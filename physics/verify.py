"""Fit and compare apparent-period hypotheses against a simplified model."""

from dataclasses import replace
from time import perf_counter
from typing import Optional

import numpy as np

from core.schemas import (
    CandidateVerification, PreparedObservation, ProgressCallback, ReconciliationResult,
    VerificationResult, check_verification_scope, report_progress, required_verification_ids,
)
from physics.fit import _check_deadline, _fit_family, _observed_sigma, normalized_error
from physics.simulator import simulate_lightcurve


def _heldout_masks(prep: PreparedObservation, n_blocks: int) -> list[np.ndarray]:
    if type(n_blocks) is not int or n_blocks < 2:
        raise ValueError("invalid n_blocks")
    rows = np.flatnonzero(prep.valid_mask)
    _, groups = np.unique(prep.t_s[rows], return_inverse=True)
    distinct = int(groups.max()) + 1 if len(groups) else 0
    if distinct < n_blocks:
        raise ValueError("too few distinct times for held-out blocks")
    masks = []
    for block in np.array_split(np.arange(distinct), n_blocks):
        mask = np.zeros_like(prep.valid_mask)
        mask[rows[(groups >= block[0]) & (groups <= block[-1])]] = True
        masks.append(mask)
    return masks


def _score_period(prep: PreparedObservation, period_s: float, cfg: dict,
                  masks: list[np.ndarray], deadline: float | None, *,
                  sigma_obs: float | None = None) -> tuple:
    if sigma_obs is None:
        sigma_obs = _observed_sigma(prep)
    shapes = cfg["fit"]["shapes"]
    by_shape = {shape: [] for shape in shapes}
    fit_errors, hold_errors = [], []
    residuals = np.full_like(prep.y_norm, np.nan)
    for hold in masks:
        fold = []
        for shape in shapes:
            _check_deadline(deadline)
            fit = _fit_family(prep, period_s, cfg, prep.valid_mask & ~hold, shape, deadline,
                              sigma_obs=sigma_obs)
            observed = getattr(prep, fit["fit_space"])[hold]
            template = simulate_lightcurve(fit["params"], prep.t_s[hold])
            scale, offset, sigma = fit["scale_a"], fit["offset_b"], fit["sigma_obs"]
            error = normalized_error(observed, template, scale, offset, sigma)
            if not np.isfinite(error):
                raise ValueError("nonfinite held-out error")
            by_shape[shape].append((error, fit))
            fold.append((fit["fit_error"], error, (observed - (scale * template + offset)) / sigma))
        train_error, hold_error, residual = min(fold, key=lambda item: item[0])
        fit_errors.append(train_error)
        hold_errors.append(hold_error)
        residuals[hold] = residual
    _check_deadline(deadline)
    geometry_errors = {shape: float(np.mean([error for error, _ in fits]))
                       for shape, fits in by_shape.items()}
    family = min(shapes, key=geometry_errors.__getitem__)
    representative = by_shape[family][0][1]
    best_params = {key: representative[key] for key in ("fit_space", "scale_a", "offset_b")}
    best_params.update(shape=family, geometry_errors=geometry_errors)
    return float(np.mean(fit_errors)), float(np.mean(hold_errors)), best_params, residuals


def _number(value, name, minimum=None):
    if (type(value) not in (int, float) or not np.isfinite(value)
            or (minimum is not None and value < minimum)):
        raise ValueError(f"invalid {name}")
    return value


def _eligible_decoys(period_s: float, linked_periods_s: list[float], window: dict,
                     bounds: tuple[float, float], ratios: list[float],
                     k: float) -> tuple[list[float], list[str]]:
    resolution = _number(window.get("freq_resolution"), "frequency resolution")
    k = _number(k, "decoy_exclusion_resolutions")
    if resolution <= 0 or k <= 0 or not np.isfinite(resolution * k):
        raise ValueError("invalid decoy exclusion width")
    peaks = window.get("peaks")
    if not isinstance(peaks, list):
        raise ValueError("invalid window peaks")
    exclusions = [(1 / period_s, "candidate frequency")]
    exclusions.extend((1 / period, "harmonic frequency") for period in linked_periods_s)
    for peak in peaks:
        offset = _number(peak["freq_hz"], "window peak offset")
        if offset <= 0:
            raise ValueError("invalid window peak offset")
        exclusions.append((offset, "alias offset"))

    eligible, notes, seen = [], [], set()
    for ratio in ratios:
        period = period_s * ratio
        if period in seen:
            cause = "duplicate period"
        elif not bounds[0] <= period <= bounds[1]:
            cause = "outside period bounds"
        else:
            cause = next((label for freq, label in exclusions
                          if abs(1 / period - freq) <= k * resolution), None)
        seen.add(period)
        if cause is None:
            eligible.append(period)
        else:
            notes.append(f"Excluded decoy {period:.12g} s ({cause})")
    return eligible, notes


def _verify_policy(prep, cfg):
    bounds = tuple(_number(cfg.get(key), key) for key in ("period_min_s", "period_max_s"))
    if not 0 < bounds[0] < bounds[1]:
        raise ValueError("invalid period bounds")
    for key in ("heldout_error_max", "harmonic_margin"):
        _number(cfg.get(key), key, minimum=0)
    _number(cfg.get("decoy_separation_min"), "decoy_separation_min")
    if _number(cfg.get("timeout_s_per_candidate"), "timeout_s_per_candidate") <= 0:
        raise ValueError("invalid timeout_s_per_candidate")
    if type(cfg.get("min_decoys")) is not int or cfg["min_decoys"] < 1:
        raise ValueError("invalid min_decoys")
    ratios = cfg.get("decoy_ratios", [1.37, 0.61])
    if (not isinstance(ratios, (list, tuple)) or not ratios
            or any(_number(value, "decoy ratio") <= 0 for value in ratios)):
        raise ValueError("invalid decoy_ratios")
    fit = cfg["fit"]
    if (not fit["shapes"] or len(set(fit["shapes"])) != len(fit["shapes"])
            or not set(fit["shapes"]) <= {"box", "cylinder"}):
        raise ValueError("invalid fit shapes")
    lower, upper = (_number(x, "aspect range") for x in fit["aspect_range"])
    if not 0 < lower < upper:
        raise ValueError("invalid aspect range")
    for key, minimum in (("coarse_samples", 1), ("restarts", 1), ("popsize", 1),
                         ("maxiter", 1), ("cylinder_sides", 3)):
        if type(fit.get(key)) is not int or fit[key] < minimum:
            raise ValueError(f"invalid fit {key}")
    if type(cfg.get("seed")) is not int or cfg["seed"] < 0:
        raise ValueError("invalid seed")
    masks = _heldout_masks(prep, cfg.get("n_blocks"))
    window = cfg.get("window_result")
    if not isinstance(window, dict):
        raise ValueError("missing or invalid window_result")
    frequencies = window.get("freq_hz")
    power = window.get("window_power")
    peaks = window.get("peaks")
    if not isinstance(frequencies, list) or not isinstance(power, list) or not isinstance(peaks, list):
        raise ValueError("invalid window_result arrays")
    if not frequencies or len(frequencies) != len(power):
        raise ValueError("invalid window_result arrays")
    freqs = np.array([_number(value, "window frequency") for value in frequencies])
    powers = np.array([_number(value, "window power") for value in power])
    if freqs[0] != 0 or powers[0] != 1 or np.any(np.diff(freqs) <= 0) or np.any((powers < 0) | (powers > 1)):
        raise ValueError("invalid window_result values")
    resolution = window.get("freq_resolution")
    if resolution != prep.meta["freq_resolution"]:
        raise ValueError("window_result frequency resolution mismatch")
    if resolution is not None and _number(resolution, "frequency resolution") <= 0:
        raise ValueError("invalid frequency resolution")
    heights = []
    for peak in peaks:
        if not isinstance(peak, dict) or set(peak) != {"freq_hz", "height"}:
            raise ValueError("invalid window_result peak")
        frequency = _number(peak["freq_hz"], "window peak frequency")
        height = _number(peak["height"], "window peak height")
        index = np.searchsorted(freqs, frequency)
        if frequency <= 0 or index == len(freqs) or freqs[index] != frequency or powers[index] != height or not 0 <= height <= 1:
            raise ValueError("invalid window_result peak")
        heights.append(height)
    if heights != sorted(heights, reverse=True):
        raise ValueError("window_result peaks out of order")
    return bounds, ratios, masks, window


def _passes(heldout_error, deltas, cfg):
    return (heldout_error <= cfg["heldout_error_max"]
            and min(deltas) >= cfg["decoy_separation_min"])


def _harmonic_partners(recon, required, by_id):
    partners = {cid: set() for cid in required}
    for cluster in recon.clusters:
        cid = cluster.cluster_id
        if cid not in partners:
            continue
        for link in cluster.harmonic_links:
            other = link["other_id"]
            if link["ratio"] not in (0.5, 2.):
                continue
            if not by_id[other].stage2["passed"]:
                partners[cid].add(other)
            elif other in required and (cid in recon.shortlist or other in recon.shortlist):
                partners[cid].add(other)
                partners[other].add(cid)
    return partners


def _compare_harmonic(record, other, partner, margin):
    comparison = {"partner_id": partner.cluster_id, "prefers": "undecided",
                  "delta_e": None, "reason": None}
    if not partner.stage2["passed"]:
        comparison["reason"] = "partner_failed_stage2"
    elif other is None or other.status != "ok" or record.status != "ok":
        comparison["reason"] = "harmonic_fit_unfinished"
    else:
        short, long = sorted((record, other), key=lambda item: item.period_s)
        delta = long.heldout_error - short.heldout_error
        comparison["delta_e"] = delta
        if delta and abs(delta) >= margin:
            comparison["prefers"] = "P" if delta > 0 else "2P"
    return comparison


def _odd_even_notes(prep, residuals, shorter_period):
    times = prep.t_s[prep.valid_mask]
    parity = np.remainder(np.floor((times - times[0]) / shorter_period), 2)
    residual = residuals[prep.valid_mask]
    notes = []
    for flag, name in enumerate(("even", "odd")):
        rows = residual[parity == flag]
        error = f"{np.sqrt(np.mean(rows**2)):.9g}" if len(rows) else "unavailable"
        notes.append(f"{name} shorter-period cycles: n={len(rows)}, normalized RMS={error}")
    return notes


def verify_candidates(prep: PreparedObservation, recon: ReconciliationResult, cfg: dict,
                      mode: str = "fast",
                      progress: Optional[ProgressCallback] = None) -> VerificationResult:
    if mode != "fast":
        raise NotImplementedError("Only Fast mode is available.")
    started = perf_counter()
    report_progress(progress, 0.0, "Starting verification")
    by_id = {cluster.cluster_id: cluster for cluster in recon.clusters}
    required = required_verification_ids(recon)
    partners = _harmonic_partners(recon, required, by_id)
    records, residuals = {}, {}
    policy_error = None
    if required:
        try:
            prep.validate()
            bounds, ratios, masks, window = _verify_policy(prep, cfg)
            sigma_obs = _observed_sigma(prep)
        except Exception as exc:
            policy_error = str(exc) or type(exc).__name__
    for cluster in recon.clusters:
        cid, period = cluster.cluster_id, cluster.period_s
        if cid not in required:
            continue
        notes, fitted_decoys, deltas = [], [], []
        try:
            if len(partners[cid]) > 1:
                raise ValueError("multiple harmonic comparisons require the contract amendment")
            if policy_error is not None:
                raise ValueError(policy_error)
            deadline = perf_counter() + cfg["timeout_s_per_candidate"]
            linked = [by_id[link["other_id"]].period_s for link in cluster.harmonic_links]
            decoys, exclusions = _eligible_decoys(
                period, linked, window, bounds, ratios, cfg.get("decoy_exclusion_resolutions"))
            notes.extend(exclusions)
            if len(decoys) < cfg["min_decoys"]:
                raise ValueError("no_valid_decoy")
            fit, heldout, params, residual = _score_period(prep, period, cfg, masks, deadline,
                                                            sigma_obs=sigma_obs)
            for decoy in decoys:
                _, error, _, _ = _score_period(prep, decoy, cfg, masks, deadline,
                                               sigma_obs=sigma_obs)
                _check_deadline(deadline)
                fitted_decoys.append(decoy)
                deltas.append(error - heldout)
            _check_deadline(deadline)
            notes.append("Representative coefficients from first chronological fold")
            record = CandidateVerification(
                cid, period, "ok", fit, heldout, fitted_decoys, deltas, None,
                _passes(heldout, deltas, cfg), params, notes,
            )
            record.validate()
            records[cid] = record
            residuals[cid] = residual
        except Exception as exc:
            notes.append(str(exc) or type(exc).__name__)
            records[cid] = CandidateVerification(
                cid, period, "failed", None, None, fitted_decoys, deltas, None, False, None, notes,
            )
        report_progress(progress, len(records) / (len(required) + 1),
                        f"Processed verification candidate {len(records)} of {len(required)}")
    for cid, record in list(records.items()):
        if len(partners[cid]) != 1:
            continue
        partner_id = next(iter(partners[cid]))
        partner = by_id[partner_id]
        comparison = _compare_harmonic(record, records.get(partner_id), partner,
                                       cfg.get("harmonic_margin"))
        notes = record.notes.copy()
        if record.status == "ok":
            notes.extend(_odd_even_notes(prep, residuals[cid], min(record.period_s, partner.period_s)))
            notes.append("AIC/BIC not compared: equal free parameter counts")
        records[cid] = replace(record, harmonic_comparison=comparison, notes=notes)
    result = VerificationResult(list(records.values()), bool(records) and all(
        record.status == "ok" for record in records.values()
    ), "fast", perf_counter() - started)
    result.validate()
    if required:
        check_verification_scope(recon, result)
    report_progress(progress, 1.0, "Verification finished")
    return result
