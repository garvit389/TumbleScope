"""Stellingwerf phase dispersion minimization on the shared classical grid."""

from math import isfinite
from numbers import Real
from time import perf_counter

import numpy as np
from scipy.signal import find_peaks
from scipy.special import betainc

from classical_analysis.lomb_scargle import _peak_width, build_frequency_grid
from core.schemas import Candidate, EstimatorResult, PDM, PreparedObservation, TOO_FEW_VALID_SAMPLES


def _integer_setting(cfg, key, default, minimum):
    value = cfg.get(key, default)
    if type(value) is not int or value < minimum:
        raise ValueError(f"{key} must be an integer of at least {minimum}")
    return value


def _phase_dispersion(times, values, frequencies, bins, occupancy):
    """Return theta and its two degrees of freedom for each trial.

    Ignore underfilled bins. Compute both variances on the same retained
    epochs so their ratio and the fixed-frequency beta test use the same data.
    Trials with fewer than two usable bins or zero total variance have no score.
    """
    theta = np.ones(frequencies.size)
    within_df = np.zeros(frequencies.size, dtype=np.intp)
    total_df = np.zeros_like(within_df)
    scale = np.max(np.abs(values))
    if scale == 0 or np.all(values == values[0]):
        return theta, within_df, total_df

    centered = values / scale
    centered -= centered.mean()
    squares = centered ** 2
    times = times - times[0]
    # Bound the trial-by-epoch workspace even for a long baseline or dense grid.
    batch_size = max(1, 500_000 // max(times.size, bins))
    for start in range(0, frequencies.size, batch_size):
        batch = frequencies[start:start + batch_size]
        phases = np.remainder(batch[:, None] * times, 1.0)
        phase_bins = np.minimum((phases * bins).astype(np.intp), bins - 1)
        phase_bins += np.arange(batch.size)[:, None] * bins
        labels = phase_bins.ravel()
        shape = (batch.size, bins)
        counts = np.bincount(labels, minlength=batch.size * bins).reshape(shape)
        sums = np.bincount(labels, weights=np.tile(centered, batch.size),
                           minlength=batch.size * bins).reshape(shape)
        sums_sq = np.bincount(labels, weights=np.tile(squares, batch.size),
                              minlength=batch.size * bins).reshape(shape)

        usable = counts >= occupancy
        bin_count = usable.sum(axis=1)
        sample_count = counts.sum(axis=1, where=usable)
        square_total = sums_sq.sum(axis=1, where=usable)
        sum_total = sums.sum(axis=1, where=usable)
        total_ss = square_total - sum_total ** 2 / np.maximum(sample_count, 1)
        bin_mean_ss = np.divide(sums ** 2, counts, where=usable, out=np.zeros_like(sums))
        within_ss = square_total - bin_mean_ss.sum(axis=1)
        good = (bin_count >= 2) & (total_ss > 0)
        indices = start + np.flatnonzero(good)
        within_df[indices] = sample_count[good] - bin_count[good]
        total_df[indices] = sample_count[good] - 1
        # Roundoff can place residual SS just outside its theoretical range.
        residual_fraction = np.clip(within_ss[good] / total_ss[good], 0.0, 1.0)
        theta[indices] = residual_fraction * total_df[indices] / within_df[indices]

    return theta, within_df, total_df


def _trial_probabilities(theta, within_df, total_df):
    probabilities = np.ones_like(theta)
    usable = within_df > 0
    residual_fraction = theta[usable] * within_df[usable] / total_df[usable]
    probabilities[usable] = betainc(within_df[usable] / 2, (total_df[usable] - within_df[usable]) / 2,
                                   np.clip(residual_fraction, 0, 1))
    return probabilities


def _candidates(frequencies, power, usable, significant, resolution, limit, cfg):
    # Local maxima of 1 - theta are local minima of theta.
    peaks = find_peaks(np.r_[-np.inf, np.where(usable, power, -np.inf), -np.inf])[0] - 1
    ranked = sorted((index for index in peaks if significant[index] and power[index] > 0),
                    key=lambda index: (-power[index], index))
    candidates = []
    unavailable = np.r_[-1, np.flatnonzero(~usable), power.size]
    for index in ranked:
        frequency = float(frequencies[index])
        if any(abs(frequency - entry.meta["freq_hz"]) < resolution for entry in candidates):
            continue
        gap = np.searchsorted(unavailable, index)
        left, right = unavailable[gap - 1] + 1, unavailable[gap]
        width = _peak_width(frequencies[left:right], power[left:right], index - left, resolution)
        period = float(np.clip(1.0 / frequency, cfg["period_min_s"], cfg["period_max_s"]))
        candidates.append(Candidate(
            period, float(power[index] / power[ranked[0]]), float(power[index]),
            {"freq_hz": frequency, "rank": len(candidates) + 1, "freq_width_hz": width},
        ))
        if len(candidates) == limit:
            break
    return candidates


def run_pdm(prep: PreparedObservation, cfg: dict) -> EstimatorResult:
    started = perf_counter()
    try:
        prep.validate()
        bins = _integer_setting(cfg, "pdm_phase_bins", 10, 2)
        occupancy = _integer_setting(cfg, "pdm_min_bin_occupancy", 3, 2)
        limit = _integer_setting(cfg, "pdm_max_candidates", 5, 1)
        plot_limit = _integer_setting(cfg, "pdm_periodogram_max_points", 1000, 2)
        alpha = cfg.get("pdm_false_alarm_alpha", 0.05)
        if isinstance(alpha, bool) or not isinstance(alpha, Real) or not isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError("pdm_false_alarm_alpha must be between 0 and 1")

        times, starts, counts = np.unique(prep.t_s[prep.valid_mask], return_index=True, return_counts=True)
        if times.size < 2 * occupancy:
            return EstimatorResult(PDM, "skipped", [], TOO_FEW_VALID_SAMPLES,
                                   {"runtime_s": perf_counter() - started})

        frequencies = build_frequency_grid(prep, cfg)
        resolution = prep.meta["freq_resolution"]
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            # Like LS, use one equally weighted mean measurement per epoch.
            values = np.add.reduceat(prep.y_norm[prep.valid_mask], starts) / counts
            theta, within_df, total_df = _phase_dispersion(times, values, frequencies, bins, occupancy)
            probabilities = _trial_probabilities(theta, within_df, total_df)
        if not np.isfinite(theta).all() or not np.isfinite(probabilities).all():
            raise ValueError("PDM produced a non-finite statistic")

        # The beta tail assumes independent, equal-variance Gaussian epoch
        # errors. Correct for all grid trials, including correlated neighbors.
        power = 1.0 - theta
        candidates = _candidates(frequencies, power, within_df > 0,
                                 probabilities < alpha / frequencies.size, resolution, limit, cfg)
        indices = np.linspace(0, frequencies.size - 1, min(plot_limit, frequencies.size), dtype=int)
        return EstimatorResult(PDM, "ok", candidates, meta={
            "runtime_s": perf_counter() - started,
            "freq_resolution": resolution,
            "periodogram": {"freq_hz": frequencies[indices].tolist(), "power": power[indices].tolist()},
        })
    except (ValueError, TypeError, KeyError, ArithmeticError, RuntimeError) as exc:
        return EstimatorResult(PDM, "failed", [], str(exc) or type(exc).__name__,
                               {"runtime_s": perf_counter() - started})


if __name__ == "__main__":
    from classical_analysis._smoke_cli import run_smoke

    raise SystemExit(run_smoke(run_pdm, "Run phase dispersion minimization on an SDLCD file."))
