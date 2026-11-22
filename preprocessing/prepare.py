"""Prepare actual measurements under CONTRACTS.md conventions."""
import numpy as np
from core.schemas import Observation, PreparedObservation


def prepare(obs: Observation, cfg: dict) -> PreparedObservation:
    obs.validate()
    # Canonical ordering within duplicate timestamps makes shuffled inputs equivalent.
    uncertainty = obs.y_err if obs.y_err is not None else np.zeros_like(obs.y)
    order = np.lexsort((uncertainty, obs.y, obs.t_s))
    times = obs.t_s[order].copy()
    times -= times[0]
    values = obs.y[order].copy()
    valid = np.isfinite(values)
    sign = -1. if obs.brightness_kind == "mag" else 1.
    directed = sign * values
    median = float(np.median(directed[valid])) if valid.any() else 0.
    mad = float(np.median(np.abs(directed[valid] - median))) if valid.any() else 0.
    scale = mad if mad > 0 else 1.
    normalized = (directed - median) / scale
    flux = None
    if obs.brightness_kind == "mag":
        with np.errstate(over="ignore", under="ignore"):
            flux = np.power(10., .4 * (directed - median))
        if not np.isfinite(flux[valid]).all() or (flux[valid] <= 0).any():
            raise ValueError("magnitude range cannot be represented as positive relative flux")
    elif obs.brightness_kind == "flux":
        if np.any(values[valid] <= 0):
            raise ValueError("non-positive flux cannot satisfy the relative-flux contract")
        denominator = float(np.median(values[valid])) if valid.any() else 1.
        flux = values / denominator
    dt = np.diff(times, prepend=times[0])
    positive = dt[dt > 0]
    cadence = float(np.median(positive)) if positive.size else 0.
    duplicate = np.zeros(len(times), dtype=bool)
    repeated = np.diff(times) == 0
    duplicate[:-1] |= repeated
    duplicate[1:] |= repeated
    gap_edge = np.zeros(len(times), dtype=bool)
    gaps = np.diff(times) > cfg.get("gap_cadence_multiple", 3.) * cadence
    gap_edge[:-1] |= gaps
    gap_edge[1:] |= gaps
    valid_times = times[valid]
    baseline = float(valid_times[-1] - valid_times[0]) if valid_times.size >= 2 else 0.
    meta = dict(obs.meta, baseline_s=baseline, freq_resolution=1. / baseline if baseline > 0 else None,
                n_rows=len(times), n_valid=int(valid.sum()), original_indices=order.tolist())
    if "exposure_s" in meta and isinstance(meta["exposure_s"], list):
        exposure = np.asarray(meta["exposure_s"], dtype=float)
        if exposure.shape != obs.t_s.shape:
            raise ValueError("exposure array must match observation length")
        meta["exposure_s"] = exposure[order].tolist()
    result = PreparedObservation(obs.obs_id, times, normalized, valid, dt, flux,
        {"duplicate": duplicate, "gap_edge": gap_edge,
         "bright_outlier": valid & (normalized > cfg.get("bright_outlier_mad", 6.))},
        {"median": median, "mad": mad, "sign": sign}, meta)
    result.validate()
    return result
