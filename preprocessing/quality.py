"""Observation quality from sampling and robust brightness statistics."""
import numpy as np
from core.schemas import LOW_QUALITY, PreparedObservation, QualityReport


def assess_stage1(prep: PreparedObservation, cfg: dict) -> QualityReport:
    prep.validate()
    values = prep.y_norm[prep.valid_mask]
    times = prep.t_s[prep.valid_mask]
    steps = np.diff(times)
    positive = steps[steps > 0]
    baseline = prep.meta["baseline_s"]
    amplitude = float(np.diff(np.percentile(values, [5, 95]))[0]) / 2 if values.size else 0.
    if values.size >= 2:
        differences = np.diff(values)
        noise = float(np.median(np.abs(differences - np.median(differences)))) * 1.4826 / np.sqrt(2.)
    else:
        noise = 0.
    # Finite convention for the noiseless limit, reported explicitly below.
    ratio = amplitude / max(noise, np.finfo(float).eps)
    largest_gap = float(positive.max()) if positive.size else 0.
    missing = float(1. - prep.valid_mask.mean())
    metrics = {"n_valid": float(values.size), "duration_s": baseline,
               "missing_fraction": missing, "largest_gap_s": largest_gap,
               "median_dt_s": float(np.median(positive)) if positive.size else 0.,
               "noise_proxy": float(noise), "amplitude_to_noise": float(ratio),
               "outlier_fraction": float(prep.flags["bright_outlier"].mean())}
    warnings = []
    reason_codes = []
    if values.size < cfg.get("min_valid_points", 21) or baseline <= 0:
        status = "insufficient"
        reason_codes.append(LOW_QUALITY)
        warnings.append("Too few valid measurements or no positive time baseline.")
    else:
        if largest_gap / baseline > cfg.get("max_gap_fraction", .25):
            warnings.append("Large sampling gap limits coverage.")
        if missing > cfg.get("max_missing_fraction", .2):
            warnings.append("Many brightness measurements are missing.")
        if ratio < cfg.get("min_amplitude_to_noise", 3.):
            warnings.append("Brightness modulation is weak relative to the noise proxy.")
        status = "limited" if warnings else "suitable"
    warnings.append("Quality uses recorded cadence; exposure smearing is not corrected.")
    result = QualityReport(status, metrics, warnings, reason_codes)
    result.validate()
    return result
