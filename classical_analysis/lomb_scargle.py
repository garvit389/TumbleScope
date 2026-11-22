"""Lomb–Scargle candidate search on the shared classical frequency grid."""

from math import ceil, isfinite
from numbers import Real
from time import perf_counter

import numpy as np
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks
from scipy.special import betainccinv

from core.schemas import Candidate, EstimatorResult, LOMB_SCARGLE, PreparedObservation, TOO_FEW_VALID_SAMPLES


def _positive_number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


def build_frequency_grid(prep: PreparedObservation, cfg: dict) -> np.ndarray:
    """Return the uniform trial frequencies in Hz for LS and PDM.

    Use the median spacing between distinct valid times for the cadence cap.
    The cap is a search policy for irregular sampling, not a Nyquist theorem.
    Both accessible endpoints are included, with spacing at most resolution / oversampling.
    """
    period_min = _positive_number(cfg.get("period_min_s"), "period_min_s")
    period_max = _positive_number(cfg.get("period_max_s"), "period_max_s")
    if period_min >= period_max:
        raise ValueError("period_min_s must be less than period_max_s")
    oversampling = _positive_number(cfg.get("frequency_oversampling", 5), "frequency_oversampling")
    cadence_factor = _positive_number(cfg.get("cadence_frequency_factor", 0.5), "cadence_frequency_factor")
    resolution = _positive_number(prep.meta["freq_resolution"], "freq_resolution")
    times = np.unique(prep.t_s[prep.valid_mask])
    if times.size < 2:
        raise ValueError("frequency grid requires distinct valid timestamps")
    cadence = float(np.median(np.diff(times)))

    lower = 1.0 / period_max
    upper = min(1.0 / period_min, cadence_factor / cadence)
    if not 0 < lower < upper < np.inf:
        raise ValueError("period bounds and cadence give no usable frequency interval")
    intervals = ceil((upper - lower) / resolution * oversampling)
    return np.linspace(lower, upper, intervals + 1)


def _power(times, values, frequencies, nterms):
    centered = values - values.mean()
    if np.ptp(values) == 0:
        return np.zeros_like(frequencies)
    total = centered @ centered
    model = LombScargle(times, values, nterms=nterms, normalization="standard")
    try:
        return model.power(frequencies, method="chi2")
    except np.linalg.LinAlgError:
        # Exact regular sampling can give rank-deficient designs at particular
        # frequencies. Solve those designs by least squares in the same basis.
        power = np.empty_like(frequencies)
        for index, frequency in enumerate(frequencies):
            design = model.design_matrix(frequency)
            coefficients = np.linalg.lstsq(design, centered, rcond=None)[0]
            residual = centered - design @ coefficients
            power[index] = 1.0 - (residual @ residual) / total
        return power


def _peak_width(frequencies, power, index, resolution):
    half_power = power[index] / 2
    left = index
    while left > 0 and power[left] > half_power:
        left -= 1
    right = index
    while right < len(power) - 1 and power[right] > half_power:
        right += 1
    if power[left] > half_power or power[right] > half_power:
        return resolution
    lower = np.interp(half_power, power[left:left + 2], frequencies[left:left + 2])
    upper = np.interp(half_power, power[right - 1:right + 1][::-1], frequencies[right - 1:right + 1][::-1])
    return float(upper - lower) if upper > lower else resolution


def _candidates(frequencies, power, resolution, cutoff, limit, period_min, period_max):
    if not np.any(power > cutoff):
        return []
    peaks = find_peaks(np.r_[-np.inf, power, -np.inf])[0] - 1
    ranked = sorted((index for index in peaks if power[index] > cutoff),
                    key=lambda index: (-power[index], index))
    selected = []
    for index in ranked:
        frequency = float(frequencies[index])
        if any(abs(frequency - candidate.meta["freq_hz"]) < resolution for candidate in selected):
            continue
        # Inverting a bound can exceed it by one rounding step.
        period = min(max(1.0 / frequency, period_min), period_max)
        selected.append(Candidate(
            period, float(power[index] / power[ranked[0]]), float(power[index]),
            {"freq_hz": frequency, "rank": len(selected) + 1,
             "freq_width_hz": _peak_width(frequencies, power, index, resolution)},
        ))
        if len(selected) == limit:
            break
    return selected


def run_lomb_scargle(prep: PreparedObservation, cfg: dict) -> EstimatorResult:
    started = perf_counter()
    try:
        prep.validate()
        nterms = cfg.get("ls_nterms", 1)
        limit = cfg.get("ls_max_candidates", 5)
        plot_limit = cfg.get("ls_periodogram_max_points", 1000)
        alpha = _positive_number(cfg.get("ls_false_alarm_alpha", 0.025), "ls_false_alarm_alpha")
        if type(nterms) is not int or not 1 <= nterms <= 3:
            raise ValueError("ls_nterms must be 1, 2, or 3")
        if type(limit) is not int or limit < 1:
            raise ValueError("ls_max_candidates must be a positive integer")
        if type(plot_limit) is not int or plot_limit < 2:
            raise ValueError("ls_periodogram_max_points must be an integer of at least 2")
        if alpha >= 1:
            raise ValueError("ls_false_alarm_alpha must be less than 1")

        times, starts, counts = np.unique(prep.t_s[prep.valid_mask], return_index=True, return_counts=True)
        values = prep.y_norm[prep.valid_mask]
        fit_parameters = 1 + 2 * nterms
        if times.size <= fit_parameters:
            return EstimatorResult(LOMB_SCARGLE, "skipped", [], TOO_FEW_VALID_SAMPLES,
                                   {"runtime_s": perf_counter() - started})

        # Use one equally weighted brightness per epoch so repeated rows do
        # not increase the sample count used by the significance policy.
        values = np.add.reduceat(values, starts) / counts
        frequencies = build_frequency_grid(prep, cfg)
        resolution = prep.meta["freq_resolution"]
        power = _power(times, values, frequencies, nterms)
        tolerance = 32 * np.finfo(float).eps
        if not np.isfinite(power).all() or np.any((power < -tolerance) | (power > 1 + tolerance)):
            raise ValueError("Lomb–Scargle produced invalid power")
        power = np.clip(power, 0, 1)

        # Under independent Gaussian errors, standard power is the fraction of
        # residual sum of squares removed by 2*nterms extra fit coefficients.
        # Test every grid frequency using a Bonferroni-adjusted upper tail.
        cutoff = betainccinv(nterms, (times.size - fit_parameters) / 2, alpha / frequencies.size)
        candidates = _candidates(frequencies, power, resolution, cutoff, limit,
                                 cfg["period_min_s"], cfg["period_max_s"])
        plot_indices = np.linspace(0, frequencies.size - 1, min(plot_limit, frequencies.size), dtype=int)
        return EstimatorResult(LOMB_SCARGLE, "ok", candidates, meta={
            "runtime_s": perf_counter() - started,
            "freq_resolution": resolution,
            "periodogram": {"freq_hz": frequencies[plot_indices].tolist(), "power": power[plot_indices].tolist()},
        })
    except (ValueError, TypeError, KeyError, ArithmeticError, RuntimeError, np.linalg.LinAlgError) as exc:
        return EstimatorResult(LOMB_SCARGLE, "failed", [], str(exc) or type(exc).__name__,
                               {"runtime_s": perf_counter() - started})


if __name__ == "__main__":
    from classical_analysis._smoke_cli import run_smoke

    raise SystemExit(run_smoke(run_lomb_scargle, "Run Lomb–Scargle on an SDLCD file."))
