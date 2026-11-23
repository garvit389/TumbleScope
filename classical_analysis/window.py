"""Sampling-only spectral window for identifying frequency offsets."""

from math import ceil, isfinite
from numbers import Real

import numpy as np
from scipy.signal import find_peaks

from core.schemas import PreparedObservation


def _frequency_span(cfg):
    """Maximum difference between two frequencies inside the injected bounds."""
    lower = cfg.get("period_min_s")
    upper = cfg.get("period_max_s")
    if not all(isinstance(value, Real) and not isinstance(value, bool) and isfinite(value)
               for value in (lower, upper)) or not 0 < lower < upper:
        return 0.0
    return 1.0 / float(lower) - 1.0 / float(upper)


def _window_power(times, frequencies):
    """Evaluate the squared magnitude of the mean unit phasor."""
    power = np.empty_like(frequencies)
    # Bound the workspace while evaluating the direct transform in batches.
    batch_size = max(1, 500_000 // times.size)
    for start in range(0, frequencies.size, batch_size):
        batch = frequencies[start:start + batch_size]
        cycles = batch[:, None] * times
        np.remainder(cycles, 1.0, out=cycles)
        cycles *= 2 * np.pi
        real = np.cos(cycles).mean(axis=1)
        imag = np.sin(cycles).mean(axis=1)
        power[start:start + batch.size] = np.clip(real ** 2 + imag ** 2, 0.0, 1.0)
    return power


def compute_window(prep: PreparedObservation, cfg: dict) -> dict:
    resolution = prep.meta["freq_resolution"]
    if resolution is not None:
        resolution = float(resolution) if isfinite(resolution) and resolution > 0 else None
    result = {"freq_hz": [0.0], "window_power": [1.0], "peaks": [], "freq_resolution": resolution}
    times = np.unique(prep.t_s[prep.valid_mask])
    if times.size < 3 or resolution is None:
        return result
    if not np.isfinite(times).all() or not times[-1] > times[0]:
        return result

    count = cfg.get("n_peaks", 5)
    harmonics = cfg.get("window_cadence_harmonics", 3)
    max_points = cfg.get("window_max_grid_points", 16385)
    min_height = cfg.get("window_min_peak_height", 0.05)
    oversampling = cfg.get("frequency_oversampling", 5)
    for key, value, minimum in (("n_peaks", count, 0), ("window_cadence_harmonics", harmonics, 1),
                                ("window_max_grid_points", max_points, 3)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{key} must be an integer of at least {minimum}")
    if isinstance(min_height, bool) or not isinstance(min_height, Real) or not 0 <= min_height <= 1:
        raise ValueError("window_min_peak_height must be in [0, 1]")
    if (isinstance(oversampling, bool) or not isinstance(oversampling, Real)
            or not isfinite(oversampling) or oversampling <= 0):
        raise ValueError("frequency_oversampling must be finite and positive")

    times -= times[0]
    cadence = float(np.median(np.diff(times)))
    upper = max(harmonics / cadence, _frequency_span(cfg))
    # Extremely large cycle counts lose their fractional phase in float64.
    if not 0 < upper < np.inf or upper * float(times[-1]) >= 2 ** 52:
        return result

    # Use the LS/PDM target spacing. Sampling aliases extend beyond the
    # signal grid's cadence cap, and offsets start at DC.
    intervals = max(1, ceil(min(max_points - 1, upper / resolution * oversampling)))
    frequencies = np.linspace(0.0, upper, intervals + 1)
    step = upper / intervals
    lookahead = upper + step
    if not isfinite(lookahead) or lookahead <= upper or np.any(np.diff(frequencies) <= 0):
        return result

    # Evaluate one extra point so a peak exactly at the upper limit can be
    # distinguished from a curve still rising there.
    power = _window_power(times, np.r_[frequencies, lookahead])
    if not np.isfinite(power).all():
        return result
    peaks = find_peaks(power)[0]
    ranked = sorted((index for index in peaks if power[index] > min_height),
                    key=lambda index: (-power[index], frequencies[index]))
    result["freq_hz"] = frequencies.tolist()
    result["window_power"] = power[:-1].tolist()
    result["peaks"] = [{"freq_hz": float(frequencies[index]), "height": float(power[index])}
                       for index in ranked[:count]]
    return result


if __name__ == "__main__":
    from classical_analysis._smoke_cli import run_smoke

    raise SystemExit(run_smoke(compute_window, "Compute the spectral window of an SDLCD file."))
