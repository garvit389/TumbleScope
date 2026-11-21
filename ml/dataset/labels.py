"""Clean-template repeats and apparent-period classification targets."""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import maximum_filter1d
from scipy.optimize import minimize_scalar
from scipy.signal import correlate, find_peaks


def flat_signal(values, flat_rtol):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("expected a finite one-dimensional curve with at least two samples")
    if not np.isfinite(flat_rtol) or not 0 <= flat_rtol < 1:
        raise ValueError("flat_rtol must be in [0,1)")
    return np.ptp(values) <= flat_rtol * np.max(np.abs(values))


def template_repeat(values, phase_cycle_s, samples_per_cycle, tolerance, flat_rtol, *, wave=None):
    """Diagnostic for an imposed cyclic waveform. Its fundamental divides the cycle."""
    n = samples_per_cycle
    values = np.asarray(values, dtype=float)
    if type(n) is not int or n < 4 or values.shape != (3*n,):
        raise ValueError("expected three equally sampled cycles")
    if not np.isfinite(phase_cycle_s) or phase_cycle_s <= 0 or not 0 <= tolerance < 1:
        raise ValueError("invalid template settings")
    if flat_signal(values, flat_rtol):
        return None
    phase = np.arange(2*n) * (phase_cycle_s/n)
    spline = CubicSpline(np.arange(3*n)*phase_cycle_s/n, values)
    # A nonconstant fundamental has at least one maximum per repeat.
    peaks = find_peaks(values)[0]
    count = np.count_nonzero((peaks >= n) & (peaks < 2*n))
    scale = np.ptp(values)
    for divisor in range(count, 0, -1):
        lag = phase_cycle_s / divisor
        if wave is None:
            residual = spline(phase + lag) - values[:2*n]
        else:
            # Include interleaved phases when checking a simulator symmetry.
            t = np.r_[phase, phase + phase_cycle_s/(2*n)]
            residual = wave(t+lag) - wave(t)
        if np.sqrt(np.mean((residual / scale)**2)) <= tolerance:
            return lag
        if wave is None and n % divisor and divisor > 1:
            # A sampled template alone may not resolve interpolation error.
            linear = np.interp(phase+lag, np.arange(3*n)*phase_cycle_s/n, values)
            error = np.sqrt(np.mean(((linear-spline(phase+lag))/scale)**2))
            if np.sqrt(np.mean((residual/scale)**2)) <= error:
                return None
    return None


def observation_label(times, values, cfg):
    """Classify one uniformly sampled raw observation; all lags are in seconds."""
    t, y = np.asarray(times, dtype=float), np.asarray(values, dtype=float)
    if t.ndim != 1 or t.shape != y.shape or not np.isfinite(t).all() or not np.isfinite(y).all():
        return "numerical failure", None
    if len(t) < 4:
        return "insufficient sampling", None
    t = t - t[0]
    dt = t[-1] / (len(t)-1)
    low, high = cfg["period_min_s"], cfg["period_max_s"]
    if dt <= 0 or dt*2 >= low or not np.allclose(np.diff(t), dt, rtol=1e-10, atol=0):
        return "insufficient sampling", None
    if t[-1] < max(3., cfg["min_cycles"]) * high:
        return "insufficient coverage", None
    if flat_signal(y, cfg["flat_rtol"]):
        return "nrp", None
    if np.all(np.diff(y) >= 0) or np.all(np.diff(y) <= 0):
        return "unresolved trend", None
    if cfg.get("sampling_tolerance") is None:
        return "unconfigured policy", None
    y = y / np.max(np.abs(y))
    y = (y-y.min()) / np.ptp(y)
    if not np.isfinite(y).all():
        return "numerical failure", None
    # Common overlap; interpolate only within consecutive samples of a validated
    # regular grid. Screen below the bounds to avoid promoting a harmonic.
    ref = t[t <= t[-1]-high]
    spline = CubicSpline(t, y)
    # Estimate local reconstruction error by predicting alternating withheld
    # samples. Use it only for fractional shifts; integer shifts compare data.
    errors = np.zeros_like(y)
    for offset in (0, 1):
        kept = np.arange(offset, len(t), 2)
        omitted = kept[:-1]+1
        errors[omitted] = abs(CubicSpline(t[kept], y[kept])(t[omitted])-y[omitted])
    interpolation_rms = np.sqrt(np.mean(errors**2))
    envelope = maximum_filter1d(errors, size=3)

    def compatible(period, score):
        if score <= tolerance:
            return True
        if score >= cfg["sampling_tolerance"]:
            return False
        positions = ref + period
        discrepancy = abs(spline(positions)-y[:len(ref)])
        fraction = period/dt % 1
        # Linear interpolation error vanishes at the anchors and peaks
        # between them. Withheld samples estimate the local midpoint error.
        allowance = 4*fraction*(1-fraction)*np.interp(positions,t,envelope)
        return np.sqrt(np.mean(np.maximum(discrepancy-allowance,0.)**2)) <= tolerance

    def residual(period):
        return np.sqrt(np.mean((spline(ref+period)-y[:len(ref)])**2))

    tolerance = cfg["repeat_tolerance"]
    threshold = cfg.get("nrp_min_residual")
    if threshold is None or not tolerance < threshold < 1:
        return "unconfigured policy", None

    def candidates(subdivision):
        step = dt/subdivision
        lags = np.unique(np.r_[np.arange(2*dt, high, step), low, high])
        scores = np.array([residual(p) for p in lags])
        indices = list(find_peaks(-scores)[0])
        if scores[0] <= scores[1]:
            indices.append(0)
        if scores[-1] <= scores[-2]:
            indices.append(len(scores)-1)
        matches = []
        best = scores.min()
        for i in indices:
            lower, upper = lags[max(i-1,0)], lags[min(i+1,len(lags)-1)]
            fit = minimize_scalar(residual, bounds=(lower,upper), method="bounded",
                                  options={"xatol": step*1e-10})
            if not fit.success or not np.isfinite(fit.fun):
                raise ArithmeticError
            score, period = min((scores[i],lags[i]), (fit.fun,fit.x))
            best = min(best,score)
            if compatible(period, score):
                matches.append(period)
        return np.sort(matches), best

    try:
        coarse, coarse_best = candidates(1)
        fine, best = candidates(2)
    except (ArithmeticError, ValueError):
        return "numerical failure", None
    if len(coarse) != len(fine) or not np.allclose(coarse, fine, atol=dt/2, rtol=0):
        return "ambiguous candidates", None
    if not len(fine):
        if min(best, coarse_best) >= threshold:
            return "nrp", None
        return "ambiguous candidates", None
    period = fine[0]
    # A near-integer multiple can interpolate much better than a shorter
    # off-grid fundamental. Estimate sampling error on withheld observations
    # before trusting the larger match.
    if interpolation_rms > cfg["sampling_tolerance"]:
        return "insufficient sampling", None
    for divisor in range(2, int(period/(2*dt))+1):
        if residual(period/divisor) <= tolerance + interpolation_rms:
            return "ambiguous candidates", None
    if period < low:
        return "repeat outside bounds", None
    multiples = np.rint(fine/period)
    if np.any(abs(fine - multiples*period) > dt/2):
        return "ambiguous candidates", None
    return "resolved", float(period)


def mixture_has_repeat(wave, times, low, high, step, tolerance, flat_rtol):
    """Analytic-wave diagnostic for legacy controls. Never used to label stored rows."""
    upper = min(high, (times[-1] - times[0]) / 3)
    if upper < low:
        raise ValueError("no adequately covered mixture lag")
    # Two full cycles of overlap at the largest tested shift, for every lag.
    duration = times[-1] - times[0]
    grid = np.linspace(times[0], times[-1], int(np.ceil(duration / step)) + 1)
    spacing = grid[1] - grid[0]
    samples = wave(grid)
    if np.ptp(samples) and (np.all(np.diff(samples) >= 0) or np.all(np.diff(samples) <= 0)):
        return False
    # Round up the maximum lag to bracket minima between grid points.
    max_lag = int(np.ceil(upper / spacing))
    ref_t = np.linspace(times[0], times[-1] - upper, grid.size - max_lag)
    ref = wave(ref_t)
    if flat_signal(ref, flat_rtol):
        return True
    scale = np.ptp(ref)

    def residual(lag, indices):
        difference = (wave(ref_t[indices] + lag) - ref[indices]) / scale
        return np.sqrt(np.mean(np.square(difference)))

    # The dot products and rolling energies give squared differences at all
    # integer lags without allocating a lag-by-time matrix.
    centered = (samples - ref.mean()) / scale
    prefix = centered[:ref.size]
    energy = np.r_[0., np.cumsum(centered**2)]
    scores = energy[ref.size:] - energy[:-ref.size] - 2 * correlate(centered, prefix, mode="valid")
    entire = slice(None)
    if min(residual(low, entire), residual(upper, entire)) <= tolerance:
        return True
    if upper == low:
        return False
    minima = find_peaks(-scores)[0]
    probe = np.linspace(0, ref.size - 1, min(32, ref.size), dtype=int)
    for index in np.unique(np.r_[int(np.floor(low / spacing)), minima, max_lag]):
        bounds = (max(low, (index - 1) * spacing), min(upper, (index + 1) * spacing))
        if bounds[0] >= bounds[1]:
            continue

        def minimum(indices):
            fit = minimize_scalar(residual, args=(indices,), bounds=bounds, method="bounded",
                                  options={"xatol": spacing * 1e-7})
            if not fit.success:
                raise ValueError("mixture repeat refinement failed")
            return fit.fun

        # The subset's sum of squared residuals lower-bounds the full sum.
        if minimum(probe) * np.sqrt(probe.size / ref.size) > tolerance:
            continue
        if minimum(entire) <= tolerance:
            return True
    return False


def encode_target(period_s, cfg):
    low, high = cfg["period_min_s"], cfg["period_max_s"]
    if not np.isfinite([low, high]).all() or not 0 < low < high:
        raise ValueError("invalid target bounds")
    target = np.zeros(129, dtype=np.float32)
    if period_s is None:
        target[128] = 1.
        return 128, target
    if not low <= period_s <= high:
        raise ValueError("apparent period outside configured bins")
    edges = np.geomspace(low, high, 129)
    label = min(int(np.searchsorted(edges, period_s, side="right") - 1), 127)
    logs = np.log(edges)
    centers = (logs[1:] + logs[:-1]) / 2
    coordinate = np.interp(np.log(period_s), centers, np.arange(128))
    left = int(coordinate)
    target[left] = 1 - (coordinate - left)
    if left < 127:
        target[left + 1] = coordinate - left
    return label, target
