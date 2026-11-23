"""Pure stress transforms on raw observations.

Time selection retains original observations and their row alignment. The
sampling transforms raise ValueError if the requested degradation is impossible.
Truth labels and experiment provenance belong to the caller's evaluation rows.
"""

from copy import deepcopy
from dataclasses import replace
from math import isfinite
from numbers import Integral, Real

import numpy as np

from core.schemas import Observation


_ROW_ALIGNED_KEYS = ("exposure_s", "original_indices")


def _finite(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _rng(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return np.random.default_rng(seed)


def _validate(obs):
    obs.validate()
    if obs.brightness_kind == "flux" and np.any(obs.y[np.isfinite(obs.y)] <= 0):
        raise ValueError("flux samples must be positive before the stress transform")


def _result(obs, *, keep=None, values=None, row_aligned_meta=()):
    meta = deepcopy(obs.meta)
    if keep is None:
        keep = slice(None)
    else:
        if not keep.any():
            raise ValueError("the requested transform would remove every row")
        for key in dict.fromkeys((*_ROW_ALIGNED_KEYS, *row_aligned_meta)):
            if key not in meta:
                continue
            data = meta[key]
            if not isinstance(data, (list, tuple, np.ndarray)):
                raise ValueError(f"row-aligned metadata {key!r} must be a sequence")
            if (isinstance(data, np.ndarray) and data.ndim == 0) or len(data) != obs.t_s.size:
                raise ValueError(f"row-aligned metadata {key!r} must match the observation length")
            if isinstance(data, np.ndarray):
                meta[key] = data[keep].copy()
            else:
                retained = [item for item, selected in zip(data, keep, strict=True) if selected]
                meta[key] = tuple(retained) if isinstance(data, tuple) else retained
    result = replace(obs, t_s=obs.t_s[keep].copy(),
                     y=(obs.y if values is None else values)[keep].copy(),
                     y_err=None if obs.y_err is None else obs.y_err[keep].copy(), meta=meta)
    _validate(result)
    return result


def add_measurement_noise(obs: Observation, *, sigma: float, seed: int) -> Observation:
    """Use additive Gaussian noise, or mean-preserving lognormal noise for flux.

    Sigma is in raw brightness units for magnitude/unknown input and in
    natural-log flux units for flux. Retain the supplied measurement errors
    and missing samples. Raise ValueError if the result cannot be represented.
    """
    _validate(obs)
    sigma = _finite(sigma, "sigma")
    if sigma < 0:
        raise ValueError("sigma must be nonnegative")
    generator = _rng(seed)
    values = obs.y.copy()
    valid = np.isfinite(values)
    if sigma and valid.any():
        draws = generator.standard_normal(int(valid.sum()))
        try:
            with np.errstate(over="raise", invalid="raise", under="ignore"):
                if obs.brightness_kind == "flux":
                    values[valid] *= np.exp(sigma * (draws - 0.5 * sigma))
                else:
                    values[valid] += sigma * draws
        except FloatingPointError as exc:
            raise ValueError("requested noise produces an unrepresentable brightness value") from exc
    return _result(obs, values=values)


def _baseline(times):
    return float(times.max()) - float(times.min()) if times.size > 1 else 0.0


def crop_baseline(obs: Observation, *, duration_s: float, start_s: float | None = None,
                  row_aligned_meta: tuple[str, ...] = ()) -> Observation:
    """Keep a closed elapsed-time window, anchored at the earliest time by default.

    At least one row must remain and the valid-sample baseline must shrink.
    """
    _validate(obs)
    duration = _finite(duration_s, "duration_s")
    if duration <= 0:
        raise ValueError("duration_s must be positive")
    start = float(obs.t_s.min()) if start_s is None else _finite(start_s, "start_s")
    end = _finite(start + duration, "crop end")
    keep = (obs.t_s >= start) & (obs.t_s <= end)
    valid = np.isfinite(obs.y)
    if not keep.any() or _baseline(obs.t_s[keep & valid]) >= _baseline(obs.t_s[valid]):
        raise ValueError("crop must retain rows and shorten the valid-sample baseline")
    return _result(obs, keep=keep, row_aligned_meta=row_aligned_meta)


def remove_gap(obs: Observation, *, start_s: float, end_s: float,
               row_aligned_meta: tuple[str, ...] = ()) -> Observation:
    """Drop samples strictly inside the interval to enlarge an internal gap.

    There must be retained valid epochs on both sides and at least one valid
    epoch removed. Thus the new local gap is strictly larger than the original
    spacings between its two bounding epochs.
    """
    _validate(obs)
    start, end = _finite(start_s, "start_s"), _finite(end_s, "end_s")
    if start >= end:
        raise ValueError("gap start must be earlier than gap end")
    before, after = obs.t_s <= start, obs.t_s >= end
    keep = before | after
    valid = np.isfinite(obs.y)
    if not ((valid & before).any() and (valid & after).any() and (valid & ~keep).any()):
        raise ValueError("a larger internal gap requires valid epochs inside and on both sides")
    return _result(obs, keep=keep, row_aligned_meta=row_aligned_meta)


def reduce_modulation(obs: Observation, *, factor: float) -> Observation:
    """Scale finite deviations from the median by factor in [0, 1].

    This reduces the measured variability, including any existing noise.
    A flat or entirely missing series has no measurable modulation to reduce.
    """
    _validate(obs)
    factor = _finite(factor, "factor")
    if not 0 <= factor <= 1:
        raise ValueError("factor must be in [0, 1]")
    values = obs.y.copy()
    valid = np.isfinite(values)
    if factor < 1 and valid.any():
        sample = np.sort(values[valid])
        middle = sample.size // 2
        centre = sample[middle] if sample.size % 2 else (
            0.5 * sample[middle - 1] + 0.5 * sample[middle]
        )
        values[valid] = factor * values[valid] + (1 - factor) * centre
    return _result(obs, values=values)


def clustered_irregular_sampling(obs: Observation, *, n_clusters: int, cluster_width_s: float,
                                 seed: int, row_aligned_meta: tuple[str, ...] = ()) -> Observation:
    """Select seeded groups of existing observations without shifting timestamps.

    Split the valid baseline into equal time strata and choose one window per
    stratum, at most cluster_width_s wide. Raise ValueError if a stratum cannot
    supply two distinct valid epochs, if the chosen windows do not separate
    groups more than their internal cadence, or if sampling discards no epochs.
    """
    _validate(obs)
    if (isinstance(n_clusters, (bool, np.bool_))
            or not isinstance(n_clusters, Integral) or n_clusters < 2):
        raise ValueError("n_clusters must be an integer of at least two")
    width = _finite(cluster_width_s, "cluster_width_s")
    if width <= 0:
        raise ValueError("cluster_width_s must be positive")
    generator = _rng(seed)
    epochs = np.unique(obs.t_s[np.isfinite(obs.y)])
    if epochs.size < 2 * n_clusters:
        raise ValueError("too few distinct valid epochs for the requested clusters")
    span = _finite(_baseline(epochs), "valid baseline")
    if width >= span / n_clusters:
        raise ValueError("cluster_width_s must be smaller than a time stratum")
    edges = np.linspace(epochs[0], epochs[-1], n_clusters + 1)
    boundaries = np.searchsorted(epochs, edges)
    boundaries[-1] = epochs.size
    keep = np.zeros(obs.t_s.size, dtype=bool)
    previous_end = None
    within, between = [], []
    for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
        stratum = epochs[left:right]
        pairs = np.flatnonzero(np.diff(stratum) <= width)
        if not pairs.size:
            raise ValueError("a time stratum has no pair of distinct valid epochs within cluster_width_s")
        start = stratum[generator.choice(pairs)]
        end = min(start + width, stratum[-1])
        selected = stratum[(stratum >= start) & (stratum <= end)]
        if selected.size < 2:
            raise ValueError("the selected time window cannot represent two distinct valid epochs")
        keep |= (obs.t_s >= start) & (obs.t_s <= end)
        within.append(np.max(np.diff(selected)))
        if previous_end is not None:
            between.append(start - previous_end)
        previous_end = selected[-1]
    if min(between) <= max(within) or np.unique(obs.t_s[keep & np.isfinite(obs.y)]).size == epochs.size:
        raise ValueError("the selected windows did not create separated clusters and discard epochs")
    return _result(obs, keep=keep, row_aligned_meta=row_aligned_meta)
