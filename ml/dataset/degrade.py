"""Seeded synthetic sampling from the summary statistics in CONTRACTS.md §9."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from ml.dataset.labels import flat_signal

PROFILE_KEYS = {"profile_id", "n_points", "duration_s", "median_dt_s", "gap_fractions",
                "noise_rel_amplitude", "missing_fraction"}


def timeline(profile, rng):
    if set(profile) != PROFILE_KEYS:
        raise ValueError("unexpected sampling profile fields")
    count, duration = profile["n_points"], profile["duration_s"]
    cadence = profile["median_dt_s"]
    gaps = np.asarray(profile["gap_fractions"], dtype=float) * duration
    ratio, fraction = profile["noise_rel_amplitude"], profile["missing_fraction"]
    if (type(count) is not int or count < 2 or duration <= 0 or cadence <= 0
            or not isinstance(profile["profile_id"], str) or not profile["profile_id"]
            or gaps.ndim != 1 or np.any(gaps <= 0) or not np.isfinite(gaps).all()
            or not np.isfinite([ratio, fraction, duration, cadence]).all()
            or ratio < 0 or not 0 <= fraction <= 1):
        raise ValueError("invalid sampling profile")
    intervals = count - 1
    fixed = intervals // 2 + 1
    residual_count = intervals - fixed - len(gaps)
    remaining = duration - fixed * cadence - gaps.sum()
    if residual_count < 0 or (residual_count == 0 and not np.isclose(remaining, 0)):
        raise ValueError("infeasible cadence and gaps")
    residual = remaining / residual_count if residual_count else cadence
    if residual <= 0:
        raise ValueError("nonpositive profile interval")
    steps = np.r_[np.full(fixed, cadence), gaps, np.full(residual_count, residual)]
    times = np.r_[0., np.cumsum(rng.permutation(steps))]
    if not np.isclose(times[-1], duration):
        raise ValueError("profile duration mismatch")
    times[-1] = duration
    if not np.all(np.diff(times) > 0):
        raise ValueError("indistinguishable profile timestamps")
    return times


def profile_pools(directory, seed):
    path = Path(directory) / "sdlcd_profiles.json"
    fallback = not path.exists()
    if fallback:
        path = path.with_name("default_profiles.json")
    records = json.loads(path.read_text(encoding="utf-8"))
    if not records:
        raise ValueError("empty sampling profile file")
    records.sort(key=lambda p: p["profile_id"])
    if len({p["profile_id"] for p in records}) != len(records):
        raise ValueError("duplicate profile identity")
    for record in records:
        timeline(record, np.random.default_rng(seed))
    if fallback:
        return {s: records for s in ("train", "val", "test")}, path
    groups = defaultdict(list)
    for record in records:
        groups[record["profile_id"].split("_")[0]].append(record)

    def digest(prefix, obj):
        return hashlib.sha256(f"{prefix}:{obj}".encode()).digest()

    dev = sorted((o for o in groups if digest(0, o)[0] % 2 == 0),
                 key=lambda o: (min(p["duration_s"] for p in groups[o]), digest(seed, o), o))
    n_val = round(len(dev) * 750 / 4250)
    validation = {dev[(2*j + 1) * len(dev) // (2*n_val)] for j in range(n_val)}
    owners = {"train": set(dev) - validation, "val": validation, "test": set(groups) - set(dev)}
    pools = {s: [p for p in records if p["profile_id"].split("_")[0] in objects]
             for s, objects in owners.items()}
    if not all(pools.values()):
        raise ValueError("profile split is empty")
    return pools, path


def degrade(times, clean_values, profile, noise_rng, mask_rng):
    if (times.shape != clean_values.shape or times.size != profile["n_points"]
            or not np.isfinite(clean_values).all()):
        raise ValueError("invalid sampled clean source")
    sigma = profile["noise_rel_amplitude"] * np.ptp(clean_values)
    values = clean_values - np.median(clean_values) + noise_rng.normal(0, sigma, len(times))
    missing = round(len(times) * profile["missing_fraction"])
    valid = np.ones(len(times), dtype=bool)
    valid[mask_rng.choice(len(times), size=missing, replace=False)] = False
    values[~valid] = np.nan
    return times, values, valid


def coverage_error(times, clean, valid, meta, cfg):
    if valid.sum() < 2:
        return "insufficient valid observations"
    source = meta["source"]
    if source in ("under_covered", "reliable"):
        cycles = np.ptp(times[valid]) / meta["parent_repeat_s"]
        if (cycles < cfg["min_cycles"]) != (source == "under_covered"):
            return "observed coverage incompatible with clean label"
        if flat_signal(clean[valid], cfg["flat_rtol"]):
            return "sampled periodic source is flat"
    return None
