"""Seeded parameter draws within disjoint simulator-variant ranges."""

import numpy as np

from physics.params import SimParams


SPLITS = ("train", "val", "test")
SOURCES = ("reliable", "noise_or_constant", "multi_frequency", "under_covered")


def interval(value, name):
    pair = np.asarray(value, dtype=np.float64)
    if pair.shape != (2,) or not np.isfinite(pair).all() or pair[0] > pair[1]:
        raise ValueError(f"invalid {name} range")
    return tuple(map(float, pair))


def validate_design(cfg):
    for name in ("cadence_s", "min_cycles", "baseline_range_s", "variants", "quotas",
                 "flat_rtol", "repeat_tolerance", "nrp_min_residual", "sampling_tolerance", "max_attempts", "samples_per_cycle"):
        if cfg.get(name) is None:
            raise ValueError(f"pilot setting requires a value: {name}")
    if interval(cfg["baseline_range_s"], "baseline")[0] <= 0:
        raise ValueError("baseline must be positive")
    for name in ("cadence_s", "min_cycles"):
        if not np.isfinite(cfg[name]) or cfg[name] <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if cfg["cadence_s"] >= cfg["period_min_s"] / 2:
        raise ValueError("pilot cadence must resolve the shortest configured cycle")
    for name in ("flat_rtol", "repeat_tolerance"):
        if not 0 <= cfg[name] < 1:
            raise ValueError(f"invalid {name}")
    for name in ("max_attempts", "samples_per_cycle", "cylinder_sides", "seed"):
        if type(cfg.get(name)) is not int or cfg[name] < 0:
            raise ValueError(f"invalid {name}")
    if cfg["max_attempts"] < 1 or cfg["samples_per_cycle"] < 4 or cfg["cylinder_sides"] < 3:
        raise ValueError("invalid retry, template or facet count")
    if not 0 < cfg["sampling_tolerance"] < 1:
        raise ValueError("invalid sampling tolerance")
    if not cfg["repeat_tolerance"] < cfg["nrp_min_residual"] < 1:
        raise ValueError("invalid NRP residual criterion")
    variants, quotas = cfg["variants"], cfg["quotas"]
    if set(variants) != set(SPLITS) or set(quotas) != set(SPLITS):
        raise ValueError("expected train, val and test designs")
    seen = []
    for split in SPLITS:
        counts = quotas[split]
        if set(counts) != set(SOURCES) or any(type(n) is not int or n < 0 for n in counts.values()):
            raise ValueError("invalid source quotas")
        if not variants[split] or not sum(counts.values()):
            raise ValueError("empty split")
        for variant in variants[split]:
            aspect = interval(variant["aspect"], "aspect")
            polar = interval(variant["polar_rad"], "polar")
            shape, surface = variant["shape"], variant["surface"]
            if shape not in ("box", "cylinder") or surface not in ("uniform", "unequal", "cap_contrast"):
                raise ValueError("unknown geometry or surface")
            if surface == "cap_contrast" and shape != "cylinder":
                raise ValueError("cap contrast requires a cylinder")
            if aspect[0] <= 0 or polar[0] < 0 or polar[1] > np.pi:
                raise ValueError("invalid aspect ratio or polar angle")
            for owner, family, other_aspect, other_polar in seen:
                overlaps = (max(aspect[0], other_aspect[0]) <= min(aspect[1], other_aspect[1])
                            and max(polar[0], other_polar[0]) <= min(polar[1], other_polar[1]))
                if owner != split and family == (shape, surface) and overlaps:
                    raise ValueError("overlapping cross-split variant ranges")
            seen.append((split, (shape, surface), aspect, polar))
    for dimension in (2, 3):
        development = [entry[dimension] for entry in seen if entry[0] != "test"]
        held_out = [entry[dimension] for entry in seen if entry[0] == "test"]
        if not all(all(r[1] < d[0] or d[1] < r[0] for d in development) for r in held_out):
            raise ValueError("test needs exclusive aspect and axis-orientation ranges")
    families = {s: {(v["shape"], v["surface"]) for v in variants[s]} for s in SPLITS}
    if not families["test"] - (families["train"] | families["val"]):
        raise ValueError("test requires an exclusive geometry family variation")


def sample_params(cfg, split, rng):
    variants = cfg["variants"][split]
    index = int(rng.integers(len(variants)))
    spec = variants[index]
    aspect = rng.uniform(*spec["aspect"])
    sides = cfg["cylinder_sides"]
    shape = spec["shape"]
    dimensions = (aspect, 1., 1.) if shape == "box" else (.5, aspect)
    rho = np.ones(6 if shape == "box" else sides + 2)
    if spec["surface"] == "unequal":
        rho = rng.uniform(0., 1., len(rho))
    elif spec["surface"] == "cap_contrast":
        rho[-2:] = rng.uniform(0., 1.)
    cycle = np.exp(rng.uniform(np.log(cfg["period_min_s"]), np.log(cfg["period_max_s"])))
    params = SimParams(shape, dimensions, tuple(rho), rng.uniform(*spec["polar_rad"]),
                       rng.uniform(0, 2*np.pi), rng.uniform(0, 2*np.pi), cycle,
                       tuple(rng.normal(size=3)), tuple(rng.normal(size=3)), sides)
    return params, index
