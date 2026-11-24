"""Materialize the independent synthetic allocation for verification ROC-AUC."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import platform
from tempfile import TemporaryDirectory

import numpy as np
import scipy

from core.config import load_config
from ml.dataset.generate import draw_curve, simulation_config
from ml.dataset.labels import template_repeat
from ml.dataset.sample_params import SPLITS
from physics.params import SimParams
from physics.simulator import simulate_lightcurve


ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / "data/synthetic_holdout/verification_auc_v1"
ALLOCATION = "verification_auc_v1"
SPLIT = "verification_auc"
SEED = 20261124
GROUP_START = 3_000_000
PER_VARIANT = 25
VARIANTS = (
    {"shape": "cylinder", "surface": "unequal", "aspect": [2.6, 2.9], "polar_rad": [0.5, 0.7]},
    {"shape": "cylinder", "surface": "unequal", "aspect": [3.6, 4.0], "polar_rad": [1.1, 1.3]},
)
_INVENTORY = ("data/synthetic/cp1_pilot/run_config.json", "data/synthetic/cp2_v1/manifest.json")
_HISTORICAL_FIT = "docs/results/verification_cp3b.json"
_SOURCE_FILES = (
    "scripts/prepare_verification_auc.py", "ml/dataset/generate.py", "ml/dataset/sample_params.py",
    "ml/dataset/labels.py", "physics/simulator.py", "physics/geometry.py", "physics/params.py",
)


def _digest(raw):
    return sha256(raw).hexdigest()


def _overlaps(left, right):
    return max(left[0], right[0]) <= min(left[1], right[1])


def _design(config):
    """Check variant isolation using published metadata, never corpus payloads."""
    cfg = simulation_config(config)
    occupied = {split: list(config.simulation["variants"][split]) for split in SPLITS}
    evidence = {}
    for relative in _INVENTORY:
        raw = (ROOT / relative).read_bytes()
        metadata = json.loads(raw)
        for split in SPLITS:
            variants = metadata["config"]["simulation"]["variants"][split]
            if not variants:
                raise ValueError("Missing corpus variant inventory")
            occupied[split].extend(variants)
        evidence[relative] = _digest(raw)
    raw = (ROOT / _HISTORICAL_FIT).read_bytes()
    fit_ranges = [config.verification["fit"]["aspect_range"],
                  json.loads(raw)["verification_policy"]["fit"]["aspect_range"]]
    evidence[_HISTORICAL_FIT] = _digest(raw)
    prior = [variant for variants in occupied.values() for variant in variants]
    for i, variant in enumerate(VARIANTS):
        if any(_overlaps(variant["aspect"], span) for span in fit_ranges):
            raise ValueError("Evaluation variant overlaps a verification fit aspect range")
        for previous in [*prior, *VARIANTS[:i]]:
            if any(_overlaps(variant[key], previous[key]) for key in ("aspect", "polar_rad")):
                raise ValueError("Evaluation variant overlaps an occupied simulator range")
        if any((variant["shape"], variant["surface"]) == (p["shape"], p["surface"]) for p in prior):
            raise ValueError("Evaluation geometry/surface combination was previously allocated")
    return {
        "allocation": ALLOCATION, "split": SPLIT, "purpose": "verification evaluation only",
        "sealed": False, "seed": SEED, "variants": list(VARIANTS),
        "curves_per_variant": PER_VARIANT, "group_start": GROUP_START,
        "curves": PER_VARIANT * len(VARIANTS),
        "label": "smallest clean repeat confirmed on the dense simulator observation",
        "sampling": "clean stratified irregular timestamps across the independently drawn baseline",
        "samples_per_curve": 2 * config.simulation["samples_per_cycle"] + 1,
        "period_bounds_s": [cfg[key] for key in ("period_min_s", "period_max_s")],
        "simulation": config.simulation, "compared_metadata_sha256": evidence,
        "sources_sha256": {name: _digest((ROOT / name).read_bytes()) for name in _SOURCE_FILES},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
    }


def _sample_times(span, count, seed_path):
    rng = np.random.default_rng(np.random.SeedSequence([*seed_path, 1]))
    interior = (np.arange(count - 2) + rng.uniform(size=count - 2)) / (count - 2)
    return np.r_[0., span * interior, span]


def _build(config, design):
    cfg = simulation_config(config)
    times, brightness, rows = [], [], []
    for index, variant in enumerate(VARIANTS):
        draw_cfg = {**cfg, "variants": {SPLIT: [variant]}}
        for slot in range(PER_VARIANT):
            rejections = []
            for attempt in range(cfg["max_attempts"]):
                seed_path = [SEED, GROUP_START, index, slot, attempt]
                seed = int(np.random.SeedSequence([*seed_path, 0]).generate_state(1, dtype=np.uint64)[0])
                draw = draw_curve(draw_cfg, SPLIT, "reliable", seed)
                if isinstance(draw, str):
                    rejections.append(draw)
                    continue
                dense_t, _, _, _, provenance = draw
                params_data = provenance["parents"][0]["params"]
                sample_t = _sample_times(float(dense_t[-1]), design["samples_per_curve"], seed_path)
                values = simulate_lightcurve(SimParams(**params_data), sample_t)
                times.append(sample_t)
                brightness.append(values)
                rows.append({
                    "group": GROUP_START + index * PER_VARIANT + slot, "variant": index,
                    "split": SPLIT, "seed_path": seed_path, "draw_seed": seed, "rejected_draws": rejections,
                    "params": params_data, "apparent_period_s": provenance["apparent_period_s"],
                    "duration_s": float(dense_t[-1]),
                })
                break
            else:
                raise ValueError(f"Clean simulator draw quota exhausted for variant {index}, slot {slot}")
    return rows, np.stack(times), np.stack(brightness)


def _checked_rows(design, rows):
    if not isinstance(rows, list) or len(rows) != design["curves"]:
        raise ValueError("Incorrect number of evaluation curves")
    draws = set()
    sim = design["simulation"]
    low, high = design["period_bounds_s"]
    for i, row in enumerate(rows):
        index, slot = divmod(i, PER_VARIANT)
        variant = VARIANTS[index]
        rejected = row["rejected_draws"]
        if (not isinstance(rejected, list) or not all(isinstance(reason, str) for reason in rejected)
                or len(rejected) >= sim["max_attempts"]):
            raise ValueError("Invalid rejected draw provenance")
        seed_path = [SEED, GROUP_START, index, slot, len(rejected)]
        seed = int(np.random.SeedSequence([*seed_path, 0]).generate_state(1, dtype=np.uint64)[0])
        params = SimParams(**row["params"])
        low_duration = max(sim["baseline_range_s"][0], max(3., sim["min_cycles"]) * high)
        if (row["group"] != GROUP_START + i or row["variant"] != index or row["split"] != SPLIT
                or row["seed_path"] != seed_path or row["draw_seed"] != seed or seed in draws
                or params.shape != variant["shape"] or params.cylinder_sides != sim["cylinder_sides"]
                or params.dimensions[0] != .5 or not variant["aspect"][0] <= params.dimensions[1] <= variant["aspect"][1]
                or not variant["polar_rad"][0] <= params.axis_polar_rad <= variant["polar_rad"][1]
                or len(set(params.reflectivity)) != params.cylinder_sides + 2
                or not low <= params.phase_cycle_s <= high or not low <= row["apparent_period_s"] <= high
                or not low_duration <= row["duration_s"] <= sim["baseline_range_s"][1]):
            raise ValueError("Invalid split, variant, seed or clean-curve provenance")
        draws.add(seed)


def read_metadata(config, cohort):
    design = _design(config)
    path = cohort / "manifest.json"
    if path.resolve().parent != cohort.absolute():
        raise ValueError("Manifest escapes the evaluation allocation")
    raw = path.read_bytes()
    manifest = json.loads(raw)
    # Versions are recorded at generation. The scientific recipe and sources must match.
    versions = manifest["design"]["versions"]
    if set(versions) != set(design["versions"]) or not all(isinstance(v, str) and v for v in versions.values()):
        raise ValueError("Invalid generation version provenance")
    if manifest["design"] != {**design, "versions": versions}:
        raise ValueError("Unrecognized evaluation allocation or changed generation inputs")
    checksum = manifest["curves_sha256"]
    if not isinstance(checksum, str) or len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
        raise ValueError("Invalid evaluation payload digest")
    _checked_rows(design, manifest["rows"])
    return manifest, _digest(raw)


def read_curves(cohort, manifest):
    path = cohort / "curves.npz"
    if path.resolve().parent != cohort.absolute():
        raise ValueError("Payload escapes the evaluation allocation")
    raw = path.read_bytes()
    if _digest(raw) != manifest["curves_sha256"]:
        raise ValueError("Evaluation payload failed checksum validation")
    with np.load(BytesIO(raw), allow_pickle=False) as curves:
        if set(curves.files) != {"t_s", "brightness"}:
            raise ValueError("Unexpected evaluation payload fields")
        times, brightness = curves["t_s"], curves["brightness"]
    design = manifest["design"]
    shape = (design["curves"], design["samples_per_curve"])
    if (times.shape != shape or brightness.shape != shape or times.dtype != np.float64
            or brightness.dtype != np.float64 or np.any(np.diff(times, axis=1) <= 0)
            or not (np.isfinite(times).all() and np.isfinite(brightness).all())):
        raise ValueError("Malformed clean evaluation curves")
    samples = design["simulation"]["samples_per_cycle"]
    for i, row in enumerate(manifest["rows"]):
        params = SimParams(**row["params"])
        if not np.array_equal(times[i], _sample_times(row["duration_s"], shape[1], row["seed_path"])):
            raise ValueError("Timestamps differ from evaluation provenance")
        if not np.array_equal(brightness[i], simulate_lightcurve(params, times[i])):
            raise ValueError("Brightness differs from evaluation provenance")
        dense = np.arange(3 * samples) / samples * params.phase_cycle_s
        repeat = template_repeat(
            simulate_lightcurve(params, dense), params.phase_cycle_s, samples,
            design["simulation"]["repeat_tolerance"], design["simulation"]["flat_rtol"],
            wave=lambda t: simulate_lightcurve(params, t),
        )
        if repeat != row["apparent_period_s"]:
            raise ValueError("Apparent period differs from the clean simulator repeat")
    return times, brightness


def publish(config, destination):
    """Publish a new allocation atomically after metadata and payload readback."""
    if destination.exists():
        raise FileExistsError(destination)
    design = _design(config)
    rows, times, brightness = _build(config, design)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as directory:
        staging = Path(directory) / ALLOCATION
        staging.mkdir()
        payload = staging / "curves.npz"
        np.savez_compressed(payload, t_s=times, brightness=brightness)
        manifest = {"design": design, "rows": rows, "curves_sha256": _digest(payload.read_bytes())}
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n",
                                               encoding="utf-8")
        checked, _ = read_metadata(config, staging)
        read_curves(staging, checked)
        staging.rename(destination)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, nargs="?", default=COHORT)
    args = parser.parse_args()
    materialized = publish(load_config(), args.destination)
    print(json.dumps({"curves": len(materialized["rows"]), "sha256": materialized["curves_sha256"]}))
