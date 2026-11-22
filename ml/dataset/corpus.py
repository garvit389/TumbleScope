"""Generate the Day 2 synthetic corpus locally. The test partition stays here."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import scipy

from core.config import config_hash, load_config
from core.schemas import Observation
from ml import features
from ml.dataset.degrade import coverage_error, degrade, profile_pools, timeline
from ml.dataset.generate import _check_parent, _draw_curve, simulation_config
from ml.dataset.sample_params import SOURCES, SPLITS, validate_design
from preprocessing.prepare import prepare

ROOT = Path(__file__).resolve().parents[2]
FEATURE_KEYS = ("window_length", "window_stride", "y_clip", "dt_floor_s", "L_min",
                "L_max", "max_windows", "min_valid_per_window")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generate_observation(job):
    config, profiles, split, category, slot, subtype = job
    cfg = simulation_config(config)
    split_id, category_id = SPLITS.index(split), SOURCES.index(category)
    base = [cfg["seed"], 2, split_id, category_id, slot]
    group = split_id * 1000000 + category_id * 100000 + slot
    failures = []
    for attempt in range(cfg["max_attempts"]):
        seed = base + [attempt]
        streams = [np.random.SeedSequence(seed + [stream]) for stream in range(6)]
        profile_rng, time_rng, source_rng, noise_rng, mask_rng = (
            np.random.Generator(np.random.PCG64(s)) for s in streams[1:])
        profile = profiles[int(profile_rng.integers(len(profiles)))]
        times = timeline(profile, time_rng)
        span = times[-1]
        if category != "under_covered":
            span = max(span, max(3., cfg["min_cycles"]) * cfg["period_max_s"])
            # observation_label checks the inclusive boundary on a floating point grid.
            span = np.nextafter(span, np.inf)
        parent_seed = int(streams[0].generate_state(1, dtype=np.uint64)[0])
        clean = _draw_curve(dict(cfg, baseline_range_s=[span, span]), split, category,
                            parent_seed, times, source_rng, subtype)
        if isinstance(clean, str):
            reason = clean
        else:
            _, brightness, label, soft, meta = clean
            t, y, valid = degrade(times, brightness, profile, noise_rng, mask_rng)
            reason = coverage_error(t, brightness, valid, meta, cfg)
            if reason is None:
                observation = Observation(str(group), t, y, None, "unknown", {})
                prepared = prepare(observation, config.preprocess)
                windows = features.build_cnn_windows(prepared, config.cnn)
                if len(windows):
                    for parent in meta["parents"]:
                        _check_parent(parent, split, cfg)
                    meta.update(group=group, category=category, seed_path=seed,
                                profile_id=profile["profile_id"], label_bin=label)
                    return observation, windows, soft, meta, failures
                reason = "no surviving CNN window"
        failures.append({"attempt": attempt, "profile_id": profile["profile_id"], "reason": reason})
    raise ValueError(f"quota exhausted for seed path {base}: {Counter(f['reason'] for f in failures)}")


def publish(config, destination):
    cfg = simulation_config(config)
    validate_design(cfg)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    pools, profile_path = profile_pools(ROOT / "data/profiles", cfg["seed"])
    multiplier = cfg["day2_quota_multiplier"]
    quotas = {s: {k: v * multiplier for k, v in cfg["quotas"][s].items()} for s in SPLITS}
    if [sum(quotas[s].values()) for s in SPLITS] != [3500, 750, 750]:
        raise ValueError("unexpected Day 2 split counts")
    source_paths = ("ml/dataset/corpus.py", "ml/dataset/generate.py", "ml/dataset/degrade.py",
                    "ml/dataset/sample_params.py", "ml/dataset/labels.py", "ml/features.py",
                    "preprocessing/prepare.py", "physics/simulator.py", "physics/geometry.py",
                    "physics/params.py", "core/schemas.py", "core/config.py")
    sources = {p: file_hash(ROOT / p) for p in source_paths}
    manifest = {"config": asdict(config), "config_hash": config_hash(config), "seed": cfg["seed"],
                "profiles_sha256": file_hash(profile_path), "profile_source": profile_path.name,
                "sources": sources, "quotas": quotas, "artifacts": {}, "counts": {},
                "versions": {"python": platform.python_version(), "numpy": np.__version__,
                             "scipy": scipy.__version__}}
    settings = {k: config.cnn[k] for k in FEATURE_KEYS}
    bounds = [cfg["period_min_s"], cfg["period_max_s"]]
    start = perf_counter()
    destination.parent.mkdir(exist_ok=True, parents=True)
    with TemporaryDirectory(dir=destination.parent) as temp, ProcessPoolExecutor(
            max_workers=cfg["workers"]) as workers:
        staging = Path(temp) / "corpus"
        staging.mkdir()
        for split in SPLITS:
            jobs = ((config, pools[split], split, category, i, None)
                    for category in SOURCES for i in range(quotas[split][category]))
            features_list, targets, ids, periods = [], [], [], []
            rejections = Counter()
            metadata = []
            for observation, windows, soft, row, failures in workers.map(generate_observation, jobs):
                row["rejections"] = failures
                metadata.append(row)
                rejections.update(f["reason"] for f in failures)
                features_list.append(windows)
                targets.append(soft)
                ids.append(row["group"])
                periods.append(row["apparent_period_s"] or 0.)
                if len(ids) % 250 == 0:
                    print(f"{split}: {len(ids)} accepted", flush=True)
            offsets = np.cumsum([0] + [len(x) for x in features_list])
            shard = staging / f"{split}.npz"
            np.savez_compressed(shard, windows=np.concatenate(features_list), offsets=offsets,
                                targets=np.stack(targets), groups=np.array(ids, dtype=np.int64),
                                periods=np.array(periods), bounds=np.array(bounds),
                                settings=json.dumps(settings, sort_keys=True))
            with np.load(shard, allow_pickle=False) as stored:
                if (not np.array_equal(stored["windows"], np.concatenate(features_list))
                        or not np.array_equal(stored["targets"], targets)
                        or not np.array_equal(stored["groups"], ids)
                        or not np.array_equal(stored["periods"], periods)
                        or not np.array_equal(stored["offsets"], offsets)
                        or json.loads(stored["settings"].item()) != settings
                        or stored["bounds"].tolist() != bounds
                        or len(set(ids)) != len(ids)
                        or Counter(r["category"] for r in metadata) != quotas[split]):
                    raise ValueError("corpus readback failed")
            private = staging / f"{split}_provenance.json"
            private.write_text(json.dumps(metadata, allow_nan=False), encoding="utf-8")
            manifest["artifacts"].update({p.name: file_hash(p) for p in (shard, private)})
            manifest["counts"][split] = {"accepted": len(ids), "windows": int(offsets[-1]),
                                         "rejected": sum(rejections.values()),
                                         "reasons": dict(rejections)}
        if sources != {p: file_hash(ROOT / p) for p in source_paths}:
            raise RuntimeError("generation source changed during publication")
        manifest["elapsed_s"] = perf_counter() - start
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        staging.rename(destination)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    # Each worker owns its linear algebra; avoid nested BLAS thread pools.
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    publish(load_config(), args.destination)
