"""Generate the shareable CP3b synthetic development holdout."""

from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from core.config import load_config, resolve_period_bounds
from ml.bundle import digest
from ml.dataset.corpus import ROOT, generate_observation
from ml.dataset.degrade import profile_pools
from ml.dataset.sample_params import SOURCES


HOLDOUT = ROOT / "data/synthetic_holdout/holdout_small.npz"
CORPUS = ROOT / "data/synthetic/cp2_v1"


def _provenance(split, manifest):
    name = f"{split}_provenance.json"
    raw = (CORPUS / name).read_bytes()
    if digest(raw) != manifest["artifacts"][name]:
        raise ValueError(f"CP2 provenance checksum mismatch: {name}")
    return json.loads(raw)


def publish_holdout(destination: Path = HOLDOUT) -> dict:
    config = load_config()
    manifest = json.loads((CORPUS / "manifest.json").read_bytes())
    train = _provenance("train", manifest)
    val = _provenance("val", manifest)
    pools, source = profile_pools(ROOT / "data/profiles", config.simulation["seed"])
    if digest(source.read_bytes()) != manifest["profiles_sha256"]:
        raise ValueError("Sampling profiles differ from the CP2 corpus")
    excluded = {row["group"] for row in [*train, *val]}
    jobs = []
    for category, count in zip(SOURCES, (140, 20, 20, 20)):
        start = 1 + max(row["seed_path"][4] for row in val if row["category"] == category)
        jobs.extend((config, pools["val"], "val", category, slot, None)
                    for slot in range(start, start + count))
    with ProcessPoolExecutor(max_workers=config.simulation["workers"]) as pool:
        records = [(observation, info) for observation, _, _, info, _ in
                   pool.map(generate_observation, jobs)]
    groups = np.array([info["group"] for _, info in records], dtype=np.int64)
    # The sealed test partition uses the disjoint split-id prefix 2.
    if len(set(groups)) != 200 or excluded.intersection(groups) or np.any(groups // 1_000_000 != 1):
        raise ValueError("Holdout group is duplicated or crosses a split")
    bounds = tuple(resolve_period_bounds(config.period_bounds))
    arrays = {
        "t_s": np.concatenate([obs.t_s for obs, _ in records]),
        "brightness": np.concatenate([obs.y for obs, _ in records]),
        "offsets": np.cumsum([0, *(len(obs.t_s) for obs, _ in records)], dtype=np.int64),
        "groups": groups,
        "periods": np.array([info["apparent_period_s"] or 0. for _, info in records]),
        "bounds": np.array(bounds),
    }
    with TemporaryDirectory(dir=destination.parent) as folder:
        staged = Path(folder) / destination.name
        np.savez_compressed(staged, **arrays)
        with np.load(staged, allow_pickle=False) as verified:
            if set(verified.files) != set(arrays) or not all(
                np.array_equal(verified[key], val, equal_nan=True) for key, val in arrays.items()
            ):
                raise ValueError("Holdout readback differs from generated observations")
        data = staged.read_bytes()
        if len(data) >= 5_000_000:
            raise ValueError(f"Holdout exceeds the approximate 5 MB target: {len(data)} bytes")
        staged.replace(destination)
    return {"count": len(groups), "size_bytes": len(data), "sha256": digest(data),
            "seed": config.simulation["seed"], "groups": groups.tolist(), "bounds": list(bounds)}


if __name__ == "__main__":
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    print(json.dumps(publish_holdout()))
