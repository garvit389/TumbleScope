"""Allowlisted transport for CP2 synthetic development data."""

import argparse
import hashlib
import json
from io import BytesIO
from pathlib import Path
import zipfile

import numpy as np

CODE = ("ml/bundle.py", "ml/model.py", "ml/train.py")
DATA = ("train.npz", "val.npz")
PAYLOAD = {*CODE, *DATA, "config.json"}
FIELDS = {"windows", "offsets", "groups", "targets", "periods"}
FEATURE_KEYS = ("window_length", "window_stride", "y_clip", "dt_floor_s", "L_min",
                "L_max", "max_windows", "min_valid_per_window")
MODEL_KEYS = ("channels", "dilations", "kernel_size")
TRAIN_KEYS = ("seed", "epochs", "batch_size", "learning_rate")
COUNTS = {"train": 3500, "val": 750}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, indent=2).encode()


def check_config(cfg):
    if set(cfg) != {"cnn", "bounds", "counts"} or cfg["counts"] != COUNTS:
        raise ValueError("unexpected training configuration")
    cnn = cfg["cnn"]
    if set(cnn) != set(FEATURE_KEYS + MODEL_KEYS + TRAIN_KEYS):
        raise ValueError("unexpected CNN settings")
    if (cnn["channels"] != [32,32,64,64,128,128] or cnn["dilations"] != [1,2,4,8,16,32]
            or cnn["kernel_size"] != 7 or cnn["window_length"] != 1024
            or not all(type(cnn[k]) in (int, float) and np.isfinite(cnn[k])
                       for k in FEATURE_KEYS + TRAIN_KEYS)
            or any(type(cnn[k]) != int or cnn[k] < 1 for k in ("epochs","batch_size"))
            or cnn["learning_rate"] <= 0):
        raise ValueError("invalid model or training settings")
    low, high = cfg["bounds"]
    if not np.isfinite([low, high]).all() or not 0 < low < high:
        raise ValueError("invalid period bounds")


def validate_split(arrays, split, cfg):
    n = COUNTS[split]
    if set(arrays) != FIELDS:
        raise ValueError("unexpected split fields")
    x, offsets, groups = (arrays[k] for k in ("windows", "offsets", "groups"))
    targets, periods = arrays["targets"], arrays["periods"]
    if (offsets.dtype != np.int64 or offsets.shape != (n+1,) or offsets[0] != 0
            or offsets[-1] != len(x) or np.any(np.diff(offsets) < 1)
            or np.any(np.diff(offsets) > cfg["cnn"]["max_windows"])
            or groups.dtype != np.int64 or groups.shape != (n,) or np.unique(groups).size != n
            or x.dtype != np.float32 or x.shape[1:] != (3,1024) or not np.isfinite(x).all()
            or np.any(np.abs(x[:,0]) > 1) or np.any((x[:,1] != 0) & (x[:,1] != 1))
            or np.any(x[:,0][x[:,1] == 0] != 0)
            or np.any(x[:,1].sum(axis=1) < cfg["cnn"]["min_valid_per_window"])
            or np.any((x[:,2] < 0) | (x[:,2] > 1))):
        raise ValueError("invalid synthetic feature arrays or curve groups")
    if (targets.shape != (n,129) or targets.dtype != np.float32
            or periods.shape != (n,) or periods.dtype != np.float64
            or not np.isfinite(periods).all() or not np.isfinite(targets).all()
            or np.any(targets < 0) or not np.allclose(targets.sum(1), 1)
            or not np.array_equal(targets[:,128], periods == 0)):
        raise ValueError("invalid synthetic targets")
    positive = periods[periods != 0]
    low, high = cfg["bounds"]
    if len(positive) != n*7//10 or np.any((positive < low) | (positive > high)):
        raise ValueError("unexpected period labels or counts")
    centers = np.log(low) + (np.arange(128)+.5) * np.log(high/low)/128
    coordinate = np.interp(np.log(positive), centers, np.arange(128))
    expected = np.maximum(1 - abs(coordinate[:,None] - np.arange(128)), 0)
    if not np.allclose(targets[periods != 0, :128], expected, atol=1e-7, rtol=0):
        raise ValueError("soft targets disagree with clean period labels")


def verify_bundle(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != PAYLOAD | {"manifest.json"}:
            raise ValueError("unexpected, duplicate or missing bundle member")
        manifest = json.loads(archive.read("manifest.json"))
        if set(manifest) != PAYLOAD:
            raise ValueError("incorrect bundle manifest")
        for name in sorted(PAYLOAD):
            if digest(archive.read(name)) != manifest[name]:
                raise ValueError(f"checksum mismatch: {name}")
        cfg = json.loads(archive.read("config.json"))
        check_config(cfg)
        splits = {}
        for split in COUNTS:
            with np.load(BytesIO(archive.read(split+".npz")), allow_pickle=False) as shard:
                if len(shard.files) != len(FIELDS) or set(shard.files) != FIELDS:
                    raise ValueError("unexpected split fields")
                splits[split] = {key: shard[key] for key in FIELDS}
            validate_split(splits[split], split, cfg)
        if np.intersect1d(splits["train"]["groups"], splits["val"]["groups"]).size:
            raise ValueError("curve identity crosses development splits")
    return cfg, splits, digest(json_bytes(manifest))


def export_bundle(corpus, output):
    from core.config import load_config, resolve_period_bounds
    from ml.dataset.corpus import ROOT

    corpus, output = Path(corpus), Path(output)
    local = load_config()
    manifest = json.loads((corpus/"manifest.json").read_bytes())
    bounds = list(resolve_period_bounds(local.period_bounds))
    cnn = {k: local.cnn[k] for k in FEATURE_KEYS + MODEL_KEYS + TRAIN_KEYS}
    cfg = {"cnn": cnn, "bounds": bounds, "counts": COUNTS}
    check_config(cfg)
    payload = {name: (ROOT/name).read_bytes() for name in CODE}
    payload["config.json"] = json_bytes(cfg)
    for split in COUNTS:
        name = split + ".npz"
        raw = (corpus/name).read_bytes()
        if digest(raw) != manifest["artifacts"][name]:
            raise ValueError("local shard checksum mismatch")
        with np.load(BytesIO(raw), allow_pickle=False) as shard:
            if (set(shard.files) != FIELDS | {"settings","bounds"}
                    or json.loads(shard["settings"].item()) != {k: cnn[k] for k in FEATURE_KEYS}
                    or shard["bounds"].tolist() != bounds):
                raise ValueError("incompatible cached shard")
            clean = {k: shard[k] for k in sorted(FIELDS)}
        clean["groups"] = np.array([
            int.from_bytes(hashlib.sha256(f"curve:{i}".encode()).digest()[:8], "big", signed=True)
            for i in clean["groups"]], dtype=np.int64)
        validate_split(clean, split, cfg)
        packed = BytesIO()
        np.savez_compressed(packed, **clean)
        payload[name] = packed.getvalue()
    hashes = {name: digest(data) for name, data in payload.items()}
    if set(hashes) != PAYLOAD:
        raise ValueError("export allowlist mismatch")
    from tempfile import TemporaryDirectory
    with TemporaryDirectory(dir=output.parent) as tmp:
        staged = Path(tmp)/output.name
        with zipfile.ZipFile(staged, "x", zipfile.ZIP_DEFLATED) as archive:
            members = {**payload, "manifest.json": json_bytes(hashes)}
            for name, data in sorted(members.items()):
                archive.writestr(zipfile.ZipInfo(name), data, compress_type=zipfile.ZIP_DEFLATED)
        verify_bundle(staged)
        if output.exists():
            raise FileExistsError(output)
        staged.rename(output)
    print(f"Validated {output}  SHA256 {digest(output.read_bytes())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export_bundle(args.corpus, args.output)
