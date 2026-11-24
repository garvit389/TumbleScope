"""Build and check local real and portable synthetic precomputed analyses."""

import argparse
from hashlib import sha256
from pathlib import Path
import sys

from core.config import load_config
from core.schemas import SCHEMA_VERSION
from ingestion import load_observation
from ml.train import atomic_write
from pipeline import cache, controller


ROOT = Path(__file__).resolve().parents[1]
REAL_ROOT = ROOT / "data/sdlcd_raw"
REAL_NAMES = ("00080B_1_2_R_20190724_DATA.txt", "01019D_2_3_4_R_20190910_DATA.txt")
SYNTHETIC = ROOT / "tests/fixtures/synthetic_sine.csv"


def _publish(root, inputs, kind, cfg):
    entries = []
    for source, expected_digest in inputs:
        raw = source.read_bytes()
        input_digest = sha256(raw).hexdigest()
        if expected_digest is not None and input_digest != expected_digest:
            raise ValueError(f"Input checksum mismatch for {source.name}")
        observation = load_observation(str(source))
        if source.read_bytes() != raw:
            raise ValueError(f"Input changed during ingestion for {source.name}")
        snapshot = cache._input_snapshot(cfg, raw)
        result = controller.analyze(observation, cfg)
        result.validate()
        payload = cache._encoded({
            "source_kind": kind, "basename": source.name, "snapshot": snapshot,
            "analysis": cache._public_result(result).to_dict(),
        })
        checksum = sha256(payload).hexdigest()
        entry = {
            "path": f"{checksum}.json", "analysis_sha256": checksum, "input_sha256": input_digest,
            "source_kind": kind, "basename": source.name,
        }
        atomic_write(root / entry["path"], lambda path: path.write_bytes(payload))
        cache._load_demo(root, entry, cfg, cache._implementation_identity(),
                         cache._artifact_identity(), kind)
        entries.append(entry)
    if not entries:
        raise ValueError("No demo inputs are available")
    index = cache._encoded({"schema_version": SCHEMA_VERSION, "entries": entries})
    atomic_write(root / "index.json", lambda path: path.write_bytes(index))


def _real_inputs(root):
    manifest = cache._read_json((root / "download_manifest.json").read_bytes())
    checksums = {record["filename"]: record["sha256"] for record in manifest["records"]}
    inputs = []
    for name in REAL_NAMES:
        if not (root / name).is_file() or not cache._is_digest(checksums.get(name)):
            print(f"Local SDLCD demo input unavailable or missing a manifest digest: {name}", file=sys.stderr)
        else:
            inputs.append((root / name, checksums[name]))
    return inputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate existing entries without regeneration")
    args = parser.parse_args(argv)
    cfg = load_config()
    original = controller._OUTPUTS
    failed = False
    try:
        controller._OUTPUTS = cache.LOCAL_DEMOS.parent
        for root, kind in ((cache.PORTABLE_DEMOS, "synthetic"), (cache.LOCAL_DEMOS, "sdlcd")):
            try:
                if not args.check:
                    inputs = [(SYNTHETIC, None)] if kind == "synthetic" else _real_inputs(REAL_ROOT)
                    failed |= kind == "sdlcd" and len(inputs) != len(REAL_NAMES)
                    if not inputs:
                        continue
                    _publish(root, inputs, kind, cfg)
                rows = cache._read_demo_index(root, cfg, kind)
                if kind == "sdlcd" and set(rows) != set(REAL_NAMES):
                    failed = True
                    print("The local demo cache is missing a curated SDLCD entry.", file=sys.stderr)
                for name, (result, _) in rows.items():
                    verdict = result.verdict.label if result.verdict else "No verdict"
                    print(f"{kind} · Precomputed · {name} · {verdict}")
            except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
                print(f"{kind} demo cache unavailable: {exc}", file=sys.stderr)
                failed = True
    finally:
        controller._OUTPUTS = original
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
