"""Local CP1 artifact acceptance, without downloads or regenerating datasets.

Run as ``python -m scripts.validate_cp1 --pilot data/synthetic/cp1_pilot``.
Reference labels are read here only to audit the evaluation table.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from core.config import config_hash, load_config, resolve_period_bounds
from core.schemas import AnalysisResult, assert_contract_equal, stage2_pending
from ingestion import load_observation
from ingestion.reference import load_reference_table
from ml.dataset.generate import _source_identity, _verify_publication
from ml.dataset.labels import template_repeat
from physics.params import SimParams
from physics.simulator import simulate_lightcurve
from pipeline import controller
from preprocessing.sampling_profile import sampling_profile


ROOT = Path(__file__).resolve().parents[1]


def _verdict_category(result):
    """Count execution states separately from scientific verdicts in the local audit."""
    result.validate()
    if result.verdict is not None:
        return result.verdict.label
    if stage2_pending(result.reconciliation, result.estimators, result.verification):
        return "stage2_pending"
    return "execution_error"


def validate(pilot):
    cfg = load_config()
    bounds = resolve_period_bounds(cfg.period_bounds)
    raw = ROOT / "data/sdlcd_raw"
    acquisition = json.loads((raw / "download_manifest.json").read_text())
    records = acquisition["records"]
    assert 20 <= len(records) <= 50
    assert {p.name for p in raw.glob("*_DATA.txt")} == {r["filename"] for r in records}
    profiles = json.loads((ROOT / "data/profiles/sdlcd_profiles.json").read_text())
    assert len(profiles) == len(records)
    by_id = {p["profile_id"]: p for p in profiles}
    assert len(by_id) == len(records)
    labels = load_reference_table(str(ROOT / "data/reference/reference_table.csv"))
    expected_periods = {}
    qualities, verdicts = Counter(), Counter()
    point_count = 0
    real_details = []
    with TemporaryDirectory() as temp, patch.object(controller, "_OUTPUTS", Path(temp)):
        for record in records:
            path = raw / record["filename"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
            assert record["source_url"] == "https://www.sdlcd.space-debris.sk/static/data/" + path.name
            measured = np.loadtxt(path)
            assert measured.shape == (record["points"], 6) and np.isfinite(measured).all()
            obs = load_observation(str(path))
            assert obs.obs_id == path.stem and len(obs.y) == len(measured)
            np.testing.assert_array_equal(obs.y, measured[:, 4])
            result = controller.analyze(obs, cfg)
            result.validate()
            assert not result.prepared.is_stub and not result.quality.is_stub
            assert not ({"prepare", "stage1"} & set(result.stage_errors))
            assert len(result.prepared.t_s) == len(measured)
            assert_contract_equal(result, AnalysisResult.from_dict(result.to_dict()))
            assert sampling_profile(result.prepared, cfg.preprocess) == by_id[obs.obs_id]
            assert result.verdict is None or result.verdict.label != "Supported"  # Later stages remain explicit stubs.
            qualities[result.quality.status] += 1
            verdicts[_verdict_category(result)] += 1
            point_count += len(measured)
            real_details.append({"obs_id": obs.obs_id, "points": len(measured),
                                 "stage1": result.quality.status,
                                 "baseline_s": result.prepared.meta["baseline_s"]})
            if record["reference_period_s"] is not None:
                expected_periods[obs.obs_id] = record["reference_period_s"]
    assert set(labels) == set(expected_periods)
    objects = {}
    for key, label in labels.items():
        assert label.period_s == expected_periods[key] and label.period_s not in (5000,6000,7000,9999)
        assert objects.setdefault(label.object_id, label.split) == label.split
    assert len(objects) >= 25
    assert bounds.min_period_s == min(expected_periods.values())
    assert bounds.max_period_s == max(expected_periods.values())

    manifest = json.loads((pilot / "run_config.json").read_text())
    _verify_publication(pilot, manifest)
    assert manifest["config"] == asdict(cfg)
    assert manifest["config_hash"] == config_hash(cfg)
    assert manifest["source_identity"] == _source_identity(ROOT)
    histogram = np.zeros(129, dtype=int)
    split_counts, group_counts = {}, {}
    seeds = set()
    train_symmetric = 0
    for split in ("train", "val", "test"):
        with np.load(pilot / f"pilot_{split}.npz", allow_pickle=False) as shard:
            rows = [json.loads(row) for row in shard["metadata"]]
            split_counts[split] = len(rows)
            groups = Counter()
            histogram += np.bincount(shard["label_bin"], minlength=129)
            for index, row in enumerate(rows):
                assert row["seed"] not in seeds
                seeds.add(row["seed"])
                source = row["source"]
                groups["noise_or_constant" if source in ("noise", "constant") else source] += 1
                left, right = shard["offsets"][index:index+2]
                times = shard["t_s"][left:right]
                assert times[0] == 0 and (np.diff(times) > 0).all()
                if source in ("reliable", "under_covered"):
                    params = SimParams(**row["parents"][0]["params"])
                    n = cfg.simulation["samples_per_cycle"]
                    truth_t = np.arange(3*n) / n * params.phase_cycle_s
                    repeat = template_repeat(simulate_lightcurve(params, truth_t), params.phase_cycle_s,
                        n, cfg.simulation["repeat_tolerance"], cfg.simulation["flat_rtol"],
                        wave=lambda t: simulate_lightcurve(params, t))
                    assert repeat == row["parent_repeat_s"]
                    if source == "reliable":
                        assert row["apparent_period_s"] == repeat
                        assert times[-1] / repeat >= cfg.simulation["min_cycles"]
                        if split == "train":
                            assert repeat == params.phase_cycle_s / 2
                            train_symmetric += 1
                    else:
                        assert row["apparent_period_s"] is None
                        assert times[-1] / repeat < cfg.simulation["min_cycles"]
            group_counts[split] = dict(groups)
    assert sum(split_counts.values()) == 1000
    assert group_counts == cfg.simulation["quotas"]
    timing = json.loads((ROOT / "outputs/cp1_pilot_timing.json").read_text())
    return {"real_files": len(records), "real_points": point_count,
            "raw_hashes_verified": True, "controller_quality_counts": dict(qualities),
            "controller_verdict_counts": dict(verdicts), "real_observations": real_details,
            "profiles": len(profiles), "reference_labels": len(labels), "reference_objects": len(objects),
            "reference_splits": dict(Counter(label.split for label in labels.values())),
            "period_bounds": cfg.period_bounds, "pilot_directory": str(pilot),
            "pilot_curves": sum(split_counts.values()), "pilot_splits": split_counts,
            "pilot_source_quotas": group_counts, "label_histogram": histogram.tolist(),
            "train_symmetric_half_cycle_labels": train_symmetric,
            "pilot_wall_seconds": timing["wall_seconds"],
            "pilot_under_three_minutes": timing["wall_seconds"] < 180,
            "pilot_manifest_and_all_shards_verified": True,
            "sampling_profile_usage": manifest["sampling_profile_usage"],
            "config_hash": manifest["config_hash"], "generator_source_sha256": manifest["source_identity"]["sha256"],
            "redistributable_samples": "BLOCKED: redistribution terms unconfirmed; local research authorized"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = validate(args.pilot.resolve())
    serialized = json.dumps(report, indent=2, allow_nan=False)
    if args.report:
        with args.report.open("x", encoding="utf-8") as handle:
            handle.write(serialized + "\n")
    print(serialized)
    if not report["pilot_under_three_minutes"]:
        raise SystemExit("FAIL: pilot exceeds the three-minute acceptance budget")
