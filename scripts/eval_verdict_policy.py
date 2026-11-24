"""Compare existing verdict policies on the checked CP3B development shard."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import hashlib
from io import BytesIO
from itertools import product
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from core.config import config_hash, load_config, resolve_period_bounds
from core.schemas import Observation, stage2_pending
from ml.train import save_json
from pipeline import cache, controller
from verdict.decide import decide


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/day4_local/verdict_policy.json"
SHARD = ROOT / "data/synthetic_holdout/holdout_small.npz"
REPORT = ROOT / "docs/results/runtime_cp3b.json"


def _load_shard(raw, identity, bounds):
    if hashlib.sha256(raw).hexdigest() != identity["sha256"]:
        raise ValueError("Development shard checksum mismatch")
    with np.load(BytesIO(raw), allow_pickle=False) as shard:
        if set(shard.files) != {"groups", "offsets", "periods", "bounds", "t_s", "brightness"}:
            raise ValueError("Unexpected development shard fields")
        groups, offsets, periods, limits = (shard[key] for key in ("groups", "offsets", "periods", "bounds"))
        times, brightness = shard["t_s"], shard["brightness"]
    if (groups.shape != (200,) or groups.dtype.kind not in "iu"
            or len(set(groups)) != 200 or np.any(groups // 1_000_000 != 1)
            or groups.tolist() != identity["groups"]
            or Counter(groups // 100_000 % 10) != {0: 140, 1: 20, 2: 20, 3: 20}
            or offsets.shape != (201,) or offsets.dtype.kind not in "iu"
            or offsets[0] != 0 or not np.all(offsets[1:] > offsets[:-1])
            or times.ndim != 1 or times.shape != brightness.shape or offsets[-1] != len(times)
            or limits.tolist() != list(bounds) or identity["bounds"] != list(bounds)
            or not np.isfinite(times).all() or np.isinf(brightness).any()
            or periods.shape != (200,) or not np.isfinite(periods).all()):
        raise ValueError("Invalid development shard")
    reliable = groups // 100_000 % 10 == 0
    if (np.any(periods[~reliable] != 0)
            or np.any((periods[reliable] < bounds[0]) | (periods[reliable] > bounds[1]))):
        raise ValueError("Invalid development truth")
    curves = []
    for i, group in enumerate(groups):
        part = slice(offsets[i], offsets[i + 1])
        obs = Observation(str(group), times[part], brightness[part], None, "unknown")
        obs.validate()
        curves.append((obs, float(periods[i])))
    return curves


def _disposition(result, verdict):
    if verdict is not None:
        return verdict.label
    if stage2_pending(result.reconciliation, result.estimators, result.verification):
        return "pending"
    return "execution_error"


def _replay(result, cfg):
    if result.verdict is None and "verdict" not in result.stage_errors:
        return None
    try:
        verdict = decide(result.quality, result.estimators, result.reconciliation, result.verification,
                         controller._decision_config(cfg))
        verdict.validate()
        return verdict
    except ValueError:
        return controller._fallback_verdict(result.quality, result.estimators,
                                            result.reconciliation, result.verification)


def _match(truth, prediction, bounds):
    if prediction is None or truth == 0:
        return None
    ratio = np.log(prediction / truth)
    width = np.log(bounds[1] / bounds[0]) / 128 + 4 * np.finfo(float).eps
    for name, offset in (("direct", 0), ("half", -np.log(2)), ("double", np.log(2))):
        if abs(ratio - offset) <= width:
            return name
    return "other"


def _ratio(numerator, denominator):
    return dict(numerator=numerator, denominator=denominator,
                value=numerator / denominator if denominator else None)


def _summary(rows):
    outcomes = Counter(row["outcome"] for row in rows)
    supported = [row for row in rows if row["outcome"] == "Supported"]
    evaluable = [row for row in supported if row["prediction_s"] is not None or row["truth_s"] == 0]
    correct = sum(row["match"] == "direct" for row in evaluable)
    incorrect = len(evaluable) - correct
    return {
        "attempted": len(rows), "groups": len({row["group"] for row in rows}),
        "outcomes": dict(outcomes), "abstained": outcomes["Ambiguous"] + outcomes["Inconclusive"],
        "matches": dict(Counter(row["match"] for row in evaluable if row["match"] is not None)),
        "supported_missing_prediction": len(supported) - len(evaluable),
        "false_supported": _ratio(incorrect, len(evaluable)),
        "false_supported_per_labelled_attempt": _ratio(incorrect, len(rows)),
        "coverage": _ratio(len(supported), len(rows)),
        "selective_accuracy": _ratio(correct, len(evaluable)),
    }


def _analyze_group(cfg_and_observation):
    cfg, obs = cfg_and_observation
    original = controller._OUTPUTS
    with TemporaryDirectory() as directory:
        try:
            controller._OUTPUTS = Path(directory)
            return controller.analyze(obs, cfg)
        finally:
            controller._OUTPUTS = original


def _single_torch_thread():
    torch.set_num_threads(1)


def evaluate(cfg, curves, identity, *, results=None):
    bounds = resolve_period_bounds(cfg.period_bounds)
    policies = [replace(cfg, verdict={**cfg.verdict, "min_agreeing_estimators": minimum,
                                     "limited_max_label": cap})
                for minimum, cap in product((2, 3), ("Supported", "Ambiguous"))]
    records = {config_hash(policy): [] for policy in policies}
    baseline = config_hash(cfg)
    if baseline not in records:
        raise ValueError("Configured baseline is outside the comparison grid")
    parity = []
    verified = set()
    if results is None:
        results = (controller.analyze(obs, cfg) for obs, _ in curves)
    for (obs, truth), result in zip(curves, results, strict=True):
        result.validate()
        if result.obs_id != obs.obs_id or result.config_hash != baseline:
            raise ValueError("Development baseline identity differs from its input")
        for policy, key in zip(policies, records):
            verdict = result.verdict if key == baseline else _replay(result, policy)
            prediction = verdict.best_period_s if verdict else None
            changed = verdict != result.verdict
            if key != baseline and changed and key not in verified:
                fresh = controller.analyze(obs, policy)
                fresh.validate()
                if fresh.verdict != verdict or _disposition(fresh, fresh.verdict) != _disposition(result, verdict):
                    raise ValueError(f"Verdict replay differs for development group {obs.obs_id}")
                parity.append({"group": obs.obs_id, "policy": key, "changed": True})
                verified.add(key)
            records[key].append({
                "group": obs.obs_id, "quality": result.quality.status if result.quality else "unavailable",
                "outcome": _disposition(result, verdict), "truth_s": truth, "prediction_s": prediction,
                "match": _match(truth, prediction, bounds), "reason_codes": verdict.reason_codes if verdict else [],
                "stage_error_keys": sorted(
                    stage for stage in result.stage_errors if stage != "verdict" or verdict is None
                ) + (["verdict"] if result.verdict is not None and verdict is None else []),
            })
        if len(records[baseline]) % 10 == 0:
            print(f"Analyzed {len(records[baseline])}/{len(curves)} development groups", flush=True)
    return {
        "scope": "Synthetic development policy diagnostic using validation variants",
        "provisional": True, "planned": len(curves), "baseline_hash": baseline,
        "input": identity, "config": asdict(cfg), "bounds": list(bounds),
        "direct_hit": "Absolute log apparent-period ratio within one configured log bin (128 bins)",
        "replay_checks": parity,
        "policies": [{
            "config_hash": key, "verdict_policy": policy.verdict, "summary": _summary(rows),
            "quality_strata": {quality: _summary([row for row in rows if row["quality"] == quality])
                               for quality in sorted({row["quality"] for row in rows})},
            "rows": rows,
        } for policy, (key, rows) in zip(policies, records.items())],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1, help="Local workers for independent observations")
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= (os.cpu_count() or 1):
        parser.error("--workers must be between one and the local logical CPU count")
    cfg = load_config()
    identity = json.loads(REPORT.read_bytes())["holdout"]
    if identity["count"] != 200 or identity["seed"] != cfg.simulation["seed"]:
        raise ValueError("Development cohort identity differs from the operating configuration")
    curves = _load_shard(SHARD.read_bytes(), identity, resolve_period_bounds(cfg.period_bounds))
    implementation = cache._implementation_identity()
    implementation["sources"]["scripts/eval_verdict_policy.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    # The parent handles replay checks, while each child analyzes a whole observation.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    _single_torch_thread()
    provenance = {"implementation": implementation, "artifacts": cache._artifact_identity(),
                  "revision": controller._repository_revision(), "workers": args.workers,
                  "torch_threads_per_worker": torch.get_num_threads()}
    original = controller._OUTPUTS
    try:
        controller._OUTPUTS = OUTPUT.parent
        if args.workers == 1:
            report = evaluate(cfg, curves, identity)
        else:
            with ProcessPoolExecutor(max_workers=args.workers, initializer=_single_torch_thread) as pool:
                jobs = ((cfg, obs) for obs, _ in curves)
                report = evaluate(cfg, curves, identity, results=pool.map(_analyze_group, jobs))
    finally:
        controller._OUTPUTS = original
    save_json(OUTPUT, {**report, "provenance": provenance})


if __name__ == "__main__":
    main()
