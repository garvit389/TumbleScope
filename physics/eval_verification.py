"""Measure verification on 50 clean synthetic development curves."""

import argparse
from collections import Counter
from importlib.metadata import version
import json
from pathlib import Path
import platform
from statistics import median

import numpy as np
import torch

from classical_analysis.window import compute_window
from core.config import load_config, resolve_period_bounds
from core.schemas import SCHEMA_VERSION, Observation, Cluster, ReconciliationResult, stage2_pending
from ingestion import load_observation
from ml.bundle import digest
from ml.dataset.corpus import ROOT
from ml.dataset.degrade import profile_pools, timeline
from ml.infer import _CALIBRATION, _CHECKPOINT, run_cnn
from ml.train import save_json
from physics.params import SimParams
from physics.simulator import simulate_lightcurve
from physics.verify import _eligible_decoys, verify_candidates
from pipeline import controller
from preprocessing.prepare import prepare
from scripts.prepare_cp3b_holdout import CORPUS, HOLDOUT, _provenance


_FILES = (
    "00029A_2_3_R_20190322_DATA.txt",
    "00080B_1_2_R_20190724_DATA.txt",
    "00041B_2_3_R_20191118_DATA.txt",
)
_RESULTS = ROOT / "docs/results"


def _clean_case(config, row, profiles):
    seed = row["seed_path"]
    profile_rng = np.random.default_rng(np.random.SeedSequence(seed + [1]))
    time_rng = np.random.default_rng(np.random.SeedSequence(seed + [2]))
    profile = profiles[int(profile_rng.integers(len(profiles)))]
    if profile["profile_id"] != row["profile_id"]:
        raise ValueError("Development sampling profile differs from recorded provenance")
    times = timeline(profile, time_rng)
    params = SimParams(**row["parents"][0]["params"])
    brightness = simulate_lightcurve(params, times)
    observation = Observation(str(row["group"]), times, brightness, None, "unknown")
    prep = prepare(observation, config.preprocess)
    period = row["apparent_period_s"]
    candidate = Cluster(1, period, {}, {}, 0., [], 1.,
                        {"passed": True, "cycles": prep.meta["baseline_s"] / period, "fail_reasons": []})
    recon = ReconciliationResult([candidate], [1], True, [])
    recon.validate()
    return prep, recon


def _run_case(case, cfg):
    prep, recon = case
    result = verify_candidates(prep, recon, cfg)
    fit = result.per_candidate[0]
    return {"group": int(prep.obs_id), "samples": int(prep.valid_mask.sum()),
            "runtime_s": result.runtime_s, **fit.to_dict()}


def _summarize(rows):
    finished = [row for row in rows if row["status"] == "ok"]
    wins = sum(bool(row["decoy_delta_e"]) and min(row["decoy_delta_e"]) > 0 for row in finished)
    failures = Counter(row["notes"][-1] for row in rows if row["status"] != "ok")
    return {"attempted": len(rows), "completed": len(finished),
            "passed_both_provisional_gates": sum(row["passed"] for row in finished),
            "beat_all_decoys": wins, "wins_per_completed": wins / len(finished) if finished else None,
            "wins_per_attempted": wins / len(rows) if rows else None,
            "failure_counts": dict(failures),
            "completed_median_runtime_s": median(row["runtime_s"] for row in finished) if finished else None,
            "completed_max_runtime_s": max((row["runtime_s"] for row in finished), default=None)}


def _environment():
    return {"platform": platform.platform(), "processor": platform.processor(),
            "python": platform.python_version(), "numpy": version("numpy"), "scipy": version("scipy"),
            "torch": str(torch.__version__), "torch_threads": torch.get_num_threads()}


def evaluate(config):
    manifest = json.loads((CORPUS / "manifest.json").read_bytes())
    rows = sorted((row for row in _provenance("val", manifest) if row["category"] == "reliable"),
                  key=lambda row: row["group"])
    if len(rows) < 50:
        raise ValueError("Fewer than 50 reliable development groups")
    pools, source = profile_pools(ROOT / "data/profiles", config.simulation["seed"])
    if digest(source.read_bytes()) != manifest["profiles_sha256"]:
        raise ValueError("Sampling profiles differ from the CP2 corpus")
    bounds = tuple(resolve_period_bounds(config.period_bounds))
    policies = {"period_min_s": bounds[0], "period_max_s": bounds[1]}
    cfg = {**config.verification, **policies}
    timing_cases = []
    cases = []
    for row in rows:
        case = _clean_case(config, row, pools["val"])
        if len(cases) < 50:
            cases.append(case)
        if len(timing_cases) < 10:
            prep, recon = case
            window = compute_window(prep, policies)
            eligible, _ = _eligible_decoys(recon.clusters[0].period_s, [], window, bounds,
                                            cfg["decoy_ratios"], cfg["decoy_exclusion_resolutions"])
            if len(eligible) >= cfg["min_decoys"]:
                timing_cases.append(case)
        if len(timing_cases) == 10 and len(cases) == 50:
            break
    if len(timing_cases) != 10:
        raise ValueError("Fewer than ten eligible development timing groups")

    def run_case(case, policy):
        window = compute_window(case[0], {**config.classical, **policies})
        return _run_case(case, {**policy, "window_result": window})

    original = [run_case(case, cfg) for case in timing_cases]
    timed = _summarize(original)
    slow = ("timeout" in timed["failure_counts"]
            or (timed["completed_median_runtime_s"] or 0.) > 30.)
    if slow:
        fit = dict(cfg["fit"], coarse_samples=8, restarts=1, popsize=4, maxiter=5)
        cfg = {**cfg, "fit": fit}
        timing_rows = [run_case(case, cfg) for case in timing_cases]
    else:
        timing_rows = original
    known = {row["group"]: row for row in timing_rows}
    measured = [known.get(int(case[0].obs_id)) or run_case(case, cfg) for case in cases]
    revision = controller._repository_revision()
    return {"schema_version": SCHEMA_VERSION, "revision": revision, "environment": _environment(),
            "population": "First 50 reliable synthetic development groups, without degradation",
            "provisional": True, "scope": "Single truth candidate per curve",
            "verification_policy": cfg, "timing_profile": "lighter" if slow else "existing",
            "initial_timing": timed, "timing": _summarize(timing_rows),
            "timing_groups": [row["group"] for row in timing_rows],
            "planned": 50, "summary": _summarize(measured), "rows": measured}


def measure_runtime(config, evidence):
    low, high = resolve_period_bounds(config.period_bounds)
    cfg = dict(config.cnn, period_min_s=low, period_max_s=high)
    files = [ROOT / "data/sdlcd_raw" / name for name in _FILES]
    typical = prepare(load_observation(str(files[0])), config.preprocess)
    predictions = [run_cnn(typical, cfg) for _ in range(4)]
    rows = []
    for file in files:
        result = controller.analyze(load_observation(str(file)), config)
        result.validate()
        group = result.reconciliation
        rows.append({
            "name": file.name, "sha256": digest(file.read_bytes()), "config_hash": result.config_hash,
            "quality": result.quality.to_dict() if result.quality is not None else None,
            "estimators": {method: {"status": estimate.status, "candidates": len(estimate.candidates),
                                    "windows": estimate.meta.get("n_windows")}
                           for method, estimate in result.estimators.items()},
            "shortlist": group.shortlist if group else [],
            "reconciliation_stub": group.is_stub if group else None,
            "verification_complete": result.verification.complete if result.verification else None,
            "verification_ids": [row.cluster_id for row in result.verification.per_candidate]
            if result.verification else [],
            "verdict": result.verdict.label if result.verdict else None,
            "reasons": result.verdict.reason_codes if result.verdict else [],
            "pending": stage2_pending(group, result.estimators, result.verification),
            "stage_errors": result.stage_errors, "timings_s": result.timings_s,
        })
    holdout = None
    if HOLDOUT.is_file():
        with np.load(HOLDOUT, allow_pickle=False) as shard:
            holdout = {"count": len(shard["groups"]), "groups": shard["groups"].tolist(),
                       "bounds": shard["bounds"].tolist(), "size_bytes": HOLDOUT.stat().st_size,
                       "sha256": digest(HOLDOUT.read_bytes()), "seed": config.simulation["seed"],
                       "fields": {"t_s": "flattened seconds", "brightness": "flattened measured values",
                                  "offsets": "indices into flattened arrays", "periods": "truth seconds, 0 for NRP",
                                  "groups": "disjoint development group IDs", "bounds": "shared period bounds"}}
    return {
        "revision": evidence["revision"], "environment": _environment(),
        "checkpoint_sha256": digest(_CHECKPOINT.read_bytes()),
        "calibration_sha256": digest(_CALIBRATION.read_bytes()),
        "cnn": {"file": files[0].name, "cold_runtime_s": predictions[0].meta["runtime_s"],
                "repeated_runtime_s": [result.meta["runtime_s"] for result in predictions[1:]],
                "statuses": [result.status for result in predictions],
                "windows": [result.meta.get("n_windows") for result in predictions]},
        "verification_timing": evidence["timing"], "verification_profile": evidence["timing_profile"],
        "verification_folds": evidence["verification_policy"]["n_blocks"],
        "verification_families": evidence["verification_policy"]["fit"]["shapes"],
        "verification_decoys_per_curve": [len(row["decoy_periods_s"]) for row in evidence["rows"]],
        "real_file_diagnostics": rows, "holdout": holdout,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=_RESULTS / "verification_cp3b.json")
    parser.add_argument("--runtime-output", type=Path, default=_RESULTS / "runtime_cp3b.json")
    args = parser.parse_args()
    config = load_config()
    evidence = evaluate(config)
    save_json(args.output, evidence)
    print(json.dumps(evidence["summary"]), flush=True)
    save_json(args.runtime_output, measure_runtime(config, evidence))
