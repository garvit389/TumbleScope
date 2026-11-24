"""Measure verification on synthetic development and independent evaluation curves."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
from statistics import median
import sys

import numpy as np
import torch

from classical_analysis.window import compute_window
from core.config import config_hash, load_config, resolve_period_bounds
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
from scripts.prepare_verification_auc import COHORT, read_curves, read_metadata


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


def _auc_statistics(rows):
    from sklearn.metrics import roc_auc_score

    labels, scores, matched = [], [], []
    failures = Counter()
    for row in rows:
        if row["status"] != "ok":
            failures[row["notes"][-1] if row["notes"] else "fit failed"] += 1
            continue
        error, deltas = row["heldout_error"], row["decoy_delta_e"]
        if (not deltas or len(deltas) != len(row["decoy_periods_s"])
                or not np.isfinite(error) or not np.isfinite(deltas).all()):
            failures["no complete finite comparison"] += 1
            continue
        decoy_errors = [error + delta for delta in deltas]
        if not np.isfinite(decoy_errors).all():
            failures["no complete finite comparison"] += 1
            continue
        labels.extend([1, *([0] * len(deltas))])
        scores.extend([-error, *(-value for value in decoy_errors)])
        matched.append(sum(1.0 if d > 0 else 0.5 if d == 0 else 0.0 for d in deltas) / len(deltas))
    return {
        "attempted": len(rows), "unique_groups": len({row["group"] for row in rows}),
        "completed": len(matched), "failed": sum(failures.values()), "failure_counts": dict(failures),
        "n_truth_scores": len(matched), "n_decoy_scores": labels.count(0),
        "n_matched_comparisons": labels.count(0),
        "n_excluded_decoys": sum(note.startswith("Excluded decoy ") for row in rows for note in row["notes"]),
        "pooled_candidate_roc_auc": float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None,
        "mean_within_curve_win_rate": sum(matched) / len(matched) if matched else None,
    }


def _independent_decoys(period, window, policy):
    """Exclude truth aliases and P/2P even for a truth-only shortlist."""
    tolerance = window["freq_resolution"] * policy["decoy_exclusion_resolutions"]
    frequency = 1 / period
    harmonics = (frequency / 2, 2 * frequency)
    aliases = [
        abs(frequency + sign * peak["freq_hz"])
        for peak in window["peaks"] for sign in (-1, 1)
    ]
    eligible, exclusions = [], []
    for ratio in policy["decoy_ratios"]:
        decoy = period * ratio
        if any(abs(1 / decoy - f) <= tolerance for f in harmonics):
            exclusions.append(f"Excluded decoy {decoy:.12g} s (truth harmonic)")
        elif any(abs(1 / decoy - f) <= tolerance for f in aliases):
            exclusions.append(f"Excluded decoy {decoy:.12g} s (truth sampling alias)")
        else:
            eligible.append(ratio)
    return eligible, exclusions


def _auc_row(job):
    prep, recon, config, bounds, strict_decoys = job
    truth_id = recon.shortlist[0]
    truth = next(cluster for cluster in recon.clusters if cluster.cluster_id == truth_id)
    exclusions = []
    try:
        window = compute_window(prep, {**config.classical, **bounds})
        policy = {**config.verification, **bounds, "window_result": window}
        if strict_decoys:
            policy["decoy_ratios"], exclusions = _independent_decoys(truth.period_s, window, policy)
            # The verifier requires at least one proposed ratio in its policy.
            if not policy["decoy_ratios"]:
                raise ValueError("no_valid_decoy")
        result = verify_candidates(prep, recon, policy)
        record = next(candidate for candidate in result.per_candidate if candidate.cluster_id == truth_id)
        row = record.to_dict()
    except (ValueError, ArithmeticError, RuntimeError) as exc:
        row = {"status": "failed", "notes": [str(exc) or type(exc).__name__]}
    row["notes"] = exclusions + row["notes"]
    return {"group": int(prep.obs_id), "apparent_period_s": truth.period_s, **row}


def evaluate_auc_cases(config, cases, *, population, workers=1, strict_decoys=False):
    """Evaluate clean cases reconstructed from separately established provenance."""
    cases = sorted(cases, key=lambda case: int(case[0].obs_id))
    if len({case[0].obs_id for case in cases}) != len(cases):
        raise ValueError("Duplicate evaluation group")
    low, high = resolve_period_bounds(config.period_bounds)
    bounds = {"period_min_s": low, "period_max_s": high}
    policy = {**config.verification, **bounds}
    if any(len(recon.shortlist) != 1 for _, recon in cases):
        raise ValueError("Each evaluation case must identify exactly one truth candidate")
    jobs = ((prep, recon, config, bounds, strict_decoys) for prep, recon in cases)
    if workers == 1:
        rows = list(map(_auc_row, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_auc_row, jobs))
    return {
        "scope": "Synthetic verification evidence", "population": population,
        "planned": len(cases), "schema_version": SCHEMA_VERSION,
        "config_hash": config_hash(config), "verification_policy": policy,
        "simulation_seed": config.simulation["seed"], "environment": _environment(),
        "workers": workers,
        "score": "negative held-out error",
        "summary": _auc_statistics(rows), "rows": rows,
    }


def _production_cases(config, manifest):
    times, brightness = read_curves(COHORT, manifest)
    for metadata, t_s, y in zip(manifest["rows"], times, brightness, strict=True):
        obs = Observation(str(metadata["group"]), t_s, y, None, "unknown")
        obs.validate()
        prepared = prepare(obs, config.preprocess)
        period = metadata["apparent_period_s"]
        truth = Cluster(1, period, {}, {}, 0., [], 1.,
                        {"passed": True, "cycles": prepared.meta["baseline_s"] / period, "fail_reasons": []})
        recon = ReconciliationResult([truth], [1], True, [])
        recon.validate()
        yield prepared, recon


def _production_auc(config, args):
    try:
        manifest, identity = read_metadata(config, COHORT)
        if not (COHORT / "curves.npz").is_file():
            raise FileNotFoundError("Independent cohort payload is missing")
        if args.check_inputs:
            print(json.dumps({"status": "eligible_metadata", "allocation": manifest["design"]["allocation"],
                              "curves": len(manifest["rows"]), "manifest_sha256": identity}))
            return 0
        cases = list(_production_cases(config, manifest))
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        print(json.dumps({"status": "blocked_no_eligible_cohort", "reason": str(exc)}), flush=True)
        return 2

    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    sources = ("physics/eval_verification.py", "physics/verify.py", "physics/fit.py",
               "preprocessing/prepare.py", "classical_analysis/window.py")
    code = {name: sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}
    evidence = evaluate_auc_cases(config, cases, population="Independent clean held-out simulator variants",
                                  workers=args.workers, strict_decoys=True)
    evidence.pop("simulation_seed")
    evidence.update({
        "cohort": manifest["design"],
        "manifest_sha256": identity,
        "curves_sha256": manifest["curves_sha256"],
        "sources_sha256": code,
        "decoy_rule": ("Contract verifier exclusions plus P/2P and truth-frequency sampling aliases "
                       "within the configured frequency resolution width"),
        "roc_auc": evidence["summary"]["pooled_candidate_roc_auc"],
        "n": evidence["summary"]["completed"],
    })
    by_group = {row["group"]: row["variant"] for row in manifest["rows"]}
    for row in evidence["rows"]:
        row["variant"] = by_group[row["group"]]
    evidence["by_variant"] = {
        str(index): _auc_statistics([row for row in evidence["rows"] if row["variant"] == index])
        for index in range(len(manifest["design"]["variants"]))
    }
    save_json(args.output or _RESULTS / "verification_auc.json", evidence)
    print(json.dumps(evidence["summary"]), flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--runtime-output", type=Path, default=_RESULTS / "runtime_cp3b.json")
    parser.add_argument("--auc", action="store_true", help="Evaluate on an eligible held-out variant cohort")
    parser.add_argument("--check-inputs", action="store_true", help="Check AUC allocation metadata only")
    parser.add_argument("--workers", type=int, default=1, help="Local workers for independent AUC curves")
    args = parser.parse_args(argv)
    if args.auc:
        if not 1 <= args.workers <= (os.cpu_count() or 1):
            parser.error("--workers must be between one and the local logical CPU count")
        return _production_auc(load_config(), args)
    if args.check_inputs:
        parser.error("--check-inputs requires --auc")
    config = load_config()
    evidence = evaluate(config)
    save_json(args.output or _RESULTS / "verification_cp3b.json", evidence)
    print(json.dumps(evidence["summary"]), flush=True)
    save_json(args.runtime_output, measure_runtime(config, evidence))
    return 0


if __name__ == "__main__":
    sys.exit(main())
