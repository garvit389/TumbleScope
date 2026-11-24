"""Capture Day 4 reconciliation inputs and replay the fixed validation cohorts."""

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from core.config import config_hash
from core.schemas import (
    SCHEMA_VERSION, EstimatorResult, PreparedObservation, ReconciliationResult,
    assert_contract_equal,
)
from evaluation import metrics, run_eval
from pipeline import controller
from reconciliation.reconcile import _reconcile

COUNTS = {"real-val": 18, "synthetic-dev": 200}
WEIGHTS = ((0.15, 0.25, 0.60), (0.10, 0.20, 0.70), (0.05, 0.15, 0.80))
PENALTIES = (0.10, 0.20, 0.30)
WEIGHT_KEYS = ("w_C", "w_L", "w_D", "lambda_A")
TOLERANCE = 0.05
CACHE_VERSION = 1
BOOTSTRAP_SEED = 17
BOOTSTRAP_DRAWS = 2000


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value):
    return sha256(_canonical(value).encode("utf-8")).hexdigest()


def _write_new(path, value):
    payload = _canonical(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload + "\n")


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _code_hashes():
    root = run_eval.ROOT
    paths = [path for name in ("evaluation", "reconciliation", "classical_analysis", "preprocessing",
                               "core", "pipeline", "ml", "physics", "verdict", "ingestion")
             for path in sorted((root / name).glob("*.py"))]
    return {path.relative_to(root).as_posix(): run_eval._digest(path) for path in paths}


def _number(value):
    return float(value) if value else None


@dataclass
class Baseline:
    dataset: str
    cfg: object
    cases: list
    rows: dict
    summary: list
    options: dict
    provenance: dict


def load_baseline(directory, dataset):
    """Resolve labels and identity from the canonical Day 4 sources."""
    count = COUNTS[dataset]
    directory = directory / dataset
    paths = {name: directory / name for name in ("manifest.json", "runs.csv", "summary.json")}
    digests = {name: run_eval._digest(path) for name, path in paths.items()}
    cfg, manifest = run_eval._read_replay(paths["manifest.json"])
    split, source = run_eval.DATASETS[dataset]
    options = manifest["options"]
    if (manifest["dataset"], manifest["split"], manifest["source"], manifest["n_observations"]) != (
        dataset, split, source, count
    ) or options != run_eval.DEFAULTS or manifest["mode"] != "fast":
        raise ValueError(f"{dataset}: expected the full clean Day 4 Fast baseline")
    root = run_eval.ROOT
    if run_eval._digest(root / source) != manifest["source_sha256"]:
        raise ValueError(f"{dataset}: baseline source differs from the current source")
    artifacts = {name: run_eval._digest(root / name) for name in run_eval.ARTIFACTS}
    if artifacts != manifest["cnn_artifacts_sha256"] or run_eval._digest(
        Path(run_eval.__file__)
    ) != manifest["runner_sha256"]:
        raise ValueError(f"{dataset}: baseline runner or CNN artifacts changed")
    if dataset == "real-val":
        cases = list(run_eval._real_cases(root / source, root / "data/sdlcd_raw", None))
    else:
        cases = list(run_eval._synthetic_cases(root / source, cfg, None, manifest["source_sha256"]))
    identities = {case.obs_id: case for case in cases}
    inputs = [dict(obs_id=case.obs_id, sha256=case.source_sha256, status=case.input_status)
              for case in cases]
    if len(identities) != count or inputs != manifest["inputs"]:
        raise ValueError(f"{dataset}: baseline observation inventory differs")
    rows = defaultdict(dict)
    with paths["runs.csv"].open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            case = identities.get(row["obs_id"])
            name = row["configuration"]
            expected_nrp = "" if case is None or case.no_reliable_period is None else str(case.no_reliable_period)
            if (case is None or row["split"] != split or row["dataset"] != dataset
                    or row["variant"] != "clean" or name not in run_eval.CONFIGURATIONS
                    or name in rows[row["obs_id"]] or _number(row["reference_period_s"]) != case.period_s
                    or row["no_reliable_period"] != expected_nrp):
                raise ValueError(f"{dataset}: unexpected baseline observation, split, label or configuration")
            rows[case.obs_id][name] = row
    if rows.keys() != identities.keys() or any(len(row) != 5 for row in rows.values()):
        raise ValueError(f"{dataset}: incomplete baseline CSV")
    return Baseline(dataset, cfg, cases, rows, _read_json(paths["summary.json"]), options,
                    dict(reports=digests, config_hash=manifest["config_hash"], artifacts=artifacts))


def _serialized_estimator(estimator):
    data = estimator.to_dict()
    data["meta"]["runtime_s"] = 0.0
    return data


def capture_case(observation, cfg):
    """Capture the original controller seam, without passing scoring data."""
    captured = {}

    def tap(prepared, estimators, settings):
        if settings["top_k"] != 3:
            raise ValueError("expected the Fast top-three reconciliation policy")
        if any(estimator.candidates for estimator in estimators) and "window_result" not in settings:
            raise ValueError("candidate-bearing reconciliation requires the injected window")
        captured.update(prepared=prepared.to_dict(), settings=settings,
                        estimators=[_serialized_estimator(item) for item in estimators])
        return original(prepared, estimators, settings)

    original = controller.reconcile
    with patch.object(controller, "reconcile", tap):
        result = controller.analyze(observation, cfg)
    result.validate()
    if result.obs_id != observation.obs_id or result.config_hash != config_hash(cfg):
        raise ValueError("analysis identity or config mismatch")
    if not captured:
        if result.reconciliation is not None or any(item.candidates for item in result.estimators.values()):
            raise ValueError("candidate evidence exists without a captured reconciliation call")
        captured.update(prepared=result.prepared.to_dict() if result.prepared else None,
                        settings=None, estimators=[
                            _serialized_estimator(item) for item in result.estimators.values()
                        ])
    captured["original"] = result.reconciliation.to_dict() if result.reconciliation else None
    captured["reconciliation_error"] = result.stage_errors.get("reconciliation")
    if captured["reconciliation_error"]:
        raise ValueError(f"cannot capture a reproducible reconciliation: {captured['reconciliation_error']}")
    return captured


def make_cache(baseline, limit=None):
    records = []
    with TemporaryDirectory(prefix="tumblescope-tuning-") as scratch:
        with patch.object(controller, "_OUTPUTS", Path(scratch)):
            for case in baseline.cases[:limit]:
                snapshot = capture_case(case.observation, baseline.cfg) if case.observation is not None else None
                records.append(dict(obs_id=case.obs_id, source_sha256=case.source_sha256, snapshot=snapshot))
    payload = dict(cache_version=CACHE_VERSION, schema_version=SCHEMA_VERSION, dataset=baseline.dataset,
                   provenance=baseline.provenance, code_sha256=_code_hashes(), records=records)
    return dict(sha256=_hash(payload), payload=payload)


def load_cache(path, baseline):
    envelope = _read_json(path)
    data = envelope["payload"]
    if (envelope["sha256"] != _hash(data) or data["cache_version"] != CACHE_VERSION
            or data["schema_version"] != SCHEMA_VERSION or data["dataset"] != baseline.dataset
            or data["provenance"] != baseline.provenance or data["code_sha256"] != _code_hashes()):
        raise ValueError("cache provenance, integrity, or schema mismatch")
    records = data["records"]
    if not records or [(row["obs_id"], row["source_sha256"]) for row in records] != [
        (case.obs_id, case.source_sha256) for case in baseline.cases[:len(records)]
    ]:
        raise ValueError("cache records do not match the approved baseline cohort")
    return records


def _csv_reading(row):
    return dict(status=row["status"], period_s=_number(row["period_s"]),
                candidate_periods_s=json.loads(row["candidate_periods_s"]),
                ranking_score=_number(row["ranking_score"]))


def _fused(result, original_row):
    if result is None:
        return dict(status=original_row["status"], period_s=None, candidate_periods_s=[], ranking_score=None)
    by_id = {cluster.cluster_id: cluster for cluster in result.clusters}
    shortlist = [by_id[key] for key in result.shortlist]
    lead = shortlist[0] if shortlist else None
    return dict(status="predicted" if lead else original_row["status"]
                if original_row["status"] != "predicted" else "abstained",
                period_s=lead.period_s if lead else None,
                candidate_periods_s=[cluster.period_s for cluster in shortlist],
                ranking_score=lead.score if lead else None)


def _hit(reference, period):
    return reference is not None and period is not None and 0.95 <= period / reference <= 1.05


def _trace(result, membership, reference):
    if result is None:
        return []
    shortlist_ranks = {key: rank for rank, key in enumerate(result.shortlist, 1)}
    rows = []
    for rank, cluster in enumerate(result.clusters, 1):
        members = membership[cluster.cluster_id]
        rows.append(dict(cluster_id=cluster.cluster_id, raw_members=members,
                         upstream_match=any(_hit(reference, item["period_s"]) for item in members),
                         selected_match=any(item["selected_for_method"] and _hit(reference, item["period_s"])
                                            for item in members),
                         representative_match=_hit(reference, cluster.period_s),
                         period_s=cluster.period_s, score=cluster.score, rank=rank,
                         stage2=cluster.stage2, shortlist_rank=shortlist_ranks.get(cluster.cluster_id),
                         truncated=cluster.stage2["passed"] and cluster.cluster_id not in shortlist_ranks))
    return rows


def _operation(row):
    if row["status"] == "abstained":
        return "incomplete_verification" if row["verification_complete"] == "False" else "completed_abstention"
    if row["status"] == "execution_error":
        return "verdict_execution_error" if "verdict" in json.loads(row["stage_errors"]) else "execution_error"
    return row["status"]


def replay(baseline, records, overrides=None, *, verify=False, trace=False):
    if set(overrides or {}) - set(WEIGHT_KEYS):
        raise ValueError("initial search may only change estimator weights and alias penalty")
    rows, traces = [], []
    for case, record in zip(baseline.cases[:len(records)], records, strict=True):
        saved = baseline.rows[case.obs_id]
        readings = {name: _csv_reading(row) for name, row in saved.items()}
        snapshot = record["snapshot"]
        result = None
        membership = {} if trace else None
        if snapshot is not None:
            estimators = [EstimatorResult.from_dict(data) for data in snapshot["estimators"]]
            settings = snapshot["settings"]
            if settings is not None:
                if settings["top_k"] != 3 or (any(item.candidates for item in estimators)
                                               and "window_result" not in settings):
                    raise ValueError("cache lacks original Fast policy or injected spectral window")
                prep = PreparedObservation.from_dict(snapshot["prepared"])
                result = _reconcile(prep, estimators, {**settings, **(overrides or {})}, trace=membership)
            if verify:
                original = snapshot["original"]
                if (original is None) != (result is None):
                    raise ValueError(f"{case.obs_id}: captured reconciliation availability differs")
                if original is not None:
                    assert_contract_equal(result, ReconciliationResult.from_dict(original))
                by_method = {item.method: item for item in estimators}
                for name, method in run_eval.METHODS.items():
                    if [item.period_s for item in by_method[method].candidates] != readings[name]["candidate_periods_s"]:
                        raise ValueError(f"{case.obs_id}: cached {name} candidates differ from baseline")
        fused = _fused(result, saved["cnn_classical"])
        if verify and fused != readings["cnn_classical"]:
            raise ValueError(f"{case.obs_id}: ordered shortlist, score or status differs from baseline report")
        readings["cnn_classical"] = fused
        complete = saved["complete"]
        rows.append(dict(obs_id=case.obs_id, object_id=case.object_id, dataset=baseline.dataset,
                         split=complete["split"], variant="clean", quality=complete["quality"] or None,
                         reference_period_s=case.period_s, no_reliable_period=case.no_reliable_period,
                         p_no_reliable=_number(complete["p_no_reliable"]), configurations=readings,
                         baseline_operation=_operation(complete)))
        if trace:
            traces.append(dict(obs_id=case.obs_id, shortlist_ids=result.shortlist if result else [],
                               clusters=_trace(result, membership, case.period_s)))
    return rows, traces


def verify_baseline(baseline, records):
    rows, traces = replay(baseline, records, verify=True, trace=True)
    if len(records) == COUNTS[baseline.dataset]:
        if run_eval.summarize(rows, baseline.options) != baseline.summary:
            raise ValueError(f"{baseline.dataset}: aggregate baseline metrics differ")
    return rows, traces


def _stats(rows, name):
    readings = [row["configurations"][name] for row in rows]
    truth = [row["reference_period_s"] for row in rows]
    periods = [reading["period_s"] for reading in readings]
    return dict(
        top1=metrics.tolerance_accuracy(truth, periods, tolerance=TOLERANCE)["overall"],
        shortlist_recall_at_3=metrics.recall_at_k(
            truth, [reading["candidate_periods_s"] for reading in readings], k=3, tolerance=TOLERANCE),
        catastrophic=metrics.catastrophic_error_rate(truth, periods, factor=2.0),
        nrp_emissions=sum(row["no_reliable_period"] is True and period is not None
                          for row, period in zip(rows, periods, strict=True)),
        n_nrp=sum(row["no_reliable_period"] is True for row in rows),
    )


def _report(rows):
    claim = _stats(rows, "complete")["top1"]
    return dict(n=len(rows), fused=_stats(rows, "cnn_classical"), pdm_fixed=_stats(rows, "pdm_only"),
                baseline_supported_claim_recall=claim,
                baseline_operations=dict(Counter(row["baseline_operation"] for row in rows)))


def _rate(metric):
    return Fraction(metric["numerator"], metric["denominator"] or 1)


def _objective(stats):
    return sum(_rate(item["top1"]) for item in stats.values()) / len(stats)


def _safeguards(stats, base):
    return (all(_rate(stats[d][metric]) >= _rate(base[d][metric])
                for d in base for metric in ("top1", "shortlist_recall_at_3"))
            and stats["synthetic-dev"]["nrp_emissions"] <= 40)


def _stability(baseline_rows, trial_rows):
    """Paired bootstrap using real objects and synthetic development groups."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    gains = []
    for dataset, rows in baseline_rows.items():
        groups = defaultdict(list)
        for original, trial in zip(rows, trial_rows[dataset], strict=True):
            reference = original["reference_period_s"]
            if reference is None:
                continue
            key = original["object_id"] if dataset == "real-val" else original["obs_id"]
            groups[key].append(
                int(_hit(reference, trial["configurations"]["cnn_classical"]["period_s"]))
                - int(_hit(reference, original["configurations"]["cnn_classical"]["period_s"]))
            )
        values = list(groups.values())
        indices = rng.integers(len(values), size=(BOOTSTRAP_DRAWS, len(values)))
        gain = np.asarray([sum(items) for items in values])[indices].sum(axis=1)
        count = np.asarray([len(items) for items in values])[indices].sum(axis=1)
        gains.append(gain / count)
    return float(np.quantile(np.mean(gains, axis=0), 0.05))


def tune(baselines, records):
    if set(baselines) != set(COUNTS) or any(len(records[d]) != n for d, n in COUNTS.items()):
        raise ValueError("tuning requires both complete Day 4 cache cohorts")
    checked = {d: verify_baseline(baselines[d], records[d]) for d in COUNTS}
    base_rows = {d: value[0] for d, value in checked.items()}
    reports = {d: _report(rows) for d, rows in base_rows.items()}
    base = {d: report["fused"] for d, report in reports.items()}
    trials = []
    winner = {}
    best_key = (_objective(base), -sum(_rate(item["catastrophic"]) for item in base.values()), 0)
    for weights, penalty in product(WEIGHTS, PENALTIES):
        proposed = dict(zip(WEIGHT_KEYS, (*weights, penalty), strict=True))
        trial_rows = {d: replay(baselines[d], records[d], proposed)[0] for d in COUNTS}
        stats = {d: _stats(rows, "cnn_classical") for d, rows in trial_rows.items()}
        objective = _objective(stats)
        safeguards = _safeguards(stats, base)
        lower_gain = _stability(base_rows, trial_rows) if safeguards and objective > _objective(base) else None
        changes = sum(value != baselines["real-val"].cfg.reconcile[key] for key, value in proposed.items())
        key = (objective, -sum(_rate(item["catastrophic"]) for item in stats.values()), -changes)
        if lower_gain is not None and lower_gain > 0 and key > best_key:
            winner, best_key = proposed, key
        trials.append(dict(overrides=proposed, stats=stats, objective=float(objective), safeguards=safeguards,
                           bootstrap_gain_p05=lower_gain, changed_settings=changes))
    traces = {d: replay(baselines[d], records[d], winner, trace=True)[1] if winner else checked[d][1]
              for d in COUNTS}
    return dict(selected_overrides=winner, selected_objective=float(best_key[0]),
                baseline_objective=float(_objective(base)), baseline=reports, trials=trials, traces=traces,
                bootstrap_seed=BOOTSTRAP_SEED, bootstrap_draws=BOOTSTRAP_DRAWS)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("capture", "replay", "tune"))
    parser.add_argument("--baseline", required=True, type=Path, help="directory containing the two baseline bundles")
    parser.add_argument("--cache", required=True, type=Path, help="directory for dataset-named cache files")
    parser.add_argument("--dataset", choices=COUNTS)
    parser.add_argument("--limit", type=int, help="capture a deterministic prefix for a smoke check")
    parser.add_argument("--output", type=Path, help="new replay or tuning JSON report")
    args = parser.parse_args(argv)
    if args.command == "tune":
        if args.dataset or args.limit is not None or not args.output:
            parser.error("tune requires --output and both full caches")
    elif not args.dataset:
        parser.error("capture and replay require --dataset")
    elif args.command == "replay" and (not args.output or args.limit is not None):
        parser.error("replay requires --output and reads the cached cohort")
    elif args.command == "capture" and (args.output or args.limit is not None and args.limit < 1):
        parser.error("capture accepts a positive --limit and writes to --cache")
    try:
        if args.output and args.output.exists():
            raise ValueError("output already exists")
        datasets = COUNTS if args.command == "tune" else (args.dataset,)
        baselines = {name: load_baseline(args.baseline, name) for name in datasets}
        if args.command == "capture":
            path = args.cache / f"{args.dataset}.json"
            if path.exists():
                raise ValueError("cache already exists")
            cache = make_cache(baselines[args.dataset], args.limit)
            records = cache["payload"]["records"]
            verify_baseline(baselines[args.dataset], records)
            _write_new(path, cache)
            print(f"Captured and verified {len(records)} observations: {path}")
            return
        records = {d: load_cache(args.cache / f"{d}.json", baseline) for d, baseline in baselines.items()}
        if args.command == "replay":
            rows, traces = verify_baseline(baselines[args.dataset], records[args.dataset])
            report = dict(dataset=args.dataset, baseline_equivalent=True,
                          full_cohort=len(rows) == COUNTS[args.dataset], summary=_report(rows), traces=traces)
        else:
            if baselines["real-val"].provenance["config_hash"] != baselines["synthetic-dev"].provenance["config_hash"]:
                raise ValueError("tuning cohorts must use the same baseline configuration")
            report = tune(baselines, records)
        report["provenance"] = {d: baseline.provenance for d, baseline in baselines.items()}
        _write_new(args.output, report)
        print(f"Wrote {args.command} report: {args.output}")
    except (ValueError, AssertionError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f"Tuning infrastructure failed: {exc}\n")


if __name__ == "__main__":
    main()
