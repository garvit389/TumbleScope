"""Evaluate one Stage 2 separation threshold at a time on existing CP4 caches."""

import argparse
from pathlib import Path

from evaluation import run_eval
from evaluation import tune_reconciliation as weights


THRESHOLD = "min_separation_resolutions"
GRID = (1.0, 1.25, 1.75, 2.0)


def replay_threshold(baseline, records, fixed_weights, value=None, *, trace=False):
    if value is not None:
        if value not in GRID:
            raise ValueError("threshold is outside the prespecified grid")
        updated = []
        for record in records:
            snapshot = record["snapshot"]
            if snapshot is not None and snapshot["settings"] is not None:
                snapshot = {**snapshot, "settings": {**snapshot["settings"], THRESHOLD: value}}
            updated.append({**record, "snapshot": snapshot})
        records = updated
    return weights.replay(baseline, records, fixed_weights, trace=trace)


def _comparison(rows, traces):
    return {d: {"metrics": weights._stats(rows[d], "cnn_classical"),
                "trace": traces[d]} for d in weights.COUNTS}


def search(baselines, records, saved):
    cohorts = weights.COUNTS
    if set(baselines) != set(cohorts) or set(records) != set(cohorts) or any(
        len(records[d]) != n for d, n in cohorts.items()
    ):
        raise ValueError("Stage 2 search requires both complete validation caches")
    if len({b.provenance["config_hash"] for b in baselines.values()}) != 1:
        raise ValueError("cohorts must share the baseline configuration")
    if saved["provenance"] != {d: baselines[d].provenance for d in cohorts}:
        raise ValueError("weight report provenance differs from the baseline")
    if (saved["bootstrap_seed"], saved["bootstrap_draws"]) != (
        weights.BOOTSTRAP_SEED, weights.BOOTSTRAP_DRAWS
    ):
        raise ValueError("weight report bootstrap policy differs")

    originals = {d: weights.verify_baseline(baselines[d], records[d])[0] for d in cohorts}
    original_reports = {d: weights._report(rows) for d, rows in originals.items()}
    original_stats = {d: report["fused"] for d, report in original_reports.items()}
    if saved["baseline"] != original_reports or saved["baseline_objective"] != float(
        weights._objective(original_stats)
    ):
        raise ValueError("weight report original baseline is not reproducible")

    fixed = saved["selected_overrides"]
    selected = [trial for trial in saved["trials"] if trial["overrides"] == fixed]
    proposed = [dict(zip(weights.WEIGHT_KEYS, (*w, penalty), strict=True))
                for w in weights.WEIGHTS for penalty in weights.PENALTIES]
    if len(selected) != 1 or fixed not in proposed:
        raise ValueError("expected a selected trial from the prespecified weight grid")
    selected = selected[0]
    loaded = {d: replay_threshold(baselines[d], records[d], fixed, trace=True) for d in cohorts}
    base_rows = {d: pair[0] for d, pair in loaded.items()}
    base_traces = {d: pair[1] for d, pair in loaded.items()}
    base = {d: weights._stats(rows, "cnn_classical") for d, rows in base_rows.items()}
    objective = weights._objective(base)
    if (base != selected["stats"] or base_traces != saved["traces"]
            or float(objective) != selected["objective"] or float(objective) != saved["selected_objective"]
            or not selected["safeguards"] or not weights._safeguards(base, original_stats)
            or not selected["bootstrap_gain_p05"] or selected["bootstrap_gain_p05"] <= 0
            or weights._stability(originals, base_rows) != selected["bootstrap_gain_p05"]):
        raise ValueError("selected weight replay or safeguards differ from the saved report")

    stage2 = {key: baselines["real-val"].cfg.reconcile[key] for key in (
        "min_cycles", "max_rel_period_uncertainty", "n_phase_bins", "min_points_per_bin",
        "min_phase_coverage", THRESHOLD,
    )}
    start = stage2[THRESHOLD]
    if start != 1.5:
        raise ValueError("prespecified separation grid requires baseline 1.5")
    best = start
    best_key = (objective, -sum(weights._rate(x["catastrophic"]) for x in base.values()), 0)
    trials = []
    for value in GRID:
        replayed = {d: replay_threshold(baselines[d], records[d], fixed, value, trace=True) for d in cohorts}
        trial = {d: pair[0] for d, pair in replayed.items()}
        trace_changes = {}
        for d, (_, traces) in replayed.items():
            pairs = list(zip(base_traces[d], traces, strict=True))
            trace_changes[d] = dict(
                changed_shortlists=sum(a["shortlist_ids"] != b["shortlist_ids"] for a, b in pairs),
                cluster_pass_flips=sum(
                    c["stage2"]["passed"] != q["stage2"]["passed"] for a, b in pairs
                    for c, q in zip(a["clusters"], b["clusters"], strict=True)
                ),
            )
        stats = {d: weights._stats(rows, "cnn_classical") for d, rows in trial.items()}
        score = weights._objective(stats)
        safeguards = weights._safeguards(stats, base)
        gain = weights._stability(base_rows, trial)
        changed = 1
        key = (score, -sum(weights._rate(x["catastrophic"]) for x in stats.values()), -changed)
        if safeguards and score > objective and gain > 0 and key > best_key:
            best, best_key = value, key
        trials.append(dict(threshold=value, metrics=stats, objective=float(score),
                           objective_gain=float(score - objective), safeguards=safeguards,
                           trace_changes=trace_changes,
                           bootstrap_gain_p05=gain))
    final = ({d: replay_threshold(baselines[d], records[d], fixed, best, trace=True) for d in cohorts}
             if best != start else loaded)
    return dict(
        baseline_equivalent=True, selected_weight_replay_equivalent=True,
        fixed_weights=fixed, threshold=THRESHOLD, grid=GRID,
        original_stage2=stage2, recommended_stage2={**stage2, THRESHOLD: best},
        baseline_objective=float(objective), recommended_objective=float(best_key[0]),
        recommended_changed=best != start, trials=trials,
        selected_weight_baseline=_comparison(base_rows, base_traces),
        recommended=_comparison({d: pair[0] for d, pair in final.items()},
                                {d: pair[1] for d, pair in final.items()}),
        baseline_operations={d: report["baseline_operations"] for d, report in original_reports.items()},
        pdm_fixed={d: report["pdm_fixed"] for d, report in original_reports.items()},
        bootstrap_seed=weights.BOOTSTRAP_SEED, bootstrap_draws=weights.BOOTSTRAP_DRAWS,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError("output already exists")
        report = args.cache / "weight-search.json"
        baselines = {d: weights.load_baseline(args.baseline, d) for d in weights.COUNTS}
        paths = {d: args.cache / f"{d}.json" for d in baselines}
        records = {d: weights.load_cache(paths[d], baseline) for d, baseline in baselines.items()}
        result = search(baselines, records, weights._read_json(report))
        result["provenance"] = {
            "weight_search_sha256": run_eval._digest(report),
            "cache_sha256": {d: run_eval._digest(path) for d, path in paths.items()},
            "baseline": {d: b.provenance for d, b in baselines.items()},
            "stage2_tuner_sha256": run_eval._digest(Path(__file__)),
        }
        weights._write_new(args.output, result)
        print(f"Wrote Stage 2 selection evidence: {args.output}")
    except (OSError, ValueError, AssertionError, TypeError, KeyError) as exc:
        parser.exit(2, f"Stage 2 tuning failed: {exc}\n")


if __name__ == "__main__":
    main()
