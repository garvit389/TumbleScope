"""Reproducible synthetic CP3a study, with a locked, separate validation step.

Run ``python -m evaluation.cp3a_calibration_study calibrate`` then
``python -m evaluation.cp3a_calibration_study validate``. Neither step overwrites.
"""
from collections import Counter
from hashlib import sha256
from importlib.metadata import version
from itertools import product
import argparse
import json
from math import sqrt
from pathlib import Path

import numpy as np
from classical_analysis.lomb_scargle import run_lomb_scargle
from classical_analysis.pdm import run_pdm
from classical_analysis.window import compute_window
from core.config import load_config
from core.schemas import Observation
from preprocessing.prepare import prepare
from reconciliation.reconcile import reconcile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results"
SCENARIOS = {
    # cycles, sampling, waveform, amplitude, noise, missing fraction
    "short": (1.7, "irregular", "sine", 1., .35, 0.),
    "borderline": (2.7, "irregular", "sine", 1., .35, 0.),
    "long": (8., "irregular", "sine", 1., .35, 0.),
    "weak": (6., "irregular", "sine", .45, 1., 0.),
    "gapped": (6., "gap", "sine", 1., .35, 0.),
    "missing": (6., "irregular", "sine", 1., .35, .4),
    "sparse": (6., "sparse", "sine", 1., .35, 0.),
    "cadence_alias": (6., "regular", "sine", 1., .35, 0.),
    "nonsinusoidal": (6., "irregular", "saw", 1., .35, 0.),
    "double_hump": (6., "irregular", "double", 1., .35, 0.),
    "nearby": (12., "irregular", "nearby", 1., .25, 0.),
    "noise": (6., "irregular", "noise", 0., 1., 0.),
}
POLICIES = {
    "baseline": {}, "cycles_3": {"min_cycles": 3.},
    "width_025": {"max_rel_period_uncertainty": .25},
    "bins_20": {"n_phase_bins": 20}, "occupancy_3": {"min_points_per_bin": 3},
    "coverage_07": {"min_phase_coverage": .7},
    "separation_1": {"min_separation_resolutions": 1.},
    "separation_2": {"min_separation_resolutions": 2.},
    "cluster_05": {"cluster_tol_resolutions": .5},
    "cluster_15": {"cluster_tol_resolutions": 1.5},
    "phase_combined": {"n_phase_bins": 20, "min_points_per_bin": 3, "min_phase_coverage": .7},
    "resolution_combined": {"cluster_tol_resolutions": .5, "min_separation_resolutions": 1.},
    "strict_combined": {"min_cycles": 3., "max_rel_period_uncertainty": .25,
                        "min_phase_coverage": .7},
}
MATCH = .02


def wilson(k, n):
    if not n:
        return None
    z = 1.959963984540054
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    h = z*sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return [max(0., c-h), min(1., c+h)]


def sample(name, seed, config):
    cycles, sampling, shape, amplitude, noise, missing = SCENARIOS[name]
    rng = np.random.default_rng(seed)
    period = 32.
    baseline = cycles * period
    n = 40 if sampling == "sparse" else 128
    if sampling == "regular":
        times = np.linspace(0., baseline, n) + rng.normal(0, .02, n)
        times.sort()
    else:
        times = np.sort(np.r_[0., rng.uniform(0., baseline, n-2), baseline])
    if sampling == "gap":
        times = times[(times < .3*baseline) | (times > .7*baseline)]
    phase = 2*np.pi*times/period
    values = rng.normal(0., noise, times.size)
    truths = []
    if shape == "sine":
        values += amplitude*np.sin(phase); truths = [period]
    elif shape == "saw":
        values += amplitude*(2*((times/period) % 1)-1); truths = [period]
    elif shape == "double":
        values += amplitude*(np.cos(2*phase)+.35*np.sin(phase)); truths = [period/2, period]
    elif shape == "nearby":
        # 1.8 baseline-resolution elements apart: distinct but hard to separate.
        other = 1/(1/period + 1.8/baseline)
        values += amplitude*(np.sin(phase)+.8*np.sin(2*np.pi*times/other+.3)); truths = [period, other]
    if missing:
        mask = rng.random(times.size) < missing
        mask[[0, -1]] = False
        values[mask] = np.nan
    prep = prepare(Observation(f"{name}-{seed}", times, values, None, "unknown"), config.preprocess)
    bounds = {"period_min_s": config.period_bounds["min_period_s"],
              "period_max_s": config.period_bounds["max_period_s"]}
    ccfg = {**config.classical, **bounds}
    estimators = [run_lomb_scargle(prep, ccfg), run_pdm(prep, ccfg)]
    if any(e.status != "ok" for e in estimators):
        raise RuntimeError(f"estimator failure {name}/{seed}: {[(e.status, e.error) for e in estimators]}")
    window = compute_window(prep, ccfg)
    return prep, estimators, {**config.reconcile, **bounds, "window_result": window}, truths


def evaluate(name, seed, prep, estimators, cfg, truths, policy):
    result = reconcile(prep, estimators, {**cfg, **policy})
    result.validate()
    def near(p, q):
        return abs(p/q - 1) <= MATCH
    matches = [[c for c in result.clusters if near(c.period_s, p)] for p in truths]
    retained = [any(c.cluster_id in result.shortlist for c in group) for group in matches]
    false_shortlist = [c for c in result.clusters if c.cluster_id in result.shortlist
                       and not any(near(c.period_s, p) for p in truths)]
    # Candidate presence is conditioned on the estimators actually returning a match.
    present = [bool(group) for group in matches]
    duplicates = sum(max(0, len(group)-1) for group in matches)
    members = [{method for c in group for method in c.members} for group in matches]
    paired_duplicates = sum(len(m) >= 2 for m in members)
    merged_duplicates = sum(any(len(c.members) >= 2 for c in group) for group in matches)
    overlap = len(truths) == 2 and bool({c.cluster_id for c in matches[0]} & {c.cluster_id for c in matches[1]})
    harmonic_link = bool(name == "double_hump" and matches[0] and matches[1] and any(
        link["other_id"] == b.cluster_id and link["ratio"] == 2.
        for a in matches[0] for b in matches[1] for link in a.harmonic_links))
    reasons = Counter(reason for c in result.clusters for reason in c.stage2["fail_reasons"])
    return {"scenario": name, "seed": seed, "truth_periods_s": truths,
            "truth_present": present, "truth_retained": retained,
            "n_clusters": len(result.clusters), "false_shortlist": len(false_shortlist),
            "shortlist_size": len(result.shortlist), "duplicate_extra_clusters": duplicates,
            "truth_with_two_methods": paired_duplicates, "truth_with_merged_methods": merged_duplicates,
            "distinct_truth_same_cluster": overlap, "harmonic_link": harmonic_link,
            "harmonic_both_retained": bool(name == "double_hump" and len(retained) == 2 and all(retained)),
            "rejections": dict(reasons)}


def summarize(rows):
    out = {}
    for scenario in SCENARIOS:
        group = [r for r in rows if r["scenario"] == scenario]
        n = len(group)
        any_retained = sum(any(r["truth_retained"]) for r in group)
        all_retained = sum(bool(r["truth_retained"]) and all(r["truth_retained"]) for r in group)
        false_any = sum(r["false_shortlist"] > 0 for r in group)
        both = sum(r["harmonic_both_retained"] for r in group)
        links = sum(r["harmonic_link"] for r in group)
        out[scenario] = {"n": n, "any_true_retained": any_retained,
                         "any_true_retained_ci95": wilson(any_retained,n),
                         "all_truth_retained": all_retained, "all_truth_retained_ci95": wilson(all_retained,n),
                         "false_shortlist_runs": false_any, "false_shortlist_ci95": wilson(false_any,n),
                         "harmonic_links": links, "harmonic_both_retained": both,
                         "harmonic_both_retained_ci95": wilson(both,n) if scenario == "double_hump" else None,
                         "distinct_truth_same_cluster": sum(r["distinct_truth_same_cluster"] for r in group),
                         "duplicate_extra_clusters": sum(r["duplicate_extra_clusters"] for r in group),
                         "truth_with_two_methods": sum(r["truth_with_two_methods"] for r in group),
                         "truth_with_merged_methods": sum(r["truth_with_merged_methods"] for r in group),
                         "rejections": dict(sum((Counter(r["rejections"]) for r in group), Counter()))}
    return out


def provenance():
    files = [Path(__file__), ROOT / "configs/reconcile.yaml", ROOT / "configs/classical.yaml",
             ROOT / "configs/period_bounds.yaml", *sorted((ROOT / "reconciliation").glob("*.py")),
             ROOT / "classical_analysis/lomb_scargle.py", ROOT / "classical_analysis/pdm.py",
             ROOT / "classical_analysis/window.py", ROOT / "preprocessing/prepare.py"]
    return {"sha256": {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in files},
            "versions": {p: version(p) for p in ("numpy", "scipy", "astropy")}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["calibrate", "validate"])
    args = parser.parse_args()
    output = OUT / f"p1_cp3a_study_{args.phase}.json"
    if output.exists():
        raise FileExistsError(output)
    freeze = OUT / "p1_cp3a_study_frozen.json"
    if args.phase == "calibrate" and freeze.exists():
        raise FileExistsError(freeze)
    config = load_config()
    if args.phase == "calibrate":
        choices = POLICIES
        seeds = range(51000, 51030)
    else:
        locked = json.loads(freeze.read_text(encoding="utf-8"))
        if locked["provenance"] != provenance() or locked["base_config"] != config.reconcile:
            raise RuntimeError("frozen inputs differ; do not validate modified inputs")
        choices = {locked["selection"]: locked["override"]}
        seeds = range(61000, 61030)
    rows = {policy: [] for policy in choices}
    for name, seed in product(SCENARIOS, seeds):
        prep, estimators, cfg, truths = sample(name, seed, config)
        for policy, override in choices.items():
            rows[policy].append(evaluate(name, seed, prep, estimators, cfg, truths, override))
        if seed == seeds.stop-1:
            print(name, flush=True)
    summaries = {name: summarize(records) for name, records in rows.items()}
    scores = {}
    for name, summary in summaries.items():
        # Equal scenario weighting and equal cost for missed signals and noise survival.
        sensitivity = sum(summary[s]["any_true_retained"]/summary[s]["n"] for s in SCENARIOS if s != "noise") / (len(SCENARIOS)-1)
        specificity = 1-summary["noise"]["false_shortlist_runs"]/summary["noise"]["n"]
        scores[name] = {"sensitivity": sensitivity, "specificity": specificity,
                        "balanced_accuracy": (sensitivity+specificity)/2}
    report = {"phase": args.phase, "seeds": list(seeds), "scenarios": SCENARIOS,
              "policies": choices, "base_config": config.reconcile, "match_relative": MATCH,
              "provenance": provenance(), "selection_metric": "equal-scenario sensitivity and noise specificity, equal weighted",
              "summaries": summaries, "scores": scores, "rows": rows}
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    if args.phase == "calibrate":
        # Fixed order favors baseline on exact ties; validation cannot alter this choice.
        winner = max(choices, key=lambda x: scores[x]["balanced_accuracy"])
        locked = {"selection": winner, "override": choices[winner], "base_config": config.reconcile,
                  "provenance": provenance(), "calibration_sha256": sha256(output.read_bytes()).hexdigest()}
        freeze.write_text(json.dumps(locked, indent=2), encoding="utf-8")
        print("Frozen:", winner, scores[winner], flush=True)
    else:
        print("Validation:", scores, flush=True)


if __name__ == "__main__":
    main()
