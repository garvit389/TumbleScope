"""Independent-seed diagnostic probe; does not select calibrated thresholds.

Run: python -m evaluation.cp3a_calibration_probe
This narrow experiment is not a complete CP3a calibration protocol.
"""
from collections import Counter
from itertools import product
import json
from math import sqrt
from pathlib import Path

import numpy as np

from core.config import load_config
from reconciliation.reconcile import reconcile
from tests.p1.test_reconcile_recovery import run_estimators


def interval(k, n):
    z = 1.959963984540054
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [center - half, center + half]


def run(split, seeds, policies):
    config = load_config()
    rows = []
    for scenario, seed in product(("sine", "double_hump", "noise"), seeds):
        rng = np.random.default_rng(seed)
        times = np.sort(np.r_[0., rng.uniform(0., 192., 126), 192.])
        phase = 2 * np.pi * times / 32.
        values = rng.normal(0, .35, times.size)
        if scenario == "sine":
            values += np.sin(phase)
        elif scenario == "double_hump":
            values += np.cos(2 * phase) + .35 * np.sin(phase)
        prep, estimators, cfg = run_estimators(times, values, config)
        for name, policy in policies.items():
            result = reconcile(prep, estimators, {**cfg, **policy})
            expected = [32.] if scenario == "sine" else [16., 32.]
            matching = [c.cluster_id for c in result.clusters if scenario != "noise"
                        and any(abs(c.period_s / p - 1) <= .02 for p in expected)]
            reasons = Counter(reason for c in result.clusters for reason in c.stage2["fail_reasons"])
            rows.append({"scenario": scenario, "seed": seed, "policy": name,
                         "candidate_match": bool(matching),
                         "retained": any(i in result.shortlist for i in matching),
                         "any_shortlist": bool(result.shortlist), "rejections": dict(reasons)})
    summary = {}
    for name, scenario in product(policies, ("sine", "double_hump", "noise")):
        group = [r for r in rows if r["policy"] == name and r["scenario"] == scenario]
        metric = "any_shortlist" if scenario == "noise" else "retained"
        count = sum(r[metric] for r in group)
        summary[f"{name}/{scenario}"] = {"n": len(group), "count": count,
                                           "rate": count / len(group),
                                           "wilson_95": interval(count, len(group))}
    return {"split": split, "seeds": list(seeds), "policies": policies,
            "configuration": config.reconcile, "rows": rows, "summary": summary,
            "limitations": ["Single cadence regime, baseline, period and noise level.",
                            "No nearby true candidate, missingness, aliases or interactions assessed.",
                            "No CNN calibration, no reference labels, and no configuration selection."]}


def main():
    policies = {"existing": {}, "cycles_3": {"min_cycles": 3.},
                "width_025": {"max_rel_period_uncertainty": .25},
                "bins_20": {"n_phase_bins": 20}, "occupancy_3": {"min_points_per_bin": 3},
                "coverage_07": {"min_phase_coverage": .7},
                "separation_1": {"min_separation_resolutions": 1.},
                "cluster_05": {"cluster_tol_resolutions": .5}}
    output = Path("docs/results")
    for split, seeds, choices in (("calibration", range(31000, 31030), policies),
                                  ("validation", range(41000, 41030), {"existing": {}})):
        path = output / f"p1_cp3a_probe_{split}.json"
        if path.exists():
            raise FileExistsError(path)
        report = run(split, seeds, choices)
        path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print(path, json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
