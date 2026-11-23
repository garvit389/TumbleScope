"""Controlled candidate-pair stress tests, separate from calibration selection.

The synthetic candidates isolate reconciliation; they are not estimator recovery.
Run with ``python -m evaluation.cp3a_controlled_pairs``.
"""
from collections import Counter
from itertools import product
import json
from pathlib import Path

import numpy as np
from classical_analysis.window import compute_window
from core.config import load_config
from core.schemas import CNN, LOMB_SCARGLE, Candidate, EstimatorResult, Observation
from preprocessing.prepare import prepare
from reconciliation.reconcile import reconcile
from evaluation.cp3a_calibration_study import wilson

ROOT = Path(__file__).resolve().parents[1]


def estimator(method, period, support, width):
    candidate = Candidate(period, support, support, {"freq_width_hz": width})
    if method == CNN:
        meta = {"runtime_s": 0., "p_no_reliable": 0., "abstained": False,
                "n_windows": 1, "window_disagreement": 0., "experimental": True}
    else:
        meta = {"runtime_s": 0., "freq_resolution": width,
                "periodogram": {"freq_hz": [], "power": []}}
    return EstimatorResult(method, "ok", [candidate], meta=meta)


def case(seed, kind, distance, config):
    rng = np.random.default_rng(seed)
    # Repeated observing blocks create measurable window sidebands in-band.
    times = np.sort(np.concatenate([64*k + rng.uniform(0., 12., 20) for k in range(12)]))
    brightness = np.sin(2*np.pi*times/10.) + rng.normal(0., .25, times.size)
    prep = prepare(Observation("controlled", times, brightness, None, "unknown"), config.preprocess)
    resolution = prep.meta["freq_resolution"]
    first = 10.
    if kind == "harmonic":
        second = 20.
    elif kind == "alias":
        second = None
    else:
        second = 1/(1/first + distance*resolution)
    bounds = {"period_min_s": config.period_bounds["min_period_s"],
              "period_max_s": config.period_bounds["max_period_s"]}
    window = compute_window(prep, {**config.classical, **bounds})
    usable = [p for p in window["peaks"] if 0 < p["freq_hz"] < .1]
    if kind == "alias":
        if not usable:
            return {"seed": seed, "kind": kind, "usable_offset": False}
        second = 1/(1/first + usable[0]["freq_hz"])
    results = [estimator(CNN, first, 1., resolution), estimator(LOMB_SCARGLE, second, .8, resolution)]
    cfg = {**config.reconcile, **bounds, "window_result": window}
    result = reconcile(prep, results, cfg)
    first_cluster = next(c for c in result.clusters if CNN in c.members)
    second_cluster = next(c for c in result.clusters if LOMB_SCARGLE in c.members)
    same = first_cluster.cluster_id == second_cluster.cluster_id
    return {"seed": seed, "kind": kind, "distance_resolutions": distance,
            "usable_offset": bool(usable), "window_peak": usable[0] if usable else None,
            "same_cluster": same, "both_pass": first_cluster.stage2["passed"] and second_cluster.stage2["passed"],
            "second_pass": second_cluster.stage2["passed"],
            "second_alias_risk": second_cluster.alias_risk,
            "harmonic_link": any(link["other_id"] == second_cluster.cluster_id and link["ratio"] == 2.
                                 for link in first_cluster.harmonic_links),
            "second_reasons": second_cluster.stage2["fail_reasons"]}


def main():
    config = load_config()
    for split, seeds in (("calibration", range(71000, 71030)), ("validation", range(81000, 81030))):
        path = ROOT / "docs" / "results" / f"p1_cp3a_controlled_{split}.json"
        if path.exists():
            raise FileExistsError(path)
        rows = [case(seed, kind, dist, config) for kind, dist, seed in product(
            ("duplicate", "distinct"), (.5, 1., 1.5, 2.), seeds)]
        rows += [case(seed, kind, None, config) for kind, seed in product(("alias", "harmonic"), seeds)]
        summary = {}
        for kind, dist in [(k, d) for k in ("duplicate", "distinct") for d in (.5, 1., 1.5, 2.)] + [("alias", None), ("harmonic", None)]:
            group = [r for r in rows if r["kind"] == kind and r.get("distance_resolutions") == dist and "same_cluster" in r]
            n = len(group)
            metrics = {}
            for field in ("same_cluster", "both_pass", "second_pass", "harmonic_link"):
                count = sum(r[field] for r in group)
                metrics[field] = {"count": count, "n": n, "wilson_95": wilson(count, n)}
            metrics["alias_risk_positive"] = {"count": sum(r["second_alias_risk"] > 0 for r in group), "n": n}
            metrics["rejections"] = dict(sum((Counter(r["second_reasons"]) for r in group), Counter()))
            summary[f"{kind}:{dist}"] = metrics
        report = {"split": split, "seeds": list(seeds), "config": config.reconcile, "rows": rows,
                  "summary": summary, "limitations": ["Controlled candidate periods and supports; not estimator performance.",
                  "Duplicate and distinct labels use identical input geometry, exposing an identifiability trade-off.",
                  "Config unchanged; this post-selection stress test is not a new validation selection."]}
        path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print(split, json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
