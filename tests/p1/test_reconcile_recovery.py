"""Controlled CP3a acceptance and a reproducible threshold sensitivity pilot.

The pilot measures candidate retention, not physical truth or false-Supported
rates. It does not select thresholds without a defined retention/error target.
"""

from collections import Counter
from dataclasses import replace
from hashlib import sha256
from itertools import product
from importlib.metadata import version
import json
import os
from pathlib import Path

import numpy as np
import pytest

from classical_analysis.lomb_scargle import run_lomb_scargle
from classical_analysis.pdm import run_pdm
from classical_analysis.window import compute_window
from core.config import load_config
from core.schemas import Observation, ReconciliationResult, assert_contract_equal
from preprocessing.prepare import prepare
from reconciliation.reconcile import reconcile


def run_estimators(times, values, config):
    prep = prepare(Observation("cp3a-injection", times, values, None, "unknown"), config.preprocess)
    bounds = {"period_min_s": config.period_bounds["min_period_s"],
              "period_max_s": config.period_bounds["max_period_s"]}
    classical_cfg = {**config.classical, **bounds}
    estimators = [run_lomb_scargle(prep, classical_cfg), run_pdm(prep, classical_cfg)]
    assert all(e.status == "ok" for e in estimators), [e.error for e in estimators]
    window = compute_window(prep, classical_cfg)
    return prep, estimators, {**config.reconcile, **bounds, "window_result": window}


@pytest.mark.parametrize("shape", ["sine", "double_hump"])
def test_real_estimators_retain_clean_injected_periods(shape, mock_cnn):
    config = load_config()
    rng = np.random.default_rng(2303)
    times = np.sort(np.r_[0., rng.uniform(0, 640, 398), 640.])
    phase = 2 * np.pi * times / 32.
    values = np.sin(phase) if shape == "sine" else np.cos(2 * phase) + .35 * np.sin(phase)
    prep, estimators, cfg = run_estimators(times, values, config)
    if shape == "double_hump":
        # Use the frozen mock's relative hypotheses at this injected time scale.
        estimators.append(replace(mock_cnn, candidates=[
            replace(c, period_s=2 * c.period_s) for c in mock_cnn.candidates]))
    result = reconcile(prep, estimators, cfg)
    expected = [32.] if shape == "sine" else [16., 32.]
    retained = []
    for period in expected:
        cluster = min(result.clusters, key=lambda c: abs(c.period_s / period - 1))
        assert cluster.period_s == pytest.approx(period, rel=.02)
        assert cluster.cluster_id in result.shortlist
        retained.append(cluster)
    if shape == "double_hump":
        assert {"other_id": retained[1].cluster_id, "ratio": 2.} in retained[0].harmonic_links
    assert_contract_equal(result, ReconciliationResult.from_dict(
        json.loads(json.dumps(result.to_dict(), allow_nan=False))))


def test_stage2_threshold_sensitivity_pilot(tmp_path):
    config = load_config()
    # Independent synthetic inputs; no reference labels, corpus, or holdout files.
    periods = (8., 32., 128.)
    cycles_grid = (1.5, 2., 3., 6.)
    samplings = ("irregular", "gapped", "phase_locked")
    waveforms = ("sine", "double_hump", "white_noise")
    seeds = (23001, 23002)
    policies = {"configured": {}}
    alternatives = {
        "min_cycles": (3.,), "max_rel_period_uncertainty": (.25,),
        "n_phase_bins": (20,), "min_points_per_bin": (3,),
        "min_phase_coverage": (.7,), "min_separation_resolutions": (1., 2.),
        "cluster_tol_resolutions": (.5, 1.5),
    }
    for key, values in alternatives.items():
        for value in values:
            policies[f"{key}={value}"] = {key: value}
    counts = {name: Counter() for name in policies}
    rows = []
    for period, cycles, sampling, shape, seed in product(
            periods, cycles_grid, samplings, waveforms, seeds):
        rng = np.random.default_rng(seed)
        baseline = cycles * period
        if sampling == "phase_locked":
            times = np.sort(np.r_[0., np.arange(1, int(cycles) + 1) * period,
                                  np.arange(0, int(cycles)) * period + .03 * period, baseline])
            # Enough distinct points to run the estimators, confined to narrow phase arcs.
            times = np.unique(np.concatenate([times + step * period for step in (0., .001, .002, .003)]))
        else:
            times = np.sort(np.r_[0., rng.uniform(0, baseline, 126), baseline])
            if sampling == "gapped":
                times = times[(times < .3 * baseline) | (times > .7 * baseline)]
        phase = 2 * np.pi * times / period
        values = rng.normal(0, .35, times.size)
        if shape == "sine":
            values += np.sin(phase)
        elif shape == "double_hump":
            values += np.cos(2 * phase) + .35 * np.sin(phase)
        prep, estimators, cfg = run_estimators(times, values, config)
        expected = (period,) if shape == "sine" else (period / 2, period)
        for name, policy in policies.items():
            result = reconcile(prep, estimators, {**cfg, **policy})
            result.validate()
            json.dumps(result.to_dict(), allow_nan=False)
            count = counts[name]
            count["runs"] += 1
            count["runs_with_shortlist"] += bool(result.shortlist)
            if shape == "white_noise":
                count["noise_runs"] += 1
                count["noise_with_shortlist"] += bool(result.shortlist)
            else:
                matches = [c for c in result.clusters
                           if any(abs(c.period_s / p - 1) <= .02 for p in expected)]
                count["signal_runs"] += 1
                count["signal_with_matching_candidate"] += bool(matches)
                count["signal_with_matching_shortlist"] += any(c.cluster_id in result.shortlist for c in matches)
            if name == "configured":
                rows.append({"period_s": period, "cycles": cycles, "sampling": sampling,
                             "waveform": shape, "seed": seed, "reconciliation": result.to_dict()})
    root = Path(__file__).resolve().parents[2]
    sources = [*sorted((root / "reconciliation").glob("*.py")), Path(__file__),
               root / "configs/reconcile.yaml", root / "configs/classical.yaml"]
    report = {
        "purpose": "CP3a diagnostic sensitivity pilot; thresholds not selected or calibrated",
        "recipe": {"periods_s": periods, "cycles": cycles_grid, "sampling": samplings,
                   "waveforms": waveforms, "seeds": seeds, "noise_sd": .35,
                   "match_relative_error": .02, "harmonic_matches": [1., .5]},
        "config": {"reconcile": config.reconcile, "classical": config.classical,
                   "preprocess": config.preprocess, "period_bounds": config.period_bounds},
        "source_sha256": {str(p.relative_to(root)): sha256(p.read_bytes()).hexdigest() for p in sources},
        "versions": {name: version(name) for name in ("numpy", "scipy", "astropy", "pytest")},
        "policies": policies, "counts": counts, "configured_runs": rows,
        "limitations": ["Two seeds per condition; not a population accuracy estimate.",
                        "No CNN recovery calibration; classical estimators only.",
                        "Half-period matches for double humps are competing hypotheses.",
                        "No predeclared retention/error target determines a preferred policy.",
                        "Real cadence and correlated-noise validation remain necessary."],
    }
    # Optional persistent test artifact; the default remains pytest's temporary directory.
    output = Path(os.environ.get("TUMBLESCOPE_CP3A_EVIDENCE_DIR", str(tmp_path)))
    output.mkdir(parents=True, exist_ok=True)
    target = output / "cp3a_threshold_sensitivity.json"
    target.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"\nCP3a sensitivity evidence: {target}\n{json.dumps(counts, sort_keys=True)}")
    assert all(c["runs"] == 216 and c["noise_runs"] == 72 and c["signal_runs"] == 144
               for c in counts.values())
    assert counts["configured"]["signal_with_matching_shortlist"] > 0
