import json
from dataclasses import replace

import numpy as np
import pytest

from classical_analysis.lomb_scargle import run_lomb_scargle
from core.config import load_config
from core.schemas import (
    CNN, LOMB_SCARGLE, NOT_RUN, PDM, Candidate, EstimatorResult, Observation,
    ReconciliationResult, STAGE2_FAIL_REASONS, assert_contract_equal,
)
from preprocessing.prepare import prepare
from reconciliation import aliases
from reconciliation.reconcile import reconcile


INSUFFICIENT_CYCLES, UNRESOLVED_FREQUENCY, POOR_PHASE_COVERAGE, MERGED_INTO_NEIGHBOUR = (
    STAGE2_FAIL_REASONS
)


@pytest.fixture
def cfg():
    config = load_config()
    return dict(config.reconcile, period_min_s=2.0, period_max_s=100.0)


def synthetic(times, values=None):
    if values is None:
        values = np.sin(2 * np.pi * times / 16)
    return prepare(Observation("stage2-test", times, values, None, "unknown"), {})


def candidate(period, support, width=None):
    meta = {} if width is None else {"freq_width_hz": width}
    return Candidate(period, support, support * 0.5, meta)


def classical(method, *candidates):
    return EstimatorResult(method, "ok", list(candidates), meta={
        "freq_resolution": 0.01, "runtime_s": 0.0,
        "periodogram": {"freq_hz": [0.1], "power": [0.5]},
    })


def test_double_humped_signal_preserves_harmonics_and_round_trips(cfg, mock_cnn):
    rng = np.random.default_rng(42)
    times = np.cumsum(rng.uniform(0.4, 1.1, 350))
    phase = 2 * np.pi * times / 16
    prep = synthetic(times, np.cos(2 * phase) + 0.27 * np.sin(phase))
    ls = run_lomb_scargle(prep, {**load_config().classical, **cfg})
    assert ls.status == "ok" and ls.candidates[0].period_s == pytest.approx(8, rel=0.01)

    result = reconcile(prep, [mock_cnn, ls], cfg)
    short = next(cluster for cluster in result.clusters if cluster.members.get(CNN) == 8.0)
    long = next(cluster for cluster in result.clusters if cluster.members.get(CNN) == 16.0)
    assert LOMB_SCARGLE in short.members
    assert {short.cluster_id, long.cluster_id} <= set(result.shortlist)
    assert {"other_id": long.cluster_id, "ratio": 2.0} in short.harmonic_links
    assert {"other_id": short.cluster_id, "ratio": 0.5} in long.harmonic_links
    assert result.stage2_passed_any and not result.is_stub
    assert_contract_equal(result, ReconciliationResult.from_dict(
        json.loads(json.dumps(result.to_dict(), allow_nan=False)),
    ))


@pytest.mark.parametrize(("period", "reciprocal"), [(12.1, 2 / 3), (16.1, 0.5), (24.1, 1 / 3)])
def test_other_harmonics_use_nominal_reciprocal_ratios(cfg, mock_cnn, period, reciprocal):
    prep = synthetic(np.linspace(0, 200, 351))
    result = reconcile(prep, [replace(mock_cnn, candidates=[
        candidate(8.0, 0.9), candidate(period, 0.8),
    ])], cfg)
    short, long = result.clusters
    assert short.harmonic_links == [{"other_id": long.cluster_id, "ratio": 1 / reciprocal}]
    assert long.harmonic_links == [{"other_id": short.cluster_id, "ratio": reciprocal}]


def test_clustering_keeps_strongest_per_method_and_is_deterministic(cfg, mock_cnn):
    prep = synthetic(np.linspace(0, 200, 301))
    cnn = replace(mock_cnn, candidates=[candidate(8.0, 0.7), candidate(8.1, 0.95), candidate(16.0, 0.95)])
    ls = classical(LOMB_SCARGLE, candidate(8.2, 0.8, 0.006))
    result = reconcile(prep, [cnn, ls], cfg)
    cluster = next(item for item in result.clusters if LOMB_SCARGLE in item.members)
    assert cluster.members == {CNN: 8.1, LOMB_SCARGLE: 8.2}
    assert cluster.evidence == {CNN: 0.95, LOMB_SCARGLE: 0.8}
    assert cluster.period_s == 8.1
    assert cluster.stage2["rel_period_uncertainty"] == pytest.approx(8.1 * 0.006)
    reordered = reconcile(prep, [ls, replace(cnn, candidates=cnn.candidates[::-1])], cfg)
    assert_contract_equal(result, reordered)


@pytest.mark.parametrize("status", ["failed", "skipped", None])
def test_weights_renormalize_over_ok_estimators_including_empty(cfg, mock_cnn, status):
    prep = synthetic(np.linspace(0, 160, 349))
    cfg = dict(cfg, w_C=0.6, w_L=0.3, w_D=0.1, lambda_A=0.0)
    cnn = replace(mock_cnn, candidates=[candidate(8.0, 0.8)])
    estimators = [cnn, classical(LOMB_SCARGLE)]
    if status is not None:
        estimators.append(EstimatorResult(PDM, status, [], NOT_RUN if status == "skipped" else "failure",
                                          {"runtime_s": 0.0}))
    result = reconcile(prep, estimators, cfg)
    assert result.clusters[0].score == pytest.approx(0.8 * 0.6 / 0.9)
    assert result.clusters[0].evidence == {CNN: 0.8}
    assert any("Renormalized" in note and PDM in note and LOMB_SCARGLE not in note
               for note in result.notes)


def test_alias_offset_uses_window_height_and_source_support(cfg, mock_cnn, monkeypatch):
    prep = synthetic(np.linspace(0, 120, 377))
    offset = abs(1 / 8.0 - 1 / 11.0)
    monkeypatch.setattr(aliases, "compute_window", lambda prep, cfg: {
        "freq_hz": [0.0, offset, offset + prep.meta["freq_resolution"] / 2],
        "window_power": [1.0, 0.8, 0.4],
        "freq_resolution": prep.meta["freq_resolution"], "peaks": [
        {"freq_hz": offset, "height": 0.8},
        {"freq_hz": offset + prep.meta["freq_resolution"] / 2, "height": 0.4},
    ]})
    cnn = replace(mock_cnn, candidates=[candidate(8.0, 0.75), candidate(11.0, 0.3), candidate(16.0, 0.2)])
    result = reconcile(prep, [cnn], cfg)
    by_period = {cluster.period_s: cluster for cluster in result.clusters}
    assert by_period[8.0].alias_risk == by_period[16.0].alias_risk == 0.0
    assert by_period[11.0].alias_risk == pytest.approx(0.8 * 0.75)
    assert by_period[11.0].score == pytest.approx(0.3 - cfg["lambda_A"] * 0.8 * 0.75)


def test_stage2_records_all_applicable_failures(cfg):
    times = np.array([0., 2., 4., 6., 8., 10., 12.])
    prep = synthetic(times)
    ls = classical(LOMB_SCARGLE, candidate(20.0, 0.9, 0.05))
    result = reconcile(prep, [ls], cfg)
    gate = result.clusters[0].stage2
    assert gate["cycles"] == pytest.approx(0.6)
    assert gate["rel_period_uncertainty"] == pytest.approx(1.0)
    assert gate["phase_coverage"] == 0.0
    assert gate["fail_reasons"] == [INSUFFICIENT_CYCLES, UNRESOLVED_FREQUENCY, POOR_PHASE_COVERAGE]
    assert not gate["passed"] and not result.stage2_passed_any and result.shortlist == []


def test_stage2_rejects_close_lower_score_but_exempts_harmonics(cfg, mock_cnn):
    prep = synthetic(np.linspace(0, 80, 351))
    cfg = dict(cfg, cluster_tol_resolutions=0.1, min_separation_resolutions=6.0, lambda_A=0.0)
    cnn = replace(mock_cnn, candidates=[candidate(8.0, 0.9), candidate(16.0, 0.8), candidate(8.45, 0.7)])
    result = reconcile(prep, [cnn], cfg)
    leader, partner, neighbour = result.clusters
    assert result.shortlist == [leader.cluster_id, partner.cluster_id]
    assert abs(1 / 8.0 - 1 / 16.0) < cfg["min_separation_resolutions"] * prep.meta["freq_resolution"]
    assert neighbour.stage2["fail_reasons"] == [MERGED_INTO_NEIGHBOUR]
    assert neighbour.stage2["merged_with"] == leader.cluster_id
    assert all(cluster.stage2["passed"] == (not cluster.stage2["fail_reasons"])
               for cluster in result.clusters)


def test_shortlist_default_and_override_keep_all_clusters(cfg, mock_cnn):
    prep = synthetic(np.linspace(0, 250, 398))
    periods = [5.0, 6.0, 7.0, 8.0, 10.0, 12.0]
    cnn = replace(mock_cnn, candidates=[candidate(period, 0.5) for period in reversed(periods)])
    cfg = dict(cfg, lambda_A=0.0)
    cfg.pop("top_k")
    default = reconcile(prep, [cnn], cfg)
    fast = reconcile(prep, [cnn], dict(cfg, top_k=3))
    assert [cluster.period_s for cluster in default.clusters] == periods
    assert len(default.shortlist) == 5 and len(fast.shortlist) == 3
    assert fast.shortlist == default.shortlist[:3]
    assert_contract_equal(default.clusters, fast.clusters)


@pytest.mark.parametrize(("status", "error", "note"), [
    ("ok", None, "no_candidates"),
    ("skipped", NOT_RUN, "no_usable_estimators"),
    ("failed", "failure", "no_usable_estimators"),
])
def test_empty_outcomes_preserve_contract(prepared, mock_cnn, status, error, note):
    estimator = replace(mock_cnn, status=status, candidates=[], error=error)
    result = reconcile(prepared, [estimator], {})
    result.validate()
    assert result.clusters == result.shortlist == []
    assert not result.stage2_passed_any and not result.is_stub
    assert result.notes == [note]


def test_out_of_bounds_candidate_is_rejected(prepared, cfg, mock_cnn):
    with pytest.raises(ValueError, match="outside injected period bounds"):
        reconcile(prepared, [mock_cnn], dict(cfg, period_max_s=10))
