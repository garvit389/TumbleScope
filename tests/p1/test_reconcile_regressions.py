"""CP3a regressions exercising the real reconciliation and Stage 2 path."""

from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pytest

from core.config import load_config
from core.schemas import (
    CNN, LOMB_SCARGLE, PDM, NOT_RUN, AnalysisResult, EstimatorResult, Observation,
    ReconciliationResult, assert_contract_equal,
)
from reconciliation import aliases
from reconciliation.reconcile import reconcile
from reconciliation.harmonics import harmonic_ratio
from tests.p1.test_reconcile import (
    candidate, classical, synthetic, INSUFFICIENT_CYCLES, UNRESOLVED_FREQUENCY,
    POOR_PHASE_COVERAGE, MERGED_INTO_NEIGHBOUR,
)


def window(prep, peaks=()):
    grid = {0.0: 1.0, 1.0: 0.0, **{p["freq_hz"]: p["height"] for p in peaks}}
    frequencies = sorted(grid)
    return {"freq_hz": frequencies, "window_power": [grid[f] for f in frequencies],
            "peaks": list(peaks), "freq_resolution": prep.meta["freq_resolution"]}


def settings(prep, **overrides):
    return {**load_config().reconcile, "period_min_s": 2.0, "period_max_s": 100.0,
            "window_result": window(prep), **overrides}


def test_injected_window_is_used_without_recomputation(mock_cnn, monkeypatch):
    prep = synthetic(np.linspace(0, 200, 401))
    injected = window(prep, [{"freq_hz": 1 / 8 - 1 / 11, "height": 0.8}])
    before = deepcopy(injected)
    monkeypatch.setattr(aliases, "compute_window", lambda *args: pytest.fail("window recomputed"))
    cnn = replace(mock_cnn, candidates=[candidate(8., 1.), candidate(11., .3)])
    result = reconcile(prep, [cnn], settings(prep, window_result=injected))
    assert next(c for c in result.clusters if c.period_s == 11).alias_risk == pytest.approx(.8)
    assert injected == before


@pytest.mark.parametrize("bad", [None, {}, {"peaks": []},
    {"freq_hz": [], "window_power": [], "peaks": [], "freq_resolution": .005},
    {"freq_hz": [0.], "window_power": [1.], "peaks": [{"freq_hz": 0., "height": 1.}], "freq_resolution": .005},
    {"freq_hz": [0.], "window_power": [1.], "peaks": [{"freq_hz": .1, "height": float("nan")}], "freq_resolution": .005},
    {"freq_hz": [0.], "window_power": [1.], "peaks": [], "freq_resolution": .1},
    {"freq_hz": [0., .1], "window_power": [1., .2], "peaks": [{"freq_hz": .1, "height": .8}], "freq_resolution": .005},
])
def test_malformed_injected_window_is_not_silently_replaced(mock_cnn, bad):
    prep = synthetic(np.linspace(0, 200, 401))
    with pytest.raises(ValueError, match="contract:.*window_result"):
        reconcile(prep, [mock_cnn], settings(prep, window_result=bad))


@pytest.mark.parametrize("field,bad", [("raw_score", float("nan")), ("raw_score", float("inf")),
    ("support", float("nan")), ("period_s", 0.), ("period_s", float("inf"))])
def test_invalid_candidate_is_rejected_even_when_duplicate_would_hide_it(mock_cnn, field, bad):
    prep = synthetic(np.linspace(0, 200, 401))
    broken = replace(candidate(8., .1), **{field: bad})
    cnn = replace(mock_cnn, candidates=[candidate(8., 1.), broken])
    with pytest.raises(ValueError, match="contract:"):
        reconcile(prep, [cnn], settings(prep))


@pytest.mark.parametrize("width", [-.1, 0., float("nan"), float("inf")])
def test_invalid_frequency_width_is_not_a_stage2_pass(mock_cnn, width):
    prep = synthetic(np.linspace(0, 200, 401))
    cnn = replace(mock_cnn, candidates=[candidate(8., 1., width)])
    with pytest.raises(ValueError, match="contract:"):
        reconcile(prep, [cnn], settings(prep))


@pytest.mark.parametrize("delta,expected", [(0., 1), (-1e-8, 1), (1e-8, 2)])
def test_frequency_cluster_boundary(mock_cnn, delta, expected):
    prep = synthetic(np.linspace(0, 100, 401))
    cnn = replace(mock_cnn, candidates=[candidate(10., 1.), candidate(1 / (.09 - delta), .8)])
    result = reconcile(prep, [cnn], settings(prep))
    assert len(result.clusters) == expected


def test_neighbour_separation_boundary_is_inclusive(mock_cnn):
    prep = synthetic(np.linspace(0, 64, 401))
    cnn = replace(mock_cnn, candidates=[candidate(8., 1.), candidate(32 / 3, .8)])
    result = reconcile(prep, [cnn], settings(prep, min_separation_resolutions=2.))
    assert result.clusters[1].stage2["fail_reasons"] == [MERGED_INTO_NEIGHBOUR]


def test_phase_coverage_and_baseline_ignore_invalid_samples(mock_cnn):
    times = np.arange(0., 121.)
    values = np.sin(2 * np.pi * times / 10)
    values[(times < 10) | (times > 100) | (times % 10 >= 2)] = np.nan
    prep = synthetic(times, values)
    cnn = replace(mock_cnn, candidates=[candidate(10., 1.)])
    result = reconcile(prep, [cnn], settings(prep))
    gate = result.clusters[0].stage2
    assert gate["cycles"] == 9.
    assert gate["phase_coverage"] == .2
    assert gate["fail_reasons"] == [POOR_PHASE_COVERAGE]


def test_missing_width_fallback_and_largest_member_width(mock_cnn):
    prep = synthetic(np.linspace(0, 200, 401))
    cnn = replace(mock_cnn, candidates=[candidate(8., 1.)])
    cfg = settings(prep)
    alone = reconcile(prep, [cnn], cfg)
    assert alone.clusters[0].stage2["rel_period_uncertainty"] == .04
    result = reconcile(prep, [cnn, classical(LOMB_SCARGLE, candidate(8., 1., .07)),
                              classical(PDM, candidate(8., 1., .02))], cfg)
    assert result.clusters[0].stage2["rel_period_uncertainty"] == pytest.approx(.56)
    assert result.clusters[0].stage2["fail_reasons"] == [UNRESOLVED_FREQUENCY]


def test_harmonics_survive_clustering_even_within_one_resolution(mock_cnn):
    prep = synthetic(np.linspace(0, 8, 401))
    cnn = replace(mock_cnn, candidates=[candidate(8., 1.), candidate(16., .9), candidate(24., .8)])
    result = reconcile(prep, [cnn], settings(prep))
    assert len(result.clusters) == 3
    for cluster in result.clusters:
        assert len(cluster.harmonic_links) == 2
        assert INSUFFICIENT_CYCLES in cluster.stage2["fail_reasons"]
    result.validate()


def test_zero_baseline_and_empty_estimators_are_serializable(mock_cnn):
    prep = synthetic(np.array([0.]), np.array([1.]))
    result = reconcile(prep, [mock_cnn], settings(prep))
    assert not result.shortlist
    assert all(UNRESOLVED_FREQUENCY in c.stage2["fail_reasons"] for c in result.clusters)
    clone = ReconciliationResult.from_dict(json.loads(json.dumps(result.to_dict(), allow_nan=False)))
    assert_contract_equal(result, clone)
    assert reconcile(prep, [], {}).notes == ["no_usable_estimators"]


def test_all_available_weights_follow_config_equation_and_ignore_raw_scores(mock_cnn):
    prep = synthetic(np.linspace(0, 200, 401))
    cnn = replace(mock_cnn, candidates=[replace(candidate(8., .9), raw_score=.001)])
    ls = classical(LOMB_SCARGLE, replace(candidate(8., .5, .005), raw_score=1000.))
    pdm = classical(PDM, candidate(8., .4, .005))
    result = reconcile(prep, [cnn, ls, pdm], settings(prep, w_C=.6, w_L=.3, w_D=.1))
    assert result.clusters[0].score == pytest.approx(.6 * .9 + .3 * .5 + .1 * .4)


def test_empty_peaks_never_penalize_and_inputs_are_unchanged(mock_cnn):
    prep = synthetic(np.linspace(0, 200, 401))
    cfg = settings(prep)
    before = (deepcopy(cfg), prep.to_dict(), mock_cnn.to_dict())
    result = reconcile(prep, [mock_cnn], cfg)
    assert all(c.alias_risk == 0. for c in result.clusters)
    assert before == (cfg, prep.to_dict(), mock_cnn.to_dict())
    assert_contract_equal(result, reconcile(prep, [mock_cnn], cfg))


def test_clustering_does_not_erase_a_members_harmonic_hypothesis(mock_cnn):
    prep = synthetic(np.linspace(0, 200, 401))
    cnn = replace(mock_cnn, candidates=[candidate(16.4, 1.), candidate(8., .9), candidate(16., .8)])
    result = reconcile(prep, [cnn], settings(prep, harmonic_ratio_tolerance=.01))
    short = next(c for c in result.clusters if c.period_s == 8.)
    partner = next(c for c in result.clusters if c.period_s == 16.)
    assert {"other_id": partner.cluster_id, "ratio": 2.} in short.harmonic_links


def test_extreme_weights_cannot_silently_zero_available_evidence(mock_cnn):
    prep = synthetic(np.linspace(0, 200, 401))
    cfg = settings(prep, w_C=1e308, w_L=1e308)
    with pytest.raises(ValueError, match="contract:"):
        reconcile(prep, [mock_cnn, classical(LOMB_SCARGLE)], cfg)


@pytest.mark.parametrize("nominal", [1.5, 2., 3.])
@pytest.mark.parametrize("delta,linked", [(0., True), (-1e-8, True), (1e-8, False)])
def test_harmonic_tolerance_boundary_is_symmetric(nominal, delta, linked):
    cfg = {"harmonic_ratio_tolerance": .03}
    period, other = 8., 8. * nominal * (1.03 + delta)
    ratio = harmonic_ratio(period, other, cfg)
    reciprocal = harmonic_ratio(other, period, cfg)
    assert (ratio is not None) == linked
    if linked:
        assert ratio == nominal and reciprocal == 1 / nominal


@pytest.mark.parametrize("cycles,expected", [(1.999, False), (2., True), (2.001, True)])
def test_cycle_threshold_boundary(mock_cnn, cycles, expected):
    prep = synthetic(np.linspace(0, cycles * 10., 401))
    cnn = replace(mock_cnn, candidates=[candidate(10., 1., .001)])
    result = reconcile(prep, [cnn], settings(prep))
    assert (INSUFFICIENT_CYCLES not in result.clusters[0].stage2["fail_reasons"]) == expected


@pytest.mark.parametrize("width,expected", [(.04999, True), (.05, True), (.05001, False)])
def test_uncertainty_threshold_boundary(mock_cnn, width, expected):
    prep = synthetic(np.linspace(0, 200, 401))
    cnn = replace(mock_cnn, candidates=[candidate(10., 1., width)])
    result = reconcile(prep, [cnn], settings(prep))
    assert (UNRESOLVED_FREQUENCY not in result.clusters[0].stage2["fail_reasons"]) == expected


@pytest.mark.parametrize("minimum,expected", [(.19999, True), (.2, True), (.20001, False)])
def test_coverage_threshold_boundary(mock_cnn, minimum, expected):
    times = np.arange(0., 121.)
    values = np.sin(2 * np.pi * times / 10.)
    values[times % 10 >= 2] = np.nan
    prep = synthetic(times, values)
    cnn = replace(mock_cnn, candidates=[candidate(10., 1.)])
    result = reconcile(prep, [cnn], settings(prep, min_phase_coverage=minimum))
    assert (POOR_PHASE_COVERAGE not in result.clusters[0].stage2["fail_reasons"]) == expected


def test_equal_score_neighbours_do_not_reject_each_other(mock_cnn):
    prep = synthetic(np.linspace(0, 64, 401))
    cnn = replace(mock_cnn, candidates=[candidate(8., 1.), candidate(32 / 3, 1.)])
    result = reconcile(prep, [cnn], settings(prep, min_separation_resolutions=3.))
    assert len(result.shortlist) == 2


@pytest.mark.parametrize("shift,risk", [(0., .8), (-1e-8, .8), (1e-8, 0.)])
def test_alias_match_at_tolerance_boundary(mock_cnn, shift, risk):
    prep = synthetic(np.linspace(0, 200, 401))
    offset = 1 / 8 - 1 / 11 + prep.meta["freq_resolution"] + shift
    cnn = replace(mock_cnn, candidates=[candidate(8., 1.), candidate(11., .3), candidate(20., .2)])
    cfg = settings(prep, window_result=window(prep, [{"freq_hz": offset, "height": .8}]))
    result = reconcile(prep, [cnn], cfg)
    by_period = {c.period_s: c for c in result.clusters}
    assert by_period[11.].alias_risk == risk
    assert by_period[8.].alias_risk == by_period[20.].alias_risk == 0.


def test_controller_uses_real_reconciliation_and_one_shared_window(monkeypatch, isolated_runs):
    from pipeline import controller

    config = load_config()
    config.cnn = {}
    config.verification = {**config.verification, "timeout_s_per_candidate": .001}
    config.classical = {**config.classical, "n_peaks": 0}
    times = np.linspace(0., 640., 401)
    phase = 2 * np.pi * times / 32.
    observation = Observation("cp3a-integration", times, np.cos(2 * phase) + .35 * np.sin(phase),
                              None, "unknown")
    # Only the unavailable CNN is isolated; classical analysis, reconciliation,
    # Stage 2, verification failure handling and controller assembly are real.
    monkeypatch.setattr(controller, "run_cnn", lambda *args: EstimatorResult(
        CNN, "skipped", [], NOT_RUN, {"runtime_s": 0.}))
    original = controller.compute_window
    windows = []

    def capture(prep, cfg):
        windows.append(original(prep, cfg))
        return windows[-1]

    monkeypatch.setattr(controller, "compute_window", capture)
    monkeypatch.setattr(aliases, "compute_window", lambda *args: pytest.fail("second window computation"))
    result = controller.analyze(observation, config)
    assert len(windows) == 1 and windows[0]["peaks"] == []
    assert "reconciliation" not in result.stage_errors
    assert result.reconciliation is not None and not result.reconciliation.is_stub
    recon = result.reconciliation
    assert len(recon.shortlist) == min(3, sum(c.stage2["passed"] for c in recon.clusters))
    assert recon.shortlist
    assert all(c.alias_risk == 0. for c in recon.clusters)
    assert result.verdict is not None and result.verdict.label != "Supported"
    assert_contract_equal(result, AnalysisResult.from_dict(
        json.loads(json.dumps(result.to_dict(), allow_nan=False))))
