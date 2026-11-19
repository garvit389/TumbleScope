from dataclasses import replace
from unittest.mock import Mock

from core.schemas import (
    ALL_CANDIDATES_FAILED_STAGE2, ALL_ESTIMATORS_UNAVAILABLE, CANDIDATE_INADEQUATE,
    ESTIMATOR_METHODS, LOW_QUALITY, LOMB_SCARGLE, NO_PERIODIC_SIGNAL, NOT_RUN, PDM,
    STAGE1_INSUFFICIENT, STAGE2_FAIL_REASONS, VERIFICATION_INCOMPLETE,
    EstimatorResult, QualityReport,
)
from pipeline import controller


def test_insufficient_stage1_skips_everything(monkeypatch, observation, injected_config):
    unexpected = Mock(side_effect=AssertionError("Stage 1 stopped the pipeline"))
    for name in ("run_cnn", "run_lomb_scargle", "run_pdm", "reconcile", "verify_candidates"):
        monkeypatch.setattr(controller, name, unexpected)
    monkeypatch.setattr(controller, "assess_stage1",
                        lambda *args: QualityReport("insufficient", {}, [], [LOW_QUALITY]))
    result = controller.analyze(observation, injected_config)
    assert all(item.error == STAGE1_INSUFFICIENT for item in result.estimators.values())
    assert result.reconciliation is result.verification is None
    assert result.verdict.label == "Inconclusive"
    assert result.verdict.reason_codes == [LOW_QUALITY]
    unexpected.assert_not_called()


def test_limited_runs_all_estimators_and_injects_identical_bounds(
    monkeypatch, observation, injected_config, mock_cnn, mock_recon
):
    seen = {}
    monkeypatch.setattr(controller, "assess_stage1",
                        lambda *args: QualityReport("limited", {}, ["A warning."], []))

    def capture(name, original):
        def call(*args, **kwargs):
            seen[name] = args[-1]
            return original(*args, **kwargs)
        return call

    for name, response in (
        ("run_cnn", mock_cnn),
        ("run_lomb_scargle", _classical_empty(LOMB_SCARGLE)),
        ("run_pdm", _classical_empty(PDM)),
    ):
        monkeypatch.setattr(controller, name, capture(response.method, Mock(return_value=response)))
    monkeypatch.setattr(controller, "reconcile", capture("reconcile", lambda *args: mock_recon))
    monkeypatch.setattr(controller, "verify_candidates", capture("verify", controller.verify_candidates))

    result = controller.analyze(observation, injected_config)
    assert set(seen) == {*ESTIMATOR_METHODS, "reconcile", "verify"}
    assert all((item["period_min_s"], item["period_max_s"]) == (2.0, 20.0) for item in seen.values())
    assert result.verdict.label == "Ambiguous"
    assert set(result.verdict.reason_codes) == {VERIFICATION_INCOMPLETE, LOW_QUALITY}
    assert result.verdict.evidence_summary["stage1_warnings"] == ["A warning."]


def _classical_empty(method):
    return EstimatorResult(method, "ok", [], meta={
        "runtime_s": 0.0, "periodogram": {"freq_hz": [], "power": []}, "freq_resolution": None,
    })


def test_all_unavailable_and_all_empty_are_distinct(monkeypatch, observation, injected_config, mock_cnn):
    def raise_failure(*args):
        raise RuntimeError("sample failure")

    monkeypatch.setattr(controller, "run_cnn", raise_failure)
    for method, name in ((LOMB_SCARGLE, "run_lomb_scargle"), (PDM, "run_pdm")):
        monkeypatch.setattr(controller, name, Mock(return_value=EstimatorResult(
            method, "skipped", [], NOT_RUN, {"runtime_s": 0.0}, is_stub=True,
        )))
    result = controller.analyze(observation, injected_config)
    assert result.estimators["cnn"].status == "failed"
    assert result.estimators[LOMB_SCARGLE].error == NOT_RUN
    assert result.reconciliation.clusters == []
    assert result.reconciliation.shortlist == []
    assert result.verification is None
    assert result.verdict.reason_codes == [CANDIDATE_INADEQUATE]
    assert result.verdict.evidence_summary["inconclusive_cause"] == ALL_ESTIMATORS_UNAVAILABLE
    assert result.stage_errors["cnn"] == "sample failure"
    assert result.verdict.evidence_summary["n_estimators_available"] == 0

    monkeypatch.setattr(controller, "run_cnn", lambda *args: replace(mock_cnn, candidates=[]))
    monkeypatch.setattr(controller, "run_lomb_scargle", lambda *args: _classical_empty(LOMB_SCARGLE))
    monkeypatch.setattr(controller, "run_pdm", lambda *args: _classical_empty(PDM))
    result = controller.analyze(observation, injected_config)
    assert all(item.status == "ok" for item in result.estimators.values())
    assert result.verdict.reason_codes == [NO_PERIODIC_SIGNAL]
    assert result.reconciliation.notes == ["no_candidates"]
    assert result.verification is None


def test_stage2_all_fail(monkeypatch, observation, injected_config, mock_cnn, mock_recon):
    first, second = STAGE2_FAIL_REASONS[:2]
    rejected = [replace(cluster, stage2={**cluster.stage2, "passed": False, "fail_reasons": reasons})
                for cluster, reasons in zip(mock_recon.clusters, ([first], [first, second]))]
    recon = replace(mock_recon, clusters=rejected, shortlist=[], stage2_passed_any=False)
    monkeypatch.setattr(controller, "run_cnn", lambda *args: mock_cnn)
    monkeypatch.setattr(controller, "reconcile", lambda *args: recon)
    verifier = Mock(side_effect=AssertionError("no candidate passed Stage 2"))
    monkeypatch.setattr(controller, "verify_candidates", verifier)
    result = controller.analyze(observation, injected_config)
    assert result.reconciliation.shortlist == []
    assert not result.reconciliation.stage2_passed_any
    assert result.verification is None
    assert result.verdict.reason_codes == [CANDIDATE_INADEQUATE]
    assert result.verdict.evidence_summary["inconclusive_cause"] == ALL_CANDIDATES_FAILED_STAGE2
    assert result.verdict.evidence_summary["stage2_fail_reasons"] == sorted([first, second])
    verifier.assert_not_called()
