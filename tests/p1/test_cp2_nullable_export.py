from dataclasses import replace
import json
from unittest.mock import Mock

import pytest

from core.schemas import (AnalysisResult, LOW_QUALITY, NOT_RUN, QualityReport,
                          VERIFICATION_INCOMPLETE, assert_contract_equal)
from evaluation.cp2_evidence import analysis_disposition, analysis_evidence
from pipeline import controller
from tests.verification_cases import with_candidate_fits


def fail(*args, **kwargs):
    raise RuntimeError("injected failure")


def test_completed_exports_preserve_legacy_fields(monkeypatch, observation, injected_config,
                                                   mock_cnn, mock_recon):
    monkeypatch.setattr(controller, "run_cnn", Mock(return_value=mock_cnn))
    monkeypatch.setattr(controller, "reconcile", Mock(return_value=mock_recon))
    ambiguous = controller.analyze(observation, injected_config)
    insufficient = replace(ambiguous, quality=QualityReport("insufficient", {}, [], [LOW_QUALITY]),
                           reconciliation=None, verification=None,
                           estimators={key: replace(item, status="skipped", candidates=[], error=NOT_RUN)
                                       for key, item in ambiguous.estimators.items()})
    inconclusive = replace(insufficient, verdict=controller._fallback_verdict(
        insufficient.quality, insufficient.estimators, None, None))
    finished = [replace(record, status="ok", passed=True, fit_error=.1, heldout_error=.2,
                        best_params={"shape": "fixture", "geometry_errors": {"fixture": .2},
                                     "fit_space": "y_norm", "scale_a": 1., "offset_b": 0.})
                for record in ambiguous.verification.per_candidate]
    finished[1] = replace(finished[1], heldout_error=.3)
    verification = with_candidate_fits(ambiguous.verification, finished)
    supported = replace(ambiguous, verification=verification,
                        verdict=replace(ambiguous.verdict, label="Supported", reason_codes=[],
                                        best_period_s=8., best_cluster_id=1))
    for result, label in ((supported, "Supported"), (ambiguous, "Ambiguous"),
                          (inconclusive, "Inconclusive")):
        assert analysis_disposition(result) == ("completed", None)
        assert analysis_evidence(result) == {
            "controller_stage_errors": result.stage_errors, "controller_verdict": label}
        assert result.verdict.reason_codes is not None


@pytest.fixture
def pending_recon(mock_recon):
    return replace(mock_recon, shortlist=[], stage2_passed_any=False,
                   notes=["stage2_not_evaluated"], is_stub=True)


def test_pending_and_all_execution_states(monkeypatch, observation, injected_config, mock_cnn, mock_recon, pending_recon):
    monkeypatch.setattr(controller, "run_cnn", Mock(return_value=mock_cnn))
    monkeypatch.setattr(controller, "reconcile", Mock(return_value=pending_recon))
    pending = controller.analyze(observation, injected_config)
    cases = [(pending, "stage2_pending", None)]
    for name, target in (("prepare", "prepare"), ("stage1", "assess_stage1"),
                         ("reconciliation", "reconcile")):
        with monkeypatch.context() as scoped:
            scoped.setattr(controller, target, fail)
            cases.append((controller.analyze(observation, injected_config), "execution_error", name))
    with monkeypatch.context() as scoped:
        scoped.setattr(controller, "reconcile", Mock(return_value=mock_recon))
        original = controller.verify_candidates
        def complete(*args, **kwargs):
            base = original(*args, **kwargs)
            return with_candidate_fits(base, [
                replace(record, status="ok", passed=True, fit_error=.1, heldout_error=.2,
                        best_params={"shape": "fixture", "geometry_errors": {"fixture": .2},
                                     "fit_space": "y_norm", "scale_a": 1., "offset_b": 0.})
                for record in base.per_candidate])
        scoped.setattr(controller, "verify_candidates", complete)
        scoped.setattr(controller, "decide", fail)
        cases.append((controller.analyze(observation, injected_config), "execution_error", "verdict"))
    for result, disposition, stage in cases:
        exported = analysis_evidence(result)
        assert analysis_disposition(result) == (disposition, stage)
        assert exported["controller_verdict"] is None
        assert exported["failed_stage"] == stage
        assert exported["controller_stage_errors"] == result.stage_errors
        assert_contract_equal(result, AnalysisResult.from_dict(json.loads(json.dumps(exported["analysis_result"]))))
        assert exported["analysis_result"]["estimators"] == result.to_dict()["estimators"]
    multiple = replace(cases[3][0], stage_errors={**cases[3][0].stage_errors, "run_config": "secondary"})
    assert analysis_evidence(multiple)["failed_stage"] == "reconciliation"
    with pytest.raises(ValueError):
        analysis_evidence(replace(pending, reconciliation=None))


def test_real_report_counts_pending_without_scientific_prediction(monkeypatch, pending_recon):
    from pathlib import Path
    from core.config import load_config, resolve_period_bounds
    from evaluation.cp2_evidence import ROOT, real_outputs
    config = load_config()
    bounds = resolve_period_bounds(config.period_bounds)
    classical = dict(config.classical, period_min_s=bounds.min_period_s,
                     period_max_s=bounds.max_period_s)
    reconcile = controller.reconcile

    def reconcile_with_pending(prep, estimators, cfg):
        if any(item.candidates for item in estimators):
            return pending_recon
        return reconcile(prep, estimators, cfg)

    monkeypatch.setattr(controller, "reconcile", reconcile_with_pending)
    report = real_outputs(config, classical, ROOT / "data/sdlcd_raw")
    assert report["trial_count"] == 3
    assert report["scientific_prediction_denominator"] == report["disposition_counts"].get("completed", 0)
    assert sum(report["disposition_counts"].values()) == 3
    assert any(row.get("analysis_disposition") == "stage2_pending" for row in report["outputs"])
    assert report["controller_matches_direct_excluding_runtime"] is True
    for row in report["outputs"]:
        if row.get("analysis_disposition") == "stage2_pending":
            assert row["controller_verdict"] is None
            assert row["analysis_result"]["estimators"]
            assert row["analysis_result"]["quality"] is not None
        else:
            assert row["controller_verdict"] in ("Supported", "Ambiguous", "Inconclusive")


def test_real_report_records_execution_failures(monkeypatch):
    from core.config import load_config, resolve_period_bounds
    from evaluation.cp2_evidence import ROOT, real_outputs
    config = load_config()
    bounds = resolve_period_bounds(config.period_bounds)
    classical = dict(config.classical, period_min_s=bounds.min_period_s,
                     period_max_s=bounds.max_period_s)
    monkeypatch.setattr(controller, "reconcile", fail)
    report = real_outputs(config, classical, ROOT / "data/sdlcd_raw")
    assert report["trial_count"] == 3
    assert report["disposition_counts"]["execution_error"] >= 1
    assert report["scientific_prediction_denominator"] == report["disposition_counts"].get("completed", 0)
    assert report["controller_matches_direct_excluding_runtime"] is None
    for row in report["outputs"]:
        if row.get("analysis_disposition") == "execution_error":
            assert row["controller_verdict"] is None
            assert row["failed_stage"] == "reconciliation"
            assert row["controller_stage_errors"]["reconciliation"] == "injected failure"
            assert row["analysis_result"]["estimators"]


@pytest.mark.parametrize("stage,target", [("prepare", "prepare"), ("stage1", "assess_stage1")])
def test_real_report_upstream_failure(monkeypatch, stage, target):
    from core.config import load_config, resolve_period_bounds
    from evaluation import cp2_evidence as exporter
    config = load_config()
    bounds = resolve_period_bounds(config.period_bounds)
    classical = dict(config.classical, period_min_s=bounds.min_period_s, period_max_s=bounds.max_period_s)
    monkeypatch.setattr(exporter, target, fail)
    monkeypatch.setattr(controller, target, fail)
    report = exporter.real_outputs(config, classical, exporter.ROOT / "data/sdlcd_raw")
    assert report["trial_count"] == 3
    assert report["disposition_counts"] == {"execution_error": 3}
    assert report["scientific_prediction_denominator"] == 0
    for row in report["outputs"]:
        result = AnalysisResult.from_dict(row["analysis_result"])
        assert row["failed_stage"] == stage and row["controller_verdict"] is None
        assert row["controller_stage_errors"][stage] == "injected failure"
        assert result.quality is None
        assert (result.prepared is None) == (stage == "prepare")


@pytest.mark.parametrize("failed", [("lomb_scargle",), ("pdm",), ("lomb_scargle", "pdm")])
def test_real_report_unavailable_estimators(monkeypatch, failed):
    from core.config import load_config, resolve_period_bounds
    from core.schemas import EstimatorResult
    from evaluation import cp2_evidence as exporter
    config = load_config()
    bounds = resolve_period_bounds(config.period_bounds)
    classical = dict(config.classical, period_min_s=bounds.min_period_s, period_max_s=bounds.max_period_s)
    for method in failed:
        def unavailable(prep, cfg, method=method):
            return EstimatorResult(method, "failed", [], "injected failure", {"runtime_s": 0.})
        monkeypatch.setattr(controller, "run_" + method, unavailable)
    report = exporter.real_outputs(config, classical, exporter.ROOT / "data/sdlcd_raw")
    assert report["trial_count"] == 3
    assert sum(report["disposition_counts"].values()) == 3
    assert report["scientific_prediction_denominator"] == report["disposition_counts"].get("completed", 0)
    assert report["controller_matches_direct_excluding_runtime"] is None
    for row in report["outputs"]:
        assert row.get("analysis_disposition") != "execution_error"
        for method in ("lomb_scargle", "pdm"):
            item = row["estimators"][method]
            assert item["method"] == method
            if method in failed:
                assert item["status"] == "failed" and item["candidates"] == []
                assert item["error"] == "injected failure" and item["direct_comparison"] is None
            else:
                assert item["status"] == "ok"
                if row.get("analysis_result"):
                    assert item["candidates"] == row["analysis_result"]["estimators"][method]["candidates"]


@pytest.mark.parametrize("methods", [("lomb_scargle",), ("pdm",), ("lomb_scargle", "pdm")])
@pytest.mark.parametrize("failure_kind", ["raises", "failed", "skipped"])
def test_optional_direct_comparison_failure(monkeypatch, methods, failure_kind):
    from core.config import load_config, resolve_period_bounds
    from core.schemas import EstimatorResult
    from evaluation import cp2_evidence as exporter
    config = load_config()
    bounds = resolve_period_bounds(config.period_bounds)
    classical = dict(config.classical, period_min_s=bounds.min_period_s, period_max_s=bounds.max_period_s)
    integrated_results = {}
    analyze = controller.analyze
    def capture(obs, cfg):
        result = analyze(obs, cfg)
        integrated_results[obs.obs_id] = result
        return result
    monkeypatch.setattr(controller, "analyze", capture)
    for method in methods:
        def comparison(prep, cfg, method=method):
            if failure_kind == "raises":
                raise RuntimeError("comparison-only failure")
            return EstimatorResult(method, failure_kind, [],
                                   NOT_RUN if failure_kind == "skipped" else "comparison-only failure",
                                   {"runtime_s": 0.})
        module = exporter.ls if method == "lomb_scargle" else exporter.pdm
        monkeypatch.setattr(module, "run_" + method, comparison)
    report = exporter.real_outputs(config, classical, exporter.ROOT / "data/sdlcd_raw")
    assert report["trial_count"] == len(exporter.SDLCD_FILES)
    assert sum(report["disposition_counts"].values()) == report["trial_count"]
    assert report["controller_matches_direct_excluding_runtime"] is None
    for row in report["outputs"]:
        result = integrated_results[row["file"].removesuffix(".txt")]
        for method in ("lomb_scargle", "pdm"):
            source = result.estimators[method]
            item = row["estimators"][method]
            assert item["method"] == method and item["status"] == source.status == "ok"
            assert item["candidates"] == source.to_dict()["candidates"]
            assert item["error"] == source.error
            if method in methods:
                assert item["direct_comparison"] is None
                assert item["direct_comparison_error"]
                assert item["meta"] == exporter.estimator_output(source)["meta"]
            else:
                assert "direct_runtime_s" in item and "controller_runtime_s" in item
                assert "direct_comparison_error" not in item
