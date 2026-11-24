from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest

from core.schemas import (
    AnalysisResult, NOT_RUN, EstimatorResult, _execution_error,
    assert_contract_equal,
)
from pipeline import controller
from tests.p2.test_controller import pending_reconciliation, unavailable_cnn


def failure(*args, **kwargs):
    raise RuntimeError("injected technical failure")


def _fixture_upload(monkeypatch):
    from io import BytesIO
    import streamlit as st

    path = Path(__file__).resolve().parents[1] / "fixtures/synthetic_sine.csv"
    upload = BytesIO(path.read_bytes())
    upload.name = path.name
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: upload)


@pytest.mark.parametrize("eager", [True, False])
@pytest.mark.parametrize("kind", ["exception", "invalid", "missing"])
@pytest.mark.parametrize("stage,function", [("prepare", "prepare"), ("stage1", "assess_stage1"),
                                            ("reconciliation", "reconcile")])
def test_execution_failures(monkeypatch, observation, injected_config, mock_cnn, eager, kind, stage, function):
    monkeypatch.setattr(controller, "run_cnn", lambda *a: mock_cnn)
    operation = failure if kind == "exception" else Mock(return_value=None if kind == "missing" else object())
    monkeypatch.setattr(controller, function, operation)
    result = controller.analyze(observation, replace(injected_config, pipeline={"validate_outputs": eager}))
    assert result.verdict is result.verification is None
    assert result.stage_errors[stage]
    assert _execution_error(result.prepared, result.quality, result.estimators,
                            result.reconciliation, result.verification, result.stage_errors) == stage
    if stage == "prepare":
        assert result.prepared is result.quality is None
    elif stage == "stage1":
        assert result.prepared is not None and result.quality is None
    else:
        assert result.quality is not None and result.estimators["cnn"].candidates
    if stage != "reconciliation":
        assert all(v.status == "skipped" and v.error == NOT_RUN for v in result.estimators.values())
    assert_contract_equal(result, AnalysisResult.from_dict(json.loads(json.dumps(result.to_dict()))))
    for errors in ({}, {stage: " "}, {"unrelated": "failure"}):
        with pytest.raises(ValueError):
            replace(result, stage_errors=errors).validate()
    for version in (1, 2, 3):
        with pytest.raises(ValueError, match="schema version"):
            AnalysisResult.from_dict({**result.to_dict(), "schema_version": version})


def completed(prep, recon, cfg, **kwargs):
    from tests.verification_cases import with_candidate_fits

    stub = controller.verify_candidates_original(prep, recon, cfg, **kwargs)
    return with_candidate_fits(replace(stub, is_stub=False), [
        replace(r, status="ok", passed=True, fit_error=.1, heldout_error=.2,
                best_params={"shape": "fixture", "geometry_errors": {"fixture": .2},
                             "fit_space": "y_norm", "scale_a": 1., "offset_b": 0.})
        for r in stub.per_candidate])


@pytest.mark.parametrize("eager", [True, False])
@pytest.mark.parametrize("kind", ["exception", "invalid", "missing"])
def test_verdict_failure_keeps_completed_evidence(monkeypatch, observation, injected_config, mock_cnn, mock_recon, eager, kind):
    monkeypatch.setattr(controller, "run_cnn", lambda *a: mock_cnn)
    monkeypatch.setattr(controller, "reconcile", lambda *a: mock_recon)
    monkeypatch.setattr(controller, "verify_candidates_original", controller.verify_candidates, raising=False)
    verifier = Mock(side_effect=completed)
    monkeypatch.setattr(controller, "verify_candidates", verifier)
    monkeypatch.setattr(controller, "decide", failure if kind == "exception" else
                        Mock(return_value=None if kind == "missing" else object()))
    result = controller.analyze(observation, replace(injected_config, pipeline={"validate_outputs": eager}))
    assert result.verdict is None
    assert result.verification.complete and all(r.passed for r in result.verification.per_candidate)
    assert result.reconciliation is mock_recon
    assert result.stage_errors["verdict"]
    assert _execution_error(result.prepared, result.quality, result.estimators,
                            result.reconciliation, result.verification, result.stage_errors) == "verdict"
    assert_contract_equal(result, AnalysisResult.from_dict(result.to_dict()))
    for invalid in (
        replace(result, stage_errors={"unrelated": "failure"}),
        replace(result, verification=None),
        replace(result, quality=None),
        replace(result, prepared=None),
        replace(result, reconciliation=None, verification=None),
    ):
        with pytest.raises(ValueError):
            invalid.validate()


@pytest.mark.parametrize("outcome", ["incomplete", "empty", "unavailable", "rejected", "quality", "mismatch"])
def test_independent_fallback(monkeypatch, observation, prepared, injected_config, mock_cnn, mock_recon, outcome):
    from core.schemas import QualityReport, LOW_QUALITY, STAGE2_FAIL_REASONS
    monkeypatch.setattr(controller, "run_cnn", lambda *a: mock_cnn)
    monkeypatch.setattr(controller, "reconcile", lambda *a: mock_recon)
    if outcome == "incomplete":
        verification = controller.verify_candidates(prepared, mock_recon, {})
        monkeypatch.setattr(controller, "verify_candidates", Mock(return_value=verification))
    elif outcome in ("empty", "unavailable"):
        results = {"run_cnn": mock_cnn}
        for method, function in (("lomb_scargle", "run_lomb_scargle"), ("pdm", "run_pdm")):
            results[function] = EstimatorResult(method, "ok", [], meta={
                "runtime_s": 0., "freq_resolution": None,
                "periodogram": {"freq_hz": [], "power": []}})
        for name, item in results.items():
            if outcome == "empty":
                item = replace(item, candidates=[])
            else:
                item = replace(item, status="skipped", candidates=[], error=NOT_RUN)
            monkeypatch.setattr(controller, name, Mock(return_value=item))
        monkeypatch.setattr(controller, "reconcile", failure)
    elif outcome == "rejected":
        rejected = replace(mock_recon, shortlist=[], stage2_passed_any=False, clusters=[
            replace(c, stage2={"passed": False, "cycles": 1., "fail_reasons": [STAGE2_FAIL_REASONS[0]]})
            for c in mock_recon.clusters])
        monkeypatch.setattr(controller, "reconcile", lambda *a: rejected)
    elif outcome == "quality":
        monkeypatch.setattr(controller, "assess_stage1", lambda *a: QualityReport("insufficient", {}, [], [LOW_QUALITY]))
    elif outcome == "mismatch":
        monkeypatch.setattr(controller, "verify_candidates_original", controller.verify_candidates, raising=False)
        finished = completed(prepared, mock_recon, {})
        verification = replace(finished, per_candidate=[
            replace(record, passed=False) for record in finished.per_candidate
        ])
        monkeypatch.setattr(controller, "verify_candidates", Mock(return_value=verification))
    expected = controller.analyze(observation, injected_config)
    monkeypatch.setattr(controller, "decide", failure)
    result = controller.analyze(observation, injected_config)
    assert result.verdict.label == expected.verdict.label
    assert result.verdict.reason_codes == expected.verdict.reason_codes
    assert result.verdict.evidence_summary == expected.verdict.evidence_summary
    assert_contract_equal(result.verification, expected.verification)
    assert result.stage_errors["verdict"]


def test_multiple_errors_and_invalid_combinations(monkeypatch, observation, injected_config, mock_cnn):
    monkeypatch.setattr(controller, "run_cnn", lambda *a: mock_cnn)
    monkeypatch.setattr(controller, "reconcile", pending_reconciliation)
    good = controller.analyze(observation, injected_config)
    monkeypatch.setattr(controller, "_write_run_config", failure)
    monkeypatch.setattr(controller, "reconcile", failure)
    result = controller.analyze(observation, injected_config)
    assert {"run_config", "reconciliation"} <= result.stage_errors.keys()
    replace(result, stage_errors={**result.stage_errors, "verdict": "secondary failure"}).validate()
    # Unrelated diagnostics cannot change the approved pending-state predicate.
    replace(good, stage_errors={"run_config": "failure"}).validate()
    for bad in (
        replace(result, prepared=None),
        replace(result, quality=None),
        replace(result, estimators={k: replace(v, candidates=[]) for k, v in result.estimators.items()}),
    ):
        with pytest.raises(ValueError):
            bad.validate()


def test_real_shortlist_consistency(mock_recon):
    real = replace(mock_recon, is_stub=False)
    with pytest.raises(ValueError, match="shortlist"):
        replace(real, shortlist=[], stage2_passed_any=False).validate()


def test_incomplete_real_shortlist_is_discarded(monkeypatch, observation, injected_config, mock_cnn, mock_recon):
    monkeypatch.setattr(controller, "run_cnn", lambda *a: mock_cnn)
    monkeypatch.setattr(controller, "reconcile", lambda *a: replace(mock_recon, is_stub=False))
    result = controller.analyze(observation, injected_config)
    assert result.reconciliation is result.verdict is None
    assert "incomplete Fast mode shortlist" in result.stage_errors["reconciliation"]


@pytest.mark.parametrize("stage,function", [("prepare", "prepare"), ("stage1", "assess_stage1"),
                                            ("reconciliation", "reconcile")])
def test_error_dashboard(monkeypatch, isolated_runs, stage, function):
    monkeypatch.setattr(controller, "run_cnn", unavailable_cnn)
    monkeypatch.setattr(controller, function, failure)
    _fixture_upload(monkeypatch)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[2] / "app/main.py")).run(timeout=60)
    app.button[0].click().run(timeout=60)
    assert not app.exception
    assert any(h.value == "Technical execution failure" for h in app.subheader)
    assert all("injected technical failure" not in c.value for c in app.caption)
    assert not any(h.value in ("Inconclusive", "Ambiguous", "Supported", "Stage 2 assessment pending") for h in app.subheader)
    assert any({"prepare": "Preparation", "stage1": "Stage 1 quality",
                "reconciliation": "Reconciliation"}[stage] + ": the stage could not finish." in c.value for c in app.caption)
    if stage == "reconciliation":
        tables = [t.value for t in app.table if {"Estimator", "Apparent (synodic) period (s)"} <= set(t.value.columns)]
        assert {t["Estimator"].iloc[0] for t in tables} == {"lomb_scargle", "pdm"}


def test_nullable_consumer(monkeypatch, observation, injected_config, mock_cnn):
    from scripts.validate_cp1 import _verdict_category
    monkeypatch.setattr(controller, "run_cnn", lambda *a: mock_cnn)
    monkeypatch.setattr(controller, "reconcile", pending_reconciliation)
    pending = controller.analyze(observation, injected_config)
    assert _verdict_category(pending) == "stage2_pending"
    monkeypatch.setattr(controller, "reconcile", failure)
    error = controller.analyze(observation, injected_config)
    assert _verdict_category(error) == "execution_error"
    from core.schemas import QualityReport, LOW_QUALITY
    monkeypatch.setattr(controller, "assess_stage1", lambda *a: QualityReport("insufficient", {}, [], [LOW_QUALITY]))
    final = controller.analyze(observation, injected_config)
    assert _verdict_category(final) == "Inconclusive"
    with pytest.raises(ValueError):
        _verdict_category(replace(final, verdict=None, stage_errors={"unrelated": "failure"}))


@pytest.mark.parametrize("stage,function", [("prepare", "prepare"), ("stage1", "assess_stage1")])
def test_earliest_error_precedence(monkeypatch, observation, injected_config, mock_cnn, stage, function):
    monkeypatch.setattr(controller, function, failure)
    result = controller.analyze(observation, injected_config)
    all_errors = dict.fromkeys(("prepare", "stage1", "reconciliation", "verdict"), "failure")
    combined = replace(result, stage_errors=all_errors)
    combined.validate()
    assert _execution_error(combined.prepared, combined.quality, combined.estimators,
                            combined.reconciliation, combined.verification, all_errors) == stage
    with pytest.raises(ValueError):
        replace(combined, estimators={**combined.estimators, "cnn": mock_cnn}).validate()
    with pytest.raises(ValueError):
        replace(combined, stage_errors={k: v for k,v in all_errors.items() if k != stage}).validate()


@pytest.mark.parametrize("name", ["00029A_2_3_R_20190322_DATA.txt", "00080B_1_2_R_20190724_DATA.txt"])
def test_real_upload_execution_error(monkeypatch, isolated_runs, name):
    from io import BytesIO
    import streamlit as st
    monkeypatch.setattr(controller, "run_cnn", unavailable_cnn)
    root = Path(__file__).resolve().parents[2]
    upload = BytesIO((root / "data/sdlcd_raw" / name).read_bytes())
    upload.name = name
    monkeypatch.setattr(st, "file_uploader", lambda *a, **kw: upload)
    monkeypatch.setattr(controller, "reconcile", failure)
    app = AppTest.from_file(str(root / "app/main.py")).run(timeout=60)
    app.button[0].click().run(timeout=60)
    assert not app.exception and not app.error
    assert any(h.value == "Technical execution failure" for h in app.subheader)
    assert all("injected technical failure" not in c.value for c in app.caption)
    assert any("Observation quality:" in m.value for m in app.markdown)
    assert any("Reconciliation: the stage could not finish." in c.value for c in app.caption)
    tables = [t.value for t in app.table if {"Estimator", "Apparent (synodic) period (s)"} <= set(t.value.columns)]
    assert {t["Estimator"].iloc[0] for t in tables} == {"lomb_scargle", "pdm"}


def test_verdict_error_dashboard(monkeypatch, injected_config, mock_cnn, mock_recon):
    from core import config
    monkeypatch.setattr(config, "load_config", lambda: injected_config)
    monkeypatch.setattr(controller, "run_cnn", lambda *a: mock_cnn)
    monkeypatch.setattr(controller, "reconcile", lambda *a: mock_recon)
    monkeypatch.setattr(controller, "verify_candidates_original", controller.verify_candidates, raising=False)
    monkeypatch.setattr(controller, "verify_candidates", completed)
    monkeypatch.setattr(controller, "decide", failure)
    _fixture_upload(monkeypatch)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[2] / "app/main.py")).run(timeout=60)
    app.button[0].click().run(timeout=60)
    assert not app.exception and not app.error
    assert any(h.value == "Technical execution failure" for h in app.subheader)
    assert all("injected technical failure" not in c.value for c in app.caption)
    assert any("Verdict: the stage could not finish." in c.value for c in app.caption)
