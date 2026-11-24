from dataclasses import replace
from io import BytesIO
from pathlib import Path
import json

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from core.config import load_config
from core.schemas import (
    ALL_CANDIDATES_FAILED_STAGE2, ALL_ESTIMATORS_UNAVAILABLE, CANDIDATE_INADEQUATE,
    NO_PERIODIC_SIGNAL, SCHEMA_VERSION, STAGE2_FAIL_REASONS, VERIFICATION_INCOMPLETE,
    AnalysisResult, QualityReport, assert_contract_equal,
)
from ingestion import load_observation
from pipeline import controller
from verdict.decide import decide
from tests.p2.test_controller import pending_reconciliation, unavailable_cnn


def test_pending_validation_and_serialization(observation, injected_config, mock_cnn, monkeypatch):
    monkeypatch.setattr(controller, "run_cnn", lambda *args: mock_cnn)
    monkeypatch.setattr(controller, "reconcile", pending_reconciliation)
    result = controller.analyze(observation, injected_config)
    assert result.verdict is None and result.verification is None
    result.validate()
    assert_contract_equal(result, AnalysisResult.from_dict(json.loads(json.dumps(result.to_dict()))))
    quality = QualityReport("suitable", {}, [], [])
    with pytest.raises(ValueError, match="pending"):
        decide(quality, result.estimators, result.reconciliation, None, {})
    final = decide(quality, {}, None, None, {})
    recon = result.reconciliation
    rejected = replace(recon, is_stub=False, notes=[], clusters=[
        replace(c, stage2={"passed": False, "cycles": 1., "fail_reasons": [STAGE2_FAIL_REASONS[0]]})
        for c in recon.clusters])
    normal = replace(result, reconciliation=rejected,
                     verdict=decide(quality, result.estimators, rejected, None, {}))
    normal.validate()
    with pytest.raises(ValueError):
        replace(normal, verdict=None).validate()
    invalid = [
        replace(result, verdict=final),
        replace(result, reconciliation=None),
        replace(result, reconciliation=replace(recon, is_stub=False)),
        replace(result, reconciliation=replace(recon, notes=[])),
        replace(result, reconciliation=replace(recon, shortlist=[recon.clusters[0].cluster_id],
                                                stage2_passed_any=True)),
        replace(result, reconciliation=replace(recon, clusters=[])),
        replace(result, estimators={k: replace(v, candidates=[]) for k,v in result.estimators.items()}),
    ]
    verification = controller.verify_candidates(result.prepared,
        replace(recon, shortlist=[recon.clusters[0].cluster_id], stage2_passed_any=True), {})
    invalid.append(replace(result, verification=verification))
    for item in invalid:
        with pytest.raises(ValueError):
            item.validate()
    payload = result.to_dict()
    payload["schema_version"] = SCHEMA_VERSION - 1
    with pytest.raises(ValueError, match="schema version"):
        AnalysisResult.from_dict(payload)


def test_existing_verdict_outcomes(mock_recon, mock_cnn):
    quality = QualityReport("suitable", {}, [], [])
    estimators = {"cnn": mock_cnn}
    rejected = replace(mock_recon, shortlist=[], stage2_passed_any=False,
        clusters=[replace(c, stage2={"passed": False, "cycles": 1.,
                         "fail_reasons": [STAGE2_FAIL_REASONS[0]]}) for c in mock_recon.clusters])
    result = decide(quality, estimators, rejected, None, {})
    assert result.reason_codes == [CANDIDATE_INADEQUATE]
    assert result.evidence_summary["inconclusive_cause"] == ALL_CANDIDATES_FAILED_STAGE2
    result = decide(quality, estimators, mock_recon, None, {})
    assert result.label == "Ambiguous" and result.reason_codes == [VERIFICATION_INCOMPLETE]
    assert decide(quality, {"cnn": replace(mock_cnn, candidates=[])}, None, None, {}).reason_codes == [NO_PERIODIC_SIGNAL]
    result = decide(quality, {}, None, None, {})
    assert result.evidence_summary["inconclusive_cause"] == ALL_ESTIMATORS_UNAVAILABLE


@pytest.mark.parametrize("name", ["00029A_2_3_R_20190322_DATA.txt",
                                "00080B_1_2_R_20190724_DATA.txt"])
def test_real_upload_pending(name, isolated_runs, monkeypatch):
    monkeypatch.setattr(controller, "reconcile", pending_reconciliation)
    monkeypatch.setattr(controller, "run_cnn", unavailable_cnn)
    root = Path(__file__).resolve().parents[2]
    path = root / "data/sdlcd_raw" / name
    upload = BytesIO(path.read_bytes())
    upload.name = name
    monkeypatch.setattr(st, "file_uploader", lambda *a, **k: upload)
    app = AppTest.from_file(str(root / "app/main.py")).run(timeout=60)
    app.button[0].click().run(timeout=60)
    assert not app.exception and not app.error
    assert any(x.value == "Stage 2 assessment pending" for x in app.subheader)
    assert any("Observation quality:" in x.value for x in app.markdown)
    assert any("have not yet been evaluated" in x.value for x in app.markdown)
    assert not any("no scientific estimate" in x.value for x in app.markdown)
    tables = [x.value for x in app.table if {"Estimator", "Apparent (synodic) period (s)"} <= set(x.value.columns)]
    assert {t["Estimator"].iloc[0] for t in tables} == {"lomb_scargle", "pdm"}
    result = controller.analyze(load_observation(str(path)), load_config())
    assert result.verdict is None
    for method in ("lomb_scargle", "pdm"):
        assert not result.estimators[method].is_stub and result.estimators[method].candidates
