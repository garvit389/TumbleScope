from dataclasses import replace
from unittest.mock import Mock

import pytest

from core.schemas import LOW_QUALITY, VERIFICATION_INCOMPLETE
from pipeline import controller


@pytest.mark.parametrize("shortlist_size", [3, 4])
def test_fast_shortlist_limit_and_harmonic_partner(
    monkeypatch, observation, injected_config, mock_cnn, mock_recon, shortlist_size,
):
    lead, harmonic = mock_recon.clusters
    extra = [
        replace(lead, cluster_id=3, period_s=7.0, score=0.9, harmonic_links=[]),
        replace(lead, cluster_id=4, period_s=9.0, score=0.8, harmonic_links=[]),
    ]
    clusters = [lead, *extra, harmonic]
    shortlist = [cluster.cluster_id for cluster in clusters[:shortlist_size]]
    output = replace(mock_recon, clusters=clusters, shortlist=shortlist)
    output.validate()
    reconcile = Mock(return_value=output)
    verify = Mock(wraps=controller.verify_candidates)
    monkeypatch.setattr(controller, "run_cnn", Mock(return_value=mock_cnn))
    monkeypatch.setattr(controller, "reconcile", reconcile)
    monkeypatch.setattr(controller, "verify_candidates", verify)
    config = replace(injected_config, reconcile={"top_k": 5})

    result = controller.analyze(observation, config)
    result.validate()
    assert reconcile.call_args.args[2]["top_k"] == 3
    assert config.reconcile == {"top_k": 5}
    if shortlist_size == 4:
        assert result.stage_errors["reconciliation"].startswith("contract_violation:")
        assert result.reconciliation is result.verification is None
        verify.assert_not_called()
    else:
        assert len(result.reconciliation.shortlist) == 3
        assert {record.cluster_id for record in result.verification.per_candidate} == {1, 2, 3, 4}
        verify.assert_called_once()


@pytest.mark.parametrize("validate_outputs", [True, False])
@pytest.mark.parametrize("stage", ["prepare", "stage1", "cnn", "reconciliation", "verification", "verdict"])
def test_invalid_stage_output_yields_valid_partial_analysis(
    monkeypatch, observation, prepared, injected_config, mock_cnn, mock_recon, validate_outputs, stage,
):
    config = replace(injected_config, pipeline={"validate_outputs": validate_outputs})
    monkeypatch.setattr(controller, "run_cnn", Mock(return_value=mock_cnn))
    monkeypatch.setattr(controller, "reconcile", Mock(return_value=mock_recon))
    good = controller.analyze(observation, config)
    record, partner = good.verification.per_candidate
    invalid_outputs = {
        "prepare": ("prepare", replace(prepared, meta={})),
        "stage1": ("assess_stage1", replace(good.quality, status="invalid")),
        "cnn": ("run_cnn", replace(mock_cnn, meta={})),
        "reconciliation": ("reconcile", replace(mock_recon, shortlist=[99])),
        "verification": ("verify_candidates", replace(
            good.verification, per_candidate=[replace(record, passed=True), partner],
        )),
        "verdict": ("decide", replace(good.verdict, evidence_summary={})),
    }
    function, invalid = invalid_outputs[stage]
    monkeypatch.setattr(controller, function, Mock(return_value=invalid))

    result = controller.analyze(observation, config)
    result.validate()
    assert result.stage_errors[stage].startswith("contract_violation:")
    if stage == "prepare":
        assert result.prepared is result.quality is None
    elif stage == "stage1":
        assert result.quality.status == "insufficient"
        assert result.verdict.reason_codes == [LOW_QUALITY]
    elif stage == "cnn":
        assert result.estimators[mock_cnn.method].status == "failed"
    elif stage == "reconciliation":
        assert result.reconciliation is result.verification is None
    elif stage == "verification":
        assert result.verification is None
        assert VERIFICATION_INCOMPLETE in result.verdict.reason_codes
    else:
        assert result.verification is not None
        assert VERIFICATION_INCOMPLETE in result.verdict.reason_codes


@pytest.mark.parametrize("validate_outputs", [True, False])
def test_controller_rejects_unsupported_verdict_and_keeps_partial_verification(
    monkeypatch, observation, injected_config, mock_cnn, mock_recon, validate_outputs,
):
    config = replace(injected_config, pipeline={"validate_outputs": validate_outputs})
    monkeypatch.setattr(controller, "run_cnn", Mock(return_value=mock_cnn))
    monkeypatch.setattr(controller, "reconcile", Mock(return_value=mock_recon))
    good = controller.analyze(observation, config)
    adverse = replace(good.verdict, label="Supported", reason_codes=[],
                      best_period_s=8.0, best_cluster_id=1)
    adverse.validate()
    monkeypatch.setattr(controller, "decide", Mock(return_value=adverse))
    result = controller.analyze(observation, config)
    result.validate()
    assert result.stage_errors["verdict"].startswith("contract_violation:")
    assert result.verification is not None and not result.verification.complete
    assert result.verdict.label == "Ambiguous"
    assert VERIFICATION_INCOMPLETE in result.verdict.reason_codes
