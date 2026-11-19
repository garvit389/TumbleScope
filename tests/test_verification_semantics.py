from dataclasses import replace
from itertools import product
from random import Random

import pytest

from core.schemas import (
    MODEL_MISMATCH, STAGE2_FAIL_REASONS, VERIFICATION_INCOMPLETE, check_verification_scope,
)
from physics.verify import verify_candidates
from pipeline import controller
from verdict.decide import decide


def finished(record, *, passed=False, heldout=0.5, decoy_error=0.75):
    return replace(record, status="ok", passed=passed, fit_error=0.4, heldout_error=heldout,
                   decoy_periods_s=[13.0], decoy_delta_e=[decoy_error - heldout],
                   best_params={"shape": "fixture", "geometry_errors": {"fixture": heldout},
                                "fit_space": "y_norm", "scale_a": 1.0, "offset_b": 0.0})


@pytest.mark.parametrize("status,passed", list(product(("ok", "skipped", "failed"), (True, False))))
def test_allowed_status_and_pass_combinations(prepared, mock_recon, status, passed):
    record = verify_candidates(prepared, mock_recon, {}).per_candidate[0]
    if status == "ok":
        finished(record, passed=passed).validate()
    elif passed:
        with pytest.raises(ValueError, match="contract: invalid unfinished verification"):
            replace(record, status=status, passed=True).validate()
    else:
        replace(record, status=status).validate()


def test_verifier_returns_each_required_partner(prepared, mock_recon):
    outcome = verify_candidates(prepared, mock_recon, {})
    assert mock_recon.shortlist == [1]
    assert [record.cluster_id for record in outcome.per_candidate] == [1, 2]
    assert not outcome.complete
    check_verification_scope(mock_recon, outcome)


def test_required_verification_ignores_failed_and_transitive_partners(prepared, mock_recon):
    lead, partner = mock_recon.clusters
    transitive = replace(partner, cluster_id=3, period_s=32.0, score=0.5, harmonic_links=[])
    linked = replace(partner, harmonic_links=[*partner.harmonic_links, {"other_id": 3, "ratio": 2.0}])
    extended = replace(mock_recon, clusters=[lead, linked, transitive])
    extended.validate()
    outcome = verify_candidates(prepared, extended, {})
    check_verification_scope(extended, outcome)
    assert {record.cluster_id for record in outcome.per_candidate} == {1, 2}

    rejected = replace(partner, stage2={**partner.stage2, "passed": False,
                                       "fail_reasons": [STAGE2_FAIL_REASONS[0]]})
    filtered = replace(mock_recon, clusters=[lead, rejected])
    filtered.validate()
    outcome = verify_candidates(prepared, filtered, {})
    check_verification_scope(filtered, outcome)
    assert [record.cluster_id for record in outcome.per_candidate] == [1]
    assert outcome.per_candidate[0].harmonic_comparison["reason"] == "partner_failed_stage2"


@pytest.fixture
def configured_pipeline(monkeypatch, injected_config, mock_cnn, mock_recon):
    monkeypatch.setattr(controller, "run_cnn", lambda *args: mock_cnn)
    monkeypatch.setattr(controller, "reconcile", lambda *args: mock_recon)
    return injected_config


@pytest.mark.parametrize("problem", ["missing", "extra", "duplicate", "incorrect completeness"])
def test_controller_rejects_invalid_verification(
    monkeypatch, observation, prepared, mock_recon, configured_pipeline, problem
):
    valid_stub = verify_candidates(prepared, mock_recon, {})
    first, second = valid_stub.per_candidate
    options = {
        "missing": replace(valid_stub, per_candidate=[first]),
        "extra": replace(valid_stub, per_candidate=[first, second, replace(first, cluster_id=99)]),
        "duplicate": replace(valid_stub, per_candidate=[first, first]),
        "incorrect completeness": replace(valid_stub, per_candidate=[finished(first), finished(second)]),
    }
    monkeypatch.setattr(controller, "verify_candidates", lambda *args, **kwargs: options[problem])
    result = controller.analyze(observation, configured_pipeline)
    assert result.verification is None
    assert result.stage_errors["verification"].startswith("contract_violation:")
    assert result.verdict.label == "Ambiguous"
    assert VERIFICATION_INCOMPLETE in result.verdict.reason_codes
    with pytest.raises(ValueError, match="^contract:"):
        replace(result, verification=options[problem]).validate()


def test_completed_failure_uses_one_diagnostic_record(
    monkeypatch, observation, prepared, mock_recon, configured_pipeline
):
    outcome = verify_candidates(prepared, mock_recon, {})
    first, second = outcome.per_candidate
    outcome = replace(outcome, complete=True, per_candidate=[
        finished(second, heldout=0.25, decoy_error=0.875),
        finished(first, heldout=0.50, decoy_error=0.75),
    ])
    monkeypatch.setattr(controller, "verify_candidates", lambda *args, **kwargs: outcome)
    result = controller.analyze(observation, configured_pipeline)
    assert result.verification.complete
    assert result.verdict.label == "Ambiguous"
    assert result.verdict.reason_codes == [MODEL_MISMATCH]
    assert result.verdict.evidence_summary["heldout_error"] == 0.5
    assert result.verdict.evidence_summary["decoy_separation"] == 0.25


@pytest.mark.parametrize("leading_passed,other_status", list(product((True, False), ("failed", "skipped"))))
def test_one_unfinished_partner_prevents_supported(
    monkeypatch, observation, prepared, mock_recon, configured_pipeline, leading_passed, other_status
):
    outcome = verify_candidates(prepared, mock_recon, {})
    first, second = outcome.per_candidate
    outcome = replace(outcome, per_candidate=[
        finished(first, passed=leading_passed), replace(second, status=other_status),
    ])
    monkeypatch.setattr(controller, "verify_candidates", lambda *args, **kwargs: outcome)
    result = controller.analyze(observation, configured_pipeline)
    assert not result.verification.complete
    assert result.verdict.label == "Ambiguous"
    assert VERIFICATION_INCOMPLETE in result.verdict.reason_codes


def test_random_incomplete_verification_cannot_support_a_candidate(
    observation, prepared, mock_recon, configured_pipeline,
):
    base = controller.analyze(observation, configured_pipeline)
    stub = verify_candidates(prepared, mock_recon, {})
    rng = Random(732)
    for _ in range(80):
        unfinished = rng.choice(stub.per_candidate).cluster_id
        records = [
            replace(record, status=rng.choice(("failed", "skipped")))
            if record.cluster_id == unfinished or rng.choice((True, False))
            else finished(record, passed=rng.choice((True, False)),
                          heldout=rng.random(), decoy_error=rng.random())
            for record in stub.per_candidate
        ]
        rng.shuffle(records)
        outcome = replace(stub, per_candidate=records, complete=False)
        outcome.validate()
        check_verification_scope(mock_recon, outcome)
        verdict = decide(base.quality, base.estimators, mock_recon, outcome, {})
        assert verdict.label == "Ambiguous"
        assert VERIFICATION_INCOMPLETE in verdict.reason_codes
        replace(base, verification=outcome, verdict=verdict).validate()
        fabricated = replace(verdict, label="Supported", reason_codes=[],
                             best_cluster_id=records[0].cluster_id,
                             best_period_s=records[0].period_s)
        with pytest.raises(ValueError, match="Supported requires complete verification"):
            replace(base, verification=outcome, verdict=fabricated).validate()


def test_decoy_delta_sign_and_worst_decoy_are_preserved(
    monkeypatch, observation, prepared, mock_recon, configured_pipeline,
):
    stub = verify_candidates(prepared, mock_recon, {})
    lead, partner = stub.per_candidate
    # Candidate error 0.50, decoy errors 0.75 and 0.25: deltas +0.25 and -0.25.
    diagnostic = replace(finished(lead, heldout=0.50, decoy_error=0.75),
                         decoy_periods_s=[13.0, 19.0], decoy_delta_e=[0.25, -0.25])
    outcome = replace(stub, per_candidate=[diagnostic, partner])
    monkeypatch.setattr(controller, "verify_candidates", lambda *args, **kwargs: outcome)
    result = controller.analyze(observation, configured_pipeline)
    assert result.verification.per_candidate[0].decoy_delta_e == [0.25, -0.25]
    assert result.verdict.evidence_summary["decoy_separation"] == -0.25
