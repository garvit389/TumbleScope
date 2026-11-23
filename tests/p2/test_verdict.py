from dataclasses import replace

import pytest

from core.schemas import (
    ALIAS_AMBIGUITY, CANDIDATE_INADEQUATE, CNN, CONFLICTING_METHODS, HARMONIC_AMBIGUITY,
    LOMB_SCARGLE, LOW_QUALITY, MODEL_MISMATCH, NO_CHECKPOINT, NO_PERIODIC_SIGNAL,
    PDM, VERIFICATION_INCOMPLETE, Candidate, CandidateVerification, EstimatorResult, HarmonicComparison,
    QualityReport, VerificationResult,
)
from verdict.decide import decide


@pytest.fixture
def case(mock_cnn, mock_recon):
    def classical(method, period):
        return EstimatorResult(method, "ok", [Candidate(period, 1., .9, {"freq_width_hz": .01})],
                               meta={"runtime_s": 0., "periodogram": {"freq_hz": [], "power": []},
                                     "freq_resolution": .01})

    lead, other = mock_recon.clusters
    lead = replace(lead, alias_risk=.2, harmonic_links=[],
                   members={CNN: 8., LOMB_SCARGLE: 8., PDM: 8.})
    other = replace(other, harmonic_links=[], members={})
    recon = replace(mock_recon, clusters=[lead, other], shortlist=[1, 2])
    fits = [CandidateVerification(
        cluster.cluster_id, cluster.period_s, "ok", .4, .5, [12.], [.2],
        cluster.cluster_id == 1,
        {"shape": "box", "geometry_errors": {"box": .5}, "fit_space": "y_norm",
         "scale_a": 1., "offset_b": 0.}, [],
    ) for cluster in recon.clusters]
    estimators = {CNN: mock_cnn, LOMB_SCARGLE: classical(LOMB_SCARGLE, 8.),
                  PDM: classical(PDM, 8.)}
    return recon, estimators, VerificationResult(fits, True, "fast", 1.)


@pytest.mark.parametrize(("scenario", "label", "reasons"), [
    ("supported", "Supported", []),
    ("no_cnn", "Supported", []),
    ("limited", "Supported", []),
    ("limited_cap", "Ambiguous", [LOW_QUALITY]),
    ("alias", "Ambiguous", [ALIAS_AMBIGUITY]),
    ("conflict", "Ambiguous", [CONFLICTING_METHODS]),
    ("few_methods", "Ambiguous", [CANDIDATE_INADEQUATE]),
    ("harmonic", "Ambiguous", [HARMONIC_AMBIGUITY]),
    ("adverse harmonic", "Ambiguous", [HARMONIC_AMBIGUITY]),
    ("missing_verification", "Ambiguous", [VERIFICATION_INCOMPLETE]),
    ("incomplete", "Ambiguous", [VERIFICATION_INCOMPLETE]),
    ("mismatch", "Ambiguous", [MODEL_MISMATCH]),
    ("abstention_only", "Inconclusive", [NO_PERIODIC_SIGNAL]),
])
def test_verdict_decision_table(case, scenario, label, reasons):
    recon, estimators, verification = case
    lead, other = recon.clusters
    record, second = verification.per_candidate
    quality = QualityReport("suitable", {}, [], [])
    cfg = {"alias_risk_max": .35}
    if scenario in ("no_cnn", "few_methods"):
        estimators[CNN] = EstimatorResult(CNN, "skipped", [], NO_CHECKPOINT, {"runtime_s": 0.})
    if scenario == "few_methods":
        estimators.pop(PDM)
        lead = replace(lead, members={LOMB_SCARGLE: 8.})
    if scenario in ("limited", "limited_cap"):
        quality = QualityReport("limited", {}, ["Limited coverage"], [])
        if scenario == "limited_cap":
            cfg["limited_max_label"] = "Ambiguous"
    if scenario == "alias":
        lead = replace(lead, alias_risk=.4)
    if scenario == "conflict":
        estimators[PDM] = replace(estimators[PDM], candidates=[
            replace(estimators[PDM].candidates[0], period_s=16.)])
        other = replace(other, members={PDM: 16.})
    if scenario in ("harmonic", "adverse harmonic"):
        lead = replace(lead, harmonic_links=[{"other_id": 2, "ratio": 2.}])
        if scenario == "adverse harmonic":
            second = replace(second, heldout_error=.4)
        verification = replace(verification, harmonic_comparisons=[HarmonicComparison(
            "1:2", 1, 2, 8., 16., 2., "ok", "undecided" if scenario == "harmonic" else "2P",
            record.heldout_error, second.heldout_error, second.heldout_error - record.heldout_error,
            .05, None, [])])
    if scenario == "mismatch":
        record = replace(record, passed=False)
    if scenario == "incomplete":
        second = replace(second, status="failed", fit_error=None, heldout_error=None, best_params=None)
        verification = replace(verification, complete=False)
    if scenario == "abstention_only":
        estimators = {CNN: replace(case[1][CNN], status="ok", candidates=[], error=None)}
    recon = replace(recon, clusters=[lead, other])
    verification = None if scenario in ("missing_verification", "abstention_only") else replace(
        verification, per_candidate=[record, second])
    result = decide(quality, estimators, recon, verification, cfg)
    result.validate()
    assert (result.label, result.reason_codes) == (label, reasons)
    assert (result.best_cluster_id, result.best_period_s) == ((1, 8.) if label == "Supported" else (None, None))
    assert [item["cluster_id"] for item in result.alternatives] == ([1, 2] if label == "Ambiguous" else [])
    assert not result.is_stub and result.evidence_summary["selective_accuracy_validation"] is None


def test_selected_record_and_alternatives(case):
    recon, estimators, verification = case
    first, second = verification.per_candidate
    other = replace(second, heldout_error=.1, decoy_delta_e=[1.])
    fits = [replace(first, passed=False), other]
    for sequence in (fits, fits[::-1]):
        result = decide(QualityReport("suitable", {}, [], []), estimators, recon,
                        replace(verification, per_candidate=sequence), {})
        assert result.reason_codes == [MODEL_MISMATCH]
        assert result.evidence_summary["heldout_error"] == .5
        assert result.evidence_summary["decoy_separation"] == .2
        assert [opt["cluster_id"] for opt in result.alternatives] == [1, 2]


@pytest.mark.parametrize("boundary", ["missing policy", "alias equality",
                                      "top-vote mapping", "geometry margin"])
def test_missing_policy_execution_boundary(case, boundary):
    recon, estimators, verification = case
    lead, other = recon.clusters
    record, second = verification.per_candidate
    cfg = {"alias_risk_max": .35}
    if boundary == "missing policy":
        cfg = {}
    if boundary == "alias equality":
        lead = replace(lead, alias_risk=.35)
    if boundary == "top-vote mapping":
        lead = replace(lead, members={CNN: 8., LOMB_SCARGLE: 7., PDM: 7.})
    if boundary == "geometry margin":
        record = replace(record, best_params={**record.best_params,
                         "geometry_errors": {"box": .5, "cylinder": .51}})
    with pytest.raises(ValueError, match="alias_risk_max|contract amendment"):
        decide(QualityReport("suitable", {}, [], []), estimators,
               replace(recon, clusters=[lead, other]),
               replace(verification, per_candidate=[record, second]), cfg)
