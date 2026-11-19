"""Phase 0 verdict placeholder, including the structural outcomes in §5."""

from typing import Optional

from core.schemas import (
    ALL_CANDIDATES_FAILED_STAGE2, ALL_ESTIMATORS_UNAVAILABLE, CANDIDATE_INADEQUATE,
    LOW_QUALITY, MODEL_MISMATCH, NO_PERIODIC_SIGNAL, VERIFICATION_INCOMPLETE,
    EstimatorResult, QualityReport, ReconciliationResult, VerificationResult, Verdict,
)


def decide(quality: QualityReport, estimators: dict[str, EstimatorResult],
           recon: Optional[ReconciliationResult], verification: Optional[VerificationResult],
           cfg: dict) -> Verdict:
    statuses = {method: result.status for method, result in estimators.items()}
    available = sum(status == "ok" for status in statuses.values())
    clusters = recon.clusters if recon is not None else []
    stage2_failures = sorted({
        reason for cluster in clusters for reason in cluster.stage2["fail_reasons"]
    })
    summary = {
        "stage1_status": quality.status,
        "stage1_warnings": list(quality.warnings),
        "estimator_status": statuses,
        "n_estimators_available": available,
        "estimators_agree": None,
        "alias_risk": None,
        "verification_complete": verification.complete if verification is not None else None,
        "heldout_error": None,
        "decoy_separation": None,
        "stage2_fail_reasons": stage2_failures,
        "selective_accuracy_validation": None,
    }
    label = "Inconclusive"
    if quality.status == "insufficient":
        reasons = [LOW_QUALITY]
    elif not available:
        reasons = [CANDIDATE_INADEQUATE]
        summary["inconclusive_cause"] = ALL_ESTIMATORS_UNAVAILABLE
    elif not any(result.candidates for result in estimators.values()):
        reasons = [NO_PERIODIC_SIGNAL]
    elif recon is None or not recon.shortlist:
        reasons = [CANDIDATE_INADEQUATE]
        summary["inconclusive_cause"] = ALL_CANDIDATES_FAILED_STAGE2
    else:
        label = "Ambiguous"
        if verification is None or not verification.complete:
            reasons = [VERIFICATION_INCOMPLETE]
        elif not any(record.passed for record in verification.per_candidate):
            reasons = [MODEL_MISMATCH]
        else:
            raise NotImplementedError("Calibrated verdicts are outside the Phase 0 placeholder.")

        if verification is not None:
            scores = {cluster.cluster_id: cluster.score for cluster in clusters}
            finished = [record for record in verification.per_candidate if record.status == "ok"]
            if finished:
                record = min(finished, key=lambda item: (
                    not item.passed, -scores[item.cluster_id], item.cluster_id
                ))
                summary["heldout_error"] = record.heldout_error
                summary["decoy_separation"] = min(record.decoy_delta_e, default=None)
    if quality.status == "limited":
        reasons.append(LOW_QUALITY)
    return Verdict(label, None, None, [], reasons, summary,
                   f"{label}. Phase 0 placeholder; no scientific estimate was made.", is_stub=True)
