"""Scientific verdicts for the contract decision table."""

from itertools import combinations
from math import isfinite
from typing import Optional

from core.schemas import (
    ALIAS_AMBIGUITY, ALL_CANDIDATES_FAILED_STAGE2, ALL_ESTIMATORS_UNAVAILABLE,
    CANDIDATE_INADEQUATE, CONFLICTING_METHODS, GEOMETRY_UNCERTAINTY, HARMONIC_AMBIGUITY, LOW_QUALITY,
    MODEL_MISMATCH, NO_PERIODIC_SIGNAL, REASON_CODES, VERIFICATION_INCOMPLETE,
    CandidateVerification, EstimatorResult, QualityReport, ReconciliationResult, VerificationResult, Verdict,
    check_verification_scope, harmonic_resolved_for, stage2_pending,
)
from verdict.reasons import _alternatives


def _select_record(recon: ReconciliationResult,
                   verification: VerificationResult | None) -> CandidateVerification | None:
    scores = {cluster.cluster_id: cluster.score for cluster in recon.clusters}
    finished = (record for record in verification.per_candidate if record.status == "ok") if verification else ()
    return min(finished, default=None, key=lambda record: (
        not record.passed, -scores[record.cluster_id], record.cluster_id,
    ))


def _passing_decision(estimators, clusters, lead, record, verification, cfg, summary):
    minimum = cfg.get("min_agreeing_estimators", 2)
    maximum = cfg.get("alias_risk_max")
    if type(minimum) is not int or not 2 <= minimum <= 3:
        raise ValueError("invalid min_agreeing_estimators")
    if type(maximum) not in (int, float) or not isfinite(maximum) or not 0 <= maximum <= 1:
        raise ValueError("missing or invalid alias_risk_max")
    if not lead.stage2["passed"]:
        raise ValueError("contract: passing verification record failed Stage 2")

    methods = {method: item for method, item in estimators.items() if item.status == "ok" and item.candidates}
    top_clusters = []
    for method, item in methods.items():
        top = item.candidates[0].period_s
        matches = [cluster for cluster in clusters if cluster.members.get(method) == top]
        if len(matches) == 1:
            top_clusters.extend(matches)
    n_votes = sum(cluster.cluster_id == lead.cluster_id for cluster in top_clusters)
    summary.update(estimators_agree=n_votes >= minimum, alias_risk=lead.alias_risk)
    reasons = []
    if n_votes < minimum:
        reasons.append(CANDIDATE_INADEQUATE)
    for left, right in combinations(top_clusters, 2):
        if left.cluster_id == right.cluster_id:
            continue
        linked = any(link["other_id"] == right.cluster_id for link in left.harmonic_links)
        linked |= any(link["other_id"] == left.cluster_id for link in right.harmonic_links)
        if not linked:
            reasons.append(CONFLICTING_METHODS)
            break

    if lead.alias_risk >= maximum:
        reasons.append(ALIAS_AMBIGUITY)

    if not harmonic_resolved_for(verification, lead):
        reasons.append(HARMONIC_AMBIGUITY)

    errors = sorted(record.best_params["geometry_errors"].values())
    if len(errors) >= 2:
        margin = cfg.get("geometry_margin")
        if type(margin) in (int, float) and isfinite(margin) and margin >= 0:
            if errors[1] - errors[0] < margin:
                reasons.append(GEOMETRY_UNCERTAINTY)
        elif HARMONIC_AMBIGUITY not in reasons:
            raise ValueError("verdict requires the contract amendment for missing or invalid geometry margin")
    return reasons


def decide(quality: QualityReport, estimators: dict[str, EstimatorResult],
           recon: Optional[ReconciliationResult], verification: Optional[VerificationResult],
           cfg: dict) -> Verdict:
    if recon is not None and recon.shortlist and verification is not None:
        try:
            verification.validate()
            check_verification_scope(recon, verification)
        except ValueError:
            verification = None
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
    best = None
    if quality.status == "insufficient":
        reasons = [LOW_QUALITY]
    elif not available:
        reasons = [CANDIDATE_INADEQUATE]
        summary["inconclusive_cause"] = ALL_ESTIMATORS_UNAVAILABLE
    elif not any(result.candidates for result in estimators.values()):
        reasons = [NO_PERIODIC_SIGNAL]
    elif recon is None or not recon.shortlist:
        if stage2_pending(recon, estimators, verification):
            raise ValueError("contract: final verdict requested for pending Stage 2 stub")
        if not clusters or not all(not cluster.stage2["passed"] for cluster in clusters):
            raise ValueError("contract: no scientific verdict without evaluated Stage 2 rejection")
        reasons = [CANDIDATE_INADEQUATE]
        summary["inconclusive_cause"] = ALL_CANDIDATES_FAILED_STAGE2
    else:
        label = "Ambiguous"
        record = _select_record(recon, verification)
        if record is not None:
            summary["heldout_error"] = record.heldout_error
            summary["decoy_separation"] = min(record.decoy_delta_e, default=None)
        if verification is None or not verification.complete:
            reasons = [VERIFICATION_INCOMPLETE]
        elif record is None or not record.passed:
            reasons = [MODEL_MISMATCH]
        else:
            lead = next(cluster for cluster in clusters if cluster.cluster_id == record.cluster_id)
            reasons = _passing_decision(estimators, clusters, lead, record, verification, cfg, summary)
            cap = cfg.get("limited_max_label", "Supported")
            if cap not in ("Supported", "Ambiguous"):
                raise ValueError("invalid limited_max_label")
            if not reasons:
                if quality.status == "limited" and cap == "Ambiguous":
                    summary["limited_cap_applied"] = True
                else:
                    label, best = "Supported", lead
    if quality.status == "limited" and label != "Supported":
        reasons.append(LOW_QUALITY)
    text = f"{label}."
    if label != "Inconclusive":
        text += " Physics checks use provisional criteria for compatibility with a simplified model."
    if quality.warnings:
        text += " " + " ".join(quality.warnings)
    return Verdict(label, best.period_s if best is not None else None,
                   best.cluster_id if best is not None else None,
                   _alternatives(recon, verification) if label == "Ambiguous" else [],
                   [reason for reason in REASON_CODES if reason in reasons], summary, text)
