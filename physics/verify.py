"""Phase 0 verifier placeholder."""

from typing import Optional

from core.schemas import (
    NOT_RUN, CandidateVerification, PreparedObservation, ProgressCallback, ReconciliationResult,
    VerificationResult, report_progress, required_verification_ids,
)


def verify_candidates(prep: PreparedObservation, recon: ReconciliationResult, cfg: dict,
                      mode: str = "fast",
                      progress: Optional[ProgressCallback] = None) -> VerificationResult:
    if mode != "fast":
        raise NotImplementedError("Only Fast mode is available.")
    report_progress(progress, 0.0, "Starting verification placeholder")
    by_id = {cluster.cluster_id: cluster for cluster in recon.clusters}
    records = []
    required = required_verification_ids(recon)
    for cluster in recon.clusters:
        if cluster.cluster_id not in required:
            continue
        comparison = None
        for link in cluster.harmonic_links:
            if link["ratio"] in (0.5, 2):
                partner_id = link["other_id"]
                reason = None if by_id[partner_id].stage2["passed"] else "partner_failed_stage2"
                comparison = {"partner_id": partner_id, "prefers": "undecided",
                              "delta_e": None, "reason": reason}
                break
        records.append(CandidateVerification(
            cluster.cluster_id, cluster.period_s, "skipped", None, None, [], [],
            comparison, False, None, [NOT_RUN],
        ))
    report_progress(progress, 1.0, "Verification placeholder complete")
    return VerificationResult(records, False, "fast", 0.0, is_stub=True)
