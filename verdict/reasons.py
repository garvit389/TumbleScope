"""Descriptions of the verification evidence available for alternatives."""

from core.schemas import harmonic_resolved_for, required_verification_ids


def _alternatives(recon, verification):
    if recon is None:
        return []
    ids = required_verification_ids(recon)
    records = {record.cluster_id: record for record in verification.per_candidate} if verification else {}
    options = []
    for cluster in sorted(recon.clusters, key=lambda item: (-item.score, item.cluster_id)):
        if cluster.cluster_id not in ids:
            continue
        record = records.get(cluster.cluster_id)
        if record is None or record.status != "ok":
            why = "Verification is unfinished."
        elif not record.passed:
            why = "The fit did not meet all provisional verification criteria."
        elif not harmonic_resolved_for(verification, cluster):
            why = "The harmonic evidence does not establish this candidate over its partners."
        else:
            why = "The candidate met both provisional verification criteria."
        options.append({"cluster_id": cluster.cluster_id, "period_s": cluster.period_s, "why": why})
    return options
