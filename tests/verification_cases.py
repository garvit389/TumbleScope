"""Construct consistent comparison evidence for tests that replace candidate fits."""

from dataclasses import replace

from core.schemas import harmonic_preference, verification_finished


def with_candidate_fits(result, fits, margin=.05):
    by_id = {fit.cluster_id: fit for fit in fits}
    pairs = []
    for pair in result.harmonic_comparisons:
        short = by_id.get(pair.short_cluster_id)
        long = by_id.get(pair.long_cluster_id)
        short_e = short.heldout_error if short else None
        long_e = long.heldout_error if long else None
        if pair.reason == "partner_failed_stage2":
            status, reason, delta = "skipped", pair.reason, None
        elif short.status != "ok" or long.status != "ok":
            status = "failed" if "failed" in (short.status, long.status) else "skipped"
            reason, delta = "harmonic_fit_unfinished", None
        else:
            status, reason, delta = "ok", None, long_e - short_e
        pairs.append(replace(pair, status=status, reason=reason, delta_e=delta,
                             margin=margin if status == "ok" else None,
                             prefers=harmonic_preference(delta, margin) if status == "ok" else "undecided",
                             short_heldout_error=short_e, long_heldout_error=long_e))
    return replace(result, per_candidate=fits, harmonic_comparisons=pairs,
                   complete=verification_finished(fits, pairs))
