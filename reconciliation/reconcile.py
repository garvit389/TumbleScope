"""Day 2 reconciliation skeleton for carrying candidates into clusters."""

from core.schemas import Cluster, EstimatorResult, PreparedObservation, ReconciliationResult


def reconcile(prep: PreparedObservation, estimators: list[EstimatorResult],
              cfg: dict) -> ReconciliationResult:
    available = [result for result in estimators if result.status == "ok"]
    entries = sorted(
        ((result.method, candidate) for result in available for candidate in result.candidates),
        key=lambda entry: (-entry[1].support, entry[1].period_s, entry[0]),
    )
    if not entries:
        note = "no_candidates" if available else "no_usable_estimators"
        return ReconciliationResult([], [], False, [note], is_stub=True)

    baseline = prep.meta["baseline_s"]
    clusters = [
        Cluster(
            cluster_id=index,
            period_s=candidate.period_s,
            members={method: candidate.period_s},
            evidence={method: candidate.support},
            alias_risk=0.0,
            harmonic_links=[],
            score=candidate.support,
            # The schema pairs an empty failure list with passed=True.
            # The empty shortlist keeps this placeholder out of verification.
            stage2={"passed": True, "cycles": float(baseline / candidate.period_s),
                    "fail_reasons": []},
        )
        for index, (method, candidate) in enumerate(entries, start=1)
    ]
    return ReconciliationResult(clusters, [], False, ["stage2_not_evaluated"], is_stub=True)
