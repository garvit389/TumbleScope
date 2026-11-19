"""Phase 0 reconciliation placeholder."""

from core.schemas import EstimatorResult, PreparedObservation, ReconciliationResult


def reconcile(prep: PreparedObservation, estimators: list[EstimatorResult],
              cfg: dict) -> ReconciliationResult:
    note = "no_candidates" if any(result.status == "ok" for result in estimators) else "no_usable_estimators"
    return ReconciliationResult([], [], False, [note], is_stub=True)
