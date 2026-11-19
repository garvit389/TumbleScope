"""Phase 0 Lomb–Scargle placeholder."""

from core.schemas import EstimatorResult, LOMB_SCARGLE, NOT_RUN, PreparedObservation


def run_lomb_scargle(prep: PreparedObservation, cfg: dict) -> EstimatorResult:
    return EstimatorResult(LOMB_SCARGLE, "skipped", [], NOT_RUN, {"runtime_s": 0.0}, is_stub=True)
