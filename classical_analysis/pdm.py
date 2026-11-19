"""Phase 0 PDM placeholder."""

from core.schemas import EstimatorResult, NOT_RUN, PDM, PreparedObservation


def run_pdm(prep: PreparedObservation, cfg: dict) -> EstimatorResult:
    return EstimatorResult(PDM, "skipped", [], NOT_RUN, {"runtime_s": 0.0}, is_stub=True)
