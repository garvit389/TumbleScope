"""Phase 0 inference placeholder."""

from core.schemas import CNN, NO_CHECKPOINT, EstimatorResult, PreparedObservation


def run_cnn(prep: PreparedObservation, cfg: dict) -> EstimatorResult:
    return EstimatorResult(CNN, "skipped", [], NO_CHECKPOINT, {"runtime_s": 0.0}, is_stub=True)
