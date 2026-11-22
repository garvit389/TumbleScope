"""Phase 0 inference placeholder."""

from core.schemas import CNN, NO_CHECKPOINT, EstimatorResult, PreparedObservation
from ml import features


def prepared_logits(model, prep, cfg):
    """CPU inference for checkpoint acceptance before calibration at CP3b."""
    import torch
    windows = features.build_cnn_windows(prep, cfg)
    with torch.inference_mode():
        return model.eval()(torch.from_numpy(windows)) if len(windows) else torch.empty((0, 129))


def run_cnn(prep: PreparedObservation, cfg: dict) -> EstimatorResult:
    return EstimatorResult(CNN, "skipped", [], NO_CHECKPOINT, {"runtime_s": 0.0}, is_stub=True)
