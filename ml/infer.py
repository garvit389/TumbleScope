"""CPU inference for the experimental apparent-period CNN."""

from io import BytesIO
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import logsumexp
import torch

from core.schemas import CNN, NO_CHECKPOINT, TOO_FEW_VALID_SAMPLES, Candidate, EstimatorResult, PreparedObservation
from ml import features
from ml.bundle import FEATURE_KEYS, MODEL_KEYS, digest
from ml.model import PeriodCNN
from ml.train import centers


_ROOT = Path(__file__).resolve().parents[1]
_CHECKPOINT = _ROOT / "models/cnn_v1.pt"
_CALIBRATION = _ROOT / "docs/results/cnn_calibration.json"


def _window_logits(model: PeriodCNN, windows: np.ndarray) -> torch.Tensor:
    with torch.inference_mode():
        return model.eval()(torch.from_numpy(windows)) if len(windows) else torch.empty((0, 129))


def prepared_logits(model, prep, cfg):
    """CPU inference for checkpoint acceptance before calibration at CP3b."""
    return _window_logits(model, features.build_cnn_windows(prep, cfg))


def _load_model(path: Path, cfg: dict, *, expected_dataset_id: str | None = None) -> tuple[PeriodCNN, str]:
    data = path.read_bytes()
    checksum = digest(data)
    if path.with_suffix(".sha256").read_text(encoding="ascii").strip().lower() != checksum:
        raise ValueError("checkpoint checksum mismatch")
    state = torch.load(BytesIO(data), map_location="cpu", weights_only=True)
    trained = state["config"]
    bounds = (cfg["period_min_s"], cfg["period_max_s"])
    if (any(type(bound) not in (int, float) for bound in bounds)
            or not np.isfinite(bounds).all() or not 0 < bounds[0] < bounds[1]):
        raise ValueError("invalid CNN period bounds")
    if tuple(trained["bounds"]) != bounds:
        raise ValueError("checkpoint_bounds_mismatch")
    if any(trained["cnn"][key] != cfg[key] for key in (*FEATURE_KEYS, *MODEL_KEYS)):
        raise ValueError("checkpoint CNN feature or model settings mismatch")
    if expected_dataset_id is not None and state["dataset_id"] != expected_dataset_id:
        raise ValueError("checkpoint dataset identity mismatch")
    model = PeriodCNN(trained["cnn"]).cpu()
    model.load_state_dict(state["model"], strict=True)
    return model.eval(), checksum


def _window_log_probabilities(logits: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    if (logits.ndim != 2 or logits.shape[1] != 129 or not np.isfinite(logits).all()
            or type(temperature) not in (int, float) or not np.isfinite(temperature)
            or temperature <= 0):
        raise ValueError("invalid CNN logits or temperature")
    scaled = (logits - logits.max(axis=1, keepdims=True)) / temperature
    if not np.isfinite(scaled).all():
        raise ValueError("invalid scaled CNN logits")
    return scaled - logsumexp(scaled, axis=1, keepdims=True)


def _aggregate(logits: np.ndarray, valid_counts: np.ndarray, temperature: float) -> tuple[np.ndarray, float]:
    logs = _window_log_probabilities(logits, temperature)
    weights = np.asarray(valid_counts, dtype=np.float64)
    if (weights.shape != (len(logs),) or not len(weights)
            or not np.isfinite(weights).all() or np.any(weights <= 0)):
        raise ValueError("invalid CNN window weights")
    log_weights = np.log(weights)
    combined_log = (np.logaddexp.reduce(logs + log_weights[:, None], axis=0)
                    - np.logaddexp.reduce(log_weights))
    period_only = logs[:, :128]
    combined_period = combined_log[:128]
    conditional = np.exp(period_only - logsumexp(period_only, axis=1, keepdims=True))
    reference = np.exp(combined_period - logsumexp(combined_period))
    tv = np.abs(conditional - reference).sum(axis=1).mean() / 2
    return np.exp(combined_log), float(np.clip(tv, 0, 1))


def _calibration(cfg: dict, checksum: str) -> tuple[float, float]:
    with _CALIBRATION.open(encoding="utf-8") as file:
        report = json.load(file)
    if (report["checkpoint_sha256"] != checksum
            or report["bounds"] != [cfg["period_min_s"], cfg["period_max_s"]]
            or report["encoder"] != {key: cfg[key] for key in FEATURE_KEYS}):
        raise ValueError("CNN calibration identity mismatch")
    threshold, temperature = report["abstain_threshold"], report["temperature"]
    if (type(threshold) not in (float, int) or not 0 <= threshold <= 1
            or type(cfg.get("abstain_threshold")) not in (float, int)
            or cfg["abstain_threshold"] != threshold
            or type(temperature) not in (float, int) or not np.isfinite(temperature)
            or temperature <= 0):
        raise ValueError("invalid CNN calibration or abstention threshold")
    return float(temperature), float(threshold)


def _period_peaks(probabilities: np.ndarray, limit: int) -> list[int]:
    edges = np.r_[0, np.flatnonzero(np.diff(probabilities)) + 1, len(probabilities)]
    peaks = []
    for left, right in zip(edges, edges[1:]):
        value = probabilities[left]
        if (value > 0 and (left == 0 or value > probabilities[left - 1])
                and (right == len(probabilities) or value > probabilities[right])):
            peaks.append(int(left))
    return sorted(peaks, key=lambda index: (-probabilities[index], index))[:limit]


def _was_subsampled(n_rows: int, cfg: dict) -> bool:
    last = max(0, n_rows - cfg["window_length"])
    steps, tail = divmod(last, cfg["window_stride"])
    return steps + bool(tail) + 1 > cfg["max_windows"]


def run_cnn(prep: PreparedObservation, cfg: dict) -> EstimatorResult:
    started = perf_counter()

    def result(status, candidates, error=None, **meta):
        return EstimatorResult(CNN, status, candidates, error,
                               {**meta, "runtime_s": perf_counter() - started})

    try:
        try:
            _CHECKPOINT.stat()
        except FileNotFoundError:
            return result("skipped", [], NO_CHECKPOINT)
        model, checksum = _load_model(_CHECKPOINT, cfg)
        temperature, threshold = _calibration(cfg, checksum)
        windows = features.build_cnn_windows(prep, cfg)
        if not len(windows):
            return result("skipped", [], TOO_FEW_VALID_SAMPLES)
        limit = cfg.get("n_candidates", 5)
        if type(limit) is not int or not 1 <= limit <= 128:
            raise ValueError("invalid n_candidates")
        logits = _window_logits(model, windows).numpy()
        combined, disagreement = _aggregate(logits, windows[:, 1, :].sum(axis=1), temperature)
        nrp = float(np.clip(combined[128], 0, 1))
        abstained = nrp >= threshold
        bounds = [cfg["period_min_s"], cfg["period_max_s"]]
        candidates = []
        if not abstained:
            peaks = _period_peaks(combined[:128], limit)
            bin_centers = centers(bounds)
            for index in peaks:
                raw = float(combined[index])
                candidates.append(Candidate(float(bin_centers[index]), float(raw / combined[peaks[0]]), raw))
        return result("ok", candidates, p_no_reliable=nrp, abstained=abstained,
                      n_windows=len(windows), window_disagreement=disagreement, experimental=True,
                      checkpoint=checksum, period_bounds=bounds,
                      windows_subsampled=_was_subsampled(len(prep.t_s), cfg))
    except Exception as exc:
        return result("failed", [], str(exc) or type(exc).__name__)
