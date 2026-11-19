"""Phase 0 placeholder. See CONTRACTS.md §16 for the real feature builder."""

import numpy as np

from core.schemas import PreparedObservation


def build_cnn_windows(prep: PreparedObservation, cfg: dict) -> np.ndarray:
    return np.empty((0, 3, 1024), dtype=np.float32)
