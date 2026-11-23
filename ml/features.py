"""Shared CNN input encoding, CONTRACTS.md §16."""

import numpy as np

from core.schemas import PreparedObservation


def build_cnn_windows(prep: PreparedObservation, cfg: dict) -> np.ndarray:
    prep.validate()
    width, stride = cfg["window_length"], cfg["window_stride"]
    cap, minimum = cfg["max_windows"], cfg["min_valid_per_window"]
    clip, floor = cfg["y_clip"], cfg["dt_floor_s"]
    low, high = cfg["L_min"], cfg["L_max"]
    if (width != 1024 or not 0 < stride <= width or cap < 2 or not 0 < minimum <= width
            or not np.isfinite([clip, floor, low, high]).all() or clip <= 0
            or floor <= 0 or low >= high or np.log10(floor) > low):
        raise ValueError("invalid CNN feature settings")
    count = len(prep.t_s)
    last = max(0, count - width)
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    if len(starts) > cap:
        indices = np.rint(np.linspace(0, len(starts) - 1, cap)).astype(int)
        starts = [starts[i] for i in indices]
    rows = np.zeros((3, max(width, count)), dtype=np.float32)
    valid = prep.valid_mask
    rows[0, :count][valid] = np.clip(prep.y_norm[valid], -clip, clip) / clip
    rows[1, :count] = valid
    log_dt = np.log10(np.maximum(prep.dt_s, floor))
    rows[2, :count] = (np.clip(log_dt, low, high) - low) / (high - low)
    windows = [rows[:, i:i + width] for i in starts if valid[i:i + width].sum() >= minimum]
    if not windows:
        return np.empty((0, 3, width), dtype=np.float32)
    return np.ascontiguousarray(np.stack(windows))
