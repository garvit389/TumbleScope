"""Phase 0 spectral window placeholder."""

from core.schemas import PreparedObservation


def compute_window(prep: PreparedObservation, cfg: dict) -> dict:
    return {
        "freq_hz": [0.0],
        "window_power": [1.0],
        "peaks": [],
        "freq_resolution": prep.meta["freq_resolution"],
    }
