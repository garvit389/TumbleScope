"""Phase 0 simulator placeholder."""

import numpy as np

from physics.params import SimParams


def simulate_lightcurve(params: "SimParams", t_s: np.ndarray) -> np.ndarray:
    return np.zeros_like(t_s, dtype=np.float64)
