"""Phase 0 placeholder using the shared prepared fixture."""

import json
from pathlib import Path

import numpy as np

from core.schemas import Observation, PreparedObservation


_FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "mock_prepared.npz"


def prepare(obs: Observation, cfg: dict) -> PreparedObservation:
    with np.load(_FIXTURE, allow_pickle=False) as data:
        return PreparedObservation(
            obs_id=obs.obs_id,
            t_s=data["t_s"],
            y_norm=data["y_norm"],
            valid_mask=data["valid_mask"],
            dt_s=data["dt_s"],
            flux_rel=None,
            flags={name: data[name] for name in ("bright_outlier", "duplicate", "gap_edge")},
            norm_params=json.loads(data["norm_params"].item()),
            meta=json.loads(data["meta"].item()),
            is_stub=True,
        )
