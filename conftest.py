"""Shared pytest root for the two owner test packages."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from core.config import PipelineConfig
from core.schemas import EstimatorResult, PreparedObservation, QualityReport, ReconciliationResult
from ingestion import load_observation
from pipeline import controller


@pytest.fixture
def fixture_dir():
    return Path(__file__).resolve().parent / "tests" / "fixtures"


@pytest.fixture
def observation(fixture_dir):
    return load_observation(str(fixture_dir / "synthetic_sine.csv"))


@pytest.fixture
def prepared(observation, fixture_dir):
    with np.load(fixture_dir / "mock_prepared.npz", allow_pickle=False) as data:
        return PreparedObservation(
            obs_id=observation.obs_id, t_s=data["t_s"], y_norm=data["y_norm"],
            valid_mask=data["valid_mask"], dt_s=data["dt_s"], flux_rel=None,
            flags={name: data[name] for name in ("bright_outlier", "duplicate", "gap_edge")},
            norm_params=json.loads(data["norm_params"].item()), meta=json.loads(data["meta"].item()),
            is_stub=True,
        )


@pytest.fixture
def stub_config():
    return PipelineConfig(period_bounds={"status": "TBD", "min_period_s": None, "max_period_s": None})


@pytest.fixture
def stub_observation_stages(monkeypatch, prepared):
    monkeypatch.setattr(controller, "prepare", lambda *args: prepared)
    monkeypatch.setattr(controller, "assess_stage1", lambda *args: QualityReport(
        "suitable", {}, ["Synthetic observation."], [], is_stub=True,
    ))


@pytest.fixture
def injected_config(isolated_runs, stub_config, stub_observation_stages):
    return replace(stub_config, period_bounds={
        "status": "SET", "min_period_s": 2.0, "max_period_s": 20.0,
    })


@pytest.fixture
def mock_cnn(fixture_dir):
    return EstimatorResult.from_dict(json.loads((fixture_dir / "mock_cnn_result.json").read_text()))


@pytest.fixture
def mock_recon(fixture_dir):
    return ReconciliationResult.from_dict(
        json.loads((fixture_dir / "mock_reconciliation.json").read_text())
    )


@pytest.fixture
def isolated_runs(monkeypatch, tmp_path):
    monkeypatch.setattr(controller, "_OUTPUTS", tmp_path)
    monkeypatch.setattr(controller, "_repository_revision", lambda: "fixture-revision")
    return tmp_path
