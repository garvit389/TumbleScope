import json

from plotly.graph_objects import Figure
import pytest

from app.plots import fig_lightcurve
from core.errors import IngestionError
from core.schemas import (
    ALL_ESTIMATORS_UNAVAILABLE, ESTIMATOR_METHODS, PERIOD_BOUNDS_NOT_SET,
    AnalysisResult, assert_contract_equal,
)
from ingestion import load_observation
from pipeline.controller import analyze


def test_stub_pipeline(observation, isolated_runs, stub_config, stub_observation_stages):
    result = analyze(observation, stub_config)
    result.validate()
    clone = AnalysisResult.from_dict(json.loads(json.dumps(result.to_dict(), allow_nan=False)))
    assert_contract_equal(result, clone)
    assert result.prepared.is_stub
    assert result.quality.is_stub
    assert tuple(result.estimators) == ESTIMATOR_METHODS
    assert all(item.status == "skipped" and item.error == PERIOD_BOUNDS_NOT_SET
               for item in result.estimators.values())
    assert result.verification is None
    assert result.verdict.evidence_summary["inconclusive_cause"] == ALL_ESTIMATORS_UNAVAILABLE
    snapshot = json.loads((isolated_runs / "run_config.json").read_text())
    assert snapshot["config_hash"] == result.config_hash
    assert snapshot["config"]["period_bounds"]["status"] == "TBD"
    assert "numpy" in snapshot["package_versions"]
    assert snapshot["commit"] == "fixture-revision"


def test_phase0_entry_points(prepared, isolated_runs, observation, stub_config):
    assert isinstance(fig_lightcurve(prepared), Figure)
    with pytest.raises(IngestionError):
        load_observation("another.csv")
    with pytest.raises(NotImplementedError, match="Only Fast"):
        analyze(observation, stub_config, mode="thorough")
