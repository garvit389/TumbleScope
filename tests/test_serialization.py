from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pytest

from core.schemas import ReferenceLabel, assert_contract_equal
from physics.verify import verify_candidates
from pipeline import controller
from pipeline.controller import analyze


def test_round_trips_all_shared_types(
    observation, prepared, mock_cnn, mock_recon, isolated_runs, stub_config, stub_observation_stages,
):
    observed = replace(observation, meta={"vector": np.array([[1, 2]], dtype=np.int32),
                                         "count": np.int64(2), "flag": np.bool_(True),
                                         "scale": np.float32(0.5)})
    result = analyze(observed, stub_config)
    verification = verify_candidates(prepared, mock_recon, {})
    objects = [observed, ReferenceLabel("obs", "object", 8.0, "val"), prepared, result.quality,
               mock_cnn.candidates[0], mock_cnn, mock_recon.clusters[0], mock_recon,
               verification.per_candidate[0], verification, result.verdict, result]
    for model in objects:
        model.validate()
        wire = json.loads(json.dumps(model.to_dict(), allow_nan=False))
        assert_contract_equal(model, type(model).from_dict(wire))
    assert None in observed.to_dict()["y"]["data"]
    assert observed.to_dict()["y_err"] is None


def test_bad_literal_unknown_field_and_nested_array(observation):
    payload = observation.to_dict()
    for update in ({"brightness_kind": "invalid"}, {"extra_top_level": 1}):
        with pytest.raises(ValueError, match="contract:"):
            type(observation).from_dict({**payload, **update})
    bad_array = deepcopy(payload)
    bad_array["t_s"]["shape"] = [2, 2]
    with pytest.raises(ValueError, match="contract:"):
        type(observation).from_dict(bad_array)


def test_infinity_refused_on_both_sides(observation):
    payload = observation.to_dict()
    payload["y"]["data"][0] = float("inf")
    with pytest.raises(ValueError, match="contract: non-finite value"):
        type(observation).from_dict(payload)
    data = observation.y.copy()
    data[0] = np.inf
    with pytest.raises(ValueError, match="contract: non-finite value"):
        replace(observation, y=data).to_dict()


def test_array_missing_values_use_json_null(observation, prepared):
    observation = replace(observation, y_err=np.where(np.isnan(observation.y), np.nan, 0.1))
    prepared = replace(prepared, flux_rel=np.where(prepared.valid_mask, 1.0, np.nan))
    for model, field in ((observation, "y_err"), (prepared, "flux_rel")):
        payload = model.to_dict()
        assert None in payload[field]["data"]
        assert_contract_equal(model, type(model).from_dict(
            json.loads(json.dumps(payload, allow_nan=False)),
        ))
        data = payload[field]["data"]
        data[data.index(None)] = float("nan")
        with pytest.raises(ValueError, match="array NaN must be encoded as null"):
            type(model).from_dict(payload)


def test_analysis_round_trip_rechecks_nested_verification_scope(
    monkeypatch, observation, injected_config, mock_cnn, mock_recon,
):
    monkeypatch.setattr(controller, "run_cnn", lambda *args: mock_cnn)
    monkeypatch.setattr(controller, "reconcile", lambda *args: mock_recon)
    result = analyze(observation, injected_config)
    payload = json.loads(json.dumps(result.to_dict(), allow_nan=False))
    assert_contract_equal(result, type(result).from_dict(payload))
    payload["verification"]["per_candidate"].pop()
    with pytest.raises(ValueError, match="verification membership|harmonic comparison references unknown fits"):
        type(result).from_dict(payload)
    payload = result.to_dict()
    payload["verification"]["harmonic_comparisons"].clear()
    with pytest.raises(ValueError, match="harmonic comparison membership"):
        type(result).from_dict(payload)


def test_analysis_schema_version_is_required(
    observation, isolated_runs, stub_config, stub_observation_stages,
):
    result = analyze(observation, stub_config)
    payload = result.to_dict()
    for version in (0, 4, True, None):
        with pytest.raises(ValueError, match="schema version"):
            type(result).from_dict({**payload, "schema_version": version})
    del payload["schema_version"]
    with pytest.raises(ValueError, match="schema version"):
        type(result).from_dict(payload)
