import json
from math import isfinite
from pathlib import Path

import numpy as np


def test_shared_fixture_seams(observation, prepared, mock_cnn, mock_recon):
    for model in (observation, prepared, mock_cnn, mock_recon):
        model.validate()
    assert (np.diff(observation.t_s) > np.median(np.diff(observation.t_s))).any()
    assert (~prepared.valid_mask).any()
    np.testing.assert_array_equal(prepared.t_s, observation.t_s)
    np.testing.assert_array_equal(~prepared.valid_mask, np.isnan(prepared.y_norm))
    valid_times = prepared.t_s[prepared.valid_mask]
    assert prepared.meta["baseline_s"] == valid_times[-1] - valid_times[0]
    assert prepared.meta["freq_resolution"] == 1 / prepared.meta["baseline_s"]
    periods = {candidate.period_s for candidate in mock_cnn.candidates}
    assert len(mock_cnn.candidates) == len(periods) == 3
    assert any(2 * period in periods for period in periods)
    assert any(link["ratio"] == 2 for cluster in mock_recon.clusters for link in cluster.harmonic_links)


def test_fallback_sampling_profile_schema():
    path = Path(__file__).resolve().parents[1] / "data" / "profiles" / "default_profiles.json"
    profiles = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(profiles, list) and profiles
    for record in profiles:
        assert type(record) is dict and set(record) == {
            "profile_id", "n_points", "duration_s", "median_dt_s", "gap_fractions",
            "noise_rel_amplitude", "missing_fraction",
        }
        assert isinstance(record["profile_id"], str) and record["profile_id"]
        assert type(record["n_points"]) is int and record["n_points"] > 0
        for key in ("duration_s", "median_dt_s", "noise_rel_amplitude", "missing_fraction"):
            assert type(record[key]) in (float, int) and isfinite(record[key]) and record[key] >= 0
        assert record["duration_s"] > 0 and record["median_dt_s"] > 0
        assert 0 <= record["missing_fraction"] <= 1
        assert type(record["gap_fractions"]) is list
        assert all(type(gap) in (float, int) and isfinite(gap) and 0 <= gap <= 1
                   for gap in record["gap_fractions"])
