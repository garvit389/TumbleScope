import json
from dataclasses import replace
from time import perf_counter

import numpy as np
import pytest

from classical_analysis import window
from core.config import load_config
from core.schemas import Observation
from preprocessing.prepare import prepare


@pytest.fixture
def cfg():
    config = load_config()
    return dict(config.classical, period_min_s=config.period_bounds["min_period_s"],
                period_max_s=config.period_bounds["max_period_s"])


def prepared(times, values=None):
    if values is None:
        values = np.ones_like(times)
    return prepare(Observation("window-test", times, values, None, "unknown"), {})


def assert_schema(result):
    assert set(result) == {"freq_hz", "window_power", "peaks", "freq_resolution"}
    frequencies = result["freq_hz"]
    power = result["window_power"]
    assert len(frequencies) == len(power)
    assert all(type(value) is float for value in frequencies + power)
    assert np.all(np.diff(frequencies) > 0)
    assert frequencies[0] == 0.0 and power[0] == 1.0
    assert all(0 <= value <= 1 for value in power)
    assert result["freq_resolution"] is None or type(result["freq_resolution"]) is float
    heights = []
    index_by_frequency = {frequency: index for index, frequency in enumerate(frequencies)}
    for peak in result["peaks"]:
        assert set(peak) == {"freq_hz", "height"}
        assert type(peak["freq_hz"]) is type(peak["height"]) is float
        assert peak["freq_hz"] > 0
        assert peak["height"] == power[index_by_frequency[peak["freq_hz"]]]
        heights.append(peak["height"])
    assert heights == sorted(heights, reverse=True)
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_regular_sampling_recovers_cadence_harmonics(cfg):
    cadence = 4.0
    prep = prepared(np.arange(140, dtype=float) * cadence)
    result = window.compute_window(prep, dict(cfg, n_peaks=3))
    assert_schema(result)
    assert len(result["peaks"]) == 3
    offsets = sorted(peak["freq_hz"] for peak in result["peaks"])
    assert offsets == pytest.approx(np.arange(1, 4) / cadence, abs=prep.meta["freq_resolution"] / 5)
    assert all(peak["height"] > 0.95 for peak in result["peaks"])
    assert np.diff(result["freq_hz"]).max() <= prep.meta["freq_resolution"] / 5 + 1e-15


def test_irregular_sampling_and_gaps_match_direct_phasor_sum(cfg):
    rng = np.random.default_rng(981)
    times = np.cumsum(rng.uniform(2.0, 4.0, 240))
    keep = (rng.random(times.size) > 0.25) & ~((times > 250) & (times < 400))
    times = times[keep]
    prep = prepared(times, rng.normal(size=times.size))
    result = window.compute_window(prep, cfg)
    assert_schema(result)
    frequencies = np.asarray(result["freq_hz"])
    check = np.linspace(0, frequencies.size - 1, 20, dtype=int)
    expected = abs(np.exp(2j * np.pi * frequencies[check, None] * prep.t_s).mean(axis=1)) ** 2
    np.testing.assert_allclose(np.asarray(result["window_power"])[check], expected, atol=1e-13)
    assert result["freq_resolution"] == prep.meta["freq_resolution"]
    assert window.compute_window(prep, cfg) == result


def test_valid_mask_and_duplicate_epochs(cfg):
    times = np.arange(130, dtype=float) * 4
    values = np.arange(130, dtype=float)
    values[::9] = np.nan
    valid = np.isfinite(values)
    repeated_times = np.r_[times, times[valid], times[valid]]
    repeated_values = np.r_[values, 2 * values[valid], -values[valid]]
    repeated = prepared(repeated_times, repeated_values)
    collapsed = prepared(times[valid])
    assert window.compute_window(repeated, cfg) == window.compute_window(collapsed, cfg)


def test_uses_only_sampling_fields(cfg):
    prep = prepared(np.arange(60, dtype=float) * 5)

    class SamplingOnly:
        t_s = prep.t_s
        valid_mask = prep.valid_mask
        meta = prep.meta

        @property
        def y_norm(self):
            raise AssertionError("sampling window accessed brightness")

    assert window.compute_window(SamplingOnly(), cfg) == window.compute_window(prep, cfg)


def test_no_peak_above_default_height_for_aperiodic_sampling(cfg):
    times = np.sort(np.random.default_rng(62).uniform(0.0, 8500.0, 2000))
    result = window.compute_window(prepared(times), cfg)
    assert_schema(result)
    assert len(result["freq_hz"]) > 1
    assert result["peaks"] == []


@pytest.mark.parametrize("times, values", [
    ([0., 1., 2.], [np.nan, np.nan, np.nan]),
    ([0., 1., 2.], [np.nan, 1., np.nan]),
    ([0., 1., 2.], [1., np.nan, 1.]),
    ([0., 0., 0.], [1., 2., 3.]),
])
def test_insufficient_sampling_returns_dc(cfg, times, values):
    prep = prepared(np.array(times), np.array(values))
    result = window.compute_window(prep, cfg)
    assert result == {"freq_hz": [0.0], "window_power": [1.0], "peaks": [],
                      "freq_resolution": prep.meta["freq_resolution"]}
    assert_schema(result)


def test_defensive_empty_arrays_return_dc(cfg):
    prep = prepared(np.arange(3, dtype=float))
    empty = replace(prep, t_s=np.array([], dtype=float), valid_mask=np.array([], dtype=bool))
    result = window.compute_window(empty, cfg)
    assert_schema(result)
    assert result["freq_hz"] == [0.0] and result["peaks"] == []


def test_unresolvable_phase_returns_dc(cfg):
    prep = prepared(np.array([0.0, 1.0, 2.0, 1e17]))
    result = window.compute_window(prep, cfg)
    assert_schema(result)
    assert result["freq_hz"] == [0.0] and result["peaks"] == []


def test_range_includes_injected_frequency_difference(cfg):
    prep = prepared(np.arange(80, dtype=float) * 20)
    narrow = window.compute_window(prep, dict(cfg, period_min_s=10, period_max_s=20))
    broad = window.compute_window(prep, dict(cfg, period_min_s=2, period_max_s=20))
    assert narrow["freq_hz"][-1] == pytest.approx(3 / 20)
    assert broad["freq_hz"][-1] == pytest.approx(1 / 2 - 1 / 20)
    assert broad["freq_hz"][0] == 0.0


def test_peak_count_and_height_limit(cfg):
    prep = prepared(np.arange(110, dtype=float) * 4)
    candidate = window.compute_window(prep, dict(cfg, n_peaks=1))
    empty = window.compute_window(prep, dict(cfg, n_peaks=0))
    height_filtered = window.compute_window(prep, dict(cfg, window_min_peak_height=1.0))
    assert len(candidate["peaks"]) == 1
    assert empty["peaks"] == height_filtered["peaks"] == []
    assert candidate["window_power"] == empty["window_power"] == height_filtered["window_power"]


def test_realistic_observation_size_and_grid_cap(cfg):
    rng = np.random.default_rng(831)
    times = np.cumsum(rng.uniform(0.5, 2.5, 2000))
    prep = prepared(times)
    start = perf_counter()
    result = window.compute_window(prep, cfg)
    elapsed = perf_counter() - start
    assert len(result["freq_hz"]) == cfg["window_max_grid_points"]
    assert np.isfinite(result["window_power"]).all()
    assert elapsed < 10, f"window calculation took {elapsed:.3f} s"
