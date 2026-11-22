import json

import numpy as np
import pytest

from classical_analysis import lomb_scargle as ls
from core.config import load_config
from core.schemas import EstimatorResult, LOMB_SCARGLE, Observation, TOO_FEW_VALID_SAMPLES
from preprocessing.prepare import prepare


@pytest.fixture
def cfg():
    config = load_config()
    return dict(config.classical, period_min_s=config.period_bounds["min_period_s"],
                period_max_s=config.period_bounds["max_period_s"])


def prepared(times, values):
    return prepare(Observation("ls-test", times, values, None, "unknown"), {})


def gapped_times():
    rng = np.random.default_rng(62)
    times = np.cumsum(rng.uniform(1.4, 2.6, 300))
    keep = (rng.random(times.size) > 0.2) & ~((times > 200) & (times < 265))
    return times[keep]


def science_payload(result):
    payload = result.to_dict()
    payload["meta"].pop("runtime_s")
    return payload


def test_gapped_sine_and_serialization(cfg):
    times = gapped_times()
    result = ls.run_lomb_scargle(prepared(times, np.sin(2 * np.pi * times / 34)), cfg)
    assert result.status == "ok", result.error
    assert result.method == LOMB_SCARGLE and not result.is_stub
    best = result.candidates[0]
    assert best.period_s == pytest.approx(34, rel=0.01)
    assert best.support == 1
    assert 0 < best.raw_score <= 1
    assert best.meta["freq_width_hz"] > 0
    assert best.meta["rank"] == 1
    best.validate()
    result.validate()
    payload = result.to_dict()
    encoded = json.dumps(payload, allow_nan=False)
    assert EstimatorResult.from_dict(json.loads(encoded)).to_dict() == payload


@pytest.mark.parametrize("nterms", [2, 3])
def test_multiterm_recovers_repeat_period(cfg, nterms):
    times = gapped_times()
    phase = 2 * np.pi * times / 44
    values = 0.55 * np.sin(phase) + np.cos(2 * phase)
    if nterms == 3:
        values += 0.45 * np.sin(3 * phase)
    result = ls.run_lomb_scargle(prepared(times, values), dict(cfg, ls_nterms=nterms))
    assert result.status == "ok", result.error
    assert result.candidates[0].period_s == pytest.approx(44, rel=0.01)


def test_double_hump_preserves_half_period(cfg):
    times = gapped_times()
    phase = 2 * np.pi * times / 44
    result = ls.run_lomb_scargle(prepared(times, np.cos(2 * phase) + 0.12 * np.sin(phase)), cfg)
    assert result.status == "ok", result.error
    assert result.candidates[0].period_s == pytest.approx(22, rel=0.01)


def test_masked_rows_match_removed_rows(cfg):
    times = gapped_times()
    values = np.sin(2 * np.pi * times / 34)
    values[1::8] = np.nan
    mask = np.isfinite(values)
    result = ls.run_lomb_scargle(prepared(times, values), cfg)
    removed = ls.run_lomb_scargle(prepared(times[mask], values[mask]), cfg)
    assert result.status == "ok", result.error
    assert science_payload(result) == science_payload(removed)


def test_duplicate_times_preserve_grid_and_recovery(cfg):
    times = gapped_times()
    repeated = np.repeat(times, np.where(np.arange(times.size) % 5 == 0, 3, 1))
    original = prepared(times, np.sin(2 * np.pi * times / 34))
    duplicated = prepared(repeated, np.sin(2 * np.pi * repeated / 34))
    np.testing.assert_array_equal(ls.build_frequency_grid(original, cfg),
                                  ls.build_frequency_grid(duplicated, cfg))
    result = ls.run_lomb_scargle(duplicated, cfg)
    assert result.status == "ok", result.error
    assert result.candidates[0].period_s == pytest.approx(34, rel=0.01)


def test_regular_sampling_with_multiple_terms(cfg):
    times = np.arange(200, dtype=float) * 2
    phase = 2 * np.pi * times / 40
    result = ls.run_lomb_scargle(prepared(times, np.sin(phase) + np.sin(2 * phase)),
                                dict(cfg, ls_nterms=3))
    assert result.status == "ok", result.error
    assert result.candidates[0].period_s == pytest.approx(40, rel=0.01)


@pytest.mark.parametrize("times, valid", [(np.arange(10, dtype=float), 3), (np.zeros(10), 10)])
def test_too_few_valid_samples(cfg, times, valid):
    values = np.arange(10, dtype=float)
    values[valid:] = np.nan
    result = ls.run_lomb_scargle(prepared(times, values), cfg)
    assert (result.status, result.error, result.candidates) == ("skipped", TOO_FEW_VALID_SAMPLES, [])
    result.validate()


@pytest.mark.parametrize("updates", [
    {"period_min_s": None}, {"period_min_s": 0}, {"period_min_s": True},
    {"period_max_s": 2}, {"period_max_s": np.inf}, {"period_max_s": 3},
    {"ls_nterms": 4}, {"ls_false_alarm_alpha": 1},
])
def test_bad_configuration_fails(cfg, updates):
    times = gapped_times()
    result = ls.run_lomb_scargle(prepared(times, np.sin(times)), dict(cfg, **updates))
    assert result.status == "failed"
    assert result.error and not result.candidates
    result.validate()


def test_numerical_failure_is_structured(cfg, monkeypatch):
    times = gapped_times()

    def nonfinite_power(self, frequencies, **kwargs):
        return np.full_like(frequencies, np.nan)

    monkeypatch.setattr(ls.LombScargle, "power", nonfinite_power)
    result = ls.run_lomb_scargle(prepared(times, np.sin(times)), cfg)
    assert result.status == "failed"
    assert "invalid power" in result.error and not result.candidates
    result.validate()


@pytest.mark.parametrize("nterms", [1, 2, 3])
def test_no_significant_candidate_on_seeded_noise(cfg, nterms):
    times = gapped_times()
    rng = np.random.default_rng(190)
    result = ls.run_lomb_scargle(prepared(times, rng.normal(size=times.size)), dict(cfg, ls_nterms=nterms))
    assert result.status == "ok" and result.error is None
    assert result.candidates == []
    assert result.meta["periodogram"]["power"]
    result.validate()


def test_constant_input_is_successful_empty_search(cfg):
    times = gapped_times()
    result = ls.run_lomb_scargle(prepared(times, np.ones_like(times)), cfg)
    assert result.status == "ok" and result.error is None
    assert result.candidates == []
    assert set(result.meta["periodogram"]["power"]) == {0.0}
    result.validate()


def test_peak_ranking_separation_and_full_grid_width(cfg, monkeypatch):
    times = gapped_times()
    prep = prepared(times, np.sin(times))
    grid = ls.build_frequency_grid(prep, cfg)
    power = np.zeros_like(grid)
    power[[20, 23, 30, grid.size - 1]] = [0.9, 0.8, 0.7, 0.6]
    monkeypatch.setattr(ls, "_power", lambda *args: power)
    result = ls.run_lomb_scargle(prep, dict(cfg, ls_max_candidates=3, ls_periodogram_max_points=12))
    assert result.status == "ok", result.error
    assert [c.raw_score for c in result.candidates] == [0.9, 0.7, 0.6]
    assert [c.support for c in result.candidates] == pytest.approx([1, 0.7 / 0.9, 0.6 / 0.9])
    peak_freqs = [c.meta["freq_hz"] for c in result.candidates]
    assert np.all(np.diff(peak_freqs) >= prep.meta["freq_resolution"])
    assert result.candidates[0].meta["freq_width_hz"] == pytest.approx(grid[1] - grid[0])
    assert result.candidates[-1].meta["freq_width_hz"] == prep.meta["freq_resolution"]
    assert len(result.meta["periodogram"]["freq_hz"]) == 12
    assert len(result.meta["periodogram"]["power"]) == 12
    result.validate()


def test_grid_bounds_and_deterministic_results(cfg):
    times = gapped_times()
    prep = prepared(times, np.sin(2 * np.pi * times / 34))
    cfg = dict(cfg, period_min_s=15, period_max_s=41)
    grid = ls.build_frequency_grid(prep, cfg)
    assert grid[[0, -1]] == pytest.approx([1 / 41, 1 / 15])
    assert np.diff(grid).max() <= prep.meta["freq_resolution"] / cfg["frequency_oversampling"]
    first = ls.run_lomb_scargle(prep, cfg)
    second = ls.run_lomb_scargle(prep, cfg)
    assert first.status == "ok", first.error
    assert first.candidates and all(15 <= candidate.period_s <= 41 for candidate in first.candidates)
    assert science_payload(first) == science_payload(second)

def test_significance_policy_is_configurable(cfg, monkeypatch):
    times = gapped_times()
    prep = prepared(times, np.sin(times))

    def trial_power(times, values, grid, nterms):
        power = np.zeros_like(grid)
        power[20] = 0.1
        return power

    monkeypatch.setattr(ls, "_power", trial_power)
    strict = ls.run_lomb_scargle(prep, dict(cfg, ls_false_alarm_alpha=0.000001))
    permissive = ls.run_lomb_scargle(prep, dict(cfg, ls_false_alarm_alpha=0.1))
    assert strict.status == permissive.status == "ok"
    assert strict.candidates == []
    assert len(permissive.candidates) == 1
