import json

import numpy as np
import pytest

from classical_analysis import pdm
from classical_analysis.lomb_scargle import build_frequency_grid, run_lomb_scargle
from core.config import load_config
from core.schemas import EstimatorResult, Observation, PDM, TOO_FEW_VALID_SAMPLES
from preprocessing.prepare import prepare


@pytest.fixture
def cfg():
    config = load_config()
    return dict(config.classical, period_min_s=config.period_bounds["min_period_s"],
                period_max_s=config.period_bounds["max_period_s"])


def prepared(times, values):
    return prepare(Observation("pdm-test", times, values, None, "unknown"), {})


def irregular_times():
    rng = np.random.default_rng(808)
    times = np.cumsum(rng.uniform(1.2, 2.8, 360))
    keep = (rng.random(times.size) > 0.2) & ~((times > 180) & (times < 273))
    return times[keep]


def science_payload(result):
    payload = result.to_dict()
    del payload["meta"]["runtime_s"]
    return payload


def test_clean_sine_agrees_with_ls_and_round_trips(cfg):
    times = np.arange(300, dtype=float) * 2
    prep = prepared(times, np.sin(2 * np.pi * times / 37))
    result = pdm.run_pdm(prep, cfg)
    ls = run_lomb_scargle(prep, cfg)
    assert result.status == ls.status == "ok", (result.error, ls.error)
    assert result.method == PDM and not result.is_stub
    assert result.candidates[0].period_s == pytest.approx(37, rel=0.01)
    assert result.candidates[0].period_s == pytest.approx(ls.candidates[0].period_s, rel=0.01)
    assert result.meta["periodogram"]["freq_hz"] == ls.meta["periodogram"]["freq_hz"]
    assert result.meta["freq_resolution"] == prep.meta["freq_resolution"]
    result.candidates[0].validate()
    result.validate()
    payload = result.to_dict()
    assert EstimatorResult.from_dict(json.loads(json.dumps(payload, allow_nan=False))).to_dict() == payload


def test_irregular_gapped_recovery_is_deterministic(cfg):
    times = irregular_times()
    phase = 2 * np.pi * times / 41
    prep = prepared(times, np.sin(phase) + 0.3 * np.cos(2 * phase))
    first = pdm.run_pdm(prep, cfg)
    second = pdm.run_pdm(prep, cfg)
    assert first.status == "ok", first.error
    assert first.candidates[0].period_s == pytest.approx(41, rel=0.01)
    assert science_payload(first) == science_payload(second)


def test_mask_and_duplicates_match_epoch_means(cfg):
    times = irregular_times()
    values = np.sin(2 * np.pi * times / 41)
    values[::11] = np.nan
    valid = np.isfinite(values)
    duplicated_times = np.concatenate((times, times[valid]))
    duplicated_values = np.concatenate((values - 0.25, values[valid] + 0.25))
    result = pdm.run_pdm(prepared(duplicated_times, duplicated_values), cfg)
    expected = pdm.run_pdm(prepared(times[valid], values[valid]), cfg)
    assert result.status == expected.status == "ok"
    assert [c.period_s for c in result.candidates] == [c.period_s for c in expected.candidates]
    np.testing.assert_allclose([c.raw_score for c in result.candidates], [c.raw_score for c in expected.candidates])
    np.testing.assert_allclose(result.meta["periodogram"]["power"], expected.meta["periodogram"]["power"])


def test_pooled_sample_variance_ignores_underfilled_bins():
    # At 1 Hz the third phase bin has one sample and is dropped.
    times = np.array([0.0, 0.1, 0.4, 0.5, 0.8])
    values = np.array([1.0, 3.0, 5.0, 7.0, 100.0])
    theta, within_df, total_df = pdm._phase_dispersion(times, values, np.array([1.0]), 3, 2)
    within_variance = (np.var(values[:2], ddof=1) + np.var(values[2:4], ddof=1)) / 2
    assert theta[0] == pytest.approx(within_variance / np.var(values[:4], ddof=1))
    assert (within_df[0], total_df[0]) == (2, 3)


def test_ranking_bounds_and_width(cfg, monkeypatch):
    times = irregular_times()
    prep = prepared(times, np.sin(times))
    cfg = dict(cfg, period_min_s=10, period_max_s=50, pdm_max_candidates=3,
               pdm_periodogram_max_points=12)
    grid = build_frequency_grid(prep, cfg)
    theta = np.ones(grid.size)
    theta[[0, 20, 23, 35, grid.size - 1]] = [0.3, 0.1, 0.2, 0.05, 0.35]
    monkeypatch.setattr(pdm, "_phase_dispersion", lambda *args: (theta,
                        np.full(grid.size, 190), np.full(grid.size, 199)))
    result = pdm.run_pdm(prep, cfg)
    assert result.status == "ok", result.error
    assert [c.raw_score for c in result.candidates] == pytest.approx([0.95, 0.9, 0.7])
    assert [c.support for c in result.candidates] == pytest.approx([1, 0.9 / 0.95, 0.7 / 0.95])
    assert all(10 <= c.period_s <= 50 for c in result.candidates)
    assert result.candidates[-1].period_s == 50
    assert result.candidates[-1].meta["freq_width_hz"] == prep.meta["freq_resolution"]
    assert result.candidates[0].meta["freq_width_hz"] == pytest.approx(grid[1] - grid[0])
    assert len(result.meta["periodogram"]["freq_hz"]) == len(result.meta["periodogram"]["power"]) == 12
    result.validate()


@pytest.mark.parametrize("kind", ["white_noise", "constant", "unusable_phase_bins"])
def test_success_without_significant_candidate(cfg, kind):
    times = irregular_times()
    values = np.random.default_rng(190).normal(size=times.size)
    if kind == "constant":
        values[:] = 1.0
    elif kind == "unusable_phase_bins":
        times = np.arange(6, dtype=float)
        values = values[:6]
        cfg = dict(cfg, pdm_phase_bins=100)
    result = pdm.run_pdm(prepared(times, values), cfg)
    assert (result.status, result.error, result.candidates) == ("ok", None, [])
    assert result.meta["periodogram"]["power"]
    result.validate()


def test_significance_policy_is_configurable(cfg, monkeypatch):
    times = irregular_times()
    prep = prepared(times, np.sin(times))
    grid = build_frequency_grid(prep, cfg)
    probabilities = np.ones(grid.size)
    probabilities[5] = 0.02 / grid.size
    monkeypatch.setattr(pdm, "_trial_probabilities", lambda *args: probabilities)
    theta = np.ones(grid.size)
    theta[5] = 0.1
    monkeypatch.setattr(pdm, "_phase_dispersion", lambda *args: (theta,
                        np.full(grid.size, 190), np.full(grid.size, 199)))
    strict = pdm.run_pdm(prep, dict(cfg, pdm_false_alarm_alpha=0.01))
    permissive = pdm.run_pdm(prep, dict(cfg, pdm_false_alarm_alpha=0.05))
    assert strict.status == permissive.status == "ok"
    assert strict.candidates == []
    assert len(permissive.candidates) == 1


@pytest.mark.parametrize("times, values", [
    (np.zeros(20), np.arange(20, dtype=float)),
    (np.arange(20, dtype=float), np.r_[np.arange(5, dtype=float), np.full(15, np.nan)]),
])
def test_too_few_distinct_valid_samples(cfg, times, values):
    result = pdm.run_pdm(prepared(times, values), cfg)
    assert (result.status, result.error, result.candidates) == ("skipped", TOO_FEW_VALID_SAMPLES, [])
    result.validate()


@pytest.mark.parametrize("updates", [
    {"period_min_s": None}, {"period_max_s": 1}, {"pdm_phase_bins": 1},
    {"pdm_min_bin_occupancy": 1}, {"pdm_false_alarm_alpha": np.inf},
])
def test_invalid_inputs_fail(cfg, updates):
    times = irregular_times()
    result = pdm.run_pdm(prepared(times, np.sin(times)), dict(cfg, **updates))
    assert result.status == "failed" and result.error and result.candidates == []
    result.validate()


def test_numerical_failure_is_structured(cfg, monkeypatch):
    times = irregular_times()

    def nonfinite_statistic(times, values, grid, bins, occupancy):
        return np.full(grid.size, np.nan), np.ones(grid.size, dtype=int), np.full(grid.size, 2)

    monkeypatch.setattr(pdm, "_phase_dispersion", nonfinite_statistic)
    result = pdm.run_pdm(prepared(times, np.sin(times)), cfg)
    assert result.status == "failed" and "non-finite" in result.error
    assert result.candidates == []
    result.validate()


def test_benchmark_two_thousand_epochs_and_five_thousand_trials(cfg):
    rng = np.random.default_rng(842)
    times = np.cumsum(rng.uniform(1.4, 2.6, 2000))
    prep = prepared(times, np.sin(2 * np.pi * times / 47))
    grid = build_frequency_grid(prep, cfg)
    oversampling = 4998.5 * prep.meta["freq_resolution"] / (grid[-1] - grid[0])
    cfg = dict(cfg, frequency_oversampling=oversampling)
    assert build_frequency_grid(prep, cfg).size == 5000
    result = pdm.run_pdm(prep, cfg)
    assert result.status == "ok", result.error
    assert result.candidates[0].period_s == pytest.approx(47, rel=0.01)
    elapsed = result.meta["runtime_s"]
    assert elapsed < 10, f"2,000 by 5,000 PDM search took {elapsed:.3f} s"
