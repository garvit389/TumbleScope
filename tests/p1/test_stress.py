import numpy as np
import pytest

from core.schemas import Observation
from evaluation.stress import (
    add_measurement_noise, clustered_irregular_sampling, crop_baseline, reduce_modulation, remove_gap,
)
from preprocessing.prepare import prepare


@pytest.fixture
def raw():
    times = np.arange(30, dtype=float)
    # Deliberately unsorted, with aligned uncertainty, known metadata, and caller-declared metadata.
    order = np.random.default_rng(3).permutation(times.size)
    values = 2 + 0.4 * np.sin(times)
    values[[2, 17]] = np.nan
    obs = Observation(
        "stress-sample", times[order], values[order], (times[order] + 1) / 10, "flux",
        {"exposure_s": (times[order] + 0.25).tolist(),
         "source_row": np.array(order), "assumptions": ["original"]},
    )
    obs.validate()
    return obs


def assert_subset(original, result, keep):
    result.validate()
    assert result.obs_id == original.obs_id and result.brightness_kind == original.brightness_kind
    np.testing.assert_array_equal(result.t_s, original.t_s[keep])
    np.testing.assert_array_equal(result.y, original.y[keep])
    np.testing.assert_array_equal(result.y_err, original.y_err[keep])
    np.testing.assert_array_equal(result.meta["exposure_s"], np.array(original.meta["exposure_s"])[keep])
    np.testing.assert_array_equal(result.meta["source_row"], original.meta["source_row"][keep])


def test_noise_is_seeded_preserves_mask_and_keeps_flux_positive(raw):
    snapshot = raw.to_dict()
    first = add_measurement_noise(raw, sigma=0.4, seed=21)
    second = add_measurement_noise(raw, sigma=0.4, seed=21)
    different = add_measurement_noise(raw, sigma=0.4, seed=22)
    valid = np.isfinite(raw.y)
    np.testing.assert_array_equal(first.y, second.y)
    assert not np.array_equal(first.y[valid], different.y[valid])
    assert (first.y[valid] > 0).all()
    np.testing.assert_array_equal(np.isnan(first.y), ~valid)
    np.testing.assert_array_equal(first.t_s, raw.t_s)
    np.testing.assert_array_equal(first.y_err, raw.y_err)
    prepared = prepare(first, {})
    assert (prepared.flux_rel[prepared.valid_mask] > 0).all()
    first.meta["assumptions"].append("new")
    first.y_err[0] = 99
    assert raw.to_dict() == snapshot


def test_magnitude_noise_uses_supplied_brightness_units():
    obs = Observation("magnitudes", np.array([2., 1., 0.]), np.array([9., np.nan, 11.]), None, "mag")
    result = add_measurement_noise(obs, sigma=0.25, seed=3)
    expected = obs.y.copy()
    expected[[0, 2]] += 0.25 * np.random.default_rng(3).standard_normal(2)
    np.testing.assert_array_equal(result.y, expected)
    assert result.y_err is None


def test_crop_preserves_alignment_and_shortens_baseline(raw):
    result = crop_baseline(raw, start_s=1, duration_s=4, row_aligned_meta=("source_row",))
    assert_subset(raw, result, (raw.t_s >= 1) & (raw.t_s <= 5))
    assert np.ptp(result.t_s[np.isfinite(result.y)]) == 4
    assert np.isnan(result.y).sum() == 1
    with pytest.raises(ValueError, match="shorten"):
        crop_baseline(raw, duration_s=100)


def test_gap_removes_only_the_interior_and_creates_a_larger_gap(raw):
    result = remove_gap(raw, start_s=8, end_s=13, row_aligned_meta=("source_row",))
    assert_subset(raw, result, (raw.t_s <= 8) | (raw.t_s >= 13))
    assert np.max(np.diff(np.sort(result.t_s[np.isfinite(result.y)]))) == 5
    with pytest.raises(ValueError, match="on both sides"):
        remove_gap(raw, start_s=-1, end_s=5)
    with pytest.raises(ValueError, match="inside"):
        remove_gap(raw, start_s=8, end_s=9)


def test_modulation_scales_about_median_and_preserves_flux_and_missing_rows(raw):
    result = reduce_modulation(raw, factor=0.25)
    valid = np.isfinite(raw.y)
    centre = np.median(raw.y[valid])
    np.testing.assert_allclose(result.y[valid] - centre, 0.25 * (raw.y[valid] - centre))
    np.testing.assert_array_equal(np.isnan(result.y), ~valid)
    np.testing.assert_array_equal(result.t_s, raw.t_s)
    np.testing.assert_array_equal(result.y_err, raw.y_err)
    assert (result.y[valid] > 0).all()
    assert not np.shares_memory(result.y, raw.y)


def test_clustered_sampling_is_seeded_and_creates_clusters_of_existing_epochs(raw):
    snapshot = raw.to_dict()
    kwargs = dict(n_clusters=3, cluster_width_s=3, row_aligned_meta=("source_row",))
    result = clustered_irregular_sampling(raw, seed=6, **kwargs)
    repeat = clustered_irregular_sampling(raw, seed=6, **kwargs)
    assert result.to_dict() == repeat.to_dict()
    assert_subset(raw, result, np.isin(raw.t_s, result.t_s))
    valid = result.t_s[np.isfinite(result.y)]
    assert valid.size < np.isfinite(raw.y).sum()
    assert np.count_nonzero(np.diff(np.sort(valid)) > 3) == 2
    assert raw.to_dict() == snapshot
    with pytest.raises(ValueError, match="no pair"):
        clustered_irregular_sampling(raw, n_clusters=3, cluster_width_s=0.25, seed=6)


def test_row_aligned_metadata_length_is_checked(raw):
    raw.meta["exposure_s"].pop()
    with pytest.raises(ValueError, match="match the observation length"):
        crop_baseline(raw, duration_s=8)
