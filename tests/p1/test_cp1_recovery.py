import json
import runpy
from pathlib import Path

import numpy as np
import pytest

from core.config import load_config
from core.errors import IngestionError
from core.schemas import Observation, PreparedObservation, AnalysisResult, assert_contract_equal
from ingestion import load_observation
from ingestion.reference import load_reference_table
from preprocessing.prepare import prepare
from preprocessing.quality import assess_stage1
from preprocessing.sampling_profile import sampling_profile
from pipeline import controller


@pytest.fixture
def cfg():
    return load_config().preprocess


def observation(times, values, kind="unknown"):
    return Observation("control", np.asarray(times, dtype=float), np.asarray(values, dtype=float), None, kind)


def test_generic_files_and_jd_conversion(tmp_path):
    path = tmp_path / "arbitrary.csv"
    path.write_text("t_s,flux,flux_err\n10,2,0.1\n12,inf,\n11,4,0.2\n")
    obs = load_observation(str(path))
    obs.validate()
    assert obs.brightness_kind == "flux"
    np.testing.assert_array_equal(obs.t_s, [0., 2., 1.])
    assert np.isnan(obs.y[1]) and np.isnan(obs.y_err[1])
    path.write_text("jd,mag\n2450000,12\n2450000.5,11\n")
    obs = load_observation(str(path))
    np.testing.assert_array_equal(obs.t_s, [0., 43200.])
    assert obs.brightness_kind == "mag"


@pytest.mark.parametrize("body", ["", "# comments only", "nonsense", "t_s,flux\n0,bad\n",
    "t_s,flux,flux_err\n0,1,-1\n", "t_s,flux\ninf,1\n", "t_s,flux,mag\n0,1,2\n", "t_s,flux,flux\n0,1,2\n",
    "t_s,flux\n0,1,2\n", "t_s flux\n0 1 2\n"])
def test_invalid_ingestion_is_human_readable(tmp_path, body):
    path = tmp_path / "bad.csv"
    path.write_text(body)
    with pytest.raises(IngestionError, match="Could not read"):
        load_observation(str(path))


def test_sdlcd_layout_without_header_period_leak(tmp_path):
    path = tmp_path / "00029A_control_DATA.txt"
    path.write_text("# Per[s] Other\n# 65.8 1\n# Time ADU ADU_err MAG MAG_err\n"
                    "0 2 -10 0.5 12 0.1\n5 2 20 0.6 11 0.2\n")
    obs = load_observation(str(path))
    assert obs.brightness_kind == "mag"
    np.testing.assert_array_equal(obs.y, [12., 11.])
    np.testing.assert_array_equal(obs.y_err, [.1, .2])
    assert obs.meta["exposure_s"] == [2., 2.]
    assert "65.8" not in json.dumps(obs.to_dict())


def test_actual_preparation_sort_mask_flags_and_originals(cfg):
    obs = observation([100, 101, 101, 102, 130], [12, 11, np.nan, 10, -20], "mag")
    original = obs.to_dict()
    result = prepare(obs, cfg)
    shuffled = observation(obs.t_s[[4, 2, 0, 3, 1]], obs.y[[4, 2, 0, 3, 1]], "mag")
    other = prepare(shuffled, cfg)
    np.testing.assert_array_equal(result.t_s, other.t_s)
    np.testing.assert_array_equal(result.y_norm, other.y_norm)
    np.testing.assert_array_equal(result.flux_rel, other.flux_rel)
    assert obs.to_dict() == original
    assert len(result.t_s) == 5 and not result.is_stub
    assert result.norm_params["sign"] == -1.
    assert result.flags["duplicate"].sum() == 2
    assert result.flags["gap_edge"].sum() == 2
    assert result.flags["bright_outlier"][-1]
    assert result.y_norm[-1] > result.y_norm[0]
    assert np.array_equal(np.isnan(result.y_norm), ~result.valid_mask)
    assert np.array_equal(np.isnan(result.flux_rel), ~result.valid_mask)
    assert result.meta["baseline_s"] == 30.
    assert result.meta["freq_resolution"] == 1 / 30.
    clone = PreparedObservation.from_dict(json.loads(json.dumps(result.to_dict(), allow_nan=False)))
    assert_contract_equal(result, clone)


def test_all_missing_and_duplicate_only(cfg):
    for obs in (observation([0, 1], [np.nan, np.nan], "mag"), observation([2, 2], [1, 2])):
        prep = prepare(obs, cfg)
        assert prep.meta["baseline_s"] == 0.
        assert prep.meta["freq_resolution"] is None
        assert assess_stage1(prep, cfg).status == "insufficient"


def test_quality_controls_and_profiles(cfg):
    times = np.arange(1000.)
    clean = prepare(observation(times, np.sin(2*np.pi*times/100)), cfg)
    assert assess_stage1(clean, cfg).status == "suitable"
    short = prepare(observation(times[:20], clean.y_norm[:20]), cfg)
    assert assess_stage1(short, cfg).status == "insufficient"
    noisy = prepare(observation(times, np.random.default_rng(9).normal(size=1000)), cfg)
    assert assess_stage1(noisy, cfg).status == "limited"
    gapped = prepare(observation(np.r_[times[:200], times[800:]], np.r_[clean.y_norm[:200], clean.y_norm[800:]]), cfg)
    assert assess_stage1(gapped, cfg).status == "limited"
    profile = sampling_profile(gapped, cfg)
    assert set(profile) == {"profile_id", "n_points", "duration_s", "median_dt_s", "gap_fractions", "noise_rel_amplitude", "missing_fraction"}
    assert profile["n_points"] == 400
    assert profile["gap_fractions"] == pytest.approx([601/999])
    assert profile["missing_fraction"] == 0.
    json.dumps(profile, allow_nan=False)


def test_brightness_direction_and_missing_flux(cfg):
    times = np.arange(5.)
    flux = np.array([1., 2., 4., 2., np.nan])
    magnitude = -2.5 * np.log10(flux)
    linear = prepare(observation(times, flux, "flux"), cfg)
    logarithmic = prepare(observation(times, magnitude, "mag"), cfg)
    np.testing.assert_allclose(linear.flux_rel, logarithmic.flux_rel)
    # Logarithmic and linear values share ordered shape, not amplitudes.
    np.testing.assert_array_equal(np.argsort(linear.y_norm), np.argsort(logarithmic.y_norm))
    assert np.isnan(linear.y_norm[-1]) and not linear.valid_mask[-1]
    assert prepare(observation(times, flux), cfg).flux_rel is None


def test_reference_filtering_determinism_and_split_conflict(tmp_path):
    path = tmp_path / "labels.csv"
    header = "obs_id,object_id,period_s\n"
    rows = ["a,one,8\n", "b,one,9\n", "c,two,10\n"]
    rows += [f"reserved{i},three,{code}\n" for i, code in enumerate((5000,6000,7000,9999))]
    path.write_text(header + "".join(rows))
    first = load_reference_table(str(path))
    path.write_text(header + "".join(reversed(rows)))
    assert first == load_reference_table(str(path))
    assert set(first) == {"a", "b", "c"}
    assert first["a"].split == first["b"].split
    path.write_text("obs_id,object_id,period_s,split\na,one,8,val\nb,one,9,test\n")
    with pytest.raises(ValueError, match="both"):
        load_reference_table(str(path))


def test_controller_uses_actual_csv_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(controller, "_OUTPUTS", tmp_path / "outputs")
    results = []
    for n in (20, 200):
        path = tmp_path / f"input{n}.csv"
        times = np.arange(n, dtype=float)
        np.savetxt(path, np.c_[times, np.sin(2*np.pi*times/20)], delimiter=",",
                   header="t_s,brightness", comments="")
        obs = load_observation(str(path))
        result = controller.analyze(obs, load_config())
        result.validate()
        assert not result.prepared.is_stub and not result.quality.is_stub
        assert result.prepared.t_s.size == n
        assert result.prepared.meta["baseline_s"] == n - 1
        assert_contract_equal(result, AnalysisResult.from_dict(result.to_dict()))
        results.append(result)
    assert results[0].quality.status == "insufficient"
    assert all(e.error == "stage1_insufficient" for e in results[0].estimators.values())
    assert results[1].quality.status == "suitable"


def test_audit_empty_input_preserves_existing_csv(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[2] / "preprocessing/audit_sdlcd.py"
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "docs/sdlcd_audit.csv"
    output.parent.mkdir()
    output.write_text("existing evidence")
    with pytest.raises(SystemExit, match="No SDLCD"):
        runpy.run_path(str(script))
    assert output.read_text() == "existing evidence"


def test_downloaded_sdlcd_controller_handoff(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    files = sorted((root / "data/sdlcd_raw").glob("*_DATA.txt"))
    assert len(files) >= 20, "CP1 acceptance requires the locally downloaded real observations"
    monkeypatch.setattr(controller, "_OUTPUTS", tmp_path)
    results = []
    for path in files[:2]:
        measured = np.loadtxt(path)
        obs = load_observation(str(path))
        result = controller.analyze(obs, load_config())
        result.validate()
        assert not result.prepared.is_stub and not result.quality.is_stub
        assert len(result.prepared.y_norm) == len(measured)
        np.testing.assert_array_equal(obs.y, measured[:, 4])
        assert result.prepared.meta["baseline_s"] == measured[-1, 0] - measured[0, 0]
        assert_contract_equal(result, AnalysisResult.from_dict(result.to_dict()))
        results.append(result)
    assert results[0].prepared.obs_id != results[1].prepared.obs_id
    assert not np.array_equal(results[0].prepared.y_norm, results[1].prepared.y_norm)
