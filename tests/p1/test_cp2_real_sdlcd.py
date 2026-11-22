from pathlib import Path

import numpy as np
import pytest

from classical_analysis.lomb_scargle import run_lomb_scargle
from classical_analysis.pdm import run_pdm
from classical_analysis.window import compute_window
from core.config import load_config, resolve_period_bounds
from core.schemas import LOMB_SCARGLE, PDM, assert_contract_equal
from ingestion import load_observation
from pipeline.controller import analyze
from preprocessing.prepare import prepare
from preprocessing.quality import assess_stage1


SDLCD_DIR = Path(__file__).resolve().parents[2] / "data" / "sdlcd_raw"


def science_payload(result):
    payload = result.to_dict()
    del payload["meta"]["runtime_s"]
    return payload


def assert_window_contract(window, resolution, max_peaks):
    assert set(window) == {"freq_hz", "window_power", "peaks", "freq_resolution"}
    assert type(window["freq_resolution"]) is float
    assert window["freq_resolution"] == pytest.approx(resolution)
    frequencies = window["freq_hz"]
    power = window["window_power"]
    assert isinstance(frequencies, list) and isinstance(power, list)
    assert len(frequencies) == len(power) and len(frequencies) > 1
    assert all(type(value) is float for value in frequencies + power)
    assert frequencies[0] == 0.0 and power[0] == 1.0
    assert np.isfinite(frequencies).all() and np.all(np.diff(frequencies) > 0)
    assert all(0.0 <= value <= 1.0 for value in power)

    power_by_frequency = dict(zip(frequencies, power, strict=True))
    peaks = window["peaks"]
    assert isinstance(peaks, list) and 0 < len(peaks) <= max_peaks
    for peak in peaks:
        assert set(peak) == {"freq_hz", "height"}
        offset, height = peak["freq_hz"], peak["height"]
        assert type(offset) is float and type(height) is float
        assert 0.0 < offset <= frequencies[-1] and 0.0 <= height <= 1.0
        assert offset in power_by_frequency
        assert height == pytest.approx(power_by_frequency[offset])
    heights = [peak["height"] for peak in peaks]
    assert heights == sorted(heights, reverse=True)


@pytest.mark.parametrize(("filename", "expected_quality", "has_candidates"), [
    pytest.param("00029A_2_3_R_20190322_DATA.txt", "suitable", True, id="suitable-with-candidates"),
    pytest.param("00080B_1_2_R_20190724_DATA.txt", "limited", True, id="limited-with-candidates"),
    pytest.param("00041B_2_3_R_20191118_DATA.txt", "limited", False, id="limited-successful-empty"),
])
def test_real_sdlcd_cp2_backend(filename, expected_quality, has_candidates, isolated_runs):
    path = SDLCD_DIR / filename
    try:
        path.stat()
    except FileNotFoundError:
        pytest.skip(f"Optional local SDLCD observation is absent: data/sdlcd_raw/{filename}")

    config = load_config()
    bounds = resolve_period_bounds(config.period_bounds)
    classical = {
        **config.classical,
        "period_min_s": bounds.min_period_s,
        "period_max_s": bounds.max_period_s,
    }
    observation = load_observation(str(path))
    prepared = prepare(observation, config.preprocess)
    quality = assess_stage1(prepared, config.preprocess)
    assert quality.status == expected_quality
    assert not prepared.is_stub and not quality.is_stub

    analysis = analyze(observation, config)
    analysis.validate()
    assert_contract_equal(analysis.prepared, prepared)
    assert_contract_equal(analysis.quality, quality)

    resolution = prepared.meta["freq_resolution"]
    for method, runner, plot_limit in (
        (LOMB_SCARGLE, run_lomb_scargle, config.classical["ls_periodogram_max_points"]),
        (PDM, run_pdm, config.classical["pdm_periodogram_max_points"]),
    ):
        direct = runner(prepared, classical)
        direct.validate()
        assert direct.method == method
        assert direct.status == "ok" and direct.error is None
        assert not direct.is_stub
        assert bool(direct.candidates) is has_candidates
        assert direct.meta["runtime_s"] >= 0
        assert direct.meta["freq_resolution"] == pytest.approx(resolution)

        periodogram = direct.meta["periodogram"]
        frequencies = periodogram["freq_hz"]
        assert 1 < len(frequencies) == len(periodogram["power"]) <= plot_limit
        assert np.isfinite(periodogram["power"]).all()
        assert np.all(np.diff(frequencies) > 0)
        assert frequencies[0] == pytest.approx(1 / bounds.max_period_s)
        assert frequencies[-1] <= 1 / bounds.min_period_s + 1e-12
        for candidate in direct.candidates:
            assert bounds.min_period_s <= candidate.period_s <= bounds.max_period_s
            assert candidate.meta["freq_width_hz"] > 0

        assert method not in analysis.stage_errors
        assert_contract_equal(science_payload(analysis.estimators[method]), science_payload(direct),
                              rtol=1e-9, atol=1e-12)

    window = compute_window(prepared, classical)
    assert_window_contract(window, resolution, config.classical["n_peaks"])
