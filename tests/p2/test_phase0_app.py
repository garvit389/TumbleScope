from pathlib import Path

from streamlit.testing.v1 import AppTest
from tests.p2.test_controller import pending_reconciliation, unavailable_cnn


def test_local_upload_uses_actual_bytes():
    from io import BytesIO
    from app.main import load_upload
    from core.errors import IngestionError
    import pytest

    raw = Path(__file__).resolve().parents[2] / "data" / "sdlcd_raw"
    for name in ("00029A_2_3_R_20190322_DATA.txt", "00041B_2_3_R_20191118_DATA.txt"):
        upload = BytesIO((raw / name).read_bytes())
        upload.name = name
        obs = load_upload(upload)
        assert obs.obs_id == Path(name).stem
        assert obs.brightness_kind == "mag"
    bad = BytesIO(b"not measurements")
    bad.name = "invalid.txt"
    with pytest.raises(IngestionError):
        load_upload(bad)


def test_bundled_demo_renders_stub_result(
    isolated_runs, monkeypatch, stub_config, stub_observation_stages,
):
    from core import config

    monkeypatch.setattr(config, "load_config", lambda: stub_config)
    entry_point = Path(__file__).resolve().parents[2] / "app" / "main.py"
    demo = AppTest.from_file(str(entry_point)).run(timeout=20)
    assert not demo.exception
    assert not demo.error
    assert any("Synthetic stub result" in warning.value for warning in demo.warning)
    assert any("Inconclusive" == heading.value for heading in demo.subheader)
    assert (isolated_runs / "run_config.json").is_file()


def test_demo_with_real_cp1_stages(tmp_path, monkeypatch):
    from pipeline import controller
    monkeypatch.setattr(controller, "_OUTPUTS", tmp_path)
    monkeypatch.setattr(controller, "reconcile", pending_reconciliation)
    monkeypatch.setattr(controller, "run_cnn", unavailable_cnn)
    entry_point = Path(__file__).resolve().parents[2] / "app" / "main.py"
    demo = AppTest.from_file(str(entry_point)).run(timeout=20)
    assert not demo.exception and not demo.error
    assert any("Stage 2 assessment pending" == heading.value for heading in demo.subheader)
    assert (tmp_path / "run_config.json").is_file()


def test_real_upload_ui_with_classical_fixtures(monkeypatch, isolated_runs):
    from io import BytesIO
    import streamlit as st
    from core.schemas import LOMB_SCARGLE, PDM, Candidate, EstimatorResult
    from pipeline import controller

    def runner(method):
        def fixture_result(prep, cfg):
            resolution = prep.meta["freq_resolution"]
            candidate = Candidate(30., 1., .8, {"freq_width_hz": resolution})
            return EstimatorResult(method, "ok", [candidate], None, {
                "runtime_s": 0., "freq_resolution": resolution,
                "periodogram": {"freq_hz": [1/30], "power": [.8]}}, is_stub=True)
        return fixture_result

    root = Path(__file__).resolve().parents[2]
    monkeypatch.setattr(controller, "run_lomb_scargle", runner(LOMB_SCARGLE))
    monkeypatch.setattr(controller, "run_pdm", runner(PDM))
    monkeypatch.setattr(controller, "reconcile", pending_reconciliation)
    monkeypatch.setattr(controller, "run_cnn", unavailable_cnn)
    for name, quality in (("00029A_2_3_R_20190322_DATA.txt", "suitable"),
                          ("00041B_2_3_R_20191118_DATA.txt", "limited")):
        upload = BytesIO((root / "data/sdlcd_raw" / name).read_bytes())
        upload.name = name
        monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: upload)
        app = AppTest.from_file(str(root / "app/main.py")).run(timeout=20)
        assert not app.exception and not app.error
        assert any(f"Observation quality: {quality}" in item.value for item in app.markdown)
        candidates = [item.value for item in app.table if "Apparent period (s)" in item.value]
        assert {table["Estimator"].iloc[0] for table in candidates} == {LOMB_SCARGLE, PDM}
