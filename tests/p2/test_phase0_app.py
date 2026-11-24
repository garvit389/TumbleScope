from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest
from tests.p2.test_controller import pending_reconciliation, unavailable_cnn
from tests.p2.test_verdict import case


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


def _synthetic_demo(monkeypatch, entry_point):
    from core.config import load_config
    from pipeline import cache, controller
    from scripts.run_demo_cache import SYNTHETIC, _publish

    root = controller._OUTPUTS / "demo"
    monkeypatch.setattr(cache, "PORTABLE_DEMOS", root)
    monkeypatch.setattr(cache, "LOCAL_DEMOS", root / "absent")
    _publish(root, [(SYNTHETIC, None)], "synthetic", load_config())

    def forbidden(*args, **kwargs):
        raise AssertionError("Demo reran the controller")

    monkeypatch.setattr(controller, "analyze", forbidden)
    app = AppTest.from_file(str(entry_point)).run(timeout=20)
    return app.radio[0].set_value("Demo").run(timeout=20)


def test_bundled_demo_renders_stub_result(
    isolated_runs, monkeypatch, stub_config, stub_observation_stages,
):
    from core import config

    monkeypatch.setattr(config, "load_config", lambda: stub_config)
    entry_point = Path(__file__).resolve().parents[2] / "app" / "main.py"
    demo = _synthetic_demo(monkeypatch, entry_point)
    assert not demo.exception
    assert not demo.error
    assert any("Stub result" in warning.value for warning in demo.warning)
    assert any("Inconclusive" == heading.value for heading in demo.subheader)
    assert (isolated_runs / "run_config.json").is_file()


def test_demo_with_real_cp1_stages(tmp_path, monkeypatch):
    from pipeline import controller
    monkeypatch.setattr(controller, "_OUTPUTS", tmp_path)
    monkeypatch.setattr(controller, "reconcile", pending_reconciliation)
    monkeypatch.setattr(controller, "run_cnn", unavailable_cnn)
    entry_point = Path(__file__).resolve().parents[2] / "app" / "main.py"
    demo = _synthetic_demo(monkeypatch, entry_point)
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
        assert any(f"Observation quality: {quality}" in item.value for item in app.markdown)
        assert not app.subheader
        app.button[0].click().run(timeout=20)
        assert not app.exception and not app.error
        assert any(f"Observation quality: {quality}" in item.value for item in app.markdown)
        candidates = [item.value for item in app.table if {"Estimator", "Apparent (synodic) period (s)"} <= set(item.value.columns)]
        assert {table["Estimator"].iloc[0] for table in candidates} == {LOMB_SCARGLE, PDM}


@pytest.mark.parametrize(("name", "raw"), [
    ("upload.csv", b"t_s,flux\n0,1\n1,2\n2,1\n"),
    ("upload.txt", b"t_s flux\n0 1\n1 2\n2 1\n"),
])
def test_generic_upload_progress_fallback_and_export(monkeypatch, isolated_runs, name, raw):
    from io import BytesIO
    from unittest.mock import Mock

    import streamlit as st
    from app import plots
    from core.schemas import AnalysisResult
    from pipeline import cache, controller

    upload = BytesIO(raw)
    upload.name = name
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: upload)
    monkeypatch.delattr(plots, "fig_quality")
    monkeypatch.setattr(cache, "PORTABLE_DEMOS", isolated_runs / "missing")
    monkeypatch.setattr(cache, "LOCAL_DEMOS", isolated_runs / "missing")
    run = Mock(wraps=controller.analyze)
    monkeypatch.setattr(controller, "analyze", run)
    captured = {}

    def capture(label, data, **kwargs):
        captured[label] = data

    monkeypatch.setattr(st, "download_button", capture)
    path = Path(__file__).resolve().parents[2] / "app/main.py"
    app = AppTest.from_file(str(path)).run()
    assert not app.exception and not app.error
    assert any("Observation quality: insufficient" in m.value for m in app.markdown)
    assert app.get("plotly_chart")
    run.assert_not_called()
    app.button[0].click().run()
    run.assert_called_once()
    assert not app.exception and not app.error
    assert app.get("progress")[0].proto.value == 100
    exported = AnalysisResult.from_dict(cache._read_json(captured["Download analysis JSON"]))
    assert exported.obs_id == "upload"
    assert exported.verdict.reason_codes == ["low_quality"]
    app.button[0].click().run()
    run.assert_called_once()
    assert any("Reused cached analysis" in c.value for c in app.caption)

    upload.seek(0)
    upload.truncate()
    upload.write(raw.replace(b"2", b"3"))
    captured.clear()
    app.run()
    assert not captured
    run.assert_called_once()


def test_supported_card_phase_fold_and_private_diagnostics(tmp_path, prepared, case):
    from dataclasses import replace
    import json

    from core.config import config_hash, load_config
    from core.schemas import AnalysisResult, QualityReport
    from pipeline import cache
    from verdict.decide import decide

    cfg = load_config()
    recon, estimators, verification = case
    private_text = r"Internal diagnostic at C:\private\unshared.txt"
    quality = QualityReport("suitable", {}, [private_text], [])
    verdict = decide(quality, estimators, recon, verification, cfg.verdict)
    assert verdict.label == "Supported"
    record = verification.per_candidate[0]
    verification = replace(verification, per_candidate=[
        replace(record, notes=[private_text]), verification.per_candidate[1],
    ])
    result = AnalysisResult(prepared.obs_id, prepared, quality, estimators, recon, verification, verdict,
                            {"run_config": private_text}, {}, "fast", config_hash(cfg))
    result.validate()
    exported = cache._public_result(result)
    assert "private" not in json.dumps(exported.to_dict())
    assert result.stage_errors["run_config"] == private_text
    assert exported.verdict.best_period_s == 8.
    assert exported.verification.per_candidate[0].heldout_error == record.heldout_error

    entry = tmp_path / "display_fixture.py"
    entry.write_text(
        "from app.main import _render\n"
        "import streamlit as st\n"
        "_render(*st.session_state['case'], show_preview=True)\n",
        encoding="utf-8",
    )
    app = AppTest.from_file(str(entry))
    app.session_state["case"] = (result, cache._input_snapshot(cfg, b"constructed fixture"))
    app.run()
    assert not app.exception and not app.error
    assert "Supported" in [heading.value for heading in app.subheader]
    assert not any("private" in item.value for item in app.caption)
    count = len(app.get("plotly_chart"))
    supported = next(option for option in app.selectbox[0].options if option.startswith("Supported"))
    app.selectbox[0].select(supported).run()
    assert not app.exception
    assert len(app.get("plotly_chart")) == count + 1
