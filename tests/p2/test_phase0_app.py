from pathlib import Path

from streamlit.testing.v1 import AppTest


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
    entry_point = Path(__file__).resolve().parents[2] / "app" / "main.py"
    demo = AppTest.from_file(str(entry_point)).run(timeout=20)
    assert not demo.exception and not demo.error
    assert any("Inconclusive" == heading.value for heading in demo.subheader)
    assert (tmp_path / "run_config.json").is_file()
