import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from classical_analysis._smoke_cli import run_smoke
from core.schemas import EstimatorResult, LOMB_SCARGLE


ROOT = Path(__file__).resolve().parents[2]
MODULES = ("lomb_scargle", "pdm", "window")


def invoke(module, path):
    return subprocess.run(
        [sys.executable, "-B", "-m", f"classical_analysis.{module}", str(path)],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )


@pytest.fixture
def sdlcd_file(tmp_path):
    path = tmp_path / "sample_lightcurve_DATA.txt"
    times = np.arange(300, dtype=float) * 2
    magnitudes = 12 + 0.15 * np.sin(2 * np.pi * times / 37)
    rows = np.column_stack((times, np.ones_like(times), np.zeros_like(times),
                            np.zeros_like(times), magnitudes, np.full_like(times, 0.01)))
    np.savetxt(path, rows, header="Synthetic SDLCD smoke input")
    return path


@pytest.mark.parametrize("module", MODULES)
def test_module_smoke_outputs_json(module, sdlcd_file):
    process = invoke(module, sdlcd_file)
    assert process.returncode == 0, process.stderr
    assert not process.stderr
    report = json.loads(process.stdout)
    assert report["status"] == "ok"
    assert report["obs_id"] == sdlcd_file.stem
    if module == "window":
        assert report["window_grid_points"] > 1
        assert report["freq_resolution"] > 0
        offsets = sorted(peak["freq_hz"] for peak in report["peaks"])
        assert offsets == pytest.approx([0.5, 1.0, 1.5], abs=report["freq_resolution"])
        assert all(peak["height"] > 0.95 for peak in report["peaks"])
    else:
        assert report["method"] == module and report["is_stub"] is False
        assert report["error"] is None
        best = report["candidates"][0]
        assert best["period_s"] == pytest.approx(37, rel=0.01)
        assert best["support"] == 1
        assert best["meta"]["freq_width_hz"] > 0
        assert report["meta"]["runtime_s"] >= 0
        assert report["meta"]["periodogram_points"] > 1


@pytest.mark.parametrize("module", MODULES)
@pytest.mark.parametrize("kind", ["missing", "malformed"])
def test_module_smoke_rejects_invalid_input(module, kind, tmp_path):
    path = tmp_path / "bad_DATA.txt"
    if kind == "malformed":
        path.write_text("0 1 2\n", encoding="utf-8")
    process = invoke(module, path)
    assert process.returncode != 0
    assert not process.stdout
    assert "Could not read" in process.stderr and path.name in process.stderr
    assert "Traceback" not in process.stderr


@pytest.mark.parametrize(("message", "expected"), [
    ("unexpected failure for test", "unexpected failure for test"),
    ("", "RuntimeError"),
])
def test_smoke_unexpected_failure_uses_stderr(sdlcd_file, monkeypatch, capsys, message, expected):
    monkeypatch.setattr(sys, "argv", ["smoke", str(sdlcd_file)])

    def fail(prepared, cfg):
        raise RuntimeError(message)

    assert run_smoke(fail, "Test unexpected failure") == 1
    captured = capsys.readouterr()
    assert not captured.out
    assert expected in captured.err


def test_smoke_estimator_failure_keeps_json(sdlcd_file, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["smoke", str(sdlcd_file)])

    def fail(prepared, cfg):
        return EstimatorResult(LOMB_SCARGLE, "failed", [], "numerical failure", {"runtime_s": 0.0})

    assert run_smoke(fail, "Test estimator failure") == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out)["status"] == "failed"
    assert "numerical failure" in captured.err
