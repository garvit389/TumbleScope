from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core.config import config_hash
from pipeline import cache, controller
from scripts import run_demo_cache as demo


def _scientific(result):
    body = result.to_dict()
    body.pop("timings_s")
    for estimate in body["estimators"].values():
        estimate["meta"].pop("runtime_s")
    if body["verification"]:
        body["verification"].pop("runtime_s")
    return body


def test_synthetic_round_trip_without_source_or_inference(tmp_path, isolated_runs,
                                                         stub_config, stub_observation_stages, monkeypatch):
    root = tmp_path / "portable"
    source = tmp_path / demo.SYNTHETIC.name
    source.write_bytes(demo.SYNTHETIC.read_bytes())
    demo._publish(root, [(source, None)], "synthetic", stub_config)
    before, snapshot = cache._read_demo_index(root, stub_config, "synthetic")[source.name]
    demo._publish(root, [(source, None)], "synthetic", stub_config)
    after, _ = cache._read_demo_index(root, stub_config, "synthetic")[source.name]
    assert _scientific(before) == _scientific(after)
    assert snapshot["input_sha256"] == sha256(source.read_bytes()).hexdigest()
    assert snapshot["config_hash"] == config_hash(stub_config)
    source.unlink()

    def forbidden(*args, **kwargs):
        raise AssertionError("cached demo ran inference")

    monkeypatch.setattr(controller, "analyze", forbidden)
    monkeypatch.setattr(cache, "PORTABLE_DEMOS", root)
    monkeypatch.setattr(cache, "LOCAL_DEMOS", tmp_path / "absent")
    from core import config

    monkeypatch.setattr(config, "load_config", lambda: stub_config)
    app = AppTest.from_file(str(demo.ROOT / "app/main.py")).run()
    app.radio[0].set_value("Demo").run()
    assert not app.exception and not app.error
    assert any("Precomputed · Synthetic" in item.value for item in app.info)
    assert any("Stub result" in item.value for item in app.warning)
    assert app.get("download_button")


def test_cache_checks_integrity_paths_and_configuration(tmp_path, isolated_runs,
                                                       stub_config, stub_observation_stages):
    root = tmp_path / "portable"
    demo._publish(root, [(demo.SYNTHETIC, None)], "synthetic", stub_config)
    with pytest.raises(ValueError, match="configuration mismatch"):
        cache._read_demo_index(root, replace(stub_config, verdict={"alias_risk_max": .3}), "synthetic")
    index = cache._read_json((root / "index.json").read_bytes())
    entry = index["entries"][0]
    original = dict(entry)
    entry["path"] = "../sample.json"
    (root / "index.json").write_bytes(cache._encoded(index))
    with pytest.raises(ValueError, match="payload path"):
        cache._read_demo_index(root, stub_config, "synthetic")
    entry.update(original)
    (root / "index.json").write_bytes(cache._encoded(index))
    payload = root / entry["path"]
    payload.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        cache._read_demo_index(root, stub_config, "synthetic")
    payload.unlink()
    with pytest.raises(FileNotFoundError):
        cache._read_demo_index(root, stub_config, "synthetic")


def test_interrupted_build_keeps_existing_index(tmp_path, isolated_runs,
                                                stub_config, stub_observation_stages):
    root = tmp_path / "portable"
    demo._publish(root, [(demo.SYNTHETIC, None)], "synthetic", stub_config)
    original = (root / "index.json").read_bytes()
    with pytest.raises(FileNotFoundError):
        demo._publish(root, [(demo.SYNTHETIC, None), (tmp_path / "missing.csv", None)], "synthetic", stub_config)
    with pytest.raises(ValueError, match="Input checksum mismatch"):
        demo._publish(root, [(demo.SYNTHETIC, "0" * 64)], "synthetic", stub_config)
    assert (root / "index.json").read_bytes() == original
    assert cache._read_demo_index(root, stub_config, "synthetic")


def test_real_inputs_use_only_basename_and_manifest_checksum(tmp_path, capsys):
    source = tmp_path / demo.REAL_NAMES[0]
    source.write_bytes(b"test bytes")
    (tmp_path / "download_manifest.json").write_bytes(cache._encoded({
        "records": [{"filename": source.name, "sha256": sha256(source.read_bytes()).hexdigest(),
                     "reference_period_s": 42.}],
    }))
    assert demo._real_inputs(tmp_path) == [(source, sha256(source.read_bytes()).hexdigest())]
    assert demo.REAL_NAMES[1] in capsys.readouterr().err
