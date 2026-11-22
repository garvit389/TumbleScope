import json
import zipfile

import numpy as np
import pytest
import torch

from core.config import load_config
from ml import bundle
from ml.model import PeriodCNN
from ml.train import baseline, centers, read_checkpoint, save_checkpoint, score


def test_validation_scoring_and_train_only_baseline():
    bounds = [2., 128.]
    h = np.log(64.)/128
    truth = np.array([8.,8.,8.,8.,8.,0.,0.])
    predicted = np.array([8.,8*np.exp(h),16.,4.,0.,0.,8.])
    metrics = score(truth, predicted, bounds)
    assert metrics["coverage"] == 4/5 and metrics["tolerance_accuracy"] == 2/5
    assert metrics["harmonic_error_rate"] == 2/5
    assert metrics["half_period_count"] == metrics["double_period_count"] == 1
    assert metrics["nrp_precision"] == metrics["nrp_recall"] == .5
    assert score(truth, 0*predicted, bounds)["median_abs_log_error"] is None
    assert score(truth[:1], np.array([8*np.exp(h+1e-9)]), bounds)["tolerance_accuracy"] == 0
    result = baseline({"periods": np.array([0.,centers(bounds)[10]])}, {"periods": truth}, bounds)
    assert result["period_s"] == centers(bounds)[10] and result["nrp_precision"] is None


def test_checkpoint_roundtrip_and_rejection(tmp_path):
    cfg = {"cnn": load_config().cnn}
    state = {"model": PeriodCNN(cfg["cnn"]).state_dict(), "config": cfg, "dataset_id": "fixture",
             "epoch": 1, "history":[{}], "launch":{"optimizer_updated": True}}
    path = tmp_path / "epoch-0001.pt"
    checksum = save_checkpoint(path, state)
    loaded, verified = read_checkpoint(path, cfg, "fixture")
    assert verified == checksum
    model = PeriodCNN(cfg["cnn"]).eval()
    model.load_state_dict(loaded["model"], strict=True)
    with torch.no_grad():
        output = model(torch.zeros(1,3,1024))
    assert output.shape == (1,129) and torch.isfinite(output).all()
    with pytest.raises(ValueError, match="incompatible"):
        read_checkpoint(path, cfg, "different")
    path.with_suffix(".sha256").write_text("incorrect")
    with pytest.raises(ValueError, match="checksum"):
        read_checkpoint(path, cfg, "fixture")


def test_cpu_acceptance_uses_shared_inference_path(tmp_path, monkeypatch):
    from unittest.mock import Mock
    from core.config import resolve_period_bounds
    from ml import features, train

    local = load_config()
    cfg = {"cnn": local.cnn, "bounds": list(resolve_period_bounds(local.period_bounds)),
           "counts": bundle.COUNTS}
    development = {"periods": np.array([0., 8.])}
    monkeypatch.setattr(train, "verify_bundle", lambda path: (cfg, {"train": development,
                                        "val": development}, "cpu-unit-fixture"))
    spy = Mock(wraps=features.build_cnn_windows)
    monkeypatch.setattr(features, "build_cnn_windows", spy)
    path = tmp_path / "test.pt"
    save_checkpoint(path, {"model": PeriodCNN(cfg["cnn"]).state_dict(), "epoch": 1,
                    "config": cfg, "dataset_id": "cpu-unit-fixture", "history": [{"train_loss": 1.}],
                    "launch": {"optimizer_updated": True}})
    train.accept_download(None, path, tmp_path/"report.json")
    spy.assert_called_once()
    result = json.loads((tmp_path/"report.json").read_bytes())["cpu_acceptance"]
    assert result["shape"] == [1,129] and result["strict_load"] and result["finite"]


def test_export_refuses_incompatible_features_or_bounds(tmp_path):
    from core.config import resolve_period_bounds
    cfg = load_config()
    source = tmp_path / "train.npz"
    good = {k: cfg.cnn[k] for k in bundle.FEATURE_KEYS}
    bounds = list(resolve_period_bounds(cfg.period_bounds))
    for settings, limits in ((dict(good, y_clip=7), bounds), (good, [1.,2.])):
        np.savez(source, **{key:[] for key in bundle.FIELDS}, settings=json.dumps(settings),
                 bounds=limits)
        (tmp_path/"manifest.json").write_text(json.dumps({
            "artifacts":{"train.npz":bundle.digest(source.read_bytes())}}))
        with pytest.raises(ValueError, match="incompatible"):
            bundle.export_bundle(tmp_path, tmp_path/"bundle.zip")
        assert not (tmp_path/"bundle.zip").exists()


@pytest.mark.parametrize("extra", [None, "test.npz", "../unexpected"])
def test_bundle_rejects_malformed_archives(tmp_path, extra):
    path = tmp_path/"data.zip"
    with zipfile.ZipFile(path, "w") as out:
        out.writestr("manifest.json", "{}")
        for name in sorted(bundle.PAYLOAD if extra else bundle.PAYLOAD - {"train.npz"}):
            out.writestr(name, "")
        if extra:
            out.writestr(extra, "")
    with pytest.raises(ValueError, match="member"):
        bundle.verify_bundle(path)
