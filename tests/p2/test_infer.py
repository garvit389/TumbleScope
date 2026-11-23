import json

import numpy as np
import pytest
import torch

from core.config import load_config, resolve_period_bounds
from core.schemas import NO_CHECKPOINT, TOO_FEW_VALID_SAMPLES
from ml import features, infer
from ml.bundle import FEATURE_KEYS, digest
from ml.calibrate import _curve_probabilities, _fit_temperature, _metrics, _select_threshold, _split_groups


def _config():
    config = load_config()
    low, high = resolve_period_bounds(config.period_bounds)
    return {**config.cnn, "period_min_s": low, "period_max_s": high}


def test_checkpoint_outcomes(prepared, monkeypatch, tmp_path):
    cfg = _config()
    monkeypatch.setattr(infer, "_CHECKPOINT", tmp_path / "absent.pt")
    result = infer.run_cnn(prepared, cfg)
    result.validate()
    assert (result.status, result.error, result.candidates) == ("skipped", NO_CHECKPOINT, [])
    checkpoint = tmp_path / "test.pt"
    monkeypatch.setattr(infer, "_CHECKPOINT", checkpoint)
    checkpoint.write_bytes(b"not a checkpoint")
    result = infer.run_cnn(prepared, cfg)
    assert result.status == "failed" and "sha256" in result.error
    checkpoint.with_suffix(".sha256").write_text(digest(checkpoint.read_bytes()))
    result = infer.run_cnn(prepared, cfg)
    result.validate()
    assert result.status == "failed" and result.candidates == [] and not result.is_stub


def test_weighted_probabilities_and_tv():
    p = np.full((2, 129), .001)
    p[0, 4], p[0, 128] = .773, .1
    p[1, 23], p[1, 128] = .573, .3
    combined, tv = infer._aggregate(np.log(p), [3, 1], 1.)
    np.testing.assert_allclose(combined, (3*p[0] + p[1])/4)
    conditional = p[:, :128] / p[:, :128].sum(axis=1, keepdims=True)
    reference = combined[:128] / combined[:128].sum()
    assert tv == pytest.approx(np.abs(conditional - reference).sum() / 4)
    assert infer._aggregate(np.log(p[[0, 0]]), [3, 1], 1.)[1] == pytest.approx(0.)
    logits = np.zeros((2, 129))
    logits[:, 128] = [10000., 10001.]
    probability, disagreement = infer._aggregate(logits, [3, 1], 1.)
    assert probability[128] == pytest.approx(1.)
    assert disagreement == pytest.approx(0.)


def test_period_peaks_and_abstention(prepared, monkeypatch):
    cfg = _config()
    monkeypatch.setattr(infer, "_load_model", lambda *args: (None, "id"))
    combined = np.zeros(129)
    combined[[0, 1, 42, 127, 128]] = [.1, .1, .4, .2, .2]
    monkeypatch.setattr(infer, "_aggregate", lambda *args: (combined, .3))
    monkeypatch.setattr(infer, "_window_logits", lambda *args: torch.zeros((1, 129)))
    monkeypatch.setattr(infer, "_calibration", lambda *args: (1., .2))
    result = infer.run_cnn(prepared, cfg)
    result.validate()
    assert result.status == "ok" and result.meta["abstained"] and result.candidates == []
    monkeypatch.setattr(infer, "_calibration", lambda *args: (1., .21))
    result = infer.run_cnn(prepared, cfg)
    result.validate()
    expected = infer.centers((cfg["period_min_s"], cfg["period_max_s"]))
    assert [candidate.period_s for candidate in result.candidates] == list(expected[[42, 127, 0]])
    assert [candidate.support for candidate in result.candidates] == [1., .5, .25]
    assert result.meta["experimental"] and result.meta["window_disagreement"] == .3


def test_window_cap_and_empty(prepared, monkeypatch):
    cfg = _config()
    assert not infer._was_subsampled(1024 + 8*512, cfg)
    assert infer._was_subsampled(1024 + 8*512 + 1, cfg)
    monkeypatch.setattr(infer, "_load_model", lambda *args: (None, "id"))
    monkeypatch.setattr(infer, "_calibration", lambda *args: (1., .5))
    monkeypatch.setattr(features, "build_cnn_windows", lambda *args: np.empty((0, 3, 1024)))
    result = infer.run_cnn(prepared, cfg)
    result.validate()
    assert (result.status, result.error) == ("skipped", TOO_FEW_VALID_SAMPLES)


def test_temperature_metrics_and_split():
    groups = np.array([44, 33, 22, 11])
    hard = np.array([0, 0, 128, 128])
    selection, check = _split_groups(groups, hard, 17)
    assert sorted(np.r_[selection, check]) == [0, 1, 2, 3]
    np.testing.assert_array_equal(_split_groups(groups, hard, 17)[0], selection)
    assert hard[selection].tolist() == hard[check].tolist() == [0, 128]
    probabilities = np.zeros((2, 129))
    probabilities[:, [0, 128]] = [[.75, .25], [.6, .4]]
    metrics = _metrics(probabilities, np.array([0, 128]))
    assert metrics["ece"] == pytest.approx((.25 + .6) / 2)
    assert metrics["brier"] == pytest.approx((2*.25**2 + 2*.6**2) / 2)
    logits = np.zeros((4, 129))
    offsets = np.arange(5)
    counts = np.ones(4)
    assert _fit_temperature(logits, counts, offsets, np.eye(129)[hard], selection) == 1.
    np.testing.assert_allclose(_curve_probabilities(logits, counts, offsets, 1.), 1/129)
    cutoff, _ = _select_threshold(np.array([.1, .2, .2, .3]), np.array([0, 0, 1, 1], bool))
    assert cutoff == .3
    cutoff, _ = _select_threshold(np.array([.4, .4]), np.array([0, 1], bool))
    assert cutoff == 1.


@pytest.mark.parametrize("key", ["checkpoint_sha256", "bounds", "encoder"])
def test_stale_calibration_rejected(prepared, monkeypatch, tmp_path, key):
    cfg = _config()
    path = tmp_path / "calibration.json"
    report = {"checkpoint_sha256": "id", "bounds": [cfg["period_min_s"], cfg["period_max_s"]],
              "encoder": {name: cfg[name] for name in FEATURE_KEYS},
              "abstain_threshold": cfg["abstain_threshold"], "temperature": 1.}
    report[key] = None
    path.write_text(json.dumps(report))
    monkeypatch.setattr(infer, "_CALIBRATION", path)
    monkeypatch.setattr(infer, "_load_model", lambda *args: (None, "id"))
    result = infer.run_cnn(prepared, cfg)
    assert (result.status, result.error) == ("failed", "CNN calibration identity mismatch")
