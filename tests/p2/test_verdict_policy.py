from dataclasses import replace
import json

import numpy as np
import pytest

from core.config import load_config, resolve_period_bounds
from core.schemas import AnalysisResult, CNN, NO_CHECKPOINT, EstimatorResult, QualityReport
from pipeline import controller
from scripts import eval_verdict_policy as study
from tests.p2.test_verdict import case


def test_verified_development_shard():
    cfg = load_config()
    identity = json.loads(study.REPORT.read_bytes())["holdout"]
    raw = study.SHARD.read_bytes()
    bounds = resolve_period_bounds(cfg.period_bounds)
    curves = study._load_shard(raw, identity, bounds)
    assert len(curves) == 200
    assert sum(truth == 0 for _, truth in curves) == 60
    assert all(not obs.meta and obs.obs_id.startswith("1") for obs, _ in curves)
    with pytest.raises(ValueError, match="checksum"):
        study._load_shard(raw + b"bad", identity, bounds)
    with pytest.raises(ValueError, match="Invalid development shard"):
        study._load_shard(raw, {**identity, "groups": list(reversed(identity["groups"]))}, bounds)


def test_metric_denominators_and_period_scoring():
    bounds = (2., 20.)
    assert study._match(8., 8. * np.exp(np.log(10.) / 128), bounds) == "direct"
    assert study._match(8., 4., bounds) == "half"
    assert study._match(8., 16., bounds) == "double"
    assert study._match(8., 10., bounds) == "other"
    assert study._match(0., 10., bounds) is None
    rows = [
        {"group": i, "truth_s": truth, "prediction_s": prediction, "outcome": outcome,
         "match": study._match(truth, prediction, bounds)}
        for i, (truth, prediction, outcome) in enumerate([
            (8., 8., "Supported"), (8., 16., "Supported"), (0., 10., "Supported"),
            (8., None, "Ambiguous"), (0., None, "pending"), (0., None, "execution_error"),
        ])
    ]
    summary = study._summary(rows)
    assert summary["false_supported"] == {"numerator": 2, "denominator": 3, "value": 2 / 3}
    assert summary["false_supported_per_labelled_attempt"]["denominator"] == 6
    assert summary["coverage"]["value"] == .5
    assert summary["selective_accuracy"]["value"] == 1 / 3
    assert summary["abstained"] == 1
    assert study._summary(rows[3:])["false_supported"]["value"] is None


def test_workflow_passes_truth_only_to_metrics(monkeypatch, observation, isolated_runs,
                                              stub_config, stub_observation_stages):
    cfg = replace(stub_config, period_bounds={"status": "SET", "min_period_s": 2., "max_period_s": 20.},
                  verdict={"min_agreeing_estimators": 2, "limited_max_label": "Supported"})
    baseline = controller.analyze(observation, stub_config)
    baseline = replace(baseline, config_hash=study.config_hash(cfg))

    def analyze(obs, policy):
        assert obs is observation
        assert "truth" not in obs.meta
        return baseline

    monkeypatch.setattr(controller, "analyze", analyze)
    report = study.evaluate(cfg, [(observation, 8.)], {"seed": 17})
    assert report["planned"] == 1
    assert len(report["policies"]) == 4
    assert all(row["rows"][0]["truth_s"] == 8. for row in report["policies"])


def test_changed_policy_predictions_checked_with_fresh_controller_call(monkeypatch, observation, prepared, case):
    cfg = load_config()
    recon, estimates, verification = case
    estimates[CNN] = EstimatorResult(CNN, "skipped", [], NO_CHECKPOINT, {"runtime_s": 0.})
    quality = QualityReport("limited", {}, [], [])
    observation = replace(observation, meta={})
    calls = []

    def analyze(obs, policy):
        assert obs is observation and not obs.meta
        digest = study.config_hash(policy)
        calls.append(digest)
        verdict = controller.decide(quality, estimates, recon, verification, controller._decision_config(policy))
        return AnalysisResult(obs.obs_id, prepared, quality, estimates, recon, verification, verdict,
                              {}, {}, "fast", digest)

    monkeypatch.setattr(controller, "analyze", analyze)
    report = study.evaluate(cfg, [(observation, 8.)], {"seed": 17})
    assert len(calls) == 4 and len(set(calls)) == 4
    assert len(report["replay_checks"]) == 3
    assert [entry["rows"][0]["outcome"] for entry in report["policies"]] == [
        "Supported", "Ambiguous", "Ambiguous", "Ambiguous",
    ]
