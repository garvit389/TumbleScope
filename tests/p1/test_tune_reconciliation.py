from dataclasses import replace
import json
from unittest.mock import Mock

import pytest

from classical_analysis.window import compute_window
from core.config import config_hash, load_config, resolve_period_bounds
from core.schemas import ESTIMATOR_METHODS, EstimatorResult, NOT_RUN
from evaluation import run_eval, tune_reconciliation as tuning
from pipeline import controller
from preprocessing.prepare import prepare
from reconciliation.reconcile import reconcile


@pytest.fixture
def replay_fixture(observation, mock_cnn):
    cfg = load_config()
    prep = prepare(observation, cfg.preprocess)
    low, high = resolve_period_bounds(cfg.period_bounds)
    bounds = dict(period_min_s=low, period_max_s=high)
    window = compute_window(prep, {**cfg.classical, **bounds})
    settings = {**cfg.reconcile, **bounds, "top_k": 3, "window_result": window}
    estimators = [mock_cnn if method == mock_cnn.method else EstimatorResult(
        method, "skipped", [], NOT_RUN, {"runtime_s": 0.0}
    ) for method in ESTIMATOR_METHODS]
    result = reconcile(prep, estimators, settings)
    snapshot = dict(prepared=prep.to_dict(), settings=settings,
                    estimators=[tuning._serialized_estimator(item) for item in estimators],
                    original=result.to_dict())
    case = run_eval.Case(observation.obs_id, "evaluation-only-object", 8.0, None,
                         "fixture", "fixture-sha", observation)
    record = dict(obs_id=case.obs_id, source_sha256=case.source_sha256, snapshot=snapshot)
    common = dict(split="val", quality="limited", p_no_reliable="", stage_errors="{}",
                  verification_complete="False")
    rows = {name: {**common, "period_s": "", "candidate_periods_s": "[]",
                   "ranking_score": "", "status": "skipped"} for name in run_eval.CONFIGURATIONS}
    for name, method in run_eval.METHODS.items():
        periods = [item.period_s for est in estimators if est.method == method for item in est.candidates]
        rows[name]["candidate_periods_s"] = json.dumps(periods)
    fused = tuning._fused(result, {"status": "abstained"})
    rows["cnn_classical"].update({key: "" if value is None else json.dumps(value)
                                  if isinstance(value, list) else str(value) for key, value in fused.items()})
    rows["complete"]["status"] = "abstained"
    baseline = tuning.Baseline("real-val", cfg, [case], {case.obs_id: rows}, [], run_eval.DEFAULTS,
                               {"config_hash": config_hash(cfg)})
    return baseline, record


def test_replay_matches_full_cluster_result_and_never_runs_controller(replay_fixture, monkeypatch):
    baseline, record = replay_fixture
    analyze = Mock(side_effect=AssertionError("inference invoked during replay"))
    monkeypatch.setattr(controller, "analyze", analyze)
    rows, traces = tuning.verify_baseline(baseline, [record])
    analyze.assert_not_called()
    assert traces[0]["shortlist_ids"] == record["snapshot"]["original"]["shortlist"]
    assert sum(len(row["raw_members"]) for row in traces[0]["clusters"]) == 3
    for cluster in traces[0]["clusters"]:
        assert sum(member["representative"] for member in cluster["raw_members"]) == 1
        assert cluster["truncated"] == (cluster["stage2"]["passed"] and cluster["shortlist_rank"] is None)
    assert tuning._report(rows)["baseline_operations"] == {"incomplete_verification": 1}
    assert tuning._report(rows)["baseline_supported_claim_recall"]["denominator"] == 1


def test_replay_detects_baseline_change_and_blocks_stage2_overrides(replay_fixture):
    baseline, record = replay_fixture
    baseline.rows[record["obs_id"]]["cnn_classical"]["candidate_periods_s"] = "[19.0]"
    with pytest.raises(ValueError, match="ordered shortlist"):
        tuning.verify_baseline(baseline, [record])
    with pytest.raises(ValueError, match="only change"):
        tuning.replay(baseline, [record], {"min_cycles": 9})


def test_cache_bytes_deterministic_and_reference_labels_stay_out(replay_fixture, monkeypatch, tmp_path):
    baseline, record = replay_fixture
    snapshot = record["snapshot"]

    def capture(observation, config):
        assert observation is baseline.cases[0].observation and config is baseline.cfg
        assert "reference_period_s" not in observation.meta
        return snapshot

    monkeypatch.setattr(tuning, "capture_case", capture)
    first = tuning.make_cache(baseline)
    changed_truth = replace(baseline, cases=[replace(baseline.cases[0], period_s=19.0)])
    second = tuning.make_cache(changed_truth)
    assert tuning._canonical(first) == tuning._canonical(second)
    path = tmp_path / "real-val.json"
    tuning._write_new(path, first)
    assert tuning.load_cache(path, baseline) == [record]


def test_cache_rejects_wrong_split_and_broken_integrity(replay_fixture, monkeypatch, tmp_path):
    baseline, record = replay_fixture
    monkeypatch.setattr(tuning, "capture_case", lambda *args: record["snapshot"])
    envelope = tuning.make_cache(baseline)
    envelope["payload"]["dataset"] = "real-test"
    envelope["sha256"] = tuning._hash(envelope["payload"])
    path = tmp_path / "cache.json"
    tuning._write_new(path, envelope)
    with pytest.raises(ValueError, match="provenance"):
        tuning.load_cache(path, baseline)
    envelope["payload"]["dataset"] = "real-val"
    envelope["payload"]["records"][0]["obs_id"] = "sealed-test"
    envelope["sha256"] = tuning._hash(envelope["payload"])
    path = tmp_path / "other.json"
    tuning._write_new(path, envelope)
    with pytest.raises(ValueError, match="approved baseline cohort"):
        tuning.load_cache(path, baseline)
    envelope["sha256"] = "invalid"
    path = tmp_path / "tampered.json"
    tuning._write_new(path, envelope)
    with pytest.raises(ValueError, match="integrity"):
        tuning.load_cache(path, baseline)


def test_tuning_rejects_prefix_and_real_test_cli_before_reading_data(replay_fixture, tmp_path):
    baseline, record = replay_fixture
    with pytest.raises(ValueError, match="complete Day 4"):
        tuning.tune({"real-val": baseline}, {"real-val": [record]})
    with pytest.raises(SystemExit, match="2"):
        tuning.main(["capture", "--dataset", "real-test", "--baseline", str(tmp_path),
                     "--cache", str(tmp_path)])


def test_safeguards_use_each_cohort_and_nrp_cap():
    def stats(top, recall, n, nrp=0):
        return dict(top1={"numerator": top, "denominator": n},
                    shortlist_recall_at_3={"numerator": recall, "denominator": n}, nrp_emissions=nrp)

    baseline = {"real-val": stats(6, 12, 18), "synthetic-dev": stats(80, 121, 140, 40)}
    trial = {"real-val": stats(7, 12, 18), "synthetic-dev": stats(91, 121, 140, 40)}
    assert tuning._safeguards(trial, baseline)
    assert float(tuning._objective(trial)) == pytest.approx((7 / 18 + 91 / 140) / 2)
    for bad in (stats(79, 122, 140, 40), stats(90, 120, 140, 40), stats(90, 122, 140, 41)):
        assert not tuning._safeguards({**trial, "synthetic-dev": bad}, baseline)


@pytest.mark.parametrize(("status", "verification", "errors", "expected"), [
    ("abstained", "True", "{}", "completed_abstention"),
    ("abstained", "", "{}", "completed_abstention"),
    ("abstained", "False", "{}", "incomplete_verification"),
    ("execution_error", "True", '{"verdict":"blocked"}', "verdict_execution_error"),
])
def test_operational_reporting(status, verification, errors, expected):
    assert tuning._operation({"status": status, "verification_complete": verification,
                              "stage_errors": errors}) == expected


def test_full_metric_gate_detects_changed_summary(replay_fixture, monkeypatch):
    baseline, record = replay_fixture
    rows, _ = tuning.verify_baseline(baseline, [record])
    baseline.summary[:] = run_eval.summarize(rows, baseline.options)
    monkeypatch.setitem(tuning.COUNTS, "real-val", 1)
    assert tuning.verify_baseline(baseline, [record])[0] == rows
    baseline.summary[0]["n"] += 1
    with pytest.raises(ValueError, match="aggregate baseline metrics differ"):
        tuning.verify_baseline(baseline, [record])
