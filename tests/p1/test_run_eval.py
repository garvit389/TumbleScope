from dataclasses import replace
import csv
import json
from unittest.mock import Mock

import numpy as np
import pytest

from core.config import config_hash
from core.schemas import CNN, LOMB_SCARGLE, PDM, Candidate, EstimatorResult
from evaluation import run_eval as runner
from pipeline import controller
from tests.verification_cases import with_candidate_fits


def case_for(obs, *, period=8.0, nrp=None):
    return runner.Case(obs.obs_id, "evaluation-only-object", period, nrp, "fixture", None, obs)


def estimator(method, period):
    candidate = Candidate(period, 1.0, 0.5, {"freq_width_hz": 0.01})
    meta = {"runtime_s": 0.0, "freq_resolution": 0.01,
            "periodogram": {"freq_hz": [], "power": []}}
    return EstimatorResult(method, "ok", [candidate], meta=meta, is_stub=True)


@pytest.fixture
def stub_chain(monkeypatch, observation, injected_config, mock_cnn, mock_recon):
    monkeypatch.setattr(controller, "run_lomb_scargle", Mock(return_value=estimator(LOMB_SCARGLE, 7.0)))
    monkeypatch.setattr(controller, "run_pdm", Mock(return_value=estimator(PDM, 9.0)))
    monkeypatch.setattr(controller, "run_cnn", Mock(return_value=mock_cnn))
    monkeypatch.setattr(controller, "reconcile", Mock(return_value=mock_recon))
    return observation, injected_config


def test_stub_controller_five_readouts_and_one_call(stub_chain):
    obs, cfg = stub_chain
    analyze = Mock(wraps=controller.analyze)
    [row] = runner.evaluate([case_for(obs)], cfg, dataset="real-val", analyze=analyze)
    analyze.assert_called_once_with(obs, cfg)
    assert list(row["configurations"]) == list(runner.CONFIGURATIONS)
    assert [item["period_s"] for item in row["configurations"].values()] == [7, 9, 8, 8, None]
    assert row["quality"] == "suitable"
    assert row["verdict"] == "Ambiguous"
    assert row["configurations"]["complete"]["status"] == "abstained"
    assert row["configurations"]["cnn_classical"]["candidate_periods_s"] == [8]


def test_pending_stage2_and_cnn_abstention(stub_chain, monkeypatch, mock_recon, mock_cnn):
    obs, cfg = stub_chain
    monkeypatch.setattr(controller, "run_cnn", Mock(return_value=replace(
        mock_cnn, candidates=[], meta={**mock_cnn.meta, "abstained": True},
    )))
    monkeypatch.setattr(controller, "reconcile", Mock(return_value=replace(
        mock_recon, shortlist=[], stage2_passed_any=False, notes=["stage2_not_evaluated"],
    )))
    [row] = runner.evaluate([case_for(obs)], cfg, dataset="real-val")
    assert row["cnn_abstained"] is True
    assert row["configurations"]["cnn_only"]["status"] == "abstained"
    assert row["configurations"]["ls_only"]["period_s"] == 7
    assert row["configurations"]["cnn_classical"]["status"] == "pending"
    assert row["configurations"]["complete"]["status"] == "pending"
    assert row["verdict"] is None


def test_verdict_failure_preserves_upstream_prediction(stub_chain, monkeypatch):
    obs, cfg = stub_chain
    original_verify = controller.verify_candidates

    def verify(*args, **kwargs):
        result = original_verify(*args, **kwargs)
        fits = [
            replace(item, status="ok", passed=True, fit_error=.1, heldout_error=.2 + index * .1,
                    best_params={"shape": "fixture", "geometry_errors": {"fixture": .2},
                                 "fit_space": "y_norm", "scale_a": 1., "offset_b": 0.})
            for index, item in enumerate(result.per_candidate)
        ]
        return with_candidate_fits(result, fits)

    base = controller.analyze(obs, cfg)
    monkeypatch.setattr(controller, "verify_candidates", verify)
    supported = replace(base.verdict, label="Supported", best_period_s=8.0,
                        best_cluster_id=1, reason_codes=[])
    monkeypatch.setattr(controller, "decide", Mock(return_value=supported))
    [completed] = runner.evaluate([case_for(obs)], cfg, dataset="real-val")
    assert completed["configurations"]["complete"]["status"] == "supported"
    assert completed["configurations"]["complete"]["period_s"] == 8
    monkeypatch.setattr(controller, "decide", Mock(side_effect=ValueError("fixture verdict error")))
    [row] = runner.evaluate([case_for(obs)], cfg, dataset="real-val")
    assert row["verification_complete"] is True
    assert row["stage_errors"]["verdict"] == "fixture verdict error"
    assert row["configurations"]["complete"]["status"] == "execution_error"
    assert row["configurations"]["complete"]["period_s"] is None
    assert row["configurations"]["cnn_classical"]["period_s"] == 8


@pytest.fixture
def real_inputs(tmp_path):
    reference = tmp_path / runner.DATASETS["real-val"][1]
    reference.parent.mkdir(parents=True)
    reference.write_text(
        "obs_id,object_id,period_s,split\n"
        "a_sealed,object-test,999,test\n"
        "b_good,object-val,8,val\n"
        "c_missing,object-missing,12,val\n"
        "d_bad,object-bad,14,val\n",
        encoding="utf-8",
    )
    raw = tmp_path / "data/sdlcd_raw"
    raw.mkdir()
    measurements = "\n".join(f"{time} 1 15 1 {5 + np.sin(time)} 0.1" for time in range(120))
    (raw / "b_good.txt").write_text("# Per[s]\n# 8\n" + measurements, encoding="utf-8")
    (raw / "a_sealed.txt").write_text("sealed", encoding="utf-8")
    (raw / "d_bad.txt").write_text("malformed", encoding="utf-8")
    return reference, raw


def test_real_split_filters_before_touching_raw_files(real_inputs, monkeypatch):
    reference, raw = real_inputs
    load = Mock(wraps=runner.load_observation)
    monkeypatch.setattr(runner, "load_observation", load)
    cases = list(runner._real_cases(reference, raw, None))
    assert [case.obs_id for case in cases] == ["b_good", "c_missing", "d_bad"]
    assert [case.input_status for case in cases] == ["ok", "missing", "failed"]
    assert [call.args[0] for call in load.call_args_list] == [
        str(raw / "b_good.txt"), str(raw / "d_bad.txt"),
    ]
    assert all(case.error for case in cases[1:])
    assert [case.obs_id for case in runner._real_cases(reference, raw, 1)] == ["b_good"]


def test_reference_rejects_cross_split_objects(real_inputs):
    reference, raw = real_inputs
    with reference.open("a", encoding="utf-8") as stream:
        stream.write("other,object-val,8,test\n")
    with pytest.raises(ValueError, match="both reference splits"):
        list(runner._real_cases(reference, raw, None))


@pytest.mark.parametrize("dataset", ["test", "real-test", "synthetic-test"])
def test_cli_has_no_test_split(tmp_path, dataset):
    with pytest.raises(SystemExit, match="2"):
        runner.main(["--dataset", dataset, "--output", str(tmp_path / "report")])


def test_synthetic_loader_checks_whole_shard_before_limit(tmp_path, injected_config):
    path = tmp_path / "holdout.npz"
    arrays = dict(groups=[1000001, 2000002], periods=[8, 0], offsets=[0, 3, 6],
                  t_s=[0, 1, 2, 0, 1, 2], brightness=[1, 2, 1, 1, 0, 1], bounds=[2, 20])
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="development groups only"):
        list(runner._synthetic_cases(path, injected_config, 1, "digest"))
    arrays["groups"] = [1000001, 1100002]
    arrays["offsets"] = [0, 3, 5]
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="offsets"):
        list(runner._synthetic_cases(path, injected_config, None, "digest"))
    arrays["offsets"] = [0, 3, 6]
    np.savez(path, **arrays)
    good, nrp = runner._synthetic_cases(path, injected_config, None, "digest")
    assert good.period_s == 8 and good.no_reliable_period is False
    assert nrp.period_s is None and nrp.no_reliable_period is True and nrp.observation.meta == {}
    assert nrp.obs_id == "1100002"
    assert nrp.observation.obs_id != nrp.obs_id
    arrays["t_s"][3] = np.inf
    np.savez(path, **arrays)
    _, malformed = runner._synthetic_cases(path, injected_config, None, "digest")
    assert malformed.input_status == "failed" and malformed.no_reliable_period is True
    path.write_bytes(b"not an npz")
    with pytest.raises(ValueError, match="cannot read"):
        list(runner._synthetic_cases(path, injected_config, None, "digest"))


def test_denominators_include_missing_rows_and_calibrate_only_clean_synthetic(stub_chain, monkeypatch, mock_cnn):
    obs, cfg = stub_chain
    monkeypatch.setattr(controller, "run_cnn", Mock(return_value=replace(
        mock_cnn, is_stub=False, meta={**mock_cnn.meta, "p_no_reliable": .25},
    )))
    missing = replace(case_for(obs, period=8, nrp=False), obs_id="missing", observation=None,
                      input_status="missing", error="missing file")
    rows = runner.evaluate([case_for(obs, nrp=False), missing, case_for(obs, period=None, nrp=True)],
                           cfg, dataset="synthetic-dev", variants=("modulation",))
    summaries = runner.summarize(rows, runner.DEFAULTS)
    clean = {item["configuration"]: item for item in summaries
             if item["quality"] == "all" and item["variant"] == "clean"}
    cnn = clean["cnn_only"]
    assert cnn["n"] == 3 and cnn["status_counts"] == {"predicted": 2, "missing": 1}
    assert cnn["tolerance_accuracy"]["overall"]["denominator"] == 2
    assert cnn["tolerance_accuracy"]["conditional"]["denominator"] == 1
    assert cnn["recall_at_k"]["denominator"] == 2
    assert cnn["brier"]["denominator"] == cnn["ece"]["denominator"] == 2
    assert cnn["brier"]["value"] == pytest.approx((.25 ** 2 + .75 ** 2) / 2)
    assert clean["complete"]["supported_metrics"]["coverage"]["denominator"] == 3
    assert clean["complete"]["supported_metrics"]["disposition_counts"]["missing"] == 1
    assert len([item for item in summaries if "brier" in item and item["variant"] != "clean"]) == 0


def test_stress_reproducibility_call_count_and_truth_isolation(stub_chain, real_inputs):
    _, cfg = stub_chain
    reference, raw = real_inputs
    [case] = runner._real_cases(reference, raw, 1)
    seen = []

    def analyze(observation, config):
        assert config is cfg
        assert not {"period_s", "reference_period_s", "split", "no_reliable_period"} & observation.meta.keys()
        seen.append(observation)
        # The controlled stub stages supply contract-valid outputs for the test.
        return controller.analyze(observation, config)

    variants = runner.STRESSES
    first = runner.evaluate([case], cfg, dataset="real-val", variants=variants, seed=81, analyze=analyze)
    first_inputs = seen.copy()
    seen.clear()
    second = runner.evaluate([replace(case, period_s=99.0)], cfg, dataset="real-val",
                             variants=variants, seed=81, analyze=analyze)
    assert len(first) == len(second) == len(seen) == len(first_inputs) == 6
    for before, after in zip(first_inputs, seen, strict=True):
        np.testing.assert_array_equal(before.t_s, after.t_s)
        np.testing.assert_array_equal(before.y, after.y)
    assert first[1]["reference_period_s"] == 8 and second[1]["reference_period_s"] == 99
    assert first[1]["variant_seed"] == second[1]["variant_seed"]
    assert first[1]["configurations"] == second[1]["configurations"]
    assert not np.array_equal(first_inputs[0].y, first_inputs[1].y)
    assert len(first_inputs[2].t_s) < len(first_inputs[0].t_s)


def test_impossible_transform_and_raised_analysis_are_retained(stub_chain):
    obs, cfg = stub_chain
    fixed = replace(obs, y=np.ones_like(obs.y))
    analyze = Mock(side_effect=RuntimeError("analysis failed"))
    clean, noisy = runner.evaluate([case_for(fixed)], cfg, dataset="real-val",
                                   variants=("noise",), analyze=analyze)
    analyze.assert_called_once_with(fixed, cfg)
    assert clean["input_status"] == "execution_error" and "analysis failed" in clean["error"]
    assert noisy["input_status"] == "skipped" and "nonzero brightness" in noisy["error"]
    assert all(item["period_s"] is None for row in (clean, noisy) for item in row["configurations"].values())


def test_cli_replay_uses_saved_config_and_rejects_changed_inputs(
    tmp_path, real_inputs, stub_chain, monkeypatch,
):
    _, cfg = stub_chain
    _, raw = real_inputs
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "load_config", Mock(return_value=cfg))
    initial, replay = tmp_path / "initial", tmp_path / "replay"
    runner.main(["--dataset", "real-val", "--limit", "1", "--stress", "noise", "--seed", "5",
                 "--output", str(initial)])
    manifest = initial / "manifest.json"
    saved = json.loads(manifest.read_text(encoding="utf-8"))
    assert saved["split"] == "val" and saved["n_observations"] == 1 and saved["options"]["seed"] == 5
    assert saved["config_hash"] == config_hash(cfg)
    assert set(path.name for path in initial.iterdir()) == {
        "runs.csv", "summary.csv", "summary.json", "risk_coverage.html", "calibration.html", "manifest.json",
    }
    monkeypatch.setattr(runner, "load_config", Mock(side_effect=AssertionError("used live config")))
    runner.main(["--replay", str(manifest), "--output", str(replay)])
    assert (initial / "summary.json").read_bytes() == (replay / "summary.json").read_bytes()

    def science(path):
        with (path / "runs.csv").open(encoding="utf-8", newline="") as stream:
            return [{key: value for key, value in row.items() if key != "stage_timings_s"}
                    for row in csv.DictReader(stream)]

    assert len(science(initial)) == 10 and science(initial) == science(replay)
    with (raw / "b_good.txt").open("a", encoding="utf-8") as stream:
        stream.write("\n# changed")
    with pytest.raises(SystemExit, match="2"):
        runner.main(["--replay", str(manifest), "--output", str(tmp_path / "changed")])
    assert not (tmp_path / "changed").exists()
