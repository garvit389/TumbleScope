import json
from dataclasses import replace
from pathlib import Path
from shutil import copytree

import numpy as np
import pytest

from core.config import load_config
from core.schemas import Cluster, Observation, ReconciliationResult
from ml.dataset.labels import template_repeat
from physics import eval_verification as evaluator, verify
from physics.simulator import simulate_lightcurve
from preprocessing.prepare import prepare
from scripts import prepare_verification_auc as cohort
from tests.p2.test_simulator import BOX


def _row(group, error, deltas):
    return {"group": group, "status": "ok", "heldout_error": error, "decoy_delta_e": deltas,
            "decoy_periods_s": list(range(len(deltas))), "notes": []}


@pytest.mark.parametrize(("delta", "value"), [(1., 1.), (-1., 0.), (0., .5)])
def test_auc_direction_and_ties(delta, value):
    summary = evaluator._auc_statistics([_row(1, 2., [delta])])
    assert summary["pooled_candidate_roc_auc"] == value
    assert summary["mean_within_curve_win_rate"] == value


def test_pooled_and_matched_accounting():
    rows = [_row(1, .5, [1., 1.]), _row(2, 10., [1.]),
            {"group": 3, "status": "failed", "notes": ["Excluded decoy 2 s (alias offset)", "timeout"]},
            _row(4, 1., [])]
    summary = evaluator._auc_statistics(rows)
    assert summary["pooled_candidate_roc_auc"] == pytest.approx(4 / 6)
    assert summary["mean_within_curve_win_rate"] == 1
    assert summary["attempted"] == 4 and summary["completed"] == 2
    assert summary["n_truth_scores"] == 2 and summary["n_decoy_scores"] == 3
    assert summary["failure_counts"] == {"timeout": 1, "no complete finite comparison": 1}
    assert summary["n_excluded_decoys"] == 1
    assert evaluator._auc_statistics(rows[2:])["pooled_candidate_roc_auc"] is None


def test_fixture_execution_uses_true_apparent_repeat_and_equal_fit_budget(monkeypatch):
    cfg = load_config()
    n = cfg.simulation["samples_per_cycle"]
    waveform = lambda t: np.sin(2 * np.pi * t / 8.)
    sample = np.arange(3 * n) * (16. / n)
    period = template_repeat(waveform(sample), 16., n, cfg.simulation["repeat_tolerance"],
                             cfg.simulation["flat_rtol"], wave=waveform)
    assert period == 8.
    times = np.sort(np.random.default_rng(cfg.simulation["seed"]).uniform(0, 300, 240))
    prep = prepare(Observation("1", times, waveform(times), None, "unknown"), cfg.preprocess)
    candidate = Cluster(1, period, {}, {}, 0., [], 1.,
                        {"passed": True, "cycles": prep.meta["baseline_s"] / period, "fail_reasons": []})
    recon = ReconciliationResult([candidate], [1], True, [])
    calls = []

    def score(prepared, proposed, policy, masks, deadline, *, sigma_obs):
        calls.append((proposed, policy["fit"], [mask.tolist() for mask in masks], sigma_obs, deadline))
        return (.1, .1 if proposed == period else .6,
                {"shape": "box", "geometry_errors": {"box": .1}, "fit_space": "y_norm",
                 "scale_a": 1., "offset_b": 0.}, np.zeros_like(prepared.t_s))

    monkeypatch.setattr(verify, "_score_period", score)
    result = evaluator.evaluate_auc_cases(cfg, [(prep, recon)], population="constructed test fixture")
    repeated = evaluator.evaluate_auc_cases(cfg, [(prep, recon)], population="constructed test fixture")
    assert result["rows"] == repeated["rows"]
    assert result["summary"]["pooled_candidate_roc_auc"] == 1.
    assert result["rows"][0]["decoy_periods_s"] == [period * ratio for ratio in cfg.verification["decoy_ratios"]]
    assert calls[0][0] == period
    assert all(call[1:] == calls[0][1:] for call in calls[1:3])
    assert calls[0][1] == cfg.verification["fit"]
    with pytest.raises(ValueError, match="Duplicate"):
        evaluator.evaluate_auc_cases(cfg, [(prep, recon), (prep, recon)], population="fixture")


def test_decoy_exclusions_follow_verifier():
    periods, notes = verify._eligible_decoys(
        10., [20.], {"freq_resolution": .001, "peaks": [{"freq_hz": .2}]},
        (2., 21.), [1., 2., .5, 3., 1.37, 1.37], 1.,
    )
    assert periods == pytest.approx([13.7])
    assert {note.rsplit("(", 1)[1].rstrip(")") for note in notes} == {
        "candidate frequency", "harmonic frequency", "alias offset", "outside period bounds", "duplicate period",
    }


def test_auc_real_fit_and_failure_on_constructed_curves():
    cfg = load_config()
    cfg = replace(cfg, verification={
        **cfg.verification, "fit": {
            **cfg.verification["fit"], "shapes": ["box"], "coarse_samples": 3,
            "restarts": 1, "popsize": 2, "maxiter": 2,
        },
    })
    times = np.sort(np.random.default_rng(cfg.simulation["seed"]).uniform(0, 250, 160))
    values = simulate_lightcurve(BOX, times)
    prep = prepare(Observation("1", times, values, None, "unknown"), cfg.preprocess)
    # The box fixture repeats twice per imposed phase cycle.
    period = BOX.phase_cycle_s / 2
    truth = Cluster(1, period, {}, {}, 0., [], 1.,
                    {"passed": True, "cycles": prep.meta["baseline_s"] / period, "fail_reasons": []})
    recon = ReconciliationResult([truth], [1], True, [])
    flat = prepare(Observation("2", times, np.ones(len(times)), None, "unknown"), cfg.preprocess)
    report = evaluator.evaluate_auc_cases(cfg, [(flat, recon), (prep, recon)], population="fixture only")
    assert report["summary"]["attempted"] == 2
    assert report["summary"]["completed"] == 1
    assert report["summary"]["failure_counts"] == {"zero observed MAD": 1}
    assert report["rows"][0]["group"] == 1
    assert report["rows"][0]["period_s"] == period


def test_both_auc_commands_refuse_without_opening_payloads(monkeypatch, tmp_path, capsys):
    original = Path.open
    missing = tmp_path / "missing"
    monkeypatch.setattr(evaluator, "COHORT", missing)
    allowed = {missing / "manifest.json", *evaluator.ROOT.glob("configs/*.yaml"),
               *(cohort.ROOT / name for name in cohort._INVENTORY),
               cohort.ROOT / cohort._HISTORICAL_FIT,
               *(cohort.ROOT / name for name in cohort._SOURCE_FILES)}
    target = tmp_path / "existing.json"
    target.write_text("untouched")

    def metadata_only(path, *args, **kwargs):
        assert path in allowed, f"Unexpected path opened: {path.name}"
        return original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", metadata_only)
        for args in (["--check-inputs"], ["--output", str(target)]):
            assert evaluator.main(["--auc", *args]) == 2
            assert json.loads(capsys.readouterr().out)["status"] == "blocked_no_eligible_cohort"
    assert target.read_text() == "untouched"


@pytest.fixture(scope="module")
def built_evaluation_allocation(tmp_path_factory):
    path = tmp_path_factory.mktemp("verification-allocation") / "cohort"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cohort, "PER_VARIANT", 1)
        # Exercise the real allocation logic without depending on local corpus files.
        patch.setattr(cohort, "_INVENTORY", ())
        cohort.publish(load_config(), path)
    return path


@pytest.fixture
def evaluation_allocation(monkeypatch, tmp_path, built_evaluation_allocation):
    monkeypatch.setattr(cohort, "PER_VARIANT", 1)
    monkeypatch.setattr(cohort, "_INVENTORY", ())
    path = tmp_path / "cohort"
    copytree(built_evaluation_allocation, path)
    cfg = load_config()
    manifest, _ = cohort.read_metadata(cfg, path)
    monkeypatch.setattr(evaluator, "COHORT", path)
    return cfg, path, manifest


def test_independent_generation_and_seeded_reproduction(evaluation_allocation, tmp_path):
    cfg, path, manifest = evaluation_allocation
    other = tmp_path / "rebuild"
    cohort.publish(cfg, other)
    assert cohort.read_metadata(cfg, other)[0] == manifest
    assert (path / "manifest.json").read_bytes() == (other / "manifest.json").read_bytes()
    assert (path / "curves.npz").read_bytes() == (other / "curves.npz").read_bytes()
    cases = list(evaluator._production_cases(cfg, manifest))
    assert [prep.obs_id for prep, _ in cases] == [str(cohort.GROUP_START), str(cohort.GROUP_START + 1)]
    assert [recon.clusters[0].period_s for _, recon in cases] == [
        row["apparent_period_s"] for row in manifest["rows"]
    ]
    assert all("apparent_period_s" not in prep.meta for prep, _ in cases)


def test_metadata_success_never_reads_observations(evaluation_allocation, monkeypatch, capsys):
    original = Path.open

    def metadata_only(path, *args, **kwargs):
        assert path.suffix != ".npz"
        assert path.name != "test_provenance.json"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", metadata_only)
    assert evaluator.main(["--auc", "--check-inputs"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "eligible_metadata"


@pytest.mark.parametrize("change", ["sealed", "split", "variant", "group", "seed", "params"])
def test_preflight_rejects_invalid_allocation_metadata(evaluation_allocation, change, capsys):
    _, path, manifest = evaluation_allocation
    design, row = manifest["design"], manifest["rows"][0]
    if change == "sealed":
        design["sealed"] = True
    elif change == "split":
        row["split"] = "val"
    elif change == "variant":
        row["variant"] = 1
    elif change == "group":
        row["group"] = 1_000_000
    elif change == "seed":
        row["draw_seed"] += 1
    else:
        row["params"]["dimensions"] = [.5, 3.2]
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert evaluator.main(["--auc", "--check-inputs"]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "blocked_no_eligible_cohort"


def test_preflight_checks_historical_variant_inventory(evaluation_allocation, monkeypatch, tmp_path):
    cfg, path, _ = evaluation_allocation
    previous = tmp_path / "historical.json"
    variants = {**cfg.simulation["variants"], "val": [cohort.VARIANTS[0]]}
    previous.write_text(json.dumps({"config": {"simulation": {"variants": variants}}}))
    monkeypatch.setattr(cohort, "_INVENTORY", (str(previous),))
    with pytest.raises(ValueError, match="overlaps an occupied"):
        cohort.read_metadata(cfg, path)


def test_corrupt_payload_cannot_publish_results(evaluation_allocation, tmp_path, capsys):
    _, path, _ = evaluation_allocation
    target = tmp_path / "output.json"
    (path / "curves.npz").write_bytes(b"invalid")
    assert evaluator.main(["--auc", "--check-inputs"]) == 0
    capsys.readouterr()
    assert evaluator.main(["--auc", "--output", str(target)]) == 2
    assert "checksum" in json.loads(capsys.readouterr().out)["reason"]
    assert not target.exists()


def test_in_bounds_tampered_truth_cannot_publish_results(evaluation_allocation, tmp_path, capsys):
    _, path, manifest = evaluation_allocation
    manifest["rows"][0]["apparent_period_s"] = manifest["design"]["period_bounds_s"][0]
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    target = tmp_path / "output.json"
    assert evaluator.main(["--auc", "--output", str(target)]) == 2
    assert "clean simulator repeat" in json.loads(capsys.readouterr().out)["reason"]
    assert not target.exists()


def test_production_excludes_truth_harmonics_and_shifted_sampling_aliases():
    period = 10.
    alias_offset = abs(1 / period - 1 / (period * 1.37))
    window = {"freq_resolution": .001, "peaks": [{"freq_hz": alias_offset}]}
    policy = {"decoy_exclusion_resolutions": 1., "decoy_ratios": [2., .5, 1.37, .61]}
    ratios, notes = evaluator._independent_decoys(period, window, policy)
    assert ratios == [.61]
    assert sum("truth harmonic" in note for note in notes) == 2
    assert sum("truth sampling alias" in note for note in notes) == 1
