from dataclasses import replace
import json
from threading import get_ident
from unittest.mock import Mock

import pytest

from core.schemas import (
    CNN, NO_CHECKPOINT, VERIFICATION_INCOMPLETE, AnalysisResult, Cluster, EstimatorResult,
    ReconciliationResult, assert_contract_equal,
)
from pipeline import controller
from pipeline import cache


def unavailable_cnn(*args):
    return EstimatorResult(CNN, "skipped", [], NO_CHECKPOINT, {"runtime_s": 0.}, is_stub=True)


def pending_reconciliation(prep, estimators, cfg):
    entries = sorted(((item.method, candidate) for item in estimators if item.status == "ok"
                      for candidate in item.candidates),
                     key=lambda pair: (-pair[1].support, pair[1].period_s, pair[0]))
    if not entries:
        note = "no_candidates" if any(item.status == "ok" for item in estimators) else "no_usable_estimators"
        return ReconciliationResult([], [], False, [note], is_stub=True)
    clusters = [
        Cluster(cid, candidate.period_s, {method: candidate.period_s}, {method: candidate.support},
                0., [], candidate.support,
                {"passed": True, "fail_reasons": [], "cycles": prep.meta["baseline_s"] / candidate.period_s})
        for cid, (method, candidate) in enumerate(entries, 1)
    ]
    return ReconciliationResult(clusters, [], False, ["stage2_not_evaluated"], is_stub=True)


def test_live_stage_wiring(observation, injected_config, mock_cnn, mock_recon, monkeypatch):
    monkeypatch.setattr(controller, "run_cnn", Mock(return_value=mock_cnn))
    runner = Mock(return_value=mock_recon)
    monkeypatch.setattr(controller, "reconcile", runner)
    result = controller.analyze(observation, injected_config)
    result.validate()
    assert tuple(result.estimators) == ("cnn", "lomb_scargle", "pdm")
    sent = runner.call_args.args[2]
    assert {key: value for key, value in sent.items() if key != "window_result"} == {
        "top_k": 3, "period_min_s": 2., "period_max_s": 20.,
    }
    assert "window_result" in sent
    assert injected_config.reconcile == {}
    assert result.verdict.label == "Ambiguous"
    assert result.verdict.reason_codes == [VERIFICATION_INCOMPLETE]
    assert len(result.verification.per_candidate) == 2 and not result.verification.complete
    assert "Incomplete candidates:" in result.stage_errors["verification"]


def test_shared_window_uses_effective_classical_settings(observation, injected_config,
                                                         mock_cnn, mock_recon, monkeypatch):
    from classical_analysis.window import compute_window

    settings = {"n_peaks": 0, "window_max_grid_points": 101}
    cfg = replace(injected_config, classical=settings)
    monkeypatch.setattr(controller, "run_cnn", Mock(return_value=mock_cnn))
    window = Mock(wraps=compute_window)
    reconcile = Mock(return_value=mock_recon)
    verifier = Mock(wraps=controller.verify_candidates)
    monkeypatch.setattr(controller, "compute_window", window)
    monkeypatch.setattr(controller, "reconcile", reconcile)
    monkeypatch.setattr(controller, "verify_candidates", verifier)
    result = controller.analyze(observation, cfg)
    result.validate()
    window.assert_called_once_with(result.prepared, {**settings, "period_min_s": 2., "period_max_s": 20.})
    shared = reconcile.call_args.args[2]["window_result"]
    assert verifier.call_args.args[2]["window_result"] is shared
    assert shared["peaks"] == []
    assert len(shared["freq_hz"]) <= 101


def test_window_failure_preserves_estimator_evidence(observation, injected_config, mock_cnn, monkeypatch):
    monkeypatch.setattr(controller, "run_cnn", Mock(return_value=mock_cnn))
    monkeypatch.setattr(controller, "compute_window", Mock(side_effect=ValueError("bad window")))
    reconcile = Mock()
    monkeypatch.setattr(controller, "reconcile", reconcile)
    result = controller.analyze(observation, injected_config)
    result.validate()
    assert result.stage_errors["reconciliation"] == "bad window"
    assert result.estimators[CNN] is mock_cnn
    assert result.reconciliation is result.verification is result.verdict is None
    reconcile.assert_not_called()


@pytest.mark.parametrize("validate_outputs", [True, False])
def test_guarded_stage_failure_matrix(observation, injected_config, mock_cnn, mock_recon,
                                      monkeypatch, validate_outputs):
    monkeypatch.setattr(controller, "run_cnn", Mock(return_value=mock_cnn))
    monkeypatch.setattr(controller, "reconcile", Mock(return_value=mock_recon))
    monkeypatch.setattr(controller, "verify_candidates", Mock(side_effect=RuntimeError("verifier failed")))
    cfg = replace(injected_config, pipeline={"validate_outputs": validate_outputs})
    result = controller.analyze(observation, cfg)
    result.validate()
    assert result.reconciliation is mock_recon and result.verification is None
    assert result.stage_errors["verification"] == "verifier failed"
    assert result.verdict.reason_codes == [VERIFICATION_INCOMPLETE]
    assert [option["cluster_id"] for option in result.verdict.alternatives] == [1, 2]


def test_progress_and_roundtrip(observation, injected_config, mock_cnn, mock_recon, monkeypatch):
    monkeypatch.setattr(controller, "run_cnn", Mock(return_value=mock_cnn))
    monkeypatch.setattr(controller, "reconcile", Mock(return_value=mock_recon))
    owner = get_ident()
    calls = []

    def progress(fraction, message):
        calls.append((fraction, message, get_ident()))
        raise RuntimeError("subscriber failure")

    result = controller.analyze(observation, injected_config, progress=progress)
    fractions = [item[0] for item in calls]
    assert fractions[0] == 0. and fractions[-1] == 1.
    assert fractions == sorted(fractions)
    assert all(message and "\n" not in message and thread == owner for _, message, thread in calls)
    assert_contract_equal(result, AnalysisResult.from_dict(json.loads(
        json.dumps(result.to_dict(), allow_nan=False)
    )))


@pytest.fixture
def upload_case(observation, injected_config, monkeypatch, isolated_runs):
    from core.schemas import LOW_QUALITY, QualityReport
    prepare = controller.prepare
    monkeypatch.setattr(controller, "prepare", lambda *args: replace(prepare(*args), is_stub=False))
    monkeypatch.setattr(controller, "assess_stage1", lambda *args: QualityReport(
        "insufficient", {}, [], [LOW_QUALITY]))
    runner = Mock(wraps=controller.analyze)
    monkeypatch.setattr(controller, "analyze", runner)
    artifact = {"checkpoint": "original", "calibration": "original"}
    implementation = {"sources": "original"}
    monkeypatch.setattr(cache, "_artifact_identity", lambda: artifact)
    monkeypatch.setattr(cache, "_implementation_identity", lambda: implementation)
    return runner, artifact, implementation


def test_upload_cache_identity(observation, injected_config, upload_case, isolated_runs):
    runner, artifacts, source = upload_case
    session = {}
    call = cache._analyze_upload
    first = call(b"original", "first.txt", observation, injected_config, session)
    snapshot = isolated_runs / "run_config.json"
    snapshot.unlink()
    progress = Mock()
    second = call(b"original", "first.txt", observation, injected_config, session, progress)
    assert_contract_equal(first, second)
    assert first is not second and runner.call_count == 1 and snapshot.exists()
    assert [args.args[0] for args in progress.call_args_list] == [0., 1.]
    second.stage_errors["mutated"] = "local only"
    assert call(b"original", "first.txt", observation, injected_config, session).stage_errors == {}
    call(b"different", "first.txt", observation, injected_config, session)
    renamed = replace(observation, obs_id="second")
    result = call(b"different", "second.txt", renamed, injected_config, session)
    assert result.obs_id == "second"
    config = replace(injected_config, verdict={"limited_max_label": "Ambiguous"})
    call(b"different", "second.txt", renamed, config, session)
    artifacts["checkpoint"] = "new"
    call(b"different", "second.txt", renamed, config, session)
    artifacts["calibration"] = "new"
    call(b"different", "second.txt", renamed, config, session)
    source["sources"] = "changed"
    call(b"different", "second.txt", renamed, config, session)
    assert runner.call_count == 7


def test_upload_cache_recovery(observation, injected_config, upload_case, monkeypatch):
    runner, *_ = upload_case
    session = {}
    call = cache._analyze_upload
    good = call(b"bytes", "first.txt", observation, injected_config, session)
    key, _ = session["analysis"]
    session["analysis"] = (key, "invalid JSON")
    call(b"bytes", "first.txt", observation, injected_config, session)
    assert runner.call_count == 2
    session["analysis"] = (key, json.dumps({**good.to_dict(), "schema_version": 4}))
    call(b"bytes", "first.txt", observation, injected_config, session)
    assert runner.call_count == 3
    session.clear()
    runner.side_effect = [replace(good, stage_errors={"cnn": "temporary failure"}), good, good]
    call(b"bytes", "first.txt", observation, injected_config, session)
    assert session == {}
    call(b"bytes", "first.txt", observation, injected_config, session)
    call(b"bytes", "first.txt", observation, injected_config, session)
    assert runner.call_count == 5


def test_upload_cache_isolation(observation, injected_config, upload_case):
    runner, *_ = upload_case
    first, second = {}, {}
    for session in (first, second, first, second):
        cache._analyze_upload(b"bytes", "first.txt", observation, injected_config, session)
    assert runner.call_count == 2
    assert first["analysis"][0] == second["analysis"][0]
    first.clear()
    cache._analyze_upload(b"bytes", "first.txt", observation, injected_config, second)
    assert runner.call_count == 2 and second["analysis"][1]


def test_invalid_upload_never_analyzes(monkeypatch):
    from io import BytesIO
    from pathlib import Path
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    file = BytesIO(b"invalid measurement input")
    file.name = "invalid.txt"
    runner = Mock(side_effect=AssertionError("invalid upload reached analyze"))
    monkeypatch.setattr(controller, "analyze", runner)
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: file)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[2] / "app/main.py")).run()
    assert app.error and not app.exception
    runner.assert_not_called()


def test_cp3b_holdout_readback(isolated_runs):
    import numpy as np

    from core.config import load_config, resolve_period_bounds
    from core.schemas import Observation
    from scripts.prepare_cp3b_holdout import CORPUS, HOLDOUT, _provenance

    cfg = load_config()
    manifest = json.loads((CORPUS / "manifest.json").read_bytes())
    used = {row["group"] for split in ("train", "val") for row in _provenance(split, manifest)}
    assert 0 < HOLDOUT.stat().st_size < 5_000_000
    with np.load(HOLDOUT, allow_pickle=False) as shard:
        data = {key: shard[key] for key in shard.files}
    assert set(data) == {"t_s", "brightness", "offsets", "groups", "periods", "bounds"}
    assert len(set(data["groups"])) == 200 and not used.intersection(data["groups"])
    assert np.all(data["groups"] // 1_000_000 == 1)
    assert data["bounds"].tolist() == list(resolve_period_bounds(cfg.period_bounds))
    assert data["offsets"][0] == 0 and data["offsets"].shape == (201,)
    assert np.all(np.diff(data["offsets"]) > 0)
    assert data["offsets"][-1] == len(data["t_s"]) == len(data["brightness"])
    assert np.isfinite(data["t_s"]).all() and not np.isinf(data["brightness"]).any()
    positive = data["periods"][data["periods"] != 0]
    assert len(positive) == 140 and np.all((positive >= data["bounds"][0]) & (positive <= data["bounds"][1]))
    segment = slice(*data["offsets"][:2])
    obs = Observation(str(data["groups"][0]), data["t_s"][segment],
                      data["brightness"][segment], None, "unknown", {})
    obs.validate()
    result = controller.analyze(obs, cfg)
    result.validate()
    assert result.obs_id == obs.obs_id


@pytest.mark.parametrize("name", [
    "00029A_2_3_R_20190322_DATA.txt", "00080B_1_2_R_20190724_DATA.txt",
    "00041B_2_3_R_20191118_DATA.txt",
])
def test_cp3b_real_sdlcd(name, isolated_runs):
    from pathlib import Path

    from core.config import load_config
    from ingestion import load_observation

    path = Path(__file__).resolve().parents[2] / "data/sdlcd_raw" / name
    assert path.is_file(), f"CP3b requires the local SDLCD file: {name}"
    result = controller.analyze(load_observation(str(path)), load_config())
    assert_contract_equal(result, AnalysisResult.from_dict(json.loads(
        json.dumps(result.to_dict(), allow_nan=False)
    )))
    if result.reconciliation is not None and result.reconciliation.is_stub:
        pytest.skip("CP3b requires P1's real Stage 2 reconciliation")
    assert result.verdict is not None
    assert result.reconciliation is None or not result.reconciliation.is_stub
    expected = {
        "00029A": {"4:2", "7:4", "8:7"},
        "00080B": {"5:3", "9:5", "10:9"},
        "00041B": set(),
    }[name.split("_")[0]]
    if expected:
        assert result.verification is not None and result.verification.complete
        assert {row.pair_id for row in result.verification.harmonic_comparisons} == expected
        assert all(row.status == "ok" for row in result.verification.harmonic_comparisons)
    else:
        assert not result.reconciliation.shortlist and result.verification is None


def test_cp3b_real_upload(monkeypatch, isolated_runs):
    from io import BytesIO
    from pathlib import Path
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    root = Path(__file__).resolve().parents[2]
    name = "00029A_2_3_R_20190322_DATA.txt"
    file = BytesIO((root / "data/sdlcd_raw" / name).read_bytes())
    file.name = name
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: file)
    app = AppTest.from_file(str(root / "app/main.py")).run(timeout=180)
    assert not app.exception and not app.error
    headings = {item.value for item in app.subheader}
    if "Stage 2 assessment pending" in headings:
        pytest.skip("Actual SDLCD upload stops at the CP2 Stage 2 skeleton")
    assert headings & {"Supported", "Ambiguous", "Inconclusive"}
    assert "Harmonic comparisons" in headings


def test_cp3b_runtime():
    from pathlib import Path
    from statistics import median

    root = Path(__file__).resolve().parents[2]
    report = json.loads((root / "docs/results/runtime_cp3b.json").read_bytes())
    cnn = report["cnn"]
    assert cnn["statuses"] == ["ok"] * 4 and all(n >= 1 for n in cnn["windows"])
    assert len(cnn["repeated_runtime_s"]) == 3 and all(t > 0 for t in cnn["repeated_runtime_s"])
    assert 0 < cnn["cold_runtime_s"] < 2 and median(cnn["repeated_runtime_s"]) < 2
    timing = report["verification_timing"]
    assert timing["attempted"] == 10 and timing["completed"] > 0
    assert timing["completed_median_runtime_s"] <= 30
    assert not timing["failure_counts"].get("timeout")
    assert report["verification_folds"] == 3 and report["verification_families"] == ["box", "cylinder"]
    assert len(report["verification_decoys_per_curve"]) == 50
    assert len(report["real_file_diagnostics"]) == 3 and report["environment"]["torch_threads"] > 0
