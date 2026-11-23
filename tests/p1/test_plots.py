from dataclasses import replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from app import plots
from classical_analysis.lomb_scargle import run_lomb_scargle
from classical_analysis.pdm import run_pdm
from core.config import load_config
from core.schemas import (
    CNN, LOMB_SCARGLE, LOW_QUALITY, NOT_RUN, PDM, CandidateVerification,
    EstimatorResult, QualityReport, ReconciliationResult, STAGE2_FAIL_REASONS, VerificationResult,
)
from preprocessing.prepare import prepare
from preprocessing.quality import assess_stage1
from reconciliation.reconcile import reconcile


@pytest.fixture
def live_inputs(observation):
    config = load_config()
    prep = prepare(observation, config.preprocess)
    bounds = dict(period_min_s=2.0, period_max_s=20.0)
    classical = {**config.classical, **bounds}
    ls, pdm = run_lomb_scargle(prep, classical), run_pdm(prep, classical)
    assert ls.status == pdm.status == "ok"
    return prep, assess_stage1(prep, config.preprocess), ls, pdm, {**config.reconcile, **bounds}


def trace(figure, name):
    return next(item for item in figure.data if item.name == name)


def annotations(figure):
    return " ".join(item.text for item in figure.layout.annotations)


def test_lightcurve_marks_flags_and_phase_fold_filters_nonfinite(prepared):
    valid = prepared.valid_mask.copy()
    indices = np.flatnonzero(valid)
    flags = {name: np.zeros_like(valid) for name in ("bright_outlier", "duplicate", "gap_edge")}
    for index, mask in zip(indices, flags.values()):
        mask[index] = True
    prep = replace(prepared, flags=flags)
    light = plots.fig_lightcurve(prep)
    np.testing.assert_array_equal(trace(light, "Valid samples").x, prep.t_s[valid])
    for index, label in zip(indices, ("Brightness outlier", "Duplicate timestamp", "Gap boundary")):
        assert tuple(trace(light, label).x) == (prep.t_s[index],)

    brightness = prep.y_norm.copy()
    brightness[indices[-1]] = np.inf
    folded = plots.fig_phase_fold(replace(prep, y_norm=brightness), 8.0)
    keep = valid & np.isfinite(brightness)
    np.testing.assert_allclose(trace(folded, "Valid samples").x, prep.t_s[keep] % 8 / 8)
    np.testing.assert_array_equal(trace(folded, "Valid samples").y, brightness[keep])
    np.testing.assert_array_equal(prep.valid_mask, valid)


@pytest.mark.parametrize("period", [None, 0, -1, np.inf, np.nan, "invalid"])
def test_invalid_period_returns_annotation(prepared, period):
    assert "finite positive" in annotations(plots.fig_phase_fold(prepared, period))


def test_periodograms_show_successful_series_and_statuses(live_inputs, mock_cnn):
    _, _, ls, pdm, _ = live_inputs
    estimators = {CNN: mock_cnn, LOMB_SCARGLE: ls, PDM: pdm}
    figure = plots.fig_periodograms(estimators)
    assert list(trace(figure, "CNN").y) == [candidate.support for candidate in mock_cnn.candidates]
    for result, name in ((ls, "Lomb–Scargle (LS)"), (pdm, "PDM")):
        series = trace(figure, name)
        np.testing.assert_allclose(series.x, 1 / np.array(result.meta["periodogram"]["freq_hz"])[::-1])
        np.testing.assert_allclose(series.y, result.meta["periodogram"]["power"][::-1])

    cnn = replace(mock_cnn, candidates=[], meta={**mock_cnn.meta, "abstained": True})
    failed = EstimatorResult(PDM, "failed", [], "fit failed", {"runtime_s": 0.0})
    partial = plots.fig_periodograms({CNN: cnn, LOMB_SCARGLE: replace(ls, candidates=[]), PDM: failed})
    assert trace(partial, "Lomb–Scargle (LS)").line.color == "#D55E00"
    assert all(text in annotations(partial) for text in ("CNN abstained", "No significant candidate", "fit failed"))
    skipped = replace(failed, status="skipped", error=NOT_RUN)
    assert "skipped" in annotations(plots.fig_periodograms({PDM: skipped}))


def test_candidates_keep_rejected_clusters_and_pending_status(mock_recon):
    short, long = mock_recon.clusters
    rejected = replace(long, alias_risk=0.7, stage2={
        **long.stage2, "passed": False, "fail_reasons": [STAGE2_FAIL_REASONS[0]],
    })
    figure = plots.fig_candidates(replace(mock_recon, clusters=[short, rejected]))
    assert tuple(trace(figure, "Shortlisted").x) == (8.0,)
    assert tuple(trace(figure, "Harmonic link").x) == (8.0, 16.0, None)
    detail = trace(figure, "Rejected by Stage 2").customdata[0]
    assert detail[1] == 0.7 and detail[3] == STAGE2_FAIL_REASONS[0]
    assert tuple(trace(figure, "CNN").x) == (8.0, 16.0)
    pending = replace(mock_recon, shortlist=[], stage2_passed_any=False, notes=["stage2_not_evaluated"])
    assert "Stage 2 assessment pending" in plots.fig_candidates(pending).layout.title.text
    assert all(item.name != "Shortlisted" for item in plots.fig_candidates(pending).data)


def test_quality_shows_units_warning_and_reason_code():
    report = QualityReport("insufficient", {"missing_fraction": 0.25, "duration_s": 12.0},
                           ["Too few valid observations."], [LOW_QUALITY])
    figure = plots.fig_quality(report)
    assert "insufficient" in figure.layout.title.text
    assert list(figure.data[0].cells.values[1]) == [
        "25.0%", "12 s", "Too few valid observations.", LOW_QUALITY,
    ]


@pytest.fixture
def verification():
    completed = CandidateVerification(
        1, 8.0, "ok", 0.2, 0.4, [6.0, 12.0], [0.3, -0.1],
        False, {"shape": "box", "geometry_errors": {"box": 0.4}, "fit_space": "y_norm",
                "scale_a": 1.0, "offset_b": 0.0}, [],
    )
    skipped = CandidateVerification(2, 16.0, "skipped", None, None, [], [], False, None, [NOT_RUN])
    from core.schemas import HarmonicComparison

    result = VerificationResult([completed, skipped], False, "fast", 0.1, harmonic_comparisons=[
        HarmonicComparison("1:2", 1, 2, 8., 16., 2., "skipped", "undecided", .4, None,
                           None, None, "harmonic_fit_unfinished", [])])
    result.validate()
    return result


def test_partial_verification_omits_missing_metrics(verification):
    figure = plots.fig_verification(verification)
    assert "Incomplete" in figure.layout.title.text
    assert tuple(trace(figure, "Fit error").y) == (0.2,)
    assert tuple(trace(figure, "Held-out error").y) == (0.4,)
    assert tuple(trace(figure, "Decoy").y) == (0.3, -0.1)
    assert list(figure.data[-1].cells.values[1]) == ["Rejected", "skipped"]
    stub = replace(verification, per_candidate=[verification.per_candidate[1]], is_stub=True)
    unavailable = plots.fig_verification(stub)
    assert "No completed fits" in annotations(unavailable)
    assert all(item.type != "bar" for item in unavailable.data)


def test_risk_coverage_groups_sorts_and_preserves_input():
    rows = pd.DataFrame({
        "coverage": [0.7, 0.2, 0.3, 0.0], "risk": [0.2, 0.1, 0.1, np.nan],
        "n_selected": [7, 2, 3, 0], "configuration": [CNN, CNN, PDM, CNN],
        "threshold": [0.6, 0.9, None, None],
    })
    original = rows.copy()
    figure = plots.fig_risk_coverage(rows)
    cnn = trace(figure, "CNN")
    np.testing.assert_array_equal(cnn.x, [0.2, 0.7])
    assert cnn.line.color == "#0072B2"
    assert list(cnn.customdata[0]) == [2.0, "0.9"]
    assert trace(figure, "PDM").line.color == "#009E73"
    pd.testing.assert_frame_equal(rows, original)


def test_calibration_shows_diagonal_and_omits_empty_bins():
    rows = pd.DataFrame({"mean_confidence": [0.8, 0.1], "observed_accuracy": [0.7, 0.0], "n": [12, 0]})
    figure = plots.fig_calibration(rows)
    assert tuple(trace(figure, "Ideal calibration").x) == (0, 1)
    assert tuple(trace(figure, "Observed calibration").y) == (0.7,)


@pytest.mark.parametrize(("function", "args", "message"), [
    (plots.fig_lightcurve, (None,), "unavailable"),
    (plots.fig_phase_fold, (None, 8), "unavailable"),
    (plots.fig_periodograms, ({},), "unavailable"),
    (plots.fig_candidates, (ReconciliationResult([], [], False, ["no_usable_estimators"]),), "No usable"),
    (plots.fig_quality, (None,), "unavailable"),
    (plots.fig_verification, (None,), "unavailable"),
    (plots.fig_risk_coverage, (pd.DataFrame(),), "No evaluation"),
    (plots.fig_calibration, (pd.DataFrame({"n": [1]}),), "Missing evaluation columns"),
])
def test_unavailable_inputs_return_annotated_figures(function, args, message):
    figure = function(*args)
    assert isinstance(figure, go.Figure)
    assert message in annotations(figure)


def test_every_figure_exports_to_html(live_inputs, mock_cnn, verification, tmp_path):
    prep, quality, ls, pdm, cfg = live_inputs
    estimators = {item.method: item for item in (mock_cnn, ls, pdm)}
    recon = reconcile(prep, list(estimators.values()), cfg)
    figures = [
        plots.fig_lightcurve(prep), plots.fig_periodograms(estimators),
        plots.fig_phase_fold(prep, 8.0), plots.fig_candidates(recon),
        plots.fig_quality(quality), plots.fig_verification(verification),
        plots.fig_risk_coverage(pd.DataFrame({"coverage": [0.5], "risk": [0.2], "n_selected": [10]})),
        plots.fig_calibration(pd.DataFrame({"mean_confidence": [0.8], "observed_accuracy": [0.7], "n": [10]})),
    ]
    for index, figure in enumerate(figures):
        target = tmp_path / f"{index}.html"
        figure.write_html(target, include_plotlyjs="cdn")
        assert "Plotly.newPlot" in target.read_text(encoding="utf-8")
