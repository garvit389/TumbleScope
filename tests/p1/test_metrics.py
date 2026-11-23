import json
from math import log

import numpy as np
import pandas as pd
import pytest

from app.plots import fig_calibration, fig_risk_coverage
from evaluation.metrics import (
    brier_score, candidate_retention, catastrophic_error_rate, expected_calibration_error,
    harmonic_error_rate, median_log_period_error, recall_at_k, risk_coverage_points,
    supported_metrics, tolerance_accuracy,
)


# Provisional unit-test conventions, not calibrated operating thresholds.
TOLERANCE = 0.05
FACTOR = 2.0
BINS = 10


def test_median_log_error_uses_only_valid_period_pairs():
    result = median_log_period_error(
        [100, 100, 100, 100, None, 0], [100, 200, 400, None, 10, 20],
    )
    assert result["value"] == pytest.approx(log(2))
    assert result["denominator"] == 3
    assert result["n_missing_prediction"] == 1
    assert result["n_missing_reference"] == 2
    assert median_log_period_error([1e-300], [1e300])["value"] == pytest.approx(600 * log(10))


def test_direct_accuracy_includes_exact_boundaries_and_counts_missing():
    result = tolerance_accuracy(
        [100] * 7 + [np.nan], [100, 95, 105, 105.001, None, np.nan, 0, 100],
        tolerance=TOLERANCE,
    )
    assert result["overall"] == {"value": 3 / 7, "numerator": 3, "denominator": 7}
    assert result["conditional"] == {"value": 3 / 4, "numerator": 3, "denominator": 4}
    assert result["n_missing_prediction"] == 3
    assert result["n_missing_reference"] == 1


def test_harmonic_mistakes_are_direct_errors_and_exclude_direct_hits():
    truth = [100] * 6
    prediction = [50, 200, 47.5, 210, 100, None]
    result = harmonic_error_rate(truth, prediction, tolerance=TOLERANCE, harmonic_tolerance=TOLERANCE)
    assert result["numerator"] == 4
    assert result["denominator"] == 5
    assert result["value"] == 0.8
    assert tolerance_accuracy(truth, prediction, tolerance=TOLERANCE)["overall"]["numerator"] == 1

    overlap = harmonic_error_rate([100], [100], tolerance=TOLERANCE, harmonic_tolerance=0.6)
    assert overlap["numerator"] == 0


def test_catastrophic_factor_is_symmetric_and_strict():
    result = catastrophic_error_rate(
        [100] * 7, [50, 49, 200, 201, 100, np.inf, -2], factor=FACTOR,
    )
    assert result["numerator"] == 2
    assert result["denominator"] == 5
    assert result["value"] == 0.4
    assert result["n_missing_prediction"] == 2


def test_recall_at_k_does_not_resort_or_promote_invalid_candidates():
    truth = [10, 10, 10, 10, None]
    predictions = [[20, 10], [10, 20], [np.nan, 10], [], [10]]
    first = recall_at_k(truth, predictions, k=1, tolerance=TOLERANCE)
    second = recall_at_k(truth, predictions, k=2, tolerance=TOLERANCE)
    assert first["numerator"] == 1 and first["denominator"] == 4
    assert second["numerator"] == 3 and second["denominator"] == 4


def test_retention_conditions_on_an_upstream_direct_hit():
    result = candidate_retention(
        [10, 10, 10, 10, None], [[20, 10], [10], [10], [20], [10]],
        [[10, 10.1], [20], None, [10], [10]], tolerance=TOLERANCE,
    )
    assert result["numerator"] == 1
    assert result["denominator"] == 3
    assert result["n_known_period"] == 4


def test_supported_metrics_keep_operational_outcomes_and_explicit_truth_separate():
    result = supported_metrics(
        [100, 100, None, None, None, None, 100, 100, 100, 100],
        [100, 200, 10, 10, None, None, None, None, None, np.nan],
        ["supported", "supported", "supported", "supported", "abstained", "abstained",
         "pending", "execution_error", "missing", "supported"],
        no_reliable_period=[False, False, True, False, True, False, False, False, False, False],
        tolerance=TOLERANCE,
    )
    assert result["coverage"] == {"value": 0.5, "numerator": 5, "denominator": 10}
    assert result["false_supported"] == {"value": 2 / 3, "numerator": 2, "denominator": 3}
    assert result["false_supported_per_labeled"]["denominator"] == 8
    assert result["supported_accuracy"]["value"] == 1 / 3
    assert result["correct_abstention"] == {"value": 0.5, "numerator": 1, "denominator": 2}
    assert result["disposition_counts"] == {
        "supported": 5, "abstained": 2, "pending": 1, "execution_error": 1, "missing": 1,
    }
    assert result["n_supported_unknown_truth"] == result["n_supported_missing_prediction"] == 1
    assert result["n_unknown_truth"] == 2
    assert result["n_missing_prediction"] == 6
    json.dumps(result, allow_nan=False)


def test_no_reliable_period_is_explicit_even_with_a_missing_claim_period():
    result = supported_metrics([None], [None], ["supported"],
                               no_reliable_period=np.array([True]), tolerance=TOLERANCE)
    assert result["false_supported"]["value"] == 1.0
    assert result["n_no_reliable_period"] == 1
    with pytest.raises(ValueError, match="cannot have period truth"):
        supported_metrics([10], [10], ["supported"], no_reliable_period=[True], tolerance=TOLERANCE)


def test_risk_coverage_keeps_tied_scores_together_and_includes_requested_thresholds():
    scores = np.array([3.0, 1.0, 3.0, -2.0, np.nan, 9.0])
    snapshot = scores.copy()
    correct = [True, True, False, False, True, None]
    rows = risk_coverage_points(scores, correct)
    assert [row["threshold"] for row in rows] == [3, 1, -2]
    assert [row["n_selected"] for row in rows] == [2, 3, 4]
    assert [row["risk"] for row in rows] == [1 / 2, 1 / 3, 2 / 4]
    assert [row["coverage"] for row in rows] == [0.5, 0.75, 1]
    assert rows[0]["n_input"] == 6 and rows[0]["n_eligible"] == 4
    explicit = risk_coverage_points(scores, correct, thresholds=[-3, 4, 2])
    assert [row["threshold"] for row in explicit] == [4, 2, -3]
    assert explicit[0]["n_selected"] == 0 and explicit[0]["risk"] is None
    assert explicit[0]["coverage"] == 0.0
    assert explicit[1]["n_selected"] == 2
    np.testing.assert_array_equal(scores, snapshot)


def test_brier_uses_only_explicit_binary_probability_pairs():
    result = brier_score([0.2, 0.75, -0.1, np.nan, 1, 0.5, 0.3],
                         [False, True, True, False, None, 2, 0.5])
    assert result["numerator"] == pytest.approx(0.1025)
    assert result["denominator"] == 2
    assert result["value"] == pytest.approx(0.05125)
    assert result["n_input"] == 7


def test_ece_weights_nonempty_bins_and_assigns_exact_boundaries():
    result = expected_calibration_error(
        [0.0, 0.09, 0.1, 0.29, 0.3, 0.95, 1, None], [0, 0, 1, 0, 1, 1, 1, 1], n_bins=BINS,
    )
    assert result["denominator"] == 7
    assert result["numerator"] == pytest.approx(0.09 + 0.9 + 0.29 + 0.7 + 0.05)
    assert result["value"] == pytest.approx(2.03 / 7)
    assert [row["n"] for row in result["bins"]] == [2, 1, 1, 1, 2]
    assert [row["bin_lower"] for row in result["bins"]] == [0, 0.1, 0.2, 0.3, 0.9]
    assert result["bins"][-1]["bin_upper"] == 1.0
    assert result["bins"][-1]["mean_confidence"] == pytest.approx(0.975)


def test_metric_rows_are_compatible_with_p1_11_figures():
    risk_rows = risk_coverage_points([5.0, 3.0], [True, False])
    calibration = expected_calibration_error([0.2, 0.8], [False, True], n_bins=BINS)
    risk = fig_risk_coverage(pd.DataFrame(risk_rows).assign(configuration="PDM"))
    reliability = fig_calibration(pd.DataFrame(calibration["bins"]).assign(configuration="CNN"))
    assert list(risk.data[0].x) == [0.5, 1.0]
    assert list(risk.data[0].y) == [0.0, 0.5]
    assert list(reliability.data[1].x) == [0.2, 0.8]
    assert list(reliability.data[1].y) == [0.0, 1.0]


def test_undefined_metrics_use_none_and_zero_denominators():
    results = [
        median_log_period_error([], []),
        tolerance_accuracy([], [], tolerance=TOLERANCE)["overall"],
        tolerance_accuracy([10], [None], tolerance=TOLERANCE)["conditional"],
        harmonic_error_rate([], [], tolerance=TOLERANCE, harmonic_tolerance=TOLERANCE),
        catastrophic_error_rate([], [], factor=FACTOR),
        recall_at_k([None], [[]], k=2, tolerance=TOLERANCE),
        candidate_retention([10], [[]], [[10]], tolerance=TOLERANCE),
        brier_score([], []),
        expected_calibration_error([], [], n_bins=BINS),
    ]
    summary = supported_metrics([], [], [], no_reliable_period=[], tolerance=TOLERANCE)
    results.extend([summary["coverage"], summary["false_supported"], summary["correct_abstention"]])
    assert all(result["value"] is None and result["denominator"] == 0 for result in results)
    assert risk_coverage_points([], []) == []
    assert risk_coverage_points([], [], thresholds=[1])[0]["coverage"] is None


@pytest.mark.parametrize(("call", "message"), [
    (lambda: median_log_period_error([10], []), "zip"),
    (lambda: tolerance_accuracy([], [], tolerance=np.nan), "tolerance"),
    (lambda: recall_at_k([], [], k=0, tolerance=TOLERANCE), "positive integer"),
    (lambda: catastrophic_error_rate([], [], factor=1), "greater than one"),
    (lambda: expected_calibration_error([], [], n_bins=0), "positive integer"),
    (lambda: risk_coverage_points([], [], thresholds=[np.inf]), "finite"),
])
def test_invalid_configuration_and_unaligned_inputs_are_rejected(call, message):
    with pytest.raises(ValueError, match=message):
        call()
