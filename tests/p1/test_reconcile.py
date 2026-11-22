import json
from dataclasses import replace

import pytest

from core.schemas import CNN, NOT_RUN, ReconciliationResult, assert_contract_equal
from reconciliation.reconcile import reconcile


def test_clusters_consume_mock_cnn_and_round_trip(prepared, mock_cnn):
    result = reconcile(prepared, [mock_cnn], {})

    assert result.is_stub
    assert result.shortlist == []
    assert result.stage2_passed_any is False
    assert result.notes == ["stage2_not_evaluated"]
    assert [cluster.period_s for cluster in result.clusters] == [8.0, 16.0, 11.0]
    assert [cluster.cluster_id for cluster in result.clusters] == [1, 2, 3]
    for cluster, candidate in zip(result.clusters, mock_cnn.candidates, strict=True):
        assert cluster.members == {CNN: candidate.period_s}
        assert cluster.evidence == {CNN: candidate.support}
        assert cluster.score == candidate.support
        assert cluster.alias_risk == 0.0
        assert cluster.harmonic_links == []
        assert cluster.stage2["cycles"] == pytest.approx(
            prepared.meta["baseline_s"] / candidate.period_s
        )
        cluster.validate()

    result.validate()
    encoded = json.dumps(result.to_dict(), allow_nan=False)
    assert_contract_equal(result, ReconciliationResult.from_dict(json.loads(encoded)))


def test_equal_support_order_is_independent_of_input_order(prepared, mock_cnn):
    candidates = [replace(candidate, support=0.5) for candidate in mock_cnn.candidates]
    first = reconcile(prepared, [replace(mock_cnn, candidates=candidates)], {})
    second = reconcile(prepared, [replace(mock_cnn, candidates=candidates[::-1])], {})
    assert [cluster.period_s for cluster in first.clusters] == [8.0, 11.0, 16.0]
    assert_contract_equal(first, second)


@pytest.mark.parametrize(("status", "error", "note"), [
    ("ok", None, "no_candidates"),
    ("skipped", NOT_RUN, "no_usable_estimators"),
])
def test_empty_outcomes_preserve_contract(prepared, mock_cnn, status, error, note):
    estimator = replace(mock_cnn, status=status, candidates=[], error=error)
    result = reconcile(prepared, [estimator], {})
    result.validate()
    assert result.clusters == result.shortlist == []
    assert result.stage2_passed_any is False
    assert result.notes == [note]
