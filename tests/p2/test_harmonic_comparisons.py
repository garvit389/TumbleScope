"""Schema-v5 harmonic pairs through fitting, serialization and verdict safety."""

from dataclasses import replace
import json

import numpy as np
import pytest

from core.config import load_config
from core.schemas import (
    HARMONIC_AMBIGUITY, LOMB_SCARGLE, VERIFICATION_INCOMPLETE,
    Candidate, EstimatorResult, Observation, QualityReport, VerificationResult,
    check_verification_scope, expected_harmonic_pairs,
)
from physics.verify import verify_candidates
from preprocessing.prepare import prepare
from tests.verification_cases import with_candidate_fits
from verdict.decide import decide


@pytest.fixture
def chain(mock_recon):
    middle, long = mock_recon.clusters
    short = replace(long, cluster_id=3, period_s=4., score=.5, members={},
                    harmonic_links=[{"other_id": 1, "ratio": 2.}])
    middle = replace(middle, harmonic_links=[*middle.harmonic_links, {"other_id": 3, "ratio": .5}])
    return replace(mock_recon, clusters=[middle, long, short])


@pytest.fixture(scope="module")
def signal():
    t = np.arange(0., 256., .5)
    return prepare(Observation("pair-fit", t, np.sin(2 * np.pi * t / 8.), None, "unknown", {}), {})


def test_real_fits_complete_both_pairs(signal, chain):
    cfg = load_config().verification
    cfg = {**cfg, "period_min_s": 2., "period_max_s": 24.,
           "window_result": {"freq_hz": [0.], "window_power": [1.], "peaks": [],
                             "freq_resolution": signal.meta["freq_resolution"]},
           "fit": {**cfg["fit"], "shapes": ["box"], "coarse_samples": 8,
                   "restarts": 1, "popsize": 4, "maxiter": 5}}
    result = verify_candidates(signal, chain, cfg)
    assert result.complete
    assert [pair.pair_id for pair in result.harmonic_comparisons] == ["1:2", "3:1"]
    assert all(pair.status == "ok" and len(pair.notes) == 5 for pair in result.harmonic_comparisons)
    check_verification_scope(chain, result)
    assert VerificationResult.from_dict(json.loads(json.dumps(result.to_dict()))) == result
    lead, *others = chain.clusters
    reordered = replace(chain, clusters=[replace(lead, harmonic_links=lead.harmonic_links[::-1]), *others])
    assert expected_harmonic_pairs(reordered) == expected_harmonic_pairs(chain)
    assert list(expected_harmonic_pairs(reordered)) == list(expected_harmonic_pairs(chain))


@pytest.mark.parametrize("fault", ["missing", "duplicate", "alternate_id", "unknown", "orientation",
                                   "ratio", "outcome", "period", "order"])
def test_invalid_pairs_never_reach_supported(signal, chain, mock_cnn, fault):
    result = verify_candidates(signal, chain, {})
    first, second = result.harmonic_comparisons
    changes = {
        "missing": [first],
        "duplicate": [first, first, second],
        "alternate_id": [first, replace(first, pair_id="other-id"), second],
        "unknown": [replace(first, long_cluster_id=9, pair_id="1:9"), second],
        "orientation": [replace(first, short_period_s=first.long_period_s), second],
        "ratio": [replace(first, ratio=3.), second],
        "outcome": [replace(first, prefers="P"), second],
        "period": [replace(first, long_period_s=17.), second],
        "order": [second, first],
    }
    bad = replace(result, harmonic_comparisons=changes[fault])
    with pytest.raises(ValueError, match="contract:"):
        bad.validate()
        check_verification_scope(chain, bad)
    verdict = decide(QualityReport("suitable", {}, [], []), {"cnn": mock_cnn}, chain, bad, {})
    assert verdict.label == "Ambiguous" and VERIFICATION_INCOMPLETE in verdict.reason_codes


def test_complete_scientific_loss_and_unfinished_pair_are_distinct(signal, chain, mock_cnn):
    result = verify_candidates(signal, chain, {})
    complete = with_candidate_fits(result, [
        replace(record, status="ok", fit_error=.2, heldout_error={1: .25, 2: .4, 3: .15}[record.cluster_id],
                passed=record.cluster_id == 1,
                best_params={"shape": "box", "geometry_errors": {"box": .25, "cylinder": .26},
                             "fit_space": "y_norm", "scale_a": 1., "offset_b": 0.})
        for record in result.per_candidate])
    first, second = complete.harmonic_comparisons
    assert complete.complete and first.prefers == "P" and second.prefers == "P"
    check_verification_scope(chain, complete)
    quality = QualityReport("suitable", {}, [], [])
    estimators = {"cnn": mock_cnn}
    verdict = decide(quality, estimators, chain, complete, {"alias_risk_max": .35})
    assert verdict.label == "Ambiguous" and HARMONIC_AMBIGUITY in verdict.reason_codes
    failed = replace(second, status="failed", prefers="undecided", delta_e=None, margin=None,
                     reason="harmonic_comparison_failed", notes=["diagnostic calculation failed"])
    incomplete = replace(complete, harmonic_comparisons=[first, failed], complete=False)
    check_verification_scope(chain, incomplete)
    verdict = decide(quality, estimators, chain, incomplete, {"alias_risk_max": .35})
    assert verdict.label == "Ambiguous" and verdict.reason_codes == [VERIFICATION_INCOMPLETE]
    with pytest.raises(ValueError, match="incorrect verification completeness"):
        check_verification_scope(chain, replace(incomplete, complete=True))
    skipped = replace(failed, status="skipped", reason="harmonic_comparison_not_run")
    unattempted = replace(incomplete, harmonic_comparisons=[first, skipped])
    check_verification_scope(chain, unattempted)
    verdict = decide(quality, estimators, chain, unattempted, {"alias_risk_max": .35})
    assert verdict.label == "Ambiguous" and verdict.reason_codes == [VERIFICATION_INCOMPLETE]


def test_two_favorable_comparisons_allow_existing_supported_path(signal, chain, mock_cnn):
    lead, *others = chain.clusters
    chain = replace(chain, clusters=[replace(lead, members={**lead.members, LOMB_SCARGLE: 8.}), *others])
    result = verify_candidates(signal, chain, {})
    complete = with_candidate_fits(result, [
        replace(record, status="ok", fit_error=.2, heldout_error=.2 if record.cluster_id == 1 else .4,
                passed=record.cluster_id == 1,
                best_params={"shape": "box", "geometry_errors": {"box": .2},
                             "fit_space": "y_norm", "scale_a": 1., "offset_b": 0.})
        for record in result.per_candidate])
    quality = QualityReport("suitable", {}, [], [])
    ls = EstimatorResult(LOMB_SCARGLE, "ok", [Candidate(8., 1., .5, {"freq_width_hz": .01})],
                         meta={"periodogram": {"freq_hz": [], "power": []},
                               "freq_resolution": .01, "runtime_s": 0.})
    verdict = decide(quality, {"cnn": mock_cnn, LOMB_SCARGLE: ls}, chain, complete, {"alias_risk_max": .35})
    assert verdict.label == "Supported" and verdict.best_cluster_id == 1


def test_standalone_v4_verification_is_rejected(signal, chain):
    payload = verify_candidates(signal, chain, {}).to_dict()
    payload.pop("harmonic_comparisons")
    payload["per_candidate"][0]["harmonic_comparison"] = None
    with pytest.raises(ValueError, match="missing harmonic_comparisons"):
        VerificationResult.from_dict(payload)
