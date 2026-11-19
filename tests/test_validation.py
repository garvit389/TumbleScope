from dataclasses import replace

import numpy as np
import pytest

from core.schemas import (
    LOW_QUALITY, LOMB_SCARGLE, STAGE2_FAIL_REASONS, Candidate, EstimatorResult, QualityReport,
    ReferenceLabel,
)
from physics.verify import verify_candidates
from pipeline.controller import analyze


def assert_invalid(*models):
    for model in models:
        with pytest.raises(ValueError, match="^contract:"):
            model.validate()


def test_observation_invalid_arrays(observation):
    assert_invalid(replace(observation, t_s=observation.t_s.astype("float32")),
                   replace(observation, t_s=observation.t_s + np.nan),
                   replace(observation, t_s=observation.t_s[:1]),
                   replace(observation, t_s=observation.t_s[:0], y=observation.y[:0]),
                   replace(observation, y=observation.y.reshape(1, -1)),
                   replace(observation, y=observation.y.astype("float32")),
                   replace(observation, y=np.full_like(observation.y, -np.inf)),
                   replace(observation, y_err=-np.ones_like(observation.y)),
                   replace(observation, y_err=observation.y[:1]),
                   replace(observation, y_err=observation.y.astype("float32")),
                   replace(observation, y_err=np.full_like(observation.y, np.inf)))


def test_prepared_invalid_arrays_and_meta(prepared):
    assert_invalid(replace(prepared, valid_mask=~prepared.valid_mask),
                   replace(prepared, valid_mask=prepared.valid_mask.astype("int64")),
                   replace(prepared, t_s=prepared.t_s + 1),
                   replace(prepared, t_s=prepared.t_s[::-1]),
                   replace(prepared, t_s=np.full_like(prepared.t_s, np.nan)),
                   replace(prepared, y_norm=prepared.y_norm[:-1]),
                   replace(prepared, y_norm=prepared.y_norm.astype("float32")),
                   replace(prepared, y_norm=np.where(prepared.valid_mask, np.inf, np.nan)),
                   replace(prepared, dt_s=-prepared.dt_s),
                   replace(prepared, dt_s=prepared.dt_s + 1),
                   replace(prepared, dt_s=np.full_like(prepared.dt_s, np.nan)),
                   replace(prepared, flux_rel=prepared.y_norm * 0),
                   replace(prepared, flux_rel=np.ones_like(prepared.y_norm)),
                   replace(prepared, flux_rel=np.full_like(prepared.y_norm, np.inf)),
                   replace(prepared, flux_rel=prepared.y_norm.astype("float32")),
                   replace(prepared, flags={}),
                   replace(prepared, flags={**prepared.flags, "duplicate": np.zeros(1, dtype=bool)}),
                   replace(prepared, flags={**prepared.flags, "duplicate": prepared.dt_s}),
                   replace(prepared, norm_params={}),
                   replace(prepared, meta={}),
                   replace(prepared, meta={**prepared.meta, "freq_resolution": 0.0}))


def test_candidate_quality_and_reference_constraints():
    assert_invalid(Candidate(0, 1, 1), Candidate(8, -0.1, 1), Candidate(8, 1.1, 1),
                   Candidate(float("nan"), 1, 1), Candidate(8, 1, float("inf")),
                   QualityReport("insufficient", {}, [], []),
                   QualityReport("invalid", {}, [], []),
                   QualityReport("limited", {}, [], [LOW_QUALITY]),
                   QualityReport("suitable", {"count": float("nan")}, [], []),
                   ReferenceLabel("obs", "object", -1, "val"),
                   ReferenceLabel("obs", "object", float("inf"), "val"),
                   ReferenceLabel("obs", "object", 1, "invalid"))


def test_estimator_and_reconciliation_constraints(mock_cnn, mock_recon):
    first, partner = mock_recon.clusters
    rejected = replace(first, stage2={**first.stage2, "passed": False,
                                     "fail_reasons": [STAGE2_FAIL_REASONS[0]]})
    rejected.validate()
    assert_invalid(replace(mock_cnn, method="missing"),
                   replace(mock_cnn, status="unrecognised"),
                   replace(mock_cnn, status="skipped"),
                   replace(mock_cnn, status="failed", error="failure"),
                   replace(mock_cnn, candidates=[], status="skipped", error="unrecognised"),
                   replace(mock_cnn, candidates=[], status="failed", error=""),
                   replace(mock_cnn, error="unexpected"),
                   replace(mock_cnn, meta={}),
                   replace(mock_cnn, candidates=list(reversed(mock_cnn.candidates))),
                   replace(first, stage2={**first.stage2, "passed": False}),
                   replace(first, alias_risk=1.5),
                   replace(first, evidence={"unknown": 0.0}),
                   replace(first, evidence={mock_cnn.method: -0.1}),
                   replace(first, score=float("nan")),
                   replace(first, stage2={**first.stage2, "passed": False,
                                          "fail_reasons": ["unrecognised"]}),
                   replace(first, stage2={"passed": True, "fail_reasons": []}),
                   replace(first, harmonic_links=[{"other_id": 2, "ratio": 0.0}]),
                   replace(mock_recon, clusters=[partner, first]),
                   replace(mock_recon, clusters=[first, first]),
                   replace(mock_recon, clusters=[first]),
                   replace(mock_recon, shortlist=[99]),
                   replace(mock_recon, shortlist=[2, 1]),
                   replace(mock_recon, clusters=[rejected, partner]),
                   replace(mock_recon, stage2_passed_any=False))


def test_classical_metadata_validation():
    candidate = Candidate(8, 1.0, 0.5, {"freq_width_hz": 0.05})
    classical = EstimatorResult(LOMB_SCARGLE, "ok", [candidate], meta={
        "runtime_s": 0.0, "periodogram": {"freq_hz": [0.125], "power": [0.5]},
        "freq_resolution": 0.05,
    })
    classical.validate()
    assert_invalid(replace(classical, candidates=[replace(candidate, meta={})]),
                   replace(classical, meta={**classical.meta, "periodogram": {
                       "freq_hz": [0.125], "power": [],
                   }}))


def test_verification_and_analysis_constraints(
    prepared, mock_recon, observation, isolated_runs, stub_config, stub_observation_stages,
):
    verification = verify_candidates(prepared, mock_recon, {})
    candidate = verification.per_candidate[0]
    result = analyze(observation, stub_config)
    supported = replace(result.verdict, label="Supported", reason_codes=[],
                        best_cluster_id=1, best_period_s=8.0)
    assert_invalid(replace(candidate, passed=True),
                   replace(candidate, fit_error=0.0),
                   replace(candidate, heldout_error=0.0),
                   replace(candidate, best_params={}),
                   replace(candidate, decoy_delta_e=[1.0]),
                   replace(candidate, decoy_periods_s=[float("inf")], decoy_delta_e=[1.0]),
                   replace(candidate, status="ok"),
                   replace(verification, complete=True),
                   replace(verification, per_candidate=[candidate, candidate]),
                   replace(result.verdict, reason_codes=[]),
                   replace(result.verdict, best_period_s=10.0),
                   replace(result.verdict, best_cluster_id=1),
                   replace(result.verdict, alternatives=[{"cluster_id": 1}]),
                   replace(result.verdict, evidence_summary={}),
                   replace(supported, reason_codes=[LOW_QUALITY]),
                   replace(supported, best_cluster_id=None),
                   replace(supported, best_period_s=None),
                   replace(result, verdict=None),
                   replace(result, estimators={}),
                   replace(result, reconciliation=mock_recon, verification=verification,
                           verdict=supported))
