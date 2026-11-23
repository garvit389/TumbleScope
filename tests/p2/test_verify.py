import numpy as np
import pytest

from core.config import load_config, resolve_period_bounds
from core.schemas import Observation
from physics.fit import affine_fit, candidate_params, fit_candidate, normalized_error
from physics.simulator import simulate_lightcurve
from physics.verify import _heldout_masks, _score_period
from preprocessing.prepare import prepare


def test_affine_and_error():
    x = np.arange(4, dtype=float)
    assert affine_fit(x, 2*x+5) == pytest.approx((2,5))
    assert normalized_error(2*x+6, x, 2, 5, 2.) == .5
    with pytest.raises(ValueError, match="rank deficient"):
        affine_fit(np.ones(4), x)
    with pytest.raises(ValueError, match="undefined"):
        normalized_error(x, x, 1, 0, 0)


def test_contiguous_fold_masks(prepared):
    from dataclasses import replace

    times = prepared.t_s.copy()
    times[1] = times[0]
    prep = replace(prepared, t_s=times)
    masks = _heldout_masks(prep, 3)
    np.testing.assert_array_equal(np.sum(masks, axis=0), prep.valid_mask)
    assert masks[0][0] and masks[0][1]
    for left, right in zip(masks, masks[1:]):
        assert times[left].max() < times[right].min()
    with pytest.raises(ValueError, match="too few distinct"):
        _heldout_masks(prep, len(times) + 1)


def test_heldout_affine_is_frozen(prepared):
    from dataclasses import replace
    from physics.fit import _fit_family

    cfg = dict(load_config().verification, period_min_s=2., period_max_s=20.)
    hold = _heldout_masks(prepared, 3)[0]
    changed = prepared.y_norm.copy()
    changed[hold] += 10
    fits = [_fit_family(p, 8., cfg, p.valid_mask & ~hold, "box", None)
            for p in (prepared, replace(prepared, y_norm=changed))]
    assert fits[0]["params"] == fits[1]["params"]
    for key in ("scale_a", "offset_b"):
        assert fits[0][key] == fits[1][key]


def test_fit_and_heldout_errors_use_all_valid_mad(monkeypatch):
    from dataclasses import replace

    from physics import fit, verify

    config = load_config()
    values = np.array([1., 2., 7., 3., 8., 9., np.nan])
    obs = Observation("nonunit-mad", np.arange(len(values), dtype=float), values, None, "unknown")
    prep = replace(prepare(obs, config.preprocess), y_norm=values)
    cfg = dict(config.verification, period_min_s=2., period_max_s=20.)
    cfg["fit"] = dict(cfg["fit"], shapes=["box"], coarse_samples=1, restarts=0)

    def simulate(params, times):
        return times % 3

    monkeypatch.setattr(fit, "simulate_lightcurve", simulate)
    monkeypatch.setattr(verify, "simulate_lightcurve", simulate)
    masks = _heldout_masks(prep, 2)
    fit_error, heldout_error, _, _ = _score_period(prep, 8., cfg, masks, None)
    sigma = 1.4826 * 3
    # Each fit block has MAD 1. The all-valid median is 5 and its MAD is 3.
    training = fit_candidate(prep, 8., cfg, ~masks[0])
    assert training["sigma_obs"] == pytest.approx(sigma)
    assert training["fit_error"] == pytest.approx(np.sqrt(8 / 9) / sigma)
    assert fit_error == pytest.approx(np.sqrt(8 / 9) / sigma)
    assert heldout_error == pytest.approx(np.sqrt(12) / sigma)


def test_family_errors_and_fit_space(prepared, monkeypatch):
    from physics import verify

    masks = _heldout_masks(prepared, 3)
    seen = []

    def fit(prep, period, cfg, mask, shape, deadline, *, sigma_obs):
        seen.append(mask.copy())
        return {"fit_space": "y_norm", "scale_a": 1., "offset_b": 0.,
                "sigma_obs": 1., "fit_error": float(shape == "cylinder"),
                "params": shape}

    monkeypatch.setattr(verify, "_fit_family", fit)
    monkeypatch.setattr(verify, "simulate_lightcurve",
                        lambda shape, t: np.full(len(t), float(shape == "box")))
    fit_error, heldout, params, residuals = _score_period(
        prepared, 8., {"fit": {"shapes": ["box", "cylinder"]}}, masks, None)
    expected = {family: np.mean([np.sqrt(np.mean((prepared.y_norm[mask] - level)**2))
                                 for mask in masks])
                for family, level in (("box", 1), ("cylinder", 0))}
    assert fit_error == 0 and heldout == pytest.approx(expected["box"])
    assert params["fit_space"] == "y_norm" and params["geometry_errors"] == pytest.approx(expected)
    assert params["shape"] == min(expected, key=expected.get)
    np.testing.assert_allclose(residuals[prepared.valid_mask],
                               prepared.y_norm[prepared.valid_mask] - 1)
    for i, hold in enumerate(masks):
        for mask in seen[2*i:2*i+2]:
            np.testing.assert_array_equal(mask, prepared.valid_mask & ~hold)

    monkeypatch.setattr(verify, "normalized_error", lambda *args: np.inf)
    with pytest.raises(ValueError, match="nonfinite held-out error"):
        _score_period(prepared, 8., {"fit": {"shapes": ["box"]}}, masks, None)


@pytest.mark.parametrize("kind,space", [("flux","flux_rel"),("mag","flux_rel"),("unknown","y_norm")])
def test_single_period_fit(kind, space):
    config = load_config()
    low, high = resolve_period_bounds(config.period_bounds)
    cfg = dict(config.verification, period_min_s=low, period_max_s=high)
    t = np.linspace(0, 55, 128)
    params = candidate_params("box", 11., [1.5,.7,.2,.5,.9], 16)
    brightness = 3 + 2*simulate_lightcurve(params, t)
    if kind == "mag":
        brightness = -2.5*np.log10(brightness) + 15.
    brightness[7] = np.nan
    prep = prepare(Observation("fit", t, brightness, None, kind, {}), config.preprocess)
    fit_rows = t < 40
    result = fit_candidate(prep, 11., cfg, fit_rows)
    assert result["fit_space"] == space
    assert result["period_s"] == result["params"].phase_cycle_s == 11.
    assert result == fit_candidate(prep, 11., cfg, fit_rows)
    template = simulate_lightcurve(result["params"], t)
    rows = fit_rows & prep.valid_mask
    y = getattr(prep, space)
    np.testing.assert_allclose(affine_fit(template[rows], y[rows]),
                               [result["scale_a"], result["offset_b"]])
    assert result["fit_error"] < np.std(y[rows])/result["sigma_obs"]
    # Scoring other rows uses frozen parameters.
    error = normalized_error(y[~fit_rows], template[~fit_rows], result["scale_a"],
                             result["offset_b"], result["sigma_obs"])
    assert np.isfinite(error)
    with pytest.raises(ValueError, match="few"):
        fit_candidate(prep, 11., cfg, np.zeros(t.size, dtype=bool))


@pytest.fixture
def protocol_cfg():
    return dict(load_config().verification, period_min_s=2., period_max_s=24.,
                window_result={"freq_hz": [0.], "window_power": [1.], "peaks": [],
                               "freq_resolution": None})


def controlled_window(monkeypatch):
    from physics import verify
    original = verify._verify_policy

    def policy(prep, cfg):
        window = {"freq_hz": [0.], "window_power": [1.], "peaks": [],
                  "freq_resolution": prep.meta["freq_resolution"]}
        bounds, ratios, masks, _ = original(prep, {**cfg, "window_result": window})
        return bounds, ratios, masks, {"freq_resolution": .001, "peaks": []}

    monkeypatch.setattr(verify, "_verify_policy", policy)


def test_decoy_exclusions():
    from physics.verify import _eligible_decoys

    periods, notes = _eligible_decoys(
        8., [16.], {"freq_resolution": 1/32, "peaks": [{"freq_hz": .25}]},
        (2., 32.), [1., 2., .5, .25, .25, .125, 4.], 1.)
    assert periods == [2.]
    assert [note.partition("(")[2] for note in notes] == [
        "candidate frequency)", "harmonic frequency)", "alias offset)",
        "duplicate period)", "outside period bounds)", "harmonic frequency)",
    ]
    with pytest.raises(ValueError, match="frequency resolution"):
        _eligible_decoys(8., [], {"peaks": []}, (2., 24.), [1.37], 1.)


def test_nondefault_shared_window_changes_decoy_eligibility(monkeypatch):
    from classical_analysis.window import compute_window
    from core.schemas import Cluster, ReconciliationResult
    from physics import verify

    times = np.arange(100, dtype=float) * 2
    prep = prepare(Observation("window", times, np.cos(times / 2), None, "unknown"), {})
    bounds = {"period_min_s": 2., "period_max_s": 30.}
    default = compute_window(prep, bounds)
    changed = compute_window(prep, {**bounds, "n_peaks": 0})
    period = 3.2892892892892895
    args = (period, [], (2., 30.), [1.37, .61], 1.)
    default_decoys, _ = verify._eligible_decoys(args[0], args[1], default, *args[2:])
    changed_decoys, _ = verify._eligible_decoys(args[0], args[1], changed, *args[2:])
    assert default_decoys == [period * 1.37]
    assert changed_decoys == [period * 1.37, period * .61]
    cfg = {**load_config().verification, **bounds, "window_result": changed}
    assert verify._verify_policy(prep, cfg)[3] is changed
    cluster = Cluster(1, period, {}, {}, 0., [], 1.,
                      {"passed": True, "cycles": prep.meta["baseline_s"] / period, "fail_reasons": []})
    recon = ReconciliationResult([cluster], [1], True, [])
    def score(prep, period, cfg, masks, deadline, *, sigma_obs):
        best = {"shape": "box", "geometry_errors": {"box": .1},
                "fit_space": "y_norm", "scale_a": 1., "offset_b": 0.}
        return .1, .1, best, np.zeros_like(prep.y_norm)
    monkeypatch.setattr(verify, "_score_period", score)
    changed_result = verify.verify_candidates(prep, recon, cfg)
    default_result = verify.verify_candidates(prep, recon, {**cfg, "window_result": default})
    assert changed_result.complete and default_result.complete
    assert changed_result.per_candidate[0].decoy_periods_s == changed_decoys
    assert default_result.per_candidate[0].decoy_periods_s == default_decoys


@pytest.mark.parametrize("window", [None, {}, {"freq_hz": [0.], "window_power": [1.],
                                                  "peaks": [{"freq_hz": .5}], "freq_resolution": None}])
def test_missing_or_invalid_shared_window_is_incomplete(prepared, mock_recon, protocol_cfg, window):
    from physics.verify import verify_candidates

    result = verify_candidates(prepared, mock_recon, {**protocol_cfg, "window_result": window})
    result.validate()
    assert not result.complete
    assert [item.cluster_id for item in result.per_candidate] == [1, 2]
    assert all(item.status == "failed" and not item.passed and item.fit_error is None
               for item in result.per_candidate)
    assert all("window_result" in item.notes[-1] for item in result.per_candidate)


def test_equal_optimizer_budget(prepared, mock_recon, protocol_cfg, monkeypatch):
    from unittest.mock import Mock

    from physics import verify

    seen = []
    scale = Mock(wraps=verify._observed_sigma)
    monkeypatch.setattr(verify, "_observed_sigma", scale)
    controlled_window(monkeypatch)

    def score(prep, period, cfg, masks, deadline, *, sigma_obs):
        seen.append((period, cfg, masks, deadline, sigma_obs))
        return .4, .5 if period in (8., 16.) else .75, {
            "shape": "box", "geometry_errors": {"box": .5}, "fit_space": "y_norm",
            "scale_a": 1., "offset_b": 0.,
        }, np.where(prep.valid_mask, .5, np.nan)

    monkeypatch.setattr(verify, "_score_period", score)
    result = verify.verify_candidates(prepared, mock_recon, protocol_cfg)
    scale.assert_called_once_with(prepared)
    assert result.complete and all(record.passed for record in result.per_candidate)
    assert [p for p, *_ in seen] == [8., 8*1.37, 8*.61, 16., 16*1.37, 16*.61]
    assert all(cfg is protocol_cfg and masks is seen[0][2] for _, cfg, masks, _, _ in seen)
    assert seen[0][3] == seen[1][3] == seen[2][3]
    assert seen[3][3] == seen[4][3] == seen[5][3]
    assert all(sigma is seen[0][4] for _, _, _, _, sigma in seen)
    assert all(record.decoy_delta_e == [.25, .25] for record in result.per_candidate)
    assert any("even shorter-period cycles" in note for note in result.per_candidate[0].notes)


def test_zero_observed_mad_marks_all_required_fits_failed(prepared, mock_recon, protocol_cfg, monkeypatch):
    from dataclasses import replace

    from physics.verify import verify_candidates

    controlled_window(monkeypatch)
    prep = replace(prepared, y_norm=np.where(prepared.valid_mask, 4., np.nan))
    result = verify_candidates(prep, mock_recon, protocol_cfg)
    assert not result.complete
    assert [record.cluster_id for record in result.per_candidate] == [1, 2]
    for record in result.per_candidate:
        assert record.status == "failed"
        assert not record.passed
        assert record.fit_error is None and record.heldout_error is None
        assert "zero observed MAD" in record.notes


def test_worst_decoy_and_gate_equality(protocol_cfg):
    from physics.verify import _passes

    assert _passes(.85, [.10, .25], protocol_cfg)
    assert not _passes(.50, [.25, -.25], protocol_cfg)
    assert not _passes(np.nextafter(.85, np.inf), [.10], protocol_cfg)


def test_required_ids_and_harmonic_direction(prepared, mock_recon, protocol_cfg, monkeypatch):
    from physics import verify

    short, long = mock_recon.clusters
    controlled_window(monkeypatch)
    for short_e, long_e, preference in ((0., .05, "P"), (.05, 0., "2P"),
                                         (.05, .05, "undecided")):
        def score(prep, period, cfg, masks, deadline, *, sigma_obs):
            error = {8.: short_e, 16.: long_e}.get(period, 1.)
            best = {"shape": "box", "geometry_errors": {"box": error},
                    "fit_space": "y_norm", "scale_a": 1., "offset_b": 0.}
            return error, error, best, np.zeros_like(prep.y_norm)

        monkeypatch.setattr(verify, "_score_period", score)
        result = verify.verify_candidates(prepared, mock_recon, protocol_cfg)
        assert result.complete
        for record, other in zip(result.per_candidate, (long, short)):
            assert record.harmonic_comparison == {
                "partner_id": other.cluster_id, "prefers": preference,
                "delta_e": long_e - short_e, "reason": None,
            }


def test_v3_multiple_partners_preserve_required_ids(prepared, mock_recon):
    from dataclasses import replace
    from physics.verify import verify_candidates

    lead, longer = mock_recon.clusters
    shorter = replace(longer, cluster_id=3, period_s=4., score=.5,
                      harmonic_links=[{"other_id": 1, "ratio": 2.}])
    lead = replace(lead, harmonic_links=[*lead.harmonic_links, {"other_id": 3, "ratio": .5}])
    recon = replace(mock_recon, clusters=[lead, longer, shorter])
    result = verify_candidates(prepared, recon, {})
    assert [record.cluster_id for record in result.per_candidate] == [1, 2, 3]
    assert not result.complete and result.per_candidate[0].status == "failed"
    assert any("contract amendment" in note for note in result.per_candidate[0].notes)
    assert all(record.harmonic_comparison is None or "pairs" not in record.harmonic_comparison
               for record in result.per_candidate)


def test_missing_policy_and_timeout(prepared, mock_recon, protocol_cfg, monkeypatch):
    from physics import verify

    missing = verify.verify_candidates(prepared, mock_recon, {})
    assert not missing.complete
    assert [record.cluster_id for record in missing.per_candidate] == [1, 2]
    assert all(record.status == "failed" and record.best_params is None
               for record in missing.per_candidate)
    controlled_window(monkeypatch)
    attempts = []

    def score(prep, period, cfg, masks, deadline, *, sigma_obs):
        attempts.append(period)
        raise TimeoutError("timeout")

    monkeypatch.setattr(verify, "_score_period", score)
    timed = verify.verify_candidates(prepared, mock_recon, protocol_cfg)
    assert attempts == [8., 16.]
    assert not timed.complete
    for record in timed.per_candidate:
        assert record.status == "failed" and not record.passed and record.fit_error is None
        assert "timeout" in record.notes


def test_verification_replay(prepared, mock_recon, protocol_cfg, monkeypatch):
    from physics import verify

    controlled_window(monkeypatch)
    cfg = dict(protocol_cfg, fit={**protocol_cfg["fit"], "coarse_samples": 8,
                                  "restarts": 1, "popsize": 4, "maxiter": 5})
    first = verify.verify_candidates(prepared, mock_recon, cfg)
    second = verify.verify_candidates(prepared, mock_recon, cfg)
    assert first.complete and second.complete
    assert first.per_candidate == second.per_candidate


def test_odd_even_diagnostics(prepared):
    from physics.verify import _odd_even_notes

    notes = _odd_even_notes(prepared, np.ones_like(prepared.y_norm), 1e6)
    assert notes == [f"even shorter-period cycles: n={prepared.valid_mask.sum()}, normalized RMS=1",
                     "odd shorter-period cycles: n=0, normalized RMS=unavailable"]


def test_clean_eval_denominators():
    from physics.eval_verification import _summarize

    rows = [
        {"status": "ok", "passed": True, "decoy_delta_e": [.4, .2], "runtime_s": 2.},
        {"status": "ok", "passed": False, "decoy_delta_e": [.4, -.2], "runtime_s": 4.},
        {"status": "failed", "notes": ["timeout"], "runtime_s": 30.},
    ]
    summary = _summarize(rows)
    assert summary["attempted"] == 3 and summary["completed"] == 2
    assert summary["beat_all_decoys"] == 1 and summary["wins_per_completed"] == .5
    assert summary["wins_per_attempted"] == 1/3
    assert summary["failure_counts"] == {"timeout": 1}
    assert summary["completed_median_runtime_s"] == 3.
