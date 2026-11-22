import numpy as np
import pytest

from core.config import load_config, resolve_period_bounds
from core.schemas import Observation
from physics.fit import affine_fit, candidate_params, fit_candidate, normalized_error
from physics.simulator import simulate_lightcurve
from preprocessing.prepare import prepare


def test_affine_and_error():
    x = np.arange(4, dtype=float)
    assert affine_fit(x, 2*x+5) == pytest.approx((2,5))
    assert normalized_error(2*x+6, x, 2, 5, 2.) == .5
    with pytest.raises(ValueError, match="rank deficient"):
        affine_fit(np.ones(4), x)
    with pytest.raises(ValueError, match="undefined"):
        normalized_error(x, x, 1, 0, 0)


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
