"""Single-candidate development fitter. Scientific acceptance is defined in §18."""

from time import perf_counter

import numpy as np
from scipy.optimize import differential_evolution

from core.schemas import PreparedObservation
from physics.params import SimParams
from physics.simulator import simulate_lightcurve


def affine_fit(template, observed):
    template, observed = np.asarray(template), np.asarray(observed)
    if (template.shape != observed.shape or template.ndim != 1 or len(template) < 2
            or not np.isfinite(template).all() or not np.isfinite(observed).all()):
        raise ValueError("invalid affine fit data")
    design = np.column_stack((template, np.ones_like(template)))
    coefficients, _, rank, _ = np.linalg.lstsq(design, observed, rcond=None)
    if rank != 2:
        raise ValueError("rank deficient template")
    return tuple(map(float, coefficients))


def normalized_error(observed, template, scale, offset, sigma):
    if not np.isfinite(sigma) or sigma <= 0 or not len(observed):
        raise ValueError("undefined normalized error")
    return float(np.sqrt(np.mean((observed - (scale * template + offset))**2)) / sigma)


def candidate_params(shape, period, values, sides):
    aspect, polar, azimuth, phase, angle = values
    dimensions = (aspect, 1., 1.) if shape == "box" else (.5, aspect)
    reflectivity = (1.,) * (6 if shape == "box" else sides+2)
    return SimParams(shape, dimensions, reflectivity, polar, azimuth, phase, period,
                     (1., 0., 0.), (np.cos(angle), np.sin(angle), 0.), sides)


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and perf_counter() >= deadline:
        raise TimeoutError("timeout")


def _observed_sigma(prep: PreparedObservation) -> float:
    series = prep.flux_rel if prep.flux_rel is not None else prep.y_norm
    observed = series[prep.valid_mask]
    if not len(observed):
        raise ValueError("zero observed MAD")
    sigma_obs = 1.4826 * np.median(np.abs(observed - np.median(observed)))
    if not np.isfinite(sigma_obs) or sigma_obs <= 0:
        raise ValueError("zero observed MAD")
    return float(sigma_obs)


def _fit_family(prep: PreparedObservation, period_s: float, cfg: dict,
                fit_mask: np.ndarray, shape: str, deadline: float | None, *,
                rng: np.random.Generator | None = None,
                sigma_obs: float | None = None) -> dict:
    _check_deadline(deadline)
    prep.validate()
    if not cfg["period_min_s"] <= period_s <= cfg["period_max_s"]:
        raise ValueError("candidate outside period bounds")
    space = "flux_rel" if prep.flux_rel is not None else "y_norm"
    observed = getattr(prep, space)
    valid = prep.valid_mask
    if fit_mask is None:
        fit_mask = valid.copy()
    else:
        fit_mask = np.asarray(fit_mask)
        if fit_mask.dtype != bool or fit_mask.shape != valid.shape:
            raise ValueError("invalid fit mask")
        fit_mask = fit_mask & valid
    if fit_mask.sum() < 2:
        raise ValueError("too few fit samples")
    y = observed[fit_mask]
    if sigma_obs is None:
        sigma_obs = _observed_sigma(prep)
    t = prep.t_s[fit_mask]
    policy = cfg["fit"]
    lower, upper = policy["aspect_range"]
    bounds = [(lower, upper), (0., np.pi), (0., 2*np.pi), (0., 2*np.pi), (0., np.pi)]
    if rng is None:
        rng = np.random.default_rng(cfg["seed"])

    def evaluate(values):
        _check_deadline(deadline)
        params = candidate_params(shape, period_s, values, policy["cylinder_sides"])
        template = simulate_lightcurve(params, t)
        try:
            a, b = affine_fit(template, y)
        except ValueError:
            return np.inf
        return normalized_error(y, template, a, b, 1.)

    draws = rng.uniform(*np.asarray(bounds).T, size=(policy["coarse_samples"], len(bounds)))
    ranked = sorted(((evaluate(draw), draw) for draw in draws), key=lambda item: item[0])
    results = [ranked[0]]
    for restart in range(policy["restarts"]):
        result = differential_evolution(
            evaluate, bounds, x0=ranked[restart % len(ranked)][1], rng=rng,
            maxiter=policy["maxiter"], popsize=policy["popsize"], polish=False, workers=1,
        )
        results.append((result.fun, result.x))
    _check_deadline(deadline)
    error, values = min(results, key=lambda item: item[0])
    fit_error = float(error) / sigma_obs
    if not np.isfinite(fit_error):
        raise ValueError("no finite candidate fit")
    params = candidate_params(shape, period_s, values, policy["cylinder_sides"])
    a, b = affine_fit(simulate_lightcurve(params, t), y)
    _check_deadline(deadline)
    return {"period_s": period_s, "fit_space": space, "params": params,
            "scale_a": a, "offset_b": b, "sigma_obs": sigma_obs,
            "fit_error": fit_error}


def fit_candidate(prep, period_s, cfg, fit_mask=None):
    rng = np.random.default_rng(cfg["seed"])
    prep.validate()
    sigma_obs = _observed_sigma(prep)
    fits = []
    for shape in cfg["fit"]["shapes"]:
        try:
            fits.append(_fit_family(prep, period_s, cfg, fit_mask, shape, None,
                                    rng=rng, sigma_obs=sigma_obs))
        except ValueError as exc:
            if str(exc) != "no finite candidate fit":
                raise
    if not fits:
        raise ValueError("no finite candidate fit")
    return min(fits, key=lambda fit: fit["fit_error"])
