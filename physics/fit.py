"""Single-candidate development fitter. Scientific acceptance is defined in §18."""

import numpy as np
from scipy.optimize import differential_evolution

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


def fit_candidate(prep, period_s, cfg, fit_mask=None):
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
    sigma = 1.4826 * np.median(abs(observed[valid] - np.median(observed[valid])))
    if sigma <= 0:
        raise ValueError("zero observed MAD")
    t = prep.t_s[fit_mask]
    policy = cfg["fit"]
    lower, upper = policy["aspect_range"]
    bounds = [(lower, upper), (0., np.pi), (0., 2*np.pi), (0., 2*np.pi), (0., np.pi)]
    rng = np.random.default_rng(cfg["seed"])
    best = None
    for shape in policy["shapes"]:
        def evaluate(values):
            params = candidate_params(shape, period_s, values, policy["cylinder_sides"])
            template = simulate_lightcurve(params, t)
            try:
                a, b = affine_fit(template, y)
            except ValueError:
                return np.inf
            return normalized_error(y, template, a, b, sigma)

        draws = rng.uniform(*np.asarray(bounds).T, size=(policy["coarse_samples"], len(bounds)))
        ranked = sorted(draws, key=evaluate)
        results = [(evaluate(ranked[0]), ranked[0])]
        for restart in range(policy["restarts"]):
            result = differential_evolution(evaluate, bounds, x0=ranked[restart % len(ranked)],
                        rng=rng, maxiter=policy["maxiter"], popsize=policy["popsize"],
                        polish=False, workers=1)
            results.append((result.fun, result.x))
        error, values = min(results, key=lambda item: item[0])
        if np.isfinite(error) and (best is None or error < best["fit_error"]):
            params = candidate_params(shape, period_s, values, policy["cylinder_sides"])
            a, b = affine_fit(simulate_lightcurve(params, t), y)
            best = {"period_s": period_s, "fit_space": space, "params": params,
                    "scale_a": a, "offset_b": b, "sigma_obs": float(sigma),
                    "fit_error": float(error)}
    if best is None:
        raise ValueError("no finite candidate fit")
    return best
