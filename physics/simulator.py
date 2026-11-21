"""Relative diffuse flux for a rigid body with fixed Sun and observer."""

import numpy as np

from physics.geometry import facets
from physics.params import SimParams


def simulate_lightcurve(params: "SimParams", t_s: np.ndarray) -> np.ndarray:
    times = np.asarray(t_s)
    if times.ndim != 1 or times.dtype.kind not in "iuf" or not np.isfinite(times).all():
        raise ValueError("t_s must be a finite one-dimensional real array")
    times = times.astype(np.float64, copy=False)
    normals, areas = facets(params)
    polar, azimuth = params.axis_polar_rad, params.axis_azimuth_rad
    axis = np.array([np.sin(polar) * np.cos(azimuth), np.sin(polar) * np.sin(azimuth),
                     np.cos(polar)])
    angle = 2 * np.pi * (np.remainder(times, params.phase_cycle_s) / params.phase_cycle_s)
    angle += np.remainder(params.phase_rad, 2 * np.pi)
    parallel = np.outer(normals @ axis, axis)
    rotated = (parallel + np.cos(angle)[:, None, None] * (normals - parallel)
               + np.sin(angle)[:, None, None] * np.cross(axis, normals))
    lit = np.maximum(rotated @ params.sun, 0.)
    visible = np.maximum(rotated @ params.observer, 0.)
    flux = (lit * visible) @ (areas * params.reflectivity)
    if not np.isfinite(flux).all():
        raise ValueError("nonfinite simulated brightness")
    return flux
