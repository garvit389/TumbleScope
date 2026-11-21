"""Convex facets in body coordinates."""

import numpy as np

from physics.params import SimParams


def facets(params: SimParams):
    if params.shape == "box":
        x, y, z = params.dimensions
        # Opposite pairs in +x,-x,+y,-y,+z,-z order.
        normals = np.repeat(np.eye(3), 2, axis=0)
        normals[1::2] *= -1
        areas = np.repeat([y * z, x * z, x * y], 2)
    else:
        radius, height = params.dimensions
        count = params.cylinder_sides
        angles = 2 * np.pi * np.arange(count) / count
        sides = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(count)))
        normals = np.vstack((sides, [0., 0., 1.], [0., 0., -1.]))
        side = (2 * np.sin(np.pi / count) * radius) * height
        cap = (count / 2 * np.sin(2 * np.pi / count) * radius) * radius
        areas = np.r_[np.full(count, side), cap, cap]
    if not np.isfinite(areas).all() or np.any(areas <= 0):
        raise ValueError("facet areas outside representable range")
    return normals, areas
