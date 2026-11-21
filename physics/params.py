"""Inputs to the fixed-geometry, single-axis simulator."""

from dataclasses import dataclass

import numpy as np


def _real_array(value):
    raw = np.asarray(value)
    if np.iscomplexobj(raw) or (raw.dtype == object and any(np.iscomplexobj(v) for v in raw.flat)):
        raise ValueError("complex physical parameters are unsupported")
    return np.asarray(raw, dtype=np.float64)


def unit_direction(vector):
    values = _real_array(vector)
    if values.shape != (3,) or not np.isfinite(values).all() or not np.any(values):
        raise ValueError("direction must be a finite nonzero three-vector")
    scaled = values / np.max(np.abs(values))
    return scaled / np.linalg.norm(scaled)


@dataclass(frozen=True)
class SimParams:
    shape: str
    dimensions: tuple[float, ...]
    reflectivity: tuple[float, ...]
    axis_polar_rad: float
    axis_azimuth_rad: float
    phase_rad: float
    phase_cycle_s: float
    sun: tuple[float, float, float]
    observer: tuple[float, float, float]
    cylinder_sides: int = 16

    def __post_init__(self):
        sizes = {"box": 3, "cylinder": 2}
        if self.shape not in sizes:
            raise ValueError("shape must be box or cylinder")
        if type(self.cylinder_sides) is not int or self.cylinder_sides < 3:
            raise ValueError("a prism needs at least three side facets")
        for name, count in (("dimensions", sizes[self.shape]),
                            ("reflectivity", 6 if self.shape == "box" else self.cylinder_sides + 2)):
            values = _real_array(getattr(self, name))
            if values.shape != (count,) or not np.isfinite(values).all():
                raise ValueError(f"invalid {name} for {self.shape}")
            if name == "dimensions" and np.any(values <= 0):
                raise ValueError("dimensions must be positive")
            if name == "reflectivity" and np.any((values < 0) | (values > 1)):
                raise ValueError("reflectivity must be in [0,1]")
            object.__setattr__(self, name, tuple(map(float, values)))
        for name in ("phase_cycle_s", "axis_polar_rad", "axis_azimuth_rad", "phase_rad"):
            value = np.asarray(getattr(self, name))
            if value.ndim or value.dtype.kind not in "iuf" or not np.isfinite(value):
                raise ValueError(f"{name} must be a finite scalar")
            object.__setattr__(self, name, float(value))
        if self.phase_cycle_s <= 0:
            raise ValueError("phase_cycle_s must be positive")
        for name in ("sun", "observer"):
            object.__setattr__(self, name, tuple(unit_direction(getattr(self, name))))
