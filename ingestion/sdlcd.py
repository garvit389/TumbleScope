"""SDLCD six-column measurement adapter described in docs/data_audit.md."""
from io import StringIO
import numpy as np
from core.schemas import Observation


def read_sdlcd(source, text):
    rows = np.loadtxt(StringIO(text), comments="#", dtype=np.float64, ndmin=2)
    if rows.shape[0] == 0 or rows.shape[1] != 6:
        raise ValueError("SDLCD requires six measurement columns")
    times = rows[:, 0] - rows[0, 0]
    values, errors = rows[:, 4].copy(), rows[:, 5].copy()
    values[~np.isfinite(values)] = np.nan
    errors[~np.isfinite(errors)] = np.nan
    exposure = rows[:, 1]
    if not np.isfinite(exposure).all() or (exposure < 0).any():
        raise ValueError("invalid exposure times")
    return Observation(source.stem, times, values, errors, "mag", {
        "source_file": source.name, "object_id": source.stem.split("_")[0],
        "exposure_s": exposure.tolist(), "assumptions": [],
    })
