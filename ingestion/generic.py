"""Header-labelled CSV or whitespace observations."""
from io import StringIO
import csv

import numpy as np
import pandas as pd

from core.schemas import Observation


def read_generic(source, text):
    first = next((line for line in text.splitlines()
                  if line.strip() and not line.lstrip().startswith("#")), None)
    if first is None:
        raise ValueError("file contains no measurements")
    separator = "," if "," in first else r"\s+"
    headers = next(csv.reader([first])) if separator == "," else first.split()
    headers = [name.strip().lower() for name in headers]
    if len(headers) != len(set(headers)):
        raise ValueError("duplicate column names")
    frame = pd.read_csv(StringIO(text), sep=separator, comment="#")
    if not isinstance(frame.index, pd.RangeIndex):
        raise ValueError("measurement rows contain more fields than the header")
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    if frame.empty or frame.columns.duplicated().any():
        raise ValueError("expected unique headers and at least one measurement")
    time_columns = [name for name in ("t_s", "time_s", "time", "mjd", "jd") if name in frame]
    brightness_columns = [name for name in ("mag", "magnitude", "flux", "adu", "brightness", "y") if name in frame]
    if len(time_columns) != 1 or len(brightness_columns) != 1:
        raise ValueError("expected one time column and one brightness column")
    time_key, key = time_columns[0], brightness_columns[0]
    times = pd.to_numeric(frame[time_key], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(times).all():
        raise ValueError("timestamps must be finite")
    times = times - times[0]
    if time_key in ("mjd", "jd"):
        times *= 86400.
    values = pd.to_numeric(frame[key], errors="raise").to_numpy(dtype=np.float64)
    values[~np.isfinite(values)] = np.nan
    errors = None
    error_keys = [name for name in (key + "_err", "y_err", "uncertainty") if name in frame]
    error_keys = list(dict.fromkeys(error_keys))
    if len(error_keys) > 1:
        raise ValueError("ambiguous uncertainty columns")
    if error_keys:
        errors = pd.to_numeric(frame[error_keys[0]], errors="raise").to_numpy(dtype=np.float64)
        errors[~np.isfinite(errors)] = np.nan
    kind = "mag" if key in ("mag", "magnitude") else "flux" if key in ("flux", "adu") else "unknown"
    assumptions = []
    if time_key == "time":
        assumptions.append("Unqualified time column assumed to contain seconds.")
    if kind == "unknown":
        assumptions.append("Brightness convention unknown; increasing values assumed brighter.")
    return Observation(source.stem, times, values, errors, kind,
                       {"source_file": source.name, "assumptions": assumptions})
