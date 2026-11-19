"""Phase 0 entry point for the bundled synthetic observation."""

from pathlib import Path
from typing import Optional

import numpy as np

from core.errors import IngestionError
from core.schemas import Observation


_DEMO = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "synthetic_sine.csv"


def load_observation(path: str, fmt: Optional[str] = None) -> Observation:
    try:
        source = Path(path).resolve()
        if source != _DEMO or fmt not in (None, "csv"):
            raise IngestionError("Phase 0 ingestion accepts only the bundled synthetic CSV.")
        rows = np.loadtxt(source, delimiter=",", skiprows=1, dtype=np.float64, ndmin=2)
        if rows.shape[1] != 2:
            raise ValueError("expected two columns")
        result = Observation(source.stem, rows[:, 0], rows[:, 1], None, "unknown",
                             {"source_file": source.name})
        result.validate()
    except (OSError, TypeError, ValueError) as exc:
        raise IngestionError(f"Could not read the synthetic fixture: {exc}") from exc
    return result
