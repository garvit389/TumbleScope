"""Observation adapters. See CONTRACTS.md for the public interface."""
from pathlib import Path
from typing import Optional

from core.errors import IngestionError
from core.schemas import Observation
from ingestion.generic import read_generic
from ingestion.sdlcd import read_sdlcd


def load_observation(path: str, fmt: Optional[str] = None) -> Observation:
    try:
        source = Path(path)
        text = source.read_text(encoding="utf-8-sig")
        if not text.strip():
            raise ValueError("file is empty")
        kind = fmt.lower() if fmt is not None else None
        if kind is None:
            kind = "sdlcd" if any("Per[s]" in line for line in text.splitlines() if line.startswith("#")) else "csv"
        if kind == "sdlcd":
            result = read_sdlcd(source, text)
        elif kind in ("csv", "txt", "generic"):
            result = read_generic(source, text)
        else:
            raise ValueError(f"unsupported format {fmt!r}")
        result.validate()
        return result
    except (OSError, UnicodeError, TypeError, ValueError, OverflowError) as exc:
        raise IngestionError(f"Could not read {path}: {exc}") from exc
