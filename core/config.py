"""YAML configuration and period-bound ownership (CONTRACTS.md §10)."""

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from typing import NamedTuple

import yaml

from core.schemas import SCHEMA_VERSION


_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
_MODULES = ("preprocess", "classical", "reconcile", "simulation", "cnn", "verification", "verdict")
_BOUND_KEYS = {"min_period_s", "max_period_s", "p_min_s", "p_max_s", "period_min_s", "period_max_s"}


class PeriodBounds(NamedTuple):
    min_period_s: float
    max_period_s: float


def _read_yaml(path: Path) -> dict:
    try:
        with path.open(encoding="utf-8") as source:
            data = yaml.safe_load(source)
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"contract: cannot read configuration {path.name}: {exc}") from exc
    if not isinstance(data, dict) or not all(isinstance(key, str) for key in data):
        raise ValueError(f"contract: {path.name} must contain a mapping with string keys")
    return data


def _default_bounds() -> dict:
    return _read_yaml(_CONFIG_DIR / "period_bounds.yaml")


def _contains_bounds(value) -> bool:
    if isinstance(value, dict):
        return bool(_BOUND_KEYS.intersection(value)) or any(
            _contains_bounds(item) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_bounds(item) for item in value)
    return False


@dataclass
class PipelineConfig:
    preprocess: dict = field(default_factory=dict)
    classical: dict = field(default_factory=dict)
    reconcile: dict = field(default_factory=dict)
    simulation: dict = field(default_factory=dict)
    cnn: dict = field(default_factory=dict)
    verification: dict = field(default_factory=dict)
    verdict: dict = field(default_factory=dict)
    pipeline: dict = field(default_factory=lambda: {"validate_outputs": True})
    period_bounds: dict = field(default_factory=_default_bounds)


def load_config(path: str | Path | None = None) -> PipelineConfig:
    manifest = _CONFIG_DIR / "default.yaml" if path is None else Path(path)
    files = _read_yaml(manifest)
    if _contains_bounds(files):
        raise ValueError(f"contract: period bounds duplicated in {manifest.name}")
    sections = {}
    for section in _MODULES:
        filename = files.get(section)
        if not isinstance(filename, str):
            raise ValueError(f"contract: missing config filename for {section}")
        data = _read_yaml(manifest.parent / filename)
        if _contains_bounds(data):
            raise ValueError(f"contract: period bounds duplicated in {filename}")
        sections[section] = data
    pipeline = files.get("pipeline", {})
    if not isinstance(pipeline, dict) or type(pipeline.get("validate_outputs", True)) is not bool:
        raise ValueError("contract: invalid pipeline configuration")
    data = _read_yaml(manifest.parent / "period_bounds.yaml")
    if data.get("status") == "TBD":
        if set(data) != {"status", "min_period_s", "max_period_s"} or any(
            data[key] is not None for key in ("min_period_s", "max_period_s")
        ):
            raise ValueError("contract: unset period bounds must be null")
    else:
        resolve_period_bounds(data)
    return PipelineConfig(**sections, pipeline={"validate_outputs": True, **pipeline},
                          period_bounds=data)


def resolve_period_bounds(data: dict) -> PeriodBounds:
    if data.get("status") == "TBD":
        raise ValueError("contract: period bounds not set")
    lower, upper = data.get("min_period_s"), data.get("max_period_s")
    if data.get("status") != "SET" or type(lower) not in (float, int) or type(upper) not in (float, int):
        raise ValueError("contract: invalid period bounds")
    try:
        finite = isfinite(lower) and isfinite(upper)
    except OverflowError:
        finite = False
    if not finite or not 0 < lower < upper:
        raise ValueError("contract: invalid period bounds")
    return PeriodBounds(float(lower), float(upper))


def get_period_bounds() -> PeriodBounds:
    return resolve_period_bounds(_default_bounds())


def config_hash(cfg: PipelineConfig) -> str:
    payload = {"schema_version": SCHEMA_VERSION, **asdict(cfg)}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(encoded.encode("utf-8")).hexdigest()
