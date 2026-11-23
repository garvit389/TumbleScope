"""Per-session reuse of completed analyses of original upload bytes."""

from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import sys
from typing import Optional

from core.config import PipelineConfig, config_hash
from core.schemas import SCHEMA_VERSION, AnalysisResult, Observation, ProgressCallback, report_progress
from ml import infer
from pipeline import controller


_ROOT = Path(__file__).resolve().parents[1]
_CACHE_HIT_MESSAGE = "Completed cached analysis"


def _artifact_identity() -> dict:
    digests = {}
    for path in (infer._CHECKPOINT, infer._CHECKPOINT.with_suffix(".sha256"), infer._CALIBRATION):
        try:
            digests[path.name] = sha256(path.read_bytes()).hexdigest()
        except FileNotFoundError:
            digests[path.name] = None
    return digests


def _implementation_identity() -> dict:
    sources = {_ROOT / "app/main.py", *(_ROOT / "ml" / name for name in (
        "infer.py", "features.py", "model.py", "bundle.py", "train.py",
    ))}
    for folder in ("ingestion", "preprocessing", "classical_analysis", "reconciliation",
                   "physics", "verdict", "pipeline", "core"):
        sources.update(path for path in (_ROOT / folder).rglob("*.py")
                       if path.name != "reference.py")
    hashes = {path.relative_to(_ROOT).as_posix(): sha256(path.read_bytes()).hexdigest()
              for path in sources}
    packages = {name: version(name) for name in ("numpy", "scipy", "astropy", "pandas",
                                                "torch", "streamlit", "PyYAML")}
    return {"python": sys.version, "packages": packages, "sources": hashes}


def _cache_key(raw: bytes, name: str, obs_id: str, digest: str) -> str:
    inputs = {"upload_sha256": sha256(raw).hexdigest(), "basename": Path(name).name,
              "obs_id": obs_id, "format": "sdlcd", "mode": "fast", "schema_version": SCHEMA_VERSION,
              "config_hash": digest, "artifacts": _artifact_identity(),
              "implementation": _implementation_identity()}
    return sha256(json.dumps(inputs, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()


def _cacheable(result: AnalysisResult) -> bool:
    if result.verdict is None or result.stage_errors:
        return False
    if result.verification is not None and not result.verification.complete:
        return False
    stages = (result.prepared, result.quality, *result.estimators.values(),
              result.reconciliation, result.verification, result.verdict)
    return not any(getattr(stage, "is_stub", False) for stage in stages)


def _analyze_upload(raw: bytes, name: str, obs: Observation, cfg: PipelineConfig, entry: dict,
                    progress: Optional[ProgressCallback] = None) -> AnalysisResult:
    try:
        digest = config_hash(cfg)
        key = _cache_key(raw, name, obs.obs_id, digest)
    except Exception:
        return controller.analyze(obs, cfg, progress=progress)

    saved = entry.get("analysis")
    if isinstance(saved, tuple) and len(saved) == 2 and saved[0] == key:
        try:
            # Decode against the active schema and recheck exact verification coverage.
            cached = AnalysisResult.from_dict(json.loads(saved[1]))
            if ((cached.obs_id, cached.mode, cached.config_hash) != (obs.obs_id, "fast", digest)
                    or not _cacheable(cached)):
                raise ValueError("Cached analysis does not match the request")
            controller._write_run_config(cfg, digest)
        except Exception:
            entry.clear()
        else:
            report_progress(progress, 0., "Loading cached analysis")
            report_progress(progress, 1., _CACHE_HIT_MESSAGE)
            return cached

    result = controller.analyze(obs, cfg, progress=progress)
    if _cacheable(result):
        try:
            result.validate()
            encoded = json.dumps(result.to_dict(), allow_nan=False)
            unchanged = key == _cache_key(raw, name, obs.obs_id, digest)
        except Exception:
            return result
        if unchanged:
            entry["analysis"] = (key, encoded)
    return result
