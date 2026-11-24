"""Live upload reuse and validation of curated precomputed analyses."""

from datetime import datetime
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import re
import sys
from typing import Optional

from core.config import PipelineConfig, config_hash
from core.schemas import SCHEMA_VERSION, AnalysisResult, Observation, ProgressCallback, report_progress
from ml import infer
from pipeline import controller


_ROOT = Path(__file__).resolve().parents[1]
_CACHE_HIT_MESSAGE = "Completed cached analysis"
PORTABLE_DEMOS = _ROOT / "outputs/demo_cache"
LOCAL_DEMOS = _ROOT / "outputs/day4_local/demo_cache"


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
              "obs_id": obs_id, "ingestion": "auto", "mode": "fast", "schema_version": SCHEMA_VERSION,
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


_QUALITY_WARNINGS = {
    "Too few valid measurements or no positive time baseline.",
    "Large sampling gap limits coverage.",
    "Many brightness measurements are missing.",
    "Brightness modulation is weak relative to the noise proxy.",
    "Quality uses recorded cadence; exposure smearing is not corrected.",
}


def _public_quality_warning(warning):
    return warning if warning in _QUALITY_WARNINGS else "Stage 1 reported an additional warning."


def _safe_verification_note(note):
    if note.startswith(("P: ", "2P: ")):
        note = note.split(": ", 1)[1]
    if note in {
        "Representative coefficients from first chronological fold",
        "AIC/BIC not compared: equal free parameter counts",
        "no_valid_decoy", "zero observed MAD", "timeout",
    }:
        return True
    number = r"[+-]?[0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?"
    return bool(re.fullmatch(
        rf"(?:even|odd) shorter-period cycles: n=[0-9]+, normalized RMS=(?:{number}|unavailable)", note,
    ) or re.fullmatch(
        rf"Excluded decoy {number} s \((?:duplicate period|outside period bounds|candidate frequency|"
        r"harmonic frequency|alias offset)\)", note,
    ))


def _public_result(result):
    """Keep diagnostics present without exporting arbitrary exception messages."""
    payload = result.to_dict()
    payload["stage_errors"] = {key: "The stage could not finish." for key in payload["stage_errors"]}
    prepared = payload["prepared"]
    if prepared is not None:
        meta = prepared["meta"]
        if "t0_iso" in meta and not _timestamp(meta["t0_iso"]):
            meta.pop("t0_iso")
        meta.pop("source_file", None)
        meta.pop("assumptions", None)
    for estimate in payload["estimators"].values():
        if estimate["status"] == "failed":
            estimate["error"] = "The estimator could not finish."
    quality = payload["quality"]
    if quality is not None:
        quality["warnings"] = [_public_quality_warning(warning) for warning in quality["warnings"]]
    verification = payload["verification"]
    if verification is not None:
        for record in verification["per_candidate"]:
            record["notes"] = [note for note in record["notes"] if _safe_verification_note(note)]
            if record["status"] != "ok":
                record["notes"].append("Verification did not finish.")
        for pair in verification["harmonic_comparisons"]:
            pair["notes"] = [note for note in pair["notes"] if _safe_verification_note(note)]
    verdict = payload["verdict"]
    if verdict is not None:
        verdict["text"] = f'{verdict["label"]}. Provisional decision criteria.'
        verdict["evidence_summary"]["stage1_warnings"] = quality["warnings"]
    return AnalysisResult.from_dict(payload)


def _timestamp(value):
    try:
        datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return False
    return True


def _read_json(raw):
    def reject_constant(value):
        raise ValueError(f"Invalid JSON constant: {value}")

    def unique_keys(pairs):
        result = dict(pairs)
        if len(result) != len(pairs):
            raise ValueError("Duplicate JSON keys")
        return result

    return json.loads(raw, parse_constant=reject_constant, object_pairs_hook=unique_keys)


def _encoded(payload):
    return json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True).encode("utf-8")


def _input_snapshot(cfg, raw):
    return {
        **controller._run_snapshot(cfg, config_hash(cfg)),
        "input_sha256": sha256(raw).hexdigest(),
        "artifacts": _artifact_identity(),
        "implementation": _implementation_identity(),
    }


def _is_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _load_demo(root, entry, cfg, implementation, artifacts, source_kind):
    checksum = entry["analysis_sha256"]
    if not _is_digest(checksum) or entry["path"] != f"{checksum}.json":
        raise ValueError("Invalid demo payload path")
    basename = entry["basename"]
    if (not isinstance(basename, str) or Path(basename).name != basename or ":" in basename
            or Path(basename).suffix not in (".txt", ".csv") or entry["source_kind"] != source_kind):
        raise ValueError("Invalid demo source")
    path = root / entry["path"]
    if path.resolve().parent != root.resolve():
        raise ValueError("Demo payload is outside cache")
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != checksum:
        raise ValueError("Demo payload checksum mismatch")
    payload = _read_json(raw)
    snapshot = payload["snapshot"]
    result = AnalysisResult.from_dict(payload["analysis"])
    if (not _is_digest(entry["input_sha256"]) or entry["input_sha256"] != snapshot["input_sha256"]
            or (payload["basename"], payload["source_kind"]) != (basename, source_kind)
            or result.obs_id != Path(basename).stem or result.mode != "fast"
            or result.schema_version != SCHEMA_VERSION or snapshot["schema_version"] != SCHEMA_VERSION
            or result.config_hash != snapshot["config_hash"]
            or config_hash(PipelineConfig(**snapshot["config"])) != result.config_hash
            or config_hash(cfg) != result.config_hash):
        raise ValueError("Demo analysis identity or configuration mismatch")
    seeds = {name: section["seed"] for name, section in snapshot["config"].items() if "seed" in section}
    recorded_artifacts = snapshot["artifacts"]
    if (snapshot["seeds"] != seeds or not snapshot["package_versions"]
            or snapshot["implementation"]["sources"] != implementation["sources"]
            or set(recorded_artifacts) != set(artifacts)
            or any(value is not None and not _is_digest(value) for value in recorded_artifacts.values())
            or any(value is not None and recorded_artifacts[name] != value for name, value in artifacts.items())):
        raise ValueError("Demo provenance is stale or invalid")
    return result, snapshot


def _read_demo_index(root, cfg, source_kind):
    index_path = root / "index.json"
    if index_path.resolve().parent != root.resolve():
        raise ValueError("Demo index is outside cache")
    index = _read_json(index_path.read_bytes())
    if (index["schema_version"] != SCHEMA_VERSION or not isinstance(index["entries"], list)
            or not index["entries"]):
        raise ValueError("Invalid demo index")
    implementation, artifacts = _implementation_identity(), _artifact_identity()
    entries = {}
    for entry in index["entries"]:
        name = entry["basename"]
        if name in entries:
            raise ValueError("Duplicate demo source")
        entries[name] = _load_demo(root, entry, cfg, implementation, artifacts, source_kind)
    return entries
