"""Shared models and their validation and JSON representation (CONTRACTS.md)."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from functools import lru_cache
from numbers import Real
from types import UnionType
from typing import Any, Callable, Literal, Optional, Union, get_args, get_origin, get_type_hints

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(f"contract: {message}")


@lru_cache
def _hints(model):
    return get_type_hints(model)


def _validate_model(self):
    for name, annotation in _hints(type(self)).items():
        _check_type(getattr(self, name), annotation, name)
    _VALIDATORS[type(self)](self)
    _check_values(self)


def _model_to_dict(self):
    _check_values(self)
    return _encode(self)


def _model_from_dict(cls, payload):
    value = _decode(payload, cls)
    value.validate()
    return value


SCHEMA_VERSION = 5

REASON_CODES = (
    "low_quality",
    "no_periodic_signal",
    "conflicting_methods",
    "harmonic_ambiguity",
    "alias_ambiguity",
    "candidate_inadequate",
    "model_mismatch",
    "geometry_uncertainty",
    "verification_incomplete",
)

STAGE2_FAIL_REASONS = (
    "insufficient_cycles",
    "unresolved_frequency",
    "poor_phase_coverage",
    "merged_into_neighbour",
)

SKIP_CODES = (
    "stage1_insufficient",
    "period_bounds_not_set",
    "no_checkpoint",
    "too_few_valid_samples",
    "not_run",
)

INCONCLUSIVE_CAUSES = (
    "all_estimators_unavailable",
    "all_candidates_failed_stage2",
)

ESTIMATOR_METHODS = ("cnn", "lomb_scargle", "pdm")

(LOW_QUALITY, NO_PERIODIC_SIGNAL, CONFLICTING_METHODS, HARMONIC_AMBIGUITY, ALIAS_AMBIGUITY,
 CANDIDATE_INADEQUATE, MODEL_MISMATCH, GEOMETRY_UNCERTAINTY, VERIFICATION_INCOMPLETE) = REASON_CODES
(STAGE1_INSUFFICIENT, PERIOD_BOUNDS_NOT_SET, NO_CHECKPOINT, TOO_FEW_VALID_SAMPLES,
 NOT_RUN) = SKIP_CODES
ALL_ESTIMATORS_UNAVAILABLE, ALL_CANDIDATES_FAILED_STAGE2 = INCONCLUSIVE_CAUSES
CNN, LOMB_SCARGLE, PDM = ESTIMATOR_METHODS

BrightnessKind = Literal["mag", "flux", "unknown"]
GateStatus = Literal["suitable", "limited", "insufficient"]
StageStatus = Literal["ok", "failed", "skipped"]
VerdictLabel = Literal["Supported", "Ambiguous", "Inconclusive"]
EstimatorMethod = Literal["cnn", "lomb_scargle", "pdm"]
ProgressCallback = Callable[[float, str], None]


@dataclass
class Observation:
    """Raw, as ingested. No sorting, no cleaning. Produced by ingestion."""
    obs_id: str
    t_s: np.ndarray
    y: np.ndarray
    y_err: Optional[np.ndarray]
    brightness_kind: BrightnessKind
    meta: dict[str, Any] = field(default_factory=dict)

    validate = _validate_model
    to_dict = _model_to_dict
    from_dict = classmethod(_model_from_dict)


@dataclass
class ReferenceLabel:
    """EVALUATION ONLY. See Section 0, rule 4."""
    obs_id: str
    object_id: str
    period_s: float
    split: Literal["val", "test"]

    validate = _validate_model
    to_dict = _model_to_dict
    from_dict = classmethod(_model_from_dict)


@dataclass
class PreparedObservation:
    obs_id: str
    t_s: np.ndarray
    y_norm: np.ndarray
    valid_mask: np.ndarray
    dt_s: np.ndarray
    flux_rel: Optional[np.ndarray]
    flags: dict[str, np.ndarray]
    norm_params: dict[str, float]
    meta: dict[str, Any] = field(default_factory=dict)
    is_stub: bool = False

    validate = _validate_model
    to_dict = _model_to_dict
    from_dict = classmethod(_model_from_dict)


@dataclass
class QualityReport:
    status: GateStatus
    metrics: dict[str, float]
    warnings: list[str]
    reason_codes: list[str]
    is_stub: bool = False

    validate = _validate_model
    to_dict = _model_to_dict
    from_dict = classmethod(_model_from_dict)


@dataclass
class Candidate:
    period_s: float
    support: float
    raw_score: float
    meta: dict[str, Any] = field(default_factory=dict)

    validate = _validate_model
    to_dict = _model_to_dict
    from_dict = classmethod(_model_from_dict)


@dataclass
class EstimatorResult:
    method: EstimatorMethod
    status: StageStatus
    candidates: list[Candidate]
    error: Optional[str] = None
    meta: dict[str, Any] = field(default_factory=dict)
    is_stub: bool = False

    validate = _validate_model
    to_dict = _model_to_dict
    from_dict = classmethod(_model_from_dict)


@dataclass
class Cluster:
    cluster_id: int
    period_s: float
    members: dict[str, float]
    evidence: dict[str, float]
    alias_risk: float
    harmonic_links: list[dict[str, Any]]
    score: float
    stage2: dict[str, Any]

    validate = _validate_model
    to_dict = _model_to_dict
    from_dict = classmethod(_model_from_dict)


@dataclass
class ReconciliationResult:
    clusters: list[Cluster]
    shortlist: list[int]
    stage2_passed_any: bool
    notes: list[str]
    is_stub: bool = False

    validate = _validate_model
    to_dict = _model_to_dict
    from_dict = classmethod(_model_from_dict)


@dataclass
class CandidateVerification:
    cluster_id: int
    period_s: float
    status: StageStatus
    fit_error: Optional[float]
    heldout_error: Optional[float]
    decoy_periods_s: list[float]
    decoy_delta_e: list[float]
    passed: bool
    best_params: Optional[dict[str, Any]]
    notes: list[str]

    validate = _validate_model
    to_dict = _model_to_dict
    from_dict = classmethod(_model_from_dict)


@dataclass
class HarmonicComparison:
    pair_id: str
    short_cluster_id: int
    long_cluster_id: int
    short_period_s: float
    long_period_s: float
    ratio: float
    status: StageStatus
    prefers: Literal["P", "2P", "undecided"]
    short_heldout_error: Optional[float]
    long_heldout_error: Optional[float]
    delta_e: Optional[float]
    margin: Optional[float]
    reason: Optional[str]
    notes: list[str]

    validate = _validate_model
    to_dict = _model_to_dict
    from_dict = classmethod(_model_from_dict)


@dataclass
class VerificationResult:
    per_candidate: list[CandidateVerification]
    complete: bool
    mode: Literal["fast", "thorough"]
    runtime_s: float
    is_stub: bool = False
    harmonic_comparisons: list[HarmonicComparison] = field(default_factory=list)

    validate = _validate_model
    to_dict = _model_to_dict
    from_dict = classmethod(_model_from_dict)


@dataclass
class Verdict:
    label: VerdictLabel
    best_period_s: Optional[float]
    best_cluster_id: Optional[int]
    alternatives: list[dict[str, Any]]
    reason_codes: list[str]
    evidence_summary: dict[str, Any]
    text: str
    is_stub: bool = False

    validate = _validate_model
    to_dict = _model_to_dict
    from_dict = classmethod(_model_from_dict)


@dataclass
class AnalysisResult:
    obs_id: str
    prepared: Optional[PreparedObservation]
    quality: Optional[QualityReport]
    estimators: dict[str, EstimatorResult]
    reconciliation: Optional[ReconciliationResult]
    verification: Optional[VerificationResult]
    verdict: Optional[Verdict]
    stage_errors: dict[str, str]
    timings_s: dict[str, float]
    mode: str
    config_hash: str
    schema_version: int = SCHEMA_VERSION

    validate = _validate_model
    to_dict = _model_to_dict
    from_dict = classmethod(_model_from_dict)


def _check_type(value, annotation, name):
    origin = get_origin(annotation)
    args = get_args(annotation)
    if annotation is Any:
        return
    if origin in (Union, UnionType):
        if len(args) == 2 and type(None) in args:
            if value is not None:
                _check_type(value, next(option for option in args if option is not type(None)), name)
            return
        for option in args:
            try:
                _check_type(value, option, name)
                return
            except ValueError:
                continue
        _require(False, f"{name} does not match any allowed type")
    elif origin is Literal:
        _require(type(value) is str and value in args, f"invalid {name}")
    elif origin is list:
        _require(isinstance(value, list), f"{name} must be a list")
        for item in value:
            _check_type(item, args[0], name)
    elif origin is dict:
        _require(isinstance(value, dict), f"{name} must be a dictionary")
        for key, item in value.items():
            _check_type(key, args[0], name)
            _check_type(item, args[1], f"{name}.{key}")
    elif annotation is float:
        _require(isinstance(value, Real) and not isinstance(value, (bool, np.bool_)),
                 f"{name} must be numeric")
        _require(np.isfinite(value), "non-finite value")
    elif annotation is int:
        _require(isinstance(value, (int, np.integer)) and not isinstance(value, bool),
                 f"{name} must be an integer")
    elif annotation is bool:
        _require(isinstance(value, (bool, np.bool_)), f"{name} must be boolean")
    else:
        _require(isinstance(value, annotation), f"{name} must be {annotation.__name__}")
        if is_dataclass(value):
            value.validate()


def _check_values(value, allow_array_nan=False):
    if isinstance(value, np.ndarray):
        _require(value.dtype.kind in "biuf", "unsupported array dtype")
        _require(not np.isinf(value).any(), "non-finite value")
        _require(allow_array_nan or not np.isnan(value).any(), "unexpected NaN")
    elif is_dataclass(value):
        allowed = {
            Observation: {"y", "y_err"},
            PreparedObservation: {"y_norm", "flux_rel"},
        }.get(type(value), set())
        for item in fields(value):
            _check_values(getattr(value, item.name), item.name in allowed)
    elif isinstance(value, dict):
        for key, item in value.items():
            _require(isinstance(key, str), "dictionary keys must be strings")
            _check_values(item)
    elif isinstance(value, list):
        for item in value:
            _check_values(item)
    elif isinstance(value, (float, np.floating)):
        _require(np.isfinite(value), "non-finite value")


def _encode(value):
    if isinstance(value, np.ndarray):
        _require(value.dtype.kind in "biuf", "unsupported array dtype")
        _require(not np.isinf(value).any(), "non-finite value")
        data = value.ravel(order="C").tolist()
        if value.dtype.kind == "f":
            data = [None if np.isnan(item) else item for item in data]
        return {"__ndarray__": True, "dtype": value.dtype.name,
                "shape": list(value.shape), "data": data}
    if isinstance(value, np.generic):
        if value.dtype.kind == "f":
            return _encode(float(value))
        return _encode(value.item())
    if is_dataclass(value):
        return {item.name: _encode(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, dict):
        _require(all(isinstance(key, str) for key in value), "dictionary keys must be strings")
        return {key: _encode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if isinstance(value, float):
        _require(np.isfinite(value), "non-finite value")
    _require(value is None or isinstance(value, (str, int, float, bool)), "unsupported JSON value")
    return value


def _decode_array(payload):
    _require(isinstance(payload, dict) and payload.get("__ndarray__") is True,
             "invalid ndarray encoding")
    _require(set(payload) == {"__ndarray__", "dtype", "shape", "data"}, "invalid ndarray keys")
    shape, data = payload["shape"], payload["data"]
    _require(isinstance(payload["dtype"], str), "invalid ndarray dtype")
    _require(isinstance(shape, list) and all(type(n) is int and n >= 0 for n in shape),
             "invalid ndarray shape")
    _require(isinstance(data, list), "invalid ndarray data")
    try:
        dtype = np.dtype(payload["dtype"])
        _require(dtype.kind in "biuf", "unsupported array dtype")
        expected = {"b": (bool,), "i": (int,), "u": (int,), "f": (int, float, type(None))}
        _require(all(type(item) in expected[dtype.kind] for item in data), "invalid ndarray data")
        if dtype.kind == "f":
            data = [np.nan if item is None else item for item in data]
        array = np.array(data, dtype=dtype).reshape(shape)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("contract: invalid ndarray encoding") from exc
    if dtype.kind == "f":
        raw = payload["data"]
        _require(not np.isinf(array).any(), "non-finite value")
        _require(not any(type(item) is float and np.isnan(item) for item in raw),
                 "array NaN must be encoded as null")
    return array


def _decode(payload, annotation=Any):
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (Union, UnionType) and type(None) in args:
        if payload is None:
            return None
        annotation = next(option for option in args if option is not type(None))
        return _decode(payload, annotation)
    if annotation is np.ndarray:
        return _decode_array(payload)
    if isinstance(annotation, type) and is_dataclass(annotation):
        _require(isinstance(payload, dict), f"{annotation.__name__} must be a dictionary")
        if annotation is AnalysisResult:
            version = payload.get("schema_version")
            _require(type(version) is int and version == SCHEMA_VERSION, "schema version mismatch")
        if annotation is VerificationResult:
            _require("harmonic_comparisons" in payload, "missing harmonic_comparisons")
        hints = _hints(annotation)
        _require(set(payload) <= hints.keys(), f"unknown {annotation.__name__} fields")
        try:
            return annotation(**{key: _decode(value, hints[key]) for key, value in payload.items()})
        except TypeError as exc:
            raise ValueError(f"contract: invalid {annotation.__name__} fields") from exc
    if isinstance(payload, dict):
        if annotation is Any and payload.get("__ndarray__") is True:
            return _decode_array(payload)
        item_type = args[1] if origin is dict else Any
        return {key: _decode(item, item_type) for key, item in payload.items()}
    if isinstance(payload, list):
        item_type = args[0] if origin is list else Any
        return [_decode(item, item_type) for item in payload]
    _require(payload is None or isinstance(payload, (str, int, float, bool)), "unsupported JSON value")
    return payload


def _dict_fields(value, name, **required):
    _require(isinstance(value, dict), f"{name} must be a dictionary")
    for key, annotation in required.items():
        _require(key in value, f"missing {name}.{key}")
        _check_type(value[key], annotation, f"{name}.{key}")


def _vector(array, dtype, n, name):
    _require(array.dtype == np.dtype(dtype) and array.shape == (n,), f"invalid {name} array")


def _observation(value):
    n = value.t_s.size
    _require(n >= 1, "empty observation")
    _vector(value.t_s, "float64", n, "t_s")
    _vector(value.y, "float64", n, "y")
    _require(np.isfinite(value.t_s).all(), "non-finite t_s")
    if value.y_err is not None:
        _vector(value.y_err, "float64", n, "y_err")
        _require(((value.y_err >= 0) | np.isnan(value.y_err)).all(), "negative y_err")


def _reference(value):
    _require(value.period_s > 0, "non-positive period_s")


def _prepared(value):
    n = value.t_s.size
    _require(n >= 1, "empty prepared observation")
    for name in ("t_s", "y_norm", "dt_s"):
        _vector(getattr(value, name), "float64", n, name)
    _vector(value.valid_mask, "bool", n, "valid_mask")
    _require(np.isfinite(value.t_s).all() and value.t_s[0] == 0 and
             (np.diff(value.t_s) >= 0).all(), "invalid prepared t_s")
    _require(np.isfinite(value.dt_s).all() and value.dt_s[0] == 0 and
             (value.dt_s >= 0).all(), "invalid dt_s")
    _require(np.array_equal(np.isnan(value.y_norm), ~value.valid_mask), "y_norm mask mismatch")
    if value.flux_rel is not None:
        _vector(value.flux_rel, "float64", n, "flux_rel")
        _require(np.array_equal(np.isnan(value.flux_rel), ~value.valid_mask), "flux_rel mask mismatch")
        _require((value.flux_rel[value.valid_mask] > 0).all(), "non-positive valid flux_rel")
    _require({"bright_outlier", "duplicate", "gap_edge"} <= value.flags.keys(), "missing flags")
    for name, array in value.flags.items():
        _vector(array, "bool", n, name)
    _dict_fields(value.norm_params, "norm_params", median=float, mad=float, sign=float)
    _dict_fields(value.meta, "meta", baseline_s=float, freq_resolution=Optional[float],
                 n_rows=int, n_valid=int)
    _require(value.meta["baseline_s"] >= 0, "negative baseline_s")
    resolution = value.meta["freq_resolution"]
    _require(resolution is None or resolution > 0, "non-positive freq_resolution")
    _require(value.meta["n_rows"] == n and value.meta["n_valid"] == value.valid_mask.sum(),
             "prepared observation counts mismatch")


def _quality(value):
    expected = [LOW_QUALITY] if value.status == "insufficient" else []
    _require(value.reason_codes == expected, "invalid Stage 1 reason codes")


def _candidate(value):
    _require(value.period_s > 0, "non-positive period_s")
    _require(0 <= value.support <= 1, "support outside [0, 1]")


def _estimator(value):
    _dict_fields(value.meta, "meta", runtime_s=float)
    if value.status != "ok":
        _require(not value.candidates, "unavailable estimator has candidates")
        if value.status == "failed":
            _require(bool(value.error and value.error.strip()), "failed estimator requires an error")
        else:
            _require(value.error in SKIP_CODES, "invalid estimator skip code")
        return
    _require(value.error is None, "successful estimator has an error")
    supports = [candidate.support for candidate in value.candidates]
    _require(supports == sorted(supports, reverse=True), "candidates out of order")
    if value.method == CNN:
        _dict_fields(value.meta, "meta", p_no_reliable=float, abstained=bool, n_windows=int,
                     window_disagreement=float, experimental=bool)
        for name in ("p_no_reliable", "window_disagreement"):
            _require(0 <= value.meta[name] <= 1, f"{name} outside [0, 1]")
    else:
        _dict_fields(value.meta, "meta", periodogram=dict, freq_resolution=Optional[float])
        graph = value.meta["periodogram"]
        _dict_fields(graph, "periodogram", freq_hz=list[float], power=list[float])
        _require(len(graph["freq_hz"]) == len(graph["power"]), "periodogram length mismatch")
        for candidate in value.candidates:
            _dict_fields(candidate.meta, "candidate.meta", freq_width_hz=float)
            _require(candidate.meta["freq_width_hz"] > 0, "non-positive freq_width_hz")


def _cluster(value):
    _require(value.period_s > 0, "non-positive cluster period_s")
    _require(value.evidence.keys() <= set(ESTIMATOR_METHODS), "invalid estimator evidence key")
    _require(all(0 <= score <= 1 for score in value.evidence.values()), "evidence outside [0, 1]")
    _require(0 <= value.alias_risk <= 1, "alias_risk outside [0, 1]")
    _dict_fields(value.stage2, "stage2", passed=bool, cycles=float, fail_reasons=list[str])
    failures = value.stage2["fail_reasons"]
    _require(value.stage2["passed"] == (not failures), "inconsistent Stage 2 pass flag")
    _require(set(failures) <= set(STAGE2_FAIL_REASONS), "invalid Stage 2 failure reason")
    for link in value.harmonic_links:
        _dict_fields(link, "harmonic link", other_id=int, ratio=float)
        _require(link["other_id"] != value.cluster_id and link["ratio"] > 0, "invalid harmonic link")


def _reconciliation(value):
    by_id = {cluster.cluster_id: cluster for cluster in value.clusters}
    _require(len(by_id) == len(value.clusters), "duplicate cluster ID")
    scores = [cluster.score for cluster in value.clusters]
    _require(scores == sorted(scores, reverse=True), "clusters out of order")
    for cluster in value.clusters:
        _require(all(link["other_id"] in by_id for link in cluster.harmonic_links),
                 "unknown harmonic cluster ID")
    _require(len(set(value.shortlist)) == len(value.shortlist), "duplicate shortlist ID")
    _require(all(cid in by_id and by_id[cid].stage2["passed"] for cid in value.shortlist),
             "shortlist contains an unknown or rejected cluster")
    if not value.is_stub:
        _require(bool(value.shortlist) == any(c.stage2["passed"] for c in value.clusters),
                 "real Stage 2 pass requires a shortlist")
    selected = [by_id[cid].score for cid in value.shortlist]
    accepted = [cluster.score for cluster in value.clusters if cluster.stage2["passed"]]
    _require(selected == accepted[:len(selected)], "shortlist scores out of order")
    _require(value.stage2_passed_any == bool(value.shortlist), "inconsistent shortlist flag")


def _candidate_verification(value):
    _require(value.period_s > 0, "non-positive verification period_s")
    _require(len(value.decoy_periods_s) == len(value.decoy_delta_e), "decoy length mismatch")
    if value.status == "ok":
        _require(value.fit_error is not None and value.heldout_error is not None,
                 "successful verification missing error metrics")
        _dict_fields(value.best_params, "best_params", shape=str, geometry_errors=dict[str, float],
                     fit_space=Literal["flux_rel", "y_norm"], scale_a=float, offset_b=float)
    else:
        _require(not value.passed and value.fit_error is None and value.heldout_error is None
                 and value.best_params is None, "invalid unfinished verification")


def harmonic_preference(delta_e: float, margin: float) -> Literal["P", "2P", "undecided"]:
    if not delta_e or abs(delta_e) < margin:
        return "undecided"
    return "P" if delta_e > 0 else "2P"


def _harmonic_comparison(value):
    short, long = value.short_cluster_id, value.long_cluster_id
    _require(short != long and 0 < value.short_period_s < value.long_period_s and value.ratio == 2.,
             "invalid harmonic relationship or orientation")
    _require(value.pair_id == f"{short}:{long}", "invalid harmonic pair ID")
    errors = value.short_heldout_error, value.long_heldout_error
    _require(all(e is None or e >= 0 for e in errors), "negative harmonic held-out error")
    if value.status == "ok":
        _require(all(e is not None for e in errors) and value.margin is not None and value.margin >= 0,
                 "completed harmonic comparison missing inputs")
        _require(value.delta_e == errors[1] - errors[0] and value.reason is None
                 and value.prefers == harmonic_preference(value.delta_e, value.margin),
                 "invalid completed harmonic outcome")
    else:
        _require(value.delta_e is None and value.margin is None and value.prefers == "undecided"
                 and value.reason is not None and bool(value.reason.strip()),
                 "invalid unfinished harmonic outcome")
        allowed = {"failed": {"harmonic_fit_unfinished", "harmonic_comparison_failed"},
                   "skipped": {"harmonic_fit_unfinished", "harmonic_comparison_not_run", "partner_failed_stage2"}}
        _require(value.reason in allowed[value.status], "invalid harmonic comparison reason for status")


def _verification(value):
    ids = {record.cluster_id for record in value.per_candidate}
    _require(len(ids) == len(value.per_candidate), "duplicate verification cluster ID")
    pairs = [(row.short_cluster_id, row.long_cluster_id) for row in value.harmonic_comparisons]
    _require(len({row.pair_id for row in value.harmonic_comparisons}) == len(pairs)
             and len({frozenset(pair) for pair in pairs}) == len(pairs), "duplicate harmonic pair")
    _require(pairs == sorted(pairs), "harmonic comparisons out of order")
    for row, pair in zip(value.harmonic_comparisons, pairs):
        count = 1 if row.reason == "partner_failed_stage2" else 2
        _require(len(ids.intersection(pair)) == count, "harmonic comparison references unknown fits")
    if value.complete:
        _require(verification_finished(value.per_candidate, value.harmonic_comparisons),
                 "verification marked complete with unfinished candidates or harmonic comparisons")


def verification_finished(records: list[CandidateVerification], comparisons: list[HarmonicComparison]) -> bool:
    return (bool(records) and all(record.status == "ok" for record in records)
            and all(row.status == "ok" or row.reason == "partner_failed_stage2" for row in comparisons))


def _verdict(value):
    supported = value.label == "Supported"
    _require(bool(value.reason_codes) != supported and
             set(value.reason_codes) <= set(REASON_CODES), "invalid verdict reason codes")
    if supported:
        _require(value.best_period_s is not None and value.best_period_s > 0
                 and value.best_cluster_id is not None, "supported verdict missing best candidate")
    else:
        _require(value.best_period_s is None and value.best_cluster_id is None,
                 "best candidate reserved for Supported")
    for option in value.alternatives:
        _dict_fields(option, "alternative", cluster_id=int, period_s=float, why=str)
    summary = value.evidence_summary
    _dict_fields(summary, "evidence_summary", stage1_status=str, stage1_warnings=list[str],
                 estimator_status=dict[str, StageStatus], n_estimators_available=int,
                 estimators_agree=Optional[bool], alias_risk=Optional[float],
                 verification_complete=Optional[bool], heldout_error=Optional[float],
                 decoy_separation=Optional[float], stage2_fail_reasons=list[str],
                 selective_accuracy_validation=Optional[dict])
    _require(set(summary["stage2_fail_reasons"]) <= set(STAGE2_FAIL_REASONS),
             "invalid evidence Stage 2 failure reason")
    if summary["selective_accuracy_validation"] is not None:
        _dict_fields(summary["selective_accuracy_validation"], "selective_accuracy_validation",
                     value=float, n=int)
    if "inconclusive_cause" in summary:
        _require(summary["inconclusive_cause"] in INCONCLUSIVE_CAUSES, "invalid inconclusive cause")
    if value.label == "Inconclusive" and CANDIDATE_INADEQUATE in value.reason_codes:
        _require("inconclusive_cause" in summary, "missing inconclusive cause")


def required_verification_ids(recon: ReconciliationResult) -> set[int]:
    """Return the exact verification scope from CONTRACTS.md §5.5."""
    by_id = {cluster.cluster_id: cluster for cluster in recon.clusters}
    required = set(recon.shortlist)
    for cid in recon.shortlist:
        for link in by_id[cid].harmonic_links:
            partner = link["other_id"]
            if link["ratio"] in (2, 0.5) and by_id[partner].stage2["passed"]:
                required.add(partner)
    return required


def expected_harmonic_pairs(recon: ReconciliationResult) -> dict[tuple[int, int], bool]:
    """Map ordered short/long IDs to whether a fit comparison is required."""
    by_id = {cluster.cluster_id: cluster for cluster in recon.clusters}
    shortlist = set(recon.shortlist)
    required = required_verification_ids(recon)
    pairs = {}
    for cid in required:
        cluster = by_id[cid]
        for link in cluster.harmonic_links:
            other = link["other_id"]
            ratio = link["ratio"]
            if ratio not in (0.5, 2.):
                continue
            partner = by_id[other]
            eligible = partner.stage2["passed"]
            if eligible and (other not in required or not (cid in shortlist or other in shortlist)):
                continue
            short, long = (cluster, partner) if ratio == 2. else (partner, cluster)
            _require(short.period_s < long.period_s, "invalid reconciliation harmonic orientation")
            pairs[short.cluster_id, long.cluster_id] = eligible
    return dict(sorted(pairs.items()))


def harmonic_resolved_for(result: VerificationResult, cluster: Cluster) -> bool:
    cid = cluster.cluster_id
    favorable = set()
    for row in result.harmonic_comparisons:
        if cid not in (row.short_cluster_id, row.long_cluster_id):
            continue
        is_short = cid == row.short_cluster_id
        if row.status != "ok" or row.prefers != ("P" if is_short else "2P"):
            return False
        favorable.add(row.long_cluster_id if is_short else row.short_cluster_id)
    return all(link["other_id"] in favorable for link in cluster.harmonic_links)


def check_verification_scope(recon: ReconciliationResult, result: VerificationResult) -> None:
    required = required_verification_ids(recon)
    records = {record.cluster_id: record for record in result.per_candidate}
    _require(bool(required) and records.keys() == required and len(result.per_candidate) == len(required),
             "verification membership differs from required cluster IDs")
    expected = expected_harmonic_pairs(recon)
    rows = result.harmonic_comparisons
    _require([(row.short_cluster_id, row.long_cluster_id) for row in rows] == list(expected),
             "harmonic comparison membership differs from expected pairs")
    periods = {cluster.cluster_id: cluster.period_s for cluster in recon.clusters}
    _require(all(record.period_s == periods[cid] for cid, record in records.items()),
             "verification period differs from reconciliation")
    for row in rows:
        short, long = row.short_cluster_id, row.long_cluster_id
        _require((row.short_period_s, row.long_period_s) == (periods[short], periods[long]),
                 "harmonic comparison periods differ from reconciliation")
        eligible = expected[short, long]
        _require((row.reason == "partner_failed_stage2") == (not eligible),
                 "incorrect harmonic Stage 2 exclusion")
        endpoints = [records.get(cid) for cid in (short, long)]
        errors = tuple(record.heldout_error if record is not None else None for record in endpoints)
        _require((row.short_heldout_error, row.long_heldout_error) == errors,
                 "harmonic comparison errors differ from verified fits")
        if not eligible:
            _require(row.status == "skipped", "excluded harmonic comparison must be skipped")
        elif any(record.status != "ok" for record in endpoints):
            status = "failed" if any(record.status == "failed" for record in endpoints) else "skipped"
            _require(row.status == status and row.reason == "harmonic_fit_unfinished",
                     "harmonic comparison status differs from unfinished fits")
        else:
            _require(row.reason != "harmonic_fit_unfinished", "harmonic comparison has finished fits")
    _require(result.complete == verification_finished(result.per_candidate, rows),
             "incorrect verification completeness")


def report_progress(progress: Optional[ProgressCallback], fraction: float, message: str) -> None:
    if progress is not None:
        try:
            progress(fraction, message)
        except Exception:
            pass


def stage2_pending(recon, estimators, verification):
    """Recognize only the pending stub exception defined in CONTRACTS.md §5.4."""
    return (recon is not None and recon.is_stub
            and "stage2_not_evaluated" in recon.notes
            and bool(recon.clusters) and not recon.shortlist and not recon.stage2_passed_any
            and verification is None
            and any(item.status == "ok" and item.candidates for item in estimators.values()))


def _structural_outcome(quality, estimators, recon, verification):
    """Existing decisions that remain available without calibrated verdict logic."""
    if quality is None:
        return None
    if quality.status == "insufficient":
        return "quality"
    if not any(item.status == "ok" for item in estimators.values()):
        return "unavailable"
    if not any(item.candidates for item in estimators.values()):
        return "empty"
    if recon is not None and not recon.shortlist and recon.clusters and all(
            not c.stage2["passed"] for c in recon.clusters):
        return "rejected"
    if recon is not None and recon.shortlist:
        if verification is None or not verification.complete:
            return "incomplete"
        if not any(item.passed for item in verification.per_candidate):
            return "mismatch"
    return None


def _execution_error(prepared, quality, estimators, recon, verification, errors):
    """Mutually exclusive execution states defined in CONTRACTS.md §5.4."""
    diagnostic = lambda name: bool(errors.get(name, "").strip())
    stopped = (recon is None and verification is None and all(
        item.status == "skipped" and item.error == NOT_RUN for item in estimators.values()))
    if prepared is None:
        return "prepare" if quality is None and stopped and diagnostic("prepare") else None
    if quality is None:
        return "stage1" if stopped and diagnostic("stage1") else None
    if quality.status == "insufficient":
        return None
    candidates = any(item.status == "ok" and item.candidates for item in estimators.values())
    if candidates and recon is None:
        return "reconciliation" if verification is None and diagnostic("reconciliation") else None
    if stage2_pending(recon, estimators, verification):
        return None
    if (candidates and recon is not None and recon.shortlist and verification is not None
            and verification.complete and diagnostic("verdict")
            and _structural_outcome(quality, estimators, recon, verification) is None):
        return "verdict"
    return None


def _analysis(value):
    _require(value.schema_version == SCHEMA_VERSION, "schema version mismatch")
    _require(value.mode == "fast", "unsupported analysis mode")
    error_state = _execution_error(value.prepared, value.quality, value.estimators,
                                   value.reconciliation, value.verification, value.stage_errors)
    _require(value.prepared is not None or value.quality is None, "quality requires preparation")
    _require(value.quality is not None or error_state in ("prepare", "stage1"),
             "missing quality requires execution failure")
    _require(set(value.estimators) == set(ESTIMATOR_METHODS), "invalid analysis estimator keys")
    _require(all(key == item.method for key, item in value.estimators.items()),
             "analysis estimator method mismatch")
    if value.verification is not None:
        _require(value.reconciliation is not None, "verification requires reconciliation")
        check_verification_scope(value.reconciliation, value.verification)
    if value.reconciliation is not None and not value.reconciliation.is_stub:
        accepted = sum(c.stage2["passed"] for c in value.reconciliation.clusters)
        _require(len(value.reconciliation.shortlist) == min(3, accepted),
                 "incomplete Fast mode shortlist")
    pending = stage2_pending(value.reconciliation, value.estimators, value.verification)
    _require((value.verdict is None) == (pending or error_state is not None),
             "verdict absence requires pending Stage 2 stub or execution failure")
    if value.verdict is not None and value.verdict.label == "Supported":
        _require(value.verification is not None and value.verification.complete,
                 "Supported requires complete verification")
        _require(any(record.cluster_id == value.verdict.best_cluster_id and
                     record.status == "ok" and record.passed
                     for record in value.verification.per_candidate),
                 "supported candidate has not passed verification")
        best_cluster = next(cluster for cluster in value.reconciliation.clusters
                            if cluster.cluster_id == value.verdict.best_cluster_id)
        _require(harmonic_resolved_for(value.verification, best_cluster),
                 "Supported requires favorable harmonic comparisons")
    outcome = _structural_outcome(value.quality, value.estimators, value.reconciliation, value.verification)
    if outcome == "incomplete":
        _require(value.verdict is not None and value.verdict.label == "Ambiguous"
                 and VERIFICATION_INCOMPLETE in value.verdict.reason_codes,
                 "required verification is incomplete")


_VALIDATORS = {
    Observation: _observation,
    ReferenceLabel: _reference,
    PreparedObservation: _prepared,
    QualityReport: _quality,
    Candidate: _candidate,
    EstimatorResult: _estimator,
    Cluster: _cluster,
    ReconciliationResult: _reconciliation,
    CandidateVerification: _candidate_verification,
    HarmonicComparison: _harmonic_comparison,
    VerificationResult: _verification,
    Verdict: _verdict,
    AnalysisResult: _analysis,
}


def assert_contract_equal(a, b, rtol=1e-12, atol=0.0):
    if isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray) and a.shape == b.shape and a.dtype == b.dtype
        assert np.array_equal(np.isnan(a), np.isnan(b))
        assert np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
    elif is_dataclass(a):
        assert type(a) is type(b)
        for item in fields(a):
            assert_contract_equal(getattr(a, item.name), getattr(b, item.name), rtol, atol)
    elif isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys()
        for key in a:
            assert_contract_equal(a[key], b[key], rtol, atol)
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b)
        for left, right in zip(a, b):
            assert_contract_equal(left, right, rtol, atol)
    elif isinstance(a, (float, np.floating)):
        assert isinstance(b, (float, np.floating)) and np.isclose(a, b, rtol=rtol, atol=atol)
    elif isinstance(a, (bool, np.bool_)):
        assert isinstance(b, (bool, np.bool_)) and bool(a) == bool(b)
    elif isinstance(a, (int, np.integer)):
        assert isinstance(b, (int, np.integer)) and not isinstance(b, (bool, np.bool_))
        assert a == b
    else:
        assert type(a) is type(b) and a == b
