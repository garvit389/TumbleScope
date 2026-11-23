"""Run Fast analysis and preserve evidence across stage failures."""

from dataclasses import asdict
from importlib.metadata import distributions
import json
from pathlib import Path
import subprocess
from tempfile import NamedTemporaryFile
from time import perf_counter
from typing import Optional

from classical_analysis.lomb_scargle import run_lomb_scargle
from classical_analysis.pdm import run_pdm
from classical_analysis.window import compute_window
from core.config import PipelineConfig, config_hash, resolve_period_bounds
from core.schemas import (
    CNN, ESTIMATOR_METHODS, LOMB_SCARGLE, LOW_QUALITY, NOT_RUN, PDM, PERIOD_BOUNDS_NOT_SET,
    SCHEMA_VERSION, STAGE1_INSUFFICIENT, VERIFICATION_INCOMPLETE, AnalysisResult, EstimatorResult, Observation,
    PreparedObservation, ProgressCallback, QualityReport, ReconciliationResult, VerificationResult,
    Verdict, check_verification_scope, report_progress,
)
from ml.infer import run_cnn
from physics.verify import verify_candidates
from preprocessing.prepare import prepare
from preprocessing.quality import assess_stage1
from reconciliation.reconcile import reconcile
from verdict.decide import decide
from verdict.reasons import _alternatives
from core.schemas import stage2_pending, _execution_error, _structural_outcome
from core.schemas import (ALL_ESTIMATORS_UNAVAILABLE, ALL_CANDIDATES_FAILED_STAGE2,
                          CANDIDATE_INADEQUATE, NO_PERIODIC_SIGNAL, MODEL_MISMATCH)


_ROOT = Path(__file__).resolve().parents[1]
_OUTPUTS = _ROOT / "outputs"
_FAST_SHORTLIST_LIMIT = 3


def _repository_revision():
    try:
        result = subprocess.run(
            ["git", "--no-optional-locks", "rev-parse", "--verify", "HEAD"],
            cwd=_ROOT, capture_output=True, text=True, check=True, timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _write_run_config(cfg, digest):
    sections = asdict(cfg)
    snapshot = {
        "config": sections,
        "config_hash": digest,
        "schema_version": SCHEMA_VERSION,
        "seeds": {section: data["seed"] for section, data in sections.items() if "seed" in data},
        "package_versions": {item.metadata["Name"]: item.version for item in distributions()},
        "commit": _repository_revision(),
    }
    _OUTPUTS.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".tmp",
                                dir=_OUTPUTS, delete=False) as target:
            tmp = Path(target.name)
            json.dump(snapshot, target, indent=2, sort_keys=True, allow_nan=False)
        tmp.replace(_OUTPUTS / "run_config.json")
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def _unavailable(method, error, status="skipped", runtime=0.0):
    return EstimatorResult(method, status, [], error, {"runtime_s": runtime})


def _validation_message(error):
    return str(error) or type(error).__name__


def analyze(obs: Observation, cfg: "PipelineConfig", mode: str = "fast",
            progress: Optional[ProgressCallback] = None) -> AnalysisResult:
    if mode != "fast":
        raise NotImplementedError("Only Fast mode is available.")

    stage_errors = {}
    timings = {}
    validate_outputs = cfg.pipeline.get("validate_outputs", True)

    def record_error(name, error):
        message = _validation_message(error)
        if message.startswith("contract:"):
            message = f"contract_violation: {message}"
        stage_errors[name] = message

    def validate_result(result, check):
        result.validate()
        if check is not None:
            check(result)

    def stage(name, operation, *, expected=None, check=None):
        start = perf_counter()
        try:
            result = operation()
            if expected is not None and not isinstance(result, expected):
                raise ValueError(f"contract: {name} must return {expected.__name__}")
            if expected is not None and validate_outputs:
                validate_result(result, check)
            return result
        except Exception as error:
            record_error(name, error)
            return None
        finally:
            timings[name] = perf_counter() - start

    def accept(name, result, check=None):
        # The optional eager check can be disabled; data consumed by the controller
        # still has to be safe to use and valid in the assembled AnalysisResult.
        if validate_outputs or result is None:
            return result
        start = perf_counter()
        try:
            validate_result(result, check)
            return result
        except Exception as error:
            record_error(name, error)
            return None
        finally:
            timings[name] += perf_counter() - start

    def reconcile_checks(result):
        if len(result.shortlist) > _FAST_SHORTLIST_LIMIT:
            raise ValueError("contract: shortlist exceeds the Fast mode limit")
        if not result.is_stub and len(result.shortlist) != min(
                _FAST_SHORTLIST_LIMIT, sum(c.stage2["passed"] for c in result.clusters)):
            raise ValueError("contract: incomplete Fast mode shortlist")

    def verdict_checks(result):
        if recon is not None and recon.shortlist and (verification is None or not verification.complete):
            if result.label != "Ambiguous" or VERIFICATION_INCOMPLETE not in result.reason_codes:
                raise ValueError("contract: required verification is incomplete")
        if result.label != "Supported":
            return
        if verification is None or not verification.complete:
            raise ValueError("contract: Supported requires complete verification")
        if not any(record.cluster_id == result.best_cluster_id and record.passed
                   for record in verification.per_candidate):
            raise ValueError("contract: supported candidate has not passed verification")

    def prepare_input():
        obs.validate()
        return prepare(obs, cfg.preprocess)

    report_progress(progress, 0.0, "Starting analysis")
    digest = config_hash(cfg)
    stage("run_config", lambda: _write_run_config(cfg, digest))
    prepared = accept("prepare", stage("prepare", prepare_input, expected=PreparedObservation))
    quality = None
    if prepared is not None:
        quality = accept("stage1", stage("stage1", lambda: assess_stage1(prepared, cfg.preprocess),
                                        expected=QualityReport))

    report_progress(progress, 0.2, "Preparation and quality complete")
    estimators = {}
    recon = verification = None
    if quality is None or quality.status == "insufficient":
        code = NOT_RUN if quality is None else STAGE1_INSUFFICIENT
        estimators = {method: _unavailable(method, code) for method in ESTIMATOR_METHODS}
    else:
        unset = cfg.period_bounds.get("status") == "TBD"
        bounds = None if unset else stage("period_bounds", lambda: resolve_period_bounds(cfg.period_bounds))
        if bounds is None:
            code = PERIOD_BOUNDS_NOT_SET if unset else NOT_RUN
            estimators = {method: _unavailable(method, code) for method in ESTIMATOR_METHODS}
        else:
            injected = {"period_min_s": bounds.min_period_s, "period_max_s": bounds.max_period_s}
            for method, runner, config in (
                (CNN, run_cnn, cfg.cnn),
                (LOMB_SCARGLE, run_lomb_scargle, cfg.classical),
                (PDM, run_pdm, cfg.classical),
            ):
                def check_estimator(result):
                    if result.method != method:
                        raise ValueError("contract: estimator method mismatch")
                    if any(not bounds.min_period_s <= entry.period_s <= bounds.max_period_s
                           for entry in result.candidates):
                        raise ValueError("contract: candidate outside injected period bounds")

                result = stage(method, lambda: runner(prepared, {**config, **injected}),
                               expected=EstimatorResult, check=check_estimator)
                result = accept(method, result, check_estimator)
                estimators[method] = result if result is not None else _unavailable(
                    method, stage_errors[method], "failed", timings[method]
                )
        for method, result in estimators.items():
            if result.status != "ok":
                stage_errors[method] = result.error

        report_progress(progress, 0.45, "Period estimation complete")
        policies = injected if bounds is not None else {}
        reconcile_cfg = {**cfg.reconcile, **policies, "top_k": _FAST_SHORTLIST_LIMIT}
        window_result = None

        def run_reconciliation():
            nonlocal window_result
            if any(item.status == "ok" and item.candidates for item in estimators.values()):
                window_result = compute_window(prepared, {**cfg.classical, **policies})
                reconcile_cfg["window_result"] = window_result
            return reconcile(prepared, list(estimators.values()), reconcile_cfg)

        recon = stage("reconciliation", run_reconciliation,
                      expected=ReconciliationResult, check=reconcile_checks)
        recon = accept("reconciliation", recon, reconcile_checks)
        if recon is not None and recon.shortlist:
            report_progress(progress, 0.6, "Starting verification")

            def verify_progress(fraction, message):
                report_progress(progress, 0.6 + 0.25 * fraction, message)

            def verify_checks(result):
                check_verification_scope(recon, result)

            verification = stage("verification", lambda: verify_candidates(
                prepared, recon, {**cfg.verification, **policies, "window_result": window_result}, mode=mode,
                progress=verify_progress if progress is not None else None,
            ), expected=VerificationResult, check=verify_checks)
            verification = accept("verification", verification, verify_checks)
            if verification is not None and not verification.complete and not verification.is_stub:
                unfinished = [
                    f"{record.cluster_id}: {record.status} ({record.notes[-1] if record.notes else 'no detail'})"
                    for record in verification.per_candidate if record.status != "ok"
                ]
                stage_errors["verification"] = "Incomplete candidates: " + "; ".join(unfinished)

    verdict = None
    if (not stage2_pending(recon, estimators, verification)
            and _execution_error(prepared, quality, estimators, recon, verification, stage_errors) is None):
        verdict = stage("verdict", lambda: decide(quality, estimators, recon, verification, cfg.verdict),
                        expected=Verdict, check=verdict_checks)
        verdict = accept("verdict", verdict, verdict_checks)
        if verdict is None:
            verdict = _fallback_verdict(quality, estimators, recon, verification)
            if verdict is not None:
                # Fallback is independent of decide and must pass both validation layers.
                validate_result(verdict, verdict_checks)
    result = AnalysisResult(obs.obs_id, prepared, quality, estimators, recon, verification,
                            verdict, stage_errors, timings, mode, digest)
    result.validate()
    report_progress(progress, 1.0, "Analysis complete")
    return result


def _fallback_verdict(quality, estimators, recon, verification):
    outcome = _structural_outcome(quality, estimators, recon, verification)
    if outcome is None:
        return None
    label, reason = {
        "quality": ("Inconclusive", LOW_QUALITY),
        "unavailable": ("Inconclusive", CANDIDATE_INADEQUATE),
        "empty": ("Inconclusive", NO_PERIODIC_SIGNAL),
        "rejected": ("Inconclusive", CANDIDATE_INADEQUATE),
        "incomplete": ("Ambiguous", VERIFICATION_INCOMPLETE),
        "mismatch": ("Ambiguous", MODEL_MISMATCH),
    }[outcome]
    stage2_failures = sorted({r for c in (recon.clusters if recon else [])
                              for r in c.stage2["fail_reasons"]})
    summary = {
        "stage1_status": quality.status, "stage1_warnings": list(quality.warnings),
        "estimator_status": {k: v.status for k, v in estimators.items()},
        "n_estimators_available": sum(v.status == "ok" for v in estimators.values()),
        "estimators_agree": None, "alias_risk": None,
        "verification_complete": verification.complete if verification is not None else None,
        "heldout_error": None, "decoy_separation": None,
        "stage2_fail_reasons": stage2_failures,
        "selective_accuracy_validation": None,
    }
    if outcome in ("unavailable", "rejected"):
        cause = ALL_ESTIMATORS_UNAVAILABLE if outcome == "unavailable" else ALL_CANDIDATES_FAILED_STAGE2
        summary["inconclusive_cause"] = cause
    if verification is not None:
        scores = {c.cluster_id: c.score for c in recon.clusters}
        finished = [r for r in verification.per_candidate if r.status == "ok"]
        if finished:
            record = min(finished, key=lambda r: (not r.passed, -scores[r.cluster_id], r.cluster_id))
            summary["heldout_error"] = record.heldout_error
            summary["decoy_separation"] = min(record.decoy_delta_e, default=None)
    reasons = [reason]
    if quality.status == "limited":
        reasons.insert(0, LOW_QUALITY)
    alternatives = _alternatives(recon, verification) if label == "Ambiguous" else []
    return Verdict(label, None, None, alternatives, reasons, summary,
                   f"{label}. Structural fallback: {reason}. " + " ".join(quality.warnings))
