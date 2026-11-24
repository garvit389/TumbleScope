"""Evaluate SDLCD validation and synthetic development observations for Day 4."""

import argparse
from collections import Counter
import csv
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from math import isfinite
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zipfile import BadZipFile

import numpy as np
import pandas as pd

from app.plots import fig_calibration, fig_risk_coverage
from core.config import PipelineConfig, config_hash, load_config, resolve_period_bounds
from core.errors import IngestionError
from core.schemas import (
    ALL_ESTIMATORS_UNAVAILABLE, CNN, LOMB_SCARGLE, PDM, SCHEMA_VERSION, Observation, stage2_pending,
)
from evaluation import metrics, stress
from ingestion import load_observation
from ingestion.reference import load_reference_table
from pipeline import controller

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "real-val": ("val", "data/reference/reference_table.csv"),
    "synthetic-dev": ("development", "data/synthetic_holdout/holdout_small.npz"),
}
METHODS = {"ls_only": LOMB_SCARGLE, "pdm_only": PDM, "cnn_only": CNN}
CONFIGURATIONS = (*METHODS, "cnn_classical", "complete")
STRESSES = ("noise", "crop", "gap", "modulation", "clustered")
CNN_EVENT = "cnn_no_reliable_period"
ARTIFACTS = ("models/cnn_v1.pt", "models/cnn_v1.sha256", "docs/results/cnn_calibration.json")
DEFAULTS = dict(limit=None, stress=[], seed=17, tolerance=0.05, harmonic_tolerance=0.05,
                catastrophic_factor=2.0, recall_k=3, bins=10)


@dataclass(frozen=True)
class Case:
    obs_id: str
    object_id: str | None
    period_s: float | None
    no_reliable_period: bool | None
    source: str
    source_sha256: str | None
    observation: Observation | None
    input_status: str = "ok"
    error: str | None = None


def _digest(path):
    return sha256(path.read_bytes()).hexdigest()


def _real_cases(reference, directory, limit):
    labels = load_reference_table(str(reference))
    selected = sorted((label for label in labels.values() if label.split == "val"),
                      key=lambda label: label.obs_id)
    if not selected:
        raise ValueError("reference table contains no validation observations")
    for label in selected[:limit]:
        identity = label.obs_id
        if Path(identity).name != identity or any(char in identity for char in "/\\:"):
            raise ValueError(f"invalid SDLCD observation ID: {identity!r}")
        path = directory / f"{identity}.txt"
        digest = None
        observation = None
        status, error = "ok", None
        try:
            if path.is_symlink() or path.resolve() != directory.resolve() / path.name:
                raise ValueError("SDLCD input must be a direct observation file")
            digest = _digest(path)
            observation = load_observation(str(path), fmt="sdlcd")
            if observation.obs_id != identity:
                raise ValueError("SDLCD observation ID does not match the reference")
        except FileNotFoundError as exc:
            status, error = "missing", str(exc)
        except (IngestionError, OSError, ValueError) as exc:
            status, error = "failed", str(exc)
        yield Case(identity, label.object_id, label.period_s, None, str(path), digest,
                   observation if status == "ok" else None, status, error)


def _synthetic_cases(path, cfg, limit, digest):
    names = ("groups", "periods", "offsets", "t_s", "brightness", "bounds")
    try:
        with np.load(path, allow_pickle=False) as shard:
            if set(shard.files) != set(names):
                raise ValueError("unexpected holdout arrays")
            groups, periods, offsets, times, brightness, bounds = (shard[name] for name in names)
    except (OSError, ValueError, TypeError, EOFError, BadZipFile) as exc:
        raise ValueError(f"cannot read synthetic development holdout {path}: {exc}") from exc
    if (groups.ndim != 1 or not groups.size or groups.dtype.kind not in "iu"
            or np.any(groups // 1_000_000 != 1) or np.unique(groups).size != groups.size):
        raise ValueError("holdout must contain unique synthetic development groups only")
    if (offsets.shape != (groups.size + 1,) or offsets.dtype.kind not in "iu"
            or times.ndim != 1 or brightness.shape != times.shape
            or times.dtype.kind not in "iuf" or brightness.dtype.kind not in "iuf"
            or offsets[0] != 0 or offsets[-1] != times.size
            or np.any(offsets[1:] <= offsets[:-1])):
        raise ValueError("invalid holdout offsets or measurement arrays")
    if (bounds.dtype.kind not in "iuf" or bounds.shape != (2,)
            or not np.array_equal(bounds, resolve_period_bounds(cfg.period_bounds))):
        raise ValueError("holdout period bounds do not match the active config")
    if periods.shape != groups.shape or periods.dtype.kind not in "iuf":
        raise ValueError("invalid holdout period labels")
    for index in np.argsort(groups, kind="stable")[:limit]:
        identity = str(int(groups[index]))
        segment = slice(*offsets[index:index + 2])
        period = float(periods[index])
        # Shard group IDs encode the simulator category and stay in evaluation.
        opaque_id = sha256(f"cp4-synthetic:{identity}".encode()).hexdigest()
        observation = Observation(opaque_id, times[segment].astype(float),
                                  brightness[segment].astype(float), None, "unknown", {})
        label_valid = period == 0 or bounds[0] <= period <= bounds[1]
        try:
            if not label_valid:
                raise ValueError("holdout period label is outside the active bounds")
            observation.validate()
        except ValueError as exc:
            yield Case(identity, None, period if label_valid and period else None,
                       period == 0 if label_valid else None, str(path), digest, None, "failed", str(exc))
        else:
            yield Case(identity, None, period or None, period == 0, str(path), digest, observation)


def _stress_variant(observation, name, seed):
    if name == "modulation":
        params = {"factor": 0.5}
        return stress.reduce_modulation(observation, **params), params
    if name == "noise":
        values = observation.y[np.isfinite(observation.y)]
        if not values.size:
            raise ValueError("noise requires finite brightness samples")
        if observation.brightness_kind == "flux":
            if np.any(values <= 0):
                raise ValueError("noise requires positive flux")
            values = np.log(values)
        sigma = float(np.std(values)) * 0.5
        if not isfinite(sigma) or sigma <= 0:
            raise ValueError("relative noise requires finite, nonzero brightness variation")
        params = {"sigma": sigma, "seed": seed}
        return stress.add_measurement_noise(observation, **params), params
    times = observation.t_s[np.isfinite(observation.y)]
    if not times.size:
        raise ValueError("sampling stress requires valid epochs")
    start = float(times.min())
    span = float(times.max()) - start
    if name == "crop":
        params = {"start_s": start, "duration_s": span * 0.5}
        return stress.crop_baseline(observation, **params), params
    if name == "gap":
        params = {"start_s": start + span * 0.35, "end_s": start + span * 0.65}
        return stress.remove_gap(observation, **params), params
    params = {"n_clusters": 3, "cluster_width_s": span * 0.10, "seed": seed}
    return stress.clustered_irregular_sampling(observation, **params), params


def _variant_seed(seed, dataset, obs_id, variant):
    key = json.dumps([seed, dataset, obs_id, variant], separators=(",", ":"))
    return int.from_bytes(sha256(key.encode()).digest()[:8], "big")


def _reading(status, *, candidates=(), period=None, score=None, error=None, is_stub=False):
    return dict(status=status, candidate_periods_s=[item.period_s for item in candidates],
                period_s=period, ranking_score=score, error=error, is_stub=is_stub)


def _read_configurations(result):
    readings = {}
    for name, method in METHODS.items():
        estimator = result.estimators[method]
        lead = estimator.candidates[0] if estimator.candidates else None
        status = estimator.status
        if status == "ok":
            status = "predicted" if lead else "abstained"
        readings[name] = _reading(
            status, candidates=estimator.candidates, period=lead.period_s if lead else None,
            score=lead.raw_score if lead else None, error=estimator.error, is_stub=estimator.is_stub,
        )

    recon, verdict = result.reconciliation, result.verdict
    pending = stage2_pending(recon, result.estimators, result.verification)
    clusters = {cluster.cluster_id: cluster for cluster in recon.clusters} if recon else {}
    shortlist = [clusters[identity] for identity in recon.shortlist] if recon else []
    if pending:
        fused = _reading("pending", is_stub=True)
    elif shortlist:
        lead = shortlist[0]
        fused = _reading("predicted", candidates=shortlist, period=lead.period_s,
                         score=lead.score, is_stub=recon.is_stub)
    elif recon is not None:
        statuses = {estimator.status for estimator in result.estimators.values()}
        status = "abstained" if "ok" in statuses else ("failed" if "failed" in statuses else "skipped")
        fused = _reading(status, is_stub=recon.is_stub)
    elif any(stage in result.stage_errors for stage in ("prepare", "stage1", "reconciliation")):
        fused = _reading("execution_error")
    else:
        fused = _reading("skipped")
    readings["cnn_classical"] = fused

    if verdict is None:
        full = _reading("pending" if pending else "execution_error", is_stub=pending)
    elif verdict.label == "Supported":
        lead = clusters.get(verdict.best_cluster_id)
        full = _reading("supported", period=verdict.best_period_s, score=lead.score if lead else None,
                        is_stub=verdict.is_stub)
        full["candidate_periods_s"] = [verdict.best_period_s]
    elif verdict.evidence_summary.get("inconclusive_cause") == ALL_ESTIMATORS_UNAVAILABLE:
        status = "execution_error" if any(
            estimator.status == "failed" for estimator in result.estimators.values()
        ) else "skipped"
        full = _reading(status, is_stub=verdict.is_stub)
    else:
        full = _reading("abstained", is_stub=verdict.is_stub)
    readings["complete"] = full
    return readings


def evaluate(cases, cfg, *, dataset, variants=(), seed=17, analyze=None):
    """Analyze each available variant once, passing only observations and config."""
    if analyze is None:
        analyze = controller.analyze
    digest = config_hash(cfg)
    rows = []
    for case in cases:
        for variant in ("clean", *variants):
            variant_seed = None if variant == "clean" else _variant_seed(seed, dataset, case.obs_id, variant)
            observation = case.observation
            status, error = case.input_status, case.error
            params = {}
            result = None
            if observation is not None and variant != "clean":
                try:
                    observation, params = _stress_variant(observation, variant, variant_seed)
                except ValueError as exc:
                    observation = None
                    status, error = "skipped", str(exc)
            if observation is not None:
                try:
                    result = analyze(observation, cfg)
                    result.validate()
                    if result.obs_id != observation.obs_id or result.config_hash != digest:
                        raise ValueError("analysis observation or configuration identity mismatch")
                except Exception as exc:
                    status, error = "execution_error", f"{type(exc).__name__}: {exc}"
                    result = None
            row = dict(dataset=dataset, split=DATASETS[dataset][0], obs_id=case.obs_id,
                       analysis_obs_id=observation.obs_id if observation is not None else None,
                       object_id=case.object_id, reference_period_s=case.period_s,
                       no_reliable_period=case.no_reliable_period, variant=variant,
                       variant_seed=variant_seed, stress_parameters=params, source=case.source,
                       source_sha256=case.source_sha256, input_status=status, error=error,
                       quality=None, p_no_reliable=None, cnn_abstained=None, verification_complete=None,
                       verdict=None, verdict_reason_codes=[], inconclusive_cause=None, stage_errors={}, stage_timings_s={},
                       configurations={name: _reading(status, error=error) for name in CONFIGURATIONS})
            if result is not None:
                cnn, verdict = result.estimators[CNN], result.verdict
                row.update(quality=result.quality.status if result.quality else None,
                           verification_complete=result.verification.complete if result.verification else None,
                           verdict=verdict.label if verdict else None,
                           verdict_reason_codes=verdict.reason_codes if verdict else [],
                           inconclusive_cause=verdict.evidence_summary.get("inconclusive_cause") if verdict else None,
                           stage_errors=result.stage_errors, stage_timings_s=result.timings_s,
                           configurations=_read_configurations(result))
                if cnn.status == "ok":
                    row["cnn_abstained"] = cnn.meta.get("abstained")
                    if not cnn.is_stub:
                        row["p_no_reliable"] = cnn.meta.get("p_no_reliable")
            rows.append(row)
    return rows


def _summarize_configuration(rows, name, options):
    tolerance = options["tolerance"]
    truth = [row["reference_period_s"] for row in rows]
    nrp = [row["no_reliable_period"] is True for row in rows]
    readings = [row["configurations"][name] for row in rows]
    predicted = [item["period_s"] for item in readings]
    candidates = [item["candidate_periods_s"] for item in readings]
    statuses = [item["status"] for item in readings]
    correctness = []
    for reference, period, no_period in zip(truth, predicted, nrp, strict=True):
        if period is None:
            correctness.append(None)
        elif no_period:
            correctness.append(False)
        elif reference is None:
            correctness.append(None)
        else:
            ratio = period / reference
            correctness.append(1 - tolerance <= ratio <= 1 + tolerance)
    n = len(rows)
    emitted = sum(period is not None for period in predicted)
    summary = dict(
        n=n, status_counts=dict(Counter(statuses)),
        prediction_coverage=dict(value=emitted / n, numerator=emitted, denominator=n),
        median_log_period_error=metrics.median_log_period_error(truth, predicted),
        tolerance_accuracy=metrics.tolerance_accuracy(truth, predicted, tolerance=tolerance),
        harmonic_error_rate=metrics.harmonic_error_rate(
            truth, predicted, tolerance=tolerance, harmonic_tolerance=options["harmonic_tolerance"]),
        catastrophic_error_rate=metrics.catastrophic_error_rate(truth, predicted, factor=options["catastrophic_factor"]),
        recall_at_k=metrics.recall_at_k(truth, candidates, k=options["recall_k"], tolerance=tolerance),
        risk_coverage=metrics.risk_coverage_points([item["ranking_score"] for item in readings], correctness),
    )
    if name == "cnn_classical":
        upstream = [[period for method in METHODS
                     for period in row["configurations"][method]["candidate_periods_s"]] for row in rows]
        summary["candidate_retention"] = metrics.candidate_retention(
            truth, upstream, candidates, tolerance=tolerance)
    if name == "complete":
        dispositions = [status if status in {"supported", "abstained", "pending", "execution_error"}
                        else "missing" for status in statuses]
        summary["supported_metrics"] = metrics.supported_metrics(
            truth, predicted, dispositions, no_reliable_period=nrp, tolerance=tolerance)
    if name == "cnn_only" and rows[0]["dataset"] == "synthetic-dev" and rows[0]["variant"] == "clean":
        probabilities = [row["p_no_reliable"] for row in rows]
        outcomes = [row["no_reliable_period"] for row in rows]
        summary["probability_event"] = CNN_EVENT
        summary["brier"] = metrics.brier_score(probabilities, outcomes)
        summary["ece"] = metrics.expected_calibration_error(probabilities, outcomes, n_bins=options["bins"])
    return summary


def summarize(rows, options):
    cohorts = {}
    for row in rows:
        key = (row["dataset"], row["split"], row["variant"])
        for quality in ("all", row["quality"] or "unavailable"):
            cohorts.setdefault((*key, quality), []).append(row)
    return [
        dict(dataset=dataset, split=split, variant=variant, quality=quality, configuration=name,
             **_summarize_configuration(cohort, name, options))
        for (dataset, split, variant, quality), cohort in sorted(cohorts.items())
        for name in CONFIGURATIONS
    ]


def _write_csv(path, records):
    columns = list(dict.fromkeys(key for record in records for key in record))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow({key: json.dumps(value, sort_keys=True, allow_nan=False)
                             if isinstance(value, (dict, list)) else value for key, value in record.items()})


def _publish(output, manifest, rows, summaries):
    flat_runs = [
        {**{key: value for key, value in row.items() if key != "configurations"},
         "configuration": name, **{key: value for key, value in reading.items() if key != "error"},
         "configuration_error": reading["error"]}
        for row in rows for name, reading in row["configurations"].items()
    ]
    overall = [item for item in summaries if item["quality"] == "all"]
    risk = [{"configuration": f'{item["configuration"]} / {item["variant"]}', **point}
            for item in overall for point in item["risk_coverage"]]
    calibration = [{"configuration": CNN_EVENT, **point} for item in overall
                   for point in item.get("ece", {}).get("bins", [])]
    risk_figure = fig_risk_coverage(pd.DataFrame(risk))
    risk_figure.update_xaxes(title_text="Coverage among scored predictions with evaluable truth")
    calibration_figure = fig_calibration(pd.DataFrame(calibration))
    if calibration:
        calibration_figure.update_xaxes(title_text="Predicted probability of No Reliable Period")
        calibration_figure.update_yaxes(title_text="Observed No Reliable Period frequency")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".tumblescope-eval-", dir=output.parent) as scratch:
        staging = Path(scratch) / "report"
        staging.mkdir()
        _write_csv(staging / "runs.csv", flat_runs)
        _write_csv(staging / "summary.csv", summaries)
        for name, data in (("summary", summaries), ("manifest", manifest)):
            payload = json.dumps(data, indent=2, sort_keys=True, allow_nan=False)
            (staging / f"{name}.json").write_text(payload + "\n", encoding="utf-8")
        risk_figure.write_html(staging / "risk_coverage.html", include_plotlyjs=True)
        calibration_figure.write_html(staging / "calibration.html", include_plotlyjs=True)
        staging.rename(output.resolve())


def _revision():
    try:
        return subprocess.check_output(
            ["git", "--no-optional-locks", "rev-parse", "--verify", "HEAD"],
            cwd=ROOT, text=True, stderr=subprocess.DEVNULL, timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _read_replay(path):
    saved = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(saved, dict) or saved.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("saved evaluation manifest requires the active analysis schema")
    config = saved["config"]
    if not isinstance(config, dict) or set(config) != {field.name for field in fields(PipelineConfig)} or not all(
        isinstance(section, dict) for section in config.values()
    ):
        raise ValueError("saved manifest must contain every configuration section")
    cfg = PipelineConfig(**config)
    if config_hash(cfg) != saved["config_hash"]:
        raise ValueError("saved configuration hash mismatch")
    return cfg, saved


def _validate_options(options):
    if not isinstance(options, dict) or options.keys() != DEFAULTS.keys():
        raise ValueError("invalid evaluation settings")
    limit = options["limit"]
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be a positive integer")
    if type(options["seed"]) is not int or options["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    variants = options["stress"]
    if (not isinstance(variants, list) or any(name not in STRESSES for name in variants)
            or len(variants) != len(set(variants))):
        raise ValueError("choose each supported stress variant at most once")
    for name in ("tolerance", "harmonic_tolerance", "catastrophic_factor"):
        value = options[name]
        if type(value) not in (float, int) or not isfinite(value):
            raise ValueError(f"{name} must be finite")
        if not (value > 1 if name == "catastrophic_factor" else 0 <= value < 1):
            raise ValueError(f"{name} is outside its valid range")
    for name in ("recall_k", "bins"):
        if type(options[name]) is not int or options[name] < 1:
            raise ValueError(f"{name} must be a positive integer")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", choices=DATASETS)
    source.add_argument("--replay", type=Path, metavar="MANIFEST")
    parser.add_argument("--output", required=True, type=Path, help="new report directory")
    parser.add_argument("--limit", type=int, help="select the first N sorted observation IDs")
    parser.add_argument("--stress", action="append", choices=STRESSES,
                        help="add this variant alongside the clean observation (repeatable)")
    parser.add_argument("--seed", type=int, help="stress seed (default 17)")
    parser.add_argument("--tolerance", type=float, help="relative direct-hit tolerance (default 0.05)")
    parser.add_argument("--harmonic-tolerance", type=float, help="relative harmonic tolerance (default 0.05)")
    parser.add_argument("--catastrophic-factor", type=float, help="strict error factor (default 2)")
    parser.add_argument("--recall-k", type=int, help="candidate recall cutoff (default 3)")
    parser.add_argument("--bins", type=int, help="equal-width calibration bins (default 10)")
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError("output directory already exists")
        if args.replay:
            if any(getattr(args, name) is not None for name in DEFAULTS):
                raise ValueError("replay uses the saved settings without overrides")
            cfg, saved = _read_replay(args.replay)
            dataset, options = saved["dataset"], saved["options"]
        else:
            cfg, saved = load_config(), None
            dataset = args.dataset
            options = {name: value if getattr(args, name) is None else getattr(args, name)
                       for name, value in DEFAULTS.items()}
        _validate_options(options)
        split, relative = DATASETS[dataset]
        if saved is not None and saved.get("split") != split:
            raise ValueError("saved split does not match the approved Day 4 dataset")
        source_path = ROOT / relative
        source_digest = _digest(source_path)
        if dataset == "real-val":
            cases = list(_real_cases(source_path, ROOT / "data/sdlcd_raw", options["limit"]))
        else:
            cases = list(_synthetic_cases(source_path, cfg, options["limit"], source_digest))
        artifacts = {name: _digest(ROOT / name) if (ROOT / name).is_file() else None for name in ARTIFACTS}
        inputs = [dict(obs_id=case.obs_id, sha256=case.source_sha256, status=case.input_status)
                  for case in cases]
        if saved is not None and (
            saved["source_sha256"] != source_digest or saved["inputs"] != inputs
            or saved["cnn_artifacts_sha256"] != artifacts
        ):
            raise ValueError("replay inputs or CNN artifacts differ from the saved run")
        manifest = dict(dataset=dataset, split=split, source=relative, source_sha256=source_digest,
                        inputs=inputs, cnn_artifacts_sha256=artifacts, options=options,
                        n_observations=len(cases), date_utc=datetime.now(timezone.utc).isoformat(),
                        commit=_revision(), schema_version=SCHEMA_VERSION, mode="fast",
                        config=asdict(cfg), config_hash=config_hash(cfg),
                        runner_sha256=_digest(Path(__file__)),
                        stress_truth="Original labels describe recoverability under degradation.",
                        package_versions={name: version(name) for name in ("numpy", "scipy", "astropy", "torch")})
        # Match CP2 evidence: isolate the controller's normal snapshot writes.
        with TemporaryDirectory(prefix="tumblescope-eval-") as scratch:
            with patch.object(controller, "_OUTPUTS", Path(scratch)):
                rows = evaluate(cases, cfg, dataset=dataset, variants=options["stress"], seed=options["seed"])
        _publish(args.output, manifest, rows, summarize(rows, options))
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
        parser.exit(2, f"Evaluation failed: {exc}\n")
    diagnostics = sum(row["input_status"] != "ok" or bool(row["stage_errors"]) for row in rows)
    print(f"{dataset} ({split}): {len(cases)} observations, {len(rows)} variants, "
          f"{diagnostics} rows with input or stage diagnostics. Reports: {args.output}")


if __name__ == "__main__":
    main()
