"""Reproduce the P1 CP2 experiments with python -m evaluation.cp2_evidence."""

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
from statistics import median
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter
from unittest.mock import patch

import numpy as np
from scipy.special import betainccinv

from classical_analysis import lomb_scargle as ls, pdm
from classical_analysis.window import compute_window
from core.config import load_config, resolve_period_bounds
from core.schemas import (LOMB_SCARGLE, PDM, EstimatorResult, Observation,
                          _execution_error, stage2_pending)
from ingestion import load_observation
from pipeline import controller
from preprocessing.prepare import prepare
from preprocessing.quality import assess_stage1


ROOT = Path(__file__).resolve().parents[1]
SDLCD_FILES = (
    "00029A_2_3_R_20190322_DATA.txt",
    "00041B_2_3_R_20191118_DATA.txt",
    "00080B_1_2_R_20190724_DATA.txt",
)
LS_ALPHAS = (0.01, 0.025, 0.05)
PDM_ALPHAS = (0.01, 0.025, 0.05, 0.10)


def digest(data):
    return sha256(data).hexdigest()


def payload_digest(value):
    return digest(json.dumps(value, sort_keys=True, allow_nan=False).encode())


def prepared(times, values, config):
    return prepare(Observation("cp2-synthetic", times, values, None, "unknown"), config)


def sampled_times(rng, n, cadence, jitter, drop, gap):
    times = np.cumsum(rng.uniform(*jitter, n) * cadence)
    keep = (rng.random(n) > drop) & ~((times > gap[0] * times[-1]) & (times < gap[1] * times[-1]))
    return times[keep]


def ls_calibration(config, preprocess):
    regimes = ((100, 2.0), (160, 3.0), (240, 1.5))
    jitter, drop, gap = (0.7, 1.3), 0.15, (0.35, 0.45)
    period = 37.0
    controls = ("white_noise", "amplitude_0.6", "amplitude_1.0")
    rows = []
    for split, start, count in (("selection", 1000, 150), ("holdout", 5000, 250)):
        for terms in (1, 2, 3):
            outcomes = np.zeros((len(controls), len(LS_ALPHAS)), dtype=int)
            for seed in range(start, start + count):
                rng = np.random.default_rng(seed)
                times = sampled_times(rng, *regimes[seed % 3], jitter, drop, gap)
                noise = rng.normal(size=times.size)
                phase = 2 * np.pi * times / period
                waveform = np.sin(phase)
                if terms > 1:
                    waveform += 0.65 * np.cos(2 * phase)
                if terms == 3:
                    waveform += 0.4 * np.sin(3 * phase)
                for row, values in enumerate((noise, noise + 0.6 * waveform, noise + waveform)):
                    prep = prepared(times, values, preprocess)
                    frequencies = ls.build_frequency_grid(prep, config)
                    power = ls._power(prep.t_s, prep.y_norm, frequencies, terms)
                    if not np.isfinite(power).all():
                        raise ValueError(f"Non-finite LS power for seed {seed}")
                    for col, alpha in enumerate(LS_ALPHAS):
                        cutoff = betainccinv(terms, (times.size - 1 - 2 * terms) / 2, alpha / frequencies.size)
                        candidates = ls._candidates(
                            frequencies, power, prep.meta["freq_resolution"], cutoff, 1,
                            config["period_min_s"], config["period_max_s"],
                        )
                        if candidates:
                            recovered = abs(candidates[0].period_s / period - 1) <= 0.01
                            outcomes[row, col] += row == 0 or recovered
            rows.append({"split": split, "seed_start": start, "seed_end_inclusive": start + count - 1,
                         "trials_per_control": count, "nterms": terms,
                         "counts_by_control": dict(zip(controls, outcomes.tolist(), strict=True))})
    return {
        "dataset": {"id": "cp2-ls-irregular-gapped-v1", "rng": "numpy.default_rng(seed), PCG64",
                    "regimes_points_cadence_s": regimes, "regime_index": "seed % 3",
                    "cadence_jitter_multiplier": jitter, "row_drop_probability": drop,
                    "gap_open_fraction_of_final_time": gap, "noise_sd": 1.0,
                    "period_s": period, "harmonic_amplitudes": [1.0, 0.65, 0.4],
                    "harmonic_phases": ["sin", "cos", "sin"], "amplitude_multipliers": [0.6, 1.0],
                    "relative_recovery_tolerance": 0.01},
        "alpha_columns": LS_ALPHAS, "configured_alpha": config["ls_false_alarm_alpha"],
        "count_definition": "White noise counts any candidate. Injections count top-ranked recovery.",
        "rows": rows,
    }


def pdm_calibration(config, preprocess):
    regimes = ((65, 1.8), (150, 2.2), (260, 1.6))
    jitter, drop, gap = (0.65, 1.35), 0.17, (0.36, 0.48)
    periods = (19.0, 33.0, 47.0)
    controls = ("white_noise", "sine_0.7", "sine_1.2", "two_harmonic_1.2", "ar1_0.65")
    calibration, correlated = [], []
    for split, start, count in (("selection", 3100, 240), ("holdout", 6100, 300)):
        outcomes = np.zeros((len(controls), len(PDM_ALPHAS)), dtype=int)
        for seed in range(start, start + count):
            rng = np.random.default_rng(seed)
            times = sampled_times(rng, *regimes[seed % 3], jitter, drop, gap)
            period = periods[(seed // 3) % 3]
            phase = 2 * np.pi * times / period + rng.uniform(0, 2 * np.pi)
            noise = rng.normal(size=times.size)
            colored = noise.copy()
            for index in range(1, colored.size):
                colored[index] = 0.65 * colored[index - 1] + np.sqrt(1 - 0.65 ** 2) * noise[index]
            sine = np.sin(phase)
            harmonic = sine + 0.55 * np.cos(2 * phase)
            for row, values in enumerate((noise, noise + 0.7 * sine, noise + 1.2 * sine,
                                          noise + 1.2 * harmonic, colored)):
                prep = prepared(times, values, preprocess)
                frequencies = ls.build_frequency_grid(prep, config)
                theta, within, total = pdm._phase_dispersion(
                    prep.t_s, prep.y_norm, frequencies,
                    config["pdm_phase_bins"], config["pdm_min_bin_occupancy"],
                )
                probabilities = pdm._trial_probabilities(theta, within, total)
                if not np.isfinite(theta).all() or not np.isfinite(probabilities).all():
                    raise ValueError(f"Non-finite PDM statistic for seed {seed}")
                for col, alpha in enumerate(PDM_ALPHAS):
                    candidates = pdm._candidates(
                        frequencies, 1 - theta, within > 0, probabilities < alpha / frequencies.size,
                        prep.meta["freq_resolution"], 1, config,
                    )
                    if candidates:
                        recovered = abs(candidates[0].period_s / period - 1) <= 0.02
                        outcomes[row, col] += row in (0, 4) or recovered
        split_info = {"split": split, "seed_start": start, "seed_end_inclusive": start + count - 1,
                      "trials_per_control": count}
        calibration.append({**split_info, "counts_by_control": dict(zip(controls[:-1], outcomes[:-1].tolist(),
                                                                       strict=True))})
        correlated.append({**split_info, "counts_by_control": dict(zip((controls[0], controls[-1]),
                                                                      outcomes[[0, -1]].tolist(), strict=True))})
    context = {
        "dataset": {"id": "cp2-pdm-irregular-gapped-v1", "rng": "numpy.default_rng(seed), PCG64",
                    "regimes_points_cadence_s": regimes, "regime_index": "seed % 3",
                    "cadence_jitter_multiplier": jitter, "row_drop_probability": drop,
                    "gap_open_fraction_of_final_time": gap, "noise_sd": 1.0,
                    "periods_s": periods, "period_index": "(seed // 3) % 3",
                    "initial_phase_rad": "uniform(0, 2*pi)", "second_harmonic_amplitude": 0.55,
                    "relative_recovery_tolerance": 0.02},
        "alpha_columns": PDM_ALPHAS, "configured_alpha": config["pdm_false_alarm_alpha"],
    }
    return (
        {**context, "count_definition": "White noise counts any candidate. Injections count top-ranked recovery.",
         "rows": calibration},
        {**context, "ar1": {"coefficient": 0.65, "innovation_sd": float(np.sqrt(1 - 0.65 ** 2)),
                            "time_step": "successive retained epochs", "initial_value": "first white-noise draw",
                            "paired_controls": "same retained cadences, seeds and Gaussian innovations"},
         "count_definition": "Both controls count any candidate.", "rows": correlated},
    )


def pdm_benchmark(config, preprocess):
    seed, count, trials, period = 842, 2000, 5000, 47.0
    rng = np.random.default_rng(seed)
    times = np.cumsum(rng.uniform(1.4, 2.6, count))
    prep = prepared(times, np.sin(2 * np.pi * times / period), preprocess)
    grid = ls.build_frequency_grid(prep, config)
    # The grid helper rounds the number of intervals upward.
    oversampling = (trials - 1.5) * prep.meta["freq_resolution"] / (grid[-1] - grid[0])
    benchmark_config = dict(config, frequency_oversampling=oversampling)
    actual_trials = ls.build_frequency_grid(prep, benchmark_config).size
    if actual_trials != trials:
        raise ValueError(f"Expected {trials} trials, got {actual_trials}")
    runtimes, periods = [], []
    for _ in range(5):
        result = pdm.run_pdm(prep, benchmark_config)
        result.validate()
        runtimes.append(result.meta["runtime_s"])
        periods.append(result.candidates[0].period_s)
    return {"dataset": {"id": "cp2-pdm-benchmark-v1", "seed": seed, "cadence_uniform_s": [1.4, 2.6],
                        "observations": count, "injected_period_s": period},
            "trial_periods": actual_trials, "config_used": benchmark_config,
            "timing_method": "Five consecutive estimator runtime_s measurements, no discarded warm-up",
            "runtime_s": runtimes, "median_runtime_s": median(runtimes), "top_period_s": periods}


def estimator_output(result):
    result.validate()
    data = result.to_dict()
    restored = EstimatorResult.from_dict(json.loads(json.dumps(data, allow_nan=False)))
    if restored.to_dict() != data or result.status != "ok" or result.is_stub:
        raise ValueError("Expected a valid non-stub estimator result with a stable JSON round-trip")
    periodogram = data["meta"].pop("periodogram")
    data["meta"]["periodogram_summary"] = {"points": len(periodogram["freq_hz"]),
                                          "sha256_canonical_json": payload_digest(periodogram)}
    return data


def analysis_disposition(analysis):
    """Classify only validated schema-v3 analyses; no missing verdict is a prediction."""
    analysis.validate()
    if analysis.verdict is not None:
        return "completed", None
    if stage2_pending(analysis.reconciliation, analysis.estimators, analysis.verification):
        return "stage2_pending", None
    failed_stage = _execution_error(analysis.prepared, analysis.quality, analysis.estimators,
                                    analysis.reconciliation, analysis.verification, analysis.stage_errors)
    if failed_stage is None:
        raise ValueError("Invalid nullable-verdict analysis")
    return "execution_error", failed_stage


def analysis_evidence(analysis):
    """Keep legacy completed records; retain the full analysis for unfinished runs."""
    disposition, failed_stage = analysis_disposition(analysis)
    if disposition == "completed":
        return {"controller_stage_errors": analysis.stage_errors,
                "controller_verdict": analysis.verdict.label}
    return {"controller_stage_errors": analysis.stage_errors, "controller_verdict": None,
            "analysis_disposition": disposition, "failed_stage": failed_stage,
            "analysis_result": analysis.to_dict()}


def real_outputs(config, classical, directory):
    outputs = []
    with TemporaryDirectory(prefix="tumblescope-cp2-") as temp, patch.object(controller, "_OUTPUTS", Path(temp)):
        for filename in SDLCD_FILES:
            path = directory / filename
            observation = load_observation(str(path))
            analysis = controller.analyze(observation, config)
            analysis.validate()
            disposition, _ = analysis_disposition(analysis)
            if disposition == "execution_error":
                outputs.append({"file": filename, "sha256": digest(path.read_bytes()),
                                **analysis_evidence(analysis)})
                continue
            prep, quality = analysis.prepared, analysis.quality
            estimators = {}
            for method, runner in ((LOMB_SCARGLE, ls.run_lomb_scargle), (PDM, pdm.run_pdm)):
                source = analysis.estimators[method]
                if source.status != "ok":
                    estimators[method] = source.to_dict()
                    estimators[method]["direct_comparison"] = None
                    continue
                integrated = estimator_output(source)
                try:
                    comparison = runner(prep, classical)
                    comparison.validate()
                    if comparison.status != "ok":
                        raise ValueError(f"{comparison.status}: {comparison.error}")
                    direct = estimator_output(comparison)
                except Exception as error:
                    estimators[method] = {**integrated, "direct_comparison": None,
                                          "direct_comparison_error": str(error) or type(error).__name__}
                    continue
                direct_runtime = direct["meta"].pop("runtime_s")
                integrated_runtime = integrated["meta"].pop("runtime_s")
                if direct != integrated:
                    raise ValueError(f"Controller {method} differs for {filename}")
                estimators[method] = {**direct, "direct_runtime_s": direct_runtime,
                                      "controller_runtime_s": integrated_runtime}
            started = perf_counter()
            window = compute_window(prep, classical)
            window_time = perf_counter() - started
            grid, power = window["freq_hz"], window["window_power"]
            peaks = window["peaks"]
            if (set(window) != {"freq_hz", "window_power", "freq_resolution", "peaks"}
                    or len(grid) != len(power) or grid[0] != 0 or power[0] != 1
                    or not np.all(np.diff(grid) > 0) or not all(0 <= height <= 1 for height in power)
                    or window["freq_resolution"] != prep.meta["freq_resolution"]
                    or len(peaks) > classical["n_peaks"]
                    or any(set(peak) != {"freq_hz", "height"} or peak["freq_hz"] <= 0 for peak in peaks)
                    or [peak["height"] for peak in peaks] != sorted((peak["height"] for peak in peaks), reverse=True)
                    or json.loads(json.dumps(window, allow_nan=False)) != window):
                raise ValueError(f"Window schema failure for {filename}")
            outputs.append({
                "file": filename, "sha256": digest(path.read_bytes()),
                "brightness_kind": observation.brightness_kind, "brightness_sign": prep.norm_params["sign"],
                "n_rows": prep.meta["n_rows"], "n_valid": prep.meta["n_valid"], "quality": quality.to_dict(),
                "estimators": estimators, "window": window, "window_runtime_s": window_time,
                **analysis_evidence(analysis),
            })
    counts = Counter(row.get("analysis_disposition", "completed") for row in outputs)
    return {"dataset": "Three local SDLCD observations used for the P1 backend smoke check",
            "selection": "Two files with candidates in both estimators and one with two successful empty lists",
            "trial_count": len(outputs),
            "controller_matches_direct_excluding_runtime": (
                True if all("estimators" in row and all(
                    item.get("direct_comparison", True) is not None
                    for item in row["estimators"].values()) for row in outputs) else None),
            "disposition_counts": dict(counts), "scientific_prediction_denominator": counts["completed"],
            "denominator_policy": "Only completed verdicts enter scientific prediction metrics; pending and execution-error records remain in trial_count and are reported separately.",
            "outputs": outputs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdlcd-dir", type=Path, default=Path("data/sdlcd_raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/results"))
    args = parser.parse_args()
    config = load_config()
    bounds = resolve_period_bounds(config.period_bounds)
    classical = dict(config.classical, period_min_s=bounds.min_period_s, period_max_s=bounds.max_period_s)
    source_files = ("evaluation/cp2_evidence.py", "classical_analysis/lomb_scargle.py",
                    "classical_analysis/pdm.py", "classical_analysis/window.py",
                    "preprocessing/prepare.py", "preprocessing/quality.py", "ingestion/sdlcd.py",
                    "pipeline/controller.py", "configs/classical.yaml", "configs/period_bounds.yaml",
                    "configs/preprocess.yaml")
    common = {
        "schema_version": 1, "owner": "P1", "checkpoint": "CP2", "type": "measured",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "command_argv": ["python", "-m", "evaluation.cp2_evidence", "--sdlcd-dir", str(args.sdlcd_dir),
                         "--output-dir", str(args.output_dir)],
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "processor": platform.processor(), "logical_cpus": os.cpu_count(),
                        "packages": {name: version(name) for name in ("numpy", "scipy", "astropy", "pytest")}},
        "git": {"branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
                "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "source_sha256": {name: digest((ROOT / name).read_bytes()) for name in source_files}},
        "pipeline_config": asdict(config), "injected_classical_config": classical,
    }

    def write(name, result, began):
        path = args.output_dir / f"p1_cp2_{name}.json"
        payload = {**common, "experiment": name, "experiment_duration_s": perf_counter() - began, **result}
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        print(path, flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    began = perf_counter()
    write("ls_calibration", ls_calibration(classical, config.preprocess), began)
    began = perf_counter()
    calibration, correlated = pdm_calibration(classical, config.preprocess)
    write("pdm_calibration", calibration, began)
    write("pdm_correlated_noise", correlated, began)
    began = perf_counter()
    write("pdm_benchmark", pdm_benchmark(classical, config.preprocess), began)
    began = perf_counter()
    write("sdlcd_outputs", real_outputs(config, classical, args.sdlcd_dir), began)


if __name__ == "__main__":
    main()
