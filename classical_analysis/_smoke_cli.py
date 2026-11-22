"""Shared input and reporting for the classical SDLCD smoke commands."""

import argparse
import json
import sys

from core.config import load_config, resolve_period_bounds
from core.schemas import EstimatorResult
from ingestion import load_observation
from preprocessing.prepare import prepare


def run_smoke(runner, description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("sdlcd_file", metavar="SDLCD-file", help="path to an SDLCD measurement file")
    args = parser.parse_args()

    try:
        observation = load_observation(args.sdlcd_file, fmt="sdlcd")
        config = load_config()
        bounds = resolve_period_bounds(config.period_bounds)
        prepared = prepare(observation, config.preprocess)
        result = runner(prepared, {
            **config.classical,
            "period_min_s": bounds.min_period_s,
            "period_max_s": bounds.max_period_s,
        })
        if isinstance(result, EstimatorResult):
            result.validate()
            report = result.to_dict()
            periodogram = report["meta"].pop("periodogram", None)
            if periodogram is not None:
                report["meta"]["periodogram_points"] = len(periodogram["freq_hz"])
        else:
            report = {
                "status": "ok",
                "freq_resolution": result["freq_resolution"],
                "window_grid_points": len(result["freq_hz"]),
                "max_freq_hz": result["freq_hz"][-1],
                "peaks": result["peaks"],
            }
        output = json.dumps({"obs_id": prepared.obs_id, **report}, allow_nan=False)
    except Exception as exc:
        print(f"Error: {str(exc) or type(exc).__name__}", file=sys.stderr)
        return 1

    print(output)
    if report["status"] == "failed":
        print(f"Error: {report['error']}", file=sys.stderr)
        return 1
    return 0
