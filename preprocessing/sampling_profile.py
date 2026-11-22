"""Sampling statistics for the P1 to P2 handoff; CONTRACTS.md section 9."""
import argparse
import json
from pathlib import Path

import numpy as np
from core.config import load_config
from ingestion import load_observation
from preprocessing.prepare import prepare
from preprocessing.quality import assess_stage1


def sampling_profile(prep, cfg):
    prep.validate()
    duration = float(prep.t_s[-1] - prep.t_s[0])
    steps = np.diff(prep.t_s)
    positive = steps[steps > 0]
    if duration <= 0 or not positive.size:
        raise ValueError("sampling profile requires a positive duration and cadence")
    cadence = float(np.median(positive))
    quality = assess_stage1(prep, cfg)
    values = prep.y_norm[prep.valid_mask]
    amplitude = float(np.ptp(values)) if values.size else 0.
    if amplitude <= 0:
        raise ValueError("noise relative to amplitude is undefined for a flat observation")
    result = {"profile_id": prep.obs_id, "n_points": len(prep.t_s),
              "duration_s": duration, "median_dt_s": cadence,
              "gap_fractions": (positive[positive > cfg.get("gap_cadence_multiple", 3.) * cadence] / duration).tolist(),
              "noise_rel_amplitude": quality.metrics["noise_proxy"] / amplitude,
              "missing_fraction": float(1. - prep.valid_mask.mean())}
    json.dumps(result, allow_nan=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    files = sorted(args.input_directory.glob("*_DATA.txt"))
    if len(files) < 20:
        parser.error("at least 20 real SDLCD observations are required")
    cfg = load_config().preprocess
    profiles = [sampling_profile(prepare(load_observation(str(p), "sdlcd"), cfg), cfg) for p in files]
    with args.output.open("x", encoding="utf-8") as target:
        json.dump(profiles, target, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
