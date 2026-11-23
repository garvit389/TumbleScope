"""Fit and check CNN calibration on the verified CP2 development bundle."""

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
import torch

from core.config import load_config, resolve_period_bounds
from ml.bundle import FEATURE_KEYS, digest, verify_bundle
from ml.dataset.labels import encode_target
from ml.infer import _CALIBRATION, _CHECKPOINT, _ROOT, _load_model, _period_peaks, _window_log_probabilities
from ml.train import centers, save_json, score


def _split_groups(groups: np.ndarray, hard_targets: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    selection, check = [], []
    for label in np.unique(hard_targets):
        ordered = sorted(np.flatnonzero(hard_targets == label), key=lambda i: (
            digest(f"{seed}:{groups[i]}".encode()), int(groups[i]),
        ))
        selection.extend(ordered[::2])
        check.extend(ordered[1::2])
    return np.array(selection, dtype=int), np.array(check, dtype=int)


def _curve_log_probabilities(logits, counts, offsets, temperature):
    logs = _window_log_probabilities(logits, temperature)
    log_weights = np.log(counts.astype(np.float64))
    summed = np.logaddexp.reduceat(logs + log_weights[:, None], offsets[:-1], axis=0)
    return summed - np.logaddexp.reduceat(log_weights, offsets[:-1])[:, None]


def _curve_probabilities(logits, counts, offsets, temperature):
    return np.clip(np.exp(_curve_log_probabilities(logits, counts, offsets, temperature)), 0, 1)


def _fit_temperature(logits, counts, offsets, targets, selection) -> float:
    def objective(log_temperature):
        logs = _curve_log_probabilities(logits, counts, offsets, float(np.exp(log_temperature)))
        return float(-np.sum(targets[selection] * logs[selection]) / len(selection))

    bounds = (float(np.log(.05)), float(np.log(20.)))
    optimum = minimize_scalar(objective, bounds=bounds, method="bounded", options={"xatol": 1e-8})
    candidates = [0., *bounds]
    if optimum.success and np.isfinite(optimum.x) and np.isfinite(optimum.fun):
        candidates.append(float(optimum.x))
    scored = [(objective(value), value) for value in candidates]
    loss, selected = min(scored, key=lambda row: (row[0], abs(np.exp(row[1]) - 1), row[1]))
    if not np.isfinite(loss):
        raise ValueError("no finite temperature objective")
    return float(np.exp(selected))


def _select_threshold(nrp: np.ndarray, truth: np.ndarray) -> tuple[float, list[dict]]:
    if not truth.any() or truth.all():
        raise ValueError("selection must contain both NRP and reliable curves")
    rows = []
    for cutoff in np.unique(np.r_[0., nrp, 1.]):
        predicted = nrp >= cutoff
        tp = int(np.sum(truth & predicted))
        tn = int(np.sum(~truth & ~predicted))
        fp = int(np.sum(~truth & predicted))
        fn = int(np.sum(truth & ~predicted))
        specificity = tn / (tn + fp)
        rows.append({"cutoff": float(cutoff), "balanced_accuracy": (tp / (tp + fn) + specificity) / 2,
                     "nrp_recall": tp / (tp + fn), "nrp_precision": tp / (tp + fp) if tp + fp else None,
                     "specificity": specificity, "periodic_coverage": specificity,
                     "false_nrp": fp, "nrp_missed": fn})
    best = max(rows, key=lambda row: (row["balanced_accuracy"], row["specificity"], row["cutoff"]))
    return best["cutoff"], rows


def _metrics(probabilities: np.ndarray, hard: np.ndarray) -> dict:
    if not len(hard):
        raise ValueError("empty calibration check partition")
    top = probabilities.argmax(axis=1)
    confidence = probabilities[np.arange(len(hard)), top]
    buckets = np.minimum((confidence * 10).astype(int), 9)
    ece = 0.
    for bucket in range(10):
        mask = buckets == bucket
        if mask.any():
            ece += mask.mean() * abs((top[mask] == hard[mask]).mean() - confidence[mask].mean())
    one_hot = np.eye(129)[hard]
    return {"n": len(hard), "ece": float(ece),
            "brier": float(np.mean(np.sum((probabilities - one_hot)**2, axis=1)))}


def _period_metrics(probs, periods, bounds, cutoff=None):
    if cutoff is None:
        bins = probs.argmax(axis=1)
    else:
        bins = [128 if row[128] >= cutoff else next(iter(_period_peaks(row[:128], 1)), 128)
                for row in probs]
    return score(periods, np.r_[centers(bounds), 0.][bins], bounds)


def calibrate(bundle_path: Path, output: Path) -> dict:
    bundle, splits, identity = verify_bundle(bundle_path)
    local = load_config()
    cfg = {**local.cnn, "period_min_s": bundle["bounds"][0], "period_max_s": bundle["bounds"][1]}
    if list(resolve_period_bounds(local.period_bounds)) != bundle["bounds"]:
        raise ValueError("bundle period bounds differ from current config")
    if any(cfg[key] != val for key, val in bundle["cnn"].items()):
        raise ValueError("bundle CNN settings differ from current config")
    model, checksum = _load_model(_CHECKPOINT, cfg, expected_dataset_id=identity)
    val = splits["val"]
    windows, offsets = val["windows"], val["offsets"]
    with torch.inference_mode():
        logits = np.concatenate([model(batch).numpy()
                                 for batch in torch.from_numpy(windows).split(cfg["batch_size"])])
    weights = windows[:, 1, :].sum(axis=1)
    periods = val["periods"]
    hard = np.array([encode_target(float(p) if p else None, cfg)[0] for p in periods])
    selection, check = _split_groups(val["groups"], hard, cfg["seed"])
    if not len(check):
        raise ValueError("empty calibration check partition")
    temp = _fit_temperature(logits, weights, offsets, val["targets"], selection)
    identity_probs = _curve_probabilities(logits, weights, offsets, 1.)
    calibrated_probs = _curve_probabilities(logits, weights, offsets, temp)
    cutoff, table = _select_threshold(calibrated_probs[selection, 128], hard[selection] == 128)
    report = {
        "checkpoint_sha256": checksum, "dataset_id": identity, "experimental": True,
        "bounds": bundle["bounds"], "encoder": {key: cfg[key] for key in FEATURE_KEYS},
        "temperature": temp, "temperature_boundary": temp in (float(np.exp(np.log(.05))),
                                                                  float(np.exp(np.log(20.)))),
        "abstain_threshold": cutoff,
        "split_rule": "Within each hard bin, alternate SHA256(seed:group_id)-ordered groups, selection first",
        "seed": cfg["seed"], "window_count": len(windows), "curve_count": len(periods),
        "selection_groups": val["groups"][selection].tolist(), "check_groups": val["groups"][check].tolist(),
        "category_counts": {
            "selection": np.bincount(hard[selection], minlength=129).tolist(),
            "check": np.bincount(hard[check], minlength=129).tolist(),
        },
        "metrics": {"ece": "ten equal-width top-label confidence bins, rightmost endpoint included",
                    "brier": "mean summed squared distance to one-hot hard truth"},
        "selection_operating_points": table,
        "before": {**_metrics(identity_probs[check], hard[check]),
                   "period_metrics": _period_metrics(identity_probs[check], periods[check], bundle["bounds"])},
        "after": {**_metrics(calibrated_probs[check], hard[check]),
                  "period_metrics": _period_metrics(calibrated_probs[check], periods[check],
                                                    bundle["bounds"], cutoff)},
        "torch_version": str(torch.__version__), "threads": torch.get_num_threads(),
    }
    report["check_ece_nonincreasing"] = report["after"]["ece"] <= report["before"]["ece"]
    save_json(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=_ROOT / "data/synthetic/cp2_colab.zip")
    parser.add_argument("--output", type=Path, default=_CALIBRATION)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    report = calibrate(args.bundle, args.output)
    print(f"Temperature {report['temperature']:.12g}, abstain_threshold {report['abstain_threshold']!r}, "
          f"check ECE {report['before']['ece']:.6f} -> {report['after']['ece']:.6f}")
