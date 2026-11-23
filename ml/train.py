"""CP2 Colab trainer and local CPU checkpoint acceptance."""

import argparse
import json
import os
import platform
import random
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch
from torch import nn

from ml.bundle import digest, json_bytes, verify_bundle
from ml.model import PeriodCNN


def centers(bounds):
    low, high = bounds
    return low * np.exp((np.arange(128) + .5) * np.log(high/low)/128)


def score(truth, prediction, bounds):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if (truth.shape != prediction.shape or not np.isfinite(prediction).all()
            or np.any(prediction < 0)):
        raise ValueError("invalid validation predictions")
    periodic, emitted = truth > 0, prediction > 0
    selected = periodic & emitted
    ratios = np.log(prediction[selected] / truth[selected])
    width = np.log(bounds[1] / bounds[0]) / 128
    # Include the mathematical boundary despite rounding the period ratio.
    slack = 4 * np.finfo(float).eps
    correct = abs(ratios) <= width + slack
    half = (~correct) & (abs(ratios + np.log(2.)) <= width + slack)
    double = (~correct) & (abs(ratios - np.log(2.)) <= width + slack)
    hits = int(np.sum(~periodic & ~emitted))
    false_nrp = int(np.sum(periodic & ~emitted))
    missed = int(np.sum(~periodic & emitted))

    def fraction(count, population):
        return float(count / population) if population else None

    n = int(periodic.sum())
    return {"curves": len(truth), "periodic": n, "scored": len(ratios),
            "median_abs_log_error": float(np.median(abs(ratios))) if len(ratios) else None,
            "tolerance_accuracy": fraction(correct.sum(), n), "coverage": fraction(len(ratios), n),
            "harmonic_error_rate": fraction(half.sum()+double.sum(), n),
            "half_period_count": int(half.sum()), "double_period_count": int(double.sum()),
            "nrp_tp": hits, "nrp_fp": false_nrp, "nrp_fn": missed, "nrp_tn": len(ratios),
            "nrp_recall": fraction(hits, hits+missed), "nrp_precision": fraction(hits, hits+false_nrp),
            "false_nrp_rate": fraction(false_nrp, n)}


def baseline(train, val, bounds):
    true = train["periods"]
    center = centers(bounds)
    median = np.median(np.log(true[true > 0]))
    distances = np.abs(np.log(center) - median)
    index = int(np.flatnonzero(distances <= distances.min() + 4*np.finfo(float).eps)[0])
    return {"period_s": float(center[index]), **score(val["periods"],
            np.full(len(val["periods"]), center[index]), bounds)}


@torch.no_grad()
def evaluate(model, data, cfg, device):
    model.eval()
    x = torch.from_numpy(data["windows"])
    prob = np.concatenate([model(batch.to(device)).softmax(1).cpu().numpy()
                           for batch in x.split(cfg["cnn"]["batch_size"])])
    weights = data["windows"][:,1].sum(axis=1)
    combined = np.add.reduceat(prob * weights[:,None], data["offsets"][:-1], axis=0)
    prediction = np.r_[centers(cfg["bounds"]), 0.][combined.argmax(axis=1)]
    return score(data["periods"], prediction, cfg["bounds"])


def atomic_write(path, writer):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as f:
        tmp = Path(f.name)
    try:
        writer(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def save_json(path, value):
    atomic_write(path, lambda p: p.write_bytes(json_bytes(value)))


def save_checkpoint(path, state):
    atomic_write(path, lambda p: torch.save(state, p))
    checksum = digest(path.read_bytes())
    atomic_write(path.with_suffix(".sha256"), lambda p: p.write_text(checksum, encoding="ascii"))
    return checksum


def read_checkpoint(path, cfg, identity):
    checksum = digest(path.read_bytes())
    if checksum != path.with_suffix(".sha256").read_text(encoding="ascii").strip():
        raise ValueError("checkpoint checksum mismatch")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["config"] != cfg or state["dataset_id"] != identity:
        raise ValueError("checkpoint incompatible with configuration or dataset")
    if (state["epoch"] != len(state["history"]) or not 1 <= state["epoch"] <= cfg["cnn"]["epochs"]
            or not state["launch"]["optimizer_updated"]):
        raise ValueError("invalid checkpoint training history")
    return state, checksum


def train_colab(bundle, output):
    import google.colab  # Only run the optimizer in the requested environment.

    cfg, data, identity = verify_bundle(bundle)
    if not torch.cuda.is_available():
        raise RuntimeError("Colab CUDA GPU unavailable. Stop and select a free GPU runtime.")
    drive = Path("/content/drive")
    if not os.path.ismount(drive) or not output.resolve().is_relative_to(drive / "MyDrive"):
        raise ValueError("Explicitly mount Drive and select a persistent MyDrive output directory")
    device = torch.device("cuda", torch.cuda.current_device())
    environment = {"gpu": torch.cuda.get_device_name(device), "device": str(device),
                   "pytorch": str(torch.__version__), "cuda": torch.version.cuda,
                   "cudnn": torch.backends.cudnn.version(), "python": platform.python_version(),
                   "numpy": np.__version__}
    print(environment, flush=True)
    cnn = cfg["cnn"]
    seed = int(np.random.SeedSequence([cnn["seed"], 3]).generate_state(1)[0])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    shuffle = torch.Generator().manual_seed(seed)
    model = PeriodCNN(cnn).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cnn["learning_rate"])
    train = data["train"]
    x = torch.from_numpy(train["windows"])
    y = torch.from_numpy(np.repeat(train["targets"], np.diff(train["offsets"]), axis=0))
    history, launch = [], None
    output.mkdir(parents=True, exist_ok=True)
    completed = sorted(output.glob("epoch-*.sha256"))
    if completed:
        latest = completed[-1].with_suffix(".pt")
        state, _ = read_checkpoint(latest, cfg, identity)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state(state["gpu_rng"], device)
        shuffle.set_state(state["shuffle_rng"])
        random.setstate(state["python_rng"])
        numpy_rng = state["numpy_rng"]
        np.random.set_state((numpy_rng[0], np.asarray(numpy_rng[1], dtype=np.uint32), *numpy_rng[2:]))
        history, launch = state["history"], state["launch"]
        print(f"Resuming after {latest.name}", flush=True)
    compare = baseline(train, data["val"], cfg["bounds"])

    for epoch in range(len(history)+1, cnn["epochs"]+1):
        model.train()
        total_loss = 0.
        for indices in torch.randperm(len(x), generator=shuffle).split(cnn["batch_size"]):
            batch, target = x[indices].to(device), y[indices].to(device)
            if batch.device != device or target.device != device or next(model.parameters()).device != device:
                raise RuntimeError("GPU device preflight failed")
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = nn.functional.cross_entropy(logits, target)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite training loss")
            loss.backward()
            if launch is None:
                before = model.head.weight.detach().clone()
                if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise RuntimeError("GPU backward preflight failed")
            optimizer.step()
            if launch is None:
                torch.cuda.synchronize(device)
                if torch.equal(before, model.head.weight):
                    raise RuntimeError("first optimizer step did not update parameters")
                groups = train["groups"][np.searchsorted(train["offsets"][1:], indices.numpy(), side="right")]
                launch = {**environment, "dataset_id": identity, "split": "train",
                          "train_curves": len(train["groups"]), "batch_size": len(batch),
                          "curve_groups_sha256": digest(groups.tobytes()), "loss": loss.item(),
                          "forward_finite": True, "backward_finite": True, "optimizer_updated": True}
                save_json(output/"gpu_launch.json", launch)
            total_loss += loss.item() * len(indices)
        history.append({"epoch": epoch, "train_loss": total_loss / len(x),
                        "validation": evaluate(model, data["val"], cfg, device)})
        numpy_rng = np.random.get_state()
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "epoch": epoch, "history": history, "config": cfg, "dataset_id": identity,
                 "launch": launch, "environment": environment, "shuffle_rng": shuffle.get_state(),
                 "cpu_rng": torch.get_rng_state(), "gpu_rng": torch.cuda.get_rng_state(device),
                 "python_rng": random.getstate(),
                 "numpy_rng": (numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:])}
        path = output/f"epoch-{epoch:04d}.pt"
        checksum = save_checkpoint(path, state)
        save_json(output/"latest.json", {"epoch": epoch, "file": path.name, "sha256": checksum})
        save_json(output/"cnn_training_log.json", {"launch": launch, "history": history,
                  "baseline": compare, "dataset_id": identity, "counts": cfg["counts"]})
        print(history[-1], path, flush=True)
    # Rebuild final artifacts even if disconnection occurred after the last epoch commit.
    last = output/f"epoch-{len(history):04d}.pt"
    state, checksum = read_checkpoint(last, cfg, identity)
    final = output/"cnn_v1.pt"
    atomic_write(final, lambda p: p.write_bytes(last.read_bytes()))
    atomic_write(final.with_suffix(".sha256"), lambda p: p.write_text(checksum, encoding="ascii"))
    save_json(output/"latest.json", {"epoch": len(history), "file": last.name, "sha256": checksum})
    save_json(output/"cnn_training_log.json", {"launch": launch, "history": history,
              "baseline": compare, "dataset_id": identity, "counts": cfg["counts"],
              "final_sha256": checksum, "experimental": True})
    print(f"Final checkpoint: {final}, SHA256 {checksum}", flush=True)


def accept_download(bundle, checkpoint, report):
    from core.config import load_config, resolve_period_bounds
    from core.schemas import Observation
    from ml.infer import prepared_logits
    from preprocessing.prepare import prepare

    cfg, data, identity = verify_bundle(bundle)
    local = load_config()
    if (cfg["bounds"] != list(resolve_period_bounds(local.period_bounds))
            or any(local.cnn.get(k) != v for k,v in cfg["cnn"].items())):
        raise ValueError("checkpoint bundle differs from local model settings")
    state, checksum = read_checkpoint(checkpoint, cfg, identity)
    model = PeriodCNN(cfg["cnn"]).cpu().eval()
    model.load_state_dict(state["model"], strict=True)
    times = np.arange(128, dtype=float)
    brightness = np.cos(2*np.pi*times/12)
    obs = Observation("cpu-check", times, brightness, None, "unknown", {})
    logits = prepared_logits(model, prepare(obs, local.preprocess), cfg["cnn"])
    if logits.shape != (1,129) or not torch.isfinite(logits).all():
        raise ValueError("CPU checkpoint inference failed")
    prior = json.loads(report.read_bytes()) if report.exists() else {}
    if prior.get("dataset_id", identity) != identity:
        raise ValueError("existing training report belongs to a different bundle")
    save_json(report, {**prior, "status": "downloaded_checkpoint_cpu_validated",
              "dataset_id": identity, "config": cfg, "launch": state["launch"],
              "history": state["history"], "baseline": baseline(data["train"], data["val"], cfg["bounds"]),
              "loss_decreased_first_to_last": (state["history"][-1]["train_loss"] <
                                               state["history"][0]["train_loss"]),
              "experimental": True, "cpu_acceptance": {"sha256": checksum, "epoch": state["epoch"],
              "input_sha256": digest(times.tobytes()+brightness.tobytes()), "shape": [1,129],
              "period_classes": 128, "nrp_index": 128, "finite": True, "strict_load": True}})
    print(f"CPU acceptance passed: {checksum}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    gpu = sub.add_parser("colab")
    gpu.add_argument("bundle", type=Path)
    gpu.add_argument("output", type=Path)
    cpu = sub.add_parser("accept")
    cpu.add_argument("bundle", type=Path)
    cpu.add_argument("checkpoint", type=Path)
    cpu.add_argument("--report", type=Path, default=Path("docs/results/cnn_training_log.json"))
    args = parser.parse_args()
    if args.command == "colab":
        train_colab(args.bundle, args.output)
    else:
        accept_download(args.bundle, args.checkpoint, args.report)
