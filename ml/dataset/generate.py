"""Generate the raw Day 1 pilot: python -m ml.dataset.generate OUTPUT_DIRECTORY."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict
from importlib.metadata import version
import hashlib
import json
import platform
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from core.config import PipelineConfig, config_hash, load_config, resolve_period_bounds
from ml.dataset.labels import (
    encode_target, flat_signal, mixture_has_repeat, observation_label, template_repeat,
)
from ml.dataset.sample_params import SPLITS, SOURCES, sample_params, validate_design
from physics.simulator import simulate_lightcurve


def simulation_config(config):
    bounds = resolve_period_bounds(config.period_bounds)
    return {**config.simulation, "period_min_s": bounds.min_period_s,
            "period_max_s": bounds.max_period_s}


def draw_curve(cfg, split, source, seed):
    """Return an accepted record or the reason this seeded draw was rejected."""
    return _draw_curve(cfg, split, source, seed)


def _draw_curve(cfg, split, source, seed, observed_times=None, source_rng=None, subtype=None):
    """Validate dense clean truth and optionally evaluate it at a second set of times."""
    if source not in SOURCES:
        raise ValueError(f"unknown pilot source: {source}")
    rng = np.random.default_rng(seed)
    lower_span, upper_span = cfg["baseline_range_s"]
    if cfg.get("condition_baseline_on_coverage", False) and source != "under_covered":
        lower_span = max(lower_span, max(3., cfg["min_cycles"]) * cfg["period_max_s"])
        if lower_span > upper_span:
            return "insufficient coverage"
    span = rng.uniform(lower_span, upper_span)
    times = np.linspace(0., span, int(np.ceil(span / cfg["cadence_s"])) + 1)
    n = cfg["samples_per_cycle"]
    tol, flat = cfg["repeat_tolerance"], cfg["flat_rtol"]
    parents = []
    parent_repeat = None
    if source == "noise_or_constant":
        kind = subtype or ("noise" if rng.integers(2) else "constant")
        grid = times
        if observed_times is not None:
            grid, inverse = np.unique(np.r_[times, observed_times], return_inverse=True)
        generator = source_rng if source_rng is not None else rng
        values = generator.normal(size=grid.size) if kind == "noise" else np.ones(grid.size)
        if observed_times is None:
            y = values
        else:
            y = values[inverse[:times.size]]
            sampled = values[inverse[times.size:]]
    else:
        kind = source
        parameter_cfg = cfg
        if source == "multi_frequency":
            parameter_cfg = dict(cfg, phase_cycle_fraction_range=cfg.get("mixture_cycle_fraction_range", [0., 1.]))
        params, index = sample_params(parameter_cfg, split, rng)
        parents.append({"split": split, "variant": index, "params": asdict(params)})
        truth_t = np.arange(3*n) / n * params.phase_cycle_s
        truth = simulate_lightcurve(params, truth_t)
        y = simulate_lightcurve(params, times)
        parent_repeat = template_repeat(
            truth, params.phase_cycle_s, n, tol, flat,
            wave=lambda t: simulate_lightcurve(params, t),
        )
        if source == "multi_frequency":
            second, index = sample_params(parameter_cfg, split, rng)
            parents.append({"split": split, "variant": index, "params": asdict(second)})
            other = simulate_lightcurve(second, np.arange(3*n)/n * second.phase_cycle_s)
            scales = (float(np.ptp(truth)), float(np.ptp(other)))
            for parent, amplitude in zip(parents, scales):
                if amplitude <= 0 or not np.isfinite(amplitude) or not np.isfinite(1.0 / amplitude):
                    return "unrepresentable mixture weight"
                parent["flux_weight"] = 1.0 / amplitude
            y = y/scales[0] + simulate_lightcurve(second, times)/scales[1]
    if observed_times is not None:
        if observed_times[0] < times[0] or observed_times[-1] > times[-1]:
            return "insufficient clean source support"
        if parents:
            sampled = simulate_lightcurve(params, observed_times)
            if source == "multi_frequency":
                sampled = sampled/scales[0] + simulate_lightcurve(second, observed_times)/scales[1]
    if source == "under_covered":
        if parent_repeat is None or not cfg["period_min_s"] <= parent_repeat <= cfg["period_max_s"]:
            return "under-covered parent has no in-range clean repeat"
        if span / parent_repeat >= cfg["min_cycles"]:
            return "under-covered quota has sufficient cycles"
        if len(times) < 4 or flat_signal(y, flat):
            return "under-covered observation has insufficient sampling or modulation"
        apparent = None
        disposition = "nrp"
    else:
        disposition, apparent = observation_label(times, y, cfg)
    if disposition not in ("resolved", "nrp"):
        return disposition
    if source == "multi_frequency" and apparent is not None:
        return "combined waveform repeats"
    if source == "multi_frequency":
        def clean_mixture(t):
            return (simulate_lightcurve(params, t) / scales[0]
                    + simulate_lightcurve(second, t) / scales[1])

        clean_step = min(params.phase_cycle_s, second.phase_cycle_s) / n
        if mixture_has_repeat(clean_mixture, times, cfg["period_min_s"],
                              cfg["period_max_s"], clean_step,
                              cfg["nrp_min_residual"], flat):
            return "clean mixture has a compatible repeat"
    if (source == "reliable") != (disposition == "resolved"):
        return "observation incompatible with quota"
    if source == "reliable":
        if parent_repeat is None or not cfg["period_min_s"] <= parent_repeat <= cfg["period_max_s"]:
            return "clean repeat outside configured bounds or unresolved"
        if abs(apparent - parent_repeat) > (times[1] - times[0]) / 2:
            return "sampled repeat disagrees with clean repeat"
        apparent = parent_repeat
    label, soft = encode_target(apparent, cfg)
    meta = {"source": kind, "parents": parents, "seed": seed, "split": split,
            "apparent_period_s": apparent, "parent_repeat_s": parent_repeat}
    json.dumps(meta, allow_nan=False)
    if observed_times is not None:
        return observed_times, sampled, label, soft, meta
    return times, y, label, soft, meta


_ROOT = Path(__file__).resolve().parents[2]
_SOURCES = ("core/config.py", "core/schemas.py", "physics/geometry.py", "physics/params.py",
            "physics/simulator.py", "ml/dataset/generate.py", "ml/dataset/labels.py",
            "ml/dataset/sample_params.py")


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_identity(root):
    files = {name: _digest(root/name) for name in sorted(_SOURCES)}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"sha256": digest, "files": files}


def _check_parent(parent, split, cfg):
    variants = cfg["variants"][split]
    index = parent["variant"]
    if parent["split"] != split or not 0 <= index < len(variants):
        raise ValueError("parent outside assigned split")
    variant = variants[index]
    p = parent["params"]
    aspect = p["dimensions"][0 if p["shape"] == "box" else 1]
    if (p["shape"] != variant["shape"] or not variant["aspect"][0] <= aspect <= variant["aspect"][1]
            or not variant["polar_rad"][0] <= p["axis_polar_rad"] <= variant["polar_rad"][1]):
        raise ValueError("parent outside declared variant")


def _verify_publication(staging, manifest):
    stored = json.loads((staging/"run_config.json").read_text(encoding="utf-8"))
    if stored != json.loads(json.dumps(manifest, allow_nan=False)):
        raise ValueError("manifest readback failed")
    for name, expected in manifest["artifacts"].items():
        path = staging/name
        if name.endswith(".npz"):
            split = path.stem.removeprefix("pilot_")
            with np.load(path, allow_pickle=False) as shard:
                rows = [json.loads(raw) for raw in shard["metadata"]]
                labels, soft = shard["label_bin"], shard["soft_target"]
                offsets, times, brightness = shard["offsets"], shard["t_s"], shard["brightness"]
                if (Counter(r["source"] for r in rows) != manifest["counts"][split]
                        or any(r["split"] != split for r in rows)
                        or offsets.shape != (len(rows)+1,) or offsets[0] != 0
                        or np.any(np.diff(offsets) < 2) or offsets[-1] != len(times)
                        or times.shape != brightness.shape or not np.isfinite(times).all()
                        or not np.isfinite(brightness).all() or labels.shape != (len(rows),)
                        or soft.shape != (len(rows),129)):
                    raise ValueError(f"inconsistent staged shard: {name}")
                bounds = manifest["config"]["period_bounds"]
                cfg = {"period_min_s": bounds["min_period_s"], "period_max_s": bounds["max_period_s"]}
                if not np.array_equal(shard["period_bounds"], list(cfg.values())):
                    raise ValueError(f"inconsistent period bounds: {name}")
                quotas = manifest["config"]["simulation"]["quotas"][split]
                groups = Counter("noise_or_constant" if r["source"] in ("noise","constant")
                                 else r["source"] for r in rows)
                if any(groups[source] != count for source,count in quotas.items()):
                    raise ValueError(f"inconsistent source quotas: {name}")
                for i, row in enumerate(rows):
                    index, target = encode_target(row["apparent_period_s"],cfg)
                    if index != labels[i] or not np.array_equal(target,soft[i]):
                        raise ValueError(f"inconsistent target: {name}")
        if path.stat().st_size == 0 or _digest(path) != expected:
            raise ValueError(f"invalid staged artifact: {name}")


def generate_pilot(config: PipelineConfig, destination, *, revision=None):
    return _publish(config, destination, revision, plots=False)


def _accepted_record(job):
    cfg, split, source, job_seed = job
    rng = np.random.default_rng(job_seed)
    rejections = Counter()
    for _ in range(cfg["max_attempts"]):
        seed = int(rng.integers(2**63 - 1))
        record = draw_curve(cfg, split, source, seed)
        if not isinstance(record, str):
            return record, rejections
        rejections[(split, source, record)] += 1
    raise ValueError(f"exhausted quota {split}/{source}: {dict(rejections)}")


def _publish(config, destination, revision, *, plots):
    started = perf_counter()
    cfg = simulation_config(config)
    validate_design(cfg)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    def git(*args):
        return subprocess.check_output(["git","--no-optional-locks",*args], cwd=_ROOT,text=True).strip()

    base = git("rev-parse","HEAD")
    dirty = bool(git("status","--porcelain","--untracked-files=all"))
    identity = _source_identity(_ROOT)
    counts = {}
    rng = np.random.default_rng(cfg["seed"])
    rejections = Counter()
    destination.parent.mkdir(parents=True, exist_ok=True)
    workers = cfg.get("workers", 1)
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else nullcontext(None)
    with executor as pool, TemporaryDirectory(dir=destination.parent) as temp:
        staging = Path(temp) / "pilot"
        staging.mkdir()
        for split in SPLITS:
            jobs = [(cfg, split, source, int(rng.integers(2**63 - 1)))
                    for source in SOURCES for _ in range(cfg["quotas"][split][source])]
            generated = pool.map(_accepted_record, jobs) if pool is not None else map(_accepted_record, jobs)
            rows = []
            for record, rejected in generated:
                rows.append(record)
                rejections.update(rejected)
            rng.shuffle(rows)
            times, flux, labels, soft, metadata = zip(*rows)
            counts[split] = dict(Counter(m["source"] for m in metadata))
            for m in metadata:
                for parent in m["parents"]:
                    _check_parent(parent, split, cfg)
            represented = {parent["variant"] for meta in metadata for parent in meta["parents"]}
            if represented != set(range(len(cfg["variants"][split]))):
                raise ValueError(f"missing simulator variants in {split}: represented {represented}")
            path = staging / f"pilot_{split}.npz"
            offsets = np.cumsum([0] + [len(t) for t in times])
            np.savez_compressed(path, t_s=np.concatenate(times), brightness=np.concatenate(flux),
                                offsets=offsets, label_bin=labels, soft_target=np.stack(soft),
                                metadata=[json.dumps(m, allow_nan=False) for m in metadata],
                                period_bounds=[cfg["period_min_s"], cfg["period_max_s"]])
            with np.load(path, allow_pickle=False) as shard:
                if not (shard["offsets"][-1] == len(shard["t_s"]) == len(shard["brightness"])
                        and np.isfinite(shard["brightness"]).all()
                        and np.allclose(shard["soft_target"].sum(axis=1), 1)):
                    raise ValueError(f"invalid generated shard {split}")
        names = [f"pilot_{s}.npz" for s in SPLITS]
        if plots:
            plot_pilot(staging)
            names += ["label_histogram.png", "inspection.png"]
        manifest = {"config": asdict(config), "config_hash": config_hash(config),
                    "commit": base, "revision_annotation": revision, "dirty": dirty,
                    "source_identity": identity, "python": platform.python_version(),
                    "label_policy": "clean-repeat-with-observation-check-v2", "counts": counts,
                    "sampling_profile_usage": "none: clean CP1 pilot; degradation is CP2",
                    "generation_and_plot_seconds": perf_counter() - started,
                    "artifacts": {name: _digest(staging/name) for name in names}, "seed": cfg["seed"],
                    "package_versions": {p: version(p) for p in ("numpy", "scipy", "matplotlib", "PyYAML")},
                    "rejections": [{"split": s, "source": k, "reason": r, "count": v}
                                   for (s, k, r), v in sorted(rejections.items())]}
        (staging / "run_config.json").write_text(json.dumps(manifest, indent=2, allow_nan=False),
                                                 encoding="utf-8")
        _verify_publication(staging, manifest)
        if _source_identity(_ROOT) != identity:
            raise ValueError("participating source changed during generation")
        staging.rename(destination)
    return destination


def plot_pilot(directory):
    """Prepare the label histogram and twenty development examples for human inspection."""
    from matplotlib.figure import Figure

    directory = Path(directory)
    counts = np.zeros(129, dtype=int)
    examples = {}
    for split in SPLITS:
        with np.load(directory / f"pilot_{split}.npz", allow_pickle=False) as shard:
            counts += np.bincount(shard["label_bin"], minlength=129)
            if split == "test":
                continue
            offsets, times, y = shard["offsets"], shard["t_s"], shard["brightness"]
            for i, raw in enumerate(shard["metadata"]):
                meta = json.loads(raw)
                group = examples.setdefault(meta["source"], [])
                if len(group) < 20:
                    segment = slice(offsets[i], offsets[i+1])
                    group.append((times[segment], y[segment], split, i, meta["source"]))
    histogram = Figure(figsize=(9, 4), layout="constrained")
    ax = histogram.subplots()
    ax.bar(np.arange(128), counts[:128], width=1)
    ax.bar(128, counts[128], label="NRP")
    ax.set(xlabel="Apparent-period bin (NRP at 128)", ylabel="Curves")
    ax.legend()
    histogram.savefig(directory / "label_histogram.png")
    figure = Figure(figsize=(14, 12), layout="constrained")
    ordered = [group[i] for i in range(20) for group in examples.values() if i < len(group)]
    for ax, (t, y, split, index, kind) in zip(figure.subplots(5, 4).flat, ordered[:20]):
        ax.plot(t, y)
        ax.set_title(f"{split}[{index}] {kind}", fontsize=8)
    figure.supxlabel("Observation time (s)")
    figure.supylabel("Relative brightness")
    figure.savefig(directory / "inspection.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    configuration = load_config()
    # The command is the official 1,000-curve pilot. Small runs use injected configs.
    validate_design(simulation_config(configuration))
    if sum(sum(q.values()) for q in configuration.simulation["quotas"].values()) != 1000:
        parser.error("Day 1 requires exactly 1,000 curves")
    _publish(configuration, args.output, None, plots=True)
