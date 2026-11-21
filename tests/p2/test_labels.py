from dataclasses import replace
import json

import numpy as np
import pytest

from core.config import load_config
from ml.dataset.generate import draw_curve, generate_pilot, plot_pilot, simulation_config
from ml.dataset.labels import encode_target, mixture_has_repeat, template_repeat
from ml.dataset.sample_params import SOURCES, SPLITS, validate_design
from physics.params import SimParams
from physics.simulator import simulate_lightcurve


@pytest.fixture
def design():
    config = load_config()
    variants = {
        "train": [dict(shape="box", aspect=[1.2, 1.5], polar_rad=[0, 0], surface="uniform")],
        "val": [dict(shape="box", aspect=[2, 2.5], polar_rad=[.3, .4], surface="unequal")],
        "test": [dict(shape="cylinder", aspect=[3, 3.5], polar_rad=[.8, 1], surface="cap_contrast")],
    }
    simulation = dict(config.simulation, variants=variants, baseline_range_s=[1, 28],
                      cadence_s=.08, flat_rtol=1e-10, repeat_tolerance=2e-5, nrp_min_residual=.08, sampling_tolerance=.08, min_cycles=2,
                      samples_per_cycle=64, max_attempts=250, seed=17,
                      quotas={s: dict(reliable=4, noise_or_constant=2, multi_frequency=2, under_covered=0) for s in SPLITS})
    return replace(config, simulation=simulation,
                   period_bounds={"status": "SET", "min_period_s": 1., "max_period_s": 8.})


@pytest.mark.parametrize("n", [256, 512])
def test_repeat_controls(n):
    params = SimParams("box", (2, 3, 4), (1.,)*6, 0, 0, 0, 8, (1, 0, 0), (1, 0, 0))
    t = np.arange(3*n) * 8/n
    for rho, expected in [((1.,)*6, 4.), ((1., .999, 1, 1, 1, 1), 8.)]:
        y = simulate_lightcurve(replace(params, reflectivity=rho), t)
        assert template_repeat(y, 8., n, 1e-6, 1e-12) == expected
    assert template_repeat(np.ones(3*n), 8., n, 1e-6, 1e-12) is None


def test_mixture_repeat_even_between_lag_grid_points():
    t = np.linspace(0, 38, 241)
    base = 3.173
    def related(x):
        return np.cos(2*np.pi*x/base) + .6*np.cos(4*np.pi*x/base)
    def unrelated(x):
        return np.cos(2*np.pi*x/base) + .6*np.cos(2*np.pi*x/(base*np.sqrt(2)))
    assert mixture_has_repeat(related, t, 1, 7, .13, 1e-5, 1e-10)
    assert not mixture_has_repeat(unrelated, t, 1, 7, .13, 1e-5, 1e-10)


def test_bins(design):
    cfg = simulation_config(design)
    edges = np.geomspace(1., 8., 129)
    for i, edge in enumerate(edges):
        index, soft = encode_target(edge, cfg)
        assert index == min(i, 127)
        assert np.isfinite(soft).all() and (soft >= 0).all()
        assert soft.sum() == pytest.approx(1.) and soft[-1] == 0
        if 0 < i < 128:
            np.testing.assert_allclose(soft[i-1:i+1], [.5, .5])
    assert encode_target(None, cfg)[0] == 128
    assert encode_target(None, cfg)[1][-1] == 1
    for bad in (.9, 8.1, np.nan):
        with pytest.raises(ValueError):
            encode_target(bad, cfg)


def test_generation_and_round_trip(tmp_path, design):
    first = generate_pilot(design, tmp_path/"first", revision="fixture")
    second = generate_pilot(design, tmp_path/"second", revision="fixture")
    groups = {}
    for split in SPLITS:
        fname = f"pilot_{split}.npz"
        with np.load(first/fname, allow_pickle=False) as a, np.load(second/fname, allow_pickle=False) as b:
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])
            seen = []
            assert a["soft_target"].shape == (8, 129)
            assert np.all(np.diff(a["offsets"]) > 1)
            for i, raw in enumerate(a["metadata"]):
                row = json.loads(raw)
                kind = row["source"]
                seen.append("noise_or_constant" if kind in ("constant", "noise") else kind)
                assert row["split"] == split
                lo, hi = a["offsets"][i:i+2]
                t = a["t_s"][lo:hi]
                assert t[0] == 0 and np.all(np.diff(t) > 0)
                if kind == "multi_frequency":
                    assert len(row["parents"]) == 2
                for parent in row["parents"]:
                    assert parent["split"] == split
                    p = parent["params"]
                    assert 1 <= p["phase_cycle_s"] <= 8
                    groups.setdefault(split, set()).add((p["shape"], p["axis_polar_rad"]))
                    v = design.simulation["variants"][split][parent["variant"]]
                    assert v["aspect"][0] <= p["dimensions"][-1 if p["shape"] == "cylinder" else 0] <= v["aspect"][1]
                    assert v["polar_rad"][0] <= p["axis_polar_rad"] <= v["polar_rad"][1]
                if kind == "reliable":
                    assert a["label_bin"][i] < 128
                    assert t[-1] / row["apparent_period_s"] >= 2
                    if split == "train":
                        assert row["parent_repeat_s"] == row["parents"][0]["params"]["phase_cycle_s"]/2
                        assert row["apparent_period_s"] == pytest.approx(
                            row["parents"][0]["params"]["phase_cycle_s"]/2, rel=.003)
                else:
                    assert row["apparent_period_s"] is None and a["label_bin"][i] == 128
                    if kind == "under_covered":
                        assert t[-1] / row["parent_repeat_s"] < 2
            assert all(seen.count(s) == design.simulation["quotas"][split][s] for s in SOURCES)
    assert not groups["train"] & (groups["val"] | groups["test"])
    assert not groups["val"] & groups["test"]
    assert json.loads((first/"run_config.json").read_text())["config"]["period_bounds"] == design.period_bounds
    plot_pilot(first)
    assert (first/"label_histogram.png").stat().st_size > 0
    assert (first/"inspection.png").stat().st_size > 0
    with pytest.raises(FileExistsError):
        generate_pilot(design, first, revision="fixture")


def test_unset_invalid_and_impossible_configs(tmp_path, design):
    cfg = simulation_config(design)
    with pytest.raises(ValueError, match="not set"):
        simulation_config(replace(design, period_bounds={"status": "TBD"}))
    with pytest.raises(ValueError, match="cadence_s"):
        validate_design(dict(cfg, cadence_s=None))
    saved_variant = cfg["variants"]["val"][0]
    cfg["variants"]["val"][0] = cfg["variants"]["train"][0].copy()
    with pytest.raises(ValueError, match="overlapping"):
        validate_design(cfg)
    cfg["variants"]["val"][0] = saved_variant
    impossible = replace(design, simulation=dict(design.simulation, min_cycles=1e20, max_attempts=3))
    with pytest.raises(ValueError, match="exhausted quota train/reliable"):
        generate_pilot(impossible, tmp_path/"partial", revision="fixture")
    assert not (tmp_path/"partial").exists()


def test_changed_injected_bounds_reach_generation(tmp_path, design):
    quotas = {s: {kind: int(kind == "reliable") for kind in SOURCES} for s in SPLITS}
    updated = replace(design, period_bounds=dict(status="SET", min_period_s=10., max_period_s=24.),
                      simulation=dict(design.simulation, baseline_range_s=[100, 100], quotas=quotas))
    directory = generate_pilot(updated, tmp_path/"changed", revision="fixture")
    for split in SPLITS:
        with np.load(directory/f"pilot_{split}.npz", allow_pickle=False) as shard:
            meta = json.loads(shard["metadata"][0])
            assert 10 <= meta["parents"][0]["params"]["phase_cycle_s"] <= 24
            period = meta["apparent_period_s"]
            assert 10 <= period <= 24
            expected_bin = np.searchsorted(np.geomspace(10, 24, 129), period, side="right") - 1
            assert shard["label_bin"][0] == min(expected_bin, 127)


def test_generator_rejects_a_repeating_mixture(design, monkeypatch):
    cfg = dict(simulation_config(design), baseline_range_s=[40, 40])
    first = SimParams("box", (2, 3, 4), (1., .3, 1., .8, 1., 1.), 0, 0, 0, 3.,
                      (1, 0, 0), (1, 0, 0))
    parents = iter([(first, 0), (replace(first, phase_cycle_s=6.), 0)])
    # Control the two parameter draws. Simulate and check their actual combined flux.
    monkeypatch.setattr("ml.dataset.generate.sample_params", lambda *args: next(parents))
    assert draw_curve(cfg, "train", "multi_frequency", 1) == "combined waveform repeats"


def test_exact_coverage_boundary_and_noise_sources(design):
    cfg = simulation_config(design)
    cfg["baseline_range_s"] = [100, 100]
    cfg["min_cycles"] = 100 / cfg["period_max_s"]
    from ml.dataset.labels import observation_label
    times = np.linspace(0.,100.,1251)
    y = np.sin(2*np.pi*times/4)
    assert observation_label(times,y,cfg)[0] == "resolved"
    cfg["min_cycles"] = np.nextafter(cfg["min_cycles"], np.inf)
    assert observation_label(times,y,cfg)[0] == "insufficient coverage"
    assert draw_curve(cfg,"val","under_covered",4) == "insufficient coverage"
    cfg["min_cycles"] = 2
    draws = [draw_curve(cfg, "train", "noise_or_constant", seed) for seed in range(12)]
    assert {row[-1]["source"] for row in draws} == {"constant", "noise"}
    for row in draws:
        assert row[2] == 128
        assert (np.ptp(row[1]) == 0) == (row[-1]["source"] == "constant")


@pytest.mark.parametrize("mixed,cadence", [(False,.4),(True,.4),(True,.2)])
def test_visible_equivalence(design, monkeypatch, mixed, cadence):
    from ml.dataset import generate
    from ml.dataset.labels import observation_label
    cfg = dict(simulation_config(design), baseline_range_s=[24,24], cadence_s=cadence)
    box = SimParams("box",(1.,)*3,(1.,0.,0.,0.,0.,0.),0,0,0,4,(1,0,0),(1,0,0))
    narrow = replace(box, observer=(-np.cos(.2),np.sin(.2),0))
    if mixed:
        parents = [box, box, replace(narrow, phase_cycle_s=2.4)]
    else:
        first = replace(narrow,phase_rad=-.1,reflectivity=(.1,.1,.8,.8,.5,.5))
        parents = [first,replace(first,reflectivity=(.1,.7,.8,.8,.5,.5))]
    draws = iter(parents)
    monkeypatch.setattr(generate,"sample_params",lambda *args:(next(draws),0))
    captured = []
    def observe(times, values, policy):
        result = observation_label(times, values, policy)
        captured.append((times.tobytes(), values.tobytes(), result))
        return result
    monkeypatch.setattr(generate,"observation_label",observe)
    draw_curve(cfg,"train","reliable",21)
    draw_curve(cfg,"train","multi_frequency" if mixed else "reliable",21)
    assert len(captured) == 2 and captured[0] == captured[1]


@pytest.mark.parametrize("phase", [0.,.37])
@pytest.mark.parametrize("n,sides,expected", [(63,0,3),(64,0,3),(255,0,3),(256,0,3),
                                               (256,5,1.2),(300,5,1.2)])
def test_fractional_symmetries(design,n,sides,expected,phase):
    from ml.dataset.labels import observation_label
    p = SimParams("box",(2,3,4),(1.,)*6,0,0,phase,6,(1,0,0),(1,0,0))
    if sides:
        p = replace(p,shape="cylinder",dimensions=(1,2),cylinder_sides=sides,
                    reflectivity=(1.,)*(sides+2),observer=(.71,.6,.02))
    def wave(t):
        return simulate_lightcurve(p,t)
    truth = wave(np.arange(3*n)*6/n)
    assert template_repeat(truth,6.,n,2e-5,1e-10,wave=wave) == expected
    times = np.linspace(0,30,1702)
    status, period = observation_label(times,wave(times),simulation_config(design))
    assert status == "resolved"
    assert period == pytest.approx(expected,abs=1e-3)


def test_span_limits_and_common_period(design):
    from ml.dataset.sample_params import sample_params
    from ml.dataset.labels import observation_label
    cfg = simulation_config(design)
    for seed in range(16):
        rng = np.random.default_rng(seed)
        parents = [replace(sample_params(cfg,"val",rng)[0],phase_cycle_s=p) for p in (1.5,2.5)]
        for span in (16,22,24):
            t = np.linspace(0,span,round(span/.08)+1)
            curves = [simulate_lightcurve(p,t) for p in parents]
            y = sum(v/np.ptp(v) for v in curves)
            status,period = observation_label(t,y,cfg)
            if span < 24:
                assert status == "insufficient coverage"
            else:
                assert status in ("resolved","ambiguous candidates","insufficient sampling")
                if period is not None:
                    assert period == pytest.approx(7.5,abs=.01)


def test_tolerance_span_and_sampling_controls(design):
    from ml.dataset.labels import observation_label
    cfg = simulation_config(design)
    for span in (16,36,96):
        t = np.linspace(0,span,round(span/.08)+1)
        y = np.cos(2*np.pi*t/4) + .6*np.cos(2*np.pi*t/4.08)
        for tol in (.003,.03):
            status, period = observation_label(t,y,dict(cfg,repeat_tolerance=tol))
            if span == 16:
                assert status == "insufficient coverage"
            else:
                assert status in ("resolved","ambiguous candidates")
                if period is not None:
                    assert 4 <= period <= 4.08
    assert observation_label(t,t,cfg)[0] == "unresolved trend"
    assert observation_label(t,1+1e-13*y,cfg)[0] == "nrp"
    assert observation_label(t[::10],y[::10],cfg)[0] == "insufficient sampling"
    assert observation_label(np.delete(t,20),np.delete(y,20),cfg)[0] == "insufficient sampling"
    assert observation_label(t,y*np.nan,cfg)[0] == "numerical failure"


@pytest.mark.parametrize("touch",[False,True])
def test_every_test_range_reserved(design,touch):
    cfg = simulation_config(design)
    val = cfg["variants"]["val"][0]
    extra = dict(val,shape="cylinder")
    if touch:
        extra.update(aspect=[val["aspect"][1],3.],polar_rad=[1.1,1.2])
    cfg["variants"]["test"].append(extra)
    with pytest.raises(ValueError,match="exclusive"):
        validate_design(cfg)


def test_unrepresentable_weight(design,monkeypatch):
    box = SimParams("box",(1.,)*3,(1e-310,0.,0.,0.,0.,0.),0,0,0,4,(1,0,0),(1,0,0))
    parents = iter([(box,0),(replace(box,reflectivity=(1.,)*6),0)])
    monkeypatch.setattr("ml.dataset.generate.sample_params",lambda *args:next(parents))
    assert draw_curve(simulation_config(design),"val","multi_frequency",3) == "unrepresentable mixture weight"


@pytest.fixture
def publication_design(design):
    quotas = {s:{k:int(k=="reliable") for k in SOURCES} for s in SPLITS}
    return replace(design,simulation=dict(design.simulation,quotas=quotas,baseline_range_s=[30,30]))


@pytest.mark.parametrize("fault",["pilot_train.npz","pilot_val.npz","readback","label_histogram.png",
                                  "inspection.png","run_config.json","validation","rename","source"])
def test_complete_publication_failures(publication_design,tmp_path,monkeypatch,fault):
    from pathlib import Path
    from matplotlib.figure import Figure
    import ml.dataset.generate as g
    dest = tmp_path/"result"
    orphan = tmp_path/"previous-staging"
    orphan.mkdir()
    def fail(*args,**kwargs):
        raise OSError(fault)
    def intercept(original):
        def wrapper(path,*args,**kwargs):
            if Path(path).name == fault:
                fail()
            return original(path,*args,**kwargs)
        return wrapper
    with monkeypatch.context() as patch:
        if fault.endswith(".npz"):
            patch.setattr(np,"savez_compressed",intercept(np.savez_compressed))
        elif fault == "readback":
            patch.setattr(np,"load",fail)
        elif fault.endswith(".png"):
            save = Figure.savefig
            def savefig(fig,path,*args,**kwargs):
                if path.name == fault:
                    fail()
                return save(fig,path,*args,**kwargs)
            patch.setattr(Figure,"savefig",savefig)
        elif fault == "run_config.json":
            patch.setattr(Path,"write_text",intercept(Path.write_text))
        elif fault == "validation":
            patch.setattr(g,"_verify_publication",fail)
        elif fault == "rename":
            patch.setattr(Path,"rename",fail)
        else:
            root = tmp_path/"source-fixture"
            root.mkdir()
            src = root/"source.py"
            src.write_bytes(b"first")
            identity = g._source_identity
            patch.setattr(g,"_SOURCES",("source.py",))
            patch.setattr(g,"_source_identity",lambda _:identity(root))
            verify = g._verify_publication
            def changed(*args):
                verify(*args)
                src.write_bytes(b"second")
            patch.setattr(g,"_verify_publication",changed)
        with pytest.raises((OSError,ValueError),match=fault):
            g._publish(publication_design,dest,None,plots=True)
    assert not dest.exists() and orphan.is_dir()
    assert {p.name for p in tmp_path.iterdir()} <= {"previous-staging","source-fixture"}
    if fault == "inspection.png":
        g._publish(publication_design,dest,None,plots=True)
        assert {p.name for p in dest.iterdir()} == {"run_config.json","label_histogram.png",
                "inspection.png",*(f"pilot_{s}.npz" for s in SPLITS)}


def test_concurrent_publication(publication_design,tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    dest = tmp_path/"shared"
    def publish(annotation):
        try:
            generate_pilot(publication_design,dest,revision=annotation)
            return annotation
        except OSError:
            return None
    with ThreadPoolExecutor(2) as pool:
        winners = [w for w in pool.map(publish,("A","B")) if w]
    assert len(winners)==1
    m = json.loads((dest/"run_config.json").read_text())
    assert m["revision_annotation"] == winners[0]
    before = {p.name:p.read_bytes() for p in dest.iterdir()}
    with pytest.raises(FileExistsError):
        generate_pilot(publication_design,dest,revision="another")
    assert before == {p.name:p.read_bytes() for p in dest.iterdir()}


def test_fingerprint_uses_source_bytes(tmp_path,monkeypatch):
    import ml.dataset.generate as g
    monkeypatch.setattr(g,"_SOURCES",("untracked.py","tracked.py"))
    for name in g._SOURCES:
        (tmp_path/name).write_bytes(b"initial bytes")
    original = g._source_identity(tmp_path)
    for name in g._SOURCES:
        (tmp_path/name).write_bytes(b"changed bytes")
        assert g._source_identity(tmp_path)["sha256"] != original["sha256"]
        (tmp_path/name).write_bytes(b"initial bytes")
        assert g._source_identity(tmp_path) == original


def test_realized_parent_violation(design):
    from ml.dataset.generate import _check_parent
    cfg = simulation_config(design)
    parent = {"split":"test","variant":0,"params":{"shape":"cylinder","dimensions":(.5,2.2),
              "axis_polar_rad":.35}}
    with pytest.raises(ValueError,match="variant"):
        _check_parent(parent,"test",cfg)
    with pytest.raises(ValueError,match="split"):
        _check_parent(parent,"val",cfg)


def test_harmonic_information_returns_with_finer_sampling(design):
    from ml.dataset.labels import observation_label
    first = SimParams("box",(1.,)*3,(.1,.1,.8,.8,.5,.5),0,0,-.1,4,
                      (1,0,0),(-np.cos(.2),np.sin(.2),0))
    second = replace(first,reflectivity=(.1,.7,.8,.8,.5,.5))
    cfg = simulation_config(design)
    for cadence in (.4,.08,.02):
        t = np.linspace(0,24,round(24/cadence)+1)
        a,b = (simulate_lightcurve(p,t) for p in (first,second))
        x,y = (observation_label(t,flux,cfg) for flux in (a,b))
        if cadence == .4:
            assert a.tobytes()==b.tobytes()
            assert x==y and x[0]=="insufficient sampling"
        else:
            assert np.any(a!=b)
            if cadence == .02:
                assert x[1]==pytest.approx(2.) and y[1]==pytest.approx(4.)


def test_undercovered_quota_excluded(design,tmp_path):
    quotas = {s:{k:int(k=="under_covered") for k in SOURCES} for s in SPLITS}
    config = replace(design,simulation=dict(design.simulation, quotas=quotas,
                     baseline_range_s=[2.,2.],max_attempts=2))
    with pytest.raises(ValueError,match="exhausted quota train/under_covered.*insufficient coverage"):
        generate_pilot(config,tmp_path/"unpublished")
    assert not (tmp_path/"unpublished").exists()


def test_base_revision_and_annotation_independent_of_source(publication_design,tmp_path,monkeypatch):
    import ml.dataset.generate as g
    base = "base-one"
    def git(args,**kwargs):
        return base if "rev-parse" in args else "?? ml/dataset/generate.py"
    monkeypatch.setattr(g.subprocess,"check_output",git)
    identities = []
    for base in ("base-one","base-two"):
        dest = generate_pilot(publication_design,tmp_path/base,revision="fixture annotation")
        manifest = json.loads((dest/"run_config.json").read_text())
        assert manifest["commit"] == base and manifest["dirty"]
        assert manifest["revision_annotation"] == "fixture annotation"
        identities.append(manifest["source_identity"])
    assert identities[0] == identities[1]
