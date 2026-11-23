import copy

import numpy as np
import pytest

from core.config import load_config
from ml.dataset import corpus
from ml.dataset.degrade import coverage_error, degrade, profile_pools, timeline
from preprocessing.prepare import prepare


def test_generator_replay_and_development_controls():
    cfg = load_config()
    pools, _ = profile_pools(corpus.ROOT / "data/profiles", cfg.simulation["seed"])
    assert not {p["profile_id"] for p in pools["val"]} & {p["profile_id"] for p in pools["train"]}
    for split in ("train", "val"):
        for category, subtype in (("reliable", None), ("noise_or_constant", "noise"),
                                  ("noise_or_constant", "constant"), ("multi_frequency", None),
                                  ("under_covered", None)):
            job = cfg, pools[split], split, category, 0, subtype
            obs, windows, soft, row, failures = corpus.generate_observation(job)
            obs2, windows2, soft2, row2, failures2 = corpus.generate_observation(job)
            np.testing.assert_equal(obs.y, obs2.y)
            np.testing.assert_array_equal(obs.t_s, obs2.t_s)
            np.testing.assert_array_equal(windows, windows2)
            np.testing.assert_array_equal(soft, soft2)
            assert row == row2 and failures == failures2
            assert row["source"] == (subtype or category)
            np.testing.assert_array_equal(windows, corpus.features.build_cnn_windows(
                prepare(obs, cfg.preprocess), cfg.cnn))
            assert soft[128] == (category != "reliable")


def test_profile_timeline_and_missingness(tmp_path):
    import shutil
    shutil.copy(corpus.ROOT / "data/profiles/default_profiles.json", tmp_path)
    pools, _ = profile_pools(tmp_path, 17)
    p = pools["train"][0]
    rng = np.random.default_rng(5)
    t = timeline(p, rng)
    assert len(t) == 29 and t[-1] == 32.0 and np.median(np.diff(t)) == 1
    for gap in p["gap_fractions"]:
        assert np.any(np.isclose(np.diff(t), gap * 32))
    _, y, mask = degrade(t, np.ones_like(t), p, rng, rng)
    assert mask.sum() == 28 and np.all(y[mask] == 0) and np.isnan(y[~mask]).all()
    broken = copy.deepcopy(p)
    broken["duration_s"] = .01
    with pytest.raises(ValueError):
        timeline(broken, rng)


def test_observed_cycle_boundary():
    cfg = {"min_cycles": 3., "flat_rtol": 1e-10}
    valid = np.array([True, True, True])
    y = np.array([0., 1., 2.])
    meta = {"parent_repeat_s": 1., "source": "under_covered"}
    for end, allowed in ((np.nextafter(3., 0.), True), (3., False),
                         (np.nextafter(3., 4.), False)):
        assert (coverage_error(np.array([0., 1., end]), y, valid, meta, cfg) is None) == allowed
    assert coverage_error(np.array([0., 1., 3.]), y, ~np.array([False,False,True]), meta, cfg) is None


def test_long_support_and_finite_source_rejection():
    from ml.dataset.generate import _draw_curve, simulation_config
    cfg = load_config()
    pools, _ = profile_pools(corpus.ROOT / "data/profiles", 17)
    longest = max(pools["train"], key=lambda p: p["duration_s"])
    obs, _, _, _, _ = corpus.generate_observation(
        (cfg, [longest], "train", "noise_or_constant", 1, "constant"))
    assert obs.t_s[-1] == longest["duration_s"] > 1851.174
    short = dict(simulation_config(cfg), baseline_range_s=[1800.,1800.])
    assert _draw_curve(short, "train", "noise_or_constant", 17, obs.t_s,
                       subtype="constant") == "insufficient clean source support"


def test_retry_exhaustion_and_parallel_replay():
    from concurrent.futures import ProcessPoolExecutor
    from dataclasses import replace
    cfg = load_config()
    pools, _ = profile_pools(corpus.ROOT / "data/profiles", 17)
    jobs = [(cfg, pools["train"], "train", "noise_or_constant", i, "constant") for i in range(2)]
    with ProcessPoolExecutor(max_workers=2) as pool:
        parallel = list(pool.map(corpus.generate_observation, jobs))
    for a, b in zip(parallel, map(corpus.generate_observation, jobs)):
        np.testing.assert_equal(a[0].y, b[0].y)
        np.testing.assert_array_equal(a[1], b[1])
        assert a[3:] == b[3:]
    impossible = replace(cfg, simulation=dict(cfg.simulation, max_attempts=1))
    profile = dict(pools["train"][0], missing_fraction=1.)
    with pytest.raises(ValueError, match="insufficient valid observations"):
        corpus.generate_observation((impossible, [profile], "train", "noise_or_constant", 0,
                                     "constant"))
