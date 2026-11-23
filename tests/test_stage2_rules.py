"""CP3a root guard for the real candidate-level Stage 2 contract."""

from dataclasses import replace

import numpy as np
import pytest

from core.config import load_config
from core.schemas import Candidate, Observation, STAGE2_FAIL_REASONS
from preprocessing.prepare import prepare
from reconciliation.reconcile import reconcile


def test_real_stage2_cycles_neighbours_and_harmonic_exemption(mock_cnn):
    times = np.linspace(0., 80., 401)
    prep = prepare(Observation("stage2-root", times, np.sin(2 * np.pi * times / 8),
                               None, "unknown"), {})
    candidates = [Candidate(period, support, support, {}) for period, support in
                  [(8., 1.), (16., .9), (8.5, .8), (80., .7)]]
    estimator = replace(mock_cnn, candidates=candidates)
    cfg = dict(load_config().reconcile, period_min_s=2., period_max_s=100.,
               cluster_tol_resolutions=.1, min_separation_resolutions=6., lambda_A=0.,
               window_result={"freq_hz": [0.], "window_power": [1.], "peaks": [],
                              "freq_resolution": prep.meta["freq_resolution"]})
    result = reconcile(prep, [estimator], cfg)
    result.validate()
    by_period = {cluster.period_s: cluster for cluster in result.clusters}
    assert set(by_period) == {8., 16., 8.5, 80.}
    for cluster in result.clusters:
        assert cluster.stage2["cycles"] == pytest.approx(prep.meta["baseline_s"] / cluster.period_s)
        assert cluster.stage2["passed"] is (not cluster.stage2["fail_reasons"])
        assert set(cluster.stage2["fail_reasons"]) <= set(STAGE2_FAIL_REASONS)
    assert STAGE2_FAIL_REASONS[0] in by_period[80.].stage2["fail_reasons"]
    assert STAGE2_FAIL_REASONS[3] in by_period[8.5].stage2["fail_reasons"]
    assert by_period[8.5].stage2["merged_with"] == by_period[8.].cluster_id
    assert abs(1 / 8. - 1 / 16.) < cfg["min_separation_resolutions"] * prep.meta["freq_resolution"]
    assert {"other_id": by_period[8.].cluster_id, "ratio": .5} in by_period[16.].harmonic_links
    assert STAGE2_FAIL_REASONS[3] not in by_period[16.].stage2["fail_reasons"]
    assert by_period[16.].cluster_id in result.shortlist
