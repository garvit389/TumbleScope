from copy import deepcopy
from types import SimpleNamespace

import pytest

from evaluation.stage2_tuning import __main__ as stage2
from evaluation import tune_reconciliation as weights


def test_stage2_replay_changes_only_separation_and_preserves_cache(monkeypatch):
    record = {"snapshot": {"settings": {"min_cycles": 2, stage2.THRESHOLD: 1.5,
                                        "window_result": {"peaks": []}}}}
    original = deepcopy(record)
    frozen = {"w_C": .05, "w_L": .15, "w_D": .8, "lambda_A": .3}

    def replay(baseline, records, overrides, *, trace):
        assert baseline == "fixture" and trace
        assert overrides is frozen
        assert records[0]["snapshot"]["settings"] == {
            **original["snapshot"]["settings"], stage2.THRESHOLD: 1.25,
        }
        assert records[1]["snapshot"] is None
        return "replayed"

    monkeypatch.setattr(weights, "replay", replay)
    assert stage2.replay_threshold("fixture", [record, {"snapshot": None}], frozen, 1.25,
                                   trace=True) == "replayed"
    assert record == original
    with pytest.raises(ValueError, match="prespecified"):
        stage2.replay_threshold("fixture", [record], frozen, 0.75)


def _row(dataset, index, predicted, *, nrp=False):
    return dict(obs_id=str(index), object_id=str(index), reference_period_s=None if nrp else 10.0,
                no_reliable_period=nrp, baseline_operation="completed_abstention",
                configurations={
                    "cnn_classical": {"period_s": predicted,
                                      "candidate_periods_s": [] if predicted is None else [predicted, 10.0]},
                    "pdm_only": {"period_s": 10.0, "candidate_periods_s": [10.0]},
                    "complete": {"period_s": None, "candidate_periods_s": []},
                })


@pytest.mark.parametrize(("stable_gain", "expected"), [(0.0, 1.5), (0.2, 1.25)])
def test_search_retains_baseline_unless_safeguards_and_stability_pass(monkeypatch, stable_gain, expected):
    cohorts = {"real-val": 2, "synthetic-dev": 43}
    monkeypatch.setattr(weights, "COUNTS", cohorts)
    cfg = SimpleNamespace(reconcile=dict(min_cycles=2.0, max_rel_period_uncertainty=.5,
                                         n_phase_bins=10, min_points_per_bin=2,
                                         min_phase_coverage=.5, min_separation_resolutions=1.5))
    baselines = {d: SimpleNamespace(provenance={"config_hash": "hash"}, cfg=cfg, dataset=d)
                 for d in cohorts}
    records = {d: [None] * n for d, n in cohorts.items()}
    original = {d: [_row(d, 0, 8), _row(d, 1, 8)] for d in cohorts}
    base = {d: [_row(d, 0, 10), _row(d, 1, 8)] for d in cohorts}
    for i in range(41):
        for collection in (original, base):
            collection["synthetic-dev"].append(_row("synthetic-dev", i + 2, None, nrp=True))
    fixed = dict(w_C=.05, w_L=.15, w_D=.8, lambda_A=.3)
    traces = {d: [] for d in cohorts}
    original_reports = {d: weights._report(rows) for d, rows in original.items()}
    stats = {d: weights._stats(rows, "cnn_classical") for d, rows in base.items()}
    saved = dict(provenance={d: b.provenance for d, b in baselines.items()},
                 bootstrap_seed=weights.BOOTSTRAP_SEED, bootstrap_draws=weights.BOOTSTRAP_DRAWS,
                 baseline=original_reports, baseline_objective=0.0, selected_objective=.5,
                 selected_overrides=fixed, traces=traces,
                 trials=[dict(overrides=fixed, stats=stats, objective=.5, safeguards=True,
                              bootstrap_gain_p05=.5)])
    monkeypatch.setattr(weights, "verify_baseline", lambda b, rows: (original[b.dataset], []))
    monkeypatch.setattr(weights, "_stability", lambda before, after:
                        .5 if before == original else stable_gain)

    def replay(b, records, chosen, value=None, *, trace=False):
        assert chosen == fixed
        rows = deepcopy(base[b.dataset])
        if value in (1.0, 1.25):
            rows[1] = _row(b.dataset, 1, 10)
        if value == 1.0 and b.dataset == "synthetic-dev":
            rows[2:] = [_row(b.dataset, i + 2, 10, nrp=True) for i in range(41)]
        return rows, []

    monkeypatch.setattr(stage2, "replay_threshold", replay)
    result = stage2.search(baselines, records, saved)
    assert result["recommended_stage2"][stage2.THRESHOLD] == expected
    assert result["recommended_stage2"]["min_cycles"] == 2
    assert result["trials"][0]["safeguards"] is False
    assert result["trials"][1]["safeguards"] is True
    assert result["baseline_equivalent"]
    saved["selected_objective"] = .6
    with pytest.raises(ValueError, match="selected weight replay"):
        stage2.search(baselines, records, saved)


def test_stage2_module_preserves_legacy_cache_fingerprint_scope():
    assert not any(path.startswith("evaluation/stage2_tuning/") for path in weights._code_hashes())
