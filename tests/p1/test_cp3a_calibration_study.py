"""Sanity checks for the synthetic calibration study's outcome accounting."""
import pytest

from core.config import load_config
from evaluation.cp3a_calibration_study import evaluate, sample, wilson
from evaluation.cp3a_controlled_pairs import case


def test_wilson_edges_and_independent_scenario_generation():
    assert wilson(0, 30)[1] == pytest.approx(.1135133932)
    assert wilson(30, 30)[0] == pytest.approx(.8864866068)
    config = load_config()
    first = sample("missing", 51000, config)
    second = sample("missing", 51000, config)
    assert first[0].to_dict() == second[0].to_dict()
    assert (~first[0].valid_mask).any()
    assert first[3] == [32.]


def test_real_double_hump_diagnostics_and_retention():
    prep, estimators, cfg, truths = sample("double_hump", 51000, load_config())
    row = evaluate("double_hump", 51000, prep, estimators, cfg, truths, {})
    assert row["truth_periods_s"] == [16., 32.]
    assert row["harmonic_link"] and row["harmonic_both_retained"]
    assert len(row["truth_present"]) == len(row["truth_retained"]) == 2


def test_controlled_pairs_have_identical_inputs_for_opposite_labels():
    config = load_config()
    duplicate = case(71000, "duplicate", .5, config)
    distinct = case(71000, "distinct", .5, config)
    assert {k: v for k, v in duplicate.items() if k != "kind"} == {
        k: v for k, v in distinct.items() if k != "kind"}
    assert duplicate["same_cluster"]
    alias = case(71000, "alias", None, config)
    assert alias["usable_offset"] and alias["second_alias_risk"] > 0
