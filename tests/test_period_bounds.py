from dataclasses import replace
from pathlib import Path
import shutil
from unittest.mock import Mock

import pytest
import yaml

from core.config import config_hash, get_period_bounds, load_config, resolve_period_bounds
from pipeline import controller


CONFIGS = Path(__file__).resolve().parents[1] / "configs"
BOUND_KEYS = {"min_period_s", "max_period_s", "p_min_s", "p_max_s", "period_min_s", "period_max_s"}


def test_only_period_bounds_file_defines_bounds():
    def find_keys(value):
        if isinstance(value, dict):
            yield from value.keys()
            for entry in value.values():
                yield from find_keys(entry)
        elif isinstance(value, list):
            for entry in value:
                yield from find_keys(entry)

    manifest = yaml.safe_load((CONFIGS / "default.yaml").read_text())
    for path in CONFIGS.rglob("*.y*ml"):
        if path == CONFIGS / "period_bounds.yaml":
            continue
        data = yaml.safe_load(path.read_text())
        assert not BOUND_KEYS.intersection(find_keys(data)), path
    assert all((CONFIGS / file).is_file() for file in manifest.values() if isinstance(file, str))


def test_unset_api_rejects_bounds(monkeypatch, tmp_path):
    from core import config

    monkeypatch.setattr(config, "_CONFIG_DIR", tmp_path)
    (tmp_path / "period_bounds.yaml").write_text("status: TBD\nmin_period_s: null\nmax_period_s: null\n")
    with pytest.raises(ValueError, match="^contract: period bounds not set$"):
        get_period_bounds()


def test_configured_bounds():
    if load_config().period_bounds["status"] == "TBD":
        pytest.skip("Period bounds are TBD. P1 supplies them by CP1.")
    lower, upper = get_period_bounds()
    assert 0 < lower < upper


@pytest.mark.parametrize(
    "lower,upper", [(0, 10), (10, 2), (True, 3), (2, float("inf")), (1, 1), (2, 10**400)],
)
def test_invalid_period_bounds(lower, upper):
    with pytest.raises(ValueError, match="contract: invalid period bounds"):
        resolve_period_bounds({"status": "SET", "min_period_s": lower, "max_period_s": upper})


def test_yaml_loader_and_config_hash(tmp_path):
    shutil.copytree(CONFIGS, tmp_path / "configs")
    config_dir = tmp_path / "configs"
    path = config_dir / "default.yaml"
    (config_dir / "period_bounds.yaml").write_text("status: SET\nmin_period_s: 2\nmax_period_s: 20\n")
    (config_dir / "cnn.yaml").write_text("z: 1\na: 2\n")
    first = load_config(path)
    assert resolve_period_bounds(first.period_bounds) == (2, 20)
    assert config_hash(first) == config_hash(replace(first, cnn={"a": 2, "z": 1}))
    assert config_hash(first) != config_hash(replace(first, cnn={"a": 3, "z": 1}))
    (config_dir / "cnn.yaml").write_text("nested:\n  - period_min_s: 2\n")
    with pytest.raises(ValueError, match="period bounds duplicated"):
        load_config(path)


def test_loader_rejects_partial_or_non_null_tbd(tmp_path):
    config_dir = tmp_path / "configs"
    shutil.copytree(CONFIGS, config_dir)
    path = config_dir / "default.yaml"
    for body in (
        "status: TBD\nmin_period_s: null\n",
        "status: TBD\nmin_period_s: 2\nmax_period_s: null\n",
    ):
        (config_dir / "period_bounds.yaml").write_text(body)
        with pytest.raises(ValueError, match="unset period bounds must be null"):
            load_config(path)


@pytest.mark.parametrize("period", [1.99, 20.01])
def test_controller_rejects_out_of_range_candidates(
    monkeypatch, observation, injected_config, mock_cnn, period,
):
    bad = replace(mock_cnn, candidates=[replace(mock_cnn.candidates[0], period_s=period)])
    bad.validate()
    runner = Mock(return_value=bad)
    monkeypatch.setattr(controller, "run_cnn", runner)
    result = controller.analyze(observation, injected_config)
    result.validate()
    call_cfg = runner.call_args.args[1]
    assert (call_cfg["period_min_s"], call_cfg["period_max_s"]) == (2.0, 20.0)
    assert result.estimators[mock_cnn.method].status == "failed"
    assert result.estimators[mock_cnn.method].candidates == []
    assert result.stage_errors[mock_cnn.method] == (
        "contract_violation: contract: candidate outside injected period bounds"
    )
