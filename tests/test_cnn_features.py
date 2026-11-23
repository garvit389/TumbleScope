import ast
from pathlib import Path


def test_single_feature_builder_definition():
    root = Path(__file__).resolve().parents[1]
    modules = ("app", "classical_analysis", "core", "evaluation", "ingestion", "ml", "physics",
               "pipeline", "preprocessing", "reconciliation", "scripts", "verdict")
    definitions = [
        path.relative_to(root)
        for module in modules
        for path in (root / module).rglob("*.py")
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "build_cnn_windows"
    ]
    assert definitions == [Path("ml/features.py")]


def test_cp2_golden(prepared, fixture_dir):
    """Input mock_prepared.npz via conftest.prepared; archive key expected_windows.

    §16 reference computed row-by-row, independently of ml.features.
    """
    import numpy as np
    from core.config import load_config
    from ml.features import build_cnn_windows

    cfg = load_config().cnn
    settings = dict(window_length=1024, window_stride=512, y_clip=6., dt_floor_s=.1,
                    L_min=-1., L_max=4., max_windows=9, min_valid_per_window=21)
    assert {key: cfg[key] for key in settings} == settings
    actual = build_cnn_windows(prepared, cfg)
    with np.load(fixture_dir / "golden_cnn_windows.npz", allow_pickle=False) as archive:
        assert archive.files == ["expected_windows"]
        expected = archive["expected_windows"]
    assert actual.shape == expected.shape == (1, 3, 1024)
    assert actual.dtype == expected.dtype == np.float32
    assert actual.flags.c_contiguous and np.isfinite(actual).all()
    np.testing.assert_array_max_ulp(actual, expected, maxulp=1)
    np.testing.assert_array_equal(actual[:, 1], expected[:, 1])
    assert np.all(actual[:, :, 29:] == 0) and actual[0, 0, 11] == 0


def test_cp2_window_boundaries():
    import numpy as np
    from core.config import load_config
    from core.schemas import Observation
    from ml.features import build_cnn_windows
    from preprocessing.prepare import prepare

    config = load_config()
    for n, starts in ((1023, [0]), (1024, [0]), (1025, [0,1]), (1600, [0,512,576]),
                      (9216, list(range(0,8193,1024)))):
        t = np.arange(n, dtype=float)
        prep = prepare(Observation("long", t, t.copy(), None, "unknown", {}), config.preprocess)
        result = build_cnn_windows(prep, config.cnn)
        assert result.shape == (len(starts), 3, 1024)
        for window, start in zip(result, starts):
            observed = min(n, 1024)
            expected = (prep.y_norm[start:start+observed] / 6).astype(np.float32)
            np.testing.assert_array_equal(window[0, :observed], expected)
            assert window[2, 0] == (np.float32(.2) if start else 0.)
            assert np.all(window[:, observed:] == 0)
    for count in (20, 21, 22):
        y = np.r_[np.arange(count, dtype=float), np.nan]
        prep = prepare(Observation("short", np.arange(len(y), dtype=float), y,
                                   None, "unknown", {}), config.preprocess)
        prep.y_norm[:8] = [-7,-6,-5,5,6,7,0,1]
        prep.dt_s[:7] = [0,.05,.1,1,10000,20000,0]
        result = build_cnn_windows(prep, config.cnn)
        assert result.shape == (int(count >= 21), 3, 1024)
        if len(result):
            np.testing.assert_array_equal(result[0,0,:6],
                                          np.array([-1,-1,-5/6,5/6,1,1], dtype="f4"))
            np.testing.assert_array_equal(result[0,2,:7], np.array([0,0,0,.2,1,1,0], dtype="f4"))


def test_generator_calls_shared_builder(monkeypatch):
    import numpy as np
    from unittest.mock import Mock
    from core.config import load_config
    from ml import features
    from ml.dataset.corpus import ROOT, generate_observation
    from ml.dataset.degrade import profile_pools
    from preprocessing.prepare import prepare

    config = load_config()
    profiles, _ = profile_pools(ROOT / "data/profiles", 17)
    original = features.build_cnn_windows
    spy = Mock(wraps=original)
    monkeypatch.setattr(features, "build_cnn_windows", spy)
    obs, windows, *_ = generate_observation((config, profiles["train"], "train",
                                             "noise_or_constant", 0, "constant"))
    spy.assert_called_once()
    np.testing.assert_array_equal(windows, original(prepare(obs, config.preprocess), config.cnn))


def test_cp3b_real_inference_calls_shared_builder(prepared, monkeypatch):
    from unittest.mock import Mock
    from core.config import load_config, resolve_period_bounds
    from ml import features, infer

    config = load_config()
    low, high = resolve_period_bounds(config.period_bounds)
    cfg = dict(config.cnn, period_min_s=low, period_max_s=high)
    builder = Mock(wraps=features.build_cnn_windows)
    monkeypatch.setattr(features, "build_cnn_windows", builder)
    result = infer.run_cnn(prepared, cfg)
    result.validate()
    builder.assert_called_once_with(prepared, cfg)
    assert result.status == "ok", result.error
    assert not result.is_stub and result.meta["n_windows"] == 1
