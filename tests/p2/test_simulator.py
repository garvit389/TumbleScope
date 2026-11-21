from dataclasses import replace
from timeit import repeat

import numpy as np
import pytest

from physics.geometry import facets
from physics.params import SimParams
from physics.simulator import simulate_lightcurve


BOX = SimParams("box", (2., 3., 4.), (1.,) * 6, 0., 0., 0., 8., (1., 0., 0.), (2., 0., 0.))


def test_box_and_harmonic():
    times = np.linspace(0, 8, 111)
    flux = simulate_lightcurve(BOX, times)
    theta = 2 * np.pi * times / 8
    np.testing.assert_allclose(flux, 12 * np.cos(theta)**2 + 8 * np.sin(theta)**2)
    np.testing.assert_allclose(simulate_lightcurve(BOX, times + 4), flux, atol=1e-9, rtol=0)
    assert flux[0] == 12
    assert simulate_lightcurve(BOX, [2])[0] == 8
    asymmetric = replace(BOX, reflectivity=(1., .2, .3, .4, .5, .6))
    assert np.ptp(simulate_lightcurve(asymmetric, [0, 4])) > 1
    side_view = replace(asymmetric, sun=(0, 1, 0), observer=(0, 1, 0))
    np.testing.assert_allclose(simulate_lightcurve(side_view, [2., 6.]), [12., 2.4])
    tilted = replace(asymmetric, axis_polar_rad=.6, axis_azimuth_rad=1.4, phase_rad=.8)
    np.testing.assert_allclose(simulate_lightcurve(tilted, times + 8),
                               simulate_lightcurve(tilted, times), rtol=0, atol=1e-9)
    assert not simulate_lightcurve(replace(BOX, observer=(-1, 0, 0)), times).any()


def test_cylinder():
    cylinder = replace(BOX, shape="cylinder", dimensions=(2., 5.), reflectivity=(.5,) * 18)
    normals, areas = facets(cylinder)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1)
    np.testing.assert_allclose(areas[-2:], 32 * np.sin(np.pi / 8))
    side = 20 * np.sin(np.pi / 16)
    np.testing.assert_allclose(areas[:-2], side)
    np.testing.assert_allclose(simulate_lightcurve(cylinder, np.linspace(0, 8, 137)), 2 * side)


def test_time_order_empty_and_reproducibility():
    t = np.array([7., 0., 7., -2.])
    y = simulate_lightcurve(BOX, t)
    assert y.dtype == np.float64 and np.all(y >= 0)
    np.testing.assert_array_equal(y, simulate_lightcurve(BOX, t))
    np.testing.assert_allclose(y, [simulate_lightcurve(BOX, [v])[0] for v in t])
    assert simulate_lightcurve(BOX, []).shape == (0,)


@pytest.mark.parametrize("change", [dict(shape="sphere"), dict(dimensions=(0, 1, 1)),
    dict(reflectivity=(2,) * 6), dict(reflectivity=(1,)), dict(phase_cycle_s=np.nan),
    dict(phase_cycle_s=0), dict(sun=(0, 0, 0)), dict(observer=(1, 2)),
    dict(cylinder_sides=2)])
def test_invalid_parameters(change):
    with pytest.raises(ValueError):
        replace(BOX, **change)


def test_invalid_times_and_overflow():
    for t in ([[1]], [np.nan], [1j]):
        with pytest.raises(ValueError):
            simulate_lightcurve(BOX, t)
    with pytest.raises(ValueError, match="facet areas"):
        simulate_lightcurve(replace(BOX, dimensions=(1e300,) * 3), [1.])


def test_box_runtime():
    times = np.linspace(0., 8., 1000)
    simulate_lightcurve(BOX, times)
    milliseconds = np.median(repeat(lambda: simulate_lightcurve(BOX, times), number=20, repeat=7)) * 50
    print(f"box median ms: {milliseconds:.6f}")
    assert milliseconds < 5


@pytest.mark.parametrize("field,size", [("sun",3),("observer",3),("dimensions",3),("reflectivity",6)])
@pytest.mark.parametrize("dtype", [complex, object])
def test_reject_complex_before_cast(field,size,dtype):
    with pytest.raises(ValueError,match="complex"):
        replace(BOX, **{field:np.array([.5+0j]*size,dtype=dtype)})
