"""TASK-3771: an exchange region's stage is the level its water fills on its own ground."""

import numpy as np
import pytest

from anuga_drainage.culverts.coupler import LevelFill


def brute_level(z, a, volume):
    lo, hi = float(np.min(z)), float(np.max(z)) + volume / float(np.sum(a)) + 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if np.sum(a * np.maximum(mid - z, 0.0)) < volume:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def test_flat_region_reads_volume_over_area_plus_bed():
    fill = LevelFill([2.0, 2.0, 2.0], [10.0, 20.0, 30.0])
    for volume in (0.0, 0.6, 60.0, 600.0):
        assert fill.stage(volume) == pytest.approx(volume / 60.0 + 2.0, abs=1e-12)


def test_wet_region_on_any_ground_matches_the_old_mean_bed_formula():
    z, a = np.array([1.0, 2.0, 4.0]), np.array([5.0, 10.0, 5.0])
    fill = LevelFill(z, a)
    volume = 200.0  # every cell under water
    assert fill.stage(volume) == pytest.approx(volume / a.sum() + np.sum(z * a) / a.sum(), abs=1e-12)


def test_dry_region_reads_its_lowest_ground():
    fill = LevelFill([5.0, 3.6, 9.0], [1.0, 1.0, 1.0])
    assert fill.stage(0.0) == pytest.approx(3.6)
    assert fill.stage(-1e-17) == pytest.approx(3.6)  # dry-cell float noise


def test_bed_plus_embankment_region_reads_the_water_level_not_a_raised_one():
    # 40 % creek bed at 3.6 m, 60 % embankment chosen so the mean ground is 5.5 m;
    # 1 m of water over the bed only. The pre-TASK-3771 formula read 5.5 + 0.4 = 5.9 m.
    z_emb = (5.5 - 0.4 * 3.6) / 0.6
    fill = LevelFill([3.6, z_emb], [40.0, 60.0])
    assert fill.mean_bed == pytest.approx(5.5)
    assert fill.stage(40.0) == pytest.approx(4.6, abs=1e-12)


def test_matches_brute_force_and_is_continuous_and_increasing():
    rng = np.random.default_rng(3771)
    z = rng.uniform(0.0, 8.0, 200)
    a = rng.uniform(1.0, 30.0, 200)
    fill = LevelFill(z, a)
    volumes = np.linspace(0.0, 2.0e4, 801)
    stages = np.array([fill.stage(v) for v in volumes])
    assert np.all(np.diff(stages) > 0.0)
    for v in volumes[::40]:
        assert fill.stage(v) == pytest.approx(brute_level(z, a, v), abs=1e-9)
