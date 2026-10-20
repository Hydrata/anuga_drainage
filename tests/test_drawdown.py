"""TASK-3771: the losing end of a headwall culvert is drawn down to a FLAT water surface.

The stock Inlet_operator removes water with set_depths(uniform): every cell of the region gets
the same depth, so on a region spanning a creek bed and its bank the bank water is LIFTED (the
Towradgi Northern Distributor bridge: bank stage 12-13.9 m against ~8.8 m in the channel, then
53-69 m3/s over the road crest). Each behavioural test below runs the stock operator on the
same state as its negative control and asserts that the stock write FAILS the property.
"""
import numpy as np
import pytest

from anuga_drainage.culverts.coupler import INERTIA_MAX_ADVERSE_HEAD_M, inertia_floor
from anuga_drainage.culverts.drawdown import drawdown_stages, withdraw

CHANNEL_LEVEL = 8.8   # the uniform water surface before the withdrawal (m)


def brute_drawdown(stages, z, a, volume):
    """Bisection on the level (independent of the sorted-breakpoint solver)."""
    depth = np.maximum(stages - z, 0.0)
    lo, hi = float(z.min()), float(stages.max())
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if np.dot(a, np.minimum(depth, np.maximum(stages - mid, 0.0))) > volume:
            lo = mid
        else:
            hi = mid
    return np.where(stages > z, np.maximum(z, np.minimum(stages, hi)), stages)


def flat_drawdown_violations(before, after, z, tol=1e-9):
    """What a flat draw-down must never do: lift a cell, leave the wet cells at more than one
    level, or leave a cell with negative depth. Returns a list of the violations found."""
    found = []
    if np.any(after > before + tol):
        found.append(f"lifted cells by up to {np.max(after - before):.3f} m")
    wet = after > z + tol
    if wet.any() and np.ptp(after[wet]) > tol:
        found.append(f"wet stages span {np.ptp(after[wet]):.3f} m")
    if np.any(after < z - tol):
        found.append("negative depth")
    return found


# ------------------------------------------------------------------ pure solver
def sloped_region(n=29):
    z = np.linspace(4.0, 11.0, n)                     # creek bed 4 m up to a bank at 11 m
    a = np.linspace(5.0, 12.0, n)
    stages = np.maximum(z, CHANNEL_LEVEL)             # a flat 8.8 m surface; bank above it dry
    return stages, z, a


def test_solver_lowers_one_flat_surface_and_removes_exactly_the_volume():
    stages, z, a = sloped_region()
    water = float(np.dot(a, stages - z))
    for volume in (1e-6, 1.0, 0.3 * water, 0.9 * water):
        new, removed = drawdown_stages(stages, z, a, volume)
        assert removed == pytest.approx(volume, rel=1e-12)
        assert float(np.dot(a, stages - new)) == pytest.approx(volume, rel=1e-12)
        assert flat_drawdown_violations(stages, new, z) == []
        wet = new > z
        assert np.all(new[wet] < CHANNEL_LEVEL)
        bank = z >= CHANNEL_LEVEL
        assert np.array_equal(new[bank], z[bank])     # the bank stays dry, untouched
        np.testing.assert_allclose(new, brute_drawdown(stages, z, a, volume), atol=1e-9)


def test_solver_matches_bisection_on_an_uneven_surface():
    rng = np.random.default_rng(3771)
    z = rng.uniform(0.0, 6.0, 300)
    stages = z + np.where(rng.random(300) < 0.7, rng.uniform(0.0, 3.0, 300), 0.0)
    a = rng.uniform(1.0, 20.0, 300)
    water = float(np.dot(a, stages - z))
    for volume in np.linspace(0.0, water, 23)[1:-1]:
        new, removed = drawdown_stages(stages, z, a, volume)
        assert removed == pytest.approx(volume, rel=1e-10)
        assert np.all(new <= stages) and np.all(new >= z)
        np.testing.assert_allclose(new, brute_drawdown(stages, z, a, volume), atol=1e-8)


def test_solver_zero_volume_is_an_exact_no_op_and_overdraw_drains_to_the_bed():
    stages, z, a = sloped_region()
    new, removed = drawdown_stages(stages, z, a, 0.0)
    assert removed == 0.0 and np.array_equal(new, stages)
    water = float(np.dot(a, stages - z))
    new, removed = drawdown_stages(stages, z, a, 10.0 * water)
    assert removed == pytest.approx(water, rel=1e-12)
    assert np.array_equal(new, z)


# ------------------------------------------------------------------ on a real ANUGA region
def sloped_domain(tmp_path, xmom=0.0):
    """A 70 x 10 m domain whose bed rises 4 -> 11 m along x, a flat 8.8 m water surface over
    the low part (dry above), and an Inlet_operator on the whole of it."""
    import anuga
    domain = anuga.rectangular_cross_domain(14, 2, len1=70.0, len2=10.0)
    domain.set_datadir(str(tmp_path))
    domain.set_quantity("elevation", lambda x, y: 4.0 + 0.1 * x, location="centroids")
    domain.set_quantity("stage", lambda x, y: np.maximum(4.0 + 0.1 * x, CHANNEL_LEVEL),
                        location="centroids")
    z = domain.quantities["elevation"].centroid_values
    depth = domain.quantities["stage"].centroid_values - z
    domain.quantities["xmomentum"].centroid_values[:] = xmom * depth   # uniform velocity
    domain.quantities["ymomentum"].centroid_values[:] = -0.5 * xmom * depth
    region = anuga.Region(domain, polygon=[[0, 0], [70, 0], [70, 10], [0, 10]])
    op = anuga.Inlet_operator(domain, region, Q=0.0)
    domain.timestep = 2.0
    return domain, op


def stock_remove(op, volume, dt):
    op.set_Q(-volume / dt)
    op()


def test_region_withdrawal_is_flat_and_the_stock_one_lifts_the_bank(tmp_path):
    dom_new, op_new = sloped_domain(tmp_path / "new")
    dom_old, op_old = sloped_domain(tmp_path / "stock")
    z = op_new.inlet.get_elevations()
    before = op_new.inlet.get_stages()
    water = op_new.inlet.get_total_water_volume()
    volume = 0.25 * water

    removed = withdraw(op_new, volume, 2.0)
    after = op_new.inlet.get_stages()
    assert removed == pytest.approx(volume, rel=1e-12)
    assert water - op_new.inlet.get_total_water_volume() == pytest.approx(volume, rel=1e-12)
    assert flat_drawdown_violations(before, after, z) == []
    assert after[after > z].max() < CHANNEL_LEVEL
    bank = z >= CHANNEL_LEVEL
    assert bank.any() and np.array_equal(after[bank], z[bank])

    # negative control: the stock uniform-depth write on the same state
    stock_remove(op_old, volume, 2.0)
    stock = op_old.inlet.get_stages()
    assert water - op_old.inlet.get_total_water_volume() == pytest.approx(volume, rel=1e-9)
    violations = flat_drawdown_violations(before, stock, z)
    assert any("lifted" in v for v in violations), violations
    assert stock[bank].max() > CHANNEL_LEVEL + 1.0    # bank water raised above the surface


def test_overdraw_takes_what_is_there_with_the_stock_bookkeeping(tmp_path):
    dom_new, op_new = sloped_domain(tmp_path / "new")
    dom_old, op_old = sloped_domain(tmp_path / "stock")
    water = op_new.inlet.get_total_water_volume()
    ask = 3.0 * water
    i0_new, i0_old = dom_new.fractional_step_volume_integral, dom_old.fractional_step_volume_integral

    removed = withdraw(op_new, ask, 2.0)
    stock_remove(op_old, ask, 2.0)

    assert removed == pytest.approx(water, rel=1e-12)
    assert op_new.inlet.get_total_water_volume() == pytest.approx(0.0, abs=1e-9)
    assert np.all(op_new.inlet.get_depths() >= 0.0)
    # the same ledger the stock operator keeps (volume_balance reads these)
    assert op_new.total_requested_volume == pytest.approx(op_old.total_requested_volume)
    assert op_new.total_requested_volume == pytest.approx(-ask)
    assert op_new.get_total_applied_volume() == pytest.approx(op_old.get_total_applied_volume())
    assert op_new.get_total_applied_volume() == pytest.approx(-water)
    assert op_new.applied_Q == pytest.approx(op_old.applied_Q) == pytest.approx(-water / 2.0)
    assert (dom_new.fractional_step_volume_integral - i0_new
            == pytest.approx(dom_old.fractional_step_volume_integral - i0_old))
    assert np.all(op_new.inlet.get_xmoms() == 0.0) and np.all(op_new.inlet.get_ymoms() == 0.0)


def test_partial_withdrawal_ledger_matches_the_stock_operator(tmp_path):
    dom_new, op_new = sloped_domain(tmp_path / "new")
    dom_old, op_old = sloped_domain(tmp_path / "stock")
    volume = 0.4 * op_new.inlet.get_total_water_volume()
    withdraw(op_new, volume, 2.0)
    stock_remove(op_old, volume, 2.0)
    for attr in ("total_requested_volume", "total_applied_volume", "applied_Q"):
        assert getattr(op_new, attr) == pytest.approx(getattr(op_old, attr), rel=1e-9), attr
    assert dom_new.fractional_step_volume_integral == pytest.approx(
        dom_old.fractional_step_volume_integral, rel=1e-9)


def test_momentum_keeps_each_cells_velocity_and_dries_to_zero(tmp_path):
    dom_new, op_new = sloped_domain(tmp_path / "new", xmom=0.8)
    dom_old, op_old = sloped_domain(tmp_path / "stock", xmom=0.8)
    inlet = op_new.inlet
    z = inlet.get_elevations()
    volume = 0.6 * inlet.get_total_water_volume()
    withdraw(op_new, volume, 2.0)
    depth = inlet.get_depths()
    assert np.all(depth >= 0.0)
    wet, dry = depth > 1e-12, depth <= 1e-12
    assert wet.any() and dry.any()
    np.testing.assert_allclose(inlet.get_xmoms()[wet] / depth[wet], 0.8, rtol=1e-12)
    np.testing.assert_allclose(inlet.get_ymoms()[wet] / depth[wet], -0.4, rtol=1e-12)
    assert np.all(inlet.get_xmoms()[dry] == 0.0) and np.all(inlet.get_ymoms()[dry] == 0.0)
    # negative control: stock gives the dry-bank cells (bed >= 8.8) water AND momentum
    stock_remove(op_old, volume, 2.0)
    bank = z >= CHANNEL_LEVEL
    assert np.all(op_old.inlet.get_depths()[bank] > 0.0)


# ------------------------------------------------------------------ the inertia tolerance
def test_inertia_floor_runs_only_up_to_a_small_adverse_head():
    assert INERTIA_MAX_ADVERSE_HEAD_M == 0.10
    q_swmm, q, dt = 140.0, 160.0, 0.5
    assert inertia_floor(q_swmm, q, dt, adverse_head=-0.3) == pytest.approx(70.0)  # with head
    assert inertia_floor(q_swmm, q, dt, adverse_head=0.0) == pytest.approx(70.0)
    assert inertia_floor(q_swmm, q, dt, adverse_head=0.09) == pytest.approx(70.0)  # tolerance
    assert inertia_floor(q_swmm, q, dt, adverse_head=0.11) == 0.0
    assert inertia_floor(q_swmm, q, dt, adverse_head=0.24) == 0.0   # the ND surge's first step
    assert inertia_floor(q_swmm, q, dt, adverse_head=0.79) == 0.0   # the Towradgi ND surge
    assert inertia_floor(-q_swmm, q, dt, adverse_head=-1.0) == 0.0  # SWMM runs the other way
    assert inertia_floor(q_swmm, q, dt, adverse_head=0.0, strict=True) == 0.0
    # negative control: without the head limit (the 1400a90 floor) 0.79 m adverse moves it all
    assert inertia_floor(q_swmm, q, dt, adverse_head=0.79,
                         max_adverse_head=None) == pytest.approx(70.0)
