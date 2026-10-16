"""TASK-3771 anti-oscillation: one coupling step never moves more than a fraction of the
volume that would bring the two exchange regions level, and nothing against the head."""
import math

import numpy as np
import pytest

from anuga_drainage.culverts.coupler import (EQUALISATION_FACTOR, LevelFill, equalising_volume,
                                             joint_fill, limit_transfer)


def brute_equalising(src, dst, v_src, v_dst):
    if src.stage(v_src) <= dst.stage(v_dst):
        return 0.0
    if src.stage(0.0) >= dst.stage(v_dst + v_src):
        return v_src
    lo, hi = 0.0, v_src
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if src.stage(v_src - mid) > dst.stage(v_dst + mid):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ------------------------------------------------------------------ LevelFill.volume
def test_volume_is_the_inverse_of_stage():
    rng = np.random.default_rng(1)
    fill = LevelFill(rng.uniform(0, 5, 50), rng.uniform(1, 9, 50))
    for v in np.linspace(0.0, 3000.0, 61):
        assert fill.volume(fill.stage(v)) == pytest.approx(v, abs=1e-8)
    assert fill.volume(fill.z[0] - 1.0) == 0.0


# ------------------------------------------------------------------ closed-form cases
def test_two_flat_equal_tanks_split_the_difference():
    a, b = LevelFill([0.0], [100.0]), LevelFill([0.0], [100.0])
    # 2 m vs 1 m of water: level at 1.5 m, so 50 m3 moves
    assert equalising_volume(a, b, joint_fill(a, b), 200.0, 100.0) == pytest.approx(50.0)


def test_flat_tanks_of_different_area_and_bed():
    a, b = LevelFill([1.0], [100.0]), LevelFill([0.0], [300.0])
    va, vb = 300.0, 300.0           # stages 4.0 and 1.0; common level L: 100(L-1)+300L = 600
    level = 700.0 / 400.0
    want = va - 100.0 * (level - 1.0)
    assert equalising_volume(a, b, joint_fill(a, b), va, vb) == pytest.approx(want)


def test_source_cannot_reach_down_to_the_other_level_gives_everything():
    a, b = LevelFill([5.0], [10.0]), LevelFill([0.0], [10.0])  # src perched 5 m above a dry dst
    assert equalising_volume(a, b, joint_fill(a, b), 7.0, 0.0) == pytest.approx(7.0)


def test_against_the_head_is_zero_and_level_is_zero():
    a, b = LevelFill([0.0], [100.0]), LevelFill([0.0], [100.0])
    j = joint_fill(a, b)
    assert equalising_volume(a, b, j, 100.0, 200.0) == 0.0     # src lower than dst
    assert equalising_volume(a, b, j, 150.0, 150.0) == pytest.approx(0.0, abs=1e-9)
    assert equalising_volume(a, b, j, 0.0, 0.0) == 0.0


def test_matches_bisection_on_random_ground():
    rng = np.random.default_rng(3771)
    for _ in range(40):
        a = LevelFill(rng.uniform(0, 4, 30), rng.uniform(1, 20, 30))
        b = LevelFill(rng.uniform(-1, 3, 25), rng.uniform(1, 20, 25))
        j = joint_fill(a, b)
        va, vb = rng.uniform(0, 800), rng.uniform(0, 800)
        got = equalising_volume(a, b, j, va, vb)
        assert got == pytest.approx(brute_equalising(a, b, va, vb), abs=1e-7)
        if 0.0 < got < va:
            assert a.stage(va - got) == pytest.approx(b.stage(vb + got), abs=1e-9)


# ------------------------------------------------------------------ limiter + guard
def test_limit_transfer_caps_at_the_factor_and_zeros_against_head():
    assert limit_transfer(10.0, 100.0, 0.5) == 10.0
    assert limit_transfer(80.0, 100.0, 0.5) == 50.0
    assert limit_transfer(80.0, 0.0, 0.5) == 0.0      # opposes the head, SWMM does not: nothing
    assert limit_transfer(80.0, 0.0, 0.5, 30.0) == 30.0   # SWMM's own inertial flow only
    assert limit_transfer(20.0, 0.0, 0.5, 30.0) == 20.0
    assert limit_transfer(80.0, 100.0, 0.5, 70.0) == 70.0  # with the head: SWMM's flow is a floor
    assert limit_transfer(80.0, 100.0, 0.5, 10.0) == 50.0
    assert limit_transfer(80.0, 0.0, None) == 80.0    # cap disabled: the pre-3771 behaviour
    assert 0.0 < EQUALISATION_FACTOR <= 1.0


# ------------------------------------------------------------------ toy two tanks + orifice
def two_tanks(factor, tau=0.0, strict=False, t_end=900.0, dt=0.1, area=260.0, cda=60.0, k=40.0):
    """The coupler's loop in miniature: two exchange regions on sloping ground, each tied to
    the rest of the domain by a linear link (reservoirs at 5 m and 3 m); an orifice flow (no
    inertia) is computed on the stages once per 1 s 'SWMM stride' and HELD, optionally
    smoothed over ``tau``, then moved every ``dt`` with the available clip and (``factor``) the
    cap, whose inertia floor is the held 'SWMM' flow unless ``strict``."""
    a = np.full(20, area / 20)
    up, dn = LevelFill(np.linspace(2.0, 2.6, 20), a), LevelFill(np.linspace(1.0, 1.6, 20), a)
    joint = joint_fill(up, dn)
    vu, vd = up.volume(3.5), dn.volume(3.5)
    q_held = q_used = 0.0
    next_stride, t, rows = 0.0, 0.0, []
    while t < t_end:
        hu, hd = up.stage(vu), dn.stage(vd)
        if t + dt >= next_stride - 1e-9:
            dh = hu - hd
            q_held = cda * math.copysign(math.sqrt(2 * 9.81 * abs(dh)), dh)
            next_stride += 1.0
        q_used = q_used + min(1.0, dt / tau) * (q_held - q_used) if tau > 0 else q_held
        vu = max(vu + k * (5.0 - hu) * dt, 0.0)
        vd = max(vd + k * (3.0 - hd) * dt, 0.0)
        src, dst, vs, vt = (up, dn, vu, vd) if q_used > 0 else (dn, up, vd, vu)
        want = min(abs(q_used) * dt, vs)
        if factor is not None:
            inertia = abs(q_held) * dt if (q_held * q_used > 0 and not strict) else 0.0
            want = limit_transfer(want, equalising_volume(src, dst, joint, vs, vt), factor, inertia)
        if q_used > 0:
            vu, vd = vu - want, vd + want
        else:
            vu, vd = vu + want, vd - want
        t += dt
        rows.append((t, math.copysign(want / dt, q_used), hu - hd))
    return np.array(rows)


def settled(rows, after=450.0):
    late = rows[rows[:, 0] > after]
    q = late[:, 1]
    flips = int(np.sum(np.diff(np.sign(q[np.abs(q) > 1e-9])) != 0))
    return flips, float(q.max() - q.min()), float(q.mean())


#: the steady flow the reservoirs drive with no lag at all (smoothing 30 s settles to it)
FREE_Q = settled(two_tanks(None, 30.0))[2]


@pytest.mark.parametrize("tau", [0.0, 5.0])
def test_toy_oscillates_without_the_cap(tau):
    """NEGATIVE CONTROL: the same loop without the cap sloshes (flow reverses, swings > 50 m3/s)."""
    flips, swing, _ = settled(two_tanks(None, tau))
    assert swing > 50.0
    if tau == 0.0:
        assert flips > 100


@pytest.mark.parametrize("tau", [0.0, 5.0, 30.0])
def test_toy_settles_with_the_strict_cap_and_is_not_starved(tau):
    flips, swing, mean = settled(two_tanks(EQUALISATION_FACTOR, tau, strict=True))
    assert flips == 0
    assert swing < 1.0
    assert mean == pytest.approx(FREE_Q, rel=0.02) and mean > 0


def test_toy_default_cap_removes_the_smoothing_slosh():
    """The default cap (SWMM's own flow is a floor) takes out what smoothing adds..."""
    _, swing_off, _ = settled(two_tanks(None, 5.0))
    flips, swing, mean = settled(two_tanks(EQUALISATION_FACTOR, 5.0))
    assert flips == 0 and swing < swing_off / 5.0
    assert mean == pytest.approx(FREE_Q, rel=0.02)


def test_toy_default_cap_leaves_a_lagging_engine_flow_alone():
    """...but not a slosh in the engine's OWN (held) flow: only the strict cap removes that."""
    np.testing.assert_array_equal(two_tanks(EQUALISATION_FACTOR, 0.0), two_tanks(None, 0.0))
