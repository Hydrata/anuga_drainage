"""TASK-3673: reverse flow through the SWMM barrel (swmm_engine.ReverseBarrel).

Steady reverse point of segment E (HW 0.5 D, TW 2.0 D: the barrel exits through J_HW at a
brink, the drowned entering end is O_TW) on all 42 entrance x geometry cases, through
SwmmHeadwallEngine + InletControl (the coupler's per-stride arithmetic), against the mirrored
HDS-5 oracle. ``reverse_losses="off"`` (SWMM as written) is the negative control: it must
fail the same 2 % gate, else the gate proves nothing.
"""
import os
import sys
import tempfile

import numpy as np
import pytest

pytest.importorskip("pyswmm")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import _culvert_unsteady_oracle as u  # noqa: E402

from anuga_drainage.culverts import ENTRANCE_TYPES, InletControl  # noqa: E402
from anuga_drainage.culverts.swmm_engine import ReverseBarrel, SwmmHeadwallEngine  # noqa: E402

CASES = [(e, g) for g in u.GEOMETRIES for e in ENTRANCE_TYPES]


def steady(e, geom, hw_d, tw_d, mode, strides=600):
    L, S = u.GEOMETRIES[geom]
    c = u.culvert_for(e, L, S)
    control = InletControl([c])
    hw, tw = u.INV + hw_d * u.D, c.downstream_invert_m + tw_d * u.D
    cap_f, cap_r = control.caps([hw], [tw])
    qs = []
    with tempfile.TemporaryDirectory() as d:
        eng = SwmmHeadwallEngine([c], d, duration_s=strides + 5, initial_stages=[(hw, tw)],
                                 reverse_losses=mode)
        for _ in range(strides):
            eng.set_stages([hw], [tw])
            eng.stride()
            qs.append(control.apply([eng.flows()[0]], cap_f, cap_r)[0][0])
        eng.close()
    q = float(np.mean(qs[-60:]))
    ref = u.QuasiSteady(e, L, S).point(hw_d * u.D, tw_d * u.D)[0]
    return q, ref, float(np.std(qs[-60:]))


@pytest.fixture(scope="module")
def reverse_points():
    return {(e.hds5, g, mode): steady(e, g, 0.5, 2.0, mode)
            for e, g in CASES for mode in ("hds5", "off")}


def test_reverse_plateau_within_2pct_all_42(reverse_points):
    bad = []
    for e, g in CASES:
        q, ref, std = reverse_points[(e.hds5, g, "hds5")]
        err = 100 * (q / ref - 1)
        if abs(err) > 2.0 or std > 1e-3 * abs(ref):
            bad.append(f"{e.hds5}@{g}: {err:+.2f} % (std {std:.1e})")
    assert bad == [], "\n".join(bad)


def test_reverse_gate_fails_swmm_as_written(reverse_points):
    """NEGATIVE CONTROL: SWMM's direction-blind losses and missing brink fail the gate."""
    errs = []
    for e, g in CASES:
        q, ref, _std = reverse_points[(e.hds5, g, "off")]
        errs.append(100 * (q / ref - 1))
    assert sum(abs(x) > 2.0 for x in errs) >= 20, errs


@pytest.mark.parametrize("key", ["rcp_groove_hw", "box_wing45_bev", "cmbox_thinproj"])
def test_reverse_drowned_exit_unchanged(key):
    """HW 1.1 D: the exit is drowned above dc, so the brink floor never binds and the
    barrel is full: the fix must leave SWMM's (already exact) answer alone."""
    e = [t for t in ENTRANCE_TYPES if t.hds5 == key][0]
    q_fix, ref, _ = steady(e, "20m_S2", 1.1, 2.0, "hds5")
    q_off, _, _ = steady(e, "20m_S2", 1.1, 2.0, "off")
    assert q_fix == pytest.approx(q_off, rel=1e-3)
    assert q_fix == pytest.approx(ref, rel=0.005)


@pytest.mark.parametrize("hw_d,tw_d", [(1.5, 0.0), (2.0, 1.8)])
def test_forward_flow_bit_identical(hw_d, tw_d):
    e = [t for t in ENTRANCE_TYPES if t.hds5 == "box_wing45_bev"][0]
    assert steady(e, "20m_S2", hw_d, tw_d, "hds5")[0] == steady(e, "20m_S2", hw_d, tw_d, "off")[0]


class _Link:
    def __init__(self):
        self.inlet_head_loss, self.outlet_head_loss, self.average_head_loss = 0.5, 1.0, 0.0


class _Node:
    depth = 0.3


def test_direction_hysteresis_and_restore():
    e = [t for t in ENTRANCE_TYPES if t.hds5 == "rcp_sq_hw"][0]
    c = u.culvert_for(e, 20.0, 0.02)
    link = _Link()
    rb = ReverseBarrel(c, link)
    tw = c.downstream_invert_m + 1.8
    rb.update(-0.5 * rb.eps, _Node(), tw)                 # inside the deadband: still forward
    assert not rb.reverse and (link.inlet_head_loss, link.outlet_head_loss) == (0.5, 1.0)
    assert rb.floor_hw(10.1) == 10.1
    rb.update(-2 * rb.eps, _Node(), tw)
    assert rb.reverse and link.inlet_head_loss > 1.0 and link.outlet_head_loss < 0.5
    assert rb.floor_hw(c.upstream_invert_m) > c.upstream_invert_m      # the brink floor binds
    rb.update(0.5 * rb.eps, _Node(), tw)                  # back inside the deadband: stays reverse
    assert rb.reverse
    rb.update(2 * rb.eps, _Node(), tw)
    assert not rb.reverse
    assert (link.inlet_head_loss, link.outlet_head_loss, link.average_head_loss) == (0.5, 1.0, 0.0)


def test_brink_depth_table_matches_bisection():
    for key in ("rcp_sq_hw", "box_wing0"):
        e = [t for t in ENTRANCE_TYPES if t.hds5 == key][0]
        rb = ReverseBarrel(u.culvert_for(e, 20.0, 0.02), _Link())
        for q in (0.05, 0.4, 1.3, 2.9):
            assert rb.critical_depth(q) == pytest.approx(rb.geom.critical_depth(q), abs=2e-3)


def test_sigma_mirrors_the_written_damping_option():
    """The convective weight in Kavg must be the one SWMM applies under the option written
    to the .inp (dwflow.c:191-208): 1 for NONE, 0 for FULL, the Froude ramp for PARTIAL."""
    e = [t for t in ENTRANCE_TYPES if t.hds5 == "rcp_sq_hw"][0]
    c = u.culvert_for(e, 20.0, 0.02)
    sig = {d: ReverseBarrel(c, _Link(), damping=d).sigma for d in ("NONE", "PARTIAL", "FULL", "none")}
    assert [sig["NONE"](fr) for fr in (0.2, 0.7, 1.2)] == [1.0, 1.0, 1.0]
    assert [sig["none"](fr) for fr in (0.2, 0.7, 1.2)] == [1.0, 1.0, 1.0]
    assert [sig["FULL"](fr) for fr in (0.2, 0.7, 1.2)] == [0.0, 0.0, 0.0]
    assert [sig["PARTIAL"](fr) for fr in (0.2, 0.75, 1.2)] == [1.0, 0.5, 0.0]
    with pytest.raises(ValueError):
        ReverseBarrel(c, _Link(), damping="SOME")
