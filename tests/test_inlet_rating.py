"""TASK-3672: the in-package HDS-5 inlet-control rating (coupler-enforced inlet control).

``anuga_drainage.culverts.inlet_control`` re-implements the HDS-5 inlet equations
without scipy; the vendored independent oracle (_hds5_oracle.py) is the reference.
"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import _hds5_oracle as hds5  # noqa: E402

from anuga_drainage.culverts import ENTRANCE_TYPES, HeadwallCulvert  # noqa: E402
from anuga_drainage.culverts import inlet_control as ic  # noqa: E402


def _culvert(e, L=20.0, S=0.02, barrels=1, blockage=0.0):
    d = dict(name="r", shape=e.shape, coordinates=[[0.0, 0.0], [L, 0.0]],
             upstream_invert_m=10.0, downstream_invert_m=10.0 - S * L,
             entrance_type=e.value, material=e.material, barrels=barrels,
             blockage_pct=blockage)
    if e.shape == "circular":
        d["diameter_m"] = 0.9
    else:
        d.update(width_m=1.5, height_m=0.9)
    return HeadwallCulvert.from_dict(d)


def _ref(e, c, S):
    shape, g1, g2 = c.effective_geometry
    return hds5.Culvert("circ" if shape == "CIRCULAR" else "box", g1,
                        B=None if shape == "CIRCULAR" else g2, L=c.length_m, S=S, n=c.n,
                        inlet=e.hds5)


@pytest.mark.parametrize("e", ENTRANCE_TYPES, ids=lambda e: f"{e.hds5}")
def test_rating_matches_oracle_inlet_equations(e):
    """Q_inlet(head) inverts the oracle's inlet_hw to 0.1 % over 0.05..8 D, both directions."""
    for S in (0.02, 0.005):
        c = _culvert(e, S=S)
        fwd, rev = ic.ratings_for(c)
        for rating, slope in ((fwd, S), (rev, -S)):
            ref = _ref(e, c, slope)
            for x in (0.05, 0.1, 0.3, 0.5, 0.8, 0.95, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 8.0):
                head = x * 0.9
                q = rating(head)
                assert q > 0
                back = hds5.inlet_hw(q, ref.geom, ref.inlet, slope)
                assert back == pytest.approx(head, rel=1e-3)


def test_rating_zero_head_zero_flow_and_monotone():
    e = ENTRANCE_TYPES[0]
    fwd, _ = ic.ratings_for(_culvert(e))
    assert fwd(0.0) == 0.0 and fwd(-1.0) == 0.0
    qs = [fwd(0.9 * k / 100) for k in range(0, 1000)]
    assert all(b >= a for a, b in zip(qs, qs[1:]))


def test_rating_blockage_uses_effective_geometry_and_cache():
    e = [t for t in ENTRANCE_TYPES if t.hds5 == "box_sq_90hw"][0]
    open_c, blocked = _culvert(e), _culvert(e, blockage=40.0)
    assert ic.ratings_for(blocked)[0](1.5) < 0.7 * ic.ratings_for(open_c)[0](1.5)
    assert ic.ratings_for(open_c)[0] is ic.ratings_for(_culvert(e))[0]      # cached


def test_inlet_control_caps_and_flags():
    e = ENTRANCE_TYPES[0]
    c = _culvert(e, barrels=3)
    ctl = ic.InletControl([c])
    fwd1 = ic.ratings_for(c)[0](0.9)
    cap_f, cap_r = ctl.caps([c.upstream_invert_m + 0.9], [c.downstream_invert_m])
    assert cap_f[0] == pytest.approx(3 * fwd1) and cap_r[0] == 0.0
    q, flags = ctl.apply([10.0], cap_f, cap_r)
    assert q == [pytest.approx(3 * fwd1)] and flags == [ic.INLET]
    q, flags = ctl.apply([0.5 * 3 * fwd1], cap_f, cap_r)
    assert q == [pytest.approx(0.5 * 3 * fwd1)] and flags == [ic.BARREL]
    # reverse: tailwater high, headwater dry
    cap_f, cap_r = ctl.caps([c.upstream_invert_m], [c.downstream_invert_m + 1.0])
    assert cap_f[0] == 0.0 and cap_r[0] > 0
    q, flags = ctl.apply([-100.0], cap_f, cap_r)
    assert q == [pytest.approx(-cap_r[0])] and flags == [ic.REVERSE_INLET]
    q, flags = ctl.apply([-0.001], cap_f, cap_r)
    assert q == [-0.001] and flags == [ic.BARREL]
