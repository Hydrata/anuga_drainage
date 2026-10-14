"""TASK-3691: the pressurised barrel (pond AND tailwater at or over the crown, the D28 regime).

The 2D flume case (box 2.4 x 1.5 m, L 8 m, S 1.25 %, square-edge headwall, TW = D, Q 7) settled
0.10 m BELOW HDS-5: once the pond fell below y_full = rise, the entrance-energy depth handed SWMM a
free-surface inlet below the crown next to a full outlet, SWMM held a spurious supercritical barrel
(12.98 m3/s) and the HDS-5 inlet cap (7.0) governed instead of the outlet-control barrel.

1D here (SwmmHeadwallEngine + InletControl, the coupler's per-stride arithmetic): the barrel is
first driven forward with the pond high and the outlet free (the 2D rising limb), then held at a
D28 point (pond 1.08 D, tailwater depth 1.007 D). The shipped engine must land on the HDS-5
outlet-control flow; the negative control (the pressurised state disabled = the 825dc0f engine)
must stay on the supercritical branch, else the gate proves nothing.
"""
import os
import sys
import tempfile

import numpy as np
import pytest

pytest.importorskip("pyswmm")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import _hds5_oracle as hds5  # noqa: E402

from anuga_drainage.culverts import HeadwallCulvert, InletControl  # noqa: E402
from anuga_drainage.culverts import swmm_engine  # noqa: E402

RISE, SPAN, L, DROP = 1.5, 2.4, 8.0, 0.1
CULVERT = dict(name="d28", coordinates=[[0.0, 0.0], [L, 0.0]], upstream_invert_m=0.0,
               downstream_invert_m=-DROP, shape="box", width_m=SPAN, height_m=RISE,
               entrance_type="square_edge_headwall", material="concrete", barrels=1)
#: the 2D trace's rising limb (pond over-shoots, outlet not yet drowned), then the D28 hold
HISTORY = [(1.85, 1.30)] * 60
HOLD = (1.6248, 1.41)          # stages: pond 1.08 D, tailwater depth 1.51 m = 1.007 D


def _run(pressurised=True, strides=600):
    c = HeadwallCulvert.from_dict(CULVERT)
    control = InletControl([c])
    qs, raw = [], []
    with tempfile.TemporaryDirectory() as d:
        eng = swmm_engine.SwmmHeadwallEngine([c], d, duration_s=len(HISTORY) + strides + 5,
                                             initial_stages=[HISTORY[0]])
        if not pressurised:
            eng._pressurised = lambda *a: False
        for hw, tw in HISTORY + [HOLD] * strides:
            cap_f, cap_r = control.caps([hw], [tw])
            eng.set_stages([hw], [tw])
            eng.stride()
            q = float(eng.flows()[0])
            raw.append(q)
            qs.append(control.apply([q], cap_f, cap_r)[0][0])
        eng.close()
    return float(np.mean(qs[-60:])), float(np.mean(raw[-60:])), float(np.std(qs[-60:]))


def _reference():
    c = HeadwallCulvert.from_dict(CULVERT)
    ref = hds5.Culvert("box", RISE, B=SPAN, L=c.length_m, S=c.slope, n=c.n, inlet=c.entrance.hds5)
    hw, tw = HOLD
    return ref.q(hw - c.upstream_invert_m, tw - c.downstream_invert_m)


def test_pressurised_barrel_lands_on_hds5_outlet_control():
    """MEASURED: +1.8 % (SWMM full pipe with the stock losses vs hds5 at the same stages)."""
    q, raw, sd = _run()
    ref = _reference()
    assert abs(q / ref - 1.0) <= 0.03, (q, raw, ref)
    assert abs(raw / ref - 1.0) <= 0.03, (q, raw, ref)      # the barrel, not the cap, governs
    assert sd <= 0.01 * ref, sd


def test_negative_control_without_the_pressurised_state_runs_supercritical():
    """NEGATIVE CONTROL: the 825dc0f engine (entrance-energy depth below the crown) holds SWMM's
    supercritical branch (raw barrel flow well over the reference) and the inlet cap governs."""
    q, raw, _ = _run(pressurised=False)
    ref = _reference()
    assert raw > 1.5 * ref, (q, raw, ref)
    assert q > 1.05 * ref, (q, raw, ref)
