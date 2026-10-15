"""TASK-3717: what SWMM's flow-routing continuity error means for a headwall culvert.

Prod run 1590 (hydrata.com) reported "Continuity Error (%) -22.166" for a 2.0 x 1.5 m box,
256 m, inverts 8.710 -> 7.205, whose outlet pond rose 0.64 m over the outlet invert by the
end of the 2 h run. Replayed SWMM-only from the run's recorded stages it reads -22.161 %.

The cause is SWMM's bookkeeping at the two FIXED-stage outfalls the coupler drives
(O_HW, O_TW), not water lost or created: under dynamic wave a conduit's volume is
built from its end depths, half to each end node, and an outfall does no continuity, so the
outfall half of each conduit fills (or empties) with the IMPOSED stage without any booked
inflow. Water standing in the outlet end of the barrel when the run stops is counted as
"Final Stored Volume" that no inflow delivered, i.e. a negative continuity error of about
-(outfall-half volume) / inflow. It is transient: when the tailwater falls back the wedge
leaves the same way, with no booked outflow. ANUGA never sees this volume (the coupler moves
exactly the water the losing exchange region gives up, coupler.py ``_exchange``), so the
flood model's mass is untouched. run_anuga reports the figure with a plain-English note
(culverts/metrics.json ``swmm_continuity``).

These tests pin that behaviour on the shipped engine: if a future SWMM or engine change
books the outfall-end storage, the first test fails and the note must be revisited.
"""
import tempfile

import pytest

pytest.importorskip("pyswmm")

from pyswmm import Links, SystemStats  # noqa: E402

from anuga_drainage.culverts import HeadwallCulvert  # noqa: E402
from anuga_drainage.culverts.swmm_engine import SwmmHeadwallEngine  # noqa: E402

FT3_TO_M3 = 0.028316846592  # pyswmm's routing_stats report storage in ft3 whatever FLOW_UNITS
INV_UP, INV_DN, RISE_TW = 8.71, 7.205, 0.64  # run 1590's "Pond outlet culvert"
STRIDES = 1200


def _culvert():
    return HeadwallCulvert.from_dict(dict(
        name="run 1590 pond outlet", shape="box", width_m=2.0, height_m=1.5,
        coordinates=[(0.0, 0.0), (256.0, 0.0)], upstream_invert_m=INV_UP,
        downstream_invert_m=INV_DN,
        inlet_region=[(-30, -30), (0, -30), (0, 30), (-30, 30)],
        outlet_region=[(256, -30), (286, -30), (286, 30), (256, 30)]))


def _run(tailwater):
    """Headwater 0.12 m over the inlet invert (run 1590's end state), the tailwater stage
    from ``tailwater(fraction of the run)``; 1 s strides. Returns the ledger in m3."""
    c = _culvert()
    with tempfile.TemporaryDirectory() as d:
        eng = SwmmHeadwallEngine([c], d, duration_s=STRIDES + 10,
                                 initial_stages=[(INV_UP, INV_DN)])
        for k in range(STRIDES):
            eng.set_stages([INV_UP + 0.12], [tailwater(k / STRIDES)])
            eng.stride()
        st = SystemStats(eng.sim).routing_stats
        links = Links(eng.sim)
        y_dn = eng._o_tw[0].depth
        # the outfall-end halves: the barrel's at O_TW, the dummy approach conduit's at O_HW
        wedge = c.width_m * y_dn * c.length_m / 2.0 + links["D_HW0"].volume / 2.0
        stored = (st["final_storage"] - st["initial_storage"]) * FT3_TO_M3
        inflow, outflow = st["external_inflow"], st["outflow"]
        err = eng.close()
    return dict(err_pct=err, inflow=inflow, outflow=outflow, stored=stored,
                residual=inflow - outflow - stored, wedge=wedge)


def test_continuity_error_is_the_outfall_end_storage_under_a_risen_tailwater():
    """Tailwater rising 0.64 m over the outlet invert (run 1590): a large negative SWMM error
    whose volume IS the outfall-end storage, within 10 %."""
    r = _run(lambda f: INV_DN + RISE_TW * f)
    assert r["err_pct"] < -10.0, r
    assert r["residual"] == pytest.approx(-r["wedge"], rel=0.10), r
    # SWMM's own percentage is that residual over the inflow (rpt rounding: 3 decimals)
    assert r["err_pct"] == pytest.approx(100.0 * r["residual"] / r["inflow"], abs=0.01), r


def test_continuity_wedge_leaves_again_when_the_tailwater_falls_back():
    """Negative control for 'water was created': the same 0.64 m rise, then back to the start
    stage. The ~165 m3 wedge is gone again (the residual matches a run whose tailwater never
    rose), so nothing was added to or taken from the water ANUGA moves."""
    risen = _run(lambda f: INV_DN + RISE_TW * f)
    back = _run(lambda f: INV_DN + RISE_TW * (2 * f if f < 0.5 else 2 * (1 - f)))
    flat = _run(lambda f: INV_DN)
    assert risen["wedge"] > 4 * flat["wedge"], (risen, flat)
    assert back["residual"] == pytest.approx(flat["residual"], abs=0.05 * abs(risen["residual"])), (back, flat)
