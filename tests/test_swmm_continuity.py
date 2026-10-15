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
import re
import tempfile

import pytest

pytest.importorskip("pyswmm")

from pyswmm import Links, SystemStats  # noqa: E402

from anuga_drainage.culverts import HeadwallCulvert  # noqa: E402
from anuga_drainage.culverts.swmm_engine import SwmmHeadwallEngine, mass_continuity  # noqa: E402

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


# ---------------------------------------------------------------- TASK-3741: judged by mass
# Operator ruling D36 (gate 637): the D9 "< 1 %" continuity check reads the MASS-based figure
# (SwmmHeadwallEngine.mass_balance["mass_error_pct"]), which counts the outfall-end storage as
# water that came in through the outfall, not SWMM's raw % above.

def _run_mass(tailwater, initial_tw=INV_DN, seepage_mm_per_hr=0.0, monkeypatch=None):
    """As ``_run`` but returns the engine's own ledger (``mass_balance``, set by close)."""
    c = _culvert()
    if seepage_mm_per_hr:
        # negative control: a GENUINE loss, water seeping out through the barrel walls. SWMM
        # books it (Exfiltration Loss), so its own % does not see it as an error; it reaches
        # neither end, so the mass figure must.
        from anuga_drainage.culverts import swmm_engine

        real = swmm_engine.write_inp

        def leaky(*a, **kw):
            layout = real(*a, **kw)
            with open(layout.path) as f:
                text = f.read()
            text = re.sub(r"^(CUL0\s+.*\sNO) 0$", rf"\1 {seepage_mm_per_hr}", text, flags=re.M)
            with open(layout.path, "w") as f:
                f.write(text)
            return layout

        monkeypatch.setattr(swmm_engine, "write_inp", leaky)
    with tempfile.TemporaryDirectory() as d:
        eng = SwmmHeadwallEngine([c], d, duration_s=STRIDES + 10,
                                 initial_stages=[(INV_UP, initial_tw)])
        for k in range(STRIDES):
            eng.set_stages([INV_UP + 0.12], [tailwater(k / STRIDES)])
            eng.stride()
        err = eng.close()
        mb = dict(eng.mass_balance)
    mb["rpt_pct"] = err
    return mb


def test_mass_continuity_counts_the_outlet_end_water_under_a_risen_tailwater():
    """AC1 (TASK-3741): run 1590's risen tailwater. SWMM's raw % is far outside 1 %, the mass
    figure (the outfall-end water counted as having come in through the outfall) within 0.5 %."""
    mb = _run_mass(lambda f: INV_DN + RISE_TW * f)
    assert mb["rpt_pct"] < -10.0, mb
    # the engine's raw figure IS SWMM's (rpt rounding: 3 decimals)
    assert mb["swmm_error_pct"] == pytest.approx(mb["rpt_pct"], abs=0.01), mb
    assert abs(mb["mass_error_pct"]) < 0.5, mb
    assert mb["final_outfall_end_m3"] > 100.0, mb
    assert mb["storage_units"] == "ft3", mb      # pyswmm 2.2: storage in ft3, flows in m3


@pytest.mark.parametrize("label, tailwater, initial_tw", [
    ("pinned at the invert", lambda f: INV_DN, INV_DN),
    ("rise and fall back", lambda f: INV_DN + RISE_TW * (2 * f if f < 0.5 else 2 * (1 - f)), INV_DN),
    ("started high, held", lambda f: INV_DN + RISE_TW, INV_DN + RISE_TW),
    ("started high, falling", lambda f: INV_DN + RISE_TW * (1 - f), INV_DN + RISE_TW),
])
def test_mass_continuity_within_half_a_percent_whatever_the_tailwater(label, tailwater, initial_tw):
    """The pinned tailwater reads about -2 % raw (the outfall halves at critical depth); a
    barrel that STARTS with water in its outlet end exercises the initial-wedge term."""
    mb = _run_mass(tailwater, initial_tw)
    assert abs(mb["mass_error_pct"]) < 0.5, (label, mb)


def test_mass_continuity_catches_a_genuine_loss(monkeypatch):
    """Negative control: inject a real loss (wall seepage) under the same risen tailwater. The
    mass figure moves by the lost volume over the inflow (water that came in and went nowhere
    reads as a POSITIVE error, SWMM's sign), so the outfall-end correction cannot hide it."""
    clean = _run_mass(lambda f: INV_DN + RISE_TW * f)
    leaky = _run_mass(lambda f: INV_DN + RISE_TW * f, seepage_mm_per_hr=200.0,
                      monkeypatch=monkeypatch)
    lost = leaky["losses_m3"]
    assert lost > 0.02 * leaky["inflow_m3"], leaky
    assert leaky["mass_error_pct"] > 1.0, leaky
    assert leaky["mass_error_pct"] == pytest.approx(
        clean["mass_error_pct"] + 100.0 * lost / leaky["denominator_m3"], abs=0.5), (clean, leaky)


def test_mass_continuity_ledger_is_the_volumes_it_names():
    """The pure ledger (m3; pyswmm storage in ft3): the outfall-end gain is counted in, a loss
    of any kind (unbooked, or booked by SWMM as seepage/evaporation/flooding) counts against."""
    def stats(outflow, seepage=0.0):
        return dict(dry_weather_inflow=0.0, wet_weather_inflow=0.0, groundwater_inflow=0.0,
                    II_inflow=0.0, external_inflow=1000.0, flooding=0.0, outflow=outflow,
                    evaporation_loss=0.0, seepage_loss=seepage, reacted=0.0, initial_storage=0.0,
                    final_storage=350.0 / FT3_TO_M3, routing_error=0.0)

    wedge = mass_continuity(stats(800.0), 0.0, 150.0)      # 150 of the 350 m3 stored: outfall end
    assert wedge["swmm_error_pct"] == pytest.approx(-15.0)
    assert wedge["mass_error_pct"] == pytest.approx(0.0, abs=1e-9)
    leak = mass_continuity(stats(750.0), 0.0, 150.0)       # and 50 m3 that went nowhere
    assert leak["mass_error_pct"] == pytest.approx(5.0)
    booked = mass_continuity(stats(750.0, seepage=50.0), 0.0, 150.0)
    assert booked["swmm_error_pct"] == pytest.approx(-15.0)  # SWMM books the seepage: not its error
    assert booked["mass_error_pct"] == pytest.approx(5.0)    # but it reached neither end
    assert booked["losses_m3"] == pytest.approx(50.0)
    assert mass_continuity(dict(stats(0.0), external_inflow=0.0, final_storage=0.0), 0.0, 0.0)[
        "mass_error_pct"] is None                            # nothing entered: no figure
    # a pyswmm that reports storage in m3 is recognised by SWMM's own error
    in_m3 = mass_continuity(dict(stats(800.0), final_storage=350.0, routing_error=-15.0), 0.0, 150.0)
    assert in_m3["storage_units"] == "m3" and in_m3["mass_error_pct"] == pytest.approx(0.0, abs=1e-9)
