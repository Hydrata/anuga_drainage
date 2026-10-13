"""Write a SWMM 5.2 .inp for N headwall culverts (epic 3585, design 1).

Per culvert ``i`` (the headwall pattern that matched HDS-5 within 0.5 % in the
culvert grill, ``KEEP_dummy_headwall_rating.py``)::

    O_HW<i> (OUTFALL, FIXED stage := 2D headwater)
       ^ D_HW<i>  1 m dummy conduit, large section, n 0.010, no losses
    J_HW<i> (JUNCTION, invert = upstream invert)
       | CUL<i>   the culvert: shape, one-barrel geometry after blockage,
       v          Barrels, [LOSSES] Ke / exit 1.0; culvert code 0 (see below)
    O_TW<i> (OUTFALL, FIXED stage := 2D tailwater)

An outfall stage only drives flow when the outfall is the DOWNSTREAM node of
its link, hence the dummy conduit runs J_HW -> O_HW. Positive CUL<i> flow is
inlet -> outlet. Fully blocked culverts are left out (the coupler holds Q = 0).

Inlet control is NOT left to SWMM (mechanism C2, TASK-3672): SWMM 5.2.4's culvert
code skips the inlet equations once both barrel ends are full, returns a stale Form 1
flow out of findroot_Ridder, and applies a -7.0 mitered slope factor (HDS-5 gives
-0.7), so the conduit is written with culvert code 0 and the coupler caps its flow
at the HDS-5 inlet-control rating (see coupler.py). ``swmm_inlet_code=True`` writes
the HDS-5 culvert code instead (the SWMM-native route, used only for the C-level
proof in the deploy spike folder).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

DEFAULT_START = datetime(2020, 1, 1)
DUMMY_LENGTH_M = 1.0
DUMMY_N = 0.010
END_MARGIN_S = 2


@dataclass
class InpNames:
    """SWMM object names for culvert ``index``."""
    index: int

    @property
    def o_hw(self):
        return f"O_HW{self.index}"

    @property
    def j_hw(self):
        return f"J_HW{self.index}"

    @property
    def d_hw(self):
        return f"D_HW{self.index}"

    @property
    def conduit(self):
        return f"CUL{self.index}"

    @property
    def o_tw(self):
        return f"O_TW{self.index}"


@dataclass
class InpLayout:
    path: str
    names: dict = field(default_factory=dict)   # culvert index -> InpNames (open culverts only)


def _hms(seconds: int) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def dummy_size(culvert) -> tuple:
    """(height, width) of the 1 m dummy approach conduit.

    Far larger in SECTION than the barrel group, so its velocity head and
    friction are negligible (a footprint-sized dummy made the rating unstable
    at low head: +102 % HW error on a 1.5 m CM box at HW/D 0.3), but SHORT,
    because its plan area is a storage tank between the 2D headwater and the
    culvert. MEASURED on the flume: a 5 m long 6 m x 10 m dummy raised the 2D
    peak headwater by 34 mm; the same section 1 m long, by 1.3 mm.
    """
    return max(6.0, 5.0 * culvert.rise_m), max(10.0, 2.0 * culvert.footprint_width_m)


def write_inp(culverts, path, *, duration_s=7 * 86400, routing_step=0.25,
              report_step_s=3600, start=DEFAULT_START, title="anuga_drainage headwall culverts",
              extra_options=None, initial_stages=None, swmm_inlet_code=False):
    """Write the .inp for ``culverts`` (HeadwallCulvert list) to ``path``.

    ``duration_s`` must cover the ANUGA run (SWMM stops at END_TIME).
    ``extra_options`` adds/overrides ``[OPTIONS]`` lines (later lines win in SWMM).
    ``initial_stages``: optional per-culvert (hw_stage, tw_stage) to start the two
    outfalls (and the junction depth) at, so the first imposed stage does not
    create barrel storage out of nothing (a SWMM continuity artefact).
    ``swmm_inlet_code``: write the entrance's SWMM culvert code (SWMM-native inlet
    control) instead of 0 (inlet control enforced by the coupler).
    Returns an :class:`InpLayout` mapping culvert index -> SWMM names.
    """
    if routing_step <= 0 or routing_step > 1:
        raise ValueError("routing_step must be in (0, 1] s (the coupler strides 1 s)")
    # pyswmm raises StopIteration on the stride that REACHES END_TIME, so keep a
    # margin: the engine refuses to stride past duration_s itself.
    end = start + timedelta(seconds=int(duration_s) + END_MARGIN_S)
    layout = InpLayout(path=str(path))
    junctions, outfalls, conduits, xsections, losses, coords = [], [], [], [], [], []
    for i, c in enumerate(culverts):
        if c.closed:
            continue
        n = InpNames(i)
        layout.names[i] = n
        dh, dw = dummy_size(c)
        shape, g1, g2 = c.effective_geometry
        e = c.entrance
        hw0, tw0 = c.upstream_invert_m, c.downstream_invert_m
        if initial_stages is not None:
            hw0 = max(hw0, float(initial_stages[i][0]))
            tw0 = max(tw0, float(initial_stages[i][1]))
        y0 = hw0 - c.upstream_invert_m
        junctions.append(f"{n.j_hw:<10} {c.upstream_invert_m:.4f} {dh:.3f} {y0:.4f} 1000 0")
        outfalls.append(f"{n.o_hw:<10} {c.upstream_invert_m:.4f} FIXED {hw0:.4f} NO")
        outfalls.append(f"{n.o_tw:<10} {c.downstream_invert_m:.4f} FIXED {tw0:.4f} NO")
        conduits.append(f"{n.d_hw:<10} {n.j_hw:<10} {n.o_hw:<10} {DUMMY_LENGTH_M:.2f} {DUMMY_N} 0 0 0 0")
        conduits.append(f"{n.conduit:<10} {n.j_hw:<10} {n.o_tw:<10} {c.length_m:.4f} {c.n:.4f} 0 0 0 0")
        xsections.append(f"{n.d_hw:<10} RECT_CLOSED {dh:.3f} {dw:.3f} 0 0 1")
        xsections.append(f"{n.conduit:<10} {shape} {g1:.5f} {g2:.5f} 0 0 {c.barrels} {e.swmm_code if swmm_inlet_code else 0}")
        losses.append(f"{n.conduit:<10} {e.ke} 1.0 0 NO 0")
        (xi, yi), (xo, yo) = c.inlet_xy, c.outlet_xy
        ux, uy = c.outward_direction("inlet")
        coords.append(f"{n.o_hw:<10} {xi + ux * DUMMY_LENGTH_M:.3f} {yi + uy * DUMMY_LENGTH_M:.3f}")
        coords.append(f"{n.j_hw:<10} {xi:.3f} {yi:.3f}")
        coords.append(f"{n.o_tw:<10} {xo:.3f} {yo:.3f}")
    if not layout.names:
        raise ValueError("no open culvert to write (all are 100 % blocked or the list is empty)")
    text = "\n".join([
        "[TITLE]", title, "",
        "[OPTIONS]",
        "FLOW_UNITS CMS",
        "FLOW_ROUTING DYNWAVE",
        "LINK_OFFSETS DEPTH",
        "ALLOW_PONDING NO",
        f"START_DATE {start:%m/%d/%Y}", f"START_TIME {start:%H:%M:%S}",
        f"REPORT_START_DATE {start:%m/%d/%Y}", f"REPORT_START_TIME {start:%H:%M:%S}",
        f"END_DATE {end:%m/%d/%Y}", f"END_TIME {end:%H:%M:%S}",
        f"REPORT_STEP {_hms(report_step_s)}",
        f"ROUTING_STEP {routing_step}",
        "VARIABLE_STEP 0",
        *[f"{k} {v}" for k, v in (extra_options or {}).items()],
        "", "[JUNCTIONS]", ";;Name Elev MaxDepth InitDepth SurDepth Aponded", *junctions,
        "", "[OUTFALLS]", ";;Name Elev Type Stage Gated", *outfalls,
        "", "[CONDUITS]", ";;Name From To Length Roughness InOffset OutOffset InitFlow MaxFlow",
        *conduits,
        "", "[XSECTIONS]", ";;Link Shape Geom1 Geom2 Geom3 Geom4 Barrels Culvert", *xsections,
        "", "[LOSSES]", ";;Link Kentry Kexit Kavg FlapGate Seepage", *losses,
        "", "[COORDINATES]", ";;Node X Y", *coords, "",
    ])
    with open(path, "w") as f:
        f.write(text)
    return layout
