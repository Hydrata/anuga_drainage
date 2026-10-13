"""SWMM side of the headwall culvert coupler: one pyswmm Simulation for N culverts.

pyswmm strides whole seconds only (``step_advance(0.5)`` raises TypeError), so
the engine advances in 1 s strides; the coupler holds Q between strides.
"""
from __future__ import annotations

import bisect
import math
import os
import re

import numpy as np

from .inlet_control import _geometry
from .inp_writer import write_inp

G = 9.81
#: How the engine treats REVERSE flow (TASK-3673), see :class:`ReverseBarrel`:
#: "hds5" (default) | "swap" (Kentry <-> Kexit + brink floor only) | "off" (SWMM as written).
REVERSE_LOSSES = ("hds5", "swap", "off")
#: hysteresis on the flow direction: |q| beyond this fraction of A_full sqrt(g D) (all barrels)
DIRECTION_DEADBAND = 1e-3
#: SWMM xsect.c RECT_ALFMAX: a closed box's top joins its wetted perimeter above this
#: fraction of the full area
RECT_ALFMAX = 0.97
_DC_POINTS = 400
#: SWMM [OPTIONS] INERTIAL_DAMPING values; the default (written nowhere) is PARTIAL
INERTIAL_DAMPING = ("NONE", "PARTIAL", "FULL")


class ReverseBarrel:
    """Reverse flow through ONE culvert's barrel in SWMM (TASK-3673).

    SWMM applies [LOSSES] Kentry at node 1 (J_HW) and Kexit at node 2 (O_TW) whatever the
    flow direction (dwflow.c findLocalLosses), and with offset1 = 0 getFlowClass never
    sets UP_CRITICAL. So in REVERSE flow (O_TW -> J_HW) the barrel leaves through J_HW at
    the 2D headwater depth even when that is below critical depth, the drowned entering
    end gets Kexit 1.0 instead of the entrance ke, and the exit gets ke instead of the
    lost velocity head: steady reverse plateaus -8.3 .. +11.2 %, monotone in ke.

    While Q < 0 (hysteresis ``DIRECTION_DEADBAND``):

    * the imposed O_HW stage is floored at invert + dc(|q|): a free exit cannot run below
      critical depth (the brink). SWMM does the same for FORWARD flow at O_TW
      (node.c outfall_setOutletDepth: min(yc, yn)); on the adverse reverse slope no
      normal depth exists, so the floor is dc;
    * "hds5": the conduit's three loss coefficients are set every stride so that SWMM's
      ONE-element momentum balance (dwflow.c, rho = 1 in reverse)::

          h2 - h1 = Sf(yMid) L + (K1/a1 + K2/a2) q^2/(2 g aMid) + Kavg q^2/(2 g aMid^2)
                    + sigma q^2 (a2 - a1) / (g aMid^3)

      carries the HDS-5 energy terms::

          h2 - h1 = V1^2/2g (velocity head lost at the brink) + ke V2^2/2g
                    + Sf_full L (entering end drowned: the barrel is pressurised up to the brink)

      i.e. K1 = aMid/a1, K2 = ke aMid/a2 and Kavg cancels SWMM's open-channel friction at
      mid depth (whose box perimeter has no top: R 0.40 m vs 0.28 m full) and its damped
      convective term. Without Kavg (scheme "swap") circles land within 1.5 % but boxes
      reach +7 %;
    * "swap": Kentry 1.0 at the exiting node 1, ke at the entering node 2, no Kavg.

    Forward flow is untouched (Kentry ke, Kexit 1.0, Kavg 0): SWMM's own forward treatment.
    The losses are implicit in SWMM's flow update (denominator), so unlike a lagged stage
    correction they do not feed back stride to stride.
    """

    __slots__ = ("geom", "d", "barrels", "ke", "n", "length", "inv_up", "inv_down", "link",
                 "eps", "reverse", "q", "r_full", "box", "mode", "damping", "_dc_q", "_dc_y")

    def __init__(self, culvert, link, mode="hds5", damping="PARTIAL"):
        if mode not in REVERSE_LOSSES:
            raise ValueError(f"reverse_losses must be one of {REVERSE_LOSSES}, got {mode!r}")
        shape, g1, g2 = culvert.effective_geometry
        self.geom = _geometry(shape, g1, g2)
        self.d = self.geom.rise
        self.box = shape != "CIRCULAR"
        self.barrels = float(culvert.barrels)
        self.ke = float(culvert.entrance.ke)
        self.n = float(culvert.n)
        self.length = float(culvert.length_m)
        self.inv_up = culvert.upstream_invert_m
        self.inv_down = culvert.downstream_invert_m
        self.link = link
        self.mode = mode
        damping = str(damping).upper()
        if damping not in INERTIAL_DAMPING:
            raise ValueError(f"damping must be one of {INERTIAL_DAMPING}, got {damping!r}")
        self.damping = damping
        self.eps = DIRECTION_DEADBAND * self.geom.a_full * math.sqrt(G * self.d) * self.barrels
        self.reverse = False
        self.q = 0.0
        self.r_full = self.geom.a_full / self._perimeter(self.d)
        # dc(q) per barrel, tabulated once (dc reaches the rise at q_top)
        t_top = max(self.geom.top_width(0.999 * self.d), 1e-9)
        q_top = self.geom.area(self.d) ** 1.5 * math.sqrt(G / t_top)
        self._dc_q = [q_top * (k / (_DC_POINTS - 1)) ** 1.5 for k in range(_DC_POINTS)]
        self._dc_y = [self.geom.critical_depth(q) for q in self._dc_q]

    def _perimeter(self, y):
        g = self.geom
        if self.box:
            ramp = (y / self.d - RECT_ALFMAX) / (1.0 - RECT_ALFMAX)
            top = g.span if y >= self.d else max(0.0, ramp) * g.span
            return g.span + 2.0 * min(y, self.d) + top
        return math.pi * self.d if y >= self.d else self.d * g._theta(y) / 2.0

    def critical_depth(self, q_barrel):
        qs, ys = self._dc_q, self._dc_y
        if q_barrel >= qs[-1]:
            return self.d
        k = bisect.bisect_right(qs, q_barrel)
        if k == 0:
            return 0.0
        return ys[k - 1] + (ys[k] - ys[k - 1]) * (q_barrel - qs[k - 1]) / (qs[k] - qs[k - 1])

    def floor_hw(self, hw_stage):
        """The O_HW stage to impose: the 2D headwater, floored at the brink in reverse."""
        if not self.reverse:
            return hw_stage
        return max(hw_stage, self.inv_up + self.critical_depth(-self.q / self.barrels))

    def update(self, q, exit_node, tw_stage):
        """After a stride: SWMM's flow ``q`` (all barrels), the J_HW node (the exit in
        reverse) and the imposed tailwater stage; sets the direction and the losses."""
        self.q = q
        if self.mode == "off":
            return
        if not self.reverse:
            if q < -self.eps:
                self.reverse = True
            else:
                return
        elif q > self.eps:
            self.reverse = False
            self._set(self.ke, 1.0, 0.0)
            return
        if self.mode == "swap":
            self._set(1.0, self.ke, 0.0)
            return
        g = self.geom
        qb = abs(q) / self.barrels
        y1 = min(max(exit_node.depth, 1e-6), self.d)
        y2 = min(max(tw_stage - self.inv_down, 1e-6), self.d)
        a1, a2 = g.area(y1), g.area(y2)
        ym = 0.5 * (y1 + y2)
        am = g.area(ym)
        kavg = 0.0
        if ym < self.d and qb > 0.0:
            vm = qb / am
            t = g.top_width(ym)
            fr = vm / math.sqrt(G * am / t) if t > 0.0 else 0.0
            sigma = self.sigma(fr)
            extra = -sigma * qb * qb * (a2 - a1) / (G * am ** 3)
            if y2 >= self.d:
                rm = am / self._perimeter(ym)
                vf = qb / g.a_full
                extra += self.n ** 2 * self.length * (vf * vf / self.r_full ** (4.0 / 3.0)
                                                      - vm * vm / rm ** (4.0 / 3.0))
            kavg = extra / (vm * vm / (2.0 * G))
        self._set(am / a1, self.ke * am / a2, kavg)

    def sigma(self, fr):
        """SWMM's convective-term weight (dwflow.c:191-208) under the INERTIAL_DAMPING option
        actually written to the .inp: 1 for NONE, 0 for FULL, the Froude ramp for PARTIAL."""
        if self.damping == "NONE":
            return 1.0
        if self.damping == "FULL":
            return 0.0
        return 1.0 if fr <= 0.5 else (0.0 if fr >= 1.0 else 2.0 * (1.0 - fr))

    def _set(self, k1, k2, kavg):
        link = self.link
        link.inlet_head_loss = k1
        link.outlet_head_loss = k2
        link.average_head_loss = kavg


class SwmmHeadwallEngine:
    def __init__(self, culverts, workdir, *, duration_s=7 * 86400, routing_step=0.25,
                 inp_name="culverts.inp", extra_options=None, initial_stages=None,
                 swmm_inlet_code=False, dummy=None, reverse_losses="hds5"):
        """``reverse_losses``: reverse-flow treatment, see :class:`ReverseBarrel`
        ("hds5" default; "off" = SWMM as written, the TASK-3672 behaviour)."""
        from pyswmm import Links, Nodes, Simulation

        os.makedirs(workdir, exist_ok=True)
        self.culverts = list(culverts)
        self.inp_path = os.path.join(workdir, inp_name)
        self.layout = write_inp(self.culverts, self.inp_path, duration_s=duration_s,
                                routing_step=routing_step, extra_options=extra_options,
                                initial_stages=initial_stages,
                                swmm_inlet_code=swmm_inlet_code, dummy=dummy)
        self.sim = Simulation(self.inp_path)
        self.sim.step_advance(1)
        self.sim.start()
        nodes, links = Nodes(self.sim), Links(self.sim)
        self._o_hw, self._o_tw, self._j_hw, self._link = {}, {}, {}, {}
        for i, names in self.layout.names.items():
            self._o_hw[i] = nodes[names.o_hw]
            self._o_tw[i] = nodes[names.o_tw]
            self._j_hw[i] = nodes[names.j_hw]
            self._link[i] = links[names.conduit]
        damping = {str(k).upper(): str(v).upper() for k, v in (extra_options or {}).items()}
        damping = damping.get("INERTIAL_DAMPING", "PARTIAL")
        self._rev = {i: ReverseBarrel(self.culverts[i], self._link[i], reverse_losses, damping)
                     for i in self._link}
        self._tw = {i: None for i in self._link}       # last imposed tailwater stage
        self.duration_s = int(duration_s)
        self.elapsed_s = 0
        self.closed = False
        self.routing_error_pct = None

    def set_stages(self, hw, tw) -> None:
        """Impose 2D headwater/tailwater STAGES (m, absolute) on every open culvert."""
        for i in self._link:
            self._o_hw[i].outfall_stage(self._rev[i].floor_hw(float(hw[i])))
            self._o_tw[i].outfall_stage(float(tw[i]))
            self._tw[i] = float(tw[i])

    def stride(self) -> None:
        """Advance SWMM by exactly one second."""
        if self.elapsed_s >= self.duration_s:
            raise RuntimeError(
                f"SWMM reached its END_TIME ({self.duration_s} s): pass duration_s >= the "
                f"ANUGA final time")
        try:
            next(self.sim)
        except StopIteration:
            self.close()
            raise RuntimeError("SWMM stopped before duration_s (END_TIME reached)") from None
        self.elapsed_s += 1
        for i, rev in self._rev.items():
            if self._tw[i] is not None:
                rev.update(self._link[i].flow, self._j_hw[i], self._tw[i])

    def flows(self) -> np.ndarray:
        """Conduit flow (m3/s, all barrels; + = inlet -> outlet); 0 for closed culverts."""
        q = np.zeros(len(self.culverts))
        for i, link in self._link.items():
            q[i] = link.flow
        return q

    def junction_heads(self) -> np.ndarray:
        h = np.full(len(self.culverts), np.nan)
        for i, node in self._j_hw.items():
            h[i] = node.head
        return h

    def close(self):
        """Finish the run; returns SWMM's flow-routing continuity error (%), read
        from the .rpt ("Flow Routing Continuity" block), or None if absent."""
        if not self.closed:
            self.sim.report()
            self.sim.close()
            self.closed = True
            self.routing_error_pct = routing_error_from_rpt(self.rpt_path)
        return self.routing_error_pct

    @property
    def rpt_path(self):
        return os.path.splitext(self.inp_path)[0] + ".rpt"


def routing_error_from_rpt(path):
    """The flow-routing 'Continuity Error (%)' from a SWMM .rpt, or None."""
    try:
        with open(path) as f:
            text = f.read()
    except OSError:
        return None
    block = text.split("Flow Routing Continuity", 1)
    if len(block) < 2:
        return None
    for line in block[1].splitlines():
        if "Continuity Error (%)" in line:
            return float(line.split()[-1])
    return None


_RPT_ERROR = re.compile(r"\bERROR\s+\d+")


def rpt_errors(path):
    """SWMM input/run errors ("ERROR 2xx: ...") reported in a .rpt."""
    with open(path) as f:
        return [ln.strip() for ln in f if _RPT_ERROR.search(ln)]
