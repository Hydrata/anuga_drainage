"""SWMM side of the headwall culvert coupler: one pyswmm Simulation for N culverts.

pyswmm strides whole seconds only (``step_advance(0.5)`` raises TypeError), so
the engine advances in 1 s strides; the coupler holds Q between strides.

Entrance energy (TASK-3672 robustness, thread T1): SWMM nodes are piezometric, so with the
2D pond stage imposed on the outfall the barrel inlet runs at the POND depth, with no
velocity head and no entrance drawdown, and SWMM's PARTIAL inertial damping drops the
convective (velocity-head) term as the barrel Froude number nears 1. In partly-full
accelerating barrels (HDS-5 outlet-partial) that over-delivers by up to ~40 % (with the
stock Kexit 1.0 standing in for the missing acceleration term, 10-37 %). With
``entrance_energy`` (the default) the engine:

* writes Kentry = Kexit = 0, INERTIAL_DAMPING NONE (the full convective term) and a tight
  dynamic-wave convergence (MAX_TRIALS 30, HEAD_TOLERANCE 0.0001 m; with the default
  8 / 0.0015 the loss-free junction rings, sd up to 9 % of Q at a fixed stage);
* imposes on the ENTERING end the depth y1 of the pond-to-barrel energy relation
  H = y1 + (1 + ke) v1^2 / 2g (EntranceEnergy), so the barrel sees the drawn-down inlet
  depth. Full flow is unchanged: the exit velocity head is lost at the piezometric
  tailwater node, H - TW = (1 + ke + friction) V^2 / 2g (HDS-5 Eq 3.6b);
* leaves a steep barrel with a free outlet (HDS-5: inlet control by definition) on the
  pond stage (no drawdown); the coupler's HDS-5 inlet cap governs there. The losses stay
  (0, 0, 0): MEASURED, switching a steep barrel to the stock (ke, 1.0) losses at run time
  rings (flow 0 <-> 3.9 m3/s every 3 s from a dry start; the steady inlet grid fails 7/126)
  once the link's hasLosses flag is set (it is, see inp_writer.HAS_LOSSES_KAVG).

Reverse flow (TASK-3673, :class:`ReverseBarrel`) and the entrance energy are ONE per-stride
state machine per culvert (``SwmmHeadwallEngine``):

* q > 0 and not steep-with-free-outlet: EntranceEnergy depth imposed on O_HW, losses (0, 0, 0);
* q > 0 and steep-with-free-outlet: pond stage on O_HW, losses (0, 0, 0), so the coupler's
  HDS-5 inlet cap governs;
* q < 0 (beyond the direction deadband): ReverseBarrel hds5 losses, the O_HW stage floored at
  the brink (dc), O_TW at the plain 2D stage (no entrance-energy lowering: stacked on the
  reverse losses it under-delivers by 15-25 %, MEASURED);
* leaving reverse restores the FORWARD losses (0, 0, 0), not the stock (ke, 1.0, 0).
"""
from __future__ import annotations

import bisect
import itertools
import math
import os
import re

import numpy as np

from .inlet_control import G, _geometry, ratings_for
from .inp_writer import ENTRANCE_ENERGY_OPTIONS, write_inp

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


class EntranceEnergy:
    """Depth to impose on the ENTERING end of one barrel (pond-to-barrel energy).

    The imposed depth y1 is driven, one damped Newton step per stride, to the solution of

        q_swmm(y1) = q_E(y1) = A(y1) sqrt(2 g (H - y1) / (1 + ke)),   H = pond depth

    (q_swmm = the last stride's flow per barrel, held to the HDS-5 inlet rating: the
    entrance passes the governing flow, never SWMM's excess). The direct form, y1 = root of
    H = y1 + (1 + ke) q^2 / 2g A^2 with the last q, is a lagged fixed point that is
    ill-conditioned near critical (dy1/dq -> -inf) and oscillates; e(y1) = q_E - q_swmm
    has a bounded slope. Per stride:

    * feedforward: the drawdown H - y1 scales with H (open channel: v^2/2g ~ H);
    * full inlet (H - (1 + ke) q^2 / 2g A_full^2 >= rise): the direct relation, which is
      well conditioned there;
    * otherwise one step y1 += gain * e / (q_swmm' - q_E'), y1 in [dc(q), H] (the barrel
      inlet cannot be supercritical; HDS-5's gvf inlet depth sits between dc and the
      choke depth on near-critical barrels);
    * the imposed depth moves at most ``rate`` rises per stride beyond the feedforward.

    ``is_steep``: HDS-5's inlet-control-by-definition test at the inlet-rating flow (S above
    the critical slope at dc, free outlet), evaluated on the cap so it cannot flip with
    SWMM's own excess flow.
    """

    __slots__ = ("k", "rise", "a_full", "gain", "rate", "slope", "dy", "n", "a_tab",
                 "t_tab", "qc_tab", "sc_tab", "y", "h", "steep", "full")

    N_TABLE = 2000
    FULL_BAND = 0.05
    FULL_RELAX = 0.5

    def __init__(self, shape, g1, g2, ke, slope=0.0, manning_n=0.013, gain=0.1, rate=0.05):
        geom = _geometry(shape, g1, g2)
        self.k = 1.0 + float(ke)
        self.rise = geom.rise
        self.a_full = geom.a_full
        self.gain = float(gain)
        self.rate = float(rate)
        self.slope = float(slope)
        n = self.n = self.N_TABLE
        self.dy = self.rise / n
        ys = [i * self.dy for i in range(n + 1)]
        self.a_tab = [geom.area(y) for y in ys]
        self.t_tab = [geom.top_width(y) if y < self.rise else 0.0 for y in ys]
        # critical flow at each depth (increasing; inf where the top width closes)
        qc = [a * math.sqrt(G * a / t) if t > 0.0 else (0.0 if a == 0.0 else math.inf)
              for a, t in zip(self.a_tab, self.t_tab)]
        self.qc_tab = list(itertools.accumulate(qc, max))
        # critical slope at each depth: steep for q if S > S_c(dc(q)) (yn < dc)
        if shape == "CIRCULAR":
            per = [self.rise * math.acos(min(max(1.0 - 2.0 * y / self.rise, -1.0), 1.0))
                   for y in ys]
        else:
            per = [g2 + 2.0 * min(y, g1) for y in ys]
        self.sc_tab = [manning_n ** 2 * (G * a / t) / (a / pp) ** (4.0 / 3.0)
                       if t > 0.0 and a > 0.0 else math.inf
                       for a, t, pp in zip(self.a_tab, self.t_tab, per)]
        self.y = self.h = None
        self.steep = self.full = False

    def _at(self, y):
        """(A, T) at depth y (linear in the tables; full above the rise)."""
        if y >= self.rise:
            return self.a_full, 0.0
        x = y / self.dy
        i = int(x)
        f = x - i
        a, t = self.a_tab, self.t_tab
        return a[i] + f * (a[i + 1] - a[i]), t[i] + f * (t[i + 1] - t[i])

    def y_crit(self, q):
        """Critical depth of per-barrel flow q (the rise at or above the crown's)."""
        i = bisect.bisect_left(self.qc_tab, q)
        if i == 0:
            return 0.0
        if i >= self.n:
            return self.rise
        q0, q1 = self.qc_tab[i - 1], self.qc_tab[i]
        return (i - 1 + (q - q0) / (q1 - q0)) * self.dy

    def is_steep(self, q, tail):
        """Steep barrel (normal depth below critical) with a free outlet (``tail`` depth below
        critical, critical below the crown): HDS-5 inlet control by definition."""
        if self.slope <= 0.0:
            return False
        dc = self.y_crit(q)
        if dc >= self.rise or tail >= dc:
            return False
        return self.slope > self.sc_tab[min(int(dc / self.dy), self.n - 1)]

    def reset(self):
        self.y = None
        self.full = False

    def depth(self, h, q, tail=0.0, cap=math.inf):
        """Depth (m above the entering invert) to impose for pond depth ``h``, the last
        stride's per-barrel SWMM flow ``q`` (>= 0, into the barrel), far-end depth ``tail``
        and the per-barrel HDS-5 inlet-rating flow ``cap`` at ``h``."""
        if h <= 0.0 or q <= 0.0:
            self.y = None
            return h
        q_gov = min(q, cap)                      # the flow the entrance passes
        self.steep = self.is_steep(q_gov if math.isinf(cap) else cap, tail)
        if self.steep:
            self.y = None
            return h
        if self.y is None:
            y0 = h
        elif self.h > 0.0:
            y0 = min(h - (self.h - self.y) * (h / self.h), h)
        else:
            y0 = min(self.y, h)
        y_full = h - self.k * q_gov * q_gov / (2.0 * G * self.a_full * self.a_full)
        # full inlet with hysteresis (no branch chatter where the inlet just fills)
        self.full = y_full >= self.rise * (1.0 if self.full else 1.0 + self.FULL_BAND)
        if self.full:                            # under-relaxed: the lagged fixed point
            y = y0 + self.FULL_RELAX * (y_full - y0)   # flips with period 2 at loop gain > 1
        else:
            # drive SWMM's flow to min(q_E(y1), cap): outlet control (energy) or inlet
            # control (the barrel carries the capped flow, not an excess)
            floor = min(self.y_crit(q_gov), h)
            y = max(y0, floor)
            a, t = self._at(y)
            c = 2.0 * G / self.k
            s = math.sqrt(c * max(h - y, 1e-6 * self.rise))
            q_e = a * s
            dq_s = 2.0 * q / max(y, 0.05 * self.rise)     # q_swmm ~ y1^2
            if q_e < cap:
                dq_e = t * s - a * c / (2.0 * s)          # > 0 below the choke depth
                y += self.gain * (q_e - q) / max(dq_s - dq_e, 0.5 * dq_s)
            else:
                y += self.gain * (cap - q) / dq_s
            y = min(max(y, floor), h)
        dmax = self.rate * self.rise
        y = min(max(y, y0 - dmax), y0 + dmax, h)
        self.y, self.h = y, h
        return y


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

    Forward flow is untouched by the loss scheme: on leaving reverse the engine's FORWARD
    losses (``forward``: stock Kentry ke / Kexit 1.0, or 0 / 0 under the entrance energy) are
    restored with Kavg 0.
    The losses are implicit in SWMM's flow update (denominator), so unlike a lagged stage
    correction they do not feed back stride to stride.
    """

    __slots__ = ("geom", "d", "barrels", "ke", "n", "length", "inv_up", "inv_down", "link",
                 "eps", "reverse", "q", "r_full", "box", "mode", "damping", "forward", "_dc_q",
                 "_dc_y")

    def __init__(self, culvert, link, mode="hds5", damping="PARTIAL", forward=None):
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
        # the losses (Kentry, Kexit) of the FORWARD state, restored on leaving reverse
        self.forward = (self.ke, 1.0) if forward is None else tuple(forward)
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
            self._set(*self.forward, 0.0)
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
                 swmm_inlet_code=False, dummy=None, reverse_losses="hds5",
                 entrance_energy=True, gain=0.1, rate=0.05):
        """``reverse_losses``: reverse-flow treatment, see :class:`ReverseBarrel`
        ("hds5" default; "off" = SWMM as written, the TASK-3672 behaviour).
        ``entrance_energy``: forward-flow entrance energy (module docstring); False = the
        stock engine (pond stage, Kentry ke / Kexit 1.0, SWMM's default options), kept as the
        negative control."""
        from pyswmm import Links, Nodes, Simulation

        os.makedirs(workdir, exist_ok=True)
        self.culverts = list(culverts)
        self.inp_path = os.path.join(workdir, inp_name)
        self.layout = write_inp(self.culverts, self.inp_path, duration_s=duration_s,
                                routing_step=routing_step, extra_options=extra_options,
                                initial_stages=initial_stages,
                                swmm_inlet_code=swmm_inlet_code, dummy=dummy,
                                entrance_energy=entrance_energy)
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
        self.entrance_energy = bool(entrance_energy)
        # the damping option SWMM actually runs with (the reverse scheme's sigma mirrors it)
        options = {**(ENTRANCE_ENERGY_OPTIONS if entrance_energy else {}), **(extra_options or {})}
        damping = {str(k).upper(): str(v).upper() for k, v in options.items()}
        damping = damping.get("INERTIAL_DAMPING", "PARTIAL")
        self._rev, self._ee = {}, {}
        for i in self._link:
            c = self.culverts[i]
            shape, g1, g2 = c.effective_geometry
            ke = c.entrance.ke
            self._rev[i] = ReverseBarrel(c, self._link[i], reverse_losses, damping,
                                         forward=(0.0, 0.0) if entrance_energy else (ke, 1.0))
            if entrance_energy:
                self._ee[i] = (EntranceEnergy(shape, g1, g2, ke, c.slope, c.n, gain, rate),
                               float(c.barrels), c.upstream_invert_m, c.downstream_invert_m,
                               ratings_for(c)[0])
        self._tw = {i: None for i in self._link}       # last imposed tailwater stage
        self.imposed = {}
        self.duration_s = int(duration_s)
        self.elapsed_s = 0
        self.closed = False
        self.routing_error_pct = None

    def set_stages(self, hw, tw) -> None:
        """Impose 2D headwater/tailwater STAGES (m, absolute) on every open culvert.

        Forward (with ``entrance_energy``) the inlet end gets the entrance-energy depth
        (EntranceEnergy) instead of the pond stage; in reverse the headwater end is floored
        at the brink and the tailwater end gets the plain 2D stage. ``self.imposed[i]`` is
        what SWMM was given."""
        for i in self._link:
            h, w = float(hw[i]), float(tw[i])
            rev = self._rev[i]
            if rev.reverse:
                h = rev.floor_hw(h)
            elif self.entrance_energy:
                h = self._forward_entrance(i, h, w)
            self.imposed[i] = (h, w)
            self._o_hw[i].outfall_stage(h)
            self._o_tw[i].outfall_stage(w)
            self._tw[i] = w

    def _forward_entrance(self, i, h, w):
        """Forward state: the O_HW stage from the last stride's flow. Steep barrel, free
        outlet (HDS-5 inlet control): the pond stage, the cap governs; otherwise the
        entrance-energy depth. The depth is NOT floored at the tailwater stage: against an
        adverse or vanishing head the drawdown (1 + ke) v^2/2g IS the dissipation that
        decelerates the column towards a reversal (floored, the decelerating column loses
        its losses and the flow crosses zero 4-6 s late, MEASURED 31/42 cases)."""
        ee, nb, inv_up, inv_dn, rate_f = self._ee[i]
        q = self._rev[i].q / nb
        if q <= 0.0:                            # not flowing forward (deadband / start)
            ee.reset()
            return h
        y = ee.depth(h - inv_up, q, w - inv_dn, rate_f(h - inv_up))
        return inv_up + y

    def stride(self) -> None:
        """Advance SWMM by exactly one second, then run the direction/loss state machine."""
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
            if self._tw[i] is None:
                continue
            was_reverse = rev.reverse
            rev.update(self._link[i].flow, self._j_hw[i], self._tw[i])
            if was_reverse != rev.reverse and i in self._ee:
                self._ee[i][0].reset()          # the forward depth restarts from the pond

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
