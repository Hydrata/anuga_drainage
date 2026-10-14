"""Coupler-enforced HDS-5 inlet control (mechanism C2, epic 3585 TASK-3672).

SWMM 5.2.4's own culvert code cannot be trusted inside dynamic wave (three C-level
defects, proven in the deploy spike folder ``results/c-proof.csv``): the inlet
equations are skipped once both barrel ends are full, Form 1 returns a stale flow out
of ``findroot_Ridder``, and mitered inlets get a -7.0 slope factor where HDS-5 has
-0.7. So the .inp carries culvert code 0 and the coupler applies the FHWA HDS-5
inlet-control rating itself::

    q_used = sign(q_swmm) * min(|q_swmm|, barrels * Q_inlet(head of the upstream side))

* forward flow: head = headwater stage - upstream invert, rating of the entrance;
* reverse flow: head = tailwater stage - downstream invert, rating of the MIRRORED
  culvert (slope -S, same entrance type at the outlet end), the same convention the
  independent oracle uses;
* which control governed each sample is returned as a flag: ``INLET`` (forward flow
  held to the inlet rating), ``BARREL`` (SWMM's barrel flow, i.e. outlet control or
  inlet capacity not reached, either direction), ``REVERSE_INLET``.

The rating is tabulated ONCE per culvert on a uniform head grid (an in-loop HDS-5
inversion cost 11 minutes per ANUGA run in a probe) and evaluated by linear
interpolation in pure Python (~0.3 us).

The HDS-5 equations (Appendix A, Eqs A.1-A.3, SI) are re-implemented here without scipy
(the package does not depend on it); ``tests/_hds5_oracle.py`` stays the independent
oracle and ``tests/test_inlet_rating.py`` pins this module to it.
"""
from __future__ import annotations

import functools
import math

import numpy as np

G = 9.81
KU = 1.811           # SI unit conversion for Q / (A D^0.5)
X_UNSUB = 3.5        # Ku Q / (A D^0.5): upper limit of the unsubmerged equations
X_SUB = 4.0          # lower limit of the submerged equation (linear HW/D in between)

#: Control flags (per sample) and their names.
BARREL, INLET, REVERSE_INLET, NOT_TRACKED = 0, 1, 2, 3
CONTROL_NAMES = ("barrel", "inlet", "reverse-inlet", "n/a")

#: Table extent: heads up to this many rises are tabulated (above it the cap is held).
TABLE_RISES = 12.0
TABLE_POINTS_PER_RISE = 200
_Q_POINTS = 1500

# hds5 key: (form, K, M, c, Y, mitered)   (HDS-5 Tables A.1 / A.2)
COEFFS = {
    "rcp_sq_hw": (1, 0.0098, 2.00, 0.0398, 0.67, False),
    "rcp_groove_hw": (1, 0.0018, 2.00, 0.0292, 0.74, False),
    "rcp_groove_proj": (1, 0.0045, 2.00, 0.0317, 0.69, False),
    "cmp_hw": (1, 0.0078, 2.00, 0.0379, 0.69, False),
    "cmp_mitered": (1, 0.0210, 1.33, 0.0463, 0.75, True),
    "cmp_proj": (1, 0.0340, 1.50, 0.0553, 0.54, False),
    "circ_bevel45": (1, 0.0018, 2.50, 0.0300, 0.74, False),
    "box_wing30_75": (1, 0.026, 1.00, 0.0347, 0.81, False),
    "box_sq_90hw": (1, 0.061, 0.75, 0.0400, 0.80, False),
    "box_wing0": (1, 0.061, 0.75, 0.0423, 0.82, False),
    "box_wing45_bev": (2, 0.510, 0.667, 0.0309, 0.80, False),
    "box_hw_bev45": (2, 0.495, 0.667, 0.0314, 0.82, False),
    "cmbox_90hw": (1, 0.0083, 2.00, 0.0379, 0.69, False),
    "cmbox_thinproj": (1, 0.0340, 1.50, 0.0496, 0.57, False),
}


# ---------------------------------------------------------------- geometry
class _Box:
    def __init__(self, span, rise):
        self.rise, self.span = float(rise), float(span)
        self.a_full = self.span * self.rise

    def area(self, y):
        return self.span * min(y, self.rise)

    def top_width(self, y):
        return self.span

    def critical_depth(self, q):
        if q <= 0:
            return 0.0
        return min((q * q / (G * self.span ** 2)) ** (1.0 / 3.0), self.rise)


class _Circle:
    def __init__(self, diameter):
        self.rise = float(diameter)
        self.a_full = math.pi * self.rise ** 2 / 4

    def _theta(self, y):
        return 2.0 * math.acos(min(max(1.0 - 2.0 * y / self.rise, -1.0), 1.0))

    def area(self, y):
        if y >= self.rise:
            return self.a_full
        th = self._theta(y)
        return self.rise ** 2 / 8.0 * (th - math.sin(th))

    def top_width(self, y):
        return 0.0 if y >= self.rise else self.rise * math.sin(self._theta(y) / 2.0)

    def critical_depth(self, q):
        """Depth where Q^2 T / (g A^3) = 1, capped at the rise (bisection)."""
        if q <= 0:
            return 0.0
        d = self.rise

        def f(y):
            return self.area(y) ** 1.5 * math.sqrt(G / max(self.top_width(y), 1e-12)) - q
        top = d * (1 - 1e-7)
        if f(top) <= 0:
            return d
        lo, hi = 1e-9 * d, top
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if f(mid) < 0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)


def _geometry(shape, g1, g2):
    return _Circle(g1) if shape == "CIRCULAR" else _Box(g2, g1)


# ---------------------------------------------------------------- inlet equations
def inlet_head(q, geom, key, slope):
    """HDS-5 inlet-control headwater (m above the inlet invert) for discharge ``q``."""
    form, k, m, c, y_c, mitered = COEFFS[key]
    d, a = geom.rise, geom.a_full
    ks = 0.7 if mitered else -0.5
    if q <= 0:
        return 0.0

    def unsub(qq):
        x = KU * qq / (a * d ** 0.5)
        if form == 1:
            dc = geom.critical_depth(qq)
            ac = geom.area(dc)
            hc = dc + qq ** 2 / (2 * G * ac ** 2)
            return d * (hc / d + k * x ** m + ks * slope)
        return d * (k * x ** m)

    def sub(qq):
        x = KU * qq / (a * d ** 0.5)
        return d * (c * x ** 2 + y_c + ks * slope)

    x = KU * q / (a * d ** 0.5)
    if x <= X_UNSUB:
        return unsub(q)
    if x >= X_SUB:
        return sub(q)
    q1 = X_UNSUB * a * d ** 0.5 / KU
    q2 = X_SUB * a * d ** 0.5 / KU
    h1, h2 = unsub(q1), sub(q2)
    return h1 + (h2 - h1) * (q - q1) / (q2 - q1)


class InletRating:
    """Tabulated per-barrel Q_inlet(head): uniform head grid, linear interpolation."""

    __slots__ = ("dh", "q", "n", "q_max")

    def __init__(self, dh, q):
        self.dh, self.q, self.n = float(dh), [float(v) for v in q], len(q)
        self.q_max = self.q[-1]

    def __call__(self, head):
        if head <= 0.0:
            return 0.0
        x = head / self.dh
        i = int(x)
        if i >= self.n - 1:
            return self.q_max
        q = self.q
        return q[i] + (q[i + 1] - q[i]) * (x - i)


def build_rating(shape, g1, g2, key, slope):
    """Tabulate Q_inlet(head) for one barrel (``shape``/g1/g2 as in
    ``HeadwallCulvert.effective_geometry``) on a uniform head grid."""
    geom = _geometry(shape, g1, g2)
    h_max = TABLE_RISES * geom.rise
    q_hi = geom.a_full * math.sqrt(G * geom.rise)
    while inlet_head(q_hi, geom, key, slope) < h_max:
        q_hi *= 2.0
    # denser at low Q, where head is a steep function of Q
    qs = q_hi * (np.linspace(0.0, 1.0, _Q_POINTS) ** 2)
    hs = np.maximum.accumulate(np.array([inlet_head(float(q), geom, key, slope) for q in qs]))
    n = int(TABLE_RISES * TABLE_POINTS_PER_RISE) + 1
    dh = h_max / (n - 1)
    return InletRating(dh, np.interp(np.arange(n) * dh, hs, qs))


@functools.lru_cache(maxsize=None)
def _cached_rating(shape, g1, g2, key, slope):
    return build_rating(shape, g1, g2, key, slope)


def ratings_for(culvert):
    """(forward, reverse) :class:`InletRating` of ONE barrel of ``culvert``: the entrance at
    the inlet end, and the mirrored culvert (slope -S) for flow from the outlet end."""
    shape, g1, g2 = culvert.effective_geometry
    key = culvert.entrance.hds5
    s = round(culvert.slope, 9)
    return (_cached_rating(shape, round(g1, 9), round(g2, 9), key, s),
            _cached_rating(shape, round(g1, 9), round(g2, 9), key, -s))


class InletControl:
    """The cap applied to SWMM's barrel flows, for a list of culverts (no ANUGA needed)."""

    def __init__(self, culverts):
        self.n = len(culverts)
        self.rates = [None if c.closed else ratings_for(c) for c in culverts]
        self.barrels = [float(c.barrels) for c in culverts]
        self.inv_up = [c.upstream_invert_m for c in culverts]
        self.inv_down = [c.downstream_invert_m for c in culverts]

    def caps(self, hw_stage, tw_stage):
        """(forward cap, reverse cap) per culvert, m3/s, all barrels, from the end stages."""
        cap_f, cap_r = [], []
        for i, rates in enumerate(self.rates):
            if rates is None:
                cap_f.append(0.0)
                cap_r.append(0.0)
                continue
            cap_f.append(self.barrels[i] * rates[0](hw_stage[i] - self.inv_up[i]))
            cap_r.append(self.barrels[i] * rates[1](tw_stage[i] - self.inv_down[i]))
        return cap_f, cap_r

    @staticmethod
    def apply(q, cap_f, cap_r):
        """Cap each flow in ``q`` (+ = inlet -> outlet); returns (q list, control flags)."""
        out, flags = [], []
        for qi, cf, cr in zip(q, cap_f, cap_r):
            if qi > cf:
                out.append(cf)
                flags.append(INLET)
            elif qi < -cr:
                out.append(-cr)
                flags.append(REVERSE_INLET)
            else:
                out.append(qi)
                flags.append(BARREL)
        return out, flags
