"""Unsteady 1D oracle: the headwall culvert coupler vs quasi-steady HDS-5 (TASK-3672).

The headwater and tailwater are IMPOSED as 1 s series (stage = invert + depth, the way the
coupler imposes the 2D end stages), the culvert is run through SwmmHeadwallEngine +
InletControl (exactly the coupler's per-stride arithmetic) and each second's flow is
compared with the independent HDS-5 calculator at that instant's (HW, TW):

    Q_qs(t) = hds5.Culvert.q(HW(t), TW(t))             forward (HW stage >= TW stage)
    Q_qs(t) = -hds5.Culvert(S = -S).q(TW(t), HW(t))    reverse: the mirrored culvert

Series (4800 s; depths in rises D above the respective invert, linear between knots)::

    A   0-1320   HW 0.3 -> 3.0 over 1200 s (part-full -> full), hold 120 s     TW 0 (free)
    B   1320-2520  HW 3.0 -> 0.3 over 1200 s (full -> part-full)
    C   2520-3240  HW 0.3 -> 2.5 over 600 s (part-full -> full), hold 120 s
    D   3240-4260  HW 2.5 -> 2.0, hold; then TW 0 -> 1.8 D over 600 s (inlet -> outlet handoff), hold
    E   4260-4800  HW 2.0 -> 0.5 D and TW 1.8 -> 2.0 D over 300 s (exactly one flow reversal), hold

Band per instant (the D9 protocol; never widened)::

    |Q(t) - Q_qs(t - lag)| <= max(2 % |Q_qs|, 5 L/s) + |dV_qs/dt| + 2 s |dQ_qs/dt|    for some lag in {0, 1, 2} s

dV_qs/dt is the quasi-steady barrel storage change (a barrel that fills or empties cannot
pass the quasi-steady flow); within +-30 s of a fill/unfill crossing (where that storage
term is a step, not a rate) instants are REPORTED but excluded. The last 60 s of every
hold (plateau) must be within 2 % of Q_qs. Chatter: sign changes of dQ per tumbling
60 s window at HW/D > 0.5, minus the oracle's own, must be <= 5.
"""
from __future__ import annotations

import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _hds5_oracle as hds5  # noqa: E402

from anuga_drainage.culverts import ENTRANCE_TYPES, MATERIAL_MANNING_N, HeadwallCulvert  # noqa: E402
from anuga_drainage.culverts.inlet_control import BARREL, CONTROL_NAMES, InletControl  # noqa: E402
from anuga_drainage.culverts.swmm_engine import SwmmHeadwallEngine  # noqa: E402

D = 0.9
SPAN = 1.5
INV = 10.0
T_END = 4800
PREROLL_S = 120
GEOMETRIES = {"20m_S2": (20.0, 0.02), "20m_S0.5": (20.0, 0.005), "flume_8m": (8.0, 0.0125)}

#: (t_s, HW/D) and (t_s, TW/D) knots; depths above the inlet / outlet invert.
HW_KNOTS = [(0, 0.3), (1200, 3.0), (1320, 3.0), (2520, 0.3), (3120, 2.5), (3240, 2.5),
            (3420, 2.0), (4260, 2.0), (4560, 0.5), (T_END, 0.5)]
TW_KNOTS = [(0, 0.0), (3540, 0.0), (4140, 1.8), (4260, 1.8), (4560, 2.0), (T_END, 2.0)]
SEGMENTS = {"A": (0, 1320), "B": (1320, 2520), "C": (2520, 3240), "D": (3240, 4260),
            "E": (4260, T_END)}
#: holds whose last 60 s are scored as plateaus.
PLATEAUS = {"A": (1260, 1320), "C": (3180, 3240), "D1": (3480, 3540), "D2": (4200, 4260),
            "E": (4740, T_END)}

REL_TOL = 0.02
ABS_TOL = 0.005          # m3/s
LAG_S = 2
EXCLUDE_S = 30
CHATTER_WINDOW_S = 60
CHATTER_MAX = 5


def culvert_for(e, L, S, barrels=1):
    d = dict(name="unsteady", shape=e.shape, coordinates=[[0.0, 0.0], [L, 0.0]],
             upstream_invert_m=INV, downstream_invert_m=INV - S * L,
             entrance_type=e.value, material=e.material, barrels=barrels)
    if e.shape == "circular":
        d["diameter_m"] = D
    else:
        d.update(width_m=SPAN, height_m=D)
    return HeadwallCulvert.from_dict(d)


def knot_series(knots, t):
    xs, ys = zip(*knots)
    return np.interp(t, xs, ys) * D


def stage_series(S, L):
    """(t, hw_depth, tw_depth) for t = 0..T_END (1 s)."""
    t = np.arange(0, T_END + 1)
    return t, knot_series(HW_KNOTS, t), knot_series(TW_KNOTS, t)


# ------------------------------------------------------------------ the oracle
class QuasiSteady:
    def __init__(self, e, L, S):
        self.e, self.L, self.S = e, L, S
        n = MATERIAL_MANNING_N[e.material]
        shape = "circ" if e.shape == "circular" else "box"
        b = None if e.shape == "circular" else SPAN
        self.fwd = hds5.Culvert(shape, D, B=b, L=L, S=S, n=n, inlet=e.hds5)
        self.rev = hds5.Culvert(shape, D, B=b, L=L, S=-S, n=n, inlet=e.hds5)
        self.n = n
        self.area_full = self.fwd.geom.Afull

    def point(self, hw, tw):
        """(signed Q, regime, barrel fill fraction of V_full) at imposed depths hw, tw (m)."""
        S, L = self.S, self.L
        if hw >= tw - S * L:                          # stage_hw >= stage_tw
            culvert = self.fwd
            q = culvert.q(hw, tw)
            sign, tail = 1.0, tw
        else:
            culvert = self.rev
            q = culvert.q(tw, hw)
            sign, tail = -1.0, hw
        if q <= 1e-9:
            return 0.0, "zero", 0.0
        _h, _hi, _ho, regime = culvert.hw(q, tail, detail=True)
        # barrel state under the governing control: full if the oracle says outlet-full, the
        # outlet is submerged, or no open-channel depth carries q; else the normal depth
        yn = hds5.normal_depth(culvert.geom, q, culvert.S, self.n)
        if tail >= D or regime == "outlet-full" or yn is None:
            fill = 1.0
        else:
            fill = culvert.geom.A(yn) / self.area_full
        return sign * q, regime, fill


def series_oracle(e, L, S, hw, tw):
    qs = QuasiSteady(e, L, S)
    out = [qs.point(h, w) for h, w in zip(hw, tw)]
    q = np.array([o[0] for o in out])
    regime = np.array([o[1] for o in out])
    fill = np.array([o[2] for o in out])
    return q, regime, fill * qs.area_full * L           # volume m3


# ------------------------------------------------------------------ the culvert under test
def run_coupler(e, L, S, hw, tw, preroll_s=PREROLL_S, extra_options=None):
    """Drive SwmmHeadwallEngine + InletControl with the imposed series; returns arrays of
    the capped flow, SWMM's own barrel flow and the control flag, for t = 1..T_END."""
    c = culvert_for(e, L, S)
    control = InletControl([c])
    hw_stage, tw_stage = INV + hw, c.downstream_invert_m + tw
    with tempfile.TemporaryDirectory() as d:
        eng = SwmmHeadwallEngine([c], d, duration_s=preroll_s + T_END + 5,
                                 initial_stages=[(hw_stage[0], tw_stage[0])],
                                 extra_options=extra_options)
        for _ in range(preroll_s):
            eng.set_stages([hw_stage[0]], [tw_stage[0]])
            eng.stride()
        q, raw, flag = np.zeros(T_END), np.zeros(T_END), np.zeros(T_END, dtype=np.int8)
        for k in range(1, T_END + 1):
            eng.set_stages([hw_stage[k]], [tw_stage[k]])
            eng.stride()
            cap_f, cap_r = control.caps([hw_stage[k]], [tw_stage[k]])
            r = eng.flows()[0]
            qq, ff = control.apply([r], cap_f, cap_r)
            q[k - 1], raw[k - 1], flag[k - 1] = qq[0], r, ff[0]
        eng.close()
    return q, raw, flag


# ------------------------------------------------------------------ scoring
def flips(x, thr):
    d = np.diff(x)
    d = d[np.abs(d) > thr]
    s = np.sign(d)
    return int(np.sum(s[1:] != s[:-1]))


def score_band(t, q, qref, vref, plateaus, segments=None, hw=None, rise=D, regime=None):
    """The per-instant band, plateau and chatter metrics for one recorded series.

    ``t`` (s, may have gaps), ``q`` the culvert flow under test, ``qref`` / ``vref`` the
    quasi-steady flow and barrel storage (m3) at the same instants, ``hw`` the headwater
    depth (m) for the chatter windows. Returns a JSON-able dict."""
    t, q, qref, vref = (np.asarray(a, dtype=float) for a in (t, q, qref, vref))
    dq = np.gradient(qref, t)
    dv = np.gradient(vref, t)
    tol = np.maximum(REL_TOL * np.abs(qref), ABS_TOL) + np.abs(dv) + LAG_S * np.abs(dq)
    err = np.full(q.shape, np.inf)
    for lag in range(LAG_S + 1):
        err = np.minimum(err, np.abs(q - np.interp(t - lag, t, qref)))
    full = vref >= vref.max() * (1 - 1e-9)
    excl = np.zeros(q.shape, dtype=bool)
    crossings = np.flatnonzero(np.diff(full.astype(np.int8)) != 0)
    for k in crossings:
        excl |= np.abs(t - t[k]) <= EXCLUDE_S
    ratio = err / tol
    scored = ~excl
    res = dict(
        n=int(scored.sum()), violations=int(((ratio > 1) & scored).sum()),
        worst_ratio=float(ratio[scored].max()), worst_t=float(t[scored][np.argmax(ratio[scored])]),
        excluded=int(excl.sum()), excluded_violations=int(((ratio > 1) & excl).sum()),
        excluded_max_ratio=float(ratio[excl].max()) if excl.any() else 0.0,
        crossings=[float(t[k]) for k in crossings])
    if regime is not None:                    # which HDS-5 regime the violating instants are in
        counts = {}
        for r_, bad_ in zip(regime, (ratio > 1) & scored):
            if bad_:
                counts[r_] = counts.get(r_, 0) + 1
        res["violations_by_regime"] = counts
    seg = {}
    for name, (a, b) in (segments or {}).items():
        m = (t > a) & (t <= b) & scored
        seg[name] = dict(n=int(m.sum()), violations=int(((ratio > 1) & m).sum()),
                         worst_ratio=float(ratio[m].max()) if m.any() else 0.0)
    res["segments"] = seg
    plat = {}
    for name, (a, b) in plateaus.items():
        m = (t > a) & (t <= b)
        ref = float(qref[m].mean())
        plat[name] = dict(q=float(q[m].mean()), q_ref=ref,
                          err_pct=float(100 * (q[m].mean() - ref) / ref) if abs(ref) > 1e-6 else 0.0)
    res["plateaus"] = plat
    thr = 0.005 * float(np.abs(qref).max())
    worst_excess = 0
    if hw is not None:
        hw = np.asarray(hw, dtype=float)
        for a in np.arange(t[0] - 1e-9, t[-1], CHATTER_WINDOW_S):
            m = (t > a) & (t <= a + CHATTER_WINDOW_S)
            if m.sum() < 3 or hw[m].mean() / rise <= 0.5:
                continue
            worst_excess = max(worst_excess, flips(q[m], thr) - flips(qref[m], thr))
    res["chatter_excess"] = int(worst_excess)
    big = np.abs(q) > 2 * ABS_TOL
    signs = np.sign(q[big])
    res["reversals"] = int(np.sum(signs[1:] != signs[:-1]))
    return res


def score(t, hw, tw, q, qref, vref, regime, flag):          # t, q, flag: 1..T_END; rest 0..T_END
    """Band metrics for one 1D case (see :func:`score_band`) plus control/regime tallies."""
    res = score_band(t, q, qref[1:], vref[1:], PLATEAUS, SEGMENTS, hw[1:], regime=regime[1:])
    res["controls"] = {CONTROL_NAMES[k]: int((flag == k).sum()) for k in (BARREL, 1, 2)}
    res["regimes"] = sorted(set(regime.tolist()))
    return res


def run_case(args):
    entrance_index, geom = args
    e = ENTRANCE_TYPES[entrance_index]
    L, S = GEOMETRIES[geom]
    t, hw, tw = stage_series(S, L)
    qref, regime, vref = series_oracle(e, L, S, hw, tw)
    q, raw, flag = run_coupler(e, L, S, hw, tw)
    res = score(t[1:], hw, tw, q, qref, vref, regime, flag)
    res.update(case=f"{e.hds5}@{geom}", hds5=e.hds5, geometry=geom)
    return res, dict(t=t[1:], hw=hw[1:], tw=tw[1:], q=q, q_raw=raw, flag=flag, q_ref=qref[1:],
                     regime=regime[1:])


def run_all(workers=None):
    jobs = [(i, g) for g in GEOMETRIES for i in range(len(ENTRANCE_TYPES))]
    workers = workers or min(16, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(run_case, jobs))
