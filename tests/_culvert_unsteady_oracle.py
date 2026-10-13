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

    |Q(t) - Q_ref(t - lag)| <= max(2 % |Q_ref|, 5 L/s) + |dV_qs/dt| + 2 s |dQ_ref/dt|    for some lag in {0, 1, 2} s

Q_ref = Q_qs except where BOTH ends are drowned (HW >= D and TW >= D: 3874-4461 s, which holds
the flow reversal): there the barrel is a full rigid water column and Q_ref is that column,
(L / (g A)) dQ/dt = (stage_hw - stage_tw) - H(Q), H the same HDS-5 loss relation, integrated
from Q_qs (see inertia_aware_reference). Plateaus are scored against Q_qs.

dV_qs/dt is the quasi-steady barrel storage change (a barrel that fills or empties cannot
pass the quasi-steady flow). The stored V_qs is A(yn) L or A_full L, a STEP at a fill/unfill
crossing, and np.gradient spreads that step over the crossing second and its neighbours, so
those instants (+-EXCLUDE_S = 1 s) are REPORTED but excluded. (The window was +-30 s; the
protocol audit found the 126 instants it hid were all outlet-partial engine misses, none a
reference artefact, so the window was cut to the step's own footprint: a tightening.) The last 60 s of every
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
import _culvert_oracle as oracle  # noqa: E402
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
#: segment E split for REPORTING: the reversal (ramp, 4260-4420 s) and the reverse hold + late
#: ramp (4420-4800 s, the steady reverse-barrel regime; not an inertia question).
E_SPLIT = {"E_cross": (4260, 4420), "E_hold": (4420, T_END)}
#: the deliberately wrong columns (L / (g A) scaled) that must FAIL segment E_cross.
WRONG_COLUMN_SCALES = (2.0, 0.5)
#: holds whose last 60 s are scored as plateaus.
PLATEAUS = {"A": (1260, 1320), "C": (3180, 3240), "D1": (3480, 3540), "D2": (4200, 4260),
            "E": (4740, T_END)}

REL_TOL = 0.02
ABS_TOL = 0.005          # m3/s
LAG_S = 2
EXCLUDE_S = 1
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


def step_window_mask(fwd, hw, tw, q_qs):
    """Instants (over the series) whose imposed headwater lies strictly inside a
    bisection-confirmed step of the HDS-5 reference's HW(Q) at a free outlet (TW at the invert,
    the only tailwater where the critical-slope step sits: a steep barrel with a free outlet is
    inlet control by definition, a mild one is outlet control, and the two forms disagree by
    up to 5.6 % of HW at the critical slope). Q(HW) is continuous but FLAT across the step, so
    inside it the reference Q is a property of the method's regime switch, not of the
    culvert. Scored against the converged march. Returns (mask, [(hw_below, hw_above, q_step)]).
    Forward flow only (the reverse reference has no step: T4 sweep)."""
    wins = oracle.hw_step_windows(fwd)
    free = np.asarray(tw) <= 1e-9
    forward = np.asarray(q_qs) > 0.0
    mask = np.zeros(len(hw), dtype=bool)
    for lo, hi, _q in wins:
        mask |= free & forward & (np.asarray(hw) > lo) & (np.asarray(hw) < hi)
    return mask, wins


def runs(t, mask):
    """[[t_first, t_last], ...] of the contiguous (1 s) runs of True in ``mask``."""
    out = []
    for k in np.flatnonzero(mask):
        if out and t[k] - out[-1][1] <= 1.0:
            out[-1][1] = float(t[k])
        else:
            out.append([float(t[k]), float(t[k])])
    return out


def score_band(t, q, qref, vref, plateaus, segments=None, hw=None, rise=D, regime=None,
               step_window=None):
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
    # reference-step windows (D20): REPORTED and excluded from the forward band like the
    # fill/unfill crossing; the count is asserted by the suite, the list is committed
    step = np.zeros(q.shape, dtype=bool) if step_window is None else np.asarray(step_window, bool)
    scored = ~excl & ~step
    res = dict(
        n=int(scored.sum()), violations=int(((ratio > 1) & scored).sum()),
        worst_ratio=float(ratio[scored].max()), worst_t=float(t[scored][np.argmax(ratio[scored])]),
        excluded=int(excl.sum()), excluded_violations=int(((ratio > 1) & excl).sum()),
        excluded_max_ratio=float(ratio[excl].max()) if excl.any() else 0.0,
        crossings=[float(t[k]) for k in crossings],
        step_excluded=int((step & ~excl).sum()),
        step_excluded_violations=int(((ratio > 1) & step & ~excl).sum()),
        step_excluded_runs=runs(t, step & ~excl),
        violation_runs=runs(t, (ratio > 1) & scored))
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


def score(t, hw, tw, q, qref, vref, regime, flag, step_window=None):  # t, q, flag: 1..T_END; rest 0..T_END
    """Band metrics for one 1D case (see :func:`score_band`) plus control/regime tallies."""
    res = score_band(t, q, qref[1:], vref[1:], PLATEAUS, {**SEGMENTS, **E_SPLIT}, hw[1:],
                     regime=regime[1:], step_window=step_window)
    res["controls"] = {CONTROL_NAMES[k]: int((flag == k).sum()) for k in (BARREL, 1, 2)}
    res["regimes"] = sorted(set(regime.tolist()))
    return res


def run_case(args):
    entrance_index, geom = args
    e = ENTRANCE_TYPES[entrance_index]
    L, S = GEOMETRIES[geom]
    t, hw, tw = stage_series(S, L)
    q_qs, regime, vref = series_oracle(e, L, S, hw, tw)
    qs = QuasiSteady(e, L, S)
    qref, _column = inertia_aware_reference(qs.fwd, qs.rev, L, S, qs.area_full, D, t, hw, tw, q_qs)
    q, raw, flag = run_coupler(e, L, S, hw, tw)
    step, windows = step_window_mask(qs.fwd, hw, tw, q_qs)
    res = score(t[1:], hw, tw, q, qref, vref, regime, flag, step_window=step[1:])
    res["step_windows"] = [dict(hw_below_m=round(a, 4), hw_above_m=round(b, 4), q_step=round(q_, 5))
                           for a, b, q_ in windows]
    segments = {**SEGMENTS, **E_SPLIT}
    # negative controls: the quasi-steady reference everywhere, and a column with L/(gA) x2, x0.5
    controls = {"quasi_steady": q_qs}
    for scale in WRONG_COLUMN_SCALES:
        controls[f"column_x{scale}"] = inertia_aware_reference(
            qs.fwd, qs.rev, L, S, qs.area_full, D, t, hw, tw, q_qs, coeff_scale=scale)[0]
    scored = {name: score_band(t[1:], q, ref[1:], vref[1:], PLATEAUS, segments)
              for name, ref in controls.items()}
    res["controls_reference"] = {name: {k: v["violations"] for k, v in sc["segments"].items()}
                                 for name, sc in scored.items()}
    res["plateaus"] = scored["quasi_steady"]["plateaus"]     # a hold is steady: Q_qs is the physics
    res["crossing_s"] = dict(coupler=crossing_time(t[1:], q), column=crossing_time(t, qref),
                             quasi_steady=crossing_time(t, q_qs))
    res.update(case=f"{e.hds5}@{geom}", hds5=e.hds5, geometry=geom)
    return res, dict(t=t[1:], hw=hw[1:], tw=tw[1:], q=q, q_raw=raw, flag=flag, q_ref=qref[1:],
                     q_qs=q_qs[1:], regime=regime[1:])


def run_all(workers=None):
    jobs = [(i, g) for g in GEOMETRIES for i in range(len(ENTRANCE_TYPES))]
    workers = workers or int(os.environ.get("CULVERT_UNSTEADY_WORKERS", 0)) or min(16, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(run_case, jobs))


# ------------------------------------------------------------------ inertia-aware reference
# The quasi-steady reference has no water-column inertia. A barrel with BOTH ends drowned is
# necessarily full: a rigid column of length L and area A_full (all barrels),
#
#     (L / (g A)) dQ/dt = (stage_hw - stage_tw) - H(Q)
#
# where H(Q) is the signed stage difference HDS-5 needs to carry Q steadily at the present
# depths (forward: Culvert.hw(Q, TW) - TW + S L; reverse: the mirrored culvert), i.e. the SAME
# loss relation as the quasi-steady reference (H(Q_qs) = stage difference). The column relaxes
# to Q_qs with the time constant tau = (L / (g A)) / H'(Q); H ~ k Q^2, so tau grows without
# bound as Q -> 0 and the quasi-steady reference stops being the physics at a FLOW REVERSAL
# (and, to a lesser degree, at any step in dQ_qs/dt, e.g. the end of a ramp). SWMM's dwflow.c
# carries exactly this term (the qOld local inertia; sigma = 0 for a surcharged closed conduit).
#
# SWITCHING RULE (applicability, no tuning constant): the reference is the rigid column over
# every contiguous interval in which both ends are drowned (HW >= rise and TW >= rise, depths
# above the respective inverts), integrated from Q_qs at the interval's first instant on the
# imposed (1D) or MEASURED (2D) stages; it is Q_qs everywhere else. In the 4800 s series this is
# 3874-4461 s (TW reaches 1 D in segment D until HW falls below 1 D in segment E); segments A-C
# and every plateau window are untouched.
G = 9.81
COLUMN_DT = 0.25          # s, explicit Euler substep of the column (stable: min tau ~ 2 s)


def loss_head(fwd, rev, L, S, q, hw, tw):
    """Signed stage difference (HW stage - TW stage, m) HDS-5 needs to carry q at depths hw, tw."""
    if q > 1e-12:
        return fwd.hw(q, max(tw, 0.0)) - tw + S * L
    if q < -1e-12:
        return hw + S * L - rev.hw(-q, max(hw, 0.0))
    return 0.0


def inertia_aware_reference(fwd, rev, L, S, area, rise, t, hw, tw, qref, coeff_scale=1.0):
    """(reference, column mask) for one series.

    ``fwd`` / ``rev``: the hds5 culvert and its mirror; ``area``: full barrel area (m2);
    ``rise``: barrel rise (m); ``t`` (s, may have gaps), ``hw`` / ``tw`` depths (m) above the
    inlet / outlet invert at those instants (stages linear in between), ``qref`` the
    quasi-steady flow there. ``coeff_scale`` multiplies L / (g A): the deliberately wrong
    column of the negative control."""
    t, hw, tw, qref = (np.asarray(a, dtype=float) for a in (t, hw, tw, qref))
    c = coeff_scale * L / (G * area)                       # s2/m2
    ref = qref.copy()
    drowned = (hw >= rise) & (tw >= rise)
    i, n = 0, len(t)
    while i < n:
        if not drowned[i]:
            i += 1
            continue
        q = qref[i]
        j = i + 1
        while j < n and drowned[j]:
            tt = t[j - 1]
            while tt < t[j] - 1e-9:
                h = min(COLUMN_DT, t[j] - tt)
                w = (tt + 0.5 * h - t[j - 1]) / (t[j] - t[j - 1])
                hw_m = hw[j - 1] + w * (hw[j] - hw[j - 1])
                tw_m = tw[j - 1] + w * (tw[j] - tw[j - 1])
                q += h / c * (hw_m - tw_m + S * L - loss_head(fwd, rev, L, S, q, hw_m, tw_m))
                tt += h
            ref[j] = q
            j += 1
        i = j
    return ref, drowned


def crossing_time(t, q):
    """First time q goes from > 0 to <= 0 (s), or None."""
    k = np.flatnonzero((q[:-1] > 0) & (q[1:] <= 0))
    return float(t[k[0] + 1]) if k.size else None
