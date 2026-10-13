"""1D oracle grid: SWMM headwall culvert vs the independent HDS-5 calculator.

For every VALID (entrance type, shape, material) combination, two steady
scenarios at imposed stages:

* ``inlet``  - L 20 m, S 2 %, free outlet (TW at the outlet invert);
* ``outlet`` - L 60 m, S 0.1 %, submerged outlet (TW 1.2 D above the outlet invert).

SWMM's settled Q at the imposed headwater HW is fed back into ``hds5.py``,
which returns the HW it needs for that Q; the error is (HW_hds5 - HW) / HW.
The regime column is hds5.py's (inlet-unsub / -trans / -sub, outlet-full /
-partial).
"""
from __future__ import annotations

import contextlib
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import _hds5_oracle as hds5  # noqa: E402

from anuga_drainage.culverts import ENTRANCE_TYPES, HeadwallCulvert, steady_flows  # noqa: E402

D = 0.9
SPAN = 1.5
INV = 10.0
SCENARIOS = {
    "inlet": dict(L=20.0, S=0.02, tw_d=None, hw_d=(0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0)),
    "outlet": dict(L=60.0, S=0.001, tw_d=1.2, hw_d=(1.4, 1.6, 2.0, 2.5, 3.0)),
}
GATE_PCT = 2.0
#: the Q window (fraction of Q) searched for a jump of the reference's HW(Q).
JUMP_WINDOW = GATE_PCT / 100.0
#: the steady scoring marches the reference's outlet profile with this many steps. The
#: vendored oracle's default (80) places the critical-slope HW(Q) step 0.74 % above the
#: converged location (Q 1.22493 vs 1.21603 at 8000 steps on cmp_hw L 20 m S 2 %), more than
#: the 0.49 % margin of the one point the step decides, so it is not evidence there.
CONVERGED_NSTEPS = 800


@contextlib.contextmanager
def converged_march(nsteps=CONVERGED_NSTEPS):
    """Scoring-side wrapper: ``outlet_hw_gvf`` marches with ``nsteps`` instead of the
    vendored default, for the duration of the block. The vendored ``_hds5_oracle.py`` stays
    unmodified and every other user of it (the unsteady oracle) keeps its 80-step march."""
    saved = hds5.outlet_hw_gvf.__defaults__
    hds5.outlet_hw_gvf.__defaults__ = (nsteps,)
    try:
        yield
    finally:
        hds5.outlet_hw_gvf.__defaults__ = saved
FIELDS = ("entrance_type", "shape", "material", "hds5", "swmm_code", "scenario", "hw_d",
          "hw_m", "tw_m", "q_swmm", "hw_hds5_m", "regime_hds5", "err_pct", "ref_jump_m",
          "score_space", "q_hds5", "err_q_pct", "pass_2pct")


def hw_jump(ref, q, tw, tol_m, window=JUMP_WINDOW, n=41):
    """Largest DISCONTINUITY (m) of the reference's HW(Q) at tailwater ``tw`` within
    q (1 +- window), or 0.0 where HW(Q) is continuous there.

    HDS-5 computes HW = max(inlet regression, outlet energy march) and suppresses outlet
    control when the barrel profile cannot be subcritical (steep barrel, free outlet; HDS-5
    3rd ed. ch. 3 / HY-8 flow types 1 and 5: backwater that does not submerge critical depth
    at the inlet => inlet control). At the critical slope the two forms disagree for the
    same flow condition (cmp_hw L 20 m S 2 %: inlet 1.064 m, outlet Ec + ke Vc^2/2g =
    1.120 m), so HW(Q) steps while its inverse Q(HW) is continuous but FLAT across the step.
    A step is a jump that survives bisection to dQ/Q < 1e-9 (a merely steep, smooth HW(Q)
    does not). Evaluated with the converged march."""
    with converged_march():
        return _hw_jump(ref, q, tw, tol_m, window, n)


def _hw_jump(ref, q, tw, tol_m, window, n):
    qs = [q * (1 - window + 2 * window * k / (n - 1)) for k in range(n)]
    hws = [ref.hw(x, tw) for x in qs]
    worst = 0.0
    for k in range(n - 1):
        lo, hi, hlo, hhi = qs[k], qs[k + 1], hws[k], hws[k + 1]
        if abs(hhi - hlo) <= tol_m:
            continue
        while (hi - lo) / hi > 1e-9:
            mid = 0.5 * (lo + hi)
            hm = ref.hw(mid, tw)
            if abs(hm - hlo) >= abs(hhi - hm):
                hi, hhi = mid, hm
            else:
                lo, hlo = mid, hm
        if abs(hhi - hlo) > tol_m:
            worst = max(worst, abs(hhi - hlo))
    return worst


def score_point(ref, q, hw, tw):
    """The 2 % gate for one settled (Q, HW) point against the HDS-5 reference.

    HW space (the D9 protocol): (HW_hds5(Q) - HW) / HW. Where the reference's HW(Q) has a
    step larger than the HW tolerance within +-2 % of Q, an HW error is not a measure of the
    culvert (any Q on the flat side of the step reads up to the full step): the point is
    scored in Q space instead, (Q - Q_hds5(HW)) / Q_hds5 within the same 2 %. Returns a dict
    of the row's scoring fields. The reference is evaluated with the converged march."""
    with converged_march():
        return _score_point(ref, q, hw, tw)


def _score_point(ref, q, hw, tw):
    hw_ref, _hi, _ho, regime = ref.hw(q, tw, detail=True)
    err = 100.0 * (hw_ref - hw) / hw
    jump = hw_jump(ref, q, tw, GATE_PCT / 100.0 * hw)
    out = dict(hw_hds5_m=round(hw_ref, 4), regime_hds5=regime, err_pct=round(err, 3),
               ref_jump_m=round(jump, 4), score_space="hw", q_hds5="", err_q_pct="",
               pass_2pct=abs(err) <= GATE_PCT)
    if jump > 0.0:
        q_ref = ref.q(hw, tw)
        err_q = 100.0 * (q - q_ref) / q_ref
        out.update(score_space="q", q_hds5=round(q_ref, 5), err_q_pct=round(err_q, 3),
                   pass_2pct=abs(err_q) <= GATE_PCT)
    return out


def culvert_for(e, L, S, barrels=1, span=SPAN):
    d = dict(name="oracle", shape=e.shape, coordinates=[[0.0, 0.0], [L, 0.0]],
             upstream_invert_m=INV, downstream_invert_m=INV - S * L,
             entrance_type=e.value, material=e.material, barrels=barrels)
    if e.shape == "circular":
        d["diameter_m"] = D
    else:
        d.update(width_m=span, height_m=D)
    return HeadwallCulvert.from_dict(d)


def run_grid():
    rows = []
    for e in ENTRANCE_TYPES:
        for name, sc in SCENARIOS.items():
            L, S = sc["L"], sc["S"]
            c = culvert_for(e, L, S)
            tw = 0.0 if sc["tw_d"] is None else sc["tw_d"] * D
            res = steady_flows(c, [(INV + x * D, INV - S * L + tw) for x in sc["hw_d"]])
            ref = hds5.Culvert("circ" if e.shape == "circular" else "box", D,
                               B=None if e.shape == "circular" else SPAN, L=L, S=S, n=c.n,
                               inlet=e.hds5)
            for x, (q, _head) in zip(sc["hw_d"], res):
                hw = x * D
                row = dict(entrance_type=e.value, shape=e.shape, material=e.material,
                           hds5=e.hds5, swmm_code=e.swmm_code, scenario=name, hw_d=x,
                           hw_m=round(hw, 4), tw_m=round(tw, 4), q_swmm=round(q, 5))
                row.update(score_point(ref, q, hw, tw))
                rows.append(row)
    return rows


def reference_for(e, L, S):
    return hds5.Culvert("circ" if e.shape == "circular" else "box", D,
                        B=None if e.shape == "circular" else SPAN, L=L, S=S,
                        n=culvert_for(e, L, S).n, inlet=e.hds5)


def write_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    out = sys.argv[1]
    rows = run_grid()
    write_csv(rows, out)
    for regime in sorted({r["regime_hds5"] for r in rows}):
        rr = [r for r in rows if r["regime_hds5"] == regime]
        bad = [r for r in rr if not r["pass_2pct"]]
        print(f"{regime:15s} points {len(rr):3d} fail {len(bad):3d} "
              f"max|err| {max(abs(r['err_pct']) for r in rr):6.2f} %")
