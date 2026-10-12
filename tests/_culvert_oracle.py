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
FIELDS = ("entrance_type", "shape", "material", "hds5", "swmm_code", "scenario", "hw_d",
          "hw_m", "tw_m", "q_swmm", "hw_hds5_m", "regime_hds5", "err_pct", "pass_2pct")


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
                hw_ref, _hi, _ho, regime = ref.hw(q, tw, detail=True)
                err = 100.0 * (hw_ref - hw) / hw
                rows.append(dict(entrance_type=e.value, shape=e.shape, material=e.material,
                                 hds5=e.hds5, swmm_code=e.swmm_code, scenario=name, hw_d=x,
                                 hw_m=round(hw, 4), tw_m=round(tw, 4), q_swmm=round(q, 5),
                                 hw_hds5_m=round(hw_ref, 4), regime_hds5=regime,
                                 err_pct=round(err, 3), pass_2pct=abs(err) <= GATE_PCT))
    return rows


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
