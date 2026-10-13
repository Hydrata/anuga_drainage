"""2D flume rise-fall-rise + reverse for the headwall culvert (TASK-3672, AC5).

The flume (_culvert_flume.py) is driven by CSV tables so that the HEADWATER follows the 1D
series' knots (_culvert_unsteady_oracle.HW_KNOTS: HW/D 0.3 -> 3.0 -> 0.3 -> 2.5 -> 2.0, then a
tailwater rise to 1.8 D (inlet -> outlet handoff), then HW 0.5 D with TW 2.0 D (reverse)):
the upstream Inlet_operator's Q(t) = Q_hds5(HW target, TW target) + A_pond dHW/dt, which is
NEGATIVE (a sink) while the culvert flows backwards, and the downstream Time_boundary imposes
the tailwater. The recorded series (write_csv, with the per-sample control flag) is scored with
the SAME band as the 1D series against hds5 evaluated at the MEASURED headwater and tailwater
(2D region stages, not the targets), on the recorded times (the 1 s series has gaps); across the
drowned-both-ends interval (which holds the flow reversal) the reference is the rigid water
column integrated on those measured stages (_culvert_unsteady_oracle.inertia_aware_reference).
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _culvert_unsteady_oracle as u  # noqa: E402
import _hds5_oracle as hds5  # noqa: E402

L, S, N = 8.0, 0.1 / 8.0, 0.013
INLET_INVERT, OUTLET_INVERT = 0.0, -0.1
POND_AREA = 28.0 * 4.0                    # upstream of the embankment toe, m2
HW_KNOTS, TW_KNOTS = u.HW_KNOTS, u.TW_KNOTS
T_FLUME = u.T_END
PLATEAUS = dict(u.PLATEAUS)
#: the SWMM barrel starts dry: its first strides are a start-up transient, not scored.
STARTUP_S = 10.0


class Flume1D:
    """hds5 quasi-steady evaluation at given depths for the flume culvert (any diameter)."""

    def __init__(self, diameter):
        self.d = diameter
        self.fwd = hds5.Culvert("circ", diameter, L=L, S=S, n=N, inlet="rcp_sq_hw")
        self.rev = hds5.Culvert("circ", diameter, L=L, S=-S, n=N, inlet="rcp_sq_hw")
        self.area_full = self.fwd.geom.Afull

    def point(self, hw, tw):
        """(signed Q, regime, barrel volume m3) at headwater / tailwater depths (m)."""
        if hw >= tw - S * L:
            culvert, head, tail, sign = self.fwd, hw, tw, 1.0
        else:
            culvert, head, tail, sign = self.rev, tw, hw, -1.0
        head, tail = max(head, 0.0), max(tail, 0.0)
        q = culvert.q(head, tail)
        if q <= 1e-9:
            return 0.0, "zero", 0.0
        regime = culvert.hw(q, tail, detail=True)[3]
        yn = hds5.normal_depth(culvert.geom, q, culvert.S, N)
        full = tail >= self.d or regime == "outlet-full" or yn is None
        fill = 1.0 if full else culvert.geom.A(yn) / self.area_full
        return sign * q, regime, fill * self.area_full * L


def tables(diameter):
    """(inflow table, tailwater table), each an (n, 2) array of (t_s, value)."""
    ref = Flume1D(diameter)
    t = np.arange(0, T_FLUME + 1)
    hw = np.interp(t, *zip(*HW_KNOTS)) * diameter
    tw = np.interp(t, *zip(*TW_KNOTS)) * diameter
    q = np.array([ref.point(h, w)[0] for h, w in zip(hw, tw)])
    inflow = q + POND_AREA * np.gradient(hw)
    return np.column_stack([t, inflow]), np.column_stack([t, OUTLET_INVERT + tw])


def write_tables(directory, diameter):
    inflow, tailwater = tables(diameter)
    paths = []
    for name, data, header in (("inflow.csv", inflow, "t_s,q_m3s"),
                               ("tailwater.csv", tailwater, "t_s,stage_m")):
        path = os.path.join(directory, name)
        np.savetxt(path, data, delimiter=",", header=header, comments="")
        paths.append(path)
    return paths


def run_flume(engine, diameter, directory, mode=1):
    """Run the unsteady flume in a subprocess; returns (info dict, series csv path)."""
    os.makedirs(directory, exist_ok=True)
    inflow_csv, tw_csv = write_tables(directory, diameter)
    out = os.path.join(directory, "run.json")
    series_dir = os.path.join(directory, "series")
    r = subprocess.run(
        [sys.executable, "-I", os.path.join(HERE, "_culvert_flume.py"), str(mode), engine,
         str(T_FLUME), out, "--diameter", str(diameter), "--inflow-csv", inflow_csv,
         "--tailwater-csv", tw_csv, "--series-dir", series_dir,
         "--initial-hw-depth", str(HW_KNOTS[0][1] * diameter)],
        capture_output=True, text=True, timeout=3000)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    with open(out) as f:
        info = json.load(f)
    return info, os.path.join(series_dir, sorted(os.listdir(series_dir))[0])


def read_series(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return dict(t=np.array([float(r["time_s"]) for r in rows]),
                q=np.array([float(r["q_m3s"]) for r in rows]),
                hw=np.array([float(r["hw_level_m"]) for r in rows]) - INLET_INVERT,
                tw=np.array([float(r["tw_level_m"]) for r in rows]) - OUTLET_INVERT,
                control=[r["control"] for r in rows])


def score_series(series, diameter, plateaus=PLATEAUS, coeff_scale=1.0):
    """Apply the 1D band to a recorded 2D series; returns a JSON-able dict."""
    ref = Flume1D(diameter)
    keep = series["t"] >= STARTUP_S
    t, q, hw, tw = (series[k][keep] for k in ("t", "q", "hw", "tw"))
    pts = [ref.point(h, w) for h, w in zip(hw, tw)]
    q_qs = np.array([p[0] for p in pts])
    vref = np.array([p[2] for p in pts])
    # inertia-aware reference: the rigid column integrated on the MEASURED stages wherever both
    # ends are drowned (_culvert_unsteady_oracle.inertia_aware_reference)
    qref, _column = u.inertia_aware_reference(ref.fwd, ref.rev, L, S, ref.area_full, diameter, t,
                                              hw, tw, q_qs, coeff_scale=coeff_scale)
    segments = {**u.SEGMENTS, **u.E_SPLIT}
    res = u.score_band(t, q, qref, vref, plateaus, segments=segments, hw=hw, rise=diameter,
                       regime=[p[1] for p in pts])
    # negative controls: the quasi-steady reference, and the wrong columns (L/(gA) x2, x0.5)
    controls = {"quasi_steady": q_qs}
    if coeff_scale == 1.0:
        for scale in u.WRONG_COLUMN_SCALES:
            controls[f"column_x{scale}"] = u.inertia_aware_reference(
                ref.fwd, ref.rev, L, S, ref.area_full, diameter, t, hw, tw, q_qs,
                coeff_scale=scale)[0]
    scored = {name: u.score_band(t, q, r, vref, plateaus, segments) for name, r in controls.items()}
    res["controls_reference"] = {name: {k: v["violations"] for k, v in sc["segments"].items()}
                                 for name, sc in scored.items()}
    res["plateaus"] = scored["quasi_steady"]["plateaus"]     # a hold is steady: Q_qs is the physics
    res["crossing_s"] = dict(coupler=u.crossing_time(t, q), column=u.crossing_time(t, qref),
                             quasi_steady=u.crossing_time(t, q_qs))
    res["regimes"] = sorted({p[1] for p in pts})
    res["peak_hw_m"] = float(hw.max())
    res["startup_excluded_s"] = STARTUP_S
    # peak headwater of every forward plateau vs the HW hds5 needs for the measured Q and TW
    peaks = {}
    for name, (a, b) in plateaus.items():
        m = (t > a) & (t <= b)
        q_m, tw_m = float(q[m].mean()), float(tw[m].mean())
        if q_m > 0:
            need = float(ref.fwd.hw(q_m, max(tw_m, 0.0)))
            peaks[name] = dict(hw_peak_m=float(hw[m].max()), hw_hds5_m=need,
                               diff_mm=float(1000 * (hw[m].max() - need)))
    res["plateau_peak_hw"] = peaks
    return res
