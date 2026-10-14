"""TASK-3672 AC3, AC5, AC6: coupler-enforced HDS-5 inlet control through rise-fall-rise,
the inlet -> outlet handoff and reverse flow.

1D: 14 catalogue entrances x {20 m S 2 %, 20 m S 0.5 %, flume 8 m} = 42 series of 4800 s
(see _culvert_unsteady_oracle.py for the series and the band). 2D: the flume driven by an
inflow CSV and a Time_boundary tailwater (see _culvert_flume_unsteady.py), SWMM engine on the
0.9 m pipe (the AC) and on a 0.6 m pipe (positive control), Boyd as coded on the 0.6 m pipe
(the must-fail negative control).

Results (per-case CSVs, unsteady-summary.json, flume series + summary) go to ``--results-dir``.
"""
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

pytest.importorskip("pyswmm")

# ~10 min (42 SWMM series + three 2D flume runs): its own scheduled CI lane (TASK-3674)
pytestmark = pytest.mark.slow

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import _culvert_flume_unsteady as fu  # noqa: E402
import _culvert_unsteady_oracle as u  # noqa: E402

from anuga_drainage.culverts import (  # noqa: E402
    CONTROL_NAMES, ENTRANCE_TYPES, HeadwallCulvertCoupler, InletControl)

CASES = len(ENTRANCE_TYPES) * len(u.GEOMETRIES)


# ---------------------------------------------------------------- 1D series (AC3)
@pytest.fixture(scope="module")
def unsteady(results_dir):
    out = u.run_all()
    folder = results_dir / "unsteady-1d"
    folder.mkdir(exist_ok=True)
    for res, s in out:
        name = res["case"].replace("@", "_")
        with open(folder / f"{name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "hw_m", "tw_m", "q_m3s", "q_swmm_m3s", "q_hds5_m3s", "control",
                        "regime_hds5", "q_ref_m3s"])
            for k in range(len(s["t"])):
                w.writerow([int(s["t"][k]), f"{s['hw'][k]:.5f}", f"{s['tw'][k]:.5f}",
                            f"{s['q'][k]:.6f}", f"{s['q_raw'][k]:.6f}", f"{s['q_qs'][k]:.6f}",
                            CONTROL_NAMES[int(s["flag"][k])], s["regime"][k], f"{s['q_ref'][k]:.6f}"])
    with open(results_dir / "unsteady-summary.json", "w") as f:
        json.dump([res for res, _ in out], f, indent=1)
    # D20: the reference-step windows' excluded instants, per case (the committed list)
    with open(results_dir / "step-window-excluded.json", "w") as f:
        json.dump({res["case"]: dict(windows=res["step_windows"], excluded=res["step_excluded"],
                                     would_violate=res["step_excluded_violations"],
                                     runs=res["step_excluded_runs"])
                   for res, _ in out if res["step_windows"]}, f, indent=1)
    return [res for res, _ in out]


def _fail_lines(results, pick):
    return [f"{r['case']}: {pick(r)}" for r in results]


def test_unsteady_runs_all_42_cases(unsteady):
    assert len(unsteady) == CASES == 42
    assert {r["geometry"] for r in unsteady} == set(u.GEOMETRIES)
    assert len({r["hds5"] for r in unsteady}) == 14


@pytest.mark.parametrize("segment", ["A", "B", "C", "D"])
def test_unsteady_forward_segment_inside_band(unsteady, segment):
    """Rise (part-full -> full), fall, rise again, inlet -> outlet handoff: every instant."""
    bad = [f"{r['case']}: {r['segments'][segment]['violations']} instants, worst "
           f"{r['segments'][segment]['worst_ratio']:.2f}x tol"
           for r in unsteady if r["segments"][segment]["violations"]]
    assert bad == [], f"{len(bad)}/42 cases outside the band in segment {segment}:\n" + "\n".join(bad)


def test_unsteady_reverse_segment_inside_band(unsteady):
    """Segment E: HW 2.0 D -> 0.5 D, TW 1.8 D -> 2.0 D, mirrored oracle."""
    bad = [f"{r['case']}: {r['segments']['E']['violations']} instants, worst "
           f"{r['segments']['E']['worst_ratio']:.2f}x tol"
           for r in unsteady if r["segments"]["E"]["violations"]]
    assert bad == [], f"{len(bad)}/42 cases outside the band in segment E:\n" + "\n".join(bad)


def test_unsteady_reversal_crossing_inside_band(unsteady):
    """Segment E's reversal ramp (4260-4420 s) alone, against the inertia-aware reference (the
    rigid column where both ends are drowned). The reverse HOLD (4420-4800 s) is the steady
    reverse-barrel regime and is scored by test_unsteady_reverse_segment_inside_band."""
    bad = [f"{r['case']}: {r['segments']['E_cross']['violations']} instants, worst "
           f"{r['segments']['E_cross']['worst_ratio']:.2f}x tol"
           for r in unsteady if r["segments"]["E_cross"]["violations"]]
    assert bad == [], f"{len(bad)}/42 cases outside the band in the reversal:\n" + "\n".join(bad)


@pytest.mark.parametrize("control", ["quasi_steady", "column_x2.0", "column_x0.5"])
def test_unsteady_wrong_reference_fails_the_reversal(unsteady, control):
    """NEGATIVE CONTROLS for the inertia-aware reference: the quasi-steady reference (no
    inertia) and a column with L/(gA) doubled or halved must each FAIL the reversal ramp in
    every case, else passing against the column proves nothing."""
    passed = [r["case"] for r in unsteady if r["controls_reference"][control]["E_cross"] == 0]
    assert passed == [], f"{control} passed the reversal in {passed}"


def test_unsteady_column_crosses_zero_with_the_coupler(unsteady):
    """The coupler's flow reverses within the band's 2 s lag of the rigid column, and more than
    that after the quasi-steady reference (the inertial delay the quasi-steady oracle lacks)."""
    for r in unsteady:
        c = r["crossing_s"]
        assert abs(c["coupler"] - c["column"]) <= u.LAG_S, (r["case"], c)
        assert c["coupler"] - c["quasi_steady"] > u.LAG_S, (r["case"], c)


def test_unsteady_plateaus_within_2pct(unsteady):
    bad = [f"{r['case']} {name}: {p['err_pct']:+.2f} %" for r in unsteady
           for name, p in r["plateaus"].items() if abs(p["err_pct"]) > 2.0]
    assert bad == [], f"{len(bad)} plateaus outside 2 %:\n" + "\n".join(bad)


def test_unsteady_forward_plateaus_within_1p1pct(unsteady):
    """The 168 forward plateaus (A, C, D1, D2 x 42) are within 1.1 % (MEASURED max 1.06 %)."""
    errs = [(abs(p["err_pct"]), f"{r['case']} {name}") for r in unsteady
            for name, p in r["plateaus"].items() if name != "E"]
    assert len(errs) == 168
    assert max(errs)[0] <= 1.1, max(errs)


#: D20: instants excluded because the imposed HW lies inside a bisection-confirmed step of the
#: HDS-5 reference's HW(Q) (free outlet, converged march; _culvert_unsteady_oracle.
#: step_window_mask). Deterministic (reference + stage series only), asserted at its measured
#: value so the exclusion cannot grow. The per-case runs are committed in
#: results/step-window-excluded.json; 144 of the 395 would violate (reported, not hidden).
STEP_WINDOW_EXCLUDED = {
    "cmp_hw@20m_S2": 76, "cmp_mitered@20m_S2": 47, "cmp_proj@20m_S2": 86,
    "rcp_sq_hw@20m_S0.5": 71, "rcp_groove_hw@20m_S0.5": 50, "rcp_groove_proj@20m_S0.5": 26,
    "circ_bevel45@20m_S0.5": 39}


def test_unsteady_reference_step_window_exclusions_are_asserted(unsteady):
    got = {r["case"]: r["step_excluded"] for r in unsteady if r["step_excluded"]}
    assert got == STEP_WINDOW_EXCLUDED
    assert sum(got.values()) == 395
    assert sum(r["step_excluded_violations"] for r in unsteady) == 144
    # only a free-outlet forward step excludes: every excluded run sits in segments A-C
    for r in unsteady:
        assert all(hi <= 3240 for _lo, hi in r["step_excluded_runs"]), r["case"]
    # the fill/unfill exclusion is the step's own 1 s footprint and hides no violation
    assert sum(r["excluded"] for r in unsteady) == 348
    assert sum(r["excluded_violations"] for r in unsteady) == 0


def test_unsteady_outlet_partial_instants_at_most_300(unsteady):
    """Violating instants in the HDS-5 outlet-partial regime, all 42 cases (stock engine:
    7675). MEASURED 160."""
    n = sum(r["violations_by_regime"].get("outlet-partial", 0) for r in unsteady)
    assert n <= 300, n


def test_unsteady_no_violation_while_the_barrel_is_a_drowned_column(unsteady):
    """3874-4461 s (both ends drowned: the reversal ramp and the TW-to-crown hand-over): no
    violating instant in any of the 42 cases (the stock 4141-4142 s instants vanish)."""
    def inside(r, a=3874, b=4461):
        return sum(max(0, min(hi, b) - max(lo, a) + 1) for lo, hi in r["violation_runs"])
    assert [(r["case"], inside(r)) for r in unsteady if inside(r)] == []


def test_unsteady_no_chatter_excess(unsteady):
    bad = [f"{r['case']}: {r['chatter_excess']}" for r in unsteady if r["chatter_excess"] > 5]
    assert bad == []


def test_unsteady_exactly_one_reversal(unsteady):
    assert [r["case"] for r in unsteady if r["reversals"] != 1] == []


def test_unsteady_control_flags_cover_inlet_and_barrel(unsteady):
    """The per-sample control flag is live: on the steep geometries (S 2 %, flume) inlet control
    caps SWMM's barrel somewhere in every case and the barrel governs somewhere in every case."""
    for r in unsteady:
        assert r["controls"]["barrel"] > 0, r["case"]
        if r["geometry"] != "20m_S0.5":
            assert r["controls"]["inlet"] > 0, r["case"]


# ---------------------------------------------------------------- per-stride overhead (AC6)
def test_coupler_per_stride_overhead_at_most_2us(results_dir):
    """The coupler's whole per-stride cost of inlet control (HeadwallCulvertCoupler._accumulate:
    raw + capped accumulation and the control flag), one culvert, best of 7 x 20000 strides."""
    e = [t for t in ENTRANCE_TYPES if t.hds5 == "rcp_sq_hw"][0]
    c = u.culvert_for(e, 20.0, 0.02)

    class Stub:
        inlet_control = InletControl([c])
        control = None
        _accumulate = HeadwallCulvertCoupler._accumulate

    stub = Stub()
    hw, tw = [c.upstream_invert_m + 1.0], [c.downstream_invert_m]
    cap_f, cap_r = stub.inlet_control.caps(hw, tw)
    n = 20000
    measured = {}
    for flow in (1.0, 5.0, -0.5):                    # barrel, inlet-capped, reverse
        raw = np.array([flow])
        best = float("inf")
        for _ in range(7):
            q_acc, raw_acc = [0.0], np.zeros(1)
            t0 = time.perf_counter()
            for _ in range(n):
                stub._accumulate(raw, cap_f, cap_r, q_acc, raw_acc)
            best = min(best, (time.perf_counter() - t0) / n)
        measured[f"stride_us_flow_{flow}"] = best * 1e6
        assert best <= 2e-6, f"flow {flow}: {best * 1e6:.2f} us per stride"
    t0 = time.perf_counter()
    for _ in range(2000):
        stub.inlet_control.caps(hw, tw)
    caps_us = (time.perf_counter() - t0) / 2000 * 1e6      # once per ANUGA step that strides
    measured["caps_us_per_stepping_step"] = caps_us
    with open(results_dir / "stride-overhead.json", "w") as f:
        json.dump(measured, f, indent=1)
    assert caps_us <= 5.0, f"caps {caps_us:.2f} us"


def test_engine_whole_step_cost_recorded(results_dir):
    """D22: the WHOLE engine step per culvert (set_stages + stride: the entrance-energy / reverse
    state machine AND SWMM's own second) forward and reverse, recorded in stride-overhead.json.
    A recorded number, NOT a gate (the real cost check is TASK-3634 on Towradgi); only sanity
    bounds so a pathological regression cannot hide."""
    import tempfile

    from anuga_drainage.culverts import SwmmHeadwallEngine
    e = [t for t in ENTRANCE_TYPES if t.hds5 == "rcp_sq_hw"][0]
    c = u.culvert_for(e, 20.0, 0.02)
    measured = {}
    for name, hw_d, tw_d in (("forward", 1.0, 0.0), ("reverse", 0.5, 2.0)):
        hw, tw = c.upstream_invert_m + hw_d * u.D, c.downstream_invert_m + tw_d * u.D
        with tempfile.TemporaryDirectory() as d:
            eng = SwmmHeadwallEngine([c], d, duration_s=4000, initial_stages=[(hw, tw)])
            for _ in range(300):                     # settle
                eng.set_stages([hw], [tw])
                eng.stride()
            best_set, best_all = float("inf"), float("inf")
            for _ in range(5):
                t_set = t_all = 0.0
                for _ in range(500):
                    t0 = time.perf_counter()
                    eng.set_stages([hw], [tw])
                    t1 = time.perf_counter()
                    eng.stride()
                    t2 = time.perf_counter()
                    t_set += t1 - t0
                    t_all += t2 - t0
                best_set, best_all = min(best_set, t_set / 500), min(best_all, t_all / 500)
            assert (eng.flows()[0] < 0) == (name == "reverse")
            eng.close()
        measured[f"engine_set_stages_us_{name}"] = best_set * 1e6
        measured[f"engine_set_stages_plus_stride_us_{name}"] = best_all * 1e6
    path = results_dir / "stride-overhead.json"
    data = json.load(open(path)) if path.exists() else {}
    data.update(measured)
    with open(path, "w") as f:
        json.dump(data, f, indent=1)
    assert all(v < 5000.0 for v in measured.values()), measured    # sanity only: < 5 ms


# ---------------------------------------------------------------- 2D flume (AC5)
@pytest.fixture(scope="module")
def flume(tmp_path_factory, results_dir):
    base = tmp_path_factory.mktemp("flume-unsteady")
    jobs = {"swmm_0.9": ("swmm", 0.9), "swmm_0.6": ("swmm", 0.6),
            "boyd_0.6": ("boyd-as-coded", 0.6)}
    with ThreadPoolExecutor(len(jobs)) as ex:
        futures = {k: ex.submit(fu.run_flume, eng, d, str(base / k)) for k, (eng, d) in jobs.items()}
        runs = {k: f.result() for k, f in futures.items()}
    out = {}
    summary = {}
    for k, (engine, d) in jobs.items():
        info, path = runs[k]
        series = fu.read_series(path)
        res = fu.score_series(series, d)
        res["engine"], res["diameter_m"], res["run"] = engine, d, info
        out[k] = (series, res)
        summary[k] = res
        with open(results_dir / f"flume-{k}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_s", "q_m3s", "hw_depth_m", "tw_depth_m", "control"])
            for row in zip(series["t"], series["q"], series["hw"], series["tw"], series["control"]):
                w.writerow([f"{row[0]:.3f}", f"{row[1]:.6f}", f"{row[2]:.5f}", f"{row[3]:.5f}", row[4]])
    with open(results_dir / "flume-summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    return out


def test_flume_swmm_series_inside_band(flume):
    """The 0.9 m pipe, SWMM engine, every recorded instant of the rise-fall-rise + reverse run."""
    _series, res = flume["swmm_0.9"]
    assert res["reversals"] == 1
    assert res["violations"] == 0, (
        f"{res['violations']}/{res['n']} instants outside the band, worst {res['worst_ratio']:.2f}x "
        f"tol at t={res['worst_t']:.0f} s")
    assert res["chatter_excess"] <= 5


@pytest.mark.parametrize("control", ["quasi_steady", "column_x2.0", "column_x0.5"])
def test_flume_wrong_reference_fails_the_reversal(flume, control):
    """NEGATIVE CONTROLS on the 2D 0.9 m run (column integrated on the measured stages)."""
    _series, res = flume["swmm_0.9"]
    assert res["controls_reference"][control]["E_cross"] > 0, res["controls_reference"]
    c = res["crossing_s"]
    assert abs(c["coupler"] - c["column"]) <= u.LAG_S < c["coupler"] - c["quasi_steady"], c


def test_flume_swmm_plateaus_and_peak_hw(flume):
    _series, res = flume["swmm_0.9"]
    assert {n: p["err_pct"] for n, p in res["plateaus"].items() if abs(p["err_pct"]) > 2.0} == {}
    bad = {n: p["diff_mm"] for n, p in res["plateau_peak_hw"].items() if abs(p["diff_mm"]) > 30.0}
    assert bad == {}, f"peak HW more than 30 mm from hds5: {bad}"


def test_flume_swmm_small_pipe_positive_control(flume):
    _series, res = flume["swmm_0.6"]
    assert res["violations"] == 0, (res["violations"], res["worst_ratio"], res["worst_t"])


def test_flume_boyd_as_coded_fails_the_band_on_the_0p6_pipe(flume):
    """NEGATIVE CONTROL: if the band cannot fail Boyd as coded on the 0.6 m pipe, it proves
    nothing about the SWMM engine passing it."""
    _series, res = flume["boyd_0.6"]
    assert res["violations"] > 0 or any(abs(p["err_pct"]) > 2.0 for p in res["plateaus"].values()), (
        "Boyd as coded passed the band: the band is not discriminating")


def test_flume_control_flag_written_to_csv(flume):
    series, _res = flume["swmm_0.9"]
    assert set(series["control"]) <= {"barrel", "inlet", "reverse-inlet"}
    assert "inlet" in series["control"]
    boyd, _ = flume["boyd_0.6"]
    assert set(boyd["control"]) == {"n/a"}
