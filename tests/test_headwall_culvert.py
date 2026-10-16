"""TASK-3630: headwall culvert backend, design 1 (host backend).

AC1 (1D oracle vs hds5.py, 2 %): inlet control is enforced by the coupler (TASK-3672,
mechanism C2: culvert code 0 in the .inp, SWMM's barrel flow capped at the HDS-5 inlet
rating), so both controls are hard gates. AC3 (mode 1 vs mode 2 peak
HW within 1 mm) and AC5 (ANUGA-side volume = sum(Q dt) clipped to the water
present) run real ANUGA simulations, each in its own subprocess (an ANUGA/MPI
abort kills the interpreter silently).
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("pyswmm")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from anuga_drainage.culverts import (  # noqa: E402
    ENTRANCE_TYPES, HeadwallCulvert, entrance_types_for, steady_flows, attach_culverts,
    HeadwallCulvertCoupler)
import _culvert_oracle as oracle  # noqa: E402
import _culvert_flume as flume  # noqa: E402


# ---------------------------------------------------------------- catalogue / spec
def test_headwall_fourteen_valid_entrance_combinations():
    assert len(ENTRANCE_TYPES) == 14
    combos = {(e.value, e.shape, e.material) for e in ENTRANCE_TYPES}
    assert len(combos) == 14
    for shape in ("box", "circular"):
        for material in ("concrete", "corrugated_metal"):
            values = [e.value for e in entrance_types_for(shape, material)]
            assert "square_edge_headwall" in values      # the default is always valid


def test_headwall_spec_from_d10_dict_defaults_and_derived():
    c = HeadwallCulvert.from_dict(dict(
        name="a", shape="box", width_m=1.5, height_m=0.9, coordinates=[[0, 0], [30, 40]],
        upstream_invert_m=10.0, downstream_invert_m=9.5, barrels=3))
    assert c.length_m == pytest.approx(50.0)
    assert c.slope == pytest.approx(0.01)
    assert c.wall_thickness_m == pytest.approx(0.20)
    assert c.footprint_width_m == pytest.approx(3 * 1.5 + 4 * 0.2)
    assert c.n == pytest.approx(0.013)
    assert c.entrance.swmm_code == 10 and c.entrance.ke == 0.5
    o = HeadwallCulvert.from_dict(dict(
        name="b", shape="circular", diameter_m=0.6, coordinates=[[0, 0], [30, 40]],
        upstream_invert_m=1.0, downstream_invert_m=0.9, material="corrugated_metal",
        manning_n=0.03, length_override_m=60.0))
    assert o.length_m == 60.0 and o.n == 0.03 and o.entrance.hds5 == "cmp_hw"


def test_headwall_multibarrel_is_n_times_single_barrel():
    e = [t for t in ENTRANCE_TYPES if t.hds5 == "box_sq_90hw"][0]
    stages = [(10.0 + 0.9 * x, 9.6) for x in (0.6, 1.5, 2.5)]
    q1 = [q for q, _ in steady_flows(oracle.culvert_for(e, 20.0, 0.02, barrels=1), stages)]
    q4 = [q for q, _ in steady_flows(oracle.culvert_for(e, 20.0, 0.02, barrels=4), stages)]
    np.testing.assert_allclose(np.array(q4), 4 * np.array(q1), rtol=1e-3)


def test_headwall_rejects_distributed_domain():
    class FakeParallelDomain:
        numproc = 2
    with pytest.raises(NotImplementedError, match="distribute"):
        HeadwallCulvertCoupler(FakeParallelDomain(), [flume.culvert_dict()], "/nonexistent")


# ---------------------------------------------------------------- AC1 oracle
@pytest.fixture(scope="module")
def oracle_rows():
    return oracle.run_grid()


def test_oracle_outlet_control_within_2pct(oracle_rows):
    rows = [r for r in oracle_rows if r["scenario"] == "outlet"]
    assert len(rows) == 14 * len(oracle.SCENARIOS["outlet"]["hw_d"])
    assert {r["regime_hds5"] for r in rows} == {"outlet-full"}
    bad = [(r["hds5"], r["hw_d"], r["err_pct"]) for r in rows if not r["pass_2pct"]]
    assert bad == []


def test_oracle_reference_step_detection():
    """The hds5 HW(Q) step at the critical slope (cmp_hw, L 20 m, S 2 %, free outlet) is found
    (against the converged march) and switches the point to Q space; a smooth neighbourhood
    stays in HW space; a real miss still fails."""
    e = [t for t in ENTRANCE_TYPES if t.hds5 == "cmp_hw"][0]
    ref = oracle.reference_for(e, 20.0, 0.02)
    assert oracle.hw_jump(ref, 1.235, 0.0, 0.02 * 1.08) > 0.05      # the 5.4 % HW step
    assert oracle.hw_jump(ref, 0.99, 0.0, 0.02 * 0.9) == 0.0        # HW/D 1.0: smooth
    # a Q on the converged step's flat side (-0.25 % in Q) passes in Q space
    s = oracle.score_point(ref, 1.2138, 1.08, 0.0)
    assert s["score_space"] == "q" and s["pass_2pct"] and abs(s["err_q_pct"]) < 0.5 < abs(s["err_pct"])
    # the stock C2 flow 1.2434 is +2.26 % in Q, 2.13 % from the step: HW space, FAILS (+4.65 %)
    s = oracle.score_point(ref, 1.2434, 1.08, 0.0)
    assert s["score_space"] == "hw" and not s["pass_2pct"] and s["err_pct"] > 4.0
    # the same rule still FAILS a real Q miss on the step: -5 % in Q
    assert not oracle.score_point(ref, 0.95 * 1.2169, 1.08, 0.0)["pass_2pct"]
    # the vendored 80-step march places the step 0.74 % higher: the default is NOT the evidence
    assert oracle.hw_jump(ref, 1.2249, 0.0, 0.02 * 1.08) > 0.05
    assert oracle.CONVERGED_NSTEPS >= 800


def test_oracle_inlet_control_within_2pct(oracle_rows):
    rows = [r for r in oracle_rows if r["scenario"] == "inlet"]
    bad = [(r["hds5"], r["hw_d"], r["err_pct"]) for r in rows if not r["pass_2pct"]]
    assert bad == []


# ---------------------------------------------------------------- entrance energy (T1)
#: rough, mildly sloped, partly full barrels (HDS-5 outlet-partial): cmp_hw on the 8 m flume
#: (S 1.25 %) at HW/D 0.435 / 0.57 / 0.84 / 0.975 and on 20 m S 0.5 % at HW/D 0.5 / 0.9
OUTLET_PARTIAL_POINTS = [(8.0, 0.0125, hw) for hw in (0.3915, 0.513, 0.756, 0.8775)] + \
    [(20.0, 0.005, hw) for hw in (0.45, 0.81)]


def _outlet_partial_errors(**engine_kw):
    import functools

    from anuga_drainage.culverts import rating, swmm_engine
    e = [t for t in ENTRANCE_TYPES if t.hds5 == "cmp_hw"][0]
    errs = []
    orig = rating.SwmmHeadwallEngine
    rating.SwmmHeadwallEngine = functools.partial(swmm_engine.SwmmHeadwallEngine, **engine_kw)
    try:
        for L, S, hw in OUTLET_PARTIAL_POINTS:
            c = oracle.culvert_for(e, L, S)
            ref = oracle.hds5.Culvert("circ", oracle.D, L=L, S=S, n=c.n, inlet="cmp_hw")
            assert ref.hw(ref.q(hw, 0.0), 0.0, detail=True)[3] == "outlet-partial"
            (q, _), = steady_flows(c, [(oracle.INV + hw, c.downstream_invert_m)])
            errs.append(100.0 * (q / ref.q(hw, 0.0) - 1.0))
    finally:
        rating.SwmmHeadwallEngine = orig
    return errs


def test_entrance_energy_outlet_partial_within_1pct():
    """The pond-to-barrel entrance energy (swmm_engine.EntranceEnergy) puts SWMM's partly-full
    barrel on the HDS-5 backwater (gvf) flow: MEASURED -0.6..+0.4 %."""
    errs = _outlet_partial_errors()
    assert max(abs(x) for x in errs) <= 1.0, errs


def test_entrance_energy_negative_control_stock_losses_over_deliver():
    """NEGATIVE CONTROL: the stock engine (pond stage on a piezometric junction, Kentry ke,
    Kexit 1.0, PARTIAL damping) over-delivers in the same points (MEASURED +3.4..+26 %),
    so the gate above can fail."""
    errs = _outlet_partial_errors(entrance_energy=False)
    assert min(errs) > 3.0, errs


def test_inlet_gate_point_cmp_hw_1p2_needs_the_entrance_energy():
    """The one inlet point decided by the reference step (cmp_hw, L 20 m, S 2 %, HW/D 1.2, free
    outlet), scored against the converged march: the entrance-energy engine passes (in Q space,
    about -0.2 %), the stock engine (entrance_energy=False: 1.2434 m3/s, +2.26 % in Q) FAILS,
    so the converged-march rule does not green the stock engine."""
    import functools

    from anuga_drainage.culverts import rating, swmm_engine
    e = [t for t in ENTRANCE_TYPES if t.hds5 == "cmp_hw"][0]
    c = oracle.culvert_for(e, 20.0, 0.02)
    ref = oracle.reference_for(e, 20.0, 0.02)
    hw = 1.2 * oracle.D
    stages = [(oracle.INV + hw, c.downstream_invert_m)]
    (q_ee, _), = steady_flows(c, stages)
    orig = rating.SwmmHeadwallEngine
    rating.SwmmHeadwallEngine = functools.partial(swmm_engine.SwmmHeadwallEngine,
                                                  entrance_energy=False)
    try:
        (q_stock, _), = steady_flows(c, stages)
    finally:
        rating.SwmmHeadwallEngine = orig
    s_ee, s_stock = oracle.score_point(ref, q_ee, hw, 0.0), oracle.score_point(ref, q_stock, hw, 0.0)
    assert s_ee["pass_2pct"] and s_ee["score_space"] == "q" and abs(s_ee["err_q_pct"]) < 0.5, s_ee
    assert not s_stock["pass_2pct"] and s_stock["err_pct"] > 4.0, s_stock


# ---------------------------------------------------------------- ANUGA runs
def _flume(mode, engine, final_time, tmp_path):
    out = tmp_path / f"flume_{mode}_{engine}.json"
    r = subprocess.run([sys.executable, "-I", os.path.join(HERE, "_culvert_flume.py"),
                        str(mode), engine, str(final_time), str(out)],
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    return json.loads(out.read_text())


@pytest.fixture(scope="module")
def flume_runs(tmp_path_factory):
    d = tmp_path_factory.mktemp("flume")
    return {m: _flume(m, "swmm", 120.0, d) for m in (1, 2)}


def test_headwall_mode1_mode2_peak_hw_within_1mm(flume_runs):
    m1, m2 = flume_runs[1], flume_runs[2]
    if m2["has_sync_bracket_api"]:               # ANUGA 4 (Hydrata engine); PyPI anuga 3.3.7 has no device path
        assert m2["cpu_sync_bracket"] is True    # design 1: the coupler forces the host bracket
    assert m1["peak_hw"] > 0.3                   # the culvert really ran surcharged-ish
    assert abs(m1["peak_hw"] - m2["peak_hw"]) <= 0.001


def test_headwall_finalize_reports_the_mass_continuity(flume_runs):
    """TASK-3741: finalize carries SWMM's raw % (unchanged key) AND the mass figure with its
    ledger, and the two differ by exactly the outfall-end change (no booked losses here).

    The flume starts DRY: the bore reaching the inlet lifts the imposed headwater ~0.5 m in
    one stride, and SWMM's wide approach dummy books ~600 m3 in and most of it back out in
    that second with a one-off residual of ~-87 m3 (constant to 0.1 m3 for the rest of the
    run, MEASURED, deploy spike engine-proofs.md "TASK-3741"). That is a real SWMM-internal
    error the mass figure must SHOW (about -12 %), not an outfall-end artefact, so no
    "< 1 %" is asserted on this flume."""
    for r in flume_runs.values():
        assert isinstance(r["swmm_routing_error_pct"], float)
        mb = r["swmm_mass_balance"]
        assert r["swmm_mass_error_pct"] == mb["mass_error_pct"]
        assert mb["inflow_m3"] > 10.0 and mb["losses_m3"] == 0.0, mb
        assert mb["swmm_error_pct"] == pytest.approx(r["swmm_routing_error_pct"], abs=0.01), mb
        dw = mb["final_outfall_end_m3"] - mb["initial_outfall_end_m3"]
        assert mb["mass_error_pct"] == pytest.approx(
            mb["swmm_error_pct"] + 100.0 * dw / mb["denominator_m3"], abs=1e-9), mb


def test_headwall_flume_volume_exact(flume_runs):
    for r in flume_runs.values():
        # every m3 drawn at the inlet arrives at the outlet, and nothing was clipped here
        assert r["op_out_applied"] == pytest.approx(-r["op_in_applied"], rel=1e-12, abs=1e-12)
        assert r["applied"] == pytest.approx(r["requested"], rel=1e-12)
        assert r["applied"] > 10.0
        # the 2D domain holds exactly what the upstream inflow put in
        assert r["v1"] - r["v0"] == pytest.approx(r["inflow"], rel=1e-9, abs=1e-9)


def test_headwall_volume_clipped_to_available(tmp_path):
    import anuga
    domain = anuga.rectangular_cross_domain(60, 8, len1=60.0, len2=4.0)
    domain.set_datadir(str(tmp_path))
    domain.set_quantity("elevation", flume.bed, location="centroids")
    domain.set_quantity("stage", lambda x, y: np.where(x < 28.0, 0.05, -0.1),
                        location="centroids")
    # the available clip on its own (the TASK-3771 equalisation cap is tested separately)
    cs = attach_culverts(domain, [flume.culvert_dict()], workdir=str(tmp_path),
                         equalisation_factor=None)
    src, dst = cs.ops_in[0], cs.ops_out[0]
    have = src.inlet.get_total_water_volume()
    assert have > 0
    domain.timestep = 0.5
    moved = cs._exchange(0, 1000.0, 0.5)               # asks for 500 m3, far more than present
    assert moved == pytest.approx(have, rel=1e-12)
    assert src.inlet.get_total_water_volume() == pytest.approx(0.0, abs=1e-12)
    assert dst.total_applied_volume == pytest.approx(have, rel=1e-12)
    assert cs._exchange(0, 1000.0, 0.5) == 0.0          # now dry: nothing moves
    cs.finalize()


def test_headwall_exchange_capped_at_half_the_level_and_never_against_head(tmp_path):
    """TASK-3771: in the live coupler a huge request moves EQUALISATION_FACTOR x the volume
    that levels the two regions, and a request against the head moves nothing."""
    import anuga
    from anuga_drainage.culverts.coupler import EQUALISATION_FACTOR, equalising_volume
    domain = anuga.rectangular_cross_domain(60, 8, len1=60.0, len2=4.0)
    domain.set_datadir(str(tmp_path))
    domain.set_quantity("elevation", flume.bed, location="centroids")
    domain.set_quantity("stage", lambda x, y: np.where(x < 28.0, 0.05, -0.05),
                        location="centroids")             # both ends wet, inlet 0.1 m higher
    cs = attach_culverts(domain, [flume.culvert_dict()], workdir=str(tmp_path))
    src, dst = cs.ops_in[0], cs.ops_out[0]
    domain.timestep = 0.5
    assert cs._exchange(0, -1000.0, 0.5) == 0.0          # outlet -> inlet: against the head
    assert cs.against_head_steps[0] == 1
    cs.q_held[0] = -0.01                                  # SWMM's barrel still runs that way
    hw, tw = cs.stages()
    assert hw[0] - tw[0] > 0.05                           # ...but against a 0.1 m adverse head:
    assert cs._exchange(0, -1000.0, 0.5) == 0.0           # beyond the inertia tolerance (3771)
    cs.inertia_max_adverse_head_m = 0.2                   # within a (wider) tolerance it runs
    before = cs.ops_out[0].inlet.get_total_water_volume()
    assert cs._exchange(0, -1000.0, 0.5) == pytest.approx(-0.005, rel=1e-9)   # inertia only
    assert before - cs.ops_out[0].inlet.get_total_water_volume() == pytest.approx(0.005, rel=1e-9)
    cs.equalisation_strict = True                         # strict: not even SWMM's own flow
    assert cs._exchange(0, -1000.0, 0.5) == 0.0
    cs.q_held[0] = 0.0
    v_src, v_dst = src.inlet.get_total_water_volume(), dst.inlet.get_total_water_volume()
    v_eq = equalising_volume(cs.fill_in[0], cs.fill_out[0], cs.fill_joint[0], v_src, v_dst)
    assert 0.0 < v_eq < v_src
    moved = cs._exchange(0, 1000.0, 0.5)
    assert moved == pytest.approx(EQUALISATION_FACTOR * v_eq, rel=1e-9)
    hw, tw = cs.stages()
    assert hw[0] > tw[0]                                  # still above: no overshoot
    cs.finalize()


def test_headwall_series_csv_has_control_column(tmp_path):
    """The per-sample control flag is in the series and in write_csv's output."""
    import csv
    for engine, allowed in (("swmm", {"barrel", "inlet", "reverse-inlet"}), ("boyd-as-coded", {"n/a"})):
        out = tmp_path / f"{engine}.json"
        series = tmp_path / f"series_{engine}"
        r = subprocess.run([sys.executable, "-I", os.path.join(HERE, "_culvert_flume.py"), "1",
                            engine, "40", str(out), "--series-dir", str(series)],
                           capture_output=True, text=True, timeout=900)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
        (path,) = list(series.iterdir())
        with open(path) as f:
            rows = list(csv.DictReader(f))
        assert list(rows[0]) == ["time_s", "q_m3s", "hw_level_m", "tw_level_m", "control",
                                "q_requested_m3s"]
        assert len(rows) >= 30 and {row["control"] for row in rows} <= allowed
        if engine == "swmm":
            assert {row["control"] for row in rows} & {"barrel", "inlet"}


def test_headwall_boyd_engine_switch_runs(tmp_path):
    r = _flume(1, "boyd-as-coded", 30.0, tmp_path)
    assert r["engine"] == "boyd-as-coded" and r["n_records"] >= 10
    assert r["peak_q"] > 0.1 and r["v1"] - r["v0"] == pytest.approx(r["inflow"], rel=1e-9)


def test_headwall_string_barrels_and_swmm_end_time_guard(tmp_path):
    from anuga_drainage.culverts import SwmmHeadwallEngine
    d = dict(flume.culvert_dict(), barrels="3.0")
    c = HeadwallCulvert.from_dict(d)
    assert c.barrels == 3
    eng = SwmmHeadwallEngine([c], str(tmp_path), duration_s=3)
    for _ in range(3):
        eng.set_stages([0.5], [-0.1])
        eng.stride()
    with pytest.raises(RuntimeError, match="END_TIME"):
        eng.stride()
    eng.close()
