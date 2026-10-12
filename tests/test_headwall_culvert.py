"""TASK-3630: headwall culvert backend, design 1 (host backend).

AC1 (1D oracle vs hds5.py, 2 %) is split by control: outlet control is a hard
gate; inlet control is a strict xfail that RECORDS the measured gate failure
(results/oracle-1d.csv in the deploy spike folder). AC3 (mode 1 vs mode 2 peak
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


@pytest.mark.xfail(strict=True, reason=(
    "D9 1D gate MEASURED FAILING for inlet control (12-10/13-10-26): SWMM 5.2.4's culvert "
    "code matches hds5.py exactly as a formula, but inside dynamic wave it is evaluated at a "
    "reset upstream head and dropped once the barrel is full, so HW errors reach -50..+21 % "
    "(cmp_mitered, cmbox_thinproj, all boxes unsubmerged, circles at HW/D 0.3 and >= 2.5). "
    "A result for the operator's design-1 / design-2 / C decision, NOT a bug to hide."))
def test_oracle_inlet_control_within_2pct(oracle_rows):
    rows = [r for r in oracle_rows if r["scenario"] == "inlet"]
    bad = [(r["hds5"], r["hw_d"], r["err_pct"]) for r in rows if not r["pass_2pct"]]
    assert bad == []


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
    assert m2["cpu_sync_bracket"] is True        # design 1: the coupler forces the host bracket
    assert m1["peak_hw"] > 0.3                   # the culvert really ran surcharged-ish
    assert abs(m1["peak_hw"] - m2["peak_hw"]) <= 0.001


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
    cs = attach_culverts(domain, [flume.culvert_dict()], workdir=str(tmp_path))
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


def test_headwall_boyd_engine_switch_runs(tmp_path):
    r = _flume(1, "boyd-as-coded", 30.0, tmp_path)
    assert r["engine"] == "boyd-as-coded" and r["n_records"] >= 29
