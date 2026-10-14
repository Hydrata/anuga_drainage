"""TASK-3700 AC1: the coupler records the flow ACTUALLY MOVED, not SWMM's request.

The case is the TASK-3699 localhost defect (inverts below the exchange regions' mean bed;
_buried_invert.py). Before the fix the series reported SWMM's 4.2 m3/s from bone-dry ground
(requested 492 m3 in 120 s, applied 0). Each run is a subprocess (an ANUGA/MPI abort kills
the interpreter silently).
"""
import json
import os
import subprocess
import sys

import pytest

pytest.importorskip("pyswmm")

HERE = os.path.dirname(os.path.abspath(__file__))


def _run(depth, tmp_path):
    out = tmp_path / f"buried_{depth}.json"
    r = subprocess.run([sys.executable, "-I", os.path.join(HERE, "_buried_invert.py"), str(depth), str(out)],
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    return json.loads(out.read_text())


def test_dry_buried_invert_reports_zero_moved_flow(tmp_path):
    r = _run(0.0, tmp_path)
    assert r["requested"] > 100.0          # the defect is still there underneath: SWMM asks ...
    assert r["q_req_max"] > 1.0            # ... (kept as the q_requested_m3s diagnostic)
    assert r["q_max"] == 0.0               # ... but nothing moves, and nothing is reported
    assert r["applied"] == 0.0 and r["sum_q_dt"] == 0.0
    assert r["v0"] == 0.0 and r["v1"] == 0.0


def test_thin_film_moved_flow_is_applied_volume_over_dt(tmp_path):
    r = _run(0.005, tmp_path)
    assert r["requested"] > 10 * r["applied"] > 0.0   # SWMM asks far more than the film holds
    # every step is a sample: the series integrates to exactly the volume moved
    assert r["sum_q_dt"] == pytest.approx(r["applied"], rel=1e-9)
    assert r["moved_le_requested"]
    assert r["v1"] == pytest.approx(r["v0"], rel=1e-6)
    assert r["t_first"] > 0.0                         # stamped at the END of the first step
