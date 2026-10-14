"""TASK-3631 AC6: the bridge_hecras validation case with the SWMM headwall coupler.

anuga_core's validation_tests/behaviour_only/bridge_hecras compares ANUGA stage with HEC-RAS at 12
stations; the Boyd_box_operator version scores mean correlation 0.995 and mean RMS 0.1133 m
(test_regression_bridge_hecras/test_bridge_hecras_regression.csv). Here the bridge is one 10 m x
1.3 m headwall box culvert driven by the coupler (see _bridge_hecras_swmm.py for exactly what is and
is not modelled). Gates: mean correlation >= 0.99 and global volume error < 0.1 % of the inflow volume.
RMS is REPORT-ONLY (D29, 13-10-26): a bridge modelled as a culvert scores 0.137 m against Boyd's 0.113 m,
which Boyd only reached with a tuned 'other 1.5' loss; tuning ours to the bar would be over-fitting. The
measured RMS is printed against REFERENCE_BOYD_RMS_M for the report; see TASK-3683 item 8.

Needs the anuga_core clone (HEC-RAS gauges.csv + the metric helpers): ANUGA_CORE_DIR, default
~/hydrata/anuga_core.
"""
import importlib.util
import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.environ.get("ANUGA_CORE_DIR", os.path.expanduser("~/hydrata/anuga_core"))
BEHAVIOUR = os.path.join(CORE, "validation_tests", "behaviour_only")
GAUGES = os.path.join(BEHAVIOUR, "bridge_hecras", "hecras_bridge_test", "gauges.csv")

MIN_CORRELATION = 0.99
REFERENCE_BOYD_RMS_M = 0.113   # report-only (D29), not asserted
MAX_VOLUME_ERROR = 1e-3



def _helpers():
    spec = importlib.util.spec_from_file_location("bridge_conftest", os.path.join(BEHAVIOUR, "conftest.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_bridge(engine, out_dir):
    env = dict(os.environ, OMP_NUM_THREADS="1")
    proc = subprocess.run([sys.executable, "-I", os.path.join(HERE, "_bridge_hecras_swmm.py"), engine, str(out_dir)],
                          capture_output=True, text=True, env=env, cwd=str(out_dir))
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    return json.loads(line[len("RESULT "):])


def hecras_metrics(out_dir):
    from anuga.utilities import plot_utils as util
    helpers = _helpers()
    p = util.get_output(os.path.join(str(out_dir), "channel_floodplain1.sww"), 0.001)
    pc = util.get_centroids(p, velocity_extrapolation=True)
    m = helpers.hecras_correlation_metrics(pc, GAUGES, x_channel=15.0)
    return {k: float(v[0]) for k, v in m.items()}


@pytest.mark.slow          # ~3 min 2D run: the scheduled slow lane (TASK-3674)
def test_bridge_hecras_swmm(tmp_path):
    assert os.path.exists(GAUGES), f"HEC-RAS gauges not found at {GAUGES} (set ANUGA_CORE_DIR)"
    res = run_bridge("swmm", tmp_path)
    m = hecras_metrics(tmp_path)
    rel_volume_error = abs(res["volume_error"]) / res["inflow_volume"]
    print("bridge_hecras (SWMM coupler):", json.dumps(dict(metrics=m, run=res, rel_volume_error=rel_volume_error,
                                                            reference_boyd_rms_m=REFERENCE_BOYD_RMS_M)))
    assert m["mean_hecras_correlation"] >= MIN_CORRELATION, m
    assert rel_volume_error < MAX_VOLUME_ERROR, (rel_volume_error, res)
