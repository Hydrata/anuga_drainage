"""A small ANUGA flume with one headwall culvert through an embankment.

Run as a script (each simulation in its own process: an ANUGA/MPI abort
kills the interpreter):

    python -I _culvert_flume.py <mode 1|2> <engine> <final_time_s> <out.json>
        [--diameter M] [--inflow-csv t,q] [--tailwater-csv t,stage] [--series-dir DIR]
        [--initial-hw-depth M] [--yieldstep S]

Bed 0 m upstream of an embankment (x 28-32 m, crest 3 m), -0.1 m downstream;
4 m wide, 60 m long, reflective walls. By default a 1.5 m3/s inflow at the upstream
end for the first 60 s raises the headwater; a 0.9 m RCP (square edge, headwall)
runs from (26, 2) to (34, 2) under the embankment.

The unsteady variant (TASK-3672) drives the same flume from CSVs: ``--inflow-csv`` is
the upstream Inlet_operator's Q(t) table (columns t_s, q_m3s), ``--tailwater-csv`` a
Time_boundary stage(t) table (t_s, stage_m) on the downstream (right) edge; the culvert
series (with its per-sample control flag) is written with ``CulvertSet.write_csv`` into
``--series-dir``.
"""
import argparse
import json
import os
import sys
import tempfile

# -I drops PYTHONPATH: import the anuga_drainage of THIS checkout (not whatever the venv
# has installed), so a scratch copy tests its own code.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src"))

DEFAULT_DIAMETER = 0.9


def culvert_dict(diameter=DEFAULT_DIAMETER):
    return dict(name="flume RCP", shape="circular", diameter_m=diameter,
                coordinates=[[26.0, 2.0], [34.0, 2.0]], upstream_invert_m=0.0,
                downstream_invert_m=-0.1, entrance_type="square_edge_headwall",
                material="concrete", barrels=1)


def bed(x, y):
    import numpy as np
    z = np.where(x < 28.0, 0.0, -0.1)
    return np.where((x >= 28.0) & (x <= 32.0), 3.0, z)


def _table(path):
    import numpy as np
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1]


def run(mode, engine, final_time, workdir, *, diameter=DEFAULT_DIAMETER, inflow_csv=None,
        tailwater_csv=None, series_dir=None, initial_hw_depth=0.0, yieldstep=10.0):
    import anuga
    import numpy as np
    from anuga_drainage.culverts import attach_culverts

    domain = anuga.rectangular_cross_domain(120, 8, len1=60.0, len2=4.0)
    domain.set_name("flume")
    domain.set_datadir(workdir)
    domain.set_store(False)
    domain.set_quantity("elevation", bed, location="centroids")
    domain.set_quantity("friction", 0.013)
    if initial_hw_depth > 0.0:
        domain.set_quantity("stage", lambda x, y: np.where(x < 28.0, initial_hw_depth, bed(x, y)),
                            location="centroids")
    else:
        domain.set_quantity("stage", expression="elevation", location="centroids")
    Br = anuga.Reflective_boundary(domain)
    right = Br
    if tailwater_csv:
        tt, ss = _table(tailwater_csv)
        right = anuga.Time_boundary(domain, lambda t: [float(np.interp(t, tt, ss)), 0.0, 0.0])
    domain.set_boundary({"left": Br, "right": right, "top": Br, "bottom": Br})
    region = anuga.Region(domain, polygon=[[0, 0], [2, 0], [2, 4], [0, 4]])
    if inflow_csv:
        qt, qq = _table(inflow_csv)
        q_in = lambda t: float(np.interp(t, qt, qq))  # noqa: E731
    else:
        q_in = lambda t: 1.5 if t < 60.0 else 0.0  # noqa: E731
    inflow = anuga.Inlet_operator(domain, region, Q=q_in)
    kw = {"workdir": workdir, "duration_s": final_time + 10} if engine == "swmm" else {}
    cs = attach_culverts(domain, [culvert_dict(diameter)], engine=engine, **kw)
    if mode == 2:
        domain.set_multiprocessor_mode(2)
    v0 = domain.get_water_volume()
    for _ in domain.evolve(yieldstep=yieldstep, finaltime=final_time):
        pass
    v1 = domain.get_water_volume()
    extra = cs.finalize() or {}
    s = cs.timeseries(0)
    if series_dir:
        cs.write_csv(series_dir)
    out = dict(mode=mode, engine=engine, peak_hw=cs.peak_headwater(0),
               peak_q=float(np.max(s["q_m3s"])), v0=v0, v1=v1,
               inflow=float(inflow.total_applied_volume),
               cpu_sync_bracket=bool(getattr(domain, "_cached_has_cpu_only_ops", False)),
               has_sync_bracket_api=hasattr(domain, "_cached_has_cpu_only_ops"),
               n_records=int(s["time_s"].size), **extra)
    if engine == "swmm":
        out.update(requested=float(cs.requested_volume[0]), applied=float(cs.applied_volume[0]),
                   op_out_applied=float(cs.ops_out[0].total_applied_volume),
                   op_in_applied=float(cs.ops_in[0].total_applied_volume))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", type=int)
    ap.add_argument("engine")
    ap.add_argument("final_time", type=float)
    ap.add_argument("out_json")
    ap.add_argument("--diameter", type=float, default=DEFAULT_DIAMETER)
    ap.add_argument("--inflow-csv")
    ap.add_argument("--tailwater-csv")
    ap.add_argument("--series-dir")
    ap.add_argument("--initial-hw-depth", type=float, default=0.0)
    ap.add_argument("--yieldstep", type=float, default=10.0)
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as d:
        result = run(a.mode, a.engine, a.final_time, d, diameter=a.diameter,
                     inflow_csv=a.inflow_csv, tailwater_csv=a.tailwater_csv,
                     series_dir=a.series_dir, initial_hw_depth=a.initial_hw_depth,
                     yieldstep=a.yieldstep)
    with open(a.out_json, "w") as f:
        json.dump(result, f)
    print(json.dumps(result))
