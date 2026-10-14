"""TASK-3700 worker: a culvert whose inverts sit BELOW the mean bed of its exchange regions.

The TASK-3699 localhost defect, reduced (200 x 100 m closed box, Box x3 1.5 x 1.2 m, inverts
5.28 -> 4.382 m under beds 6.0275 / 4.7703 m): a dry region reads its bed as headwater, so
SWMM asks for ~4.2 m3/s from bone-dry ground. Every ANUGA step is recorded
(record_every_s 1e-6) so sum(q dt) over the series can be compared with the volume moved.

    python -I _buried_invert.py <initial depth m> <out.json> [finaltime s]

Runs in its own subprocess (an ANUGA/MPI abort kills the interpreter).
"""
import json
import os
import sys
import tempfile

import numpy as np

SPEC = dict(name="Box x3", shape="box", width_m=1.5, height_m=1.2, barrels=3,
            upstream_invert_m=5.28, downstream_invert_m=4.382,
            entrance_type="square_edge_headwall", material="concrete", manning_n=0.013,
            coordinates=[[70.0, 50.0], [120.0, 50.0]],
            inlet_region=[[70, 36.5], [43, 36.5], [43, 63.5], [70, 63.5]],
            outlet_region=[[120, 36.5], [147, 36.5], [147, 63.5], [120, 63.5]])
BED_UP, BED_DOWN, X_STEP = 6.0275, 4.7703, 95.0


def bed(x, y):
    return np.where(x < X_STEP, BED_UP, BED_DOWN)


def main(depth, out, finaltime=120.0):
    import anuga
    from anuga_drainage.culverts import attach_culverts

    work = tempfile.mkdtemp(prefix="buried_invert_")
    os.chdir(work)
    d = anuga.rectangular_cross_domain(40, 20, len1=200.0, len2=100.0)
    d.set_name("buried_invert")
    d.set_datadir(work)
    d.set_store(False)
    d.set_quantity("elevation", bed)
    d.set_quantity("friction", 0.04)
    d.set_quantity("stage", lambda x, y: bed(x, y) + depth)
    d.set_boundary({k: anuga.Reflective_boundary(d) for k in d.get_boundary_tags()})
    cs = attach_culverts(d, [SPEC], engine="swmm", workdir=os.path.join(work, "swmm"),
                         duration_s=finaltime + 60, smoothing_timescale=0.0, record_every_s=1e-6)
    v0 = float(d.get_water_volume())
    for _ in d.evolve(yieldstep=30.0, finaltime=finaltime):
        pass
    v1 = float(d.get_water_volume())
    s = cs.timeseries(0)
    t = s["time_s"]
    dt = np.diff(np.concatenate([[0.0], t]))
    q_req = s.get("q_requested_m3s", s["q_m3s"])   # absent before TASK-3700 (negative control)
    result = dict(depth=depth, v0=v0, v1=v1, n=int(t.size),
                  q_max=float(np.abs(s["q_m3s"]).max()), q_req_max=float(np.abs(q_req).max()),
                  sum_q_dt=float(np.sum(s["q_m3s"] * dt)), applied=float(cs.applied_volume[0]),
                  requested=float(cs.requested_volume[0]),
                  moved_le_requested=bool(np.all(np.abs(s["q_m3s"]) <= np.abs(q_req) + 1e-12)),
                  t_first=float(t[0]), t_last=float(t[-1]))
    cs.finalize()
    with open(out, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main(float(sys.argv[1]), sys.argv[2], *(float(a) for a in sys.argv[3:4]))
