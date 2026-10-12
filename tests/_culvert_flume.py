"""A small ANUGA flume with one headwall culvert through an embankment.

Run as a script (each simulation in its own process: an ANUGA/MPI abort
kills the interpreter):

    python -I _culvert_flume.py <mode 1|2> <engine> <final_time_s> <out.json>

Bed 0 m upstream of an embankment (x 28-32 m, crest 3 m), -0.1 m downstream;
4 m wide, 60 m long, reflective walls. A 1.5 m3/s inflow at the upstream end
for the first 60 s raises the headwater; a 0.9 m RCP (square edge, headwall)
runs from (26, 2) to (34, 2) under the embankment.
"""
import json
import sys
import tempfile


def culvert_dict():
    return dict(name="flume RCP", shape="circular", diameter_m=0.9,
                coordinates=[[26.0, 2.0], [34.0, 2.0]], upstream_invert_m=0.0,
                downstream_invert_m=-0.1, entrance_type="square_edge_headwall",
                material="concrete", barrels=1)


def bed(x, y):
    import numpy as np
    z = np.where(x < 28.0, 0.0, -0.1)
    return np.where((x >= 28.0) & (x <= 32.0), 3.0, z)


def run(mode, engine, final_time, workdir):
    import anuga
    import numpy as np
    from anuga_drainage.culverts import attach_culverts

    domain = anuga.rectangular_cross_domain(120, 8, len1=60.0, len2=4.0)
    domain.set_name("flume")
    domain.set_datadir(workdir)
    domain.set_store(False)
    domain.set_quantity("elevation", bed, location="centroids")
    domain.set_quantity("friction", 0.013)
    domain.set_quantity("stage", expression="elevation", location="centroids")
    Br = anuga.Reflective_boundary(domain)
    domain.set_boundary({"left": Br, "right": Br, "top": Br, "bottom": Br})
    region = anuga.Region(domain, polygon=[[0, 0], [2, 0], [2, 4], [0, 4]])
    inflow = anuga.Inlet_operator(domain, region, Q=lambda t: 1.5 if t < 60.0 else 0.0)
    cs = attach_culverts(domain, [culvert_dict()], engine=engine,
                         **({"workdir": workdir} if engine == "swmm" else {}))
    if mode == 2:
        domain.set_multiprocessor_mode(2)
    v0 = domain.get_water_volume()
    for _ in domain.evolve(yieldstep=10.0, finaltime=final_time):
        pass
    v1 = domain.get_water_volume()
    extra = cs.finalize() or {}
    s = cs.timeseries(0)
    out = dict(mode=mode, engine=engine, peak_hw=cs.peak_headwater(0),
               peak_q=float(np.max(s["q_m3s"])), v0=v0, v1=v1,
               inflow=float(inflow.total_applied_volume),
               cpu_sync_bracket=bool(getattr(domain, "_cached_has_cpu_only_ops", False)),
               n_records=int(s["time_s"].size), **extra)
    if engine == "swmm":
        out.update(requested=float(cs.requested_volume[0]), applied=float(cs.applied_volume[0]),
                   op_out_applied=float(cs.ops_out[0].total_applied_volume),
                   op_in_applied=float(cs.ops_in[0].total_applied_volume))
    return out


if __name__ == "__main__":
    mode, engine, final_time, out_path = int(sys.argv[1]), sys.argv[2], float(sys.argv[3]), sys.argv[4]
    with tempfile.TemporaryDirectory() as d:
        result = run(mode, engine, final_time, d)
    with open(out_path, "w") as f:
        json.dump(result, f)
    print(json.dumps(result))
