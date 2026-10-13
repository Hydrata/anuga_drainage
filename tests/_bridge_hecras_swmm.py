"""bridge_hecras with the headwall SWMM coupler in place of Boyd_box_operator (TASK-3631 AC6).

A copy of anuga_core/validation_tests/behaviour_only/bridge_hecras/channel_floodplain1.py (same
mesh, topography, hydrograph, boundaries, DE1) whose ONLY change is the bridge: the
``Boyd_box_operator`` becomes one 10 m x 1.3 m headwall box culvert from ``attach_culverts``.

What is modelled (and what is not):
* opening 10 m wide x 1.3 m high, 20.02 m long (the 490-510 m deck plus the 0.01 m exchange gaps),
  inverts = the channel bed at the two exchange lines (-3.44995 / -3.55005 m);
* entrance ``square_edge_headwall`` (HDS-5 box_sq_90hw, Ke 0.5) + the fixed exit loss 1.0 = 1.5,
  which is the Boyd run's ``other: 1.5`` summed loss, so NO loss override is needed; Manning n 0.03
  as in the Boyd run (``manning_n`` is a culvert field, not a loss);
* exchange regions = the 10 m x 10 m channel rectangles upstream / downstream of the deck (Boyd's
  ``enquiry_gap`` was 10 m), wall thickness 0;
* the deck itself is the ANUGA bed (elevation -1.0 m across the whole 30 m width for y 490-510), so
  deck OVERTOPPING is 2D shallow-water flow over that strip, not a culvert feature.
Not carried over: Boyd's smoothing_timescale 30 s (the coupler is not smoothed; use_momentum_jet
stays on) and the Boyd operator's own discharge law.

    python -I _bridge_hecras_swmm.py <engine swmm|boyd-as-coded> <out_dir>

Writes ``<out_dir>/channel_floodplain1.sww`` and prints one JSON line (volume budget, SWMM continuity).
"""
import json
import os
import sys

import numpy

engine, out_dir = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src"))

import anuga  # noqa: E402
from anuga import Inlet_operator  # noqa: E402
from anuga_drainage.culverts import attach_culverts  # noqa: E402

floodplain_length = 1000.0
floodplain_width = 30.0
floodplain_slope = 1. / 200.
chan_initial_depth = 0.25
chan_bankfull_depth = 1.0
chan_width = 10.0
bankwidth = 0.01
man_n = 0.03
l0 = 5.000

boundary_polygon = [[0., 0.], [0., floodplain_length],
                    [floodplain_width / 2. - chan_width / 2., floodplain_length],
                    [floodplain_width / 2. + chan_width / 2., floodplain_length],
                    [floodplain_width, floodplain_length], [floodplain_width, 0.],
                    [floodplain_width / 2. + chan_width / 2., 0.],
                    [floodplain_width / 2. - chan_width / 2., 0.]]
breakLines = {
    'n1': [[floodplain_width / 2.0 - chan_width / 2., floodplain_length, -999.],
           [floodplain_width / 2.0 - chan_width / 2., 0., -999.0]],
    'n3': [[floodplain_width / 2.0 + chan_width / 2.0, floodplain_length, -999.],
           [floodplain_width / 2.0 + chan_width / 2.0, 0., -999.]],
    'n4': [[5., 490., -999.], [floodplain_width - 5., 490., -999]],
    'n5': [[5., 510., -999.], [floodplain_width - 5., 510., -999]],
}
regionPtAreas = [[0.01, 0.01, 0.5 * l0 * l0 * 4], [floodplain_width / 2., 0.01, 0.5 * l0 * l0],
                 [floodplain_width - 0.01, 0.01, 0.5 * l0 * l0 * 4],
                 [0.01, floodplain_length - 0.01, 0.5 * l0 * l0 * 4],
                 [floodplain_width / 2., floodplain_length - 0.01, 0.5 * l0 * l0],
                 [floodplain_width - 0.01, floodplain_length - 0.01, 0.5 * l0 * l0 * 4]]

pmesh = anuga.create_pmesh_from_regions(
    boundary_polygon,
    boundary_tags={'left': [0], 'top1': [1], 'chan_out': [2], 'top2': [3], 'right': [4],
                   'bottom1': [5], 'chan_in': [6], 'bottom2': [7]},
    maximum_triangle_area=1.0e+06, minimum_triangle_angle=28.0, interior_regions=[],
    breaklines=breakLines.values(), regionPtArea=regionPtAreas, verbose=False)
domain = anuga.Domain(pmesh)
domain.set_name('channel_floodplain1')
domain.set_datadir(out_dir)
domain.set_flow_algorithm('DE1')
domain.set_store_vertices_uniquely(True)


def topography(x, y):
    elev1 = -y * floodplain_slope - chan_bankfull_depth * \
        (x > (floodplain_width / 2. - chan_width / 2.)) * (x < (floodplain_width / 2. + chan_width / 2.))
    leftbnk = floodplain_width / 2. - chan_width / 2.
    rightbnk = floodplain_width / 2. + chan_width / 2.
    elev2 = elev1 + (chan_bankfull_depth - chan_bankfull_depth / bankwidth * (x - leftbnk)) * \
        (x > leftbnk) * (x < leftbnk + bankwidth)
    elev2 = elev2 + (chan_bankfull_depth + chan_bankfull_depth / bankwidth * (x - rightbnk)) * \
        (x > rightbnk - bankwidth) * (x < rightbnk)
    return elev2 * ((y < 490.) + (y > 510.)) - 1.0 * ((y >= 490.) * (y <= 510.))


def stagetopo(x, y):
    return -y * floodplain_slope - chan_bankfull_depth + chan_initial_depth


domain.set_quantity('elevation', topography, location='centroids')
domain.set_quantity('friction', man_n)
domain.set_quantity('stage', stagetopo)

Qdata = [3., 3., 3., 3., 3., 3., 4., 5., 6., 6., 6., 6., 6., 7., 8., 9., 10., 10., 10., 10., 10.,
         11., 12., 13., 14., 15., 15., 15., 15., 16., 17., 18., 19., 20., 21., 23., 25.,
         30., 35., 40., 45., 50., 55., 60., 65., 70., 70., 70., 70.] + 10 * [70.]
dtQdata = 300.


def Qin(t):
    t_hour = t / dtQdata
    indL = numpy.floor(t_hour).astype(int)
    indU = numpy.ceil(t_hour).astype(int)
    return Qdata[indL] * (1.0 * indU - t_hour) + Qdata[indU] * (t_hour - 1.0 * indL)


inflow = Inlet_operator(domain, [[floodplain_width / 2. - chan_width / 2., 10.0],
                                 [floodplain_width / 2. + chan_width / 2., 10.0]], Qin)

# BRIDGE_OVERRIDES (JSON) is a diagnostic knob for the sensitivity table in engine-proofs.md; the
# gate (test_bridge_hecras_swmm.py) runs with none of them set.
OV = json.loads(os.environ.get("BRIDGE_OVERRIDES", "{}"))
DEPTH = float(OV.get("depth", 10.0))
xl, xr = floodplain_width / 2. - chan_width / 2. + 0.01, floodplain_width / 2. + chan_width / 2. - 0.01
bed_in = -(490. - 0.01) * floodplain_slope - chan_bankfull_depth
bed_out = -(510. + 0.01) * floodplain_slope - chan_bankfull_depth
bridge = dict(name="bridge", shape="box", width_m=chan_width, height_m=1.3, barrels=1, wall_thickness_m=0.0,
              coordinates=[[15.0, 490. - 0.01], [15.0, 510. + 0.01]],
              upstream_invert_m=bed_in, downstream_invert_m=bed_out,
              entrance_type="square_edge_headwall", material="concrete", manning_n=0.03,
              inlet_region=[[xl, 490. - 0.01], [xr, 490. - 0.01], [xr, 490. - 0.01 - DEPTH], [xl, 490. - 0.01 - DEPTH]],
              outlet_region=[[xl, 510. + 0.01], [xr, 510. + 0.01], [xr, 510. + 0.01 + DEPTH], [xl, 510. + 0.01 + DEPTH]])
bridge["manning_n"] = float(OV.get("manning_n", 0.03))
bridge["entrance_type"] = OV.get("entrance_type", "square_edge_headwall")
finaltime = dtQdata * (len(Qdata) - 2)
if engine == "swmm":
    cs = attach_culverts(domain, [bridge], engine="swmm", workdir=out_dir, duration_s=finaltime + 20,
                         use_momentum_jet=bool(OV.get("jet", True)),
                         smoothing_timescale=float(OV.get("smoothing", 0.0)))
else:
    cs = attach_culverts(domain, [bridge], engine="boyd-as-coded")

Br = anuga.Reflective_boundary(domain)
Bout_tmss = anuga.shallow_water.boundaries.Transmissive_momentum_set_stage_boundary(
    domain, function=lambda t: -floodplain_length * floodplain_slope + chan_initial_depth - chan_bankfull_depth)
domain.set_boundary({'left': Br, 'right': Br, 'top1': Bout_tmss, 'top2': Bout_tmss, 'bottom1': Br,
                     'bottom2': Br, 'chan_out': Bout_tmss, 'chan_in': Br})

# The inherited initial stage (-y/200 - 0.75) sits BELOW the -1.0 m deck bed for y 490-510, so the
# raw initial volume is negative; ANUGA clips it on the first step.  Budget against the clipped one.
_st, _el = domain.quantities['stage'].centroid_values, domain.quantities['elevation'].centroid_values
V0 = float(numpy.sum(numpy.maximum(_st - _el, 0.0) * domain.areas))
for t in domain.evolve(yieldstep=10.0, finaltime=finaltime):
    pass
V1 = domain.get_water_volume()
bf, fs = domain.get_boundary_flux_integral(), domain.get_fractional_step_volume_integral()
extra = cs.finalize() or {}
series = cs.timeseries(0)
out = dict(engine=engine, V0=V0, V1=V1, dV=V1 - V0, boundary_flux=bf, fractional_step_volume=fs,
           inflow_volume=float(inflow.total_applied_volume),
           volume_error=(V1 - V0) - (bf + fs), peak_q=float(numpy.max(series["q_m3s"])),
           min_q=float(numpy.min(series["q_m3s"])), peak_hw=float(cs.peak_headwater(0)), **extra)
if engine == "swmm":
    out.update(requested=float(cs.requested_volume[0]), applied=float(cs.applied_volume[0]),
               out_in_balance=float(cs.ops_out[0].total_applied_volume + cs.ops_in[0].total_applied_volume))
cs.write_csv(out_dir)
print("RESULT " + json.dumps(out))
