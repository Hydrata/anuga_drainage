# Headwall culverts (live SWMM 5.2 culvert hydraulics)

`anuga_drainage.culverts` couples N headwall-to-headwall culverts to an ANUGA
domain, with the barrel hydraulics computed by EPA SWMM 5.2 (pyswmm) and inlet
control enforced by the coupler from the FHWA HDS-5 inlet rating (SWMM's own
culvert code is not used, see "Inlet control" below). It is the Hydrata fork's addition (epic 3585, spike
**design 1, host backend**); the pit/manhole `Coupler` elsewhere in this
package is unrelated.

Status: SPIKE. Single process only (no `distribute()`); see "Limits" below.

## Attach culverts to a domain

```python
import anuga
from anuga_drainage.culverts import attach_culverts

culverts = [
    dict(name="Main St box", shape="box", width_m=1.8, height_m=1.2, barrels=3,
         coordinates=[[x_in, y_in], [x_out, y_out]],          # inlet -> outlet, absolute
         upstream_invert_m=12.40, downstream_invert_m=12.10,
         entrance_type="wingwalls_30_75", material="concrete",
         blockage_pct=10),
    dict(name="Creek RCP", shape="circular", diameter_m=0.9,
         inlet_xy=(x1, y1), outlet_xy=(x2, y2),               # alternative to coordinates
         upstream_invert_m=8.0, downstream_invert_m=7.9),
]

cs = attach_culverts(domain, culverts, engine="swmm", workdir="out/culverts_swmm",
                     duration_s=6 * 3600)          # SWMM END_TIME; must cover the run
for t in domain.evolve(yieldstep=60, finaltime=6 * 3600):
    pass
info = cs.finalize()        # closes SWMM; {'swmm_routing_error_pct': ...}
```

Call `attach_culverts` after the domain's elevation and initial stage are set
and before `evolve` (the initial 2D stages seed the SWMM outfalls). With
`engine="swmm"` the coupler registers ONE fractional-step operator; ANUGA calls
it every internal step.

### The culvert dicts (D10 fields)

Same keys as the Hydrata package's `culvert.geojson` properties (run_anuga
`culverts.features.Culvert`):

| key | meaning |
|---|---|
| `name` | label |
| `shape` | `box` or `circular` |
| `width_m`, `height_m` / `diameter_m` | one barrel's internal span and rise (box) or diameter |
| `upstream_invert_m`, `downstream_invert_m` | barrel inverts (m, same datum as the DEM) |
| `barrels` | 1-20, identical and parallel (default 1) |
| `wall_thickness_m` | default 0.20; sets the footprint width = barrels x span + (barrels + 1) x wall |
| `entrance_type` | one of the 14 HDS-5 entries valid for the shape + material (see below); default `square_edge_headwall` |
| `material` | `concrete` (n 0.013) or `corrugated_metal` (n 0.024); default concrete |
| `manning_n` | overrides the material's n |
| `blockage_pct` | 0-100, reduced-area blockage (box loses width; circle becomes the equal-area circle, invert fixed); 100 = closed (Q = 0) |
| `length_override_m` | barrel length; default = drawn length of `coordinates` |
| `coordinates` or `inlet_xy` + `outlet_xy` | absolute coordinates, inlet first |
| `inlet_region`, `outlet_region` | optional exchange polygons (run_anuga's mesh-sized D15 regions); default = a footprint-wide rectangle on the approach side of each end, `exchange_depth_m` deep (default footprint width, >= 1 m) |

`HeadwallCulvert.from_dict(d)` validates a dict (plain-English ValueError) and
is what `attach_culverts` uses. `entrance_types_for(shape, material)` lists the
valid entrance types; `resolve_entrance(value, shape, material)` gives the
SWMM culvert code and Ke.

### Engine switch

| `engine=` | what runs | extra kwargs |
|---|---|---|
| `"swmm"` (default) | `HeadwallCulvertCoupler`: live SWMM 5.2, headwall pattern below | `workdir`, `duration_s` (default 7 days), `routing_step` (0.25 s), `smoothing_timescale` (0 = none), `use_momentum_jet` (False), `record_every_s` (1.0), `extra_options` ({SWMM [OPTIONS] key: value}) |
| `"boyd-as-coded"` | `BoydAsCodedCulverts`: ANUGA's own `Boyd_box_operator` / `Boyd_pipe_operator`, unmodified (the negative control), losses {inlet: Ke, outlet: 1.0}, blockage passed as Boyd's `blockage` | `record` (True), `record_every_s`, `smoothing_timescale` |

### Reading results

Both engines return a `CulvertSet`:

- `cs.timeseries(i)` -> dict of numpy arrays `time_s`, `q_m3s` (the flow ANUGA
  actually MOVED: the signed volume the losing end gave up in the step / the step,
  + = inlet -> outlet), `hw_level_m`, `tw_level_m` (the end stages that drove the step),
  `q_requested_m3s` (diagnostic: the flow asked of ANUGA after the inlet cap and smoothing;
  larger than `q_m3s` only when the losing end held less water than asked),
  `q_engine_m3s` (SWMM's own barrel flow before the inlet-control cap; both = `q_m3s` for Boyd).
  Sampled every `record_every_s` of model time (relative to the attach time); each
  SWMM-engine sample is stamped at the END of the ANUGA step it covers (TASK-3700).
- `cs.peak_headwater(i)`; `cs.names()`.
- `cs.write_csv(directory)` -> one CSV per culvert with run_anuga's
  `TIMESERIES_HEADER` (`time_s, q_m3s, hw_level_m, tw_level_m`) plus `control` and
  `q_requested_m3s`.
- `cs.timeseries(i)["control"]` (int8, index into `CONTROL_NAMES` = barrel,
  inlet, reverse-inlet, n/a) says which control governed each sample; `write_csv`
  adds it as a last `control` column (`n/a` for Boyd).
- SWMM engine only: `cs.requested_volume[i]` (sum of |Q| dt asked of ANUGA),
  `cs.applied_volume[i]` (signed volume actually moved), `cs.finalize()`
  returns SWMM's flow-routing continuity error read from the .rpt.

For Boyd the HW/TW are the enquiry-point stages (`get_enquiry_stages()`),
for SWMM they are the exchange-region stages below.

## How the SWMM engine works (design 1)

Per culvert `i`, the .inp (`write_inp`, one model for all culverts) holds

```
O_HW<i> (OUTFALL, FIXED stage := 2D headwater)
   ^ D_HW<i>  1 m dummy conduit, large section, n 0.010, no losses
J_HW<i> (JUNCTION at the upstream invert)
   | CUL<i>   the culvert: shape, one barrel after blockage, Barrels, culvert code 0,
   v          [LOSSES] Kentry = Ke of the entrance type, Kexit = 1.0
O_TW<i> (OUTFALL, FIXED stage := 2D tailwater)
```

An outfall stage only drives flow when the outfall is the downstream node of
its link, hence the dummy runs J_HW -> O_HW. Each ANUGA step the coupler:

1. reads each end's stage on the host as region volume / area + area-weighted
   mean bed (in mode 2 the coupler is not on the GPU-safe whitelist, so
   `apply_fractional_steps` syncs the host copy before the operator loop and
   pushes it back after);
2. strides SWMM by whole seconds (pyswmm cannot do less) while the ANUGA step
   end passes the next whole second, imposing the stages before each stride;
   every stride's barrel flow is capped at the HDS-5 inlet rating (below) and
   the result is HELD between strides (SWMM lags ANUGA by < 1 s);
3. moves the water with two stock `Inlet_operator`s it calls itself (they are
   removed from the domain's operator list): the losing end first, its request
   clipped to the water present, then exactly the volume it gave up at the
   other end. Mass is exact on the ANUGA side.

## Inlet control (coupler-enforced, TASK-3672)

SWMM 5.2.4's culvert code (`culvert.c`) is not trusted inside dynamic wave. Three
C-level defects are proven in the deploy spike folder
(`docs/reports/2026-10-12-culvert-spike/results/c-proof.csv`, traces and patches
beside it):

1. `dwflow.c:251` skips `culvert_getInflow` once BOTH barrel ends are full (a free
   outfall's depth saturates at the rise), and the code runs only for `q > 0`, so
   there is no inlet control on reverse flow at all (documented as intended in the
   SWMM Reference Manual vol. II);
2. `getForm1Flow` returns `culvert->qc` from the LAST `form1Eqn` evaluation, but
   `findroot_Ridder` breaks before evaluating at the new root (the last evaluation is
   a bracket midpoint): flow errors of -11 % and +36 % on the two traced box cases;
3. mitered inlets (codes 5/37/46) get `scf = -7.0 * slope` where HDS-5's Ks = +0.7
   means `-0.7 * slope` (-25 % flow on the traced case).

(The earlier suspicion that `findSurfArea`'s UP_CRITICAL reset corrupts the head is
REFUTED: it needs a non-zero upstream offset and the writer emits 0.)

So the .inp carries culvert code 0 (the Ke / exit losses stay) and the coupler
applies `inlet_control.py` to every stride:

```
q_used = sign(q_swmm) * min(|q_swmm|, barrels * Q_inlet(head of the upstream side))
```

* forward flow: head = headwater stage - upstream invert, rating of the entrance type;
* reverse flow: head = tailwater stage - downstream invert, rating of the MIRRORED
  culvert (slope -S, same entrance type at the outlet end);
* `Q_inlet` is the FHWA HDS-5 inlet equations (Form 1/2, submerged, linear HW/D
  interpolation between Ku Q/(A D^0.5) = 3.5 and 4.0), tabulated ONCE per culvert on a
  uniform head grid (12 rises, 200 points per rise) and interpolated linearly in pure
  Python; per-stride cost of the whole mechanism is < 1 us (measured, see
  `tests/test_inlet_control_unsteady.py`);
* blockage uses the reduced-area geometry; closed culverts carry 0.

Cost: SWMM's barrel, which sees no inlet control, can carry up to ~2.3x the applied
Q internally, so SWMM's own barrel diagnostics (velocity, depth, Froude, continuity) are
not those of the applied flow. Pipe NETWORKS will need the patched-library route
(scripts/swmm-patches/ in the spike folder) instead.

## Limits (spike)

- No MPI: `distribute()`d domains raise `NotImplementedError` (master-rank
  SWMM + collective reads are design 2 / W3).
- No device backend: every step costs a full host <-> device sync in mode 2.
- Accuracy against the independent HDS-5 calculator (see the spike folder's
  `engine-proofs.md`, TASK-3672 section, for the measured numbers): steady
  inlet control 125/126, outlet control 70/70 within 2 %; the unsteady band holds
  in the inlet-control and full-barrel regimes, and does NOT hold (a) within ~1 min of a
  flow reversal (the quasi-steady oracle has no water-column inertia: a rigid-column
  reference crosses zero with the coupler to the second), (b) in rough, mildly sloped
  partly-full barrels where HDS-5's outlet-control march and SWMM's dynamic wave
  differ, (c) in the steady reverse plateau of long barrels.
- SWMM's own continuity report includes a forcing artefact: each imposed
  outfall stage changes the barrel's stored volume without a matching flow.
- `pyswmm` must be installed (`pip install -e .[swmm]`).
