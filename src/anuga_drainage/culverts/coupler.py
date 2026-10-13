"""Fractional-step coupler: N headwall culverts, live SWMM, one ANUGA domain.

Design 1 (epic 3585 D6/D7, spike scope, host backend):

* one operator registered on the domain; ANUGA calls it every internal step;
* end stages are read on the HOST: stage = volume / area + area-weighted mean
  bed level of each exchange region (in mode 2 the operator is not on the
  GPU-safe whitelist, so ``apply_fractional_steps`` syncs the host copy before
  the operator loop and pushes it back after; no device backend is needed);
* SWMM advances in whole 1 s strides whenever the ANUGA step end passes the
  next whole second (so SWMM lags ANUGA by < 1 s), and the conduit flow of the
  last stride is HELD between strides;
* Q is exchanged through two stock ``Inlet_operator``s per culvert, which the
  coupler calls itself (they are taken off the domain's operator list): the
  losing end is drawn down first, its request clipped to the water present,
  and exactly the volume it gave up is added at the other end (mass exact);
* MPI (``distribute``) is NOT supported in the spike: the master-rank gather
  is deferred to design 2 / W3.

* inlet control is enforced HERE, not by SWMM (mechanism C2, TASK-3672): the .inp
  carries culvert code 0 and every stride's barrel flow is capped at the HDS-5 inlet
  rating of the upstream end (forward: the entrance at the inlet; reverse: the mirrored
  culvert at the outlet), see inlet_control.py; the control that governed each sample is
  recorded. SWMM's own culvert code skips inlet control once both ends are full.
* reverse flow through the barrel is handled by the engine (TASK-3673,
  swmm_engine.ReverseBarrel): while Q < 0 the imposed headwater stage is floored at the
  brink (invert + dc) and the conduit's loss coefficients follow the flow direction.

Sign convention: Q > 0 flows inlet -> outlet.
"""
from __future__ import annotations

import csv
import math
import os

import numpy as np

from .inlet_control import BARREL, CONTROL_NAMES, NOT_TRACKED, InletControl
from .spec import HeadwallCulvert
from .swmm_engine import SwmmHeadwallEngine

ENGINE_SWMM = "swmm"
ENGINE_BOYD = "boyd-as-coded"
ENGINES = (ENGINE_SWMM, ENGINE_BOYD)
TIMESERIES_HEADER = ("time_s", "q_m3s", "hw_level_m", "tw_level_m")
#: write_csv adds the governing control (barrel | inlet | reverse-inlet | n/a) as a last column.
CSV_HEADER = TIMESERIES_HEADER + ("control",)


def _as_culverts(culverts):
    return [c if isinstance(c, HeadwallCulvert) else HeadwallCulvert.from_dict(c)
            for c in culverts]


class _Series:
    def __init__(self):
        self.t, self.q, self.hw, self.tw, self.q_engine, self.control = [], [], [], [], [], []

    def as_arrays(self):
        arrays = {k: np.asarray(v, dtype=float) for k, v in
                  dict(time_s=self.t, q_m3s=self.q, hw_level_m=self.hw, tw_level_m=self.tw,
                       q_engine_m3s=self.q_engine).items()}
        arrays["control"] = np.asarray(self.control, dtype=np.int8)
        return arrays


def _region_stage(inlet, mean_bed):
    area = inlet.get_area()
    return inlet.get_total_water_volume() / area + mean_bed


#: Stage below bed by less than this (m) is float noise on a dry cell.
DRY_NOISE_M = 1e-9


def _clamp_dry_noise(op):
    """Lift dry-cell stages that sit a hair BELOW the bed back onto it.

    In mode 2 a dry exchange region can come back from the device with
    stage - elevation = -1e-17 (MEASURED on the flume at t = 0), and the stock
    Inlet_operator asserts its region volume >= 0. The write is on the host
    inside apply_fractional_steps' sync bracket, so it reaches the device.
    A real negative depth (more than DRY_NOISE_M) is a bug and raises.
    """
    inlet = op.inlet
    depths = inlet.get_depths()
    low = depths < 0.0
    if not low.any():
        return
    if depths.min() < -DRY_NOISE_M:
        raise RuntimeError(f"{op.label}: negative depth {depths.min():.3g} m in an exchange region")
    idx = inlet.triangle_indices[low]
    stage = op.domain.quantities["stage"].centroid_values
    stage[idx] = op.domain.quantities["elevation"].centroid_values[idx]


def _base_operator():
    import anuga
    return anuga.Operator


class CulvertSet:
    """Common read API for both engines (see docs/culverts.md)."""

    engine = None

    def __init__(self, culverts):
        self.culverts = culverts
        self.series = [_Series() for _ in culverts]

    def names(self):
        return [c.name for c in self.culverts]

    def timeseries(self, index):
        """dict of numpy arrays: time_s, q_m3s, hw_level_m, tw_level_m, q_engine_m3s and
        control (int8 index into CONTROL_NAMES: barrel, inlet, reverse-inlet, n/a)."""
        return self.series[index].as_arrays()

    def peak_headwater(self, index):
        hw = self.timeseries(index)["hw_level_m"]
        return float(np.nanmax(hw)) if hw.size else float("nan")

    def write_csv(self, directory):
        """One ``<name>.csv`` per culvert: run_anuga's TIMESERIES_HEADER columns plus
        ``control`` (which control governed the sample: barrel | inlet | reverse-inlet)."""
        os.makedirs(directory, exist_ok=True)
        paths = []
        for i, c in enumerate(self.culverts):
            s = self.timeseries(i)
            path = os.path.join(directory, f"{i:03d}_{_safe(c.name)}.csv")
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(CSV_HEADER)
                for row in zip(*(s[k] for k in TIMESERIES_HEADER), s["control"]):
                    w.writerow([f"{v:.6g}" for v in row[:-1]] + [CONTROL_NAMES[int(row[-1])]])
            paths.append(path)
        return paths

    def finalize(self):
        return None


def _next_sample(rel_t, every):
    """The next sampling time strictly after rel_t (no backlog after a long step)."""
    return (math.floor(rel_t / every + 1e-9) + 1) * every


def _safe(name):
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)[:60] or "culvert"


class HeadwallCulvertCoupler(CulvertSet):
    """The SWMM engine. Build it with :func:`attach_culverts`."""

    engine = ENGINE_SWMM

    def __init__(self, domain, culverts, workdir, *, duration_s=None, routing_step=0.25,
                 smoothing_timescale=0.0, use_momentum_jet=False, record_every_s=1.0,
                 extra_options=None, label="headwall_culverts"):
        import anuga

        if getattr(domain, "numproc", 1) > 1:
            raise NotImplementedError(
                "headwall culverts under distribute() are not supported in the spike "
                "(master-rank SWMM + collective reads are W3 / design 2)")
        CulvertSet.__init__(self, _as_culverts(culverts))
        self.domain = domain
        self.smoothing_timescale = float(smoothing_timescale)
        self.use_momentum_jet = bool(use_momentum_jet)
        self.record_every_s = float(record_every_s)
        if duration_s is None:
            duration_s = 7 * 86400
        n = len(self.culverts)
        self.ops_in, self.ops_out, self.bed_in, self.bed_out = [], [], np.zeros(n), np.zeros(n)
        for i, c in enumerate(self.culverts):
            for end, ops, bed in (("inlet", self.ops_in, self.bed_in),
                                  ("outlet", self.ops_out, self.bed_out)):
                region = anuga.Region(domain, polygon=c.exchange_polygon(end))
                if len(region.indices) == 0:
                    raise ValueError(f"culvert {c.name!r}: the {end} exchange region holds "
                                     f"no triangle centroid")
                op = anuga.Inlet_operator(domain, region, Q=0.0,
                                          label=f"{label}_{i}_{end}")
                domain.fractional_step_operators.remove(op)
                ops.append(op)
                inl = op.inlet
                bed[i] = float(np.sum(inl.get_elevations() * inl.get_areas()) / inl.get_area())
        hw0, tw0 = self.stages()
        self.engine_swmm = SwmmHeadwallEngine(self.culverts, workdir, duration_s=duration_s,
                                              routing_step=routing_step,
                                              extra_options=extra_options,
                                              initial_stages=list(zip(hw0, tw0)))
        self.inlet_control = InletControl(self.culverts)
        self.q_raw = np.zeros(n)           # SWMM's own barrel flow (mean of the last strides)
        self.q_held = np.zeros(n)          # barrel flow capped at the HDS-5 inlet rating
        self.control = [BARREL] * n        # what governed the last stride
        self.q_used = np.zeros(n)          # smoothed flow requested of ANUGA
        self.requested_volume = np.zeros(n)  # sum |Q| dt requested
        self.applied_volume = np.zeros(n)    # sum of volume actually moved (signed, + in->out)
        self.t0 = None
        self._next_record = 0.0
        self.operator = _CouplerOperator(domain, self, label)

    # ---------------------------------------------------------------- per step
    def stages(self):
        hw = np.array([_region_stage(op.inlet, b) for op, b in zip(self.ops_in, self.bed_in)])
        tw = np.array([_region_stage(op.inlet, b) for op, b in zip(self.ops_out, self.bed_out)])
        return hw, tw

    def step(self):
        d = self.domain
        dt = float(d.get_timestep())
        t = float(d.get_time())
        if self.t0 is None:
            self.t0 = t
        rel_end = t - self.t0 + dt
        hw, tw = self.stages()
        strides = 0
        n = len(self.culverts)
        q_acc, raw_acc = [0.0] * n, np.zeros(n)
        if self.engine_swmm.elapsed_s + 1 <= rel_end + 1e-9:
            cap_f, cap_r = self.inlet_control.caps(hw.tolist(), tw.tolist())
            while self.engine_swmm.elapsed_s + 1 <= rel_end + 1e-9:
                self.engine_swmm.set_stages(hw, tw)
                self.engine_swmm.stride()
                self._accumulate(self.engine_swmm.flows(), cap_f, cap_r, q_acc, raw_acc)
                strides += 1
        if strides:
            self.q_held = np.array(q_acc) / strides
            self.q_raw = raw_acc / strides
        if dt <= 0:
            return
        if self.smoothing_timescale > 0:
            a = min(1.0, dt / self.smoothing_timescale)
            self.q_used = self.q_used + a * (self.q_held - self.q_used)
        else:
            self.q_used = self.q_held.copy()
        for i, c in enumerate(self.culverts):
            q = 0.0 if c.closed else float(self.q_used[i])
            self.requested_volume[i] += abs(q) * dt
            moved = self._exchange(i, q, dt)
            self.applied_volume[i] += moved
        if t - self.t0 + 1e-9 >= self._next_record:
            self._record(t - self.t0, hw, tw)
            self._next_record = _next_sample(t - self.t0, self.record_every_s)

    def _accumulate(self, raw, cap_f, cap_r, q_acc, raw_acc):
        """Add one stride's flows: SWMM's own (``raw_acc``) and the inlet-controlled ones
        (``q_acc``, python floats); remembers which control governed (``self.control``).
        This is the whole per-stride cost of inlet control (measured in
        tests/test_inlet_control_unsteady.py)."""
        raw_acc += raw
        q, self.control = self.inlet_control.apply(raw.tolist(), cap_f, cap_r)
        for i, v in enumerate(q):
            q_acc[i] += v

    def _exchange(self, i, q, dt):
        """Move |q| dt from the losing end to the other; returns the signed volume moved."""
        if q == 0.0:
            return 0.0
        src, dst = (self.ops_in[i], self.ops_out[i]) if q > 0 else (self.ops_out[i], self.ops_in[i])
        _clamp_dry_noise(src)
        _clamp_dry_noise(dst)
        available = src.inlet.get_total_water_volume()
        want = min(abs(q) * dt, available)
        if want <= 0.0:
            src.applied_Q = dst.applied_Q = 0.0
            return 0.0
        before = src.total_applied_volume
        src.set_Q(-want / dt)
        src()
        taken = -(src.total_applied_volume - before)      # >= 0, what src really gave
        if self.use_momentum_jet:
            dst.velocity = self._jet_velocity(i, taken / dt, toward_outlet=q > 0)
        dst.set_Q(taken / dt)
        dst()
        return taken if q > 0 else -taken

    def _jet_velocity(self, i, q, toward_outlet):
        c = self.culverts[i]
        shape, g1, g2 = c.effective_geometry
        area = (math.pi * g1 * g1 / 4.0 if shape == "CIRCULAR" else g1 * g2) * c.barrels
        v = min(q / area, 10.0)
        ux, uy = c.outward_direction("outlet" if toward_outlet else "inlet")
        return [v * ux, v * uy]

    def _record(self, t, hw, tw):
        for i, s in enumerate(self.series):
            s.t.append(t)
            s.q.append(float(self.q_used[i]))
            s.hw.append(float(hw[i]))
            s.tw.append(float(tw[i]))
            s.q_engine.append(float(self.q_raw[i]))
            s.control.append(self.control[i])

    def finalize(self):
        """Close SWMM; returns {'swmm_routing_error_pct': float}."""
        return {"swmm_routing_error_pct": self.engine_swmm.close()}


def _CouplerOperator(domain, coupler, label):
    Operator = _base_operator()

    class HeadwallCulvertOperator(Operator):
        def __init__(self):
            Operator.__init__(self, domain, description="headwall culverts (SWMM 5.2)",
                              label=label)
            self.coupler = coupler

        def __call__(self):
            self.coupler.step()

        def parallel_safe(self):
            return False

        def statistics(self):
            return f"Headwall culvert coupler: {len(self.coupler.culverts)} culvert(s)"

        def timestepping_statistics(self):
            q = ", ".join(f"{v:.3f}" for v in self.coupler.q_used[:5])
            return f"Headwall culverts Q[:5] = [{q}] m3/s"

    return HeadwallCulvertOperator()


class BoydAsCodedCulverts(CulvertSet):
    """The negative-control engine: anuga's Boyd box/pipe operators as coded."""

    engine = ENGINE_BOYD

    def __init__(self, domain, culverts, *, record_every_s=1.0, record=True,
                 smoothing_timescale=0.0, label="boyd_culverts"):
        import anuga

        CulvertSet.__init__(self, _as_culverts(culverts))
        self.domain = domain
        self.ops = []
        for i, c in enumerate(self.culverts):
            common = dict(losses={"inlet": c.entrance.ke, "outlet": 1.0},
                          barrels=c.barrels, blockage=c.blockage,
                          end_points=[list(c.inlet_xy), list(c.outlet_xy)],
                          invert_elevations=[c.upstream_invert_m, c.downstream_invert_m],
                          manning=c.n, smoothing_timescale=smoothing_timescale,
                          label=f"{label}_{i}")
            if c.shape == "circular":
                op = anuga.Boyd_pipe_operator(domain, diameter=c.diameter_m, **common)
            else:
                op = anuga.Boyd_box_operator(domain, width=c.width_m, height=c.height_m, **common)
            self.ops.append(op)
        self.record_every_s = float(record_every_s)
        self._next_record = 0.0
        self.t0 = None
        self.recorder = _BoydRecorder(domain, self, label) if record else None

    def record(self):
        t = float(self.domain.get_time())
        if self.t0 is None:
            self.t0 = t
        if t - self.t0 + 1e-9 < self._next_record:
            return
        self._next_record = _next_sample(t - self.t0, self.record_every_s)
        for op, s in zip(self.ops, self.series):
            hw, tw = op.get_enquiry_stages()
            s.t.append(t - self.t0)
            s.q.append(float(op.discharge))
            s.hw.append(float(hw))
            s.tw.append(float(tw))
            s.q_engine.append(float(op.discharge))
            s.control.append(NOT_TRACKED)


def _BoydRecorder(domain, culvert_set, label):
    Operator = _base_operator()

    class BoydRecorder(Operator):
        def __init__(self):
            Operator.__init__(self, domain, description="Boyd culvert recorder",
                              label=f"{label}_recorder")

        def __call__(self):
            culvert_set.record()

        def parallel_safe(self):
            return False

        def statistics(self):
            return "Boyd culvert recorder"

        def timestepping_statistics(self):
            return "Boyd culvert recorder"

    return BoydRecorder()


def attach_culverts(domain, culverts, engine=ENGINE_SWMM, *, workdir=None, **kwargs):
    """Attach N culverts to ``domain``; returns a :class:`CulvertSet`.

    ``culverts``: HeadwallCulvert objects or D10-field dicts (see spec.py).
    ``engine``: "swmm" (HeadwallCulvertCoupler, live SWMM 5.2) or
    "boyd-as-coded" (anuga Boyd operators, the negative control).
    ``workdir`` (swmm only): where the .inp/.rpt/.out go (default:
    ``<domain datadir>/culverts_swmm``).
    """
    if engine == ENGINE_SWMM:
        if workdir is None:
            workdir = os.path.join(domain.get_datadir() or ".", "culverts_swmm")
        return HeadwallCulvertCoupler(domain, culverts, workdir, **kwargs)
    if engine == ENGINE_BOYD:
        return BoydAsCodedCulverts(domain, culverts, **kwargs)
    raise ValueError(f"engine must be one of {ENGINES}, got {engine!r}")
