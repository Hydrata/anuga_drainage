"""Withdraw water from an exchange region by lowering a FLAT water surface (TASK-3771).

The stock ``anuga.Inlet_operator`` adds water with ``set_stages_evenly`` (the lowest stages
are raised to one common level) but removes it with ``set_depths(uniform)``: every cell of
the region is given the SAME DEPTH. On a region that spans a creek bed and the bank beside it
(Towradgi Northern Distributor bridge: bed ~4 m, bank 10.6-11.4 m) that lifts the water on
the bank cells to bank + depth (12-13.9 m against ~8.8 m in the channel), which then pours
over the road crest as a spurious 2D flow path with 11-12 m/s jets.

:func:`drawdown_stages` is the mirror of ``set_stages_evenly``: it lowers the HIGHEST water
surfaces to one common level L, never lifts a cell, and clamps each cell at its bed:

    new_stage_i = max(z_i, min(stage_i, L)),   sum a_i (stage_i - new_stage_i) = volume

L is found exactly (no iteration) on the sorted breakpoints: the volume removed above L is
piecewise linear in L with kinks at each wet cell's stage and bed.

:func:`withdraw` applies it to an ``Inlet_operator``'s region with the stock operator's
bookkeeping (``total_requested_volume``, ``total_applied_volume``, ``applied_Q``,
``domain.fractional_step_volume_integral``), so ``volume_balance`` and the coupler's
``taken`` accounting read it exactly as before. It writes the HOST centroid arrays through
``inlet.set_stages``/``set_xmoms``/``set_ymoms`` like the stock host path; in
multiprocessor mode 2 the headwall coupler runs inside ``apply_fractional_steps``'
sync_from_device()/sync_to_device() bracket (it is a CPU-only fractional operator), so the
write reaches the device. Do NOT call it from an operator the domain treats as GPU-safe.
"""
from __future__ import annotations

import numpy as np


def drawdown_stages(stages, elevations, areas, volume):
    """Remove ``volume`` (m3, >= 0) by lowering the highest water surfaces to a common level.

    Returns ``(new_stages, removed)``: ``removed`` is the volume the new stages actually take
    off (``sum a (stage - new_stage)``), equal to ``volume`` to float round-off, or every
    drop of water in the region when it holds less than ``volume`` (drained to the bed).
    Cells at or below their bed are left untouched. ``volume <= 0`` is an exact no-op.
    """
    stages = np.asarray(stages, dtype=float)
    z = np.asarray(elevations, dtype=float)
    a = np.asarray(areas, dtype=float)
    new = stages.copy()
    volume = float(volume)
    wet = stages > z
    if volume <= 0.0 or not wet.any():
        return new, 0.0
    s_w, z_w, a_w = stages[wet], z[wet], a[wet]
    water = float(np.dot(a_w, s_w - z_w))
    if volume >= water:
        new[wet] = z_w
        return new, float(np.dot(a_w, s_w - z_w))

    # Sweep the level down from the highest stage. A wet cell joins the draw-down when the
    # level passes below its stage and leaves it (dry) when the level reaches its bed.
    levels = np.concatenate([s_w, z_w])
    delta = np.concatenate([a_w, -a_w])
    order = np.argsort(-levels, kind="stable")
    lv, active = levels[order], np.cumsum(delta[order])
    removed_at = np.zeros_like(lv)                      # volume removed down to lv[k]
    removed_at[1:] = np.cumsum(active[:-1] * (lv[:-1] - lv[1:]))
    k = int(np.searchsorted(removed_at, volume, side="right")) - 1
    k = min(max(k, 0), len(lv) - 1)
    tiny = 1e-12 * float(a_w.sum())
    while k > 0 and active[k] <= tiny:                   # a kink where the region runs dry
        k -= 1
    level = lv[k] - (volume - removed_at[k]) / active[k]
    new[wet] = np.maximum(z_w, np.minimum(s_w, level))
    return new, float(np.dot(a_w, s_w - new[wet]))


def withdraw(op, volume, dt):
    """Take ``volume`` m3 (>= 0) out of ``op``'s region (an ``anuga.Inlet_operator``) by
    draw-down, with the stock operator's bookkeeping; returns the volume really removed.

    Momentum: each cell keeps its velocity (xmom, ymom scaled by new depth / old depth, zero
    where the cell dries), or follows the operator's ``zero_velocity`` / ``velocity``
    settings as the stock operator does.
    """
    inlet = op.inlet
    domain = op.domain
    volume = max(float(volume), 0.0)
    stages = inlet.get_stages()
    elevations = inlet.get_elevations()
    old_depth = np.maximum(stages - elevations, 0.0)
    new_stages, removed = drawdown_stages(stages, elevations, inlet.get_areas(), volume)
    op.total_requested_volume -= volume
    if removed > 0.0:
        inlet.set_stages(new_stages)
        new_depth = np.maximum(new_stages - elevations, 0.0)
        if op.zero_velocity:
            inlet.set_xmoms(0.0)
            inlet.set_ymoms(0.0)
        elif op.velocity is not None:
            inlet.set_xmoms(new_depth * op.velocity[0])
            inlet.set_ymoms(new_depth * op.velocity[1])
        else:
            ratio = np.divide(new_depth, old_depth, out=np.ones_like(new_depth),
                              where=old_depth > 0.0)          # a dry cell is left as it was
            inlet.set_xmoms(inlet.get_xmoms() * ratio)
            inlet.set_ymoms(inlet.get_ymoms() * ratio)
    op.total_applied_volume -= removed
    domain.fractional_step_volume_integral -= removed
    op.applied_Q = -removed / dt if dt > 0 else 0.0
    return removed
