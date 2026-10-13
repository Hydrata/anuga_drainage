"""Tests for the coupling driver (smooth_Q, limit_outflow, Coupler.step).

These exercise the pure flux mechanics with a fake backend and fake inlets, so
they need no ANUGA/SWMM/pipedream install. calculate_Q itself pulls anuga.g, so
Coupler.step tests that call it are skipped when ANUGA is unavailable.
"""
import numpy as np
import pytest

from anuga_drainage.coupler import smooth_Q, limit_outflow, Coupler


# --- pure helpers -----------------------------------------------------------

def test_smooth_Q_blends_old_and_new():
    Q_old = np.array([0.0, 0.0])
    Q_new = np.array([10.0, 10.0])
    # dt=1, time_average=10 -> 10% of the new value moves in.
    out = smooth_Q(Q_new, Q_old, dt=1.0, time_average=10.0)
    assert out == pytest.approx([1.0, 1.0])


def test_smooth_Q_disabled_returns_new():
    Q_new = np.array([3.0, -2.0])
    assert smooth_Q(Q_new, np.zeros(2), dt=1.0, time_average=0.0) is Q_new


def test_smooth_Q_dt_equals_time_average_is_full_step():
    Q_new = np.array([5.0])
    out = smooth_Q(Q_new, np.array([99.0]), dt=2.0, time_average=2.0)
    assert out == pytest.approx([5.0])


def test_limit_outflow_caps_positive_flux():
    Q = np.array([100.0, 0.5, -100.0])
    available = np.array([1.0, 1.0, 1.0])  # /dt=1 -> limit 1.0
    out = limit_outflow(Q, available, dt=1.0)
    # positive flux capped at available/dt; small positive and surcharge untouched.
    assert out == pytest.approx([1.0, 0.5, -100.0])


def test_limit_outflow_safety_factor():
    Q = np.array([100.0])
    out = limit_outflow(Q, np.array([2.0]), dt=2.0, safety_factor=0.5)
    # 0.5 * 2.0 / 2.0 = 0.5
    assert out == pytest.approx([0.5])


# --- Coupler.step orchestration with fakes ----------------------------------

class _FakeInlet:
    def __init__(self, depth, volume):
        self._depth = depth
        self._volume = volume
        self.Q_set = None

    class _I:
        pass

    @property
    def inlet(self):
        i = self._I()
        i.get_average_depth = lambda: self._depth
        i.get_total_water_volume = lambda: self._volume
        return i

    def set_Q(self, q):
        self.Q_set = q


class _FakeBackend:
    """Records the flux it was stepped with; feeds back -Q_in (pipedream-like)."""

    def __init__(self, heads):
        self._heads = np.asarray(heads, dtype=float)
        self.stepped_with = None

    def get_heads(self):
        return self._heads

    def step(self, Q_in, dt):
        self.stepped_with = np.array(Q_in)

    def anuga_flux(self, Q_in, dt):
        return -np.asarray(Q_in)


def test_coupler_step_feeds_realised_flux_to_inlets():
    inlets = [_FakeInlet(depth=1.0, volume=10.0), _FakeInlet(depth=1.0, volume=10.0)]
    backend = _FakeBackend(heads=[-1.0, -1.0])  # heads below bed -> weir inflow > 0
    coupler = Coupler(inlets, beds=[0.0, 0.0], weir_lengths=[2.0, 2.0],
                      manhole_areas=[1.0, 1.0], backend=backend, time_average=0.0,
                      g=9.81)

    step = coupler.step(dt=1.0)

    # Positive surface->pipe flux was passed to the backend ...
    assert np.all(step.Q_in > 0)
    assert backend.stepped_with == pytest.approx(step.Q_in)
    # ... and the realised flux (-Q_in here) was applied to each inlet.
    for op, f in zip(inlets, step.anuga_flux):
        assert op.Q_set == pytest.approx(f)
    assert np.all(step.anuga_flux < 0)


def test_coupler_smoothing_state_persists_across_steps():
    inlets = [_FakeInlet(depth=1.0, volume=10.0)]
    backend = _FakeBackend(heads=[-1.0])
    coupler = Coupler(inlets, beds=[0.0], weir_lengths=[2.0], manhole_areas=[1.0],
                      backend=backend, time_average=10.0, g=9.81)

    first = coupler.step(dt=1.0).Q_in.copy()
    second = coupler.step(dt=1.0).Q_in.copy()
    # With heads/depths constant, smoothing ramps the flux up toward the target,
    # so the second step exceeds the first.
    assert second[0] > first[0]


# --- TASK-3677 AC4 (D25): C2 vs the patched libswmm (fix4) on the steady grid -----------------
# tests/data/c2_vs_libswmm.csv is written by the deploy repo's
# docs/reports/2026-10-12-culvert-spike/scripts/c2_vs_libswmm.py (196 points). D25: the 0.5 %
# gate applies where fix4 itself is within 0.5 % of HDS-5; the other points are excluded and
# reported against HDS-5. The excluded list is asserted so it cannot grow silently.

D25_EXCLUDED = [   # (entrance, scenario, HW/D): fix4 more than 0.5 % from HDS-5
    ("cmp_hw", "inlet", 1.2), ("cmp_hw", "inlet", 1.5), ("cmp_hw", "inlet", 2.0),
    ("cmp_hw", "inlet", 2.5), ("cmp_hw", "inlet", 3.0), ("cmp_mitered", "inlet", 3.0),
]


def _c2_vs_libswmm_rows():
    import csv
    import pathlib
    path = pathlib.Path(__file__).parent / "data" / "c2_vs_libswmm.csv"
    with open(path) as f:
        return [dict(r, hw_d=float(r["hw_d"]), diff=float(r["diff_pct"]), lib=float(r["libswmm_vs_hds5_pct"]),
                     c2=float(r["c2_vs_hds5_pct"])) for r in csv.DictReader(f)]


def test_c2_vs_libswmm_d25_excluded_points_are_asserted():
    rows = _c2_vs_libswmm_rows()
    assert len(rows) == 196
    partial = [r for r in rows if r["regime_hds5"] == "outlet-partial"]
    excluded = [r for r in rows if r["regime_hds5"] != "outlet-partial" and abs(r["lib"]) > 0.5]
    assert partial == []
    assert [(r["hds5"], r["scenario"], r["hw_d"]) for r in excluded] == D25_EXCLUDED
    # reported against HDS-5: C2 is within 1 % of the independent reference at every excluded point
    assert max(abs(r["c2"]) for r in excluded) < 1.0


def test_c2_vs_libswmm_gated_points_within_half_percent():
    rows = _c2_vs_libswmm_rows()
    gated = [r for r in rows if r["regime_hds5"] != "outlet-partial" and abs(r["lib"]) <= 0.5]
    bad = [(r["hds5"], r["scenario"], r["hw_d"], r["diff"]) for r in gated if abs(r["diff"]) > 0.5]
    assert len(gated) == 190
    assert bad == [], f"{len(bad)}/{len(gated)} gated points differ from libswmm by more than 0.5 %: {bad}"
