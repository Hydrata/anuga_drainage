# Vendored UNMODIFIED (below this line) from Hydrata/deploy
# docs/reports/2026-10-07-boyd-culvert-robustness/scripts/hds5.py: the independent
# HDS-5 oracle for the headwall culvert tests (TASK-3630). Do not "fix" it to match SWMM.
"""
Independent FHWA HDS-5 (3rd ed., 2012, FHWA-HIF-12-026) culvert calculator, SI units.

Inlet control  : Appendix A, Eqs A.1 (Form 1), A.2 (Form 2), A.3 (submerged),
                 Ku = 1.811 (SI), Ks = -0.5 (mitered +0.7).  Unsubmerged valid to
                 Ku*Q/(A D^0.5) = 3.5, submerged from 4.0; linear interpolation of HW/D
                 in between (SWMM culvert.c / HDS-5 "transition line").
                 Coefficients: HDS-5 Table A.1 / A.2 (identical to EPA SWMM 5.2 culvert.c).
Outlet control : (a) 'full'  HDS-5 Eq 3.6b:  HWo = ho + H - S L,
                       H = (1 + ke + 2 g n^2 L / R^(4/3)) V^2/2g,  ho = max(TW, (dc+D)/2)
                 (b) 'gvf'   backwater (energy-equation march from the outlet, HY-8 style):
                       outlet water surface = max(TW, dc) (or pressure head TW if TW >= D),
                       pressurised wherever E >= D + V_full^2/2g, entrance loss ke V^2/2g,
                       exit loss = full outlet velocity head (outlet WS forced to TW).
                       If the subcritical profile cannot exist (steep barrel, free outlet, or
                       S1 profile drops below critical) outlet control does not govern.
Controlling HW = max(inlet, outlet).  Q(HW, TW) obtained by inverting HW(Q) (monotone).

All depths are measured from the respective invert: HW from the inlet invert, TW from the
outlet invert (outlet invert = inlet invert - S*L).
"""
import math
from scipy.optimize import brentq

G = 9.81
KU = 1.811           # SI unit conversion for Q/(A D^0.5)
X_UNSUB = 3.5        # Ku Q/(A D^0.5) upper limit of unsubmerged equations
X_SUB = 4.0          # lower limit of submerged equation

# name: (form, K, M, c, Y, ke, mitered, shape)
INLETS = {
    # Circular concrete (Chart 1)
    'rcp_sq_hw':       (1, 0.0098, 2.00, 0.0398, 0.67, 0.5, False, 'circ'),
    'rcp_groove_hw':   (1, 0.0018, 2.00, 0.0292, 0.74, 0.2, False, 'circ'),
    'rcp_groove_proj': (1, 0.0045, 2.00, 0.0317, 0.69, 0.2, False, 'circ'),
    # Circular CMP (Chart 2)
    'cmp_hw':          (1, 0.0078, 2.00, 0.0379, 0.69, 0.5, False, 'circ'),
    'cmp_mitered':     (1, 0.0210, 1.33, 0.0463, 0.75, 0.7, True,  'circ'),
    'cmp_proj':        (1, 0.0340, 1.50, 0.0553, 0.54, 0.9, False, 'circ'),
    # Circular beveled ring (Chart 3)
    'circ_bevel45':    (1, 0.0018, 2.50, 0.0300, 0.74, 0.2, False, 'circ'),
    # Concrete box (Chart 8, 9, 10)
    'box_wing30_75':   (1, 0.026, 1.00, 0.0347, 0.81, 0.4, False, 'box'),
    'box_sq_90hw':     (1, 0.061, 0.75, 0.0400, 0.80, 0.5, False, 'box'),   # 90 & 15 deg flares
    'box_wing0':       (1, 0.061, 0.75, 0.0423, 0.82, 0.7, False, 'box'),
    'box_wing45_bev':  (2, 0.510, 0.667, 0.0309, 0.80, 0.2, False, 'box'),  # 45 deg flare, top bevel
    'box_hw_bev45':    (2, 0.495, 0.667, 0.0314, 0.82, 0.2, False, 'box'),  # 90 deg headwall, 45 deg bevels
    # Corrugated-metal box (Table A.2)
    'cmbox_90hw':      (1, 0.0083, 2.00, 0.0379, 0.69, 0.5, False, 'box'),
    'cmbox_thinproj':  (1, 0.0340, 1.50, 0.0496, 0.57, 0.9, False, 'box'),
}


# ---------------------------------------------------------------- geometry
class Box:
    def __init__(self, B, D):
        self.B, self.D = float(B), float(D)
        self.Afull = self.B * self.D
        self.Pfull = 2 * (self.B + self.D)

    def A(self, y):
        return self.B * min(y, self.D)

    def T(self, y):
        return self.B

    def P(self, y):
        return self.B + 2 * min(y, self.D) if y < self.D else self.Pfull


class Circle:
    def __init__(self, D):
        self.D = float(D)
        self.Afull = math.pi * self.D ** 2 / 4
        self.Pfull = math.pi * self.D

    def _th(self, y):
        r = min(max(1 - 2 * y / self.D, -1.0), 1.0)
        return 2 * math.acos(r)

    def A(self, y):
        if y >= self.D:
            return self.Afull
        th = self._th(y)
        return self.D ** 2 / 8 * (th - math.sin(th))

    def T(self, y):
        if y >= self.D:
            return 0.0
        return self.D * math.sin(self._th(y) / 2)

    def P(self, y):
        if y >= self.D:
            return self.Pfull
        return self.D * self._th(y) / 2


def make_geom(shape, B=None, D=None):
    return Box(B, D) if shape == 'box' else Circle(D)


def critical_depth(geom, Q):
    """Critical depth (capped at D). Q^2 T / (g A^3) = 1."""
    D = geom.D
    if Q <= 0:
        return 0.0

    def f(y):
        return geom.A(y) ** 1.5 * math.sqrt(G / max(geom.T(y), 1e-12)) - Q
    ytop = D * (1 - 1e-7)
    if f(ytop) <= 0:
        return D
    return brentq(f, 1e-9 * D, ytop, xtol=1e-10 * D)


def normal_depth(geom, Q, S, n):
    """Normal depth, or None if the barrel cannot carry Q in uniform flow (or S<=0)."""
    if S <= 0:
        return None

    def cap(y):
        A = geom.A(y)
        P = geom.P(y)
        return A * (A / P) ** (2 / 3) * math.sqrt(S) / n
    D = geom.D
    # max open-channel capacity (circle peaks ~0.938 D)
    ys = [D * k / 200 for k in range(1, 200)]
    caps = [cap(y) for y in ys]
    imax = max(range(len(caps)), key=lambda i: caps[i])
    if Q > caps[imax]:
        return None
    return brentq(lambda y: cap(y) - Q, 1e-9 * D, ys[imax], xtol=1e-10 * D)


# ---------------------------------------------------------------- inlet control
def inlet_hw(Q, geom, inlet, S):
    form, K, M, c, Y, ke, mitered, _ = INLETS[inlet]
    D, A = geom.D, geom.Afull
    Ks = 0.7 if mitered else -0.5
    if Q <= 0:
        return 0.0

    def unsub(q):
        x = KU * q / (A * D ** 0.5)
        if form == 1:
            dc = critical_depth(geom, q)
            Ac = geom.A(dc)
            Hc = dc + q ** 2 / (2 * G * Ac ** 2)
            return D * (Hc / D + K * x ** M + Ks * S)
        return D * (K * x ** M)

    def sub(q):
        x = KU * q / (A * D ** 0.5)
        return D * (c * x ** 2 + Y + Ks * S)

    x = KU * Q / (A * D ** 0.5)
    if x <= X_UNSUB:
        return unsub(Q)
    if x >= X_SUB:
        return sub(Q)
    q1 = X_UNSUB * A * D ** 0.5 / KU
    q2 = X_SUB * A * D ** 0.5 / KU
    h1, h2 = unsub(q1), sub(q2)
    return h1 + (h2 - h1) * (Q - q1) / (q2 - q1)


# ---------------------------------------------------------------- outlet control
def outlet_hw_full(Q, geom, L, S, n, ke, TW):
    D, A, P = geom.D, geom.Afull, geom.Pfull
    R = A / P
    V = Q / A
    dc = critical_depth(geom, Q)
    ho = max(TW, (min(dc, D) + D) / 2)
    H = (1 + ke + 2 * G * n ** 2 * L / R ** (4 / 3)) * V ** 2 / (2 * G)
    return ho + H - S * L


def outlet_hw_gvf(Q, geom, L, S, n, ke, TW, nsteps=80):
    """Backwater outlet-control headwater (m above inlet invert); None if not governing."""
    D, Af, Pf = geom.D, geom.Afull, geom.Pfull
    if Q <= 0:
        return max(TW - S * L, 0.0)
    Vf = Q / Af
    hvf = Vf ** 2 / (2 * G)
    Sf_full = n ** 2 * Vf ** 2 / (Af / Pf) ** (4 / 3)
    dc = critical_depth(geom, Q)
    Ac = geom.A(dc)
    Ec = dc + Q ** 2 / (2 * G * Ac ** 2)

    if TW < dc and dc < D:
        yn = normal_depth(geom, Q, S, n)
        if yn is not None and yn < dc:
            return None            # steep barrel, free outlet: supercritical, inlet control

    if TW >= D:
        E = TW + hvf
    elif dc >= D:
        E = D + hvf
    else:
        y0 = max(TW, dc)
        E = y0 + Q ** 2 / (2 * G * geom.A(y0) ** 2)

    def state(E):
        """returns (Sf, V) or None if E < Ec (no subcritical solution)."""
        if E >= D + hvf:
            return Sf_full, Vf
        if E < Ec - 1e-12:
            return None
        if dc >= D:
            return Sf_full, Vf
        f = lambda y: y + Q ** 2 / (2 * G * geom.A(y) ** 2) - E
        lo, hi = dc, D
        if f(lo) >= 0:
            y = lo
        else:
            y = brentq(f, lo, hi, xtol=1e-9 * D)
        A = geom.A(y)
        P = geom.P(y)
        V = Q / A
        return n ** 2 * V ** 2 / (A / P) ** (4 / 3), V

    dx = L / nsteps
    st = state(E)
    if st is None:
        return None
    for _ in range(nsteps):
        Sf0 = st[0]
        E1 = E + (Sf0 - S) * dx
        st1 = state(E1)
        if st1 is None:
            return None
        E = E + (0.5 * (Sf0 + st1[0]) - S) * dx
        st = state(E)
        if st is None:
            return None
    V = st[1]
    global LAST_FULL_AT_INLET
    LAST_FULL_AT_INLET = E >= D + hvf
    return E + ke * V ** 2 / (2 * G)


LAST_FULL_AT_INLET = False


# ---------------------------------------------------------------- culvert
class Culvert:
    def __init__(self, shape, D, B=None, L=30.0, S=0.01, n=0.013, inlet=None,
                 ke=None, outlet_method='gvf', extra_loss=0.0):
        self.geom = make_geom(shape, B=B, D=D)
        self.shape, self.L, self.S, self.n = shape, L, S, n
        self.inlet = inlet
        self.ke = (INLETS[inlet][5] if ke is None else ke) + extra_loss
        self.outlet_method = outlet_method

    def hw(self, Q, TW, detail=False):
        hi = inlet_hw(Q, self.geom, self.inlet, self.S)
        if self.outlet_method == 'full':
            ho = outlet_hw_full(Q, self.geom, self.L, self.S, self.n, self.ke, TW)
        else:
            ho = outlet_hw_gvf(Q, self.geom, self.L, self.S, self.n, self.ke, TW)
        if ho is None:
            ho = -1e9
        h = max(hi, ho, 0.0)
        if detail:
            if hi >= ho:
                x = KU * Q / (self.geom.Afull * self.geom.D ** 0.5)
                reg = 'inlet-unsub' if x <= X_UNSUB else ('inlet-sub' if x >= X_SUB else 'inlet-trans')
            else:
                full = (TW >= self.geom.D) or (self.outlet_method == 'gvf' and LAST_FULL_AT_INLET)
                reg = 'outlet-full' if full else 'outlet-partial'
            return h, hi, ho, reg
        return h

    def q(self, HW, TW, qmax=None):
        """Discharge for headwater HW (above inlet invert) and tailwater TW (above outlet invert)."""
        if HW <= 0 or HW + self.S * self.L <= TW:
            return 0.0
        if qmax is None:
            qmax = 10.0 * self.geom.Afull * math.sqrt(2 * G * max(HW, self.geom.D))
        f = lambda Q: self.hw(Q, TW) - HW
        lo = 1e-9
        if f(lo) > 0:
            return 0.0
        hi = qmax
        while f(hi) < 0:
            hi *= 2
        return brentq(f, lo, hi, xtol=1e-7, rtol=1e-7)


# ---------------------------------------------------------------- unit helpers
FT = 0.3048
CFS = FT ** 3
