"""A headwall culvert built from the D10 per-feature fields (epic 3585).

``HeadwallCulvert.from_dict`` takes the same keys the Hydrata package writes
into ``culvert.geojson`` properties (run_anuga ``culverts.features.Culvert``):

    name, shape ('box'|'circular'), width_m + height_m (box) or diameter_m
    (circular), upstream_invert_m, downstream_invert_m, barrels (1-20),
    wall_thickness_m (default 0.20), entrance_type, material
    ('concrete'|'corrugated_metal'), manning_n (default from material),
    blockage_pct (0-100), length_override_m

plus the line geometry, either ``coordinates`` ([[x, y], ...] drawn inlet ->
outlet, absolute coordinates) or ``inlet_xy`` and ``outlet_xy``. Optional
``inlet_region`` / ``outlet_region`` polygons override the default exchange
rectangles (run_anuga generates mesh-sized D15 regions and passes them here).

Blockage is D11 "Reduced-area blockage": a box loses width, a circular barrel
becomes the equal-area circle (invert fixed). HW/D is still reported against
the UNBLOCKED rise.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from . import catalogue

MAX_BARRELS = 20
DEFAULT_WALL_THICKNESS_M = 0.20
BLOCKAGE_METHOD = "reduced-area"


def _num(d, key, default=None):
    v = d.get(key, default)
    if v is None or v == "":
        return default
    v = float(v)
    if not math.isfinite(v):
        raise ValueError(f"{key} must be finite, got {v!r}")
    return v


@dataclass
class HeadwallCulvert:
    name: str
    shape: str
    inlet_xy: tuple
    outlet_xy: tuple
    upstream_invert_m: float
    downstream_invert_m: float
    width_m: Optional[float] = None
    height_m: Optional[float] = None
    diameter_m: Optional[float] = None
    barrels: int = 1
    wall_thickness_m: float = DEFAULT_WALL_THICKNESS_M
    entrance_type: str = catalogue.DEFAULT_ENTRANCE_TYPE
    material: str = catalogue.DEFAULT_MATERIAL
    manning_n: Optional[float] = None
    blockage_pct: float = 0.0
    length_override_m: Optional[float] = None
    inlet_region: Optional[list] = None
    outlet_region: Optional[list] = None
    exchange_depth_m: Optional[float] = None
    coordinates: list = field(default_factory=list)

    # ------------------------------------------------------------ construction
    @classmethod
    def from_dict(cls, d: dict) -> "HeadwallCulvert":
        d = dict(d)
        coords = d.get("coordinates")
        if coords:
            coords = [(float(p[0]), float(p[1])) for p in coords]
            inlet_xy, outlet_xy = coords[0], coords[-1]
        else:
            inlet_xy = tuple(map(float, d["inlet_xy"]))
            outlet_xy = tuple(map(float, d["outlet_xy"]))
            coords = [inlet_xy, outlet_xy]
        material = d.get("material") or catalogue.DEFAULT_MATERIAL
        barrels = d.get("barrels")
        c = cls(
            name=str(d.get("name") or "culvert"),
            shape=(d.get("shape") or "").strip().lower(),
            inlet_xy=inlet_xy,
            outlet_xy=outlet_xy,
            upstream_invert_m=_num(d, "upstream_invert_m"),
            downstream_invert_m=_num(d, "downstream_invert_m"),
            width_m=_num(d, "width_m"),
            height_m=_num(d, "height_m"),
            diameter_m=_num(d, "diameter_m"),
            barrels=1 if barrels in (None, "") else int(barrels),
            wall_thickness_m=_num(d, "wall_thickness_m", DEFAULT_WALL_THICKNESS_M),
            entrance_type=d.get("entrance_type") or catalogue.DEFAULT_ENTRANCE_TYPE,
            material=material,
            manning_n=_num(d, "manning_n"),
            blockage_pct=_num(d, "blockage_pct", 0.0),
            length_override_m=_num(d, "length_override_m"),
            inlet_region=d.get("inlet_region"),
            outlet_region=d.get("outlet_region"),
            exchange_depth_m=_num(d, "exchange_depth_m"),
            coordinates=coords,
        )
        c.validate()
        return c

    def validate(self) -> None:
        problems = []
        if self.shape not in catalogue.SHAPES:
            problems.append(f"shape must be box or circular, got {self.shape!r}")
        elif self.shape == catalogue.SHAPE_BOX and not (
                self.width_m and self.width_m > 0 and self.height_m and self.height_m > 0):
            problems.append("a box culvert needs width_m and height_m > 0")
        elif self.shape == catalogue.SHAPE_CIRCULAR and not (self.diameter_m and self.diameter_m > 0):
            problems.append("a circular culvert needs diameter_m > 0")
        if self.upstream_invert_m is None or self.downstream_invert_m is None:
            problems.append("upstream_invert_m and downstream_invert_m are required")
        if not 1 <= self.barrels <= MAX_BARRELS:
            problems.append(f"barrels must be 1..{MAX_BARRELS}, got {self.barrels}")
        if not 0.0 <= self.blockage_pct <= 100.0:
            problems.append(f"blockage_pct must be 0..100, got {self.blockage_pct}")
        if self.material not in catalogue.MATERIAL_MANNING_N:
            problems.append(f"material must be one of {sorted(catalogue.MATERIAL_MANNING_N)}, "
                            f"got {self.material!r}")
        if self.length_m <= 0:
            problems.append("the culvert has zero length")
        if not problems and self.shape in catalogue.SHAPES:
            try:
                self.entrance
            except ValueError as e:
                problems.append(str(e))
        if problems:
            raise ValueError(f"culvert {self.name!r}: " + "; ".join(problems))

    # ------------------------------------------------------------ derived
    @property
    def entrance(self) -> catalogue.EntranceType:
        return catalogue.resolve_entrance(self.entrance_type, self.shape, self.material)

    @property
    def n(self) -> float:
        return self.manning_n if self.manning_n else catalogue.MATERIAL_MANNING_N[self.material]

    @property
    def rise_m(self) -> float:
        """Unblocked barrel rise (D): HW/D is always reported against this."""
        return float(self.diameter_m if self.shape == catalogue.SHAPE_CIRCULAR else self.height_m)

    @property
    def span_m(self) -> float:
        return float(self.diameter_m if self.shape == catalogue.SHAPE_CIRCULAR else self.width_m)

    @property
    def footprint_width_m(self) -> float:
        """barrels * span + (barrels + 1) * wall thickness (D10)."""
        return self.barrels * self.span_m + (self.barrels + 1) * self.wall_thickness_m

    @property
    def drawn_length_m(self) -> float:
        return sum(math.dist(a, b) for a, b in zip(self.coordinates, self.coordinates[1:]))

    @property
    def length_m(self) -> float:
        return float(self.length_override_m) if self.length_override_m else self.drawn_length_m

    @property
    def slope(self) -> float:
        return (self.upstream_invert_m - self.downstream_invert_m) / self.length_m

    @property
    def blockage(self) -> float:
        return self.blockage_pct / 100.0

    @property
    def closed(self) -> bool:
        """100 % blocked: no flow, and the culvert is left out of the SWMM model."""
        return self.blockage >= 1.0

    @property
    def effective_geometry(self) -> tuple:
        """(swmm_shape, geom1, geom2) of ONE barrel after reduced-area blockage."""
        keep = 1.0 - self.blockage
        if self.shape == catalogue.SHAPE_CIRCULAR:
            return "CIRCULAR", self.diameter_m * math.sqrt(keep), 0.0
        return "RECT_CLOSED", self.height_m, self.width_m * keep

    def outward_direction(self, end: str) -> tuple:
        """Unit vector pointing AWAY from the barrel at ``end`` ('inlet'|'outlet')."""
        if end == "inlet":
            tip, nb = self.coordinates[0], self.coordinates[1]
        else:
            tip, nb = self.coordinates[-1], self.coordinates[-2]
        dx, dy = tip[0] - nb[0], tip[1] - nb[1]
        norm = math.hypot(dx, dy)
        return dx / norm, dy / norm

    def exchange_polygon(self, end: str) -> list:
        """The exchange region at ``end``: the given polygon, else a rectangle on the
        approach side, footprint-wide and ``exchange_depth_m`` deep (default: the
        footprint width, at least 1 m), starting at the end point."""
        given = self.inlet_region if end == "inlet" else self.outlet_region
        if given:
            return [tuple(map(float, p)) for p in given]
        ux, uy = self.outward_direction(end)
        px, py = -uy, ux
        half = 0.5 * self.footprint_width_m
        depth = self.exchange_depth_m or max(self.footprint_width_m, 1.0)
        x0, y0 = self.inlet_xy if end == "inlet" else self.outlet_xy
        return [
            (x0 + px * half, y0 + py * half),
            (x0 + px * half + ux * depth, y0 + py * half + uy * depth),
            (x0 - px * half + ux * depth, y0 - py * half + uy * depth),
            (x0 - px * half, y0 - py * half),
        ]
