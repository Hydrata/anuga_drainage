"""Entrance types, materials and their SWMM culvert codes (epic 3585, D10).

The 14 entrance types are the FHWA HDS-5 inlet-control entries of the epic's
oracle (``hds5.py`` INLETS). Each is valid for exactly one shape and the
material its chart covers, so the valid (entrance type, shape, material)
combinations are the 14 rows below, not 14 x 2 x 2. The ``value`` strings are
the product's stored codes (Hydrata gmc ``culvertForm.js`` ENTRANCE_TYPES);
'square_edge_headwall' exists for every shape x material so a default culvert
is always valid.

``swmm_code`` is the EPA SWMM 5.2 ``[XSECTIONS]`` culvert code whose
FORM/K/M/c/Y row (``culvert.c`` Params) equals the HDS-5 row; ``ke`` is the
HDS-5 entrance-loss coefficient used for outlet control.
"""
from __future__ import annotations

from dataclasses import dataclass

SHAPE_BOX = "box"
SHAPE_CIRCULAR = "circular"
SHAPES = (SHAPE_BOX, SHAPE_CIRCULAR)

CONCRETE = "concrete"
CORRUGATED_METAL = "corrugated_metal"

#: D10 default Manning n per material (the user may override n).
MATERIAL_MANNING_N = {CONCRETE: 0.013, CORRUGATED_METAL: 0.024}

#: D10: fixed exit-loss coefficient.
EXIT_LOSS = 1.0

DEFAULT_ENTRANCE_TYPE = "square_edge_headwall"
DEFAULT_MATERIAL = CONCRETE


@dataclass(frozen=True)
class EntranceType:
    value: str
    shape: str
    material: str
    hds5: str
    swmm_code: int
    ke: float
    label: str


ENTRANCE_TYPES = (
    # Circular concrete (HDS-5 Chart 1) and bevelled ring (Chart 3)
    EntranceType("square_edge_headwall", SHAPE_CIRCULAR, CONCRETE, "rcp_sq_hw", 1, 0.5,
                 "Square edge with headwall"),
    EntranceType("groove_end_headwall", SHAPE_CIRCULAR, CONCRETE, "rcp_groove_hw", 2, 0.2,
                 "Groove end with headwall"),
    EntranceType("groove_end_projecting", SHAPE_CIRCULAR, CONCRETE, "rcp_groove_proj", 3, 0.2,
                 "Groove end projecting"),
    EntranceType("bevelled_ring_45", SHAPE_CIRCULAR, CONCRETE, "circ_bevel45", 7, 0.2,
                 "Bevelled ring, 45 deg bevels"),
    # Circular corrugated metal (Chart 2)
    EntranceType("square_edge_headwall", SHAPE_CIRCULAR, CORRUGATED_METAL, "cmp_hw", 4, 0.5,
                 "Headwall"),
    EntranceType("mitred_to_slope", SHAPE_CIRCULAR, CORRUGATED_METAL, "cmp_mitered", 5, 0.7,
                 "Mitred to conform to slope"),
    EntranceType("projecting", SHAPE_CIRCULAR, CORRUGATED_METAL, "cmp_proj", 6, 0.9,
                 "Projecting"),
    # Concrete box (Charts 8, 9, 10)
    EntranceType("square_edge_headwall", SHAPE_BOX, CONCRETE, "box_sq_90hw", 10, 0.5,
                 "Square edge, 90 deg headwall"),
    EntranceType("wingwalls_30_75", SHAPE_BOX, CONCRETE, "box_wing30_75", 9, 0.4,
                 "Wingwalls flared 30 to 75 deg"),
    EntranceType("wingwalls_0", SHAPE_BOX, CONCRETE, "box_wing0", 11, 0.7,
                 "Wingwalls parallel (0 deg flare)"),
    EntranceType("wingwalls_45_top_bevel", SHAPE_BOX, CONCRETE, "box_wing45_bev", 12, 0.2,
                 "Wingwalls flared 45 deg, top edge bevelled"),
    EntranceType("headwall_bevelled_45", SHAPE_BOX, CONCRETE, "box_hw_bev45", 15, 0.2,
                 "Headwall, 45 deg bevelled edges"),
    # Corrugated-metal box (HDS-5 Table A.2)
    EntranceType("square_edge_headwall", SHAPE_BOX, CORRUGATED_METAL, "cmbox_90hw", 27, 0.5,
                 "Square edge, 90 deg headwall"),
    EntranceType("thin_wall_projecting", SHAPE_BOX, CORRUGATED_METAL, "cmbox_thinproj", 29, 0.9,
                 "Thin wall projecting"),
)


def entrance_types_for(shape: str, material: str) -> list:
    """The entrance types offered for a shape + material (the D10 filter)."""
    return [t for t in ENTRANCE_TYPES if t.shape == shape and t.material == material]


def resolve_entrance(value: str, shape: str, material: str) -> EntranceType:
    """The EntranceType for a stored (value, shape, material); ValueError if invalid."""
    for t in ENTRANCE_TYPES:
        if t.value == value and t.shape == shape and t.material == material:
            return t
    valid = ", ".join(t.value for t in entrance_types_for(shape, material)) or "none"
    raise ValueError(
        f"entrance type {value!r} is not valid for a {shape} {material} culvert "
        f"(valid: {valid})")
