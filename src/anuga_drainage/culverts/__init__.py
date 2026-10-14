"""Headwall culverts: live SWMM 5.2 culvert hydraulics coupled to ANUGA (epic 3585).

See docs/culverts.md for the public API.
"""
from .catalogue import (ENTRANCE_TYPES, EntranceType, MATERIAL_MANNING_N,
                        entrance_types_for, resolve_entrance)
from .spec import HeadwallCulvert, BLOCKAGE_METHOD
from .inp_writer import write_inp, InpLayout
from .swmm_engine import SwmmHeadwallEngine
from .rating import steady_flows
from . import inlet_control
from .inlet_control import InletControl, CONTROL_NAMES
from .coupler import (attach_culverts, CulvertSet, HeadwallCulvertCoupler, BoydAsCodedCulverts,
                      ENGINE_SWMM, ENGINE_BOYD, ENGINES, TIMESERIES_HEADER)

__all__ = [
    "ENTRANCE_TYPES", "EntranceType", "MATERIAL_MANNING_N", "entrance_types_for",
    "resolve_entrance", "HeadwallCulvert", "BLOCKAGE_METHOD", "write_inp", "InpLayout",
    "SwmmHeadwallEngine", "steady_flows", "attach_culverts", "CulvertSet",
    "HeadwallCulvertCoupler", "BoydAsCodedCulverts", "ENGINE_SWMM", "ENGINE_BOYD", "ENGINES",
    "TIMESERIES_HEADER", "inlet_control", "InletControl", "CONTROL_NAMES",
]
