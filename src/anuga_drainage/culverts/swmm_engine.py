"""SWMM side of the headwall culvert coupler: one pyswmm Simulation for N culverts.

pyswmm strides whole seconds only (``step_advance(0.5)`` raises TypeError), so
the engine advances in 1 s strides; the coupler holds Q between strides.
"""
from __future__ import annotations

import os
import re

import numpy as np

from .inp_writer import write_inp


class SwmmHeadwallEngine:
    def __init__(self, culverts, workdir, *, duration_s=7 * 86400, routing_step=0.25,
                 inp_name="culverts.inp", extra_options=None, initial_stages=None):
        from pyswmm import Links, Nodes, Simulation

        os.makedirs(workdir, exist_ok=True)
        self.culverts = list(culverts)
        self.inp_path = os.path.join(workdir, inp_name)
        self.layout = write_inp(self.culverts, self.inp_path, duration_s=duration_s,
                                routing_step=routing_step, extra_options=extra_options,
                                initial_stages=initial_stages)
        self.sim = Simulation(self.inp_path)
        self.sim.step_advance(1)
        self.sim.start()
        nodes, links = Nodes(self.sim), Links(self.sim)
        self._o_hw, self._o_tw, self._j_hw, self._link = {}, {}, {}, {}
        for i, names in self.layout.names.items():
            self._o_hw[i] = nodes[names.o_hw]
            self._o_tw[i] = nodes[names.o_tw]
            self._j_hw[i] = nodes[names.j_hw]
            self._link[i] = links[names.conduit]
        self.elapsed_s = 0
        self.closed = False
        self.routing_error_pct = None

    def set_stages(self, hw, tw) -> None:
        """Impose 2D headwater/tailwater STAGES (m, absolute) on every open culvert."""
        for i in self._link:
            self._o_hw[i].outfall_stage(float(hw[i]))
            self._o_tw[i].outfall_stage(float(tw[i]))

    def stride(self) -> None:
        """Advance SWMM by exactly one second."""
        next(self.sim)
        self.elapsed_s += 1

    def flows(self) -> np.ndarray:
        """Conduit flow (m3/s, all barrels; + = inlet -> outlet); 0 for closed culverts."""
        q = np.zeros(len(self.culverts))
        for i, link in self._link.items():
            q[i] = link.flow
        return q

    def junction_heads(self) -> np.ndarray:
        h = np.full(len(self.culverts), np.nan)
        for i, node in self._j_hw.items():
            h[i] = node.head
        return h

    def close(self):
        """Finish the run; returns SWMM's flow-routing continuity error (%), read
        from the .rpt ("Flow Routing Continuity" block), or None if absent."""
        if not self.closed:
            self.sim.report()
            self.sim.close()
            self.closed = True
            self.routing_error_pct = routing_error_from_rpt(self.rpt_path)
        return self.routing_error_pct

    @property
    def rpt_path(self):
        return os.path.splitext(self.inp_path)[0] + ".rpt"


def routing_error_from_rpt(path):
    """The flow-routing 'Continuity Error (%)' from a SWMM .rpt, or None."""
    try:
        with open(path) as f:
            text = f.read()
    except OSError:
        return None
    block = text.split("Flow Routing Continuity", 1)
    if len(block) < 2:
        return None
    for line in block[1].splitlines():
        if "Continuity Error (%)" in line:
            return float(line.split()[-1])
    return None


_RPT_ERROR = re.compile(r"\bERROR\s+\d+")


def rpt_errors(path):
    """SWMM input/run errors ("ERROR 2xx: ...") reported in a .rpt."""
    with open(path) as f:
        return [ln.strip() for ln in f if _RPT_ERROR.search(ln)]
