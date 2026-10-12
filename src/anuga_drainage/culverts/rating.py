"""Steady SWMM rating of one headwall culvert (1D oracle and rating tables)."""
from __future__ import annotations

import tempfile

from .swmm_engine import SwmmHeadwallEngine


def steady_flows(culvert, stages, *, settle_s=600, routing_step=0.25, workdir=None,
                 extra_options=None):
    """Settled culvert flow for each (hw_stage, tw_stage) pair, one fresh model each.

    Returns a list of (Q m3/s, J_HW head m). ``settle_s`` whole-second strides at
    fixed stages; the result is the mean of the last 60 s.
    """
    out = []
    with tempfile.TemporaryDirectory(dir=workdir) as d:
        for k, (hw, tw) in enumerate(stages):
            eng = SwmmHeadwallEngine([culvert], d, duration_s=settle_s + 10,
                                     routing_step=routing_step, inp_name=f"rating{k}.inp",
                                     extra_options=extra_options)
            qs, hs = [], []
            for s in range(settle_s):
                eng.set_stages([hw], [tw])
                eng.stride()
                if s >= settle_s - 60:
                    qs.append(eng.flows()[0])
                    hs.append(eng.junction_heads()[0])
            eng.close()
            out.append((sum(qs) / len(qs), sum(hs) / len(hs)))
    return out
