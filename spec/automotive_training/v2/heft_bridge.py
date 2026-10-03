#!/usr/bin/env python3
"""Bridge to the frozen HEFT v2 reference plan (re-exported for the v2 gate)."""

from __future__ import annotations

from ..heft_reference_v2 import heft_reference_v2_plan


def heft_plan_for_graph(graph):
    """Return (actions_in_decoder_order, nominal_makespan_s) for a frozen v1 graph."""
    actions, nominal, _result = heft_reference_v2_plan(graph.as_record(), co_physical=True)
    return list(actions), float(nominal)
