#!/usr/bin/env python3
"""Loader for the constrained-training profile (`spec/constraints.yaml`).

The trainer entry point reads `MARGO_CONSTRAINTS` (path to a YAML, or empty /
"off") and calls `constraints_from_env()`.  Keeping the budgets in YAML means a
constrained run is a path change on the GPU host, not a code edit.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

DEFAULT_PATH = Path(__file__).resolve().parent / "constraints.yaml"
OFF_VALUES = ("", "off", "0", "none", "false", "no")


def load_constraints(path: str | Path | None = None) -> tuple[Any, float]:
    """Return (ConstraintSpec | None, dual_lr). Missing file → unconstrained."""
    import yaml

    from env.mec_offloaing_envs.scheduler.constraints import ConstraintSpec

    resolved = Path(path) if path is not None else DEFAULT_PATH
    if not resolved.exists():
        raise FileNotFoundError(f"constraint config not found: {resolved}")
    doc = yaml.safe_load(resolved.read_text()) or {}
    dual_lr = float(doc.get("dual_lr", 0.05))
    spec = ConstraintSpec.from_dict(doc.get("constraints"))
    if not spec.enabled:
        return None, dual_lr
    return spec, dual_lr


def constraints_from_env(var: str = "MARGO_CONSTRAINTS") -> tuple[Any, float]:
    """Unconstrained (None, 0.05) unless MARGO_CONSTRAINTS points at a YAML."""
    raw = str(os.environ.get(var, "")).strip()
    if raw.lower() in OFF_VALUES:
        return None, 0.05
    return load_constraints(raw)
