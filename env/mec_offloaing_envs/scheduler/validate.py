"""Fail-fast numeric validation for canonical scheduling inputs."""

from __future__ import annotations

import math


def require_finite(name: str, value: float) -> float:
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return v


def require_nonneg_int(name: str, value: object) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative int, got bool")
    try:
        v = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative int, got {value!r}") from exc
    if v != float(value):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if v < 0:
        raise ValueError(f"{name} must be >= 0, got {v}")
    return v


def require_positive_rate(name: str, value: object) -> float:
    v = require_finite(name, float(value))
    if v <= 0.0:
        raise ValueError(f"{name} must be > 0, got {v}")
    return v


def require_nonneg_float(name: str, value: object) -> float:
    v = require_finite(name, float(value))
    if v < 0.0:
        raise ValueError(f"{name} must be >= 0, got {v}")
    return v
