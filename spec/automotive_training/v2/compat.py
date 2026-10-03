#!/usr/bin/env python3
"""Runtime-compatibility helpers for the v2 modules.

The frozen TensorFlow 1.15 environment is **Python 3.7**, so v2 code must run there:

* `statistics.fmean` was added in Python 3.8 — `fmean()` below falls back to
  `statistics.mean` (identical value for floats) on 3.7 instead of raising
  `AttributeError` part-way through a measurement;
* no PEP-604 `X | Y` unions and no walrus operator at runtime.
"""

from __future__ import annotations

import statistics

__all__ = ["fmean", "end_lineno_of"]


def fmean(values) -> float:
    """Arithmetic mean of floats, available on Python 3.7 and 3.8+ alike."""
    values = list(values)
    if not values:
        raise ValueError("fmean() requires at least one data point")
    fn = getattr(statistics, "fmean", None)
    if fn is not None:
        return float(fn(values))
    return float(statistics.mean(values))


def end_lineno_of(node, default=None):
    """`ast` node end line, with a Python 3.7 fallback.

    `end_lineno` was added to `ast` nodes in Python 3.8; on 3.7 the attribute is absent, so
    the source-span helpers that use it raise AttributeError instead of degrading.
    """
    value = getattr(node, "end_lineno", None)
    if value is not None:
        return int(value)
    if default is not None:
        return int(default)
    # Python 3.7: take the deepest descendant line. `node.lineno` alone would truncate a
    # multi-line block to its first line (and then fail to `exec`/compile).
    import ast as _ast

    return max(int(getattr(child, "lineno", 0) or 0) for child in _ast.walk(node))
