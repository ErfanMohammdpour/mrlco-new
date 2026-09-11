"""STATUS: baseline/ablation only (ADR-007)

Load a frozen π_BC copy from a core_policy joblib ckpt. Diagnostic only.

Graph2Seq MeanAggregator names use a process-global UID
(`meanaggregator_12` vs `meanaggregator_45`). Match by canonical slot + UID order.
"""

from __future__ import annotations

import re
from collections import defaultdict

_UID_RE = re.compile(
    r"(meanaggregator|gatedmeanaggregator|maxpoolingaggregator|dense)_(\d+)",
    re.I,
)


def canonical_weight_slot(name, scope):
    rest = name
    prefix = scope + "/"
    if rest.startswith(prefix):
        rest = rest[len(prefix) :]
    return _UID_RE.sub(r"\1_#", rest)


def _uid_sort_key(name):
    nums = [int(m.group(2)) for m in _UID_RE.finditer(name)]
    return tuple(nums) if nums else (0,)


def pair_ckpt_to_vars(loaded, dst_vars, src_scope, dst_scope):
    """Return list of (var, numpy_value) matched by canonical slot and UID order."""
    src_by_slot = defaultdict(list)
    src_prefix = src_scope + "/"
    for key, value in loaded.items():
        if not key.startswith(src_prefix):
            continue
        src_by_slot[canonical_weight_slot(key, src_scope)].append((key, value))
    for slot in src_by_slot:
        src_by_slot[slot].sort(key=lambda kv: _uid_sort_key(kv[0]))

    dst_by_slot = defaultdict(list)
    dst_prefix = dst_scope + "/"
    for var in dst_vars:
        name = var.name if hasattr(var, "name") else str(var)
        if not name.startswith(dst_prefix):
            raise ValueError("expected scope %s, got %s" % (dst_prefix, name))
        dst_by_slot[canonical_weight_slot(name, dst_scope)].append(var)
    for slot in dst_by_slot:
        dst_by_slot[slot].sort(key=lambda v: _uid_sort_key(v.name))

    paired = []
    missing_slots = []
    for slot, dst_list in sorted(dst_by_slot.items()):
        src_list = src_by_slot.get(slot, [])
        if len(src_list) != len(dst_list):
            missing_slots.append((slot, len(src_list), len(dst_list)))
            continue
        for (_src_key, arr), var in zip(src_list, dst_list):
            paired.append((var, arr))
    if missing_slots:
        raise KeyError("ckpt slot mismatch %s" % (missing_slots[:8],))
    if not paired:
        raise ValueError("no variables paired from %s into %s" % (src_scope, dst_scope))
    return paired


from spec.cavia_objective import is_cavia_var_name


def load_named_policy_from_core_ckpt(policy, ckpt_path, sess, src_scope="core_policy"):
    """Assign `policy` vars from a Seq2SeqPolicy.save_variables dump of `src_scope`."""
    import joblib
    import numpy as np

    loaded = joblib.load(str(ckpt_path))
    dst = policy.name
    dst_vars = [v for v in policy.get_variables() if not is_cavia_var_name(v.name)]
    paired = pair_ckpt_to_vars(loaded, dst_vars, src_scope, dst)
    restores = []
    for var, arr in paired:
        arr = np.asarray(arr)
        shape = tuple(int(x) for x in var.shape.as_list())
        if arr.shape != shape:
            raise ValueError(
                "shape mismatch %s ckpt%s var%s" % (var.name, arr.shape, shape)
            )
        restores.append(var.assign(arr))
    sess.run(restores)
    return len(restores)
