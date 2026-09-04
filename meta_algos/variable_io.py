"""TF 1.x policy weight copy / snapshot. Used so held-out eval never mutates core."""

from __future__ import annotations

import tensorflow as tf


def _session(sess):
    return sess or tf.compat.v1.get_default_session()


def assign_trainable(src_policy, dst_policy, sess=None):
    sess = _session(sess)
    src = src_policy.get_trainable_variables()
    dst = dst_policy.get_trainable_variables()
    if len(src) != len(dst):
        raise ValueError(
            "trainable variable count mismatch: src=%d dst=%d" % (len(src), len(dst))
        )
    sess.run([d.assign(s) for d, s in zip(dst, src)])


def snapshot_trainable(policy, sess=None):
    sess = _session(sess)
    return sess.run(policy.get_trainable_variables())


def restore_trainable(policy, values, sess=None):
    sess = _session(sess)
    variables = policy.get_trainable_variables()
    if len(variables) != len(values):
        raise ValueError(
            "snapshot length %d != trainable count %d" % (len(values), len(variables))
        )
    sess.run([var.assign(val) for var, val in zip(variables, values)])
