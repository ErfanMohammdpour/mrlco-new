#!/usr/bin/env python3
"""Dump meanagg encoder_outputs for a frozen packed fixture.

Run on Kish with TensorFlow 1.15 BEFORE trusting encoder_type edits:
  python spec/dump_meanagg_fixture.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import tensorflow as tf

from spec.encoder_fixture import HIDDEN, OUTPUT_PATH, PACKED_PATH, SEED, save_packed_fixture
from policies.graph2seq_encoder import Graph2SeqEncoderAdapter


def main():
    packed_path = save_packed_fixture()
    packed = np.load(str(packed_path))
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.set_random_seed(SEED)
    np.random.seed(SEED)
    ph = tf.compat.v1.placeholder(tf.float32, [None, None, packed.shape[-1]], name="packed")
    adapter = Graph2SeqEncoderAdapter(
        input_dim=packed.shape[-1],
        hidden_dim=HIDDEN,
        num_layers=2,
        bidirectional=True,
        mode="eval",
    )
    outputs, _state = adapter.encode(ph)
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        out = sess.run(outputs, feed_dict={ph: packed})
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(OUTPUT_PATH), np.asarray(out, dtype=np.float32))
    print("packed", packed.shape, packed_path)
    print("outputs", out.shape, OUTPUT_PATH, "finite", bool(np.isfinite(out).all()))


if __name__ == "__main__":
    main()
