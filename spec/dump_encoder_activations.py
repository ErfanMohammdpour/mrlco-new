#!/usr/bin/env python3
"""⑤b GPU-side step: dump REAL encoder activations for the effective-rank probe.

Runs the frozen Graph2Seq encoder (TF 1.15 / tf.contrib) over a slice of graphs
with obs v3 and saves the tensors that `effective_rank_probe.py --mode
activations` consumes:

    node_h    [G, 20, 256]   encoder_outputs (Luong memory)
    proj256   [G, 256]       readout_proj output (before state projection)
    state128  [G, 128]       state projection output fed to the LSTM
    labels    [G, 20]        per-task teacher actions (greedy_from_mec)

Loose vs tight deadlines are dumped as separate keys (`*_loose`, `*_tight`) so the
rank difference caused by deadline information is measurable.

Usage (on the GPU host, or CPU for a smoke run):
    python spec/dump_encoder_activations.py --graphs 120 --out acts.npz
    python spec/effective_rank_probe.py --mode activations --npz acts.npz
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MARGO_OBS_VERSION", "v3")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", type=int, default=1)
    ap.add_argument("--graphs", type=int, default=120)
    ap.add_argument("--deadline-factor", type=float, default=1.05)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--ckpt", type=str, default=None, help="optional policy checkpoint dir")
    ap.add_argument(
        "--obs-version", type=str, default="v3",
        help=(
            "obs version used to BUILD the encoder. A checkpoint trained with "
            "v1/v2 has a different packed width and cannot consume v3 obs; pass "
            "--obs-version v1 together with a v1/v2 checkpoint for a trained "
            "baseline without deadline features."
        ),
    )
    args = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    import numpy as np
    import tensorflow as tf

    from env.mec_offloaing_envs.scheduler import encoder_obs as eo

    eo.set_obs_version(str(args.obs_version))
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "effective_rank_probe", str(ROOT / "spec" / "effective_rank_probe.py")
    )
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)

    rows, resources = probe.build_dataset(args.dist, args.graphs, args.deadline_factor)
    loose = probe.stack(rows, "loose")
    tight = probe.stack(rows, "tight")
    labels = probe.teacher_labels(rows).reshape(len(rows), -1)

    from policies.graph2seq_encoder import create_graph2seq_encoder

    tf.compat.v1.disable_eager_execution()
    packed_dim = int(loose.shape[2])
    with tf.compat.v1.Graph().as_default():
        ph = tf.compat.v1.placeholder(tf.float32, [None, 20, packed_dim], name="obs")
        # Scope MUST be "encoder" to match the variable names stored by the policy
        # (policies/meta_seq2seq_policy.py builds it with scope_name="encoder").
        # Using a different scope makes a checkpoint restore silently impossible.
        enc = create_graph2seq_encoder(
            encoder_inputs=ph, encoder_units=128, num_layers=2,
            is_bidirectional=False, mode="train", scope_name="encoder",
        )
        node_h = enc[0]
        state = enc[1]
        # readout_proj is not returned separately; recover the 256-d pre-projection
        # by running the encoder and, if available, the policy's state projection.
        print("encoder_outputs shape:", node_h.shape)
        print("encoder_state type:", type(state))
        with tf.compat.v1.Session() as sess:
            if args.ckpt:
                ckpt = tf.train.latest_checkpoint(args.ckpt)
                if ckpt is None:
                    raise SystemExit("no checkpoint found under %s" % args.ckpt)
                enc_vars = [
                    v for v in tf.compat.v1.global_variables()
                    if v.name.startswith("encoder/")
                ]
                if not enc_vars:
                    raise SystemExit("no encoder/ variables were built; scope mismatch")
                reader = tf.compat.v1.train.NewCheckpointReader(ckpt)
                stored = set(reader.get_variable_to_shape_map().keys())
                missing = [v.name for v in enc_vars if v.name.split(":")[0] not in stored]
                print(
                    "checkpoint=%s  encoder vars=%d  missing_in_ckpt=%d"
                    % (ckpt, len(enc_vars), len(missing))
                )
                if missing:
                    print(
                        "NOTE: %d encoder variables are absent from the checkpoint "
                        "(e.g. %s). They keep their random init; treat the probe as "
                        "PARTIAL." % (len(missing), missing[:3])
                    )
                saver = tf.compat.v1.train.Saver(var_list=enc_vars)
                saver.restore(sess, ckpt)
                print("restored encoder variables")
            else:
                sess.run(tf.compat.v1.global_variables_initializer())
            out = {}
            for name, packed in (("loose", loose), ("tight", tight)):
                h = sess.run(node_h, {ph: packed})
                out[f"node_h_{name}"] = h
                # encoder_state is an LSTMStateTuple((c, h)) tuple of 2 layers here
                try:
                    st = sess.run(state, {ph: packed})
                    arr = np.asarray(st[0] if isinstance(st, (tuple, list)) else st)
                    out[f"state_{name}"] = arr.reshape(len(packed), -1)
                except Exception as exc:  # pragma: no cover - diagnostic
                    print("state dump skipped: %s" % exc)
    np.savez_compressed(args.out, labels=labels, **out)
    print("wrote %s with keys: %s" % (args.out, sorted(np.load(args.out).files)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
