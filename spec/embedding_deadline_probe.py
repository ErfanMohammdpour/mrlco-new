#!/usr/bin/env python3
"""⑤a GPU-side probe: does the ENCODER EMBEDDING depend on the deadline?

The numpy half of this test (obs rows differ) lives in tests/test_obs_v3.py.
This script is the second half the review asked for, and it needs TF 1.15
(tf.contrib) on the GPU host:

    # two states identical except for the deadline
    #   A: deadline = 1e6 s (loose)   B: deadline = 1e-3 s (tight)
    # 1) pre-training: encoder_outputs must differ
    # 2) post-training: the action distribution must differ           (Phase ⑧)

Usage:
    python spec/embedding_deadline_probe.py                 # pre-training check
    python spec/embedding_deadline_probe.py --ckpt <dir>    # after training
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MARGO_OBS_VERSION", "v3")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None, help="policy checkpoint dir")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy as np
    import tensorflow as tf

    from env.mec_offloaing_envs.scheduler import encoder_obs as eo
    from env.mec_offloaing_envs.scheduler.model import CanonicalDAG, CanonicalTask
    from env.mec_offloaing_envs.scheduler.resources import ResourceConfig

    eo.set_obs_version("v3")
    resources = ResourceConfig.from_frozen_yaml(model="physical_v1")

    def dag(deadline):
        tasks = [
            CanonicalTask(
                task_id=i,
                compute_workload_bytes=4_166_700 if i == 0 else 2048,
                task_output_bytes=4_166_700 if i == 0 else 2048,
                deadline_s=deadline if i == 0 else None,
                deadline_type="hard" if (i == 0 and deadline is not None) else "none",
            )
            for i in range(3)
        ]
        return CanonicalDAG.from_records(tasks, [(0, 1, 4_166_700), (1, 2, 2048)])

    order = [0, 1, 2]
    obs = {
        "loose": eo.encode_canonical_dag(dag(1e6), order, resources=resources),
        "tight": eo.encode_canonical_dag(dag(1e-3), order, resources=resources),
    }

    from policies.graph2seq_encoder import create_graph2seq_encoder

    tf.compat.v1.disable_eager_execution()
    results: dict[str, object] = {"obs_differ": bool(not np.allclose(obs["loose"], obs["tight"]))}

    with tf.compat.v1.Graph().as_default():
        ph = tf.compat.v1.placeholder(tf.float32, [1, 3, eo.PACKED_DIM], name="obs")
        enc = create_graph2seq_encoder(
            encoder_inputs=ph,
            encoder_units=128,
            num_layers=2,
            is_bidirectional=False,
            mode="train",
            scope_name="probe_encoder",
        )
        out = enc[0]
        with tf.compat.v1.Session() as sess:
            if args.ckpt:
                ckpt = tf.train.latest_checkpoint(args.ckpt)
                if ckpt is None:
                    raise SystemExit("no checkpoint in %s" % args.ckpt)
                saver = tf.compat.v1.train.Saver()
                saver.restore(sess, ckpt)
                results["ckpt"] = ckpt
            else:
                sess.run(tf.compat.v1.global_variables_initializer())
            a = sess.run(out, {ph: obs["loose"][None, :, :]})
            b = sess.run(out, {ph: obs["tight"][None, :, :]})
            diff = float(np.max(np.abs(a - b)))
            results["encoder_output_max_abs_diff"] = diff
            results["encoder_sees_deadline"] = diff > 1e-6

    print(json.dumps(results, indent=2))
    if not results.get("encoder_sees_deadline", False):
        print(
            "FAIL: the encoder output is identical for loose/tight deadlines — "
            "deadline information did not reach the representation."
        )
        return 1
    print("PASS: encoder output depends on the deadline.")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
