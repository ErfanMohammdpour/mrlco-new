#!/usr/bin/env python3
"""Full paired diagnostic: run the REAL evaluator on the current checkpoint and dump everything.

Restores the CURRENT generation (read-only) and calls `evaluate_all(k_steps=0)` and
`(k_steps=3)` exactly as the scheduled validation does, then writes the full result so the
model's own latency, its per-graph plan and the paired baselines (all_MEC / greedy / HEFT)
can be compared on the SAME worlds.
"""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, "/work")
os.environ["MARGO_OBS_VERSION"] = "automotive_v2_obs_v1"

import json
import numpy as np
import tensorflow as tf

from spec.automotive_training.v2.stack import build_automotive_v2_stack
from spec.automotive_training.v2.run_tf import _resolve_resume_ckpt

CKPT = os.environ.get("CKPT_DIR", "/out/ckpt")
OUT = os.environ.get("OUT_JSON", "/evalspace/eval_full.json")

tfv1 = tf.compat.v1
tfv1.disable_eager_execution()
tfv1.reset_default_graph()
tfv1.set_random_seed(0)
np.random.seed(0)

trainer, algo = build_automotive_v2_stack(
    seed=0, n_itr=1, ckpt_dir="/evalspace/tmp_ckpt",
    reward_mode="latency_only", use_energy=True, link_regime="moderate", mec_workers=1,
    reliability=False, background_dags=2, background_policy="all_mec",
    constraints_enabled=True, budget_fractions={"total_energy_mec": 0.5, "ue_energy": 1.0},
    r_select=2, s_select=2, validation_interval=50, checkpoint_interval=10)

cfg = tfv1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
cfg.intra_op_parallelism_threads = 1
cfg.inter_op_parallelism_threads = 1
sess = tfv1.Session(config=cfg)

with sess.as_default():
    prefix = _resolve_resume_ckpt(CKPT)
    saver = tfv1.train.Saver()
    saver.restore(sess, prefix)
    ev = trainer.held_out_evaluator
    k0 = ev.evaluate_all(k_steps=0)
    k3 = ev.evaluate_all(k_steps=3)
    out = {"schema": "v2_eval_full_v1", "checkpoint": prefix, "k0": k0, "k3": k3}
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True, default=str)
    # compact summary
    def summ(b, tag):
        return {
            "tag": tag,
            "model_latency_k%s" % tag: b.get("query_mean_latency_seconds_k%d" % {"k0": 0, "k3": 3}[tag]) if tag in ("k0",) else b.get("query_mean_latency_seconds"),
            "model_latency_s": b.get("query_mean_latency_seconds"),
            "oracle_s": b.get("oracle_s"),
            "regret_s": b.get("k0_regret_s") if tag == "k0" else b.get("k3_regret_s"),
        }
    print(json.dumps({"k0": summ(k0, "k0"), "k3": summ(k3, "k3")}, indent=2, default=str))
