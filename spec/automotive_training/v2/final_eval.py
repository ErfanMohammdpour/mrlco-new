#!/usr/bin/env python3
"""FINAL evaluation of the completed campaign: final checkpoint + best-validation.

Restores the CURRENT generation (read-only) and the best-validation weights, then runs the
production `evaluate_all(k_steps=0/3)` on the full validation query split and dumps, per
checkpoint: model latency/energy, the paired baselines (greedy/all-MEC/HEFT), regret, the
deadline/criticality rates and the model's deterministic per-graph action mix.
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
BESTVAL = os.environ.get("BESTVAL_CKPT", "/out/ckpt/meta_model_best_val.ckpt")
OUT = os.environ.get("OUT_JSON", "/evalspace/final_eval.json")

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
LOC = {0: "UE", 1: "MEC", 2: "HELPER"}


def eval_one(label):
    ev = trainer.held_out_evaluator
    k0 = ev.evaluate_all(k_steps=0)
    k3 = ev.evaluate_all(k_steps=3)
    # model deterministic plan + action mix over the full query set
    qg = list(ev.query_graphs)
    env = ev._paired_env(qg, 0, 0, role="query")
    paths = ev._rollout_on(env, ev.policy, adapt_steps=0, seed=0, deterministic=True)
    mix = {"UE": 0, "MEC": 0, "HELPER": 0}
    for p in paths:
        acts = [int(a) for a in np.asarray(p["actions"]).reshape(-1)]
        for a in acts:
            mix[LOC[a]] += 1
    n = len(paths)
    return {"label": label, "k0": k0, "k3": k3, "model_location_mix": mix,
            "model_fraction": {k: round(v / (n * 20), 4) for k, v in mix.items()},
            "n_query_graphs": n}


with sess.as_default():
    # final generation
    prefix = _resolve_resume_ckpt(CKPT)
    saver = tfv1.train.Saver()
    saver.restore(sess, prefix)
    final = eval_one("final")
    final["checkpoint"] = prefix

    # best-validation weights (restored into the same evaluator stack)
    try:
        saver2 = tfv1.train.Saver()
        saver2.restore(sess, BESTVAL)
        best = eval_one("best_validation")
        best["checkpoint"] = BESTVAL
    except Exception as exc:  # the best-val saver may not map 1:1; report instead of fail
        best = {"label": "best_validation", "restore_error": "%s: %s" % (type(exc).__name__, exc)}

    out = {"schema": "v2_final_eval_v1", "final": final, "best_validation": best}
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True, default=str)
    print(json.dumps(out, indent=2, sort_keys=True, default=str)[:6000])
