#!/usr/bin/env python3
"""Meta-test: evaluate the BEST weights on UNSEEN graphs (the held-out meta_test split).

Training is complete and the user has authorised opening meta_test for the final generalization
check (this is the first and only access). Restores the best-validation weights
(meta_model_best_val.ckpt, a joblib weight dict saved by `Seq2SeqPolicy.save_variables`) into the
core policy, repoints the held-out evaluator at the meta_test graphs, and runs the production
evaluate_all on them, then additionally decodes the deterministic per-graph plan.

Env: META_GRAPHS (default 1), CKPT_DIR, BESTVAL_CKPT, OUT_JSON.
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
from spec.automotive_training.automotive_loader import load_dataset
from spec.automotive_training.v2.run_tf import _resolve_resume_ckpt

N = int(os.environ.get("META_GRAPHS", "1"))
CKPT = os.environ.get("CKPT_DIR", "/out/ckpt")
BESTVAL = os.environ.get("BESTVAL_CKPT", "/out/ckpt/meta_model_best_val.ckpt")
OUT = os.environ.get("OUT_JSON", "/evalspace/meta_test.json")
USE_BESTVAL = os.environ.get("USE_BESTVAL", "1") == "1"

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

with sess.as_default():
    sess.run(tfv1.global_variables_initializer())
    core = algo.policy.core_policy
    if USE_BESTVAL:
        core.load_variables(BESTVAL, sess=sess)      # joblib weight dict
        weights_src = "best_validation (meta_model_best_val.ckpt)"
    else:
        saver = tfv1.train.Saver()
        saver.restore(sess, _resolve_resume_ckpt(CKPT))
        weights_src = "final generation"

    ds = load_dataset()
    graphs = list(ds.meta_test(allow_meta_test=True))[:N]
    guard = __import__("spec.automotive_training.automotive_loader",
                       fromlist=["META_TEST_GUARD"]).META_TEST_GUARD

    ev = trainer.held_out_evaluator
    ev.query_graphs = graphs
    ev.support_graphs = graphs

    k0 = ev.evaluate_all(k_steps=0)
    try:
        k3 = ev.evaluate_all(k_steps=3)
    except Exception as exc:
        k3 = {"error": "%s: %s" % (type(exc).__name__, exc)}

    # deterministic per-graph plan
    env = ev._paired_env(graphs, 0, 0, role="query")
    paths = ev._rollout_on(env, ev.policy, adapt_steps=0, seed=0, deterministic=True)
    plans = []
    for i, p in enumerate(paths):
        acts = [int(a) for a in np.asarray(p["actions"]).reshape(-1)]
        plans.append({"graph_id": str(graphs[i].graph_id),
                      "plan": "".join({"UE": "U", "MEC": "M", "HELPER": "H"}[LOC[a]] for a in acts),
                      "location_mix": {LOC[a]: [int(a) for a in acts].count(a) for a in (0, 1, 2)},
                      "makespan_s": float(np.asarray(p["finish_time"]).reshape(-1)[-1])})

    out = {
        "schema": "v2_meta_test_v1",
        "weights_src": weights_src,
        "n_meta_test_graphs": N,
        "meta_test_access_count_after": int(guard.meta_test_access_count()),
        "k0": k0,
        "k3": k3,
        "per_graph_plans": plans,
    }
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True, default=str)
    print(json.dumps({
        "weights_src": weights_src,
        "n_meta_test_graphs": N,
        "meta_test_access_count": int(guard.meta_test_access_count()),
        "k0_latency_s": k0.get("query_mean_latency_seconds_k0"),
        "k0_oracle_s": k0.get("oracle_s"),
        "k0_regret_s": k0.get("k0_regret_s"),
        "plans": plans,
    }, indent=2, default=str))
