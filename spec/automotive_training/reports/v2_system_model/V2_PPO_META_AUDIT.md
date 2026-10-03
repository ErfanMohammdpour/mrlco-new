# V2_PPO_META_AUDIT — what is verified, by which method, for the v2 PPO / MRLCO path

**Status: PARTIALLY VERIFIED. Executed (numpy contract + source guards): `test_v2_ppo_contract.py`
(8 tests). NOT RUN: any execution of the TensorFlow 1.15 graph, so no inner-PPO / outer-update
numerics, no checkpoint save/restore and no adapted-policy evaluation are claimed.**

The frozen algorithm is unchanged (inner lr 5e-4, outer lr 5e-4, 3 inner grad steps,
clip 0.2, gamma/GAE unchanged); the v2 stack only swaps the environment family and the
observation version.

## 1. Likelihood-ratio contract — EXECUTED (numpy reference)

`categorical_pd.likelihood_ratio_sym(x, old, new) = exp(log p_new(x) - log p_old(x))` with
`x = x_var`. A numpy reference of that expression is executed and checked:

| property | result |
|---|---|
| unchanged logits give a ratio of exactly 1.0 | PASS (atol 1e-12) |
| ratio equals `p_new(a)/p_old(a)` at the **stored** action | PASS (rtol 1e-12) |
| the ratio would differ if the action were re-sampled | PASS (fixture discriminates, max diff > 1e-6) |
| the ratio is per token, not collapsed over the sequence | PASS (only the changed token moves) |

## 2. Source-level contract — REVIEWED (static, not executed)

| property | evidence |
|---|---|
| the ratio uses the stored actions, the stored old logits and the current new logits | `meta_algos/MRLCO.py` — `likelihood_ratio_sym(self.actions[i], self.old_logits[i], self.new_logits[i])`; `self.actions` is bound to `meta_policies[i].decoder_targets` |
| no action is re-sampled inside the ratio | the same call site; the Gumbel/CRN sampling happens in the sampler, not in the loss |
| each task resets its inner optimizer BEFORE adapting | `UpdatePPOTargetPerTask` calls `self.reset_inner_optimizer(task_id)` first, which runs `variables_initializer(slot_vars)` for that task's Adam slots |
| no adaptation survives into the next meta-batch | `meta_trainer.py` calls `self.algo.sync_task_policies_from_core()` at the TOP of every `for itr in ...` body, before `sampler.update_tasks()` |
| the outer update is a declared FIRST-ORDER pseudogradient | `mean_pseudogradient(theta0, adapted, inner_lr, num_inner_grad_steps)` fed to `outer_optimizer.apply_gradients`; there is no `tf.gradients` anywhere in `MRLCO.py`, i.e. no gradient-of-gradient graph |

## 3. What is NOT verified (and must not be inferred from §1–§2)

* that the TF graph actually produces those numbers (shapes, masking, padding, dynamic
  dimensions, terminal rewards, GAE, advantage normalisation, recurrent-state resets);
* support/query independence in an executed run, and the measured inner/outer update
  magnitudes (`UpdateMetaPolicy` returns no displacement statistics here);
* checkpoint contents (duals, obs schema, normalisation, resolved config, protocol, RNG state)
  and resume equivalence;
* any held-out adaptation result for v2.

Each of those requires the TensorFlow 1.15 environment. The exact commands are in
`V2_RUN_MANIFEST.json`. A passing source guard only proves the code still *intends* the right
thing; it is not evidence that the run is correct.
