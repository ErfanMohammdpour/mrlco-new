# ⑥b — Masked-PPO Interface Contract (design + code sites)

Status: **design frozen, reference semantics implemented and tested in numpy; TF wiring is commit 2**
Scope: make the LSTM decoder emit *only feasible* placements (hard shield) without breaking the
PPO importance ratio, the value head, or entropy. No architecture change, no new reward terms.

---

## 1. The one invariant that drives every decision

> **The mask applied when the action was sampled must be the mask applied when its log-probability is
> recomputed at update time.** If they differ, `exp(new_logp − old_logp)` is no longer a likelihood
> ratio and the PPO surrogate (and its clipping) becomes meaningless.

Everything below is a consequence of this. Two masking regimes follow from it:

| Regime | Mask depends on | Safe to derive in-graph? |
|---|---|---|
| **Static** (PROOF mask, obs-derived) | `obs` only (v3 feasibility channels) | **Yes** — deterministic function of the stored observation, so rollout and update agree by construction |
| **Runtime / prefix shield** | episode calendar, duals, contention, `skip` timing | **No** — must be stored with the batch and fed at update |

Corollary already locked into the design: the **mask is stored with the trajectory** either way. The
in-graph derivation is an optimisation for the static case, not a licence to recompute freely.

The failure mode this prevents is reproduced in `env/mec_offloaing_envs/scheduler/tests/test_masking.py`
(`TestTest2RatioCollapsesWhenMaskChanges`): identical logits, mask tightened between rollout and
update under the chosen action ⇒ ratio collapses to `< 1e-6`. This is now a regression test.

---

## 2. Reference semantics — `env/mec_offloaing_envs/scheduler/masking.py`

Numpy/tf-agnostic reference used to define and test the contract before touching the graph.

| Symbol | Meaning |
|---|---|
| `NEG_LARGE = -1e9` | replacement for masked logits (never `-inf`: `-inf` produces NaN gradients) |
| `normalise_mask(feasible)` | bool/float/int → bool, shape-checked |
| `dead_end_guard(feasible)` | rows with no valid action are force-unmasked; returns `(guarded, any_all_invalid)` |
| `apply_mask(logits, feasible, neg)` | `where(feasible, logits, neg)` |
| `masked_softmax` / `masked_log_softmax` | guard → mask → softmax / log-softmax; probability of infeasible action is exactly `0.0` in float32 |
| `entropy_valid` | entropy computed over valid actions only (masked terms contribute `p·log p = 0`) |
| `distribution(logits, feasible)` | `MaskedDistribution(probabilities, log_probabilities, valid_count, any_all_invalid)` |
| `sample(logits, feasible, rng)` | samples from the masked categorical; 1000-draw test never returns an invalid action |
| `likelihood_ratio(action, old_logits, new_logits, feasible, new_feasible=None)` | masked log-probs; `new_feasible` exists **only** to demonstrate the hazard |
| `mask_from_observation(packed, channel_indices)` | v3 feasibility channels → bool mask `(B, T, 3)` |

Numerical facts locked by the 19 tests in `tests/test_masking.py`:
- fixed mask ⇒ `likelihood_ratio ≡ 1` for any logits (even if logits are wildly rescaled);
- masked action probability `== 0.0` exactly;
- all-invalid row ⇒ guard fires, distribution stays finite and sums to 1, counter increments;
- sampling never returns an infeasible action (1000 draws, 2 rows);
- `mask_from_observation` is a pure function of the observation.

Static-only derivation is also enforced: `encoder_obs.feasibility_channel_indices()` raises
`EncoderGraphError` for v1/v2 (the feasibility channels are a v3-only part of the schema; indices are
26/28/30 = `feasible_ue`, `feasible_mec`, `feasible_helper`).

---

## 3. Exact tensor contract

Notation: `B` = trajectories in batch, `T` = packed task slots (`PACKED_DIM`), `A` = `action_dim = 3`
(UE / MEC / HELPER-V2V).

### 3.1 Rollout (sampling) path

| Tensor | Shape / dtype | Produced at | Consumed at |
|---|---|---|---|
| `policy.obs` | `[None, None, obs_dim]` f32 (`obs_dim = PACKED_DIM` for v1/v2/v3 = 19/23/31) | `meta_seq2seq_policy.py:570` | encoder |
| `decoder_full_length` | `[None]` i32 | `:571` | `SampleEmbeddingHelper` |
| `sample_decoder_logits` | `[B, T, A]` f32 | `:251` (`sample_decoder_outputs.rnn_output`) | sampler (stored), `sample_pi`, `sample_q`, `sample_vf` |
| `sample_decoder_prediction` | `[B, T]` i32 | `:258` (helper `sample_id`) | environment step, stored as `actions` |
| `sample_neglogp` | `[B, T]` f32 | `:265` (`softmax_cross_entropy_with_logits_v2` on `sample_decoder_logits`) | diagnostics |
| `sample_pi` / `sample_q` / `sample_vf` | `[B,T,A]` / `[B,T,A]` / `[B,T]` | `:252/:253/:256` | `get_actions` → `values` |
| **`feasible` (NEW)** | `[B, T, A]` bool (or f32 0/1) | derived from obs v3 channels **or** fed by env shield | feed into helper + stored |
| **`masked_sample_logits` (NEW)** | `[B, T, A]` f32 | `where(feasible, sample_logits, NEG_LARGE)` | helper `sample()`, `sample_pi`, `sample_q`, `sample_vf`, `sample_neglogp` |

### 3.2 Update path (inner PPO, evaluator + trainer)

| Tensor | Shape / dtype | Where |
|---|---|---|
| `self.actions` | `[B, T]` i32 | `ppo_offloading.py:86` ← stored `actions` |
| `self.old_logits` | `[None, None, A]` f32 placeholder | `ppo_offloading.py:83` ← **stored, already-masked** logits |
| `new_logits` | `[B, T, A]` f32 | `ppo_offloading.py:81` ← `policy.network.decoder_logits` (training decoder) |
| `likelihood_ratio` | `[B, T]` f32 | `ppo_offloading.py:96` → `CategoricalPd.likelihood_ratio_sym` |
| `old_v`, `advs`, `r` | `[None, None]` f32 | `:91-93` |
| `vf` | `[B, T]` f32 | `policy.vf` (`pi·q`) — **must be computed from masked `pi`/`q`** |
| `total_loss` | scalar | `:114` |

**Decision (frozen): `old_logits` carries the masked logits.** The sampler stores
`masked_sample_logits`, so `old_logits` is consistent with whatever distribution actually produced
the action. The mask is additionally stored (`feasible`) so that the update recomputes the *same*
masked `new_logits`, and so that a runtime shield can be replayed exactly.

---

## 4. Exact code sites for commit 2

Four small edits, each with a single responsibility:

1. **`policies/meta_seq2seq_policy.py`**
   - `Seq2SeqNetwork.__init__` (~`:229-231`, `:251-256`): add
     `self.feasible_mask` placeholder `[None, None, A]` (f32, default `None` ⇒ no masking) and define
     `self.decoder_logits_masked`, `self.sample_decoder_logits_masked`,
     `self.greedy_decoder_logits_masked`; build `pi`, `q`, `vf`, `neglogp` from the **masked** logits.
   - `FixedSequenceLearningSampleEmbedingHelper.sample` (`:63-87`): apply the mask to `outputs`
     **before** `categorical.Categorical(logits=...)`, and before the top-p branch. This is the
     critical site — the environment action comes from this helper, so masking only `pi` would leave
     sampling unmasked.
   - `Seq2SeqPolicy.get_actions` (`:654-664`): accept `feasible_mask=None`, feed it, return the masked
     logits for storage.
2. **`policies/distributions/categorical_pd.py`** (`:68-106`): no arithmetic change; document that
   `logits` are pre-masked (`NEG_LARGE`) so `softmax_cross_entropy_with_logits_v2` yields `-1e9`
   (finite) for infeasible targets. Entropy (`:131`) inherits correct "valid-only" behaviour for free
   because `p = 0` on masked entries.
3. **`meta_algos/ppo_offloading.py`** (`:80-98`, `:132-203`): add `self.feasible` placeholder, feed it
   into the policy graph, and add `feasible` to `task_samples` handling. Keep `old_logits` = stored
   masked logits. Add assertions: `feasible.shape == new_logits.shape`; if `feasible` is absent, no
   masking (byte-exact legacy path).
4. **`meta_algos/MRLCO.py`** (~`:135`) — the **second** `likelihood_ratio_sym` site (meta inner-update
   over task slots). Same treatment; missing this is a silent correctness bug.
5. **`samplers/seq2seq_meta_sampler.py`** (`:126-141`): store `masked_logits` and `feasible` in
   `path_dict` (additive keys, backward compatible). Also stash the raw logits for diagnostics.

Flag: `MARGO_MASK_MODE = off | static | runtime` (default `off`). `off` reproduces every historical
number byte-exactly; `static` derives the mask in-graph from the obs v3 channels; `runtime` additionally
feeds an env-provided shield mask and stores it.

---

## 5. Why mask-before-softmax (and not "renormalise after")

- Masking after softmax changes the *support* of the distribution but not `q`; `vf = Σ pi·q` and
  entropy would then be computed from a renormalised `pi` that never saw the mask, so the value head
  and entropy would silently disagree with the sampling distribution.
- `NEG_LARGE = -1e9` instead of `-inf`: `exp(-1e9) = 0.0` exactly in f32 (prob 0 as required), while
  gradients stay finite. `-inf` gives `NaN` in `softmax_cross_entropy_with_logits_v2` when the
  target is masked, and `NaN` gradients for the whole minibatch.
- Dead-end guard (`all-masked ⇒ unmask all`) is the only sanctioned way to avoid a `NaN`
  distribution. It must **count** its activations (`mask/dead_end_rows`) because a rising count means
  the feasibility model is too aggressive, not that the policy is learning.

---

## 6. Static vs runtime mask — decision table

| Question | Static (PROOF / obs v3) | Runtime shield |
|---|---|---|
| Where does the mask come from? | `encoder_obs.feasibility_channel_indices()` → 26/28/30 | env step, after calendar/duals/contention |
| Can it be derived at update from stored `obs`? | Yes (pure function) | No — prefix-dependent |
| Stored in batch? | Yes (cheap, enables the assertion) | Yes (mandatory) |
| `mask/applied` in graph? | Yes | Yes |
| Risk if forgotten | none (deterministic) | **ratio collapse** (see §1 test) |
| First wiring | commit 2 | after ⑥b lands (placeholder already reserved) |

Commit 2 wires **static** only, with the runtime placeholder reserved and inert (`None`), so the
runtime shield can be added later without touching the PPO loss again.

---

## 7. TF smoke plan (must run on kish — local Python 3.14 has no TF)

1. **Graph build + shapes**
   `python -c "build policy with MARGO_OBS_VERSION=v3; assert sample_decoder_logits.shape[-1]==3; assert feasible_mask placeholder present"`.
2. **Ratio identity (the invariant)**
   One graph, two runs with the *same* `feasible`, different `obs`→ same logits fed as `old_logits`:
   assert `max|ratio − 1| < 1e-5` over all `(B,T)`.
3. **Zero probability**
   For every `(b,t,a)` with `feasible[b,t,a] == False`: `sample_pi[b,t,a] == 0.0`, and
   `log_likelihood_sym` for that action `≤ -1e8`.
4. **Masked sampling**
   100 rollouts on a seeded env: every emitted action satisfies `feasible[b,t,a]`; count
   `mask/dead_end_rows` and assert `0` for the standard dataset.
5. **Gradient sanity**
   One `sess.run(self._train)`; assert all grads finite (`np.isfinite`) and
   `total_loss` finite; assert `off`-mode loss equals the legacy loss bit-for-bit.
6. **One-iteration trainer run** (v3 obs, `objective_mode="log_only"`, no deadlines):
   1 meta-iteration on kish with masking `off` then `static`; compare `.npz` logs — everything except
   the mask counters must match, proving the flag path is additive.

---

## 8. Test status (local, non-TF)

- `tests/test_masking.py`: **19 passed**.
- Full non-TF scheduler suite (excluding the two TF modules that require `tensorflow`):
  **381 passed, 5 skipped**.
- Two failures found while writing the tests were fixed and are worth recording because both were
  real interface traps:
  - the ratio-collapse test originally used a single mask argument, so both log-probs shared the
    changed mask and the ratio stayed `1.0`; `likelihood_ratio(..., new_feasible=...)` now models
    "mask changed under the action".
  - the batch-shape test used a 3-row mask against 2-row logits; the contract is one mask row per
    logits row, and `apply_mask` rejects anything else loudly instead of broadcasting.
- A third, subtler trap: `encoder_obs`'s obs version is **module-global**, so a test that sets `v3`
  leaks into later modules and breaks the encoder pack tests. `TestTest5MaskIsDeterministicFromState`
  now saves and restores `eo.OBS_VERSION` in `setUp`/`tearDown`.

---

## 9. Rollback / observability

New scalar summaries (all prefixed `mask/`): `applied` (0/1), `frac_masked`, `dead_end_rows`,
`all_masked_batches`, `sampled_infeasible` (must stay 0), `ratio_max_dev` (must stay `< 1e-5` under
`static`). Any non-zero `sampled_infeasible` or `dead_end_rows` in a standard run is a bug in the
feasibility model, not a policy issue — treat it as a build failure.

## 10. Explicit non-goals for commit 2

No entropy-coefficient change, no clip-ratio change, no reward change, no decoder width change, no
encoder change, no runtime shield activation. Commit 2 ships the plumbing plus counters only; the
PPO log-prob/entropy correction (commit 3) and dead-end wiring (commit 4) use the same interface.
