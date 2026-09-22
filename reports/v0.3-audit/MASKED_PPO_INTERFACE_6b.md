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

- `tests/test_masking.py`: **32 passed** (19 reference-semantics + 13 mode/shield/wiring).
- Full non-TF scheduler suite (excluding the two TF modules that require `tensorflow`):
  **394 passed, 5 skipped**.
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

---

## 11. Commit 2 implementation notes (what actually landed)

Implemented exactly the four edits above plus the sampler/environment side. The
guiding choice was **fail loud, never silently degrade**:

- `MARGO_MASK_MODE` (`off|static|runtime`, default `off`) is resolved **at policy
  construction**. In `off` the mask is `None`, so `mask_logits_tf` returns the raw
  logits tensor and the graph is the legacy graph — no extra placeholder, no extra
  op, byte-exact reproduction. In `static`/`runtime` a real `feasible_mask`
  placeholder is created; code paths that do not know about masks now fail loudly
  instead of silently running unmasked.
- The mask is applied to **all three decoders** (train / sample / greedy) *and*
  inside the sampling helper. The helper is the site that actually shields the
  environment: `sample_decoder_prediction` comes from
  `FixedSequenceLearningSampleEmbedingHelper.sample`, not from `pi`. Inside the
  decoder loop the per-step logits are `[B, A]`, so the helper passes `time` and
  the mask is gathered at that step (`tf.gather(mask, time, axis=1)`) — a 3-D mask
  broadcast onto 2-D logits would have failed at runtime.
- `pi`, `q`, `vf`, `entropy`, `neglogp` and both `CategoricalPd` log-likelihoods
  are built from the **masked** logits, so the value head and the importance ratio
  agree with the sampling distribution by construction.
- The dead-end guard is in-graph too, and its count is exposed as
  `network.dead_end_rows` / `network.sample_dead_end_rows` for later logging.
- **The sampler stores the mask the policy reports it used** (`last_feasible_mask`
  / `last_feasible_masks`), rather than recomputing it. `path_dict["feasible"]` is
  threaded through both sample processors into `samples_data["feasible"]`, and
  `PPO.UpdatePPOTarget` / `MRLCO.UpdatePPOTargetPerTask` raise if the key is
  missing while masking is active. This is the §1 invariant enforced in code.
- Feed builders were centralised: `policies.meta_seq2seq_policy.feasibility_feed`
  is now used by `spec/bc_greedy_mec.policy_feed`, so BC/phase-4 scripts and the
  context baselines get the shield without per-script changes.
- The frozen BC reference in the KL term is fed an **all-feasible** mask, i.e. the
  unmasked legacy distribution: `p=0` on infeasible actions makes those entries
  contribute nothing to `KL(pi_new || pi_BC)`.

Verification available locally: `masking.py` semantics, the wiring-completeness
test (`test_tf_wiring_covers_every_decoder`), and `py_compile` on every edited
module. The graph itself cannot be built here (no TF), so
`spec/masked_ppo_smoke.py` is the gate to run on kish:

    MARGO_OBS_VERSION=v3 MARGO_MASK_MODE=off    python -m spec.masked_ppo_smoke
    MARGO_OBS_VERSION=v3 MARGO_MASK_MODE=static python -m spec.masked_ppo_smoke

It checks placeholder/mode agreement, shapes, `sampled_actions_are_feasible`,
`infeasible_probability_is_zero`, `probabilities_sum_to_one`,
`ratio_identity_with_fixed_mask` (max deviation < 1e-5), the dead-end guard count,
`infeasible_action_logp_is_neg_large`, and finite gradients, and exits non-zero on
any failure so it can gate a commit.

Known unwired call sites (they must run with `MARGO_MASK_MODE=off`, and now fail
loudly otherwise): `spec/pair_sup.py`, `spec/rewrite_mec.py`,
`spec/cavia_loop.py`, `spec/best_of_k.py`, `spec/oracle_dist.py`,
`spec/pair_head.py`, `comprehensive_encoder_verification.py`. Wiring them is
mechanical (`feasibility_feed`) but belongs to commit 4/6 when those scripts are
actually used with v3.

---

## 12. Post-audit P0 fixes (external audit of `08d74d8`)

An independent audit of the branch found three blockers. All three were verified
in the code and are fixed here; the audit's positive findings (mask applied at the
real sampling site, stored mask replayed at update, both update paths covered)
were confirmed.

### P0-1 critic contaminated by `-1e9` (real, most severe)

`q = dense(masked_logits)` was wrong: a dense layer mixes *all* three logits, so a
single masked entry at `-1e9` reaches the `q` of the valid actions too and
`vf = sum(masked_pi * q)` lands around `1e8`-`1e9`. The value clip becomes
meaningless and `vf_loss` explodes. Fixed by masking the **support** and not the
critic features:

    q  = dense(raw_logits)          # train / sample / greedy
    vf = sum(masked_pi * q)

In `off` mode `decoder_logits_raw is decoder_logits`, so the legacy path is
untouched. The smoke now asserts `max|vf| < 1e3` and that `vf` equals
`(masked_pi * q).sum(-1)` in numpy — a direct regression test for this blocker.
The auditor's longer-term suggestion (critic from the decoder hidden state rather
than the policy logits) is recorded as a separate design item; it changes the
architecture and is out of scope for ⑥b.

### P0-2 smoke ratio test used the wrong prefix (real)

The smoke fed all-zero `decoder_inputs` while `old_logits` came from the
autoregressive sample decoder and `new_logits` from the teacher-forced train
decoder, so the identity could fail for reasons unrelated to masking. Fixed with
the same shifted-action prefix the real PPO update uses
(`[0, a_0, ..., a_{T-2}]`). The smoke also now: runs an **actual** Adam step on a
PPO-shaped loss (clipped surrogate + value clip) and asserts finite losses, a
bounded value loss, non-empty finite gradients and moved parameters. The ratio
check is named `ratio_identity_before_first_update` — after an update the ratio is
expected to leave 1, so no metric may require `ratio≈1` during training.

### P0-3 soft/firm deadlines were silently hard-shielded (real)

`static_bounds.feasible_by_deadline` returns proof feasibility for *any* deadline
type, and `observation_mask` turned it straight into a shield. A task with a soft
deadline therefore lost actions that merely incur tardiness — contradicting the
locked curriculum (soft -> objective, firm -> constraint channel, hard -> shield).
Fixed in `masking.observation_mask`:

    mask = proof_mask OR NOT hard_deadline      # hard => proof mask; else all allowed

`static_base_mask()` keeps the featurised proof signal available for soft/firm (the
policy can still learn to price lateness); only the shield is gated. Feature names
are unchanged because renaming `feasible_*` would break the v3 schema and stats
file, so the distinction is documented instead: these are *proof* channels, not
shield channels.

### P0-4 runtime mode no longer degrades silently

`observation_mask(mode="runtime")` now raises: the shield is prefix-dependent and
the environment consumes a whole decoded plan at once, so a per-prefix mask cannot
exist yet. A caller must pass the env shield explicitly; `intersect_masks` is the
documented way to combine it with the static base. No call site feeds such a shield
yet, so `runtime` is formally **reserved/unsupported**, not "working".

### P0-5 smaller items from the audit

- `network.decoder_prediction` (teacher-forced argmax) now comes from the **masked**
  logits, so BC/label diagnostics cannot report a shield-forbidden action. In `off`
  mode this equals the previous `sample_id` (argmax of the same logits).
- `spec/kish_gpu.sh` now forwards `MARGO_OBS_VERSION` / `MARGO_MASK_MODE` when set on
  the host, and has explicit `mask-smoke` (off + static) and `mask-runtime` targets.
  Previously the launcher passed neither, and `phase4_train_driver` forces
  `MARGO_OBS_VERSION=v2` at several entry points — in mask mode those paths now fail
  loudly through `observation_mask` instead of silently ignoring the shield.
- Smoke CLI: `--mask-mode {off,static,runtime}`, `--obs-version v3`; the script sets
  the env vars itself before importing, so the command is reproducible.
- Wiring-completeness test also asserts the critic reads raw logits (no
  `dense(self.*decoder_logits,` may reappear).

Still open and explicitly NOT claimed: the `train` decoder's `TrainingHelper` feeds
targets (teacher forcing), so `decoder_logits` are computed from ground-truth
actions — correct for the PPO importance ratio, which is what PPO uses; and the
remaining unwired decode call sites (`spec/pair_sup.py`, `spec/rewrite_mec.py`,
`spec/cavia_loop.py`, `spec/best_of_k.py`, `spec/oracle_dist.py`,
`spec/pair_head.py`, `comprehensive_encoder_verification.py`) still require
`MARGO_MASK_MODE=off`. Metrics (`mask/*`, `policy/*`, `critic/*`) are the next
commit, before any 500-iteration run.

---

## 13. Audit round 2 — gate and watcher fixes (`1f7edbd` review)

The P0 fixes in section 12 were confirmed correct by the second audit round. Three
defects in the *gate tooling* (not in the shield) were reported and are fixed:

### 13.1 `branch_watch` lost commits (definite bug)

`git ls-remote` reveals a SHA without fetching its objects, so `git log prev..head`
printed nothing for the new head, the report showed `new_commits: []`, and the
marker was advanced anyway — the commit disappeared from the feed permanently. The
reproduction (synthetic bare remote, push a second commit) previously produced
exactly that. Fixed by always fetching the head into
`refs/remotes/<remote>/<branch>` before computing the log, and by treating a failed
fetch as fatal **even when a stale tracking ref still resolves** (the second
variant of the same bug: stale ref compares equal to the marker and reports "no
new commits"). The marker is now advanced only after a report is computed, so a
failed run retries next time.

Force-push / rebase is detected with `git merge-base --is-ancestor` and reported as
`history_rewritten: true` with an explanatory note; the range listing still shows
what is new. Verified end to end against a synthetic remote: first look, new
commit, force-push (`history_rewritten: true`), fetch failure with a stale ref
(exit 1, marker untouched), and the real `erfan/phase4-eval`.

### 13.2 `mask-runtime` swallowed every failure

`gpu_run_masked runtime ... || true` would have hidden an import error, a CUDA
failure or a missing image behind a green exit code. Removed. The smoke itself
catches the expected `ValueError` and exits 0, and now records the caught exception
type and message in the JSON report, so a green run is evidence rather than
silence.

### 13.3 the smoke re-initialised the whole graph

The Adam step was preceded by a second `global_variables_initializer()`, which
wiped the weights that produced `old_logits` / `old_v`; the "PPO-shaped update" was
therefore not replaying the rollout it had just sampled. Now only the Adam slot
variables are initialised. Related fix: gradients are kept paired with their
variables (`[(g, v) ...]`) instead of filtering a gradient list against the full
parameter list, which would have silently applied gradients to the wrong variables.

### 13.4 the watcher fix is now covered by the suite

`tests/test_branch_watch_tool.py` (6 tests, run inside a throwaway repo so the
project's own refs stay clean) asserts: first look records the head; a newly pushed
commit is reported; the marker only advances after a report; a force-push sets
`history_rewritten`; and a failed fetch — with and without a stale tracking ref —
exits non-zero and leaves the marker untouched. Non-TF suite: **400 passed, 5 skipped**.

### 13.5 corrupt/unknown marker is fatal too (audit round 3)

`git merge-base --is-ancestor` returns neither 0 nor 1 when `previous` is not a
commit known to the repository (fake or corrupt marker, or an object that was never
fetched). The code read that as "not rewritten", then `git log` failed inside
`check=False` and returned an empty list, so the run printed `new_commits: []`,
exited 0 and advanced the marker past real commits. Both halves are now fatal:
an undecidable comparison raises `cannot compare previous ... with head ...`, and
`log_entries` uses `check=True` so a failed log can never masquerade as "no new
commits". Reproduced with the auditor's fake marker: exit 1, `state_advanced:
false`, marker untouched. Test 7 covers it.

The runtime smoke now also asserts the required message substring
(`runtime shield masks cannot be derived`), not merely that a `ValueError` was
raised, so an unrelated error cannot make the runtime gate look green.

Non-TF suite: **401 passed, 5 skipped**.

### 13.6 kish TF/GPU gate: PASSED (`34fd49d`)

Host `kish-ai`: RTX 4090 24GB idle (1 MiB used, 0% util), image
`margo-phase4-tf115-nv2212` present, 427G free. The existing checkout
`/opt/margo/mrlco-new` is stale (`39ea0ed`) with 3661 dirty entries, so it was
left untouched; a clean clone lives at `/opt/margo/mrlco-new-6b`.

    MARGO_ROOT=/opt/margo/mrlco-new-6b bash spec/kish_gpu.sh mask-smoke
    MARGO_ROOT=/opt/margo/mrlco-new-6b bash spec/kish_gpu.sh mask-runtime

Run from the committed SHA (dirty=0), TF 1.15.5 / Python 3.8, all three exit 0
with `failures: []` — off 18/18 checks, static 21/21, runtime 2/2. Evidence:
`reports/v0.3-audit/mask_smoke/kish_34fd49d_evidence.json`.

The three checks that directly refute the audit blockers, now measured rather
than argued: `critic_value_scale_is_sane` and `value_is_masked_support_dot_raw_q`
(the `-1e9` critic contamination), `shield_is_hard_deadline_only` (soft/firm no
longer shielded), and `ratio_identity_before_first_update` (correct prefix). The
runtime gate reports the expected
`ValueError: runtime shield masks cannot be derived from the observation`.

Reaching a green run also exposed three bugs in the smoke harness itself, fixed in
`34fd49d`: the sample-decoder fetches never fed `obs`/`decoder_full_length`;
embedding gradients arrive as `IndexedSlices` and came back as a ragged object
array; and the stored mask is float32, so `~mask` raised `TypeError`.

Order from here is unchanged: the `mask/*`, `policy/*`, `critic/*` metrics commit,
then the no-deadline 500-iteration sanity run — in which
`active_rate = forced_rate = all_invalid_rate = invalid_action_rate =
argmax_masked_rate = 0` is the expected result, not a failure.
`critic/*` metrics commit, then the no-deadline 500-iteration sanity run — in which
`active_rate = forced_rate = all_invalid_rate = invalid_action_rate =
argmax_masked_rate = 0` is the expected result, not a failure.
