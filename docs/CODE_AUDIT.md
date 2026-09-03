# Audit: `MARGO_BASELINE/mrlco-new` only

لیست کامل باگ منطق/یادگیری (بدون هایپرپارامتر و لاگ): [`LOGIC_AND_LEARNING_BUGS.md`](./LOGIC_AND_LEARNING_BUGS.md).

Compared against origin copy `MRLCO/metarl-offloading` to see what this fork changed and what it inherited. No paper, no other trees.

**This code:** Graph2Seq encoder, `obs_dim=20`, `vocab_size=3` {Local, MEC, V2V}, `USE_ENERGY=True`, V2V half-duplex, `clip=0.2`, `inner_batch_size=10`, `n_itr=3500`.

**MRLCO origin:** LSTM encoder, `obs_dim=17`, `vocab_size=2` {Local, MEC}, energy yes but no V2V, `clip=0.3`, `inner_batch_size=1000`, `n_itr=3000`.

---

## What this fork added (real code)

- `policies/graph2seq_encoder.py` + modules; policy calls `create_graph2seq_encoder` instead of LSTM `create_encoder`.
- Feature +3 V2V costs → 20-D. HEFT rank `w[i] = min(local, MEC, V2V)`.
- Action `x==2` with half-duplex channel + helper CPU. Invalid action raises.
- V2V energy: `ptx_v2v`, `prx_v2v`, `rho_v2v`.
- Triple readout (attn/mean/max) for LSTM init.

Shared with MRLCO (not new bugs): Reptile+Adam, PPO no entropy, GAE, `ValueFunctionBaseline` = network values, `max_path_length=20000` sample budget, `end_token` in hparams, loss-accum loop, same graph folder list.

---

## Problems in THIS code

### 1. MEC does not wait for V2V predecessor (logic bug)

Origin MRLCO local wait: `max(FT_locally, FT_wr)`. MEC wait: `max(FT_locally, FT_ws)`. Fine for binary actions.

This fork patched **local** (and V2V) to also wait on `FT_v2v_dl`. **MEC line copied from MRLCO unchanged:**

```python
max([max(FT_locally[j], FT_ws[j]) for j in task_graph.pre_task_sets[i]])
```

No `FT_v2v_dl`. If pred = V2V and succ = MEC, uplink can start before V2V downlink done. Same hole in `greedy_solution`. Introduced by adding V2V, not present in origin.

### 2. `end_token=2` now equals a real action (inherited, now harmful)

MRLCO: `vocab_size=2`, actions `{0,1}`, `end_token=2` is a **sentinel outside** the action set.

This fork: `vocab_size=3`, action `2` = V2V, **same** `end_token=2`.

Train/eval `get_actions` uses sample decoder + `sequence_length` → currently does not stop early. `GreedyEmbeddingHelper` would stop at first V2V. Leftover hyperparam became a landmine.

### 3. `inner_batch_size=10` vs origin `1000` (training scale)

Same sampler budget (~1000 traj/task = 10 rollouts × 100 graphs).

- MRLCO `inner_batch_size=1000` → `batch_number ≈ 1` → Reptile `update_numbers ≈ 1`
- Here `inner_batch_size=10` → `batch_number ≈ 100` → `update_numbers ≈ 100`

Meta-grad divides by `update_numbers`. Outer step ~**100× smaller** than origin for same data. Weights still move; meta-learning is much weaker / noisier sequential Adam.

### 4. Graph2Seq adjacency is a 20-node clique

`sequence_to_graph` tiles full `seq_len×seq_len` adj. Not pred/succ from `.gv`. Mean-agg over everyone, including self.

Origin LSTM at least reads tasks in HEFT order (topo-ish). This GCN does not use DAG edges in the graph. Pred/succ ids only live in the 20-D features (same idea as MRLCO’s 17-D, plus three V2V times).

### 5. Loss accum not reset (`MRLCO.py` / `ppo_offloading.py`)

Identical to origin. `vf_loss`/`pg_loss` init once; after each inner step divide by `K` without reset.

Train `K=1` → printed train loss OK. Eval `K=3` → printed eval loss wrong. **Updates themselves OK.**

### 6. Energy / greedy (design)

- Per-step energy reward = (one graph-level energy score) × (step energy / total). Heuristic.
- Energy min bound = all-MEC TX, ignores cheaper V2V TX.
- Greedy still min finish time only (same as origin); energy just logged. With V2V this matters more.

### 7. Train/eval knobs differ (same pattern as origin, worse gap)

Train: Reptile, `5e-4`, `K=1`, 3500 iters, 20 dist paths.  
Eval: PPO only, `1e-4`, `K=3`, 101 iters, default dist 12.

Origin eval was `n_itr=21`. Here 101. Not a crash; numbers from eval are a different loop than train.

### 8. Small / inherited

- Encoder adapter `dropout=0.1` not passed into `MeanAggregator` (stays 0).
- Train log averages only first 5 of 10 meta-tasks (same as origin).
- `clip_value` 0.2 here vs origin train 0.3.
- `calculate_optimal_solution` unpacks 2-tuple; crashes if energy on. Script leaves energy off. `3^20` unused in train.
- Comment `inner_lr=5e-4  # Increased from 5e-4` is nonsense.
- `Dense` dropout commented (origin Graph2Seq modules; LSTM path unused).

---

## Not problems

- V2V half-duplex channel vs helper CPU: implemented.
- Energy formulas for local/MEC/V2V: live, not commented.
- `get_running_cost` calls schedule **once** per full plan (origin already fixed this; this folder kept the fix).
- `greedy_finish_time[task_id]` with `total_task=20` is per-distribution list; trainer `np.mean` is valid.
- 256-D encoder vs 128-D decoder: Luong + state projection handle it.
- System trains and env.step runs.

---

## Diff summary vs MRLCO (this folder only)

| | Origin MRLCO | This code |
|---|---|---|
| Encoder | LSTM on HEFT sequence | Graph2Seq, **clique** adj + triple readout |
| Actions | 2; `else` = MEC | 3; explicit V2V |
| `end_token=2` | outside action set | **collides with V2V** |
| MEC wait pred | local/ws | **same, missing V2V finish** |
| `inner_batch_size` | 1000 | **10** |
| PPO clip (train) | 0.3 | 0.2 |
| Iters | 3000 | 3500 |
| obs | 17 | 20 |

---

## Fix order if changing this code

1. MEC (and greedy MEC) wait: add `FT_v2v_dl[j]`.
2. `end_token` unused or not `2`.
3. `inner_batch_size` back toward 1000 (or stop dividing Reptile by `update_numbers`).
4. Reset `vf_loss`/`pg_loss` each inner step (eval logs).
5. If encoder should see the DAG: adj = pred∪succ, not clique.
