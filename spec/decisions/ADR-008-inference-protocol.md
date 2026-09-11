# ADR-008: What counts as the learned method at inference

Status: Accepted  
Decision date: 2026-09-08

## Decision

The learned method at inference is the policy network π plus, optionally, **sampling k plans from π** and calling `schedule()` once per sample, then keeping the lowest-T (or lowest-J) plan.

Maximum k that may be reported as the method: **k ≤ 64**. Every reported T MUST be paired with `evals_per_graph` (equals k for this protocol; 1 for greedy decode).

Handcrafted neighbourhood search is **not** π:

- `pair_head` / `pair_ksweep` / `pair_ranker` / `pair_seq`
- Hamming-2 (`hamming2_probe.py`)
- 2-opt (`iterate_2opt`)

Those operators are (a) **teachers**: labels for BC on train graphs only, or (b) **reference ceilings** in tables. They must not appear inside the paper method's test loop.

Greedy decode (`model="greedy"`, k=1, `evals_per_graph=1`) is always the zero-shot number. Best-of-k is additional.

## Why

POMO (Kwon 2020), PolyNet, and EAS treat multi-sample decoding as neural inference: the search distribution is π, not a hand-built neighbourhood. Kish already showed the skill lives in the tail (`bc_greedy_tmec` iter 0: T_p10 **445** vs mean **668**, log §4; `bc_only_eval` train p10 **320** vs mean **533**, log §5). Pair-seq bestimp k20 val **447.9** at ~718 evals (log §18) is a heuristic ceiling, not a network.

Cap k=64 so the method stays cheaper than motif search (pair k20 ≈ 180 evals already; k50 sequential ~1671). If Phase 2 needs more than 64 samples to move T, that is a negative result, not an excuse to raise k.

## Consequence

- Phase 2 implements `spec/best_of_k.py` with k ∈ {1,4,8,16,32,64}. Sample 0 is greedy, so best-of-k ≥ greedy by construction.
- Phase 3 adaptation may use k=16 **on support only**; query reports greedy and best-of-32 separately.
- Ledger JSON always includes `evals_per_graph`.
- `paper_result` rows in Phase 6 that omit evals are invalid.
