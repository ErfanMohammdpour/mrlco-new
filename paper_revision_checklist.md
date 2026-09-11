# MARGO paper revision checklist

## Figures — Nano Banana (replace temporary `figures/fig_*.png`)

- [ ] **Fig 1** `fig_system_model.png` — final vector-style system diagram (current file may be deck export).
- [ ] **Fig 2** `fig_margo_architecture.png` — full pipeline with tensor shapes and PPO/Reptile.
- [ ] **Fig 3** `fig_graph2seq_encoder.png` — DAG neighborhoods + two-layer aggregation + 256-D.
- [ ] **Fig 4** `fig_triple_readout.png` — three branches + concat + projection.
- [ ] **Fig 5** `fig_decoder_action_generation.png` — **replace placeholder** (currently copied from architecture export).
- [ ] **Fig 6** `fig_meta_reptile_training.png` — inner/outer loop, M=10, K=1.
- [ ] **Fig 7** `fig_joint_objective.png` — conceptual α trade-off (no fake data ticks).
- [x] **Fig 8 (evaluation / benchmark)** `fig_graph_fatness_density.png` — used in the paper for DAG diversity (fatness/density); **not** generating `fig_dataset_generation`, `fig_evaluation_pipeline`, or `fig_results_summary`.

Prompts: `figure_prompts_for_nano_banana.md` (Figs. 1–7 + note on Fig.~8 asset).

## Figures — experimental JPGs (`results/`, `results/mrlco-compare/1/`, `results/mrlco-compare/2/`)

- [ ] Restore or add all referenced `*.jpg` files if missing from the repo clone (LaTeX `\graphicspath` expects them).
- [ ] **Task 2 panels:** verify `task2-*.jpg` paths match filenames on disk (typos: `policy-loses` vs `policy-losses` in task1 filenames).
- [ ] Optional: combine groups into `figure*` multi-panel layouts for IEEE readability (latency/energy, reward/loss, policy/value, greedy, MRLCO, three-way).

## LaTeX / IEEEtran

- [x] **Do not use `subcaption`** with `IEEEtran` (it loads `caption` and can trigger ``Missing number, treated as zero'' near `\maketitle`). Multi-panel figures in `MARGO_paper_final.tex` use **`minipage`** panels instead.
- [x] **Avoid `[`…`]` placeholders in `\author`** (can confuse `cite` / `hyperref` PDF-string parsing). Use plain text or `\url{...}` for email.
- [x] **`\method` / `\baseline`** use `\texorpdfstring{...}{...}` so bookmarks stay valid; expectation macro is **`\Expect`** (not one-letter `\E`, which can surface as a stray `E` in TeX’s ``missing number'' diagnostics).
- [x] Removed unused **`adjustbox`** and **`balance`** packages (extra `AtBeginDocument` hooks).
- [ ] `Font shape OT1/ptm/m/scit undefined` is a harmless Times fallback when `\textsc{...}` appears near italics; ignore or wrap in `\textnormal{...}` if needed.

## Bibliography (`paper/MARGO_references.bib`)

- [x] Added: `mach2017mobile`, `dinh2018learning`, `ning2019deep`, `zhan2020deep`, `botvinick2019reinforcement`, `seo2018lte`.
- [x] Corrected entry types: `wang2020fast`, `arabnejad2014list` are now `@article` with proper `journal` fields.
- [ ] Run BibTeX + two-pass LaTeX locally; resolve any venue-specific warnings.
- [ ] Run BibTeX and resolve **undefined citations** / **unused** entries.

## Writing / claims

- [ ] Abstract: keep **uncited** unless journal requires otherwise.
- [ ] Introduction: citation density checked for MEC, DAG, DRL, meta-RL, VEC/V2V, energy.
- [ ] Related Work: subsection structure **6.x** aligned with IEEE auto-numbering; no over-claims vs closest prior art.
- [ ] Evaluation: tie qualitative sentences to **specific figures**; add numeric summaries **only** from logs/CSV extraction (not from eyeballing JPG).
- [ ] Discussion: real-world limitations (fading, multi-helper, traces) + **V2X** references if half-duplex discussion is expanded.

## Human review

- [ ] Author names, affiliations, acknowledgment, funding.
- [ ] Notation consistency: **α** vs **λ_lat**, **λ_ene** across objective figure vs equations.
- [ ] Algorithm vs code: confirm **M, K**, meta-iterations, and reward normalization match implementation.

## Done in repo (snapshot)

- [x] `figure_prompts_for_nano_banana.md` created.
- [x] `paper/MARGO_paper_final.tex` updated: `fig_*.png` naming, expanded Related Work, denser citations, `figure*` multi-panel result layouts (task~1 / task~2); **removed** standalone dataset / pipeline / results-summary figures in favor of **`fig_graph_fatness_density.png`**.
