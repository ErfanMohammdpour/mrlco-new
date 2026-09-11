# Figure generation prompts for Nano Banana (MARGO paper)

Use these prompts **verbatim in English** in Nano Banana. Export **PNG** (or PDF vector if available) with **white background**, **IEEE Transactions–style** flat technical diagrams (no 3D, no clipart, no decorative backgrounds, no people). **Color palette (consistent across all figures):** UE/local **dark blue**; MEC server **teal/cyan**; V2V helper **orange**; graph/DAG nodes **navy**; neural blocks **light gray / blue**; meta-training blocks **purple**; latency–energy trade-off **green/orange** accents. Typography: clean sans-serif, legible at two-column width (~3.3 in). Arrows: simple vector arrows. **Do not** show futuristic cities, cartoon vehicles, glossy AI art, or crowded slide titles.

**Repository note:** Temporary copies of deck exports may exist as `figures/fig_*.png` for LaTeX build continuity; **replace** those with diagrams generated from the prompts below. The paper uses **`fig_graph_fatness_density.png`** for benchmark DAG diversity in Sec.~V (no separate Nano Banana prompts for dataset protocol, evaluation pipeline, or conceptual results summary).

---

## Figure 1 — V2V-assisted MEC System Model

**Title:** V2V-assisted MEC system considered by MARGO  
**Purpose:** Communicate execution options, duplex assumptions, and where the policy sits.  
**Must show:** Ego UE with **local CPU**; **MEC/RSU** with edge server; **V2V helper** vehicle; **MARGO policy** block inside UE; three execution paths: **Local**, **MEC** (uplink + downlink drawn as **two separate full-duplex** cellular links), **V2V** (single shared radio link labeled **half-duplex**); outputs **actions 0/1/2** = Local/MEC/V2V; small **DAG icon** near UE (optional).  
**Nano Banana prompt:**

> IEEE Transactions–style flat technical diagram on white background. Left: User Equipment (ego vehicle) in dark blue containing a small “local processor” block and a “MARGO policy” controller. Center-top: MEC server at roadside in teal/cyan. Right: V2V helper vehicle in orange. Show three execution choices as clearly separated paths: (1) local execution loop inside UE; (2) offload to MEC via two labeled channels “UL” and “DL” with a note “full-duplex cellular”; (3) offload to V2V helper via one bidirectional link labeled “half-duplex V2V”. Arrows for data/control flow only. Output arrows from policy labeled “a∈{0,1,2}” mapping to “Local / MEC / V2V”. Minimal text, vector arrows, no photos, no 3D, no decorative background.

**Required labels:** UE, Local CPU, MEC/RSU, UL, DL, full-duplex (cellular), V2V helper, half-duplex (V2V), MARGO policy, Local (0), MEC (1), V2V (2), optional small DAG.  
**Style:** Horizontal layout, single-column friendly height.  
**Suggested filename:** `fig_system_model.png`  
**LaTeX caption (suggested):** “V2V-assisted MEC system considered by MARGO. The UE can execute a DAG task locally, offload it to the MEC server through full-duplex cellular links, or offload it to a nearby V2V helper through a half-duplex V2V channel. MARGO selects one of the three execution modes for each task.”  
**Must NOT show:** Photorealistic cars, cityscapes, vendor logos, cluttered equations.

---

## Figure 2 — Overall MARGO Architecture

**Title:** End-to-end MARGO neural pipeline  
**Purpose:** Match paper’s tensor shapes and training loops.  
**Must show:** **DAG** → **task feature matrix** annotated **\[B, 20, 20\]** (or equivalent) → **embedding 20→128** → **Graph2Seq** with “DAG-edge message passing” → encoder output **\[B, 20, 256\]** → **triple readout** (attention + mean + max) → fused **\[B, 256\]** → **2-layer LSTM + Luong attention** → **3-way logits** (Local/MEC/V2V) → side boxes **PPO inner loop** and **Reptile outer loop** with short arrows to “policy weights”.  
**Nano Banana prompt:**

> Wide horizontal IEEE-style architecture diagram, white background. Flow left to right: “DAG tasks” → tensor “[B,20,20]” → linear “20→128” → block “Graph2Seq (DAG-edge MP)” → “[B,20,256]” → “Triple readout (attn/mean/max)” → “[B,256] graph state” → “LSTM decoder + Luong attention” → “logits: 3 classes”. Use light gray/blue blocks for neural modules, purple side panels for “PPO inner update (GAE)” and “Reptile outer update”. Minimal text inside blocks; include only key tensor shapes. Flat vector style, consistent palette (dark blue/teal/orange accents), no decorative icons.

**Required labels:** Graph2Seq, DAG-edge MP, Triple readout, LSTM, Luong attention, Local/MEC/V2V, PPO, Reptile, shapes as above.  
**Suggested filename:** `fig_margo_architecture.png`  
**LaTeX caption (suggested):** “Overall architecture of MARGO. DAG task features are encoded by a Graph2Seq encoder using DAG-edge message passing. A triple readout forms a graph-level state, which initializes an attention-based LSTM decoder for Local/MEC/V2V decisions. Training combines PPO inner-loop adaptation and Reptile outer-loop meta-learning.”  
**Must NOT show:** Training hyperparameter tables, dataset screenshots, fake accuracy numbers.

---

## Figure 3 — Graph2Seq Encoder (DAG-edge message passing)

**Title:** Dependency-aware encoding  
**Purpose:** Explain predecessor/successor neighborhoods and two-layer mean aggregation.  
**Must show:** Small **DAG** (4–6 nodes); highlight node **i** with **pred(i)** and **succ(i)**; **N(i)=pred∪succ**; **two layers** of aggregation; final **256-D** embedding per node; equations as small callouts: **h_neigh = mean(neighbors)**; **h_i = σ(W_self h_i + W_neigh h_neigh)** (or concat+linear if you prefer, but label clearly).  
**Nano Banana prompt:**

> Technical diagram: small directed acyclic graph in navy nodes with directed edges. Select a node i and color its predecessors and successors; show set N(i)=pred(i)∪succ(i). Two stacked layers labeled “layer 1” and “layer 2”: each layer computes neighbor mean pooling along DAG edges only (no full graph clique), then combines with self transformation. Annotate “256-D node embedding”. Include two compact equation callouts: h_neigh = mean(neighbors); h_i = σ(W_self h_i + W_neigh h_neigh). White background, flat vector, minimal text, IEEE style.

**Suggested filename:** `fig_graph2seq_encoder.png`  
**LaTeX caption (suggested):** “Graph2Seq encoder used in MARGO. Each task aggregates information from predecessor and successor neighborhoods through two layers of mean aggregation, producing dependency-aware node embeddings.”  
**Must NOT show:** Full GraphSAGE on arbitrary graphs; social-network illustrations.

---

## Figure 4 — Triple Readout Mechanism

**Title:** Attention + mean + max → graph embedding  
**Purpose:** Explain complementary pooling roles.  
**Must show:** Input **\[B, 20, 256\]** fan-out to three branches: **attention pooling** (“learned task importance”), **mean pooling** (“global workload average”), **max pooling** (“bottleneck / extreme features”); **concat → \[B, 768\]**; **dense → \[B, 256\]**; “graph embedding”.  
**Nano Banana prompt:**

> Block diagram starting from a tensor strip labeled “[B,20,256] node embeddings”. Three parallel branches: (1) attention pooling with softmax weights α over tasks; (2) mean pooling; (3) max pooling (element-wise max over tasks). Concatenate to “[B,768]” then a single linear/projection block to “[B,256] graph embedding”. Short role labels only. White background, flat vector, consistent colors, IEEE Transactions look.

**Suggested filename:** `fig_triple_readout.png`  
**LaTeX caption (suggested):** “Triple readout mechanism. Attention pooling captures task importance, mean pooling summarizes global workload characteristics, and max pooling highlights bottleneck features. The concatenated representation is projected into a graph-level embedding for the decoder.”  
**Must NOT show:** Heatmaps of fake attention weights.

---

## Figure 5 — Attention-based Decoder & Ternary Actions

**Title:** Sequential action generation  
**Purpose:** Link graph state, autoregression, attention, and 3-class outputs.  
**Must show:** **Graph embedding** initializes decoder; steps **t=1…N**; **Luong attention** over encoder outputs; **previous action** fed back; per-step **softmax over {Local,MEC,V2V}**.  
**Nano Banana prompt:**

> Decoder diagram: “graph embedding” initializes a 2-layer LSTM block. For timesteps t=1..N, show autoregressive input “prev action embedding”, LSTM hidden state, and Luong attention attending to the sequence of node embeddings from the encoder. At each step, output head maps to three logits labeled Local(0), MEC(1), V2V(2). White background, flat vector, minimal equations, IEEE style, consistent palette.

**Suggested filename:** `fig_decoder_action_generation.png`  
**LaTeX caption (suggested):** “Attention-based action decoder. At each scheduling step, the decoder attends to encoded task embeddings and outputs a ternary offloading action for Local, MEC, or V2V execution.”  
**Must NOT show:** Natural-language token generation; machine translation toy examples.

---

## Figure 6 — Meta-RL Training (PPO + Reptile)

**Title:** Inner/outer loop workflow  
**Purpose:** Clarify meta-parameters, batch of tasks, adaptation, and meta-test.  
**Must show:** **θ** meta parameters → sample **M=10** tasks → for each: **copy θ**, **rollouts**, **PPO update with K=1** → **θ′**; aggregate; **Reptile outer update** → updated **θ**; separate small panel **meta-test: fast adaptation** on unseen task.  
**Nano Banana prompt:**

> Flowchart on white background. Start with “meta parameters θ”. Branch to “sample M tasks (M=10)”. For each task i: “θ_i ← θ”, “collect trajectories”, “PPO inner update (K=1)”, produce “θ_i’”. Aggregate across tasks (mean icon) then “Reptile outer update: θ ← θ − η·(θ − mean(θ_i’))” as a simple labeled box (equation optional but small). Add a small inset “meta-test: few-shot / fast adaptation on held-out distribution”. Purple accents for meta-training blocks, minimal text, IEEE style, no screenshots.

**Suggested filename:** `fig_meta_reptile_training.png`  
**LaTeX caption (suggested):** “Meta-training workflow of MARGO. PPO performs task-specific inner-loop adaptation, while Reptile updates the shared initialization toward adapted task policies, enabling fast adaptation to unseen task distributions.”  
**Must NOT show:** Generic “AutoML” funnel graphics.

---

## Figure 7 — Joint Latency–Energy Objective (conceptual)

**Title:** Weighted objective and trade-off curve  
**Purpose:** Visualize **min J(π) = E[α C_latency + (1−α) C_energy]** as a **conceptual** trade-off (not experimental data).  
**Must show:** Smooth **Pareto-style** curve in the latency–energy plane; mark three **conceptual** operating points for **α=1** (latency-oriented), **α=0** (energy-oriented), **α=0.5** (balanced); axes **Latency / makespan** vs **Energy**.  
**Nano Banana prompt:**

> Conceptual 2D plot, white background, IEEE style. Axes: x-axis “Energy consumption”, y-axis “Latency / makespan”. Draw a clean convex trade-off curve (schematic, not data-derived). Mark three annotated points along the curve for α=1, α=0.5, α=0 with short captions “latency-oriented”, “balanced”, “energy-oriented”. Include a small equation banner: min J(π)=E[α·C_latency+(1−α)·C_energy]. No numeric tick labels claiming measured results; ticks generic. Flat vector, minimal grid.

**Suggested filename:** `fig_joint_objective.png`  
**LaTeX caption (suggested):** “Conceptual latency–energy trade-off optimized by MARGO. The weighting factor α controls the preference between makespan minimization and energy conservation.”  
**Must NOT show:** Fake experiment markers, error bars, p-values.

---

## Figure 8 (paper) — DAG benchmark diversity (graph fatness & density)

**Note:** The manuscript **does not** use separate Nano Banana figures for dataset protocol, evaluation pipeline, or conceptual results summary. Instead it includes **`fig_graph_fatness_density.png`** (author-provided asset under `figures/`) to illustrate structural diversity of generated DAGs. The **19×100** layout, **20 tasks** per graph, **Graphviz .gv** generation, and **meta-train (dist. 1–15) / meta-test (dist. 16–19)** split are described **in prose** in Sec.~V of the LaTeX source—no extra infographic required unless you later choose to redraw this panel.

**Suggested filename:** `fig_graph_fatness_density.png`  
**LaTeX caption (in paper):** See `MARGO_paper_final.tex` (label `fig:graph_fatness_density`).

---

## Experimental curve figures (from repository plots; do not Nano-Banana)

Regenerate **only** if you have underlying CSV/logs. Otherwise use the existing JPG curves under `results/` and `results/mrlco-compare/{1,2}/` and compose multi-panel LaTeX figures as in `MARGO_paper_final.tex`.
