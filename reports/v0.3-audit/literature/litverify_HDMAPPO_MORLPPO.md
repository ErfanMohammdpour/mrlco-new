# Literature verification: CLAIM A (HDMAPPO reward) and CLAIM B (MORL-PPO 2023 / IEEE TSC)

All statements below are backed by sources I actually fetched. Full-text sources are marked **[full text]**; metadata-only sources are marked **[metadata only]**. Anything I could not retrieve is flagged explicitly.

---

## CLAIM A — "HDMAPPO uses -(alpha*D + beta*E + phi*dropout) with a separate deadline-violation penalty branch."

### Verdict: **VERIFIED** — the acronym is real and the reward structure matches almost verbatim.

The earlier check failed because it only queried arXiv. HDMAPPO is **not** on arXiv (`https://export.arxiv.org/api/query?search_query=all:HDMAPPO&max_results=20` → `<opensearch:totalResults>0</opensearch:totalResults>`), but it **is** a published MDPI method. It comes from:

**Citation:** Yu Sun, Qijie He, "Computational Offloading for MEC Networks with Energy Harvesting: A Hierarchical Multi-Agent Reinforcement Learning Approach," *Electronics* **2023**, 12(6), 1304. DOI 10.3390/electronics12061304. Published 2023-03-09.
- Landing page (blocked to me): https://www.mdpi.com/2079-9292/12/6/1304
- **[full text] PDF I actually fetched (HTTP 200, 702,849 bytes):** https://mdpi-res.com/d_attachment/electronics/electronics-12-01304/article_deploy/electronics-12-01304.pdf
  (this is the redirect target of the Unpaywall/Semantic Scholar OA URL `https://res.mdpi.com/...`; `www.mdpi.com` returns HTTP 403 to this agent)
- Metadata confirmation: OpenAlex https://api.openalex.org/works/doi:10.3390/electronics12061304 (title, *Electronics*, 2023-03-09, authors Yu Sun / Qijie He); Unpaywall https://api.unpaywall.org/v2/10.3390/electronics12061304 (journal *Electronics*, year 2023, is_oa true)

### Acronym expansion (verbatim, p. 2 of PDF)

> "To address this, we design a hierarchical **double** multi-agent proximal policy optimization (**HDMAPPO**) task offloading method, where the high level uses discrete MAPPO [11] to generate server location selection for each task and the low level uses continuous MAPPO to generate the task offloading ratio."

So **HDMAPPO = "Hierarchical Double Multi-Agent Proximal Policy Optimization"** — a two-level MAPPO (high-level discrete server/discard selection, low-level continuous offload ratio), **not** a single hierarchical actor-critic.

### The objective / cost, Eq. (14), verbatim

> "min_{L_t,X_t} lim_{τ→∞} (1/τ) Σ_{t=0}^{τ} [ α Σ_{i=1}^{N} D_i^t / N + β Σ_{i=1}^{N} E_i^t / N + φ·dropout^t ]"

with the surrounding text:

> "The cost loss is composed of three parts: delay, energy consumption, and punishment for task drop, where punishment is used to describe the scale of the system's task dropout rate. **α, β, and φ represent the weights of latency, energy consumption, and task discard rate**, respectively"

Constraints listed with it: `C1: L_i^t ∈ {0,1,...,m}`, `C2: 0 ≤ x_i^t ≤ 1`, `C3: T_i^t < T_max`, `C4: E_i^tot(t) < b_i^t + E_i^har(t)`.

### The reward function, Eq. (29), verbatim — **this is the exact structure in the claim**

> "r_i^t = { −penalt        D_i^t > T_max or h_i^{a_t} = 0
>         { −(αD_i^t + βE_i^t + φdropout^t)   else"

Preceding text (verbatim): *"The reward function is designed to consider the task delay, energy consumption, and task drop rate, and can be formulated as follows:"*
And: *"The purpose of the low-level algorithm is to reduce the task latency, energy consumption, and task discard rate, which is consistent with the high-level algorithm, so the reward function of the low-level algorithm is the same as the reward function of the high-level algorithm. It is calculated by (29)."*

### Answers to the specific verification questions

| Question | Finding (verbatim/structural) |
|---|---|
| Exact reward formula | Eq. (29) above; else-branch is exactly `-(αD + βE + φ·dropout)`. ✔ matches claim |
| What happens on deadline violation | The **first branch fires**, i.e. the reward is forced to the single constant `−penalt`. Trigger condition is **`D_i^t > T_max` OR the high-level action is `h_i^{a_t} = 0`** (active discard). |
| Is the episode terminated? | **No termination is stated anywhere.** There is no "done"/"terminal"/"reset" language in the paper; the task is "discarded" and *"If the task execution time exceeds the size of the time slot or the total energy consumption exceeds the remaining battery power, the task will be discarded."* Algorithm 2 simply continues `for t = 1..T`. The penalty is **added/substituted in the reward**, not an episode kill. |
| Is reward forced to a fixed negative constant? | **Yes for the violating step:** `−penalt` replaces the whole weighted cost for that step. |
| Penalty coefficients | Symbols only: `α` (latency), `β` (energy), `φ` (discard rate), `penalt`. **No numeric values for α, β, φ or penalt appear anywhere in the paper** (I grepped the full extracted text: the only occurrences are in Eqs. 14 and 29 and the sentence on p. 8). |
| γ (discount) | **0.99** — Table 3 "Hyperparameters of HDMAPPO": `Reward discount 0.99` |
| GAE λ | λ is **symbolically defined but never given a numeric value**. Eq. (25): `Â_i(t) = Σ_{l=0}^{∞} (γλ)^l ( r_i(t+l) + γV_i(S(t+1+l)) − V_i(S(t+l)) )`, with *"λ is the parameter of GAE for bias–variance tradeoff in estimation."* Table 3 lists only learning rates (actor 0.0003, critic 0.0004), Adam, `K_epochs 10`, `Clip_rate 0.2`. |
| Environment step / horizon | Discrete time slots `T = {t1,...,tk}`; a single episode runs `for t = 1..T`, and **one offloading decision is made per UE per time slot** (per-slot sequential, not single-shot planning). Episodes loop `for episode = 1..E`. Time-slot length τ is **not** given numerically. `T_max` (max tolerated delay) = **1 s**. Task data size = [300, 500] Kbits; computational density = [800, 1200] cycles/bit; UE CPU = [1,2] GHz; MEC CPU = [2,3] GHz; harvested energy per slot = [50,150] mJ. |
| Normalization | **None described.** Grep for "normaliz" in the full text returns zero hits. |

---

## CLAIM B — "MORL-PPO (2023) and a newer IEEE TSC paper train preference-conditioned policies to produce a Pareto front because the weights are unknown a priori."

### Verdict
- **(a) "MORL-PPO (2023)": VERIFIED as a real 2023 MORL+PPO offloading paper; PARTIALLY VERIFIED as "preference-conditioned policy"** — it is a **multi-policy** method (one PPO network *per* preference), with the preference entering the scalarized reward and the critic, not the actor input.
- **(b) "newer IEEE TSC paper": VERIFIED** — there is a 2025 IEEE TSC paper that uses one preference-conditioned policy (preference fed as context) and approximates the Pareto front explicitly because the weights are unknown a priori. It uses **Discrete-SAC, not PPO**.
- **Correction to the draft's strong lead:** IEEE document **11626586 is NOT an IEEE TSC paper** — it is **IEEE Transactions on Vehicular Technology (2026)**. See §"Strong lead" below.

---

### B(a) The 2023 MORL-PPO offloading paper

**Citation:** Ning Yang, Junrui Wen, Meng Zhang, Ming Tang, "Multi-objective Deep Reinforcement Learning for Mobile Edge Computing," **2023 21st International Symposium on Modeling and Optimization in Mobile, Ad Hoc, and Wireless Networks (WiOpt)**, 2023-08-24, DOI **10.23919/wiopt58741.2023.10349870**.
- **[full text] arXiv abs:** https://arxiv.org/abs/2307.14346 (comments field: *"Received by IEEE WiOpt 2023"*)
- **[full text] HTML:** https://arxiv.org/html/2307.14346v1
- **[full text] LaTeX source (file `Offloading.tex`):** https://arxiv.org/e-print/2307.14346v1
- Venue confirmed via Crossref (https://api.crossref.org/works?query.bibliographic=Multi-objective+Deep+Reinforcement+Learning+for+Mobile+Edge+Computing+WiOpt+2023): title / container "2023 21st International Symposium on Modeling and Optimization in Mobile, Ad Hoc, and Wireless Networks (WiOpt)" / 2023-08-24 / DOI 10.23919/wiopt58741.2023.10349870.

**Unknown-preference motivation (verbatim, from the arXiv abstract):**
> "However, conventional single-objective scheduling solutions cannot be directly applied to practical systems in which the preferences of these applications (i.e., the weights of different objectives) are often unknown or challenging to specify in advance."

**Exact reward formulation (verbatim LaTeX from `Offloading.tex`):**
```
r_E(s_t,a_t) = -\hat{E}_m                                   (Eq. 20)
R_E = \sum_{t=1}^{T} r_E(s_t,a_t) = -\sum_{m \in M} \hat{E}_m   (Eq. 21)
r_T(s_t,a_t) = -(\hat{T}_m + \sum_{m' \in M_e(\tau_t)} \Delta \hat{T}_{m'}^{a_t})   (Eq. 22)
r_{\omega}(s_t,a_t) = \omega^{T} \times (\alpha_T r_T(s_t,a_t), \alpha_E r_E(s_t,a_t))   (Eq. 27, "fun:scalarized reward")
R_{\omega} = \sum_{t=1}^{T} r_{\omega}(s_t,a_t)             (Eq. 28)
```
> "where $\alpha_{\rm T}$ and $\alpha_{\rm E}$ are coefficients for adjusting delay ${r}_{\rm T}(t)$ and energy consumption ${r}_{\rm E}(t)$ to the same order of magnitude."

**The full expanded `r_T` (Eq. "Reward of delay summarize", verbatim):**
```
r_T(s_t,a_t) = -\hat{T}_m^{off} + \sum_{i=1}^{n_e^{exe}(\tau_t)} (n_e^{exe}(\tau_t)-i+1) \hat{T}_{i,e}^{dur}
  - \sum_{i=1}^{n_e^{exe}(\tau_t)} (n_e^{exe}-i+1) min( \hat{T}_{i,e}^{dur}, max( \hat{T}_m^{off} - \sum_{j=1}^{i-1} \hat{T}_{j,e}^{dur}, 0 ) )
  - \sum_{i=1}^{n_e^{exe}(\tau_t')} \frac{\eta}{f_e} (n_e^{exe}(\tau_t')-i+1)^2 (L_{i,e}^{sort}(\tau_t') - L_{i-1,e}^{sort}(\tau_t'))
```

- **How latency and energy are combined: weighted sum with a preference vector** (Pareto via sweeping ω, not a constraint).
- **Penalty terms: none.** There is no dropout/discard/deadline penalty and no constraint-violation branch anywhere in the reward.
- **γ = 0.9**; **GAE λ = 0.95**; **clip ε = 0.2** (Table I: "Discount factor γ | 0.9", "GAE discount factor λ | 0.95", "Clip parameter ε | 0.2"). GAE (Eq. 24): `Â_i(t) = Σ_{t'=t}^{T-1} γλ( α_i r_i(s_{t'},a_{t'}) + γV_{i,θ}(s_{t'+1}) − V_{i,θ}(s_{t'}) )`.
- **α_T, α_E numeric values: not given** (only described qualitatively as order-of-magnitude scalers). **No normalization method** is described (grep "normaliz" in the full text: 0 hits).
- **Environment step:** per-task sequential. Continuously-timed system with discrete decision steps, one offloading decision per step, one episode = T steps, FIFO task queue, Poisson arrivals.
- **Preference-conditioned? NO — multi-policy.** Verbatim: *"In the training phase, the MORL algorithm trains a parametric network for each preference."* Also: *"For each preference ω in set Ω, we train a policy with PPO method to maximize reward R_ω and approximate Pareto front PF(Π)"*, and *"We set the preference set as Ω with an equal interval 0.02 and obtain 50 preferences to fit the Pareto front."* The actor is `π_θ(a_t|s_t)` (ω is **not** an input); only the critic is preference-weighted: *"the output is estimated value ω^T[V_{T,θ}(s_t), V_{E,θ}(s_t)] for preference ω."*

---

### B(b) The newer IEEE TSC paper

**Citation:** Ning Yang, Junrui Wen, Meng Zhang, Ming Tang, "Generalizable Pareto-Optimal Offloading With Reinforcement Learning in Mobile Edge Computing," **IEEE Transactions on Services Computing**, 2025 (issue date 2025-11), DOI **10.1109/TSC.2025.3604371**.
- **[full text] arXiv (accepted version):** https://arxiv.org/abs/2509.10474 ; https://ar5iv.labs.arxiv.org/html/2509.10474
- **[full text] LaTeX source — file literally named `IEEE_TSC_R1_arxiv.tex`:** https://arxiv.org/e-print/2509.10474v1
- Venue confirmed via Crossref (https://api.crossref.org/works?query.bibliographic=Generalizable+Pareto-Optimal+Offloading...): container-title "IEEE Transactions on Services Computing", year 2025-11, DOI 10.1109/tsc.2025.3604371; Unpaywall confirms journal *IEEE Transactions on Services Computing*, 2025 (is_oa false, hence using the arXiv version).
- IEEE landing page (abstract, blocked to this agent): https://ieeexplore.ieee.org/abstract/document/11144916

**Unknown-preference + Pareto motivation (verbatim, abstract):**
> "However, conventional single-objective scheduling solutions cannot be directly applied to practical systems in which the preferences (i.e., the weights of different objectives) are often unknown or challenging to specify in advance. … To address the challenge of unknown preferences and the potentially diverse MEC systems, we propose a generalizable multi-objective (deep) reinforcement learning (GMORL)-based tasks offloading framework, which employs the Discrete Soft Actor-Critic (Discrete-SAC) method. Our method uses a single policy model to efficiently schedule tasks based on varying preferences…"

And in the intro (verbatim):
> "It is worth noting that the direct application of single-objective DRL through scalarization, which involves taking a weighted sum, is not a valid approach due to the following issues: 1. Impossibility: Weights may be unknown when designing or learning an offloading scheme. 2. Infeasibility: Weights may be diverse … 3. Undesirability: …"

**Exact reward formulation (verbatim LaTeX from `IEEE_TSC_R1_arxiv.tex`) — same two-branch structure as B(a):**
```
r_E(s_t,a_t) = -\hat{E}_m
R_E = \sum_{t=1}^{T} r_E(s_t,a_t) = -\sum_{m \in M} \hat{E}_m
r_T(s_t,a_t) = -(\hat{T}_m + \sum_{m' \in M_e(\tau_t)} \Delta \hat{T}_{m'}^{a_t})
r_{\omega}(s_t,a_t) = \omega^{T} \times (\alpha_T r_T(s_t,a_t), \alpha_E r_E(s_t,a_t))     (Eq. 29)
R_{\omega} = \sum_{t=1}^{T} r_{\omega}(s_t,a_t)
```
> "To achieve the GMORL algorithm, we compute a scalarized reward given preference $\boldsymbol{\omega}$ … where $\alpha_{\rm T}$ and $\alpha_{\rm E}$ are coefficients for adjusting delay ${r}_{\rm T}(t)$ and energy consumption ${r}_{\rm E}(t)$ to the same order of magnitude."

The expanded `r_T` (Eq. "Reward of delay summarize") is byte-identical in form to B(a) above.

- **Latency/energy combination:** weighted sum with preference ω, single policy conditioned on ω → Pareto front.
- **Penalty terms: none** (no dropout / deadline / constraint penalty). Constraints appear only in the optimization problem as hard constraints `x_{m,e} ∈ {0,1}`, `Σ_e x_{m,e} = 1`, not in the reward.
- **Note the exact preference-conditioning statement** (commented line in the LaTeX, i.e. an author remark retained in the source): *"To address the unknown preferences, a scalarized reward r_ω that depends on ω is designed, where ω is used as a context input rather than a fixed parameter, allowing the policy to dynamically receive an arbitrary preference vector during inference."* The state includes the context: optimizer objective is `π* = arg max_π Σ_t E[ γ^t ( r_ω(s_t,a_t) + α_H H(π(·|s_t)) ) ]`.
- **γ = 0.95** (Table III: "Discount factor γ | 0.95"); **SAC temperature α_H = 0.05**; learning rates λ_π = λ_Q = 1e-6, λ_{α_H} = 0. **No GAE λ** — it is off-policy Discrete-SAC, GAE does not apply. Hyperparameters: N_ep = 4000 epochs, N_g = 64 environments/epoch, N_up = 10, replay memory 1e5, batch 4096.
- **α_T, α_E numeric values: not given**; **normalization:** only softmax normalization of the policy over masked actions (`π_φ = softmax(mask(π'_φ))`) and the SAC partition function `Z^{π_old}`; **no reward/objective normalization method and no normalization reference** is given (grep "normaliz" returns only those policy-distribution hits).
- **Environment step:** `"Consider one episode consisting of T steps, and each step is denoted by t ∈ {1,...,T}, each with a duration of Δt seconds."` Table: **`The number of steps for one episode T | 100`**, **`Step duration Δt | 1 s`**. Also `"In each step, the system will offload the first task in the queue to one of the servers. Then the task is removed from the queue."` → **per-slot sequential (one task per step), one episode = 100 steps**.
- It is **hierarchical in neither policy nor action space** — single-level; it cites and baselines against the 2023 multi-policy MORL-PPO (reference `[yang2023multi]`).

---

### The draft's "strong lead" — correction

**IEEE Xplore document 11626586** is:

**Citation:** Yanran Liu, Zhengli Liu, Weiyu Yuan, Cheng Zeng, Peng He, Tao Peng, Jian Wang, Bing Li, "A Hierarchical Preference-Guided Multi-Objective Reinforcement Learning Method for Task Offloading in Vehicular Edge Computing," **IEEE Transactions on Vehicular Technology**, 2026, DOI **10.1109/TVT.2026.3717633**.
- **VENUE IS TVT, NOT TSC.** Confirmed independently by three metadata APIs:
  - Crossref: https://api.crossref.org/works?query.bibliographic=A+Hierarchical+Preference-Guided+Multi-Objective+Reinforcement+Learning+Method+for+Task+Offloading+in+Vehicular+Edge+Computing → container-title `IEEE Transactions on Vehicular Technology`, issued 2026, DOI `10.1109/tvt.2026.3717633`
  - Semantic Scholar: https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/TVT.2026.3717633 → venue "IEEE Transactions on Vehicular Technology", year 2026, 8 authors as listed; **abstract field is empty**
  - OpenAlex: https://api.openalex.org/works/doi:10.1109/tvt.2026.3717633 → venue "IEEE Transactions on Vehicular Technology", publication_date 2026-01-01, `is_oa: false, oa_status: "closed", any_repository_has_fulltext: false`, **abstract_inverted_index: none**
- **[metadata only] — abstract and full text NOT VERIFIED.** `https://ieeexplore.ieee.org/document/11626586` returns HTTP 202 with an empty body to this agent (both `web_fetch` and `curl` with a browser user-agent); Unpaywall reports `is_oa: false` with no OA locations; no preprint exists (arXiv API search for `all:"preference guided" AND all:"task offloading"` → 0 results); there is no abstract in Crossref/OpenAlex/Semantic Scholar. I therefore **cannot** state its reward formulation, whether it is preference-conditioned, or whether it produces a Pareto front — only its title, venue, year, and author list are verified.

**Other near-miss papers found and checked (for triage):**
- "Graph-Augmented MAPPO for Dynamic Task Offloading in Edge Networks," *IEEE Communications Letters*, 2026, DOI 10.1109/lcomm.2026.3674662 — **[metadata only]** via Crossref; MAPPO (not hierarchical-preference MORL).
- "Cooperative UAV Resource Allocation and Task Offloading in Hierarchical Aerial Computing Systems: A MAPPO-Based Approach," *IEEE Internet of Things Journal*, 2023, DOI 10.1109/JIOT.2023.3240173 — **[abstract only]** via Semantic Scholar API (https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/JIOT.2023.3240173). Real **hierarchical MAPPO offloading** paper: objective is *"to maximize the amount of computed tasks while satisfying tasks' heterogeneous Quality-of-Service (QoS) requirements"*; *"state normalization and action mask are also adopted to improve training efficiency."* It does **not** use the `-(αD+βE+φ·dropout)` reward and is not preference-conditioned.
- "Task Offloading Algorithm Based on Preference Weight Adaptive Multi-objective Reinforcement Learning for Cloud-Assisted Mobile Edge Computing in Internet of Vehicles," Springer, DOI 10.1007/s42154-024-00350-8 — **[metadata only]**, surfaced by search, not fetched.
- `https://www.emergentmind.com/topics/hierarchical-multi-agent-proximal-policy-optimization-h-mappo` **[fetched]** — a survey/topic page of H-MAPPO works (wind-farm control, shepherding, emergency-corridor traffic, privacy-aware edge inference, electric-bus charging). **No offloading paper with the HDMAPPO acronym appears there**, and it states the label is *"best understood as a family of structured MAPPO instantiations rather than a single canonical algorithm."*

---

## Bottom line for the draft

1. **CLAIM A: VERIFIED (and the draft understates it).** HDMAPPO = "Hierarchical Double Multi-Agent PPO," Sun & He, *Electronics* 12(6):1304, 2023, DOI 10.3390/electronics12061304. Its reward is exactly `−(αD + βE + φ·dropout)` with a **separate constant-penalty branch** for `D > T_max` **or** active discard `h=0`, where the penalty is a **fixed constant `−penalt` substituted for the reward on that step — the episode is not terminated**. Caveats to fix in the draft: (i) the violation condition includes *active discard*, not only deadline violation; (ii) **α, β, φ, penalt, and GAE λ have no numeric values in the paper**; (iii) γ = 0.99, clip = 0.2, K_epochs = 10; (iv) no normalization is described; (v) `T_max` = 1 s, decisions are **per-slot sequential** (one per UE per time slot), not a single-shot plan.
2. **CLAIM B(a): PARTIALLY VERIFIED.** The 2023 MORL-PPO paper exists (Yang et al., WiOpt 2023, DOI 10.23919/wiopt58741.2023.10349870, arXiv:2307.14346) and matches the "weights unknown a priori → Pareto front" rationale, and it *is* preference-parameterized — but it trains **one PPO network per preference** (50 preferences at 0.02 spacing), i.e. **not** a single preference-conditioned policy; α_T/α_E are unreported and there are **no penalty terms**.
3. **CLAIM B(b): VERIFIED, but attribute it correctly.** The "newer IEEE TSC paper" is **"Generalizable Pareto-Optimal Offloading With Reinforcement Learning in Mobile Edge Computing," IEEE TSC 2025, DOI 10.1109/TSC.2025.3604371** (arXiv:2509.10474). It is preference-conditioned (ω as context), explicitly justified by unknown a priori weights, and produces a Pareto front — but with **Discrete-SAC, not PPO**, γ = 0.95, T = 100 steps of Δt = 1 s, and **no penalty terms**.
4. **Do not cite 11626586 as IEEE TSC.** That paper is **IEEE Transactions on Vehicular Technology 2026**, and its content is unverifiable from open sources (closed access, no abstract available anywhere I could fetch).
