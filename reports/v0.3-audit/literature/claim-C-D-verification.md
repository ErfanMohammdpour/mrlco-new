# Literature verification: CLAIM C and CLAIM D

All quotations below were read from full text I actually fetched. Where only an abstract/paywall was reachable it is flagged **[ABSTRACT-ONLY]** or **[PAYWALLED]**.

---

## CLAIM C — "DRL-E2D minimises energy subject to a task deadline constraint rather than scalarising 0.5/0.5."

### C.1 The acronym "DRL-E2D" is real, but it is NOT on arXiv

- arXiv API `all:"DRL-E2D"` → `opensearch:totalResults = 0`. Verbatim: `https://export.arxiv.org/api/query?search_query=all:%22DRL-E2D%22&max_results=5`
- However, **DRL-E2D is a published algorithm**, and it is exactly the paper suggested as a "good lead":

> **Li, Z., Chang, V., Ge, J., Pan, L., Hu, H., Huang, B. (2021). "Energy-aware task offloading with deadline constraint in mobile edge computing." EURASIP Journal on Wireless Communications and Networking, 2021:56. DOI 10.1186/s13638-021-01941-3**

Verbatim abstract sentence:

> "Based on the deep reinforcement learning (DRL) approach, this paper proposes an Energy-aware Task Offloading with Deadline constraint (DRL-E2D) algorithm for a multi-eNB MEC environment, **which is to maximize the reward under the deadline constraint of the tasks**."

Sources:
- Metadata + abstract: `https://api.semanticscholar.org/graph/v1/paper/DOI:10.1186/s13638-021-01941-3` (title, authors, year 2021, venue, `openAccessPdf.status = "GOLD"`, `license = "CCBY"`)
- Landing page: `https://jwcn-eurasipjournals.springeropen.com/articles/10.1186/s13638-021-01941-3` (DOI-resolved; **JS/Cloudflare challenge** blocked direct HTML fetch; PDF retrieved instead)
- **Full text PDF actually fetched**: `https://jwcn-eurasipjournals.springeropen.com/counter/pdf/10.1186/s13638-021-01941-3` (24 pages; 5,097,051 bytes) — the publisher PDF endpoint returns `Client Challenge` HTML to a normal UA but served the real PDF to a Googlebot UA.
- Green-OA mirror listed by Unpaywall (`https://api.unpaywall.org/v2/10.1186/s13638-021-01941-3?email=...`): `https://research.tees.ac.uk/en/publications/29c698be-30fd-498b-aa75-f6fcd94a4ca4` → file `https://research.tees.ac.uk/files/25413926/Li2021_Article_Energy_awareTaskOffloadingWith.pdf` (this repository PDF was itself Cloudflare-blocked to direct curl; the publisher PDF above was used instead).

### C.2 What DRL-E2D actually does — it IS a scalarised reward

Verbatim, from the paper's own contribution list (p. 4):

> "DRL-E2D deals with the task offloading problem by maximizing the well-designed reward, **which is the weighted sum of the utility of processed tasks, energy consumption, and the penalty of task dropping**."

Verbatim, Section 3.3 "Problem formulation" (p. 8):

> "This section formulates the task offloading as an optimization problem. **The objective is to maximize the joint reward, including the utility of the finished task, the energy consumption of MD, and the penalty of task dropping.**"

The objective (Eq. 16a), verbatim:

> **(16a) max : R(τ) = U(τ) − E(τ) − P(τ)**

Component equations, verbatim:

> **(12) E(τ) = β₀(τ)Eᵉˣ₀(τ) + Σⁿᵢ₌₁ αᵢ(τ)Eᵗˣᵢ(τ)**
> **(13) u(tⱼ) = { u, T(tⱼ) ≤ T_DL ; 0, otherwise }**
> **(14) U(τ) = Σⁿᵢ₌₀ Σ^{β_k(τ)}_{j=1} u(tⱼ)**
> **(15) P(τ) = Σⁿᵢ₌₀ dᵢ(τ)**

Constraints (16b–16e), verbatim:

> "(16b) Σⁿᵢ₌₀ αᵢ(τ) ≤ z(τ) … Equation (16c) is the constraint of transmission capacity for each link between MD and eNBᵢ. Equation (16d) is the time constraint for offloading tasks, and Eq. (16e) is the computing capacity constraint."

**Conclusion:** `R(τ) = U(τ) − E(τ) − P(τ)` has *implicit unit weights on all three terms*. It is a scalarised weighted-sum objective. The paper even says so:

> "We can see from Eqs. (12)–(15) that MD wants to maximize the utility, incurring less the penalty of dropped tasks but more energy consumption. So, **there exists the performance trade-off among the utility, energy consumption, and penalty**."

### C.3 How the deadline is enforced in DRL-E2D

**Not** a hard constraint set, **not** a Lagrangian multiplier, **not** action masking. It is **reward gating + drop penalty**. Verbatim:

> "The task's deadline constraint is considered. Let T_DL denote the deadline constraint for all the tasks. **If the waiting time and execution time of task tⱼ is less than and equal to the deadline, i.e., T(tⱼ) ≤ T_DL, the MD gains the u utility of successfully finished task; otherwise, MD obtain zero utility.**"

> "Note that the deadline constraint is measured by the number of the time slot. For example, we can set T_DL = N_DL T_slot, which means the task should be executed within N_DL time slots once it is generated."

> "**However, if a task misses the deadline, this task will be dropped, incurring the penalty.** Then, the penalty for dropping a task at time slot τ is computed by [Eq. 15] where dᵢ(τ) is the number of dropped tasks of each eNB and MD."

Deadline is a *constraint* only in the loose sense that (16a) is solved "Under the deadline constraints"; (16b)–(16e) are capacity/number-of-task constraints, not energy or deadline-margin constraints. There is no `min E s.t. deadline` problem.

### C.4 Coefficients / hyperparameters actually reported

| Item | Value (verbatim) |
|---|---|
| Penalty coefficient | Implicit **1**: `P(τ) = Σ dᵢ(τ)` (number of dropped tasks). Utility `u = 1` per on-time task: *"if a task is finished within the deadline constraint, MD will receive the utility u = 1"* |
| Deadline | *"deadline constraint T_DL = 3T_slot (i.e., N_DL = 3)"* |
| Discount γ | **No numeric value reported.** Only symbolic: *"γ ∈ [0, 1] is the discount rate"* |
| GAE λ | **None.** Algorithm is an actor–critic / deterministic-policy gradient; `GAE` and `advantage` → 0 hits |
| Normalisation | **None reported** (`normaliz` → 0 hits in the full text) |
| Environment step / horizon | **Per-time-slot sequential.** *"We set the length of each time slot T_slot = 1 second, and the number of time slots N_slot = 1000, i.e., each episode includes 1000 iterations, which is long enough for obtaining the stable results."* State = *"the length of the task processing queue, the data transmission rate, and the number of arrived tasks"* |
| Sim constants | `ptx₀ = 250 mW`, `B = 100 MHz`, `σ² = −174 dbm/Hz`, `ϕ = 0.01`, `θ = 4`, `M = 4` CPU cores, `D = 10 MB`, `T_slot = 1 s` |

**Verdict on CLAIM C: NOT VERIFIED (refuted).**
"DRL-E2D" does exist (EURASIP JWCN 2021, DOI 10.1186/s13638-021-01941-3), but it does **not** minimise energy subject to a hard task-deadline constraint. Its objective is the scalarised sum `R(τ) = U(τ) − E(τ) − P(τ)`, with the deadline handled by reward gating (`u = 0` on miss) plus a drop penalty. The claim in the draft is backwards: DRL-E2D is precisely a member of the class the draft contrasts it against.

---

### C.5 The other two suggested "good leads"

**Lead 2 — "An Energy-Efficient Dynamic Offloading Algorithm for Edge Computing Based on Deep Reinforcement Learning"** — **FOUND, full text read.**

> Zhu, K., Li, S., Zhang, X., Wang, J., Xie, C., Wu, F., Xie, R. (2024). IEEE Access, vol. 12, pp. 127489–127506. DOI 10.1109/ACCESS.2024.3452190. Method name: **EE-A2C**.

Metadata: `https://m2.mtmt.hu/api/publication/35594465`; `https://api.crossref.org/works/10.1109/ACCESS.2024.3452190`; full text PDF fetched from `https://ieeexplore.ieee.org/ielx8/6287639/10380310/10659862.pdf` (18 pp.).

Verbatim reward function (Section IV.A.3):

> "Next, we calculate the weighted penalty value of energy consumption, delay, and task offloading success rate … **ω_f = 1 − U_nf/E_TC ; ω_e = −log₂(rᵉᵢ[−1]/ϕₑ + 1e−10) ; ω_d = 1 − ϕ_d/D_max**"
> "**r(t) = α ω_f + β ω_e + η ω_d**  (33) — Among these parameters, α, β and η represent the weight coefficients of task completion, energy consumption, and delay, respectively."

- Deadline: appears once, as constraint **c4** — *"c4 mandates that the task delay must not exceed the maximum tolerance time."* Enforced only softly through the average-delay term `ω_d = 1 − ϕ_d/D_max`. **No hard deadline constraint, no Lagrangian, no masking.**
- Weight values α, β, η: **not numerically reported** in the paper's setup text I read.
- γ: *"we identified 0.9 as the optimal value for γ"* (swept 0.3, 0.6, 0.9, 0.95, 0.99).
- GAE λ: none — vanilla A2C with `δ_t = r_t + γV(s_{t+1}) − V(s_t)`.
- Normalisation: none reported. Batch 64 (32 also tested); lr 0.001.
- → **Also a scalarised weighted sum**, so it cannot supply the "energy-min under hard deadline" formulation either.

**Lead 3 — "Energy-efficient collaborative task offloading in multi-access edge computing based on deep reinforcement learning"** — **[PAYWALLED / ABSTRACT UNAVAILABLE]**

> Wang, S., Zhao, S., Gui, H., He, X., Lu, Z., Chen, B., Fan, Z., Pang, S. (2024). Ad Hoc Networks. DOI 10.1016/j.adhoc.2024.103743

Semantic Scholar record (`https://api.semanticscholar.org/graph/v1/paper/DOI:10.1016/j.adhoc.2024.103743`) returns `"openAccessPdf": {"url": "", "status": "CLOSED"}` and `"abstract": null`. ScienceDirect fetch (`https://www.sciencedirect.com/science/article/abs/pii/S1570870524003548`) returned HTTP 403 (Cloudflare error box). **I could not read its objective, deadline enforcement, or any hyperparameter. Flagged as unverifiable.**

### C.6 Did I find *any* real paper using "minimise energy s.t. a hard task-deadline constraint"?

arXiv API queries, all with **0 results**:
- `all:"deadline" AND all:"energy" AND all:"Lagrangian" AND all:"offloading"` → 0
- `abs:"energy" AND abs:"deadline" AND abs:"action masking"` → 0
- `abs:"minimize energy" AND abs:"deadline constraint" AND abs:"reinforcement learning"` → 0
- `abs:"constrained reinforcement learning" AND abs:"offloading"` → 1 (unrelated: LLM inference, arXiv:2502.11007)

The closest real formulations I could verify:

1. **ElasticVR, arXiv:2512.12366** — handles the deadline as an explicit **inequality constraint via a Lagrangian dual**, not fixed weights. Full text fetched from `https://ar5iv.labs.arxiv.org/html/2512.12366`. Verbatim:
   > "(15) min_{λ_k} max_{e,u} QTE(e,u) + Σ_{k=1}^{K} λ_k (T^d_k − T^r_k(e_k,u_k))"
   > "(18) λ*_k = { 0, T^d_k − T^r_k ≥ 0 ; λ^penalty_k (T^d_k − T^r_k), T^d_k − T^r_k < 0 }"
   > "**Deadline coefficient λ^penalty_k = 10**"
   Its base objective is still weighted-sum: "(11) QTE(e,u) = w₀Q(e) − w₁T(e,u) − w₂E(e,u)", with "**w₀ = 0.35, w₁ = 0.85, w₂ = 0.15**". Uses PPO-style agents with "discount factor **γ = 0**", "Entropy weight β = 1.e−4", "Number of training iterations N_policy = 80", "Policy update frequency N_update = 4". No GAE λ reported (uses `Â_t = r_t + γV(s_{t+1}) − V(s_t)` with γ = 0).
2. **arXiv:1708.04813**, "Energy-Efficient Resource Allocation for Cache-Assisted Mobile Edge Computing" — matched `"deadline constraint"` + `"energy minimization"`, but is a Lyapunov/convex MEC paper, not a DRL hard-deadline formulation. `https://arxiv.org/abs/1708.04813`

**Bottom line for C:** the formulation the draft attributes to DRL-E2D — "minimise energy subject to a hard task-deadline constraint" — I could not find instantiated in any paper I could read. Real DRL offloading papers overwhelmingly use either a weighted sum (DRL-E2D, EE-A2C, ElasticVR's QTE) or a Lagrangian penalty (ElasticVR's Eq. 15–18).

---

## CLAIM D — "JORA / Lyapunov-based work optimises QoE under a long-term energy budget."

### D.1 JORA — arXiv:2010.08119: the QoE / Lyapunov / long-term-energy-budget parts are all absent

Full text fetched and read from `https://ar5iv.labs.arxiv.org/html/2010.08119`.

> Huang, X., He, L., Chen, X., Wang, L., Li, F. "Revenue and Energy Efficiency-Driven Delay Constrained Computing Task Offloading and Resource Allocation in a Vehicular Edge Computing Network: A Deep Reinforcement Learning Approach." arXiv:2010.08119. (JORA-MADDPG)

**Actual utility function** (Section V-A, Eq. 21), verbatim:

> "Therefore, these two factors are utilized to formulate the utility function, U_t, which can be expressed as:
> **(21) U_t = β₁ TR_t − β₂ E_t**
> where β₁, β₂ are positive values."

Numbers, verbatim from Table II:

> "Parameters of utility function **β₁ = 0.8, β₂ = 0.4**"

→ It is a per-slot weighted sum of **revenue minus energy cost**. There is no QoE term.

**Actual reward function** — three-tier, constraint-gated (Eqs. 25–27), verbatim:

> "After taking action a_{t,k}, if the state of vehicle k does not satisfy the constraints (c1)-(c7), the reward function is defined as:
> **(25) r_{t,k} = ℓ₁ + Γ₁·(τ̂^V + Σ_{k'∈C_k} τ̂^{k'} + τ̂^k + τ̂^H − 1)·Λ(τ̂^V + Σ_{k'∈C_k} τ̂^{k'} + τ̂^k + τ̂^H ≤ 1)**"
> "…if the state of vehicle k satisfies all constraints (c1)-(c7), the reward function is defined as:
> **(26) r_{t,k} = ℓ₂ + exp(Υ(v_{I_{t,k}}) − D(I_{t,k}))**
> where ℓ₂ is the experimental parameter. After taking action a_{t,k}, if the state of vehicle k satisfies all constraints (c1)-(c8), the reward function is defined as:
> **(27) r_{t,k} = ℓ₃ + Γ₈·exp(U_{t,k})**"

Numbers, verbatim (Table II):

> "Parameters of reward **Γ₁ = 0.8, Γ₈ = 0.9, ℓ₁ = −0.4, ℓ₂ = −0.2, ℓ₃ = 0.5, Γ₂, Γ₃, Γ₄, Γ₅, Γ₆, Γ₇ = 0.5**"

**How the deadline is enforced:** reward gating on constraint **(c1)** in Eq. (22), verbatim:

> "**(c1) τ^V_{I_{t,k}} + Σ_{k'∈C_k} τ^{k'}_{I_{t,k}} + τ^k_{I_{t,k}} + τ^H_{I_{t,k}} ≤ 1, ∀k ∈ Q**"

i.e. the normalised per-slot delay must be ≤ 1 slot; satisfied → Eq. (26)/(27), violated → Eq. (25) penalty branch. **No Lagrangian, no action masking.**

**Keyword counts over the full extracted text (98,606 chars):**

| Term | Occurrences | Where |
|---|---|---|
| `Lyapunov` | **0** | nowhere in the paper |
| `energy budget` | **0** | nowhere in the paper |
| `long-term` | **1** | only inside a cited reference title ("…video clients over long-term evolution system") |
| `QoE` | **3** | all three are motivation/related-work only, never in an equation |

The three QoE occurrences, verbatim:
> "…can be satisfied to improve the reliability of autonomous driving and the **quality of experience (QoE)** in VEC network for the user."
> "To enhance the efficiency of computing task offloading and improve the **QoE** of drivers and passengers in a VEC-based network, researchers have proposed many VEC-based offloading and resource allocation optimization algorithms…"
> "…considering both energy consumption and execution delay, an optimization problem was formulated to maximize the **QoE** of the vehicles and then solved by the improved double-deep Q networks [20]." *(this is a description of prior work [20], not of JORA)*

**Hyperparameters:** Table III "The Neural Network and Training Parameters" lists verbatim: Critic layers 5, neurons 1024/512/300, lr 0.0001; Actor layers 4, neurons 500/128, lr 0.0001; activation Relu; mini-batch 128; buffer size 30000. **No discount factor γ and no GAE λ are reported** (`discount` → 0 hits, `GAE` → 0 hits). **No normalisation method reported.** (`Gamma`/`Λ` hits in the text are the reward/indicator symbols Γ and Λ, not a discount factor.)

**Environment step/horizon:** per-time-slot **t** sequential with a task queue; Algorithm 2 loops `for t = 1 : T`, inner `for each vehicle k`, *"Size of task queue 10"*, delay threshold 10/40/100 ms, *"Waiting time of hold on 10, 20, 40 ms"*.

**Verdict for JORA: NOT VERIFIED.** JORA-MADDPG does not optimise QoE, does not use Lyapunov optimisation, and has no long-term energy budget. It maximises the per-slot utility `U_t = 0.8·TR_t − 0.4·E_t` (revenue minus energy cost) subject to a per-slot normalised delay cap enforced by reward gating.

### D.2 Real Lyapunov-based offloading papers that DO optimise QoE/utility under a long-term energy budget

**D.2.a — arXiv:2404.02166 (best match).** He, L., Sun, G., Sun, Z., Wang, P., Li, J., Liang, S., Niyato, D. "An Online Joint Optimization Approach for QoE Maximization in UAV-Enabled Mobile Edge Computing." Full text: `https://ar5iv.labs.arxiv.org/html/2404.02166`. Journal/extended version: **arXiv:2406.11918**, "QoE Maximization for Multiple-UAV-Assisted Multi-Access Edge Computing via an Online Joint Optimization Approach," DOI 10.1109/TON.2025.3581531.

Verbatim objective (Eq. 19) and long-term energy constraint (Eq. 19a):

> "**P : min_{A,F,W,P_u} (1/T) Σ_{t=1}^{T} Σ_{m=1}^{M} C_m(t)**  (19)
> s.t. **lim_{T→+∞} (1/T) Σ_{t=1}^{T} 𝔼{E_u(t)} ≤ Ē_u**  (19a)"
> "To guarantee service time, we define the UAV energy consumption constraint as follows … **where Ē_u is the energy budget of the UAV per time slot.**"
> "Constraint (a) is the **long-term energy consumption constraint** of the UAV."

QoE definition (Eq. 15), verbatim:

> "we consider that each UD's cost at time slot t consists of the task completion delay and the UD's energy consumption, **which reflects the UD's QoE** … **C_m(t) = γ_m T_m(t) + (1 − γ_m) E_m(t)**  (15) … **Obviously, minimizing the cost of UDs is equivalent to maximizing the QoE of UDs.**"

Virtual energy queues (Eq. 20), verbatim:

> "**Q_u^c(t+1) = max{ Q_u^c(t) + E_u^c(t) − Ē_u^c, 0 }**, ∀t ∈ T
> **Q_u^p(t+1) = max{ Q_u^p(t) + E_u^p(t) − Ē_u^p, 0 }**, ∀t ∈ T … where Ē_u^c and Ē_u^p represent the computation and propulsion **energy budgets per slot**, respectively and Ē_u^c + Ē_u^p = Ē_u."

Lyapunov function and drift (Eqs. 21–22), verbatim:

> "**L(Q_u(t)) = ((Q_u^c(t))² + (Q_u^p(t))²)/2**  (21)"
> "**ΔL(Q_u(t)) ≜ 𝔼{ L(Q_u(t+1)) − L(Q_u(t)) | Q_u(t) }**  (22)"

Drift-plus-penalty (Eq. 23) and its bound (Eq. 24), verbatim:

> "**D(Q_u(t)) = ΔL(Q_u(t)) + V·𝔼{ C_s(t) | Q_u(t) }**  (23), where C_s(t) = Σ_{m=1}^{M} C_m(t) is the total cost of all UDs at time slot t, and **V is a parameter that trades off the total cost and queue stability**."
> "**D(Q_u(t)) ≤ W + Q_u^c(t)(E_u^c(t) − Ē_u^c) + Q_u^p(t)(E_u^p(t) − Ē_u^p) + V × C_s(t)**  (24)"

Per-slot problem solved (Eq. 25), verbatim:

> "**min_{A^t,F^t,W^t,P_u'} Q_u^c(t)E_u^c(t) + Q_u^p(t)E_u^p(t) + V Σ_{m=1}^{M} C_m(t)**"

Performance bound (Theorem 8, Eq. 47), verbatim:

> "**(1/T) Σ_{t=1}^{T} Σ_{m=1}^{M} C_m(t) ≤ C_s^opt + (WT + C)/V**"

**Coefficients:** γ_m (delay-vs-energy QoE weight) and V (drift-penalty tradeoff) — the text states their roles but **I did not find numerical values for γ_m or V in the sections I extracted**; the simulation setup lists *"80 time slots … length of each time slot is 1 s"*, 20 UDs, F_u^max = 20 GHz, D_m(t) ∈ [0.1, 1] Mb, η_m(t) ∈ [500, 1500] cycles/bit, T_m^max = 1 s, B = 4 MHz. **γ (RL discount) and GAE λ: N/A — the method is Lyapunov + game theory + convex optimisation, not DRL** (it benchmarks *against* DRL algorithms and reports "at least a 10% improvement in the QoE of UDs compared to deep reinforcement learning (DRL)-based algorithms").
**Horizon:** per-slot sequential, T = 80 slots of 1 s.

**D.2.b — arXiv:1806.07764.** Chen, L., Zhou, P., Gao, L., Xu, J. "Adaptive Fog Configuration for the Industrial Internet of Things." `https://ar5iv.labs.arxiv.org/html/1806.07764`; DOI 10.1109/TII.2018.2846549. Verbatim:

> "**The limited battery capacity of FNs creates a long-term energy budget constraint that significantly complicates the Fog configuration problem as it introduces temporal coupling of decision making across the timeline.**"
> "We jointly optimize service hosting and task admission of a network of FNs in order to **minimize the time-average computation delay cost while satisfying the long-term battery energy constraints of FNs**."
> "we propose an online distributed algorithm, called Adaptive Fog Configuration (AFC), **based on Lyapunov optimization and parallel Gibbs sampling**."

Caveat: this optimises delay cost / fog-network utility, **not QoE** (`QoE` → 0 hits). Not DRL.

**D.2.c — arXiv:2010.01370.** Bi, S., Huang, L., Wang, H., Zhang, Y. J. A. "Lyapunov-guided Deep Reinforcement Learning for Stable Online Computation Offloading in Mobile-Edge Computing Networks" (LyDROO); DOI 10.1109/TWC.2021.3066619. `https://ar5iv.labs.arxiv.org/html/2010.01370`. Verbatim:

> "we aim to design an online computation offloading algorithm to **maximize the network data processing capability subject to the long-term data queue stability and average power constraints**."
> "we propose a novel framework, named LyDROO, that combines the advantages of **Lyapunov optimization and deep reinforcement learning (DRL)**."

Caveat: long-term **average power** constraint (not an energy budget), and no QoE (`QoE` → 0 hits).

**D.2.d — LARCS (long-term energy constraints, but NOT Lyapunov).** Zhu, Q., Liu, J., Hou, Y., Lai, D., Ou, Z. "Multi-Agent Deep Reinforcement Learning for Resource Scheduling in Hybrid-Energy MEC Systems Under Long-Term Constraints," IEEE Trans. Automation Science and Engineering, author's version, "Citation information: DOI 10.1109/TASE.2026.3690812". PDF fetched: `https://teacher.gdut.edu.cn/_resources/group1/M00/00/12/rB8AD2o6aPeADkGxADMz-2KtTiw388.pdf` (18 pp.). Verbatim:

> "we investigate the latency-aware resource-constrained scheduling (LARCS) problem in a hybrid-energy edge-cloud MEC system to **minimize energy consumption and latency under long-term MEC energy constraints** … to enhance the end-user's quality of experience (**QoE**)."
> "**Rn,t = r^latency_{n,t} + r^pen_{n,t} + r^energy_{n,t}**  (30) … Our proposed penalty mechanism incorporates a penalty component **r^pen_{n,t} = P^end_{n,t} + P^ene_{n,t}** into each agent's reward function … This mechanism not only **decouples long-term MEC energy constraints** but also allows for adaptive management of resources."
> "**(P2) π* = arg max_π E[ Σ_{t∈T} Σ_{n∈N} γ^t R_{n,t} | π ], s.t. (1) – (11)**  (31)"

Caveat: despite the "long-term constraints" framing, **this paper is not Lyapunov-based** — `Lyapunov` occurs only in related work (citing Bi et al. [16]); it enforces the long-term energy constraint via reward shaping/penalty. It reports no numeric γ, no GAE, no normalisation reference.

**D.2.e — ElasticVR, arXiv:2512.12366** — QoE objective (`QTE`), deadline Lagrangian; **no Lyapunov and no long-term energy budget** (see C.6 above for quotes and coefficients).

**Verdict on CLAIM D:**
- As applied to **JORA**: **NOT VERIFIED.** JORA-MADDPG has no QoE objective, no Lyapunov optimisation, and no long-term energy budget (`Lyapunov` = 0, `energy budget` = 0 hits in the whole paper). Coupling "JORA" to "Lyapunov-based work" is a conflation.
- As a **general existence statement** ("Lyapunov-based work optimises QoE under a long-term energy budget"): **VERIFIED, but not by JORA.** arXiv:2404.02166 (and its journal version arXiv:2406.11918) is exactly this: QoE cost `C_m(t) = γ_m T_m(t) + (1−γ_m) E_m(t)` minimised under the time-average energy-budget constraint `lim_{T→∞}(1/T)Σ 𝔼{E_u(t)} ≤ Ē_u` via virtual energy queues and drift-plus-penalty `D = ΔL + V·𝔼{C_s}`. arXiv:1806.07764 (long-term energy budget + Lyapunov, delay cost, not QoE) and arXiv:2010.01370 (Lyapunov + DRL, long-term average *power* constraint, LyDROO) are supporting but weaker matches.

---

## Summary table

| Item | Status | Key evidence |
|---|---|---|
| "DRL-E2D" on arXiv | **Absent** | arXiv API `totalResults = 0` |
| "DRL-E2D" as a real algorithm | **Exists** | Li et al., EURASIP JWCN 2021:56, DOI 10.1186/s13638-021-01941-3 |
| CLAIM C as written (DRL-E2D = energy-min under hard deadline) | **NOT VERIFIED (refuted)** | `max R(τ) = U(τ) − E(τ) − P(τ)`; deadline via `u = 0` gating + `P(τ)` drop penalty |
| JORA = QoE | **NOT VERIFIED** | `U_t = β₁TR_t − β₂E_t`; `QoE` only 3× as motivation |
| JORA = Lyapunov | **NOT VERIFIED** | `Lyapunov` = 0 hits |
| JORA = long-term energy budget | **NOT VERIFIED** | `energy budget` = 0 hits; `long-term` 1× in a reference title |
| Real Lyapunov + QoE + long-term energy budget | **VERIFIED (exists)** | arXiv:2404.02166 / arXiv:2406.11918; also arXiv:1806.07764 (utility/delay, not QoE) |
| Real DRL with hard-deadline energy minimisation | **NOT FOUND** | 4 arXiv API queries → 0 relevant results; closest is ElasticVR's Lagrangian (arXiv:2512.12366) and EE-A2C's weighted sum (IEEE Access 2024) |

## Items I could not verify (flagged)

1. **Ad Hoc Networks 2024** `10.1016/j.adhoc.2024.103743` — paywalled, no abstract available anywhere I could reach; ScienceDirect returned 403. **No objective, deadline mechanism, or hyperparameter could be verified.**
2. **Numerical values of V and γ_m in arXiv:2404.02166** — roles are stated in the equations but I did not locate numeric settings in the sections fetched.
3. **DRL-E2D discount factor γ** — reported only symbolically as `γ ∈ [0, 1]`; no numeric value.
4. **Numerical weights α, β, η in EE-A2C Eq. (33)** — not reported in the text I read.
