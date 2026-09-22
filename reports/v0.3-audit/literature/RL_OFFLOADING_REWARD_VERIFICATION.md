# Reward / constraint / discounting design in DRL offloading literature — verification report

**Scope.** How published DRL / PPO / MAPPO / MORL offloading papers formulate (a) reward, (b) latency–energy
weighting, (c) constraints & penalties, (d) γ and GAE λ, (e) normalisation and its reference.
Bias to VEC / V2X / vehicular settings, plus strong MEC/UAV-MEC entries.

**Verification policy.** Every formula, coefficient and value below was read from a source actually fetched this
session (arXiv HTML/LaTeX/PDF, Europe PMC full-text XML, mdpi-res OA PDFs, Nature, publisher OA PDFs). Nothing is
from memory. Where a paper does not print a value it is marked **not stated** — never guessed. Where a claim could
not be substantiated it is marked **NOT VERIFIED** with the closest real counter-evidence given instead.

**Marker legend:** `[FT]` full text read · `[PART]` full text read but a requested field is genuinely not printed ·
`[PAYWALL]` only metadata reachable · `[BLOCKED]` source unreachable.

---

## PART 1 — Verdicts on the seven draft claims

| # | Claim as drafted | Verdict |
|---|---|---|
| 1 | 2025 PPO: `r = -Σ(αT + βE)`, α+β=1; γ=0.93 converged in 4000 iters, γ=0.99 and 0.90 failed at 6000 | **NOT VERIFIED** |
| 2 | QoS-aware VEC papers evaluate balanced 0.5/0.5, delay-sensitive 0.8/0.2, energy-sensitive 0.2/0.8 | **VERIFIED** (exact, verbatim) |
| 3 | HDMAPPO: `-(αD + βE + φ·dropout)` with a separate deadline-violation penalty branch | **VERIFIED**, with 3 corrections |
| 4 | VEC/P-PPO uses `10/(10 + λt + μe)` | **PARTIALLY VERIFIED** |
| 5 | DRL-E2D minimises energy subject to a task-deadline constraint rather than scalarising 0.5/0.5 | **NOT VERIFIED (refuted)** |
| 6 | JORA / Lyapunov work optimises QoE under a long-term energy budget | **NOT VERIFIED for JORA; VERIFIED for a different paper** |
| 7 | MORL-PPO (2023) and a newer IEEE TSC paper train preference-conditioned policies for a Pareto front because weights are unknown a priori | **VERIFIED with one correction** |

---

### Claim 1 — `NOT VERIFIED`

No paper was found that reports γ = 0.93 as an adopted value, nor convergence at 4000 vs failure at 6000
iterations as a function of γ. **Do not cite the draft's numbers.** However, *both halves of the phenomenon the
draft is reaching for are real, and there are now much better citations than the ones the draft used* — including
one that is a **DAG-based VEC** paper whose γ grid contains exactly 0.93.

**Best replacements found** (all `[FT]`, and both top entries re-verified directly in this pass):

1. **Jiang, Jin, Xin, Chen & Guo**, *"Vehicle as a Service: Fuzzy Reward-Based Multi-Agent Deep Reinforcement
   Learning for Task Scheduling in Vehicular Edge Computing"*, **Systems 2026, 14(9), 1103**, DOI
   [10.3390/systems14091103](https://doi.org/10.3390/systems14091103) —
   <https://mdpi-res.com/d_attachment/systems/systems-14-01103/article_deploy/systems-14-01103.pdf>
   **FRMPPO (multi-agent PPO) for DAG-task VEC scheduling, objective = minimise task completion latency *and*
   vehicle energy consumption.** This is the only offloading paper found whose γ grid covers all three values the
   draft claims. Table 4 verbatim: `Discount factor γ {0.90, 0.91, 0.93, 0.95, 0.97, 0.99}`,
   `GAE coefficient λ 0.95`, `Clipping ratio ϵ 0.20`, learning rate grid `{0.0001 … 0.005}`. Verbatim reasoning:
   > "the discount factor γ ∈ {0.90, 0.91, 0.93, 0.95, 0.97, 0.99} is evaluated. The converged reward also exhibits
   > a peak at γ = 0.95, where the reward distribution is the most compact, and the median is the highest. When the
   > discount factor is too small (0.90 and 0.91), agents place excessive emphasis on immediate rewards and ignore
   > long-term cooperative gains. When the discount factor is too large (0.97 and 0.99), future rewards are
   > insufficiently discounted, making the policy less responsive to the current environmental state. Both cases are
   > unfavorable for cooperative multi-vehicle task scheduling. … this paper finally selects ηθ = ηϕ = 0.0005 and
   > **γ = 0.95** as the optimal hyperparameter configuration."
   It **selects 0.95, not 0.93** — but it is the right paper to cite for "the γ trade-off was swept and both
   extremes underperform", and it is directly on-topic for a DAG VEC project.

2. **"Maritime Mobile Edge Computing for Sporadic Tasks: A PPO-Based Dynamic Offloading Strategy" (ON-PPO)**,
   **Mathematics 2025, 13(16), 2643**, DOI [10.3390/math13162643](https://doi.org/10.3390/math13162643) —
   <https://mdpi-res.com/d_attachment/mathematics/mathematics-13-02643/article_deploy/mathematics-13-02643-v2.pdf>
   The only offloading paper found that reports **high γ failing to converge**. Verbatim:
   > "Figure 6 illustrates the impact of different gamma values on algorithm convergence and performance. When
   > gamma is 0.5, the training outcome is the worst as the agent focuses excessively on immediate rewards. In
   > contrast, **when gamma is 0.99 or 0.999, the agent's overemphasis on long-term rewards results in poor
   > algorithm performance and a failure to converge.** Results show that a gamma of 0.9 balances convergence speed
   > and average performance well, so it is used as the default gamma for later experiments."
   Table 2: `Discount rate γ 0.9`, `Number of episode T 1000`. **γ = 0.9, not 0.93; and no iteration counts** —
   but this is the real "high γ does not converge" citation the draft needs.

3. **Wang et al., *Electronics* 2024, 13(12), 2387** (M-GNRL, MEC) — ablates
   `discount factor γ: 0.93, 0.95, 0.97, 0.99` and selects **0.97**: *"the highest average reward achieved when
   γ = 0.97, outperforming other values."* Convergence is discussed in **episodes (~920–1050)**, not iterations.
   Also verbatim: *"Conversely, when γ = 1, convergence issues may emerge"*. `[FT]`

4. **Chen, Y.; Tong, Y.**, *Computation Offloading in Space–Air–Ground Integrated Networks …*, **Future Internet
   2025, 17(12), 542**, DOI [10.3390/fi17120542](https://doi.org/10.3390/fi17120542) — ablation over
   **{0.8, 0.95, 0.99}**, selects **0.95**. Verbatim: *"With a discount factor of 0.99 (blue line), although the
   optimization performance is poor before 700 episodes, there is an improvement later, though it still does not
   outperform the 0.95 case."* `[FT]`

**No paper was found anywhere in the 33-paper sweep that adopts γ = 0.93.** The draft's specific numbers
(γ = 0.93 converging at 4000 iterations, 0.90/0.99 failing at 6000) correspond to **no published result**: the
0.93 value appears only as one grid point among six in Systems 14:1103, and no paper reports 4000/6000-iteration
convergence thresholds. **The numbers appear fabricated or garbled — but the qualitative claim (γ is swept, both
extremes underperform, and high γ can fail to converge) is well supported by citations 1 and 2 above.**

---

### Claim 2 — `VERIFIED` (verbatim, single source)

**Citation.** Chenhong Cao, Meijia Su, Shengyu Duan, Miaoling Dai, Jiangtao Li, Yufeng Li, *"QoS-Aware Joint Task
Scheduling and Resource Allocation in Vehicular Edge Computing"*, **Sensors 2022, 22(23), 9340**, DOI
[10.3390/s22239340](https://doi.org/10.3390/s22239340). Full text fetched from the Europe PMC REST endpoint
(the PMC landing page is reCAPTCHA-gated):
<https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9736212/fullTextXML>

**Verbatim (results §5.2.2):**
> "Our MOV algorithm achieves better performance compared with other algorithms under three average QoS weights:
> **"Balanced" (λ = 0.5, μ = 0.5), "Delay-sensitive" (λ = 0.8, μ = 0.2), and "Energy-sensitive" (λ = 0.2, μ = 0.8)**."

Caveats for the draft: (i) the three operating points are a *comparison protocol* over an NSGA-II Pareto selector
(MOV) and a PPO baseline — not a claim that every QoS-aware paper does this; no second independent paper with the
exact same trio was found. (ii) It is a **weight sweep on evaluation**, not a training-time weight ablation.
`[FT]`. Reported failure mode: at 50 vehicles the PPO arm becomes untrainable — *"we find that the neural network
model is very difficult to train when the number of vehicles is 50 because of the high dimensionality"*.

---

### Claim 3 — `VERIFIED`, with three corrections

**Citation.** Yu Sun, Qijie He, *"Computational Offloading for MEC Networks with Energy Harvesting: A Hierarchical
Multi-Agent Reinforcement Learning Approach"*, **Electronics 2023, 12(6), 1304**, DOI
[10.3390/electronics12061304](https://doi.org/10.3390/electronics12061304) —
<https://mdpi-res.com/d_attachment/electronics/electronics-12-01304/article_deploy/electronics-12-01304.pdf>
(`mdpi.com` 403s; the `mdpi-res.com` OA mirror is the reliable route). **HDMAPPO** is not on arXiv
(`all:HDMAPPO` → 0 results).

Acronym: *"hierarchical double multi-agent proximal policy optimization (HDMAPPO) task offloading method, where the
high level uses discrete MAPPO to generate server location selection for each task and the low level uses continuous
MAPPO to generate the task offloading ratio."*

**Cost, Eq. (14):** `min_{L_t,X_t} lim_{τ→∞} (1/τ) Σ_{t=0}^{τ} [ α Σᵢ Dᵢᵗ/N + β Σᵢ Eᵢᵗ/N + φ·dropoutᵗ ]`.

**Reward, Eq. (29):**
```
r_i^t = { −penalt                            if D_i^t > T_max  or  h_i^{a_t} = 0
        { −(α·D_i^t + β·E_i^t + φ·dropout^t)  else
```
*"The reward function is designed to consider the task delay, energy consumption, and task drop rate."* The
low-level MAPPO uses the same Eq. (29).

**Three corrections:**
1. The penalty branch fires on **deadline violation `D_i^t > T_max` OR active discard `h_i^{a_t} = 0`** — not
   deadline alone.
2. On violation the reward is replaced by a **single fixed constant `−penalt`**; the episode is **not terminated**
   (no done/terminal/reset language anywhere; Algorithm 2 loops `t = 1..T`). The task is simply discarded.
3. **α, β, φ, penalt and GAE λ have no numeric values anywhere in the paper.** Only γ is printed: Table 3
   `Reward discount 0.99` (independently confirmed by direct grep: `Reward discount 0.99`). GAE λ appears only
   symbolically in Eq. (25) as `(γλ)^l(...)`. Clip 0.2, K_epochs 10, actor lr 3e-4, critic lr 4e-4.

**T/E combination:** weighted sum, α+β not constrained to 1 (values unreported).
**Normalisation:** none — `normaliz` returns 0 hits.
**Step/horizon:** per-slot sequential (one offloading decision per UE per slot), episode = T slots; T_max = 1 s,
task size [300,500] Kbit, density [800,1200] cycles/bit. `[FT]`/`[PART]`

---

### Claim 4 — `PARTIALLY VERIFIED`

The formula is real and verbatim, **but it is stated inside the Sensors 2022 paper above, which attributes it to
P-PPO as a baseline**; the underlying P-PPO paper is paywalled.

**Verbatim (Cao et al., Sensors 2022, 22(23):9340):**
> "In Table 4, we set the reward value model as follows: **reward = 10 / (10 + λ × t + μ × e)**, where t and e are
> the delay and energy consumption, respectively, of each vehicle. **In the P-PPO algorithm, the reward value model
> is used as the reward function.**"

So the exact string matches the claim. However:
- It is this paper's **own comparison reward model**, applied to the P-PPO baseline's outputs; MOV (non-RL) results
  are converted into the same reward post-hoc.
- "P-PPO" resolves to reference **[21]** of that paper: **Wu, Y.; Xia, J.; Gao, C.; Ou, J.; Fan, C.; Ou, J.; Fan, D.**
  *"Task offloading for vehicular edge computing with imperfect CSI: A deep reinforcement approach"*,
  **Physical Communication 2022, 55, 101867**, DOI
  [10.1016/j.phycom.2022.101867](https://doi.org/10.1016/j.phycom.2022.101867) — **`[PAYWALL]`**: ScienceDirect
  abstract page, ACM DL, typeset.io all gated; the reward function could **not** be independently confirmed in the
  source paper. The acronym "P-PPO" itself is not confirmed to originate there.
- Structure: weighted share inside a saturating reciprocal; λ + μ = 1. No penalty terms. γ and GAE λ **not stated**
  (the words "discount" and "episode" do not occur in the article). `[PART]`

**Draft-safe phrasing:** *"A VEC comparison study evaluates PPO with the saturating reward r = 10/(10 + λt + μe)
[Cao et al., Sensors 2022]"* — not *"VEC/P-PPO uses …"* as a property of the P-PPO method.

---

### Claim 5 — `NOT VERIFIED (refuted)`

**DRL-E2D is a real algorithm** — but not on arXiv (`all:"DRL-E2D"` → 0 results). It is:

**Li, Z.; Chang, V.; Ge, J.; Pan, L.; Hu, H.; Huang, B.**, *"Energy-aware task offloading with deadline constraint in
mobile edge computing"*, **EURASIP Journal on Wireless Communications and Networking 2021:56**, DOI
[10.1186/s13638-021-01941-3](https://doi.org/10.1186/s13638-021-01941-3). Full text retrieved (24 pp.) via the
SpringerOpen counter-PDF endpoint; landing page and the Teesside green-OA mirror are Cloudflare-gated.
`[FT]` on the PDF body.

**Its formulation is the opposite of the claim.** Verbatim from its contribution list:
> "DRL-E2D deals with the task offloading problem by maximizing the well-designed reward, **which is the weighted sum
> of the utility of processed tasks, energy consumption, and the penalty of task dropping**."

Objective (Eq. 16a) verbatim: **`max : R(τ) = U(τ) − E(τ) − P(τ)`**, with
`E(τ) = β₀(τ)Eᵉˣ₀(τ) + Σⁿᵢ₌₁ αᵢ(τ)Eᵗˣᵢ(τ)` (12),
`u(tⱼ) = u if T(tⱼ) ≤ T_DL else 0` (13),
`U(τ) = Σⁿᵢ₌₀ Σ^{β_k(τ)}_{j=1} u(tⱼ)` (14),
`P(τ) = Σⁿᵢ₌₀ dᵢ(τ)` (15).
The three terms carry **implicit unit weights** — it *is* a scalarised weighted sum. Constraints (16b–16e) are task
count / transmission capacity / offloading time / computing capacity; **there is no `min E s.t. deadline` problem.**

**Deadline mechanism:** reward gating + drop-count penalty (not a hard constraint, not Lagrangian, not masking):
*"If the waiting time and execution time of task tⱼ is less than and equal to the deadline … the MD gains the u
utility of successfully finished task; otherwise, MD obtain zero utility"*; *"if a task misses the deadline, this
task will be dropped, incurring the penalty."* `u = 1`, penalty coefficient implicit 1 (count of dropped tasks),
`T_DL = 3·T_slot`. **γ: no numeric value** (`γ ∈ [0,1]` symbolic). **No GAE.** **No normalisation reported.**
Per-time-slot sequential: `T_slot = 1 s`, `N_slot = 1000` slots per episode. `[FT]`/`[PART]`

**Two other candidate "hard-deadline energy-min" papers also fail:**
- *"An Energy-Efficient Dynamic Offloading Algorithm for Edge Computing Based on DRL"* = **Zhu, K. et al., IEEE
  Access 12:127489–127506, 2024**, DOI [10.1109/ACCESS.2024.3452190](https://doi.org/10.1109/ACCESS.2024.3452190)
  (method "EE-A2C"). `[FT]`. Another weighted sum: `r(t) = α·ω_f + β·ω_e + η·ω_d` (Eq. 33) with
  `ω_f = 1 − U_nf/E_TC`, `ω_e = −log₂(rᵉᵢ[−1]/φₑ + 1e−10)`, `ω_d = 1 − φ_d/D_max`. The deadline appears only as
  soft constraint c4 via `ω_d`. γ = 0.9 (swept 0.3/0.6/0.9/0.95/0.99: *"we identified 0.9 as the optimal value for
  γ"*). α, β, η **not printed**. No GAE, no normalisation.
- Ad Hoc Networks 2024, DOI [10.1016/j.adhoc.2024.103743](https://doi.org/10.1016/j.adhoc.2024.103743) —
  `[PAYWALL]`: Semantic Scholar reports `openAccessPdf.status = CLOSED` and `abstract = null`; ScienceDirect 403.
  Not usable as evidence.

**Closest real hard-constraint mechanism found:** **ElasticVR, arXiv:2512.12366** (full text
<https://ar5iv.labs.arxiv.org/html/2512.12366>) encodes the deadline as an explicit inequality via a Lagrangian
dual — `min_{λ_k} max_{e,u} QTE(e,u) + Σ_k λ_k (T^d_k − T^r_k)` (15), with
`λ*_k = 0` if `T^d_k − T^r_k ≥ 0`, else `λ^penalty_k (T^d_k − T^r_k)` (18), **λ^penalty_k = 10**; base objective
still a weighted sum `QTE = w₀Q − w₁T − w₂E` (11) with **w₀=0.35, w₁=0.85, w₂=0.15**, **γ = 0**. `[FT]`

**Bottom line:** four arXiv API queries for hard-deadline energy minimisation returned nothing relevant. The
sentence should be deleted or rewritten; the honest contrast is *weighted-sum / gated reward* (DRL-E2D, EE-A2C)
versus *Lagrangian deadline penalty* (ElasticVR) versus *Lyapunov drift-plus-penalty* (below).

---

### Claim 6 — `NOT VERIFIED` for JORA; `VERIFIED` for a different paper

**(a) JORA = JORA-MADDPG, arXiv:2010.08119** — Huang, X.; He, L.; Chen, X.; Wang, L.; Li, F., *"Revenue and Energy
Efficiency-Driven Delay Constrained Computing Task Offloading and Resource Allocation in a Vehicular Edge Computing
Network: A Deep Reinforcement Learning Approach"* ([arXiv:2010.08119](https://arxiv.org/abs/2010.08119)).
Full text `[FT]` at <https://ar5iv.labs.arxiv.org/html/2010.08119>.

- **Utility (Eq. 21), verbatim:** `U_t = β₁·TR_t − β₂·E_t` — *"where β₁, β₂ are positive values"*; Table II
  `β₁ = 0.8, β₂ = 0.4`. This is **revenue minus energy cost**, not QoE.
- **Reward:** three-tier, constraint-gated (Eqs. 25–27): fails (c1)–(c7) → `r = ℓ₁ + ΣΓᵢ·(violation terms)·Λ(·)`;
  satisfies (c1)–(c7) → `r = ℓ₂ + exp(Υ(v_I) − D(I))`; satisfies (c1)–(c8) → `r = ℓ₃ + Γ₈·exp(U)`. Table II:
  `Γ₁ = 0.8, Γ₈ = 0.9, ℓ₁ = −0.4, ℓ₂ = −0.2, ℓ₃ = 0.5, Γ₂…Γ₇ = 0.5`.
- **Deadline:** constraint (c1) `τ^V + Σ τ^{k'} + τ^k + τ^H ≤ 1`, enforced by **reward-branch gating**.
- **Keyword counts over the entire full text:** `Lyapunov` = **0**, `energy budget` = **0**, `long-term` = 1 (inside
  a cited reference title), `QoE` = **3**, all motivation — and one of those three describes **prior work [20]**.
- **γ and GAE λ: not stated** (`discount` = 0 hits, `GAE` = 0 hits). **No normalisation reported.**
  Per-time-slot sequential over a task queue (Algorithm 2: `for t = 1:T`, then per vehicle); delay thresholds
  10/40/100 ms. `[FT]`/`[PART]`

→ JORA supports *none* of QoE, Lyapunov, or a long-term energy budget.

**(b) The real "Lyapunov + QoE + long-term energy budget" paper:**

**He, L.; Sun, G.; Sun, Z.; Wang, P.; Li, J.; Liang, S.; Niyato, D.**, *"An Online Joint Optimization Approach for
QoE Maximization in UAV-Enabled Mobile Edge Computing"*, [arXiv:2404.02166](https://arxiv.org/abs/2404.02166)
(extended version [arXiv:2406.11918](https://arxiv.org/abs/2406.11918); DOI
[10.1109/TON.2025.3581531](https://doi.org/10.1109/TON.2025.3581531)). Full text `[FT]`.

- **QoE cost (Eq. 15), verbatim:** `C_m(t) = γ_m·T_m(t) + (1 − γ_m)·E_m(t)`; *"each UD's cost … consists of the task
  completion delay and the UD's energy consumption, which reflects the UD's QoE … minimizing the cost of UDs is
  equivalent to maximizing the QoE of UDs."*
- **Long-term energy budget constraint (19a), verbatim:** `lim_{T→+∞} (1/T) Σ_{t=1}^{T} 𝔼{E_u(t)} ≤ Ē_u`;
  *"where Ē_u is the energy budget of the UAV per time slot"*; *"Constraint (a) is the long-term energy consumption
  constraint of the UAV."*
- **Virtual energy queues (20):** `Q_u^c(t+1) = max{Q_u^c(t) + E_u^c(t) − Ē_u^c, 0}` (and propulsion analogue).
- **Lyapunov function / drift (21–22):** `L(Q_u(t)) = ((Q_u^c(t))² + (Q_u^p(t))²)/2`,
  `ΔL(Q_u(t)) ≜ 𝔼{L(Q_u(t+1)) − L(Q_u(t)) | Q_u(t)}`.
- **Drift-plus-penalty (23):** `D(Q_u(t)) = ΔL(Q_u(t)) + V·𝔼{C_s(t) | Q_u(t)}`; bound (24):
  `D ≤ W + Q_u^c(t)(E_u^c(t) − Ē_u^c) + Q_u^p(t)(E_u^p(t) − Ē_u^p) + V·C_s(t)`; per-slot problem (25):
  `min Q_u^c(t)E_u^c(t) + Q_u^p(t)E_u^p(t) + V·Σ_m C_m(t)`; bound (Thm 8, Eq. 47):
  `(1/T)ΣΣC_m(t) ≤ C_s^opt + (WT+C)/V`.
- **Caveat:** **numeric V and γ_m are not printed** in the extracted sections; and this is **Lyapunov + game theory
  + convex optimisation, not DRL** (it benchmarks against DRL). Sim: 80 slots × 1 s, 20 UDs, T_m^max = 1 s.

**Supporting real Lyapunov papers** (neither optimises QoE): arXiv:1806.07764 (long-term battery energy budget,
minimises delay cost; `QoE` = 0 hits) and **LyDROO** arXiv:2010.01370 (Lyapunov + DRL, but a long-term average
*power* constraint, no QoE). The LARCS paper (IEEE TASE, DOI 10.1109/TASE.2026.3690812) *looks* like a match but is
**not** Lyapunov-based (`Lyapunov` only in related work) — it uses reward shaping.

---

### Claim 7 — `VERIFIED`, with one correction

**(a) 2023 MORL-PPO offloading — VERIFIED, but it is multi-policy, not preference-conditioned.**

**Yang, N.; Wen, J.; Zhang, M.; Tang, M.**, *"Multi-objective Deep Reinforcement Learning for Mobile Edge
Computing"*, **2023 21st Int. Symp. on Modeling and Optimization in Mobile, Ad Hoc, and Wireless Networks (WiOpt)**,
DOI [10.23919/wiopt58741.2023.10349870](https://doi.org/10.23919/wiopt58741.2023.10349870) —
[arXiv:2307.14346](https://arxiv.org/abs/2307.14346) · <https://arxiv.org/html/2307.14346v1>. `[FT]`

- Unknown-weights → Pareto rationale: **VERIFIED**. *"conventional single-objective scheduling solutions cannot be
  directly applied to practical systems in which the preferences of these applications (i.e., the weights of
  different objectives) are often unknown or challenging to specify in advance."*
- Preference-conditioned single policy: **NO.** Verbatim: *"In the training phase, the MORL algorithm **trains a
  parametric network for each preference**"*; *"We set the preference set as Ω with an equal interval 0.02 and obtain
  50 preferences to fit the Pareto front."* The **actor** is `π_θ(a_t|s_t)` — ω is not an actor input; only the
  critic is ω-weighted.
- Reward: `r_E = −Ê_m`, `R_E = −Σ_m Ê_m`; `r_T = −(T̂_m + Σ_{m'∈M_e(τ_t)} ΔT̂^{a_t}_{m'})`;
  `r_ω = ω^T × (α_T·r_T, α_E·r_E)`; `R_ω = Σ_t r_ω`. *"where α_T and α_E are coefficients for adjusting delay
  r_T(t) and energy consumption r_E(t) to the same order of magnitude."* **α_T/α_E numeric values not given.**
- **No penalty terms at all** (no dropout, no deadline, no constraint branch).
- Table I (independently re-read from the arXiv HTML): `Discount factor γ 0.9`, `GAE discount factor λ 0.95`,
  `Clip parameter ε 0.2`, lr 1e-6.
- No normalisation described. Per-step sequential (one task per step, m = t).

**(b) Newer IEEE TSC paper — VERIFIED, but it is Discrete-SAC, not PPO.**

**Yang, N.; Wen, J.; Zhang, M.; Tang, M.**, *"Generalizable Pareto-Optimal Offloading With Reinforcement Learning in
Mobile Edge Computing"*, **IEEE Transactions on Services Computing, 2025**, DOI
[10.1109/TSC.2025.3604371](https://doi.org/10.1109/TSC.2025.3604371) —
[arXiv:2509.10474](https://arxiv.org/abs/2509.10474) · <https://arxiv.org/html/2509.10474v1>. `[FT]`

- Genuinely **single preference-conditioned policy**, explicitly *because* weights are unknown a priori:
  *"Impossibility: Weights may be unknown when designing or learning an offloading scheme. Infeasibility: Weights may
  be diverse … Undesirability: Even if weights are known, nonlinear objective functions may lead to non-stationary
  optimal policies."* … *"Our method uses a single policy model to efficiently schedule tasks based on varying
  preferences."*
- Objective (12a): `min_π 𝔼_{x∼π}[ Σ_{m∈M} γ^m (ω_T·T_m + ω_E·E_m) ]`, with `ω_T + ω_E = 1`.
- Vector reward (22)–(26): `r_E(s_t,a_t) = −Ê_m`; `r_T(s_t,a_t) = −(T̂_m + Σ_{m'∈M_e(τ_t)} ΔT̂^{a_t}_{m'})`.
  Scalarised (30): `r_ω(s_t,a_t) = ω^T × (α_T·r_T, α_E·r_E)`; *"where α_T and α_E are coefficients for adjusting
  delay r_T(t) and energy consumption r_E(t) to the same order of magnitude."*
- **No penalty terms.** Action masking for dummy edge servers (softmax renormalisation), not a reward penalty.
- Hyperparameters (independently re-read): `Discount factor γ 0.95`, `learning rate of temperature λ_{α_H} 0`,
  γ ∈ [0,1) in the MOMDP tuple. **No GAE** (off-policy SAC). No reward normalisation; the only scaling is the
  softmax over masked action probabilities.
- Horizon: *"Consider one episode consisting of T steps, each with a duration of Δt seconds"*; Table III T = 100
  steps, Δt = 1 s; one task offloaded per step. Hypervolume reference point = *"the maximum delay and energy
  consumption across all Pareto fronts"*.

**(c) Correction on the strong lead.** IEEE document **11626586** is **NOT** IEEE TSC. It is **Yanran Liu, Zhengli
Liu, Weiyu Yuan, Cheng Zeng, Peng He, Tao Peng, Jian Wang, Bing Li**, *"A Hierarchical Preference-Guided
Multi-Objective Reinforcement Learning Method for Task Offloading in Vehicular Edge Computing"*, **IEEE Transactions
on Vehicular Technology, 2026**, DOI [10.1109/TVT.2026.3717633](https://doi.org/10.1109/TVT.2026.3717633)
(venue/year confirmed 3× via Crossref, Semantic Scholar, OpenAlex). **Its content is `NOT VERIFIED`**: closed
access, empty abstract, IEEE returns an empty HTTP 202, and no preprint exists
(`all:"preference guided" AND all:"task offloading"` → 0 arXiv results). Do not cite its reward or Pareto claims.

---

## PART 2 — Per-paper extraction (verified set)

`[FT]` unless flagged. Format: **citation** · reward/objective · T/E combination · penalties · γ / GAE λ ·
normalisation (reference) · step/horizon.

### Vehicular / V2X / VEC

**P1. Cao et al., *Sensors* 22(23):9340, 2022** — VEC, PPO baseline vs NSGA-II (MOV).
<https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9736212/fullTextXML>
- Reward: `reward = 10/(10 + λ·t + μ·e)` (weighted share inside a saturating reciprocal), λ+μ=1.
- T/E: fixed-weight weighted sum inside a reciprocal; λ/μ = 0.5/0.5, 0.8/0.2, 0.2/0.8.
- Penalties: none.
- γ / GAE λ: **not stated**.
- Normalisation: reward self-normalises via the `10/(10+·)` saturating form; MOV's non-RL QoS is
  `QoS_zn = λ_n·(t_z^max − t_z^n)/(t_z^max − t_z^min) + μ_n·(E_z^max − E_z^n)/(E_z^max − E_z^min)` — **min–max over
  pure strategies**, *"where t_z^max and E_z^max are the maximal latency and maximal energy consumption with
  strategy z."*
- Horizon: per-RSU per-slot; 100 slots; 5/7/10 vehicles in the comparison (50/60/70 elsewhere). `[PART]`

**P2. Sun & He, *Electronics* 12(6):1304, 2023 (HDMAPPO)** — MEC with energy harvesting.
<https://mdpi-res.com/d_attachment/electronics/electronics-12-01304/article_deploy/electronics-12-01304.pdf>
- Reward: Eq. (29) — `−penalt` on {deadline miss OR active discard}, else `−(αD + βE + φ·dropout)`.
- T/E: fixed-weight weighted sum (α, β unreported); dropout is a third weighted term.
- Penalties: single constant `−penalt` on the violation branch; **no episode termination**; `penalt` unreported.
- γ = **0.99**; GAE λ symbolic only (no number).
- Normalisation: **none**.
- Horizon: per-slot decision per UE; episode = T slots; T_max = 1 s. `[FT]`/`[PART]`

**P3. Yang et al., WiOpt 2023 (arXiv:2307.14346)** — MEC MORL-PPO, multi-policy.
<https://arxiv.org/html/2307.14346v1>
- Reward: `r_ω = ω^T×(α_T r_T, α_E r_E)` with `r_T = −(T̂ + ΣΔT̂)`, `r_E = −Ê`.
- T/E: preference-vector weighted sum over a **vector reward**; one network per preference (50 prefs @ 0.02).
- Penalties: **none**.
- γ = **0.9**; GAE λ = **0.95**; clip 0.2.
- Normalisation: none stated (α_T/α_E are magnitude-alignment coefficients, values unreported).
- Horizon: per-step sequential, one task per step. `[FT]`

**P4. Yang et al., IEEE TSC 2025 (arXiv:2509.10474) — GMORL** — MEC, Discrete-SAC, single preference-conditioned.
<https://arxiv.org/html/2509.10474v1>
- Reward: `r_ω = ω^T×(α_T r_T, α_E r_E)`; vector `R: S×C×A → ℝ²`.
- T/E: preference-conditioned scalarisation; single policy generalises across ω and across server counts.
- Penalties: **none** (action masking instead).
- γ = **0.95**; no GAE (SAC).
- Normalisation: action masking + softmax; **no reward normalisation**. Hypervolume reference = max delay & energy
  across all Pareto fronts.
- Horizon: episodic, T = 100 steps × Δt = 1 s, one task per step. `[FT]`

**P5. *DRLO-VANET*, Scientific Reports 2026** — VANET, DQN + SAC, NS-3/ns3-gym.
<https://www.nature.com/articles/s41598-026-46336-w>
- Objective (3): `π* = argmin_π E[αL_t + βE_t + γP_miss]`;
  Reward (5): `R_t = −(α·L̃_t + β·Ẽ_t + γ·1{L_t > DL_t} + ν·H̃_t)`.
- T/E: fixed-weight weighted sum. Weights α, β, γ, ν **not numerically printed**.
- Penalties: **deadline miss** as a hard indicator `1{L_t > DL_t}`; **handover overhead** `H̃`; both weighted.
- γ / GAE λ: `ζ ∈ (0,1)` symbolic, **value not stated**; no GAE (off-policy).
- Normalisation (6): **running min–max per component** —
  `X̃_t = (X_t − X_min)/(X_max − X_min + ε)`, individually for `X ∈ {L, E, H}`; *"latency, energy, and handover
  costs normalized to [0,1]"*.
- Horizon: per-slot decision epoch, one task per vehicle per epoch. `[FT]`/`[PART]`

**P6. Hevesli et al., arXiv:2503.03391 (MAPPO-BD / JUTQORA)** — hierarchical air-ground MEC.
<https://arxiv.org/html/2503.03391v1>
- IoTD reward (46): `r_n(t) = −(E^{com2}_n(t) + ω·Σ_m β_{n,m}(t)E^{traj}_m(t) + p_n(t))`.
- Delay violations (47): `p_n(t) = ψ₁·ReLU(Q̄^l_n − Q^l_max) + ψ₂·ReLU(Q̄^o_{n,m} − Q^o_max)`.
- UAV reward (48): `r_m(t) = −(β_{n,m}Σ_n E^{com2}_n + ωE^{traj}_m + p_m(t)) + r^guide_m(t)`,
  with `p_m = p^delay + p^flyout + p^collision` (49):
  `p^delay = ψ₃β_{n,m}Σ_n ReLU(Q̄^e_{n,m} − Q^e_max)` (50);
  `p^flyout = μ_o|q_m − clip(q_m,0,W)|` (51);
  `p^collision = μ_c Σ_j min((|q_m − q_j| − d_min)/d_min, 0)` (52);
  guidance `r^guide_m = μ_g·U_cov(q_m)/N^max_m` (53).
- T/E: **energy-dominated objective with ReLU-gated hard-ish queue-delay penalties**; latency is *not* additively
  traded, it is enforced as a constraint-shaped penalty.
- Penalty coefficients: ψ₁, ψ₂, ψ₃, μ_o, μ_c, μ_g, ω are **symbolic only (no numbers printed)**.
- γ = 0.99 (Table: `Discount factor of IoTD, UAV, and HAPS agents γu 0.99`); GAE λ symbolic only.
- Normalisation: none; queue maxima `Q_max` are the penalty reference.
- Horizon: per-slot; episode length `I` (numeric not stated). `[FT]`/`[PART]`

**P7. Chen & Tong, *Future Internet* 17(12):542, 2025 (D-MAPPO)** — SAGIN offloading.
<https://mdpi-res.com/d_attachment/futureinternet/futureinternet-17-00542/article_deploy/futureinternet-17-00542-v2.pdf>
- Reward (32), piecewise:
  `−[(1−k)E + kT]` if `δ_ur=1 ∧ δ_n=1`;
  `−[(1−k)E + kT] − ι(1−δ_n)N_n` if `δ_ur=1 ∧ δ_n≠1`;
  `−[(1−k)E + kT] − ι(1−δ_n)N_n − ι(1−δ_ur)N_ur` if both fail.
- T/E: **fixed-weight weighted sum with k = 0.5** (weights sum to 1), energy weighted `(1−k)`.
- Penalties: **per-uninstalled/failed task**, **ι = 7** (`ι` = penalty coefficient, `(1−δ)N` = count of failures);
  separate counts for normal and urgent task classes.
- γ = **0.95**; GAE λ = **0.95**; clip 0.2; rollout length 75 (Table 3).
- Normalisation: none for the reward; timeout tasks get fixed delay/energy values.
- Horizon: per-time-slot control (`for timestep = 1 to Max_Timesteps`), offloading-ratio vector per slot. `[FT]`

**P8. arXiv:2312.01499 (Qin, Lu, Chen, Chong, Wu)** — user-centric MEC, IPPO + MAPPO.
<https://arxiv.org/html/2312.01499v1>
- IPPO reward: `r_m(t) = −T_m(t) + κ^c(τ_c − T_m(t))`, **κ^c = 0.6**.
- MAPPO reward: `r(t) = −(1/M)Σ_m T_m(t) + κ^n (1/M)Σ_m (τ_c − T_m(t))`, **κ^n = 0.8**.
- T/E: **latency-only** (no energy term); the second term is a deadline-slack penalty.
- Penalties: soft deadline penalty with a **positive** coefficient on remaining slack; τ_c = max tolerable delay.
- γ = **0.99**, GAE λ = **0.95**, clip 0.2. **Explicitly stated in one sentence.**
- Normalisation: none reported.
- Horizon: per-time-slot; *"For each episode, we assume the episode length T is 300."* `[FT]`

**P9. Li et al., arXiv:2305.01536 (FlexEdge)** — UAV-aided VEC, PPO.
<https://ar5iv.labs.arxiv.org/html/2305.01536>
- Reward: `r_n = Σ_k(E^u_k[n] + E^{rc}_k[n]) + E^f[n] + P^l_n`,
  `P^l_n = (μ/K)Σ_k max{T^l_k[n] − t_k[n], T^e_k[n] − t_k[n], 0}` (linear latency-violation penalty).
- T/E: **energy cost + linear soft-constraint penalty**; no additive T/E trade-off weight.
- Penalty coefficient: **μ = 100**.
- γ = **0.95**; GAE formula given, λ not stated.
- Normalisation: none reported.
- Horizon: per-slot; *"the length of an episode is equal to N"* (N unreported). `[FT]`/`[PART]`

**P10. Paknejad et al., arXiv:2507.09341** — VEC real-time offloading, PPO + DQN.
- Table III: `DQN discount factor 0.9`, `PPO discount factor 0.95`, **`Number of tasks per vehicle 1`**.
- Structurally the closest thing to a **single-shot** offloading decision in the verified set — **and PPO still
  uses γ = 0.95, not 1.** GAE λ not stated. `[FT]`

**P11. arXiv:2605.18437 (Huang, Luo, Wang) — FedMAGS** — **DAG** offloading in VEC, PPO inner + meta outer (MAML),
federated. <https://arxiv.org/html/2605.18437v1> · <https://ar5iv.labs.arxiv.org/html/2605.18437>
- Reward (24): **incremental completion-time reward only** —
  `r_t = CE(t_j) − CE(t_i)` if subtask `t_j` follows `t_i`, else `−CE(t_i)` for the first subtask;
  return (25) `G = Σ_{t=1}^{T} γ^{t−1} r_t`.
- T/E: **latency-only, no energy term at all**; no weighted trade-off.
- Penalties: **none**.
- γ / GAE λ: MDP tuple includes γ but **no numeric value is printed**; GAE `Â_t` is used in the clipped PPO
  surrogate (26) but **λ is not printed**. Clip ε symbolic.
- Normalisation: none reported.
- Horizon: per-subtask sequential over a DAG; **n = 20 subtasks** per DAG, four topologies crossing
  density ∈ {0.7, 0.9} and fat ∈ {0.4, 0.6}. Structurally the closest match to a "20-action plan per episode".
  `[FT]`/`[PART]`

**P12. Gholipour et al., arXiv:2312.11739 (TPTO)** — Transformer-PPO, **DAG** offloading, IEEE ICPADS 2023.
- Param table: `Clip ratio 0.2`, `Discount Factor 0.99`, `Entropy coefficient 0.5`, lr 0.1, batch 100.
- Reward: `ΔALO_i = ALO_i − ALO_{i−1}` (negative latency increment).
- T/E: latency-only. Penalties: none. GAE λ symbolic only (**not stated numerically**).
- Horizon: **per-task sequential over an application DAG** — the offloading goal is a plan
  `O_n = (o_1, …, o_n)`, binary action per task. `[FT]`

**P12b. Jiang, Jin, Xin, Chen & Guo, *Systems* 14(9):1103, 2026 (FRMPPO)** — **DAG-task VEC scheduling**, fuzzy
reward multi-agent PPO. DOI [10.3390/systems14091103](https://doi.org/10.3390/systems14091103) —
<https://mdpi-res.com/d_attachment/systems/systems-14-01103/article_deploy/systems-14-01103.pdf>. `[FT]`
- Objective: *"minimizing task completion latency and vehicle energy consumption"*; system integrates a
  **directed acyclic graph task model**, dynamic communication model, and computation model.
- Reward: a **fuzzy reward mechanism** that takes edge-node load pressure and communication state as inputs and
  *"adaptively combine[s] local immediate rewards and the global reward"* — i.e. a *learned/adaptive* local–global
  fusion rather than a fixed α:β weight. Reward weight α is a configured parameter (Table 4).
- T/E: latency **and** energy in the objective, combined through the adaptive fuzzy fusion rather than a fixed
  scalarisation. (The exact fusion expression is in the binary-extracted PDF text and was not cleanly recoverable —
  flagged rather than guessed.)
- Penalties: none reported as a separate branch.
- **γ = 0.95** chosen from the grid `{0.90, 0.91, 0.93, 0.95, 0.97, 0.99}`; **GAE λ = 0.95**; clip ϵ = 0.20;
  `Lmax = 500`; learning rate `ηθ = ηϕ = 0.0005` (Table 4 + Fig. 4).
- Normalisation: none reported for the reward; local-vs-global reward fusion is the mechanism instead.
- Horizon: episode-based training over per-slot DAG subtask scheduling; 20 vehicles / 5 RSUs in the parameter study.
- **Why it matters for this project:** it is simultaneously (i) DAG-based, (ii) VEC, (iii) latency+energy, (iv)
  MAPPO-family, and (v) the only paper whose γ grid contains 0.93 — yet it selects **0.95** and explicitly rejects
  both extremes. It is the single best citable precedent for the γ discussion.

### MEC / UAV-MEC / other

**P13. Sun et al., arXiv:2501.06410 (EMODRL)** — UAV-MEC, evolutionary multi-objective DRL with PPO.
<https://arxiv.org/html/2501.06410v1>
- Reward (23): `r_t = −W` if `S_{t+1} = Ω` (invalid state set), else the **vector** `(−D_t, −E_t)`.
- T/E: **vector reward** with evolutionary weight adjustment to obtain non-dominated policies; violated
  constraints collapse the step to a large scalar penalty.
- Penalties: **`−W`, "a sufficiently large and reasonable positive number"** — numeric value not printed. No
  separate constraint branch coefficients.
- γ: vector-valued `γ = [γ_1, …, γ_m]` with `γ_i ∈ [0,1]`, **no numeric values**; no GAE.
- Normalisation: none reported; invalid states → `Ω`.
- Horizon: per-slot with simulated-annealing action-space reduction. `[FT]`/`[PART]`

**P14. Liu et al., arXiv:2406.06986 (MAD2RL)** — VEC DNN partitioning + offloading, **Lyapunov-guided** diffusion
MARL. <https://arxiv.org/html/2406.06986v1>
- Objective: minimise DNN task completion time subject to system stability.
- Lyapunov drift-plus-penalty (31): `Λ(Q(t)) = Δ(Q(t)) + V·𝔼[Σ_{i∈I} d_i(t) | Q(t)]`.
- Reward (57): `r(t) = −Σ_i Q^loc_i(t)[…−f^loc_i τ] − Σ_k Q^rsu_k(t)[…−f^rsu_k(t)τ] − Σ_{j≠1} Q^veh_j(t)[…−f^veh_j τ] − V·Σ_i d_i(t)`.
- T/E: **neither a weighted sum nor a constraint on T/E** — the reward *is* the drift-plus-penalty expression;
  virtual queues encode the constraint backlogs, `−V·Σd_i` is the penalty on completion time.
- Penalties: virtual-queue backlog terms (implicit multipliers, no separate coefficient); `V` trades cost vs
  stability (**numeric V not printed**). Keyword `virtual queue` / `energy budget` / `battery` → 0 hits in the HTML.
- γ: `ω` is the discount factor for future rewards (target Q), **no numeric value**; DQN-family (mixing network).
- Normalisation: none reported.
- Horizon: per-slot. `[FT]`/`[PART]`

**P15. Zhan et al., arXiv:2511.05789 (Ly-DTMPPO)** — ISAC-enabled IoV, digital twin, MAPPO.
<https://ar5iv.labs.arxiv.org/html/2511.05789>
- Reward (44): `R(t) = −{ V·∁(t) + Σ_{C_i} Q^loc_i(t)(λ^loc_i(t) − C·CV_i^{cpu}τ) + Σ_{R_k} Q^rsu_k(t)(Σ λ^rsu_i(t) − F^{cpu}_{RSU_k}(t)τ) + Σ V_i(t)(E_total(t) − E_max) }`.
- T/E: **Lagrangian / Lyapunov drift-plus-penalty** — cost×V plus virtual-queue-weighted constraint terms.
- Penalties: local and RSU queue backlogs; **energy-budget term `V_i(t)(E_total − E_max)`**. **V = 5** default
  (swept 5/10/50/100; *"reaching 0.552 at V=50 and 0.781 at V=100"*).
- γ = **0.99**, GAE λ = **0.95**, clip 0.2, 10 update epochs.
- Normalisation: no min–max; C4 pins bandwidth to "the normalized unit bandwidth"; `E_max` is the budget reference.
- Horizon: per-slot; **30 slots/episode, max 1800 episodes**, τ = 1 s.
- Reported multiplier sensitivity: *"an excessively large V prioritizes immediate energy minimization but neglects
  delay control, while an overly small V may lead to aggressive queue draining at the expense of energy efficiency."* `[FT]`

**P16. Li et al., EURASIP JWCN 2021:56 (DRL-E2D)** — multi-eNB MEC.
DOI [10.1186/s13638-021-01941-3](https://doi.org/10.1186/s13638-021-01941-3)
- Objective (16a): `max R(τ) = U(τ) − E(τ) − P(τ)`; components (12)–(15) as in Claim 5 above.
- T/E: **weighted sum** (implicit unit weights), plus a dropped-task penalty.
- Penalties: utility gating `u = 0` on deadline miss + `P(τ) = Σ dᵢ(τ)` (count of dropped tasks), implicit
  coefficient 1.
- γ: **no numeric value**; **no GAE**; **no normalisation reported**.
- Horizon: per-slot; `T_DL = 3·T_slot`, `T_slot = 1 s`, `N_slot = 1000` slots/episode. `[FT]`/`[PART]`

**P17. *QoE-Driven Multi-Task Offloading for Semantic-Aware Edge Computing*, arXiv:2407.11018** — MAPPO.
<https://ar5iv.labs.arxiv.org/html/2407.11018>
- Reward (12): `r_n = { QoE_n if success; t_max − t_n if t-max violated; E_n^max − E_n^M if e-max violated;
  ε_n^M − ε_min if a-max violated }`; `r_t = Σ_n r_n`; `R_t(τ) = Σ_{t'=t}^{T} γ^{t'−t} r_{t'}` (13).
- QoE (10): `QoE_n = Σ_{q_n} ( ω_t/(1+e^{−λ(t^l_n − t^M_n)}) + ω_e/(1+e^{−β(E^l_n − E^M_n)}) + ω_a/(1+e^{−η(ε^M_n − ε^l_n)}) )`.
- T/E: weighted sum `ω_t, ω_e, ω_a` (sum to 1) inside **sigmoids relative to local execution**.
- Penalties: latency `t_max − t_n`; **energy-budget violation `E_n^max − E_n^M`**; accuracy `ε_n^M − ε_min`.
  All coefficient 1; **numeric values not printed**. No dropout/queue-overflow penalty.
- γ = **0.99**; GAE λ = **0.95**; clip ε = 0.2; entropy weight b2 = 0.1.
- Normalisation: **local execution** — `t^l_n = l_n^U/c_n`, `E^l_n`, `ε^l_n`; *"The significant performance
  variations observed across task categories underscore the necessity of employing locally normalized QoE metrics
  for fair evaluation."*
- Horizon: per-timestep over task queues (Q_n = 20); 300 train / 200 test episodes.
- **Reported collapse:** *"A significant jump occurs at (0.8, 0.2) when the weight for accuracy increases, as tasks
  are executed locally, causing a substantial increase in energy consumption and a corresponding rise in accuracy."*
  Balanced point is 1/3 each, corners (1,0,0)/(0,1,0)/(0,0,1), pairs (0.8,0.2) and (0.3,0.7). `[FT]`

**P18. Zulfiqar, Mirza, Qureshi, arXiv:2602.18797 (CADDTO-PPO)** — carbon-aware MIMO-MEC, MAPPO.
<https://ar5iv.labs.arxiv.org/html/2602.18797>
- Reward (26): `r_u(t) = −( w₁·B_u(t)/B_max + w₂·ς_u(t) + w₃·(BO_u(t) + EW_u(t)) )`.
- T/E: negative weighted sum, **w = 1/3 each**; delay via normalised buffer, energy via carbon + wastage.
- Penalties: queue overflow `BO_u(t) = max(0, RB_u(t) − B_max(t))` (10) and energy wastage `EW_u`, both under w₃.
- γ = **0.99**, GAE λ = **0.95**, clip 0.2, entropy 0.01, n_steps 2048.
- Normalisation: `B̄_u = B_u/B_max` (**buffer capacity**); `ψ̄_u = clip(ψ_u/ψ_target, 0, 1)` (**target SINR**).
- Horizon: per-slot; 13,000 episodes × T_max = 100. `[FT]`

**P19. Lan et al., arXiv:2404.15278 (IEEE TMC 2024)** — security-sensitive satellite-terrestrial offloading, PPO.
<https://ar5iv.labs.arxiv.org/html/2404.15278>
- Objective (14): `SP1: min (T_total + β₁E_total + β₂A_total)`; `r(τ) = −(T_total(τ) + β₁E_total(τ) + β₂A_total(τ))`.
- T/E: **weighted sum with fixed weights**; `A_total` is an attack-risk term.
- Penalties: reliability is a **constraint** `r_total(τ) ≥ ρ` with **ρ = 70%**; **β₁ = β₂ = 1**.
- γ = **0.99**, GAE λ = **0.95**, clip 0.2; 5×10⁵ timesteps, update interval 5, batch 64.
- Normalisation: **none reported** (`normaliz` absent) — mixed units are summed raw.
- Horizon: per-scheduling-period; episode length T not stated. `[FT]`/`[PART]`

**P20. Chen & Wang, arXiv:1812.07394** — multi-user MEC, DDPG + DQN.
<https://ar5iv.labs.arxiv.org/html/1812.07394>
- Reward (24)+(29): `r_{m,t} = −10w_m·(p_{l,m}(t) + p_{o,m}(t)) − (1 − w_m)·B_m(t)`.
- T/E: negated weighted sum; **queue length as a delay proxy** (Little's theorem).
- Penalties: none — the delay term *is* the queue-length penalty with weight `(1−w_m)`.
- γ: symbolic only (`R_t = Σ γ^{i−t} r`), **no numeric γ**; no λ (off-policy).
- Normalisation: **none** (raw watts and raw bits) — which is exactly why the `10·w_m` scale factor exists.
- Horizon: per-slot; `K_max = 2000` episodes × `T_max = 200` steps.
- Weight sweep: `w₁ = 0.5` and `w₁ = 0.8` (in the (24) parameterisation: (5, 0.5) and (8, 0.2)). At 0.8:
  *"slightly compromises the buffering delay to achieve the lowest energy consumption."* **No mirrored
  energy-sensitive 0.2/0.8 point and no "balanced 0.5/0.5" label.** `[FT]`

**P21. Li, Jia, He, Guo, Wu, arXiv:2507.05722** — hierarchical SAC, UAV-assisted VEC.
<https://ar5iv.labs.arxiv.org/html/2507.05722>
- Objective (23a): `max ω₁R_succ − ω₂β_T Σ T_i^total(t) − ω₃β_E E_sys(t)`; reward (26) identical.
- T/E: weighted sum with **ω = 0.6 / 0.2 / 0.2** (success / delay / energy) and **per-term normalisers β_T, β_E**.
- Penalties: no reward penalty; `R_succ = (1/|I|)Σ 1{T_i^total ≤ T_i^max}` supplies the deadline incentive; hard
  budgets `Σ_t E^fly ≤ E^max`, `Σ_t E^hover ≤ E^max`.
- γ / GAE λ: **not stated** (SAC, auto-temperature α symbolic). β_T, β_E numeric values **not printed**.
- Normalisation: β_T/β_E "normalization parameters"; `R_succ ∈ [0,1]`; deadline `T_i^max ∈ [0.1, 1] s` used only
  inside the indicator.
- Horizon: per-slot, two-layer hierarchy; episode length not stated. `[FT]`/`[PART]`

**P22. Shi, Zhang, Loo, Huang, Wang, arXiv:2505.04272 (IEEE TII 2025)** — MEC offloading + channel allocation, D3QN.
<https://ar5iv.labs.arxiv.org/html/2505.04272>
- Cost (11): `cost_{n,i} = ω·Σ_{m'} d_{n,i,m'} + (1−ω)·Σ_{m'} e_{n,i,m'}`; reward (18):
  `r_n[t] = tanh(gain_cost) = tanh(cost_{n,i,0} − cost_{n,i,m'})`.
- T/E: weighted-sum cost with **ω = 0.5**, turned into a reward via a **tanh-scaled gain versus local execution**.
- Penalties: none.
- γ = **0.99**, lr 3e-4; no λ (DQN family).
- Normalisation: **local execution + tanh** — the cost gain of offloading relative to local (`m' = 0`), then `tanh`.
- Horizon: per-slot; "each episode means the completion of a mobile application"; total slots not printed. `[FT]`/`[PART]`

**P23. Ebrahimi & Afghah, arXiv:2501.06242** — MEC 5G, PPO.
<https://ar5iv.labs.arxiv.org/html/2501.06242>
- Objective (6): `R = Σ_{t=1}^{∞} γ^t Σ_j w_s r_{j,t}`.
- URLLC reward (8): `r^{URLLC} = 2/(1 + e^{−δ·r_u}) − 1`, with
  `r_u = α·(t^local_process − t_exe)/t^local_process + β·(τ − t_exe)/τ`.
- mMTC reward (9): same sigmoid shell with
  `r_m = α·(t^local_process − t_exe)/t^local_process + β·(E^local_process − E_exe)/E^local_process`.
- T/E: **weighted sum inside a sigmoid squash**; separate URLLC (delay-only) and mMTC (delay+energy) rewards.
- Penalties: the deadline enters *inside* the reward as `β(τ − t_exe)/τ`, going negative after violation.
  **α = 0.5, β = 0.5, δ = 3**.
- γ = **0.99** (Table IV); λ not stated.
- Normalisation: **task deadline τ** and **local execution time/energy** — the only paper in the set whose
  normaliser is explicitly the deadline.
- Horizon: per-task/per-timestep; 15,000 episodes (length not stated). `[FT]`

**P24. Ji, Qin, Tao, Zhu, arXiv:2301.08376 (IEEE TWC 2024)** — semantic-aware offloading, MAPPO.
<https://ar5iv.labs.arxiv.org/html/2301.08376>
- Cost (9): `Cost_i = β·E_i/E^Q_0 + (1−β)·t_i/t^Q_0`; QoE (10): `QoE_i = ε^Q_i/(ε^Q_0·Cost_i)`.
- Reward (13): `r_i = { QoE_i if success; t_max − t_i if (c6) fails; ε_i − ε_min if (c1) fails }`;
  terminal (14): `r_T = { ξ_S(T − t_0) if t_0 ≤ T; −ξ_F Σ_{i∈I} d_i(T)E^Q_0 otherwise }`.
- T/E: single preference coefficient β (energy weight; latency 1−β) inverted into a reciprocal share.
- Penalties: latency violation `t_max − t_i`; accuracy-floor `ε_i − ε_min`; terminal failure
  `−ξ_F Σ d_i(T)E^Q_0`. ξ_S, ξ_F are *"constant values"* — **no numeric values printed**. Piece coefficient 1.
- γ = **0.95**; GAE λ = **0.98**; clip 0.3; c^CR = 0.5, c^E = 0.005.
- Normalisation: **local execution** — `E^Q_0` and `t^Q_0`; *"we can measure the QoE for different UEs fairly"*,
  *"we standardize the task QoE with the tasks executed locally."*
- Horizon: per-timestep over a task queue; 1000 train / 100 test episodes, batch 256; ~600 episodes to converge.
- Weight handling: fixed `β^f = 0.5`; battery-adaptive `β^d = 1 − B/(2B_max)` (28).
- **Reported failure mode:** sparse reward → policy degenerates —
  *"If a UE can hardly receive a positive reward, which is called the sparse reward issue, it cannot learn from a
  good policy"*; *"the DDQN algorithm cannot be well-trained due to the sparse reward issue, with even worth QoE
  performance compared to the local execution scheme."* `[FT]`

**P25. Li, Liu, Xie, Zhang, Zhang, arXiv:2310.17470 (IEEE TGCN 2023)** — DT UAV ISCC, ATB-MAPPO.
<https://ar5iv.labs.arxiv.org/html/2310.17470>
- Reward (32): `r_t^k = −Ē^ω_k[n]·P^k_{T,t}`; deadline penalty (33)–(34):
  `P^k_{t,T} = 𝒫(Σ_m α_{k,m}·max{t^l_k[n], t^o_k[n]+t^e_k[n]}, t^max_k[n], t^max_k[n])`,
  `𝒫(x,a,b) = 2 − exp(−⌈(x−a)/b⌉⁺)`.
- T/E: **negated product** of weighted energy × a bounded deadline penalty — minimises energy subject to a delay
  penalty *multiplier*, i.e. neither an additive sum nor a hard constraint.
- Penalties: deadline violation penalty bounded in (1,2]; **μ_o = μ_t = 0.1**, **ω = 0.001**; reward clipped to
  **[−5, 5]**.
- γ = **0.98** (`γ_u = 0.98`); λ appears symbolically in `Â = Σ(γλ)^l(…)` — **no numeric λ**.
- Normalisation: *"value normalization is used and the reward is clipped into [−5, 5]"*; the deadline is the
  penalty reference `b`.
- Horizon: per-slot; `M_e = 300` episodes × `E_l = 200` steps.
- Weight sweep: `ω_s` from 0.2 to 0.6 and >0.6; no collapse reported. `[FT]`

**P26. Wang & Zhou, *CMC* 2025** — smart-grid PPO offloading. DOI
[10.32604/cmc.2025.065465](https://doi.org/10.32604/cmc.2025.065465) —
<https://file.techscience.com/files/cmc/2025/online/CMC0612/TSP_CMC_65465/TSP_CMC_65465.pdf>
- **The one in-domain γ = 1 argument found.** Verbatim: *"For γ = 1, the cumulative reward represents the sum of
  all delays and energy costs. Thus, finding the optimal strategy π* aligns with minimizing the original objective
  function."*
- **But it does not train at γ = 1.** Table 3: `Discount factor γ 0.99`, `Adv. discount factor φ 0.95`,
  clip range 0.1, entropy 0.05, lr 0.0003.
- GAE λ = **0.95**. Horizon: per-time-slot, time step 0.01 s; episode length not stated. `[FT]`/`[PART]`

**P27. Dai, Zhang, Maharjan, Zhang, arXiv:2011.08442** — edge intelligence, 5G beyond, DDPG.
- Uses **ε (not γ)** as the discount symbol. Verbatim: *"To evaluate the impact of discount factors, we set ε from
  0.5 to 0.7 … when ε = 0.6, the normalized cumulative reward is clearly higher than the cases when ε = 0.65 and
  ε = 0.7 … Thus, we can conclude that ε = 0.6 is the best discount factor for the proposed algorithm."*
- **A downward sweep that selects 0.6** — direct evidence against a field-wide drift toward γ → 1.
- Horizon: 6000 episodes × 20 steps. `[FT]`

**P28. Jiang, Tavakkolnia, Han, *npj Wireless Technology* 2:64, 2026** — THz cell-free MEC, MAPPO.
DOI [10.1038/s44459-026-00072-9](https://doi.org/10.1038/s44459-026-00072-9)
- Table 1: `The discount factor, γ 0.9`, `The GAE parameter, λ 0.9`, `ϵ 0.2`, `E 2000` episodes, `N 40` slots.
- Discusses the γ → 1 limit explicitly but does **not** adopt it: *"γ → 1 means future rewards are emphasized
  equally with immediate rewards, while γ → 0 means immediate rewards dominate."* `[FT]`

**P29. *"Maritime Mobile Edge Computing for Sporadic Tasks: A PPO-Based Dynamic Offloading Strategy"* (ON-PPO),
*Mathematics* 13(16):2643, 2025** — the only offloading paper found reporting outright non-convergence at high γ.
DOI [10.3390/math13162643](https://doi.org/10.3390/math13162643) —
<https://mdpi-res.com/d_attachment/mathematics/mathematics-13-02643/article_deploy/mathematics-13-02643-v2.pdf>. `[FT]`
- γ ablation verbatim: *"When gamma is 0.5, the training outcome is the worst as the agent focuses excessively on
  immediate rewards. In contrast, when gamma is 0.99 or 0.999, the agent's overemphasis on long-term rewards results
  in poor algorithm performance and a failure to converge. Results show that a gamma of 0.9 balances convergence
  speed and average performance well, so it is used as the default gamma for later experiments."*
- Table 2: `Discount rate γ 0.9`, `Number of episode T 1000`. GAE λ not stated.
- Uses per-episode training over sporadic-task MEC offloading. `[FT]`

---

## PART 3 — Synthesis tables

### Table A — Formulation families: advantages and failure modes

| Formulation | Papers (verified) | Advantages | Failure modes |
|---|---|---|---|
| **Fixed weighted sum** `r = −(αT + βE)` (α+β=1 or unnormalised) | **DRL-E2D** (JWCN 2021:56) `U−E−P`; **EE-A2C** (IEEE Access 2024) `αω_f+βω_e+ηω_d`; **P7** Future Internet `−[(1−k)E+kT]` k=0.5; **P19** TMC `−(T+β₁E+β₂A)` β=1; **P22** TII `ωd+(1−ω)e` ω=0.5; **P20** `−10w(p_l+p_o)−(1−w)B`; **P21** `ω₁R_succ−ω₂β_T T−ω₃β_E E` ω=0.6/0.2/0.2; **P18** `−(w₁B/B_max+w₂ς+w₃(BO+EW))` 1/3 each; **P13** vector `(−D,−E)` | Simple, well-understood; single scalar critic; easy to implement; works when a defensible relative scale exists | **Cannot express a hard deadline** (a huge deadline overrun can be offset by a small energy gain); **weight sensitivity** — one weight pair is one point on the Pareto front, so results are only valid for that pair; **reward-scale mismatch** — T in seconds vs E in joules differ by orders of magnitude, so α/β silently encode the unit conversion unless coefficients are explicitly used for it (this is exactly why GMORL introduces α_T/α_E "to adjust … to the same order of magnitude"); **policy collapse** to the dominant term (verified: P17 collapses to local execution at (0.8, 0.2)) |
| **Preference-conditioned scalarisation / MORL** (ω as policy input or one policy per ω) | **P4** TSC 2025 GMORL (single policy, ω input, Discrete-SAC); **P3** WiOpt 2023 (multi-policy, 50 prefs @0.02); **P13** EMODRL (evolutionary weight adjustment); **P6/others** QoE weighted-sigmoid families | Handles **unknown / diverse preferences** without committing to one α:β; produces a Pareto front in one training run (P4) or a cheap set of policies (P3); avoids re-training per deployment | Multi-policy variants cost O(#preferences) in compute/storage (P4 explicitly lists this as a limitation); single-policy variants trade peak performance for generality (P4's own Pareto front is 0.3 % below its multi-policy upper bound); **hybridisation caveat**: P3 is *called* MORL but the actor never sees ω — only the critic does, so it is not preference-conditioned in the modern sense |
| **Hard-constraint / feasibility-gated reward** (violation branch) | **P2** HDMAPPO `−penalt`; **P16** DRL-E2D `u=0` gating + drop count; **P9** FlexEdge linear penalty μ=100; **P8** deadline-slack penalty κ^c=0.6 / κ^n=0.8; **P17** `t_max−t_n` branch; **P19** reliability ρ=70 % constraint; **P13** `−W` invalid-state branch | Expresses a hard deadline in principle; keeps the constraint visible to the learner; HDMAPPO's two-branch form is the cleanest template | **Penalty magnitude is the real constraint**: a branch to a constant `−penalt` only enforces the deadline if `penalt` dominates every feasible alternative — and HDMAPPO **does not report `penalt` at all**, so the constraint is uncalibrated; `−W`/`ξ_S`/`ξ_F`/`μ_o`/`ψ_i` are likewise unreported in P13/P24/P6; **no episode termination** in the verified set (the violation just replaces one step's reward), so a single infeasible action is not structurally excluded; large penalties risk **gradient domination** and can flatten the reward signal |
| **Lagrangian / Lyapunov drift-plus-penalty** | **P15** `−{V·∁ + ΣQ(·) + ΣV_i(E_total−E_max)}` V=5; **P14** MAD2RL `Δ+V𝔼[Σd_i]` with virtual queues; **(non-DRL)** arXiv:2404.02166 `D = ΔL + V·𝔼{C_s}`; LyDROO arXiv:2010.01370 | The only family that genuinely handles a **long-term / time-average budget** (energy, power) rather than a per-step cap; gives an analytic optimality bound (P15 Thm 8: `≤ C^opt + (WT+C)/V`); constraint satisfaction is asymptotically guaranteed, not hoped for | **V is a hand-tuned trade-off with a documented degeneracy**: *"an excessively large V prioritizes immediate energy minimization but neglects delay control, while an overly small V may lead to aggressive queue draining at the expense of energy efficiency"* (P15); virtual-queue terms make the reward **non-stationary during training** (queues drift); needs a per-slot decision structure — it does **not** map onto a single-shot plan episode |
| **Reciprocal / ratio / saturating reward** | **P1** `10/(10+λt+μe)`; **P24** `QoE = ε/(ε₀·Cost)` with `Cost = βE/E₀+(1−β)t/t₀`; **P23** `2/(1+e^{−δr_u})−1`; **P22** `tanh(Δcost)`; **P9** arXiv:2603.20238 `Σ(l)/(ΣE) − βI` | **Bounded by construction** — kills reward-scale blow-up and is invariant to units in a way a raw weighted sum is not; naturally encodes "diminishing returns"; the bounded range makes advantage normalisation unnecessary | **Weight sensitivity is hidden and sometimes perverse**: in `10/(10+λt+μe)`, λ and μ relative to the constant 10 set the entire operating regime, so a change in the *unit* of t or e changes the policy — the calibration is implicit; **gradient vanishing**: once `λt+μe ≫ 10` the reward is nearly flat, so late-training improvement signals are tiny; ratio forms can be dominated by a near-zero denominator; `QoE = ε/(ε₀·Cost)` explodes as Cost → 0 |
| **Latency-only reward** (with deadline shaping) | **P11** FedMAGS `CE(t_j) − CE(t_i)`; **P12** TPTO `ΔALO`; **P8** `−T + κ(τ_c−T)`; **P10** (PPO, 1 task/vehicle) | **Eliminates the α:β problem entirely** — no arbitrary trade-off to justify or tune; well-matched to a makespan objective; the natural choice when energy is reported separately as a metric rather than optimised | Ignores energy entirely (P11, P12 have **no** energy term); energy outcomes are then emergent, not controlled — fine if energy is a reported metric, **not** fine if there is an energy budget or battery constraint; single-objective optima are often the extreme points the draft is trying to avoid |
| **Lexicographic / constrained-cost** | **not found** in any verified paper | Would express "minimise energy subject to deadline" exactly | Nobody in this literature does it in the DRL reward — the closest verified mechanism is ElasticVR's Lagrangian deadline term (arXiv:2512.12366, λ^penalty=10) |

**Cross-cutting counts.** Of the papers with a resolvable formulation: **9/13** in the wider sweep use an explicit
weighted sum of T and E; **only 1/13** is a true Lagrangian/Lyapunov reward; **3/13** use reciprocal/ratio forms;
**no lexicographic reward was found anywhere**.

### Table B — γ and GAE λ in episodic offloading PPO work

33 full-text-verified papers (`γ` read from a table or body sentence; **zero abstract-only entries**).

| γ | count | examples (algorithm, episode structure) |
|---|---|---|
| **0.99** | 14 | arXiv:2312.01499 (IPPO/MAPPO, per-slot, T=300) · arXiv:2603.20238 (PPO, per-slot, N=100) · arXiv:2404.15278 (PPO, per-period) · arXiv:2407.11018 (MAPPO, per-step, 300 ep) · arXiv:2503.03391 (MAPPO-BD, per-slot) · arXiv:2602.18797 (MAPPO, per-slot, T_max=100) · arXiv:2605.24972 (MARL, 40 steps/ep) · arXiv:2606.26293 (HPPO manager, 3000 ep) · arXiv:2607.09295 (MAPPO) · arXiv:2312.11739 (**TPTO, per-task DAG plan**) · CMC 2025 (smart-grid PPO) · Sensors 25:1428 (PPO+GIN, **episode = one scheduling scenario**) · Electronics 12:1304 (HDMAPPO, per-slot) · Electronics 14(17):3444 (TD3, DT-VEC, 2025) |
| **0.95** | 7 | **Systems 14(9):1103 (FRMPPO, DAG-VEC; γ grid {0.90,0.91,0.93,0.95,0.97,0.99}, selects 0.95)** · Future Internet 17:542 (D-MAPPO, per-slot) · arXiv:2301.08376 (MAPPO) · arXiv:2302.09021 (AB-MAPPO, UAVs; MUs use 0.8) · arXiv:2305.01536 (FlexEdge PPO) · arXiv:2507.09341 (PPO, **1 task/vehicle**) · arXiv:2606.26293 (HPPO worker) |
| **0.9** | 8 | **Mathematics 13(16):2643 (ON-PPO, maritime MEC — "0.99 or 0.999 … failure to converge")** · Electronics 13:2933 (PPO) · arXiv:2412.13676 (REDQ) · arXiv:2311.02525 (QECO, DQN) · arXiv:2502.03403 (PPO) · npj WT 2:64 (MAPPO; λ also 0.9) · arXiv:2507.09341 (DQN arm) · arXiv:2307.14346 (MORL-PPO, WiOpt 2023) |
| 0.995 | 1 | arXiv:2212.05757 (Co-MAPPO, reward only at sub-task completion) |
| 0.98 | 1 | arXiv:2308.12756 (MAPPO, 200 steps/ep) |
| 0.97 | 1 | Electronics 13(12):2387 (M-GNRL; **the 0.93/0.95/0.97/0.99 ablation**, selects 0.97) |
| 0.8 | 1 | arXiv:2302.09021 (MUs, in the same run as UAVs at 0.95) |
| **0.01** | 1 | Drones 9(4):288 (PER-DDPG, UAV-swarm MEC, 2025) — **the lowest γ found anywhere** |
| 0.6 | 2 | arXiv:2011.08442 (DDPG, swept 0.5–0.7 **downward** and selected 0.6; symbol is ε) · Electronics 15:936 (PPO, RIS-VEC) |

**Reading the distribution.** 33 papers contribute 38 γ values, because several papers report different γ per agent
population or per algorithm (arXiv:2302.09021 = 0.95 UAVs / 0.8 MUs; arXiv:2606.26293 = 0.99 manager / 0.95 worker;
arXiv:2507.09341 = 0.95 PPO / 0.9 DQN). Counting the primary value per paper: **0.99 dominates (14), then 0.95 (7)
and 0.9 (7–8)**; the corpus spans **0.01 → 0.995**, so there is no consensus value, only a modal default.
**No paper in this set trains at γ = 1 or 0.999.**

**GAE λ, where stated (14 of 28 state nothing):** **0.95 → 9 papers** (arXiv:2312.01499 · arXiv:2603.20238 ·
arXiv:2404.15278 · arXiv:2407.11018 · arXiv:2602.18797 · arXiv:2605.24972 · arXiv:2607.09295 · CMC 2025 ·
Future Internet 17:542) · **0.98 → 1** (arXiv:2301.08376) · **0.9 → 1** (npj WT 2:64). Plus the separately verified
WiOpt 2023 MORL-PPO at **λ = 0.95**. Papers that give λ only symbolically (no number): HDMAPPO, arXiv:2305.01536,
arXiv:2503.03391, arXiv:2311.02525, arXiv:2312.11739, arXiv:2508.06863, arXiv:2308.12756, arXiv:2507.05722.

**Modal recipe for an offloading PPO paper:** `γ = 0.99, λ = 0.95, clip = 0.2, entropy ≈ 0.01, batch 64–128`.

**Is there any γ = 1 for episodic / single-shot scheduling?** **No — not one.** Specifically:
- Explicitly episodic papers still use γ < 1: arXiv:2508.06863 (*"The task is episodic, meaning that after T time
  slots the POMDP ends. Still, we can select T arbitrarily large to model an infinite horizon problem."*) uses
  **0.99**; Sensors 25:1428 (*"each episode representing a complete scheduling scenario"*) uses **0.99**;
  arXiv:2507.09341 (`Number of tasks per vehicle 1`) uses **0.95**; **TPTO** (arXiv:2312.11739, per-task DAG plan —
  structurally the closest analogue to a single-shot plan) uses **0.99**.
- The literature largely treats γ = 1 as a **warning**: Electronics 13(12):2387 verbatim — *"Conversely, when
  γ = 1, convergence issues may emerge, underscoring the importance of selecting an appropriate discount factor for
  policy training."* **Direct counter-evidence to any "use γ = 1" recommendation is strong and comes from
  offloading papers specifically:** Mathematics 13(16):2643 reports that **γ = 0.999 fails to converge outright**
  (*"when gamma is 0.99 or 0.999, the agent's overemphasis on long-term rewards results in poor algorithm
  performance and a failure to converge"*); Systems 14(9):1103 finds **0.97 and 0.99 "too large"**; Drones 9(4):288
  selects **γ = 0.01**; and arXiv:2011.08442 sweeps γ **downward** (0.5–0.7) and picks 0.6.
- **One in-domain mathematical argument for γ = 1 exists** — CMC 2025 (DOI 10.32604/cmc.2025.065465):
  *"For γ = 1, the cumulative reward represents the sum of all delays and energy costs. Thus, finding the optimal
  strategy π* aligns with minimizing the original objective function."* — **but that paper trains at γ = 0.99**
  (with Adv. discount φ = 0.95). Justification exists; deployment does not.
- The only **actual γ = 1 PPO run** found is out of domain (LIACS bachelor thesis on molecular minimum-energy-path
  discovery, Table 2 `Discount factor γ 1.0` / `GAE λ 0.95`), justified by a *time-independent* objective:
  *"Since the MEP objective itself does not depend on time, a reward function that encodes this objective should
  ideally remain well-defined at γ = 1."* The same thesis warns that *"discounting biases the agent to obtain
  low-energy rewards within a shorter horizon, which leads to heuristic shortcuts."*
- The npj WT paper is the only one to discuss the γ → 1 limit in-domain, and it chooses **0.9**.

**Extraction hazard worth flagging for the draft:** γ is not a stable symbol in this literature. It denotes
**compression ratio** (arXiv:2412.13676), **SINR** (Electronics 15:936), **offloading ratio** (arXiv:2606.26293),
and **antenna spacing** (arXiv:2310.17470); the discount factor is called **ε** (arXiv:2011.08442), **µ_RL**
(arXiv:2607.09295), **ζ** (Scientific Reports DRLO-VANET), **τ** (arXiv:2312.08714) and **ω** (arXiv:2406.06986);
and the hyperparameter row may be labelled **`Gamma`** rather than "discount factor" (arXiv:2404.15278). Any
automated sweep for "γ" over this corpus produces false positives without manual reading.

### Table C — Penalty designs: formulas and coefficients

| Penalty type | Formula (verbatim) | Coefficient | Paper |
|---|---|---|---|
| **Deadline violation (fixed-constant branch, no termination)** | `r_i^t = −penalt` if `D_i^t > T_max or h_i^{a_t} = 0`; else `−(αD_i^t + βE_i^t + φ·dropout^t)` | `penalt` **unreported** | Sun & He, *Electronics* 12(6):1304, 2023 (HDMAPPO) — <https://mdpi-res.com/d_attachment/electronics/electronics-12-01304/article_deploy/electronics-12-01304.pdf> |
| **Deadline violation (linear, unbounded)** | `P^l_n = (μ/K)Σ_k max{T^l_k[n] − t_k[n], T^e_k[n] − t_k[n], 0}` | **μ = 100** | Li et al., arXiv:2305.01536 (FlexEdge) — <https://ar5iv.labs.arxiv.org/html/2305.01536> |
| **Deadline violation (ReLU on queue backlog)** | `p_n(t) = ψ₁·ReLU(Q̄^l_n − Q^l_max) + ψ₂·ReLU(Q̄^o_{n,m} − Q^o_max)` | ψ₁, ψ₂ **unreported** | Hevesli et al., arXiv:2503.03391 — <https://arxiv.org/html/2503.03391v1> |
| **Edge-queue delay violation (ReLU)** | `p^delay(t) = ψ₃·β_{n,m}Σ_n ReLU(Q̄^e_{n,m} − Q^e_max)` | ψ₃ **unreported** | Hevesli et al., arXiv:2503.03391 |
| **Deadline violation (smooth exponential, bounded)** | `P(x,a,b) = 2 − exp(−⌈(x−a)/b⌉⁺)` with `a = b = t^max_k[n]`; reward `r = −Ē^ω·P` | **μ_o = μ_t = 0.1, ω = 0.001**, reward clipped **[−5, 5]** | Li et al., arXiv:2310.17470 (IEEE TGCN 2023) — <https://ar5iv.labs.arxiv.org/html/2310.17470> |
| **Deadline indicator (hard 0/1 multiplicity in mixed units)** | `R_t = −(α·L̃_t + β·Ẽ_t + γ·1{L_t > DL_t} + ν·H̃_t)` | α, β, γ, ν **unreported** | *DRLO-VANET*, Sci. Rep. 2026 — <https://www.nature.com/articles/s41598-026-46336-w> |
| **Deadline-slack penalty (positive on remaining slack)** | `r_m(t) = −T_m(t) + κ^c(τ_c − T_m(t))`; global `r(t) = −(1/M)ΣT_m + κ^n(1/M)Σ(τ_c − T_m(t))` | **κ^c = 0.6 (IPPO), κ^n = 0.8 (MAPPO)** | Qin et al., arXiv:2312.01499 — <https://arxiv.org/html/2312.01499v1> |
| **Deadline gating + dropped-task count** | `u(tⱼ) = u if T(tⱼ) ≤ T_DL else 0`; `P(τ) = Σⁿᵢ₌₀ dᵢ(τ)`; `max R = U − E − P` | `u = 1`, penalty coefficient **implicit 1**; `T_DL = 3·T_slot` | Li et al., *EURASIP JWCN* 2021:56 (DRL-E2D) — DOI [10.1186/s13638-021-01941-3](https://doi.org/10.1186/s13638-021-01941-3) |
| **Deadline-violation branch inside a multi-objective reward** | `r_n = t_max − t_n` if the max-latency constraint fails | coefficient **1**, symbols unvalued | Chen et al., arXiv:2407.11018 (IEEE TNSE) — <https://ar5iv.labs.arxiv.org/html/2407.11018> |
| **Energy-budget violation (per-task cap)** | `r_n = E_n^max − E_n^M` if the max-energy constraint fails | coefficient **1**, symbols unvalued | Chen et al., arXiv:2407.11018 |
| **Energy-budget violation (long-term, Lyapunov)** | `Σ_i V_i(t)(E_total(t) − E_max)` inside `R(t) = −{V·∁ + ΣQ(·) + ΣV_i(·)}` | **V = 5** default (swept 5/10/50/100) | Zhan et al., arXiv:2511.05789 — <https://ar5iv.labs.arxiv.org/html/2511.05789> |
| **Energy budget (time-average constraint + virtual queue)** | `lim_{T→∞}(1/T)Σ𝔼{E_u(t)} ≤ Ē_u`; `Q_u^c(t+1) = max{Q_u^c(t)+E_u^c(t)−Ē_u^c, 0}`; `D = ΔL + V·𝔼{C_s}` | **V and γ_m unreported** | He et al., arXiv:2404.02166 — <https://ar5iv.labs.arxiv.org/html/2404.02166> |
| **Battery-aware preference (not a penalty)** | `β^d = 1 − B/(2B_max)` | coefficient 1 | Ji et al., arXiv:2301.08376 (IEEE TWC 2024) — <https://ar5iv.labs.arxiv.org/html/2301.08376> |
| **Task dropout / uninstalled-task count** | `−ι(1−δ_n)N_n` and `−ι(1−δ_ur)N_ur` | **ι = 7** | Chen & Tong, *Future Internet* 17(12):542, 2025 — <https://mdpi-res.com/d_attachment/futureinternet/futureinternet-17-00542/article_deploy/futureinternet-17-00542-v2.pdf> |
| **Task-dropout rate (weighted third term)** | `−(αD + βE + φ·dropout)` | φ **unreported** | Sun & He, *Electronics* 12(6):1304, 2023 (HDMAPPO) |
| **Queue overflow (buffer beyond capacity)** | `BO_u(t) = max(0, RB_u(t) − B_max(t))` | **w₃ = 1/3** (shares the reward with delay and carbon) | Zulfiqar et al., arXiv:2602.18797 — <https://ar5iv.labs.arxiv.org/html/2602.18797> |
| **Energy wastage** | `EW_u(t)` added to `BO_u(t)` under the same weight | **w₃ = 1/3** | Zulfiqar et al., arXiv:2602.18797 |
| **UAV fly-out-of-bounds** | `p^flyout(t) = μ_o|q_m[n] − clip(q_m[n], 0, W)|` | μ_o **unreported** | Hevesli et al., arXiv:2503.03391 |
| **UAV collision / safety distance** | `p^collision = μ_c Σ_j min((|q_m[n] − q_j[n]| − d_min)/d_min, 0)` | μ_c **unreported** | Hevesli et al., arXiv:2503.03391 |
| **Infeasibility / invalid state (large constant)** | `r_t = −W` if `S_{t+1} = Ω`; else vector `(−D_t, −E_t)` | **W: "a sufficiently large and reasonable positive number"** — no number | Sun et al., arXiv:2501.06410 — <https://arxiv.org/html/2501.06410v1> |
| **Constraint-set violation (tiered indicator sums)** | `r = ℓ₁ + Σᵢ Γᵢ·(violation) ·Λ(·)` for (c1)–(c7); `r = ℓ₂ + exp(Υ−D)` if (c1)–(c7) hold; `r = ℓ₃ + Γ₈·exp(U)` if (c1)–(c8) hold | **Γ₁ = 0.8, Γ₂…Γ₇ = 0.5, Γ₈ = 0.9, ℓ₁ = −0.4, ℓ₂ = −0.2, ℓ₃ = 0.5** | Huang et al., arXiv:2010.08119 (JORA-MADDPG) — <https://ar5iv.labs.arxiv.org/html/2010.08119> |
| **Accuracy-floor violation** | `r_i = ε_i − ε_min` if the accuracy constraint fails | coefficient **1**, unvalued | Ji et al., arXiv:2301.08376 |
| **Terminal task-set failure** | `r_T = −ξ_F Σ_{i∈I} d_i(T)·E^Q_0` | ξ_F *"constant value"* — **not printed** | Ji et al., arXiv:2301.08376 |
| **Reliability constraint** | `r_total(τ) ≥ ρ` | **ρ = 70 %** (constraint, not penalty) | Lan et al., arXiv:2404.15278 — <https://ar5iv.labs.arxiv.org/html/2404.15278> |
| **Deadline as Lagrangian dual term (closest to a hard constraint)** | `min_{λ_k} max_{e,u} QTE + Σ_k λ_k(T^d_k − T^r_k)`; `λ*_k = 0` if `T^d_k − T^r_k ≥ 0` else `λ^penalty_k(T^d_k − T^r_k)` | **λ^penalty_k = 10**; base weights **w₀=0.35, w₁=0.85, w₂=0.15**, **γ = 0** | ElasticVR, arXiv:2512.12366 — <https://ar5iv.labs.arxiv.org/html/2512.12366> |

**Pattern.** The literature's *reported* penalty coefficients are all over the place — μ = 100 (FlexEdge),
ι = 7 (Future Internet), λ^penalty = 10 (ElasticVR), μ_o = μ_t = 0.1 (TGCN), β = 1 (TMC), w = 1/3 (CADDTO),
V = 5 (Ly-DTMPPO), and **unreported** in at least five papers (HDMAPPO, MAPPO-BD, EMODRL, arXiv:2407.11018,
arXiv:2301.08376). There is **no convention** for penalty scale, and the majority of papers that use a "hard"
violation branch never state the constant they gate on — which makes the constraint unverifiable and, in practice,
uncalibrated.

---

## PART 4 — What this implies for a DAG / V2V-MEC single-shot-plan design

Flagged as **inference from the verified evidence**, not a literature claim.

1. **The draft's γ = 0.93 story has no basis in the numbers but a real basis in the phenomenon.** Delete the
   specific figures (no paper adopts γ = 0.93; no paper reports 4000/6000-iteration thresholds). Keep the *claim*
   and re-cite it to the two strongest real sources: **Systems 14(9):1103** — a **DAG-VEC** MAPPO paper that sweeps
   `γ ∈ {0.90, 0.91, 0.93, 0.95, 0.97, 0.99}`, finds converged reward peaks at 0.95, and states that both 0.90/0.91
   (too myopic) and 0.97/0.99 (too far-sighted) "are unfavorable for cooperative multi-vehicle task scheduling" —
   and **Mathematics 13(16):2643**, which is the only offloading paper reporting outright non-convergence at high γ
   (*"when gamma is 0.99 or 0.999 … a failure to converge"*, selecting 0.9). If a γ = 1 argument is wanted, cite
   CMC 2025 for the finite-horizon/objective-alignment argument and Mathematics 13(16):2643 plus Electronics
   13(12):2387 for the counter-argument. **No offloading paper actually trains at γ = 1**, and the modal field
   value is γ = 0.99 (14/33) with λ = 0.95.
2. **A single 20-action plan per episode is structurally unusual** in this literature — the modal environment is
   per-slot control with one task per step. Three DAG-based precedents were verified: **Systems 14(9):1103**
   (DAG task model in VEC, MAPPO, objective = task completion latency + vehicle energy consumption, γ = 0.95,
   **λ = 0.95**, clip 0.20 — the most directly reusable hyperparameter setting for this project), **TPTO**
   (arXiv:2312.11739, per-task sequential over an application DAG, γ = 0.99, clip 0.2, **λ not stated**), and
   **FedMAGS** (arXiv:2605.18437, per-subtask over a DAG with n = 20, PPO + MAML inner/outer + FedAvg — the closest
   match to "PPO inner loop + first-order meta outer loop"). **FedMAGS prints no numeric γ or GAE λ**, so it offers
   no γ = 1 precedent either. Notably **none of the three uses γ = 1** despite DAG structure and finite-horizon
   episodes.
3. **Because `schedule()` is deterministic and returns makespan + energy in one shot, the weighted-sum failure modes
   in Table A apply directly.** Specifically: (i) the weight-sensitivity failure and the policy-collapse failure
   are documented, not hypothetical (arXiv:2407.11018 collapses to local execution at (0.8, 0.2)); (ii) the
   reward-scale failure is the one GMORL explicitly patches with α_T/α_E — a single-shot makespan (seconds) and
   energy (joules) will differ by orders of magnitude, so α:β will silently encode units unless normalisation is
   made explicit.
4. **Normalisation reference choice is the highest-leverage decision**, and the literature offers four verified
   precedents: **local execution** (arXiv:2301.08376, arXiv:2407.11018, arXiv:2505.04272) — natural because a
   local-only plan is a well-defined baseline; **task deadline** (arXiv:2501.06242 only) — natural for a
   delay-sensitive plan; **pure-strategy min–max** (Sensors 22(23):9340) — natural when a Pareto set is being
   scored; **buffer/battery capacity** (arXiv:2602.18797, arXiv:2301.08376) — natural for a battery constraint.
   For a deterministic simulator with known pure strategies, the Sensors min–max over pure strategies is the
   closest published precedent and avoids the "flat reward at large λt+μe" problem of the reciprocal form.
5. **If the deadline must be hard, no verified paper does it via a plain weighted sum** — the honest options are
   (a) a feasibility-gated reward branch (HDMAPPO pattern, but the gate constant must actually be published and
   calibrated), (b) a Lagrangian deadline term (ElasticVR, λ^penalty = 10), or (c) a long-term Lyapunov
   drift-plus-penalty (arXiv:2404.02166 for a time-average energy budget). Option (c) does not map cleanly onto a
   single-shot plan episode, since it relies on per-slot virtual-queue updates.
6. **The 0.5/0.5 scalarisation the draft contrasts against is real and common** (9/13 papers) — so if MARGO uses a
   constraint or Pareto formulation instead, the *contrast* is legitimate and citable (Sensors 22(23):9340 for the
   0.5/0.5/0.8-0.2/0.2-0.8 protocol; IEEE TSC 2025 GMORL for the explicit impossibility/infeasibility/
   undesirability argument against scalarisation). What is **not** citable is "DRL-E2D minimises energy subject to a
   hard deadline" — DRL-E2D is itself a weighted sum with implicit unit weights.

---

## Appendix — Sources that could not be used

`[PAYWALL]` **Wu, Y. et al.**, *Task offloading for vehicular edge computing with imperfect CSI*, Phys. Commun. 55:101867, 2022, DOI [10.1016/j.phycom.2022.101867](https://doi.org/10.1016/j.phycom.2022.101867) — the underlying "P-PPO" paper; reward unconfirmed.
`[PAYWALL]` Ad Hoc Networks 2024, DOI [10.1016/j.adhoc.2024.103743](https://doi.org/10.1016/j.adhoc.2024.103743) — Semantic Scholar `openAccessPdf.status = CLOSED`, `abstract = null`; ScienceDirect 403.
`[BLOCKED]` *IEEE 11626586* (Liu et al., IEEE TVT 2026, DOI [10.1109/TVT.2026.3717633](https://doi.org/10.1109/TVT.2026.3717633)) — closed access, empty abstract, HTTP 202, no preprint. Venue/year/authors verified only.
`[BLOCKED]` MDPI direct HTML (`mdpi.com/...`) returns 403 for every article; the `mdpi-res.com` OA PDF mirror was used instead throughout.
`[BLOCKED]` PMC landing pages (reCAPTCHA); the Europe PMC REST `fullTextXML` endpoint was used instead.
`[BLOCKED]` techrxiv.org and some IEEE PDF endpoints return Cloudflare interstitials.

*Raw extracted texts for this report are cached under `cache/`. Cross-check reports written by the four parallel
verification passes are `VERIFY_gamma_GAE_offloading_literature.md`, `lit/reward_verification_report.md`,
`claim-C-D-verification.md`, and `litverify_HDMAPPO_MORLPPO.md`.*
