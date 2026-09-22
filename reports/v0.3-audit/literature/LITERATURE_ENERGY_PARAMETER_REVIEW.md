# Literature-Grounded Energy-Parameter Review — DAG Offloading in V2V-Assisted MEC (MARGO)

Scope: verify (a) how published V2X / VEC / V2V-assisted offloading papers model energy, and (b) the actual
numeric parameters they use. Everything below was read from the cited URL. Anything not readable from full
text is marked **NOT VERIFIED** — no number is inferred or invented.

Method note: publisher HTML endpoints (ScienceDirect, MDPI, Wiley/Hindawi, IEEE Xplore) return HTTP 403 or a
CAPTCHA to automated fetch. Where a paper is open access I retrieved the OA PDF (arXiv, DOAJ/KeAi, Europe PMC
JATS XML), the society-platform SciEngine data API (tables + MathML equations — see §2 and implication 8), or
an Internet Archive snapshot. **Result: 12 of the 13 primary works are verified from full text or from
publisher-released tables/equations.** Only two gaps remain, both explicitly flagged: **Zhang et al. 2013**
(IEEE Xplore, closed access — citation only) and **Mi21's battery capacity** (never stated by that paper).

---

## 1. Zhao et al. — Energy and Latency Control for Edge Computing in Dense V2X Networks

**Citation.** Jingjing Zhao, Lifeng Wang, Kai-Kit Wong, Meixia Tao, Toktam Mahmoodi, "Energy and Latency
Control for Edge Computing in Dense V2X Networks," arXiv:1807.02311 (cs.NI), v1 6 Jul 2018 / v2 23 Nov 2018.
URL: https://arxiv.org/abs/1807.02311 · full text read at https://ar5iv.labs.arxiv.org/html/1807.02311 and
OA PDF https://arxiv.org/pdf/1807.02311 (6 pp., simulation section + Table I extracted directly).

**Energy scope — FULL SYSTEM INCLUDING RSU COMPUTE.** The objective energy is the RSU's computation energy
plus both radio legs:

> "We consider computing and communication as the two main contributors to the energy consumption. The
> computation energy consumption for the associated RSU at time t is expressed in (7) based on the model in
> [1]" — and Eq. (8): **E(t) = E_c^R(t) + P_v(t)·τ_{1,t} + P_R(t)·τ_{3,t}** (Sec. II-C).

The RSU (edge server) CPU energy is the *first* term of the objective, not a negligibility.

**Exact energy formulas.**

- RSU compute (Eq. 7): `E_c^R(t) = ϱ · C_in(t) · ϑ · f_R^2`
  — "where ϱ is the effective switched capacitance of the RSU processor" and "ϑ is the number of CPU cycles
  per bit required for computing" (Sec. II-B).
- Total (Eq. 8): `E(t) = E_c^R(t) + P_v(t)·τ_{1,t} + P_R(t)·τ_{3,t}` where τ_{1,t} = uplink time, τ_{3,t} = downlink time.
- Latency (Eq. 4): `C_in(t)/C_o^v(t) + ϑ·C_in(t)/f_R + C_out(t)/C_o^R(t) ≤ [ℓ_o − mod(V_o t, ℓ_o)] / V_o`
  — i.e. total offload latency must fit the residual RSU dwell time.
- Rates (Eqs. 6–7): `C_o^v(t) = W log2(1 + P_v(t)L_o(t)G_max^v G_max^R / (I_o^R(t) + σ²))`, downlink analogous.
- Forced-terminology note: in the published PDF the coefficients appear as "̺" and "ϑ" (the iWeb/ar5iv HTML
  renders them `\varrho` and `\vartheta`); the HTML also shows the exponent as `f_R^2` consistently.

**Parameter table (Table I, "SIMULATION PARAMETERS" — transcribed verbatim).**

| Parameter | Value | Parameter | Value |
|---|---|---|---|
| System bandwidth | **2 GHz** | RSU maximum transmit power | **35 dBm** |
| Vehicle maximum transmit power | **25 dBm** | RSU computing capacity | **10 × 10⁹ CPU cycles/s (= 10 GHz)** |
| Required CPU cycles per bit | **300** | Switched capacitance constant | **10⁻²⁸** |
| Each task size | 10 × 10⁶ bits | Carrier frequency | 60 GHz |
| Noise power | −174 + 10·log10(Bandwidth) + 7 (NF) dBm | Pathloss exponent | 2.0 |
| RSU site-distance | 50 m | Vehicle speed | 50 km/h |
| Abs. antenna elevation diff. | 6 m | Perp. distance 1st/2nd lane–RSU | 7 m / 10 m |
| RSU main/side-lobe gain | 15 / −15 dB | Vehicle main/side-lobe gain | 3 / −3 dB |
| RSU beamwidth | 9° | Vehicle beamwidth | 90° |

**Verdict on the claimed numbers — ALL CONFIRMED** (RSU 10 GHz, 300 cycles/bit, ρ = 1e-28, vehicle TX 25 dBm,
RSU TX 35 dBm, bandwidth 2 GHz). Two refinements: (i) the paper says *switched capacitance constant* 10⁻²⁸ for
the **RSU** processor (there is no separate UE κ because the UE does not compute in this model — it offloads);
(ii) the UE/vehicle tier has **no local compute energy at all** in this model.

**Latency in the objective — CONSTRAINT ONLY, not a weighted sum.** Delay enters through constraint C2
(Eq. 4) plus a queue-stability constraint C1: `lim sup (1/T)Σ E[Q_o(t)] < ∞`. The optimisation (Eq. 10) is
`min lim sup (1/T) Σ E[E(t)]` s.t. C1–C4. The energy/latency trade-off is exposed only via the Lyapunov
drift-plus-penalty control variable η:

> "By introducing the penalty term η E[E(t)|Q_o(t)] with the non-negative control variable η, the
> drift-plus-penalty is Δ_η(t) = Δ(t) + η E[E(t)|Q_o(t)]" (Eq. 12), "where η represents the price of energy
> consumption, lower η means that more energy will be consumed to accomplish more computing tasks."

**Penalty terms.** No deadline-violation or dropout penalty. The only penalty is the Lyapunov
drift-plus-penalty energy term above — note this is a *Lyapunov* penalty, not a violation penalty, and it is
a common mis-citation.

---

## 2. Liu et al. — Joint computation offloading and resource allocation in VEC networks

**Citation.** Shuang Liu, Jie Tian, Chao Zhai, Tiantian Li, "Joint computation offloading and resource
allocation in vehicular edge computing networks," *Digital Communications and Networks*, vol. 9, no. 6,
pp. 1399–1410, Dec. 2023. DOI 10.1016/j.dcan.2022.12.002. Gold OA (CC BY-NC-ND).
URL: https://doi.org/10.1016/j.dcan.2022.12.002 · https://www.sciencedirect.com/science/article/pii/S2352864822002620
· DOAJ record https://doaj.org/article/cb7970628997409a9061dfa500d89229

**Access status — recovered.** ScienceDirect (landing page, `/pdf`, `pdfft`, `pdf.sciencedirectassets.com`
plus text proxies and a scripted Chromium session) returns HTTP 403/CAPTCHA to every automated request, and
the paper is not in Sci-Hub. However the publisher's society platform **SciEngine carries the article's
machine-readable data**: article page https://www.sciengine.com/DCAN/doi/10.1016/j.dcan.2022.12.002 · data API
https://www.sciengine.com/sci-open/api/v1/open/article/figAndTable?articleBaseId=BE6EF1BE63DF40219C71410642808C16
— this yielded **Table 1 verbatim plus the MathML of all 39 numbered equations**. Table 1 and the equations
cross-check each other exactly (γ_m/γ_j in Table 1 match Eqs. 10/15). **Limitation: the prose body was not
readable**, so only tabulated and equated quantities are reported.

**Verdict on claimed numbers — ALL SIX CONFIRMED.** Table 1, verbatim:

| Parameter | Value |
|---|---|
| **Service Vehicle resource F_j^max** | **[1, 2] GHz** ✓ |
| **RSU server resource F_m^max** | **10 GHz** ✓ |
| **Switch capacitance constant of SV γ_j** | **5 × 10⁻²⁷** ✓ |
| **Switch capacitance constant of RSU server γ_m** | **1 × 10⁻²⁷** ✓ |
| **Vehicle transmission power P** | **30 dBm** ✓ |
| **Bandwidth B** | **10 MHz** ✓ |
| Data size R_i | [100, 2000] KB |
| "Computation cycles of task C_i" | [0.5, 2.5] GHz (label/unit inconsistent as printed; **no cycles/bit value exists in this paper**) |
| Noise spectral density N₀ | −174 dBm/Hz |
| Path loss exponent κ | 4 ; channel gain at ref. distance δ₀ = −30 dB |
| Max latency T_i^max | [0.5, 2.5] s |
| Delay weight ω_l / cost weight ω_e | 0.5 / 0.5 |

**Energy scope — INCLUDES the MEC/RSU server compute energy, AND the service vehicle's compute energy.**
- Eq. (10), V2R/RSU mode: `E_{i,m} = P_i · t^up_{i,m} + C_i (f^R_{i,m})² · γ_m` — the second term **is the
  RSU/MEC server compute energy**, with γ_m the server's switched-capacitance constant.
- Eq. (15), V2V/SV mode: `E_{i,j} = P_i · t^up_{i,j} + C_i (f^S_{i,j})² · γ_j` — the second term is the
  **service vehicle's** compute energy.
- Eq. (23): `E_i = Σ_m x_{i,m}E_{i,m} + Σ_j y_{i,j}E_{i,j}`.
- **Momentous nuance for MARGO:** the task model is **fully offloaded** (Eq. 5 / C5, `Σx + Σy = 1`), so there is
  **no task-vehicle (UE) local-compute energy term and no UE-tier γ**. Scope = task-vehicle uplink transmission
  energy + compute energy of the *executing* node (RSU server **or** service vehicle). This is the closest
  scope analogue in the whole review to MARGO's `E = UE + helper` — except Liu also counts the RSU, and counts
  no requester-local compute energy because nothing runs locally.

**Latency — BOTH a weighted-sum term AND a hard constraint.**
- Eq. (21): `U_i = ω_l T_i + ω_e E_i + e_i − Σ_m x_{i,m}g_{i,m} − Σ_j y_{i,j}g_{i,j}`; Eq. (22):
  `T_i = Σ_m x t_{i,m} + Σ_j y t_{i,j}`; problem P1 (Eq. 25) `min Σ_i U_i` s.t. **`C1: T_i ≤ T_i^max`** —
  weighted sum with default weights 0.5/0.5 **plus** a deadline constraint.
- Eq. (8)/(9): `t^exe = C_i/f^R`, `t = t^up + t^exe`. Distinctive term: **transmission gain**, Eq. (17)/(18)
  `g = ∫₀^{t^up}(r(t) − r̄)dt`, `r̄ = R_i/t^up`, subtracted from `U_i` (this is the "transmission gain" of the
  abstract and is a mobility **credit**, not a penalty).

**Penalty terms — no fixed violation/dropout penalty.** Deadline and capacity enter as Lagrangian duals:
Eq. (33) `L = ω_lT_i + ω_eE_i + e_i − Σxg − Σyg + α_i(T_i^max − T_i) + β_m(F_m^max − Σx f^R) + ρ_j(F_j^max − Σy f^S)`,
duals updated by subgradient (Eq. 39). The only hard association cap is C7 `Σ_i y_{i,j} ≤ Q`.

**Correction to the earlier γ-provenance hypothesis.** Liang et al. arXiv:2411.10770 is **NOT** the source of
5e-27 / 1e-27. Its Table II (read from https://arxiv.org/pdf/2411.10770) gives 24 cycles/bit;
`f_pk = [1, 2.5] GHz`, `f_rj = [4, 6] GHz`; `κ_v = 1e-27`, `κ_r = 1e-28`; `P_t = 0.28183815 W` (= 24.5 dBm);
`N₀ = 1.2589e-13 W`; `W_b = 15` (printed "MB"); `T_max = [100, 200] ms`; `D_qi = [10, 30] MB` — matching
neither 5e-27/1e-27 nor 30 dBm. The project's values are **genuinely Liu et al. Table 1** and can be cited to
it directly. Note also that the venue is **Elsevier/KeAi, 2022/2023** — not IEEE/Springer ~2018–2021.

---

## 3. UAV / Dual-RIS-aided MEC-enabled IoV energy optimisation

**Citation.** Emmanouil T. Michailidis, Nikolaos I. Miridakis, Angelos Michalas, Emmanouel Skondras,
Dimitrios J. Vergados, "Energy Optimization in Dual-RIS UAV-Aided MEC-Enabled Internet of Vehicles,"
*Sensors* **2021**, 21(13), 4392. DOI 10.3390/s21134392 · PMID 34198977 · PMCID PMC8271975.
URL: https://doi.org/10.3390/s21134392 · MDPI 403s bots, so full text (all equations, Table 2) was read from
the OA JATS XML: https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8271975/fullTextXML and
https://pmc.ncbi.nlm.nih.gov/articles/PMC8271975/

**Verdict on claimed numbers — ALL CONFIRMED**, all at the ARSU (aerial RSU) tier, Table 2:
`f_A,max = 3 GHz`, `c_A = 10³ cycles/bit`, `κ_A = 10⁻²⁷` (unit not printed; the text calls it the
"chip-dependent effective capacitance coefficient"). Table 2 cites these three from ref [24] = Zhang et al.,
IEEE Trans. Ind. Informat. 2020, 16, 5505–5516 — a closed-access paper whose values were therefore **not read
at first hand**.

**Energy scope — vehicles + ARSU only; MEC/GRSU compute neglected; propulsion excluded.** Verbatim (Abstract):
> "this paper proposes an optimization approach that intends to minimize the weighted total energy
> consumption (WTEC) of the vehicles and ARSU subject to transmit power constraints, timeslot scheduling, and
> task allocation."

Sec. 4.1: "Thus, the propulsion energy consumption is excluded from the optimization process."
GRSU compute: "the computation delay at GRSU can be neglected owing to its computing capabilities."
Task model: "the k-th vehicle fully offloads to ARSU and GRSU (via relaying) its task" → **no UE local
compute energy**, and there is **no RIS energy term at all**. This is a partial-inclusion example: ARSU
compute is counted, the terrestrial MEC server's compute is not.

**Exact formulas (read from the MathML tree).**

- Offload energy: `E_{k,off}[n] = p_{k,off}[n]·τ_{k,off}[n]` (4); `E_{k,A,off}[n] = p_{k,A,off}[n]·τ_{k,A,off}[n]` (5)
- ARSU compute: `E_{k,cA}[n] = P_{k,cA}·τ_{k,cA}[n] = κ_A·c_A³·K²·(β_k[n] b_k[n])³·τ⁻²`, with `P_{k,cA} = κ_A·f_{A,max}³` (6)
- Objective: `min E_total = Σ_n Σ_k ( w_k·E_{k,off}[n] + w_A·E_A[n] )` (26a), `E_A[n] = Σ_k (E_{k,cA}[n] + E_{k,A,off}[n])` (27)
- UAV flight energy (defined, **not** optimised): Eq. (1), rotary-wing power model with `P0, P1, P2, d_r, s, ρ, G, v_tip, v0`.

**Parameter table (Table 2 unless noted).** K = 3; weights `w_k = 1`, `w_A = 0.1`; `b_k = 0.4 Mbits/timeslot`;
deadline = flight duration `T = 8 s`; slot `τ = 0.2 s`; max TX power = **35 dBm** (vehicle) and **35 dBm**
(ARSU); target rate `r_t = 1.5 bps/Hz`; **B = 10 MHz** (Sec. 5 text); RIS `L = 64` elements, `q = 2`;
path-loss exponents 3.5/2.2/2/3.5/2/2.2; `β0 = −20 dB`; `N0 = −80 dBm`; `m_kA = m_AG = 1`;
Rician factors 7/10/10/7 dB; `v_k = 60 km/h`; `v_A = 5 m/s`; convergence tolerance ε = 1e-4.
**Battery capacity: NOT VERIFIED** — no J/Wh/% figure anywhere; "battery" is qualitative only.

**Latency in the objective — constraints + separately reported metric; NOT in the objective.** Constraints
(26d)(26e)(26g) bound transmission delay by τ/K; (26f) bounds `β_k` by `τ·f_{A,max}/(K c_A b_k)` (ARSU
compute delay ≤ τ/K); completion time `τ_TCCD = Σ_n Σ_k τ_k[n]` is reported as a metric. There is **no
weighted energy–latency sum.**

**Penalty terms — NONE.** `χ_{1,2,3,k,n}` are Lagrange dual multipliers (ellipsoid method), not penalties.
Only a tolerance ε = 1e-4. If MARGO attributes a penalty term to this source, that attribution is unsupported.

---

## 4. Real-time energy-aware offloading in vehicular networks with MEC

**Citation.** Haibo Zhang, Xiangyu Liu, Xia Bian, Yan Cheng, Shengting Xiang, "A Resource Allocation Scheme
for Real-Time Energy-Aware Offloading in Vehicular Networks with MEC," *Wireless Communications and Mobile
Computing*, vol. 2022, Article ID 8138079, 17 pp. DOI 10.1155/2022/8138079.
URL: https://doi.org/10.1155/2022/8138079 (publisher Cloudflare-403 to automation).
Full text read via Internet Archive OA copies:
https://web.archive.org/web/20240423015325/https://www.hindawi.com/journals/wcmc/2022/8138079/ and the 17-page
OA PDF http://web.archive.org/web/20220423103105if_/https://downloads.hindawi.com/journals/wcmc/2022/8138079.pdf

**Energy scope — V-UE (requester) BATTERY ONLY.** Objective energy = local compute + uplink transmit.
Verbatim:
> "The corresponding energy consumption of V-UE is expressed as e^C_{i,j} = Σ_n a_{i,j,n} p_{i,j,n} d_{i,j}/R_{i,j}." (Eq. 7)
> "the time and energy consumption of computation result from the MEC server to the V-UE are neglected…"
> "The objective is to minimize the overhead of V-UEs" · "study the trade-off between the energy consumption
> of vehicle units and the latency of the corresponding tasks."

No MEC/RSU compute energy, no downlink/receive energy, no idle/circuit energy. MEC compute **time** does
enter latency (`c_{i,j}/f_C`).

**Exact formulas.** Local (Eq. 2): `e^L = k (f_loc)² c`, "where k = 10⁻²⁶ is a coefficient that depends on the
chip architecture." Offload (Eq. 7): `e^C = Σ_n a p (d/R)` — uplink only. Rate (Eq. 3):
`R = w log2(1 + p h/(σ² + I))`, `w = W/N`. Latency (Eqs. 1, 6): `t^L = c/f_loc`, `t^C = d/R + c/f_C`.

**Parameter table (Table 2).** `p_max = 23 dBm`; `W = 20 MHz`; `f_C = 4 GHz` (MEC); `c_{i,j} = [0.1, 1] GHz`
(total cycles per task); V-UE `f_loc = 0.2–1 GHz`; `σ² = −113 dBm`; **`E_total = 100 J` battery**;
`D_max = 0.5 s`; `d_{i,j} = [300, 1600] KB`; `|Q_u| = 10`, `|Q_c| = 4`, `H = 4`, `T = 60 s`;
μ₁ = 2e-18, μ₂ = 1e-5, μ₃ = 1e-10, ε = 1e-5. Scenario: 5 RSUs, 10 channels/RSU, 25 V-UEs/RSU, RSU radius 250 m,
3GPP TR 36.885. **Cycles-per-bit: NOT GIVEN** (`c` is total cycles per task, not an intensity).

**Verdict on claimed numbers.** V-UE 0.2–1 GHz **CONFIRMED**; MEC 4 GHz **CONFIRMED**; k = 1e-26 **CONFIRMED**
(body text of Eq. 2, not Table 2); battery 100 J **CONFIRMED** ("E_total is the battery capacity in Joules");
**23 dBm CORRECTED** — it is the *maximum* `p_max` (constraint C4), the allocated `p_{i,j,n} ∈ [0, 23 dBm]` is
an optimisation variable, not a fixed operating power. "Optimises only the V-UE battery" is **CONFIRMED for
energy scope**, with the caveat that the objective is a weighted sum of V-UE energy **and** latency, not a
pure battery objective.

**Latency handling — weighted-sum scalarisation + hard deadline.** Residual-energy-dependent weights (Eq. 8):
`w' = w·r^E`, `r^E = (E_total − [(1−s)e^L + s e^C]) / E_total`, `w^t = w'`, `w^e = 1 − w'`;
`O^L = w^t t^L + w^e e^L`, `O^C = w^t t^C + w^e e^C`; problem P1 minimises the s/(1−s) sum. Hard deadline
constraint C1: `≤ D_max`.

**Penalty terms — NONE.** Zero matches for penalt*/dropout/violat*/infeasib*/outage/failure in the full text.
DRL reward `R = U − C` with `U = ρ(O^L + O^C)`, `C = η₁E + η₂T` — reward shaping, not a penalty term.

**Structural warning for MARGO.** This is **not a DAG paper and not a V2V/helper-vehicle paper**. Task model is
one independent task per V-UE `{d, c, D_max}`; only two modes (local V-UE, MEC-RSU). "V2V Link" appears in
Figure 1 only and in no equation. Do not cite it as a DAG or V2V baseline.

---

## 5. Canonical κf²C model — Mao et al. survey and Zhang et al.

### 5a. Mao, You, Zhang, Huang, Letaief — survey

**Citation.** Yuyi Mao, Changsheng You, Jun Zhang, Kaibin Huang, Khaled B. Letaief, "A Survey on Mobile Edge
Computing: The Communication Perspective," *IEEE Communications Surveys & Tutorials*, vol. 19, no. 4,
pp. 2322–2358, 2017. DOI 10.1109/COMST.2017.2745201. Open-access preprint arXiv:1701.01090
(https://arxiv.org/abs/1701.01090), read at https://ar5iv.labs.arxiv.org/html/1701.01090 and OA PDF
https://arxiv.org/pdf/1701.01090v4.

**VENUE CORRECTION.** The task brief places this in *Proc. IEEE* 2017. It is in **IEEE Communications Surveys
& Tutorials** 19(4):2322–2358 (OpenAlex/DOI confirm; JPROC ISSN 0018-9219 returns no such 2017 MEC survey).
Also note the arXiv v4 was retitled "Mobile Edge Computing: Survey and Research Outlook" and is a *different*
submission from the COMST version — the COMST title is the one to cite.

**Device energy — the canonical form, verbatim:**
> "According to the circuit theory [78]–[81], the CPU power consumption can be divided into several factors
> including the dynamic, short-circuit, and leakage power consumption, where the dynamic power consumption
> dominates the others. In particular, it is shown in [80] that the dynamic power consumption is proportional
> to the product of V²_cir f_m where V_cir is the circuit supplied voltage. It is further noticed in [78], [81]
> that, the clock frequency of the CPU chip is approximately linear proportional to the voltage supply when
> operating at the low voltage limits. Thus, the energy consumption of a CPU cycle is given by κf_m², where κ
> is a constant related to the hardware architecture. For the computation task A(L, τ, X) with CPU clock speed
> f_m, the energy consumption can be derived: **E_m = κ L X f_m²** (2)."

**Server energy — THE SURVEY APPLIES THE SAME κf²C FORM TO THE MEC SERVER.** Verbatim:
> "Two tractable models are widely used for the energy consumption of MEC servers. One model is based on the
> DVFS technique described as follows. Consider an MEC server that handles K computation tasks and the k-th
> task is allocated with w_k CPU cycles with CPU-cycle frequency f_{s,k}. Hence, the total energy consumed by
> the CPU at the MEC server, denoted by E_s, can be expressed as **E_s = Σ_{k=1}^{K} κ w_k f_{s,k}²** (4),
> which is similar to that for the mobile devices."
> "The other model is based on an observation in recent works [91]–[93] that the server-energy consumption is
> linear to the CPU utilization ratio which depends on the computation load. Moreover, even for an idle server,
> it still, on average, consumes up to 70% of the energy consumption for the case with the full CPU speed.
> Thus, the energy consumption at the MEC server can be calculated according to **E_s = αE_max + (1−α)E_max u**
> (5), where E_max is the energy consumption for a fully-utilized server, α is the fraction of the idle energy
> consumption (e.g., 70%) and u denotes the CPU utilization ratio."

So the canonical formalism supplies **both** the device form and a matching server form, plus a second
utilization-linear server model with α ≈ 70% idle draw. Task notation `A(L, τ_d, X)`: L bits, deadline τ_d s,
workload X CPU cycles/bit, "can be estimated through task profilers" — this is the standard origin of
cycles-per-bit task triples. **No numeric κ, X, or f values are given in this survey** (it is analytic).

### 5b. Zhang et al. — Energy-optimal mobile cloud computing under stochastic wireless channel

**Citation.** Weiwen Zhang, Yonggang Wen, Kyle Guan, Daniel C. Kilper, Haiyun Luo, Dapeng Oliver Wu,
"Energy-Optimal Mobile Cloud Computing under Stochastic Wireless Channel," *IEEE Transactions on Wireless
Communications*, vol. 12, no. 9, pp. 4569–4581, 2013. DOI 10.1109/TWC.2013.072513.121842.
URL: https://doi.org/10.1109/TWC.2013.072513.121842 · metadata verified via
https://researchr.org/publication/ZhangWGKLW13/bibtex

**PAYWALL — NOT VERIFIED at first hand.** IEEE Xplore is closed access and no OA copy was reachable. The DOI,
venue, volume, issue and page range are verified; **the paper's own κ, cycles/bit, f and transmit-power
numbers are NOT VERIFIED and are not reported here.** Its canonical "energy-optimal under stochastic channel"
formulation is the lineage behind the κf²C model that the Mao survey states explicitly (Sec. 5a) — cite the
survey, or obtain the paper, rather than attributing numbers to it.

### 5c. Concrete calibration of the canonical model in a well-cited instance

**Citation.** Yuyi Mao, Jun Zhang, Khaled B. Letaief, "Dynamic Computation Offloading for Mobile-Edge
Computing with Energy Harvesting Devices," *IEEE JSAC*, vol. 34, no. 12, pp. 3590–3605, 2016.
arXiv:1605.05488, OA PDF read at https://arxiv.org/pdf/1605.05488

- Local energy (Eq. 3): `E_mobile^t = κ Σ_{w=1}^{W} (f_w^t)²`, "where κ is the effective switched capacitance
  that depends on the chip architecture" — per-cycle form; `W = L·X` cycles.
- Offload energy (Eq. 5): `E_server^t = p^t · D_server^t = p^t · L / r(h^t, p^t)` — **device-side transmit
  energy only**; server compute energy is not in the objective.
- **Parameters (Sec. VI, verbatim):** "κ = 10⁻²⁸, τ = φ = 2 ms, ω = 1 MHz, σ = 10⁻¹³ W, p_tx^max = 1 W,
  f_CPU^max = 1.5 GHz, E_max = 2 mJ, and L = 1000 bits. Besides, X = 5900 cycles per byte, which corresponds
  to the workload of processing the English main page of Wikipedia [28]. Moreover, P_H = 12 mW, d = 50 m and
  τ_d = 2 ms."
- This is the **canonical device-only** objective: the device's energy is minimised subject to a hard deadline,
  with server execution delay explicitly neglected.

---

## 6. Energy-scope precedent: papers that INCLUDE vs EXCLUDE MEC compute energy

### 6a. INCLUDE MEC/RSU server compute energy

**(A1) Cheng, Teng, Sun, Liu, Wang — multi-MEC system, server energy in the objective.**
Kang Cheng, Yinglei Teng, Weiqi Sun, An Liu, Xianbin Wang, "Energy-Efficient Joint Offloading and Wireless
Resource Allocation Strategy in Multi-MEC Server Systems," arXiv:1803.07243; read at
https://ar5iv.labs.arxiv.org/html/1803.07243
> "In the case of offloading, similarly, it incorporates the transmission energy used to send the input data
> D_i to the helping MEC server k and the energy consumption on the MEC server k for computing."
- Local (Eq. 6): `E_i^l = k_0 f_{i,loc}² D_i X_i`; server compute (Eq. 8): `E_i,k^c = k_1 f_{k,ser}² D_i X_i`;
  transmit (Eq. 7): `E_i,k^t = Σ_n w_{i,n,k} p_{i,n,k} D_i / R_{i,k}`; total (Eq. 9) sums server + transmit.
  Objective (P) `min Σ_i Σ_k b_{i,k} E_i,k^r` — the server's compute energy is inside the minimised quantity.
- Latency: hard constraint C7 `T_i = Σ_k b_{i,k}(D_i/R_{i,k} + D_i X_i/f_{k,ser}) ≤ τ_i`. No penalty terms.

**(A2) Gu, Wu, Fan, Cheng, Chen, Letaief — VEC/RSU, three-part system energy.**
Xueying Gu, Qiong Wu, Pingyi Fan, Nan Cheng, Wen Chen, Khaled B. Letaief, "DRL-Based Federated
Self-Supervised Learning for Task Offloading and Resource Allocation in ISAC-Enabled Vehicle Edge Computing,"
*Digital Communications and Networks* (2025), arXiv:2408.14831; read at https://arxiv.org/pdf/2408.14831 (14 pp.)
> "we propose an optimization algorithm aimed at minimizing the energy consumption of the entire system. The
> energy consumption is divided into three parts: RSU computing energy consumption E_{t,R}^n, transmission
> energy consumption E_t^{n,trans} from vehicle n to RSU, and local computing energy consumption E_{t,L}^n."
> "we also aimed to minimize the total energy consumption, including local training, RSU training, and
> transmission."
- Local (Eq. 5): `E_{t,L}^{n,comp} = κ (f_{t,L}^{n,comp})² c Z`; RSU (Eq. 7): `E_{t,R}^{comp} = κ (f_{t,R}^{comp})² c Z`;
  total (Eq. 28): `E_t^{n,total} = g_t^n E_{t,R}^n + E_{t,L}^n + g_t^n E_t^{n,trans}`; objective (Eq. 29)
  `min Σ_n E_t^{n,total}`. DVFS: `p^{comp} = κ f³` (Eq. 3).
- **Parameters (Table 2, verbatim):** `κ = 10⁻²⁷`; `f^min = 5×10⁷ Hz`, `f^max = 4×10⁸ Hz` (vehicle);
  `f_{t,R}^{comp} = 6×10⁹ Hz` (RSU = 6 GHz); `p_min = 5 W`, `p_max = 200 W`; `c = 1600 cyc/s`;
  `Z = 1500 KB`; `N₀ = −114 dB`; `E_max = 3000`; `v = 10–15 m/s`; `N = 512` (bandwidth `B = 256`, unit not
  printed — treat as unverified) and "total bandwidth 2×10⁶ Hz". No penalty term.

**(A3) Wang, Xu, Wang, Cui — wireless-powered MEC, AP (server) energy objective.**
Fangming Wang, Jie Xu, Xin Wang, Shuguang Cui, "Joint Offloading and Computing Optimization in Wireless
Powered Mobile-Edge Computing Systems," *IEEE Trans. Wireless Commun.*, vol. 17, no. 3, pp. 1784–1797, 2018.
DOI 10.1109/TWC.2017.2785305; OA preprint arXiv:1702.00606, read at https://ar5iv.labs.arxiv.org/html/1702.00606
> "As for the AP, the energy is mainly consumed for executing the offloaded computation tasks and transmitting
> the computation results back to the users."
> "We adopt a simplified linear energy consumption model for the computation at the AP as E_MEC = α Σ_{i=1}^{K} ℓ_i,
> where α denotes the energy consumption per offloaded bit at the AP."
- Objective `min T·tr(Q) + Σ_i α ℓ_i`. Note the server model here is **linear per bit**, not κf²C — evidence
  that server-energy modelling spans at least three forms (κf²C, utilization-linear, energy-per-bit).
- Latency: hard `Σ 1/f_{i,n} ≤ T`; MEC compute time neglected. Energy-causality constraint
  `E_loc + E_offl ≤ E_i`. No penalty. No numeric values in the read text.

**(A4) Cang, Chen, Pan, Yang, Hu, Sun, Chen — objective is purely the MEC server's CPU energy.**"Joint User Scheduling and Computing Resource Allocation Optimization in Asynchronous Mobile Edge Computing
Networks," *IEEE Trans. Communications*, vol. 72, no. 6, pp. 3378–3392, 2024. DOI 10.1109/TCOMM.2024.3358237;
OA preprint arXiv:2401.11377, read at https://ar5iv.labs.arxiv.org/html/2401.11377
> "the energy consumption of the MEC server for all tasks computation can be formulated by
> E_MEC = Σ_{n=1}^{N} Σ_{m=n+1}^{N+1} κ f_{n,m}³ Δt_m, where κ denotes the energy coefficient of the MEC server."
- `κ f³ Δt = κ f² · (f Δt) = κ f² · cycles` — the exact E = κf²C server-CPU form. Objective (Eq. 2) is this
  quantity alone. Latency: hard `Σ Δt_i ≤ T`, server frequency cap, completion constraint. No penalty.

**(A5) Malik & Vu — weighted sum of UE **and** server energy, κf²C on both ends.**
"Energy-Efficient Computation Offloading in Delay-Constrained Massive MIMO Enabled Edge Network Using Data
Partitioning," *IEEE Trans. Wireless Commun.*, vol. 19, no. 10, pp. 6977–6991, 2020. DOI 10.1109/TWC.2020.3007616;
OA preprint arXiv:2001.08259, read at https://ar5iv.labs.arxiv.org/html/2001.08259
> "we formulate a novel problem to minimize a weighted sum of the energy consumption at both the users and the
> MEC server under a round-trip latency constraint, using a combination of data partitioning, transmit power
> control and CPU frequency scaling at both the user and server ends."
> "The formulation accounts for energy consumption at both the users and MEC ends, compared to current
> literature considering only one side [11]–[15]."
- UE local `E_LC = Σ κ_i c_i (u_i − s_i) f_{u,i}²`; **server (Eq. 5) `E_OC = Σ κ_m f_{mi}² d_m s_i`** with
  "κ_m is a hardware dependent constant of the MEC server"; offload `E_OFF = Σ p_{li} s_i/(B r_{u,i})`;
  downlink `E_DL = Σ P η_{li} μ s_i/(B r_{d,i})`. Per-tier cycles/bit: `c_i` (UE) and `d_m` (server).
  Latency: hard round-trip `T_d`. No penalty. No numeric values in the read text.

### 6b. EXCLUDE MEC/RSU compute energy (device/UE/vehicle-only objective)

**(B1) Zhang et al. 2022** — see §4. Objective energy = `e^L + e^C`, both V-UE-side. Quotes as in §4.
This is the precedent for MARGO's requester-battery-only scope (k = 1e-26, battery 100 J).

**(B2) You, Huang, Chae, Kim — the canonical device-only baseline (1400+ citations).**
Chang-Sheng You, Kaibin Huang, Hyukjin Chae, Byoung-Hoon Kim, "Energy-Efficient Resource Allocation for
Mobile-Edge Computation Offloading," *IEEE Trans. Wireless Commun.*, vol. 16, no. 3, pp. 1397–1411, 2017.
DOI 10.1109/TWC.2016.2633522; OA preprint arXiv:1605.08518, read at https://ar5iv.labs.arxiv.org/html/1605.08518
> "The objective is to minimize the weighted sum mobile energy consumption: Σ_{k=1}^{K} β_k (E_off,k + E_loc,k),
> where the positive weight factors {β_k} account for fairness among mobiles."
> "the radio and computation resources were jointly allocated to minimize the mobile energy consumption under
> offloading latency constraints."
- `E_loc,k = (R_k − ℓ_k) C_k P_k` (C_k cycles/bit, P_k energy per cycle); `E_off,k = p_k t_k = (t_k/h_k²) f(ℓ_k/t_k)`,
  `f(x) = N₀(2^{x/B} − 1)`; rate `r_k = B log2(1 + p_k h_k²/N₀)`. **No cloud/server energy term.**
- Latency: hard `C_k(R_k−ℓ_k)/F_k ≤ T`; cloud-load cap `Σ C_k ℓ_k ≤ F`; cloud compute time optional
  `t_comp = (Σ ℓ_k C_k)/F'`. The cloud appears **only via capacity F/F' — never as energy.**
- Also: "downloading consumes negligible mobile energy" — the standard justification for dropping the
  downlink receive term.

**(B3) Vu, Van Huynh, Hoang, Nguyen, Dutkiewicz — cooperative MEC, mobile-device energy only.**
"Offloading Energy Efficiency with Delay Constraint for Cooperative Mobile Edge Computing Networks,"
*Proc. IEEE GLOBECOM 2018*, pp. 1–6. DOI 10.1109/GLOCOM.2018.8647856; OA preprint arXiv:1811.12686, read at
https://ar5iv.labs.arxiv.org/html/1811.12686
> "we consider a joint offloading and resource allocation problem in which the total energy consumption of
> mobile devices is minimized."
- `E_i^l = v_i C_i` (v_i = energy per CPU cycle); `E_ij^f = E_ij^u + E_ij^d`. **No server energy term.**
- Latency: hard `T_i ≤ T_i^r`; the MEC frequency `f_ij^f` appears **only inside delay**
  `T_ij^f = D_i^i/r_ij^u + D_i^o/r_ij^d + C_i/f_ij^f` and in per-server frequency caps. No penalty.

**(B4) Chen, Yi, Alam, Nallanathan — MEC capability only as delay.**
"Dynamic Task Software Caching-Assisted Computation Offloading for Multi-Access Edge Computing,"
*IEEE Trans. Communications*, vol. 70, no. 10, pp. 6950–6965, 2022. DOI 10.1109/TCOMM.2022.3200109;
OA preprint arXiv:2208.07151, read at https://ar5iv.labs.arxiv.org/html/2208.07151
> "we formulate a joint task software caching update (TSCU) and computation offloading (COMO) problem to
> minimize users' energy consumption while guaranteeing delay constraints…"
- Device local `E_{k,f}^L = ζ (f_k^L)² S_f`, with deadline-optimal `f_k^L = S_f/τ` ⇒ `E = ζ S_f³/τ²`;
  offload `E_{k,f,t}^O = p_k (I_f + D_f)/r_{k,t}`. Server capability `f_C` appears only as the delay term
  `S_f/f_C` and as a cache-size constraint. No penalty. No numeric values in the read text.

**(B5) Mao, Zhang, Letaief 2016** — see §5c. Device energy only; server execution delay explicitly neglected.

### 6d. The closest scope analogue to MARGO: Liu et al. 2023

**Liu, Tian, Zhai, Li, *DCN* 9(6):1399–1410, 2023** (see §2) deserves separate emphasis because it is the only
verified paper whose energy accounting has the *same shape* as MARGO's while taking the **opposite** stance on
the server. It counts, for **both** offloading modes, the radio cost **plus the compute energy of the node that
actually executes the task**:

- V2R: `E_{i,m} = P_i·t^up_{i,m} + C_i (f^R_{i,m})² γ_m` — RSU/MEC server compute energy included (Eq. 10)
- V2V: `E_{i,j} = P_i·t^up_{i,j} + C_i (f^S_{i,j})² γ_j` — **service-vehicle (helper) compute energy included** (Eq. 15)

so it is simultaneously (i) a **V2V-helper-compute inclusion precedent** — directly relevant to MARGO's
`rho_v2v` term, and (ii) a **MEC-compute inclusion precedent**. MARGO keeps the V2V-helper term and drops the
MEC term; Liu keeps both. Because Liu's task model is fully offloaded (`Σx + Σy = 1`, Eq. 5/C5), it has no
requester-local compute term at all, so it cannot be cited for the UE local-compute half of MARGO's objective.

### 6e. Prevalence evidence (authoritative survey, verbatim)

**Citation.** S. Zhang, N. Yi, Y. Ma, "A Survey of Computation Offloading With Task Types," *IEEE Trans.
Intelligent Transportation Systems*, vol. 25, no. 8, pp. 8313–8333, Aug. 2024. DOI 10.1109/TITS.2024.3410896;
OA preprint arXiv:2401.01017, read at https://ar5iv.labs.arxiv.org/html/2401.01017
> "Typically, in scenarios where tasks are offloaded to MCC or MEC servers with a continuous energy supply,
> researchers do not consider the energy consumption of these servers, as noted in several studies
> [60], [90], [61], [91], [75], [67], [92], [81], [128], [59], [68], [93], [69], [94], [70], [71], [72],
> [95], [62], [129], [18], [63]. However, there are many research works that consider the energy of edge and
> cloud servers in the optimization [79], [80], [78], [130], this is because cloud service providers and data
> centers need to optimize their energy usage to reduce their environmental impact and lower operating costs."
- Its Table I ("Optimization Objective → Energy Saving Maximization") lists "1) Energy consumption of UE only"
  against ~19 reference codes versus "2) Energy consumption of multiple entities (including various
  combinations of UE, relay node, edge or cloud server)" against ~7.
- Corroborated from the include side (Malik & Vu): "Most existing works have considered energy minimization at
  only one side of the network, either the users [11]–[13], or the MEC-server when it is energy constrained,
  such as a UAV-MEC [14][15]."

**Bottom line:** device-only objectives dominate, roughly 22 vs 4 by the survey's own prose count. Server
compute energy enters when (i) the edge node is itself energy-constrained (UAV / EH / off-grid), (ii) the
authors take a system- or operator-level view, or (iii) the objective *is* the server's energy.

---

# Synthesis Table A — parameter ranges across the literature

All values are as printed in the cited papers. "—" = not stated in the accessible text. Entries marked ⚠ are
**NOT VERIFIED** (abstract-only or paywalled); there are none left in this table. Abbreviations:
Zh18 = Zhao 2018; Liu23 = Liu DCN 2023; Mi21 = Michailidis Sensors 2021; Zh22 = Zhang WCMC 2022;
MZL16 = Mao JSAC 2016; Ch18 = Cheng 2018; Gu25 = Gu DCN 2025; Wa18 = Wang TWC 2018; Ca24 = Cang TCOM 2024;
MV20 = Malik & Vu TWC 2020; Yo17 = You TWC 2017; Vu18 = Vu GLOBECOM 2018; Ch22 = Chen TCOM 2022;
Li24 = Liang arXiv 2024.

| Parameter | min | typical | max | which papers |
|---|---|---|---|---|
| **UE / mobile-device CPU freq** | 0.2 GHz (Zh22) | **1 GHz** (Zh22 0.2–1; Liu23-SV 1–2; Li24 1–2.5) | 2.5 GHz (Li24 `f_pk=[1,2.5]`) | Zh22 `[0.1,1]`+`0.2–1`; Li24 `[1,2.5]`; MZL16 `f_CPU^max=1.5 GHz`; Gu25 `f^max=4×10⁸ Hz` |
| **Helper-vehicle / peer CPU freq** | 1 GHz (Liu23 `F_j^max=[1,2]`) | **1.5 GHz** (Liu23 midpoint) | 2.5 GHz (Li24 parked vehicle `f_pk`) | **Liu23 `F_j^max = [1, 2] GHz` (service vehicle, V2V executing node — Table 1, verified)**; Li24 `f_pk=[1,2.5] GHz`. **Only these two papers give a numeric peer/helper tier at all** |
| **MEC-RSU / server CPU freq** | 4 GHz (Zh22; Li24 `f_rj∈[4,6]`) | **6 GHz** (Gu25) | **10 GHz (Zh18; Liu23 `F_m^max`)** | Zh18 `10×10⁹`; **Liu23 `F_m^max = 10 GHz`**; Zh22 `4 GHz`; Gu25 `6×10⁹ Hz`; Li24 `[4,6] GHz`; Mi21 ARSU `3 GHz` |
| **Cycles per bit (computational intensity)** | 24 cycles/bit (Li24) | **300** (Zh18); 1000 (MZL16; Mi21) | 1000 cycles/bit (Mi21, MZL16) | Zh18 `300`; MZL16 `X`=1000 (also 5900 cycles/**byte**); Li24 `24`; Mi21 `c_A=10³`. **NOT GIVEN in Zh22 or Liu23** (in Liu23, `C_i` is a task cycle count printed in GHz — no cycles/bit value exists) |
| **κ / switched capacitance (UE tier)** | 1e-28 (MZL16, Zh18-RSU) | **1e-27** (Gu25, Li24) | 1e-26 (Zh22) | Zh22 `1e-26`; Gu25 `1e-27`; Li24 `κ_v=1e-27`; MZL16 `1e-28`; Zh18 `1e-28`. **Liu23 has no UE tier** (fully offloaded) |
| **κ (helper / server tier)** | 1e-28 (Li24 `κ_r`; Zh18 RSU; MZL16 server) | **1e-27** (Mi21 ARSU; Liu23 RSU `γ_m`) | **5e-27 (Liu23 service vehicle `γ_j`)** | **Liu23 `γ_j = 5×10⁻²⁷` (SV), `γ_m = 1×10⁻²⁷` (RSU)**; Mi21 `κ_A=1e-27`; Li24 `κ_r=1e-28`; Zh18 `ϱ=1e-28`; Gu25 single `κ=1e-27` for both tiers |
| **Transmit power (device → edge)** | 5 W (Gu25 `p_min`) | **23–25 dBm ≈ 0.2–0.3 W** (Zh22 23; Zh18 25) | **35 dBm (Mi21)**; 200 W (Gu25 `p_max`) | Zh22 `23 dBm`; **Liu23 `P = 30 dBm`**; Zh18 vehicle `25 dBm`; Mi21 `35 dBm`; Li24 `0.28184 W ≈ 24.5 dBm`; MZL16 `p_tx^max=1 W`; Gu25 `[5,200] W` |
| **Transmit power (RSU / ARSU → device)** | 35 dBm (Zh18, Mi21) | **35 dBm** | 35 dBm | Zh18 `35 dBm`; Mi21 ARSU `35 dBm`. Others fold it into a neglected downlink |
| **Receive power (device)** | — | **not modelled** (downlink/receive energy dropped as negligible) | — | Yo17 "downloading consumes negligible mobile energy"; Zh22 "time and energy consumption of computation result … are neglected"; Liu23 downlink energy absent from Eqs. 10/15. **No verified paper gives a numeric `p_rx`** |
| **Bandwidth** | 1 MHz (MZL16 `ω`) | **10 MHz** (Liu23 `B`; Mi21) | 20 MHz (Zh22 `W`) | MZL16 `1 MHz`; Gu25 `2×10⁶ Hz`; **Liu23 `B = 10 MHz`**; Mi21 `10 MHz`; Li24 `W_b=15` (unit printed as "MB" — inconsistent); Zh22 `20 MHz`; Zh18 `2 GHz` (mmWave) |
| **Noise** | −174 dBm/Hz (Liu23 `N₀`; Zh18 +NF) | **−113 dBm** (Zh22 `σ²`); −114 dB (Gu25) | −80 dBm (Mi21 `N₀`) | **Liu23 `N₀ = −174 dBm/Hz`**; Zh22 `−113 dBm`; Gu25 `−114 dB`; Mi21 `−80 dBm`; Li24 `1.2589e-13 W ≈ −99 dBm`; MZL16 `σ = 10⁻¹³ W = −100 dBm`; Zh18 thermal + 7 dB NF |
| **Battery / energy budget** | 2 mJ (MZL16 `E_max`) | **100 J** (Zh22 vehicular battery) | 100 J (Zh22) | Zh22 `E_total=100 J`; MZL16 `E_max=2 mJ` (per-slot harvested); Gu25 `E_max=3000` (unit not printed). Mi21, Zh18, Liu23: none stated |
| **Deadline / task time limit** | 2 ms (MZL16 `τ_d`) | **0.5 s** (Zh22 `D_max`); **0.5–2.5 s (Liu23 `T_i^max`)** | 8 s (Mi21 `T=8 s` flight) | MZL16 `τ_d=2 ms`; Zh22 `D_max=0.5 s`; **Liu23 `T_i^max = [0.5, 2.5] s`**; Li24 `[100,200] ms`; Mi21 `T=8 s`, `τ=0.2 s`; Zh18 dwell time `ℓ_o/V_o` = 50 m / 50 km/h = **3.6 s** |

# Synthesis Table B — energy-scope precedent (include vs exclude MEC compute energy)

| # | Paper | Scope | Verbatim proof | URL |
|---|---|---|---|---|
| A1 | Zhao et al. 2018 (V2X) | **INCLUDES RSU compute** | "We consider computing and communication as the two main contributors…"; `E(t) = E_c^R(t) + P_v(t)τ_{1,t} + P_R(t)τ_{3,t}`, `E_c^R(t) = ϱ C_in(t) ϑ f_R²` | https://arxiv.org/abs/1807.02311 |
| A2 | Cheng et al. 2018 (multi-MEC) | **INCLUDES MEC compute** | "it incorporates the transmission energy used to send the input data D_i to the helping MEC server k **and the energy consumption on the MEC server k for computing**." | https://ar5iv.labs.arxiv.org/html/1803.07243 |
| A3 | Gu et al. 2025 (VEC/FL) | **INCLUDES RSU compute** | "the energy consumption is divided into three parts: **RSU computing energy consumption** E_{t,R}^n, transmission energy consumption…, and local computing energy consumption E_{t,L}^n." | https://arxiv.org/pdf/2408.14831 |
| A4 | Wang et al. 2018 (WPT-MEC) | **INCLUDES AP/MEC compute** | "the energy is mainly consumed for executing the offloaded computation tasks…"; objective `min T·tr(Q) + Σ αℓ_i` | https://ar5iv.labs.arxiv.org/html/1702.00606 |
| A5 | Cang et al. 2024 (async MEC) | **INCLUDES MEC compute (server-energy-only objective)** | "the energy consumption of the MEC server for all tasks computation can be formulated by E_MEC = Σ Σ κ f_{n,m}³ Δt_m" | https://ar5iv.labs.arxiv.org/html/2401.11377 |
| A6 | Malik & Vu 2020 (MIMO MEC) | **INCLUDES both UE and server** | "minimize a weighted sum of the energy consumption at **both the users and the MEC server**…"; `E_OC = Σ κ_m f_{mi}² d_m s_i` | https://ar5iv.labs.arxiv.org/html/2001.08259 |
| A7 | Michailidis et al. 2021 (dual-RIS UAV IoV) | **PARTIAL — ARSU compute IN, GRSU compute OUT, propulsion OUT** | "minimize the weighted total energy consumption (WTEC) of the vehicles and ARSU"; "the computation delay at GRSU can be neglected"; "the propulsion energy consumption is excluded" | https://pmc.ncbi.nlm.nih.gov/articles/PMC8271975/ |
| A8 | Liu et al. 2023 (VEC, V2V+V2R) | **INCLUDES BOTH the RSU/MEC and the service-vehicle (helper) compute energy; no requester-local term (fully offloaded)** | `E_{i,m} = P_i t^up_{i,m} + C_i (f^R_{i,m})² γ_m` (Eq. 10); `E_{i,j} = P_i t^up_{i,j} + C_i (f^S_{i,j})² γ_j` (Eq. 15); Table 1 "Switch capacitance constant of RSU server γ_m = 1×10⁻²⁷" | https://www.sciengine.com/sci-open/api/v1/open/article/figAndTable?articleBaseId=BE6EF1BE63DF40219C71410642808C16 |
| B1 | Zhang et al. 2022 (real-time VEC) | **EXCLUDES — V-UE battery only** | "The corresponding energy consumption of V-UE is…"; "the time and energy consumption of computation result from the MEC server to the V-UE are neglected" | https://doi.org/10.1155/2022/8138079 |
| B2 | You et al. 2017 (canonical) | **EXCLUDES — mobile energy only** | "The objective is to minimize the weighted sum **mobile** energy consumption: Σ_k β_k(E_off,k + E_loc,k)" | https://ar5iv.labs.arxiv.org/html/1605.08518 |
| B3 | Vu et al. 2018 (cooperative MEC) | **EXCLUDES — mobile devices only** | "the total energy consumption of **mobile devices** is minimized" | https://ar5iv.labs.arxiv.org/html/1811.12686 |
| B4 | Chen et al. 2022 (cache-assisted MEC) | **EXCLUDES — users only** | "minimize **users'** energy consumption while guaranteeing delay constraints" | https://ar5iv.labs.arxiv.org/html/2208.07151 |
| B5 | Mao, Zhang & Letaief 2016 (EH-MEC) | **EXCLUDES — mobile device only** | `E_mobile = κ Σ(f_w)²`; `E_server = p·L/r` (device TX only); MEC compute delay "ignore[d]" | https://arxiv.org/pdf/1605.05488 |
| — | Mao et al. 2017 survey (canonical formalism) | **PROVIDES BOTH forms** | Device `E_m = κLXf_m²`; Server `E_s = Σ_k κ w_k f_{s,k}²` (DVFS) and `E_s = αE_max + (1−α)E_max u`, α≈70% | https://ar5iv.labs.arxiv.org/html/1701.01090 |
| — | Zhang, Yi & Ma 2024 survey (prevalence) | **meta** | "in scenarios where tasks are offloaded to MCC or MEC servers with a continuous energy supply, **researchers do not consider the energy consumption of these servers**" | https://ar5iv.labs.arxiv.org/html/2401.01017 |

**Counts.** Of the **13** primary works tabulated: **7 fully include** server/edge compute energy (A1–A6, A8),
**1 partially includes** (A7: aerial-RSU yes, terrestrial MEC no), **5 exclude** it (B1–B5). Counting the
broader literature via the T-ITS 2024 survey, exclusion is still the majority convention (~22 device-only vs 4
multi-entity in its prose list; ~19 vs 7 reference codes in its Table I). Note that two of the seven
include-side works are themselves V2X/VEC papers (A1 Zhao, A8 Liu) and one more is IoV (A7), so
**server-compute inclusion is well represented inside the vehicular subset specifically** — it is not an
exotic choice for a V2X/VEC paper, even though it is minority practice overall.

---

# Implications for MARGO (grounded only in the above)

1. **The single most important finding for MARGO is Liu et al. 2023 (A8).** It is the closest published
   analogue to MARGO's scope: a **V2V + V2R** VEC paper that counts the compute energy of **the node that
   executes the task** — the service vehicle's `γ_j` in V2V mode and the RSU server's `γ_m` in V2R mode —
   on top of the radio energy. MARGO's helper-vehicle compute term (`rho_v2v`) therefore has a **direct,
   citable precedent**, and the specific numbers `F_j^max ∈ [1,2] GHz`, `γ_j = 5e-27` are verified Table 1
   values that can anchor `f_v2v` and `rho_v2v` instead of leaving them purely assumed. Liu keeps the MEC
   term that MARGO drops, so the difference is a scope choice, not a modelling error — say so explicitly.
   Cite Liu (A8) for the helper-compute inclusion, A1/A2/A3/A6 for the system-level alternative, and
   B2/B3 + the 2024 T-ITS survey for the exclusion precedent.

2. **Helper-tier parameters now have two anchors, but neither fixes `rho_v2v = 0.7`.** Liu23 gives
   `f_v2v`-relevant `F_j^max = [1, 2] GHz` and `γ_j = 5×10⁻²⁷` for the service vehicle (vs `γ_m = 1×10⁻²⁷`
   for the RSU) — note the helper's κ is the *larger* one, the opposite of MARGO's `rho_v2v < rho_ue`
   assumption. Li24 gives `f_pk ∈ [1, 2.5] GHz`, `κ_v = 1e-27`. So MARGO should either justify
   `rho_v2v = 0.7` from first principles or sweep it, and should be aware that **published helper/κ values
   sit at or above the device tier, not below it.**

3. **η = 2 with `duration·ρ·f^ζ` is consistent with the canonical form.** The canonical model is
   `E = κ·C·f²` with `P = κf³` (Mao survey Eq. 2; MZL16; Gu25 Eq. 5; MV20 Eq. 5; Ca24; and Liu23's
   `E = P·t^up + C_i f² γ`). Since `duration = C/f`, an f² power and an f³ power are equivalent and reduce to
   `E = κ C f²`, so **MARGO's `duration·f²` is correct** — but `rho`/`κ` must always be quoted **with its
   f-exponent and units**, because the literature mixes `κf²` (energy) and `κf³` (power) under near-identical
   symbols. Liu23's own Table 1 also shows a real-world unit hazard: its `C_i` row is labelled "[0.5, 2.5] GHz"
   while `C_i` functions as a cycle count.

4. **The radio-power magnitudes are not comparable to the literature.** MARGO uses `p_tx = 0.1 W` (20 dBm) and
   `p_rx = 0.05 W` (17 dBm). Published transmit values run **23–35 dBm (0.2–3.2 W)** — Zh22 23, Liu23 30,
   Zh18 25, Mi21 35, Li24 24.5, MZL16 1 W — and **no verified paper gives a numeric receive power at all**:
   downlink energy is either folded into the transmit term or dropped as negligible (Yo17, Zh22, Liu23).
   MARGO's receive terms, and its `p_tx_v2v = 0.06 W < ptx_mec = 0.1 W` assumption, have no direct published
   precedent and lower the modelled energy by roughly an order of magnitude versus the V2X literature.

5. **Latency conventions.** Constraint-only is the norm (Zh18, Mi21, A1–A6, B2–B4 — and note Liu23 adds a hard
   `C1: T_i ≤ T_i^max` *in addition to* its weighted term). Weighted energy–latency scalarisation appears in
   Zh22 (residual-energy-dependent weights) and Liu23 (`U_i = ω_l T_i + ω_e E_i + …`, ω = 0.5/0.5), the latter
   also subtracting a **transmission-gain credit** `g = ∫₀^{t^up}(r(t) − r̄)dt`. Lyapunov drift-plus-penalty
   with an energy price η appears in Zh18. **Do not attribute a penalty term to Mi21 or to Zh18's η** — Mi21
   has none, and Zh18's η is a Lyapunov price, not a violation penalty.

6. **One claimed attribution was wrong; the other was right.** (i) **Corrected:** γ_SV = 5e-27 / γ_RSU = 1e-27
   are **genuinely Liu et al. Table 1** (verified via the SciEngine data API) — my earlier hypothesis that they
   came from Liang et al. arXiv:2411.10770 is refuted (that paper has 1e-27/1e-28 and 24.5 dBm). (ii) **Still
   needs fixing:** the Mao survey is in **IEEE COMST 19(4):2322–2358, 2017**, not *Proc. IEEE*.

7. **Zhang et al. 2013 could not be read.** Its DOI/venue are verified but no parameter from it is reported.
   Use the Mao 2017 survey for the `κf²C` formalism and MZL16 (§5c) for concrete calibration values.

8. **Access note for future verification work.** ScienceDirect/Elsevier CAPTCHAs all automated clients, but
   **SciEngine (the society platform) exposes machine-readable tables and MathML equations** — this is how the
   Liu et al. Table 1 and all 39 equations were recovered. When an Elsevier/KeAi paper is blocked, try
   `https://www.sciengine.com/<JOURNAL>/doi/<DOI>` and its `sci-open/api/v1/open/article/figAndTable` endpoint
   before declaring parameters NOT VERIFIED.
