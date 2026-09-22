# Energy-scope verification: does the objective include the MEC/RSU server compute energy?

All quotes below were read from **open-access full text** (arXiv/ar5iv HTML or publisher HTML). Where a paper is also published in a paywalled IEEE journal, the quote URL is the open-access version I actually read. Nothing is paraphrased as a quote. No parameter value is reported unless it appeared in the text I read.

---

## CATEGORY A — objective EXPLICITLY INCLUDES server/edge compute energy

### A1. Wang, Xu, Wang, Cui — "Joint Offloading and Computing Optimization in Wireless Powered Mobile-Edge Computing Systems"

**Citation.** F. Wang, J. Xu, X. Wang, S. Cui, "Joint Offloading and Computing Optimization in Wireless Powered Mobile-Edge Computing Systems," *IEEE Transactions on Wireless Communications*, vol. 17, no. 3, pp. 1784–1797, Mar. 2018. DOI: 10.1109/TWC.2017.2785305. Preprint: arXiv:1702.00606.
**URL read.** https://ar5iv.labs.arxiv.org/html/1702.00606 (open access; the IEEE version is paywalled)
**Year / OA.** 2017 preprint, 2018 journal. Open access via arXiv.

**Verbatim scope quotes (server compute energy is inside the objective):**
> "As for the AP, the energy is mainly consumed for executing the offloaded computation tasks and transmitting the computation results back to the users."

> "Therefore, we adopt a simplified linear energy consumption model for the computation at the AP as  E_MEC = α Σ_{i=1}^{K} ℓ_i,  where α denotes the energy consumption per offloaded bit at the AP."

> "We aim to minimize the energy consumption at the AP (including the energy consumption Σ_{i=1}^{K} α ℓ_i in (5) for computation and T tr(Q) for WPT) while ensuring the successful execution of the K users' computation tasks per time block."

> "minimizing the total energy consumption at the AP including the radiated energy for WPT and the energy for computing the offloaded tasks"

**Objective as printed (P1), Eq. (9a):**
min_{Q⪰0, t, ℓ, f}  T·tr(Q) + Σ_{i=1}^{K} α·ℓ_i

**Energy formulas used.**
- AP/MEC compute (linear-per-bit, *not* κf²C): `E_MEC = α Σ_{i=1}^{K} ℓ_i`, α = energy per offloaded bit at AP.
- UE local compute (κf²C form): `E_loc,i = Σ_{n=1}^{C_i q_i} κ_i f_{i,n}²`, κ_i = effective capacitance coefficient (chip-architecture dependent).
- UE offloading: `E_offl,i = (t_i / g̃_i)·β(ℓ_i/t_i) + p_{c,i} t_i`, with `β(x) ≜ σ²(2^{x/B} − 1)`.
- Uplink rate: `r_i = B log₂(1 + p_i g̃_i /(Γσ²))`, Γ = capacity gap (set to 1).
- Harvested energy: `E_i = T ζ tr(Q H_i)`, ζ = EH efficiency.

**Parameters seen.** UE CPU frequency per cycle `f_{i,n} ∈ (0, f_i^max]`; cycles per bit `C_i`; capacitance `κ_i`; circuit power `p_{c,i}`; bandwidth `B`; noise `σ²`; block length `T`; AP energy-per-bit `α`. **No numeric values** were stated in the sections I read.

**Latency handling / penalties.** Hard constraint, no penalty term: local-compute time `Σ_{n=1}^{C_i q_i} 1/f_{i,n} ≤ T`. Server compute time explicitly assumed away: *"Due to the sufficient CPU capability at the MEC server, the computation time consumed at the MEC server are relatively small and negligible."* Result-download time/energy also ignored: *"we ignore the downloading time, i.e. t_i ≈ 0 … and also ignore the energy consumption for transmitting and receiving the computation results in this paper."* Energy-causality constraint `E_loc,i + E_offl,i ≤ E_i`.

---

### A2. Cang, Chen, Pan, Yang, Hu, Sun, Chen — "Joint User Scheduling and Computing Resource Allocation Optimization in Asynchronous Mobile Edge Computing Networks"

**Citation.** Y. Cang, M. Chen, Y. Pan, Z. Yang, Y. Hu, H. Sun, M. Chen, "Joint User Scheduling and Computing Resource Allocation Optimization in Asynchronous Mobile Edge Computing Networks," *IEEE Transactions on Communications*, vol. 72, no. 6, pp. 3378–3392, Jun. 2024. DOI: 10.1109/TCOMM.2024.3358237. Preprint: arXiv:2401.11377.
**URL read.** https://ar5iv.labs.arxiv.org/html/2401.11377 (open access)
**Year / OA.** 2024. Open access via arXiv.

**Verbatim scope quotes (objective is purely the MEC server's CPU energy):**
> "Besides, the energy consumption of the MEC server for all tasks computation can be formulated by  E_MEC = Σ_{n=1}^{N} Σ_{m=n+1}^{N+1} κ f_{n,m}³ Δt_m,  where κ denotes the energy coefficient of the MEC server."

> "Our goal is to minimize the MEC server's energy consumption of completing the tasks offloaded by all devices, which is formulated as an optimization problem as"

**Objective as printed, Eq. (2):**
min_{Δt, A, f}  Σ_{n=1}^{N} Σ_{m=n+1}^{N+1} κ f_{n,m}³ Δt_m

Note `κ f³ Δt = κ f² · (f Δt) = κ f² · (number of CPU cycles)` — i.e. this **is** the E = κ f² C server-CPU form, written with f³Δt.

**Other energy formulas.**
- Device offload transmit power (monomial model, order 3): `p_{k,n} = λ(r_{k,n})³ / h_k = λ(A_k)³ / (h_k (Δt_n)³)`; λ = energy coefficient related to bandwidth and noise power.
- Harvested energy: `E_k^H = Σ_{n=1}^{N} Σ_{i=0}^{n−1} a_{k,n} Δt_i h_k η P_0`; η = EH efficiency, `P_0` = server transmit power.
- Device energy-causality constraint (2a): `Σ_n a_{k,n} λ(A_k)³/(h_k (Δt_n)²) ≤ Σ_n Σ_{i=0}^{n−1} a_{k,n} Δt_i h_k η P_0`.

**Parameters seen.** Server energy coefficient `κ`; server max frequency `F_max`; task bits `A_k`; computation intensity `I_k` (CPU cycles/bit), so task cycles `F_k = A_k I_k`; period `T`; path-loss parameters. **No numeric values** in the sections read.

**Latency handling / penalties.** Hard constraints, no penalty: `Σ_{i=0}^{N+1} Δt_i ≤ T`; server frequency cap `Σ_{n=1}^{m−1} f_{n,m} ≤ F_max`; deadline/completion `Σ_n Σ_m a_{k,n} f_{n,m} Δt_m ≥ F_k`. The paper's novelty is *asynchronous* server execution, but the optimised energy is the server's only.

---

### A3. Malik & Vu — "Energy-Efficient Computation Offloading in Delay-Constrained Massive MIMO Enabled Edge Network Using Data Partitioning"

**Citation.** R. Malik, M. Vu, "Energy-Efficient Computation Offloading in Delay-Constrained Massive MIMO Enabled Edge Network Using Data Partitioning," *IEEE Transactions on Wireless Communications*, vol. 19, no. 10, pp. 6977–6991, Oct. 2020. DOI: 10.1109/TWC.2020.3007616. Preprint: arXiv:2001.08259 (preprint title omits "Computation").
**URL read.** https://ar5iv.labs.arxiv.org/html/2001.08259 (open access)
**Year / OA.** 2020. Open access via arXiv.

**Verbatim scope quotes (weighted sum of UE **and** MEC-server energy, with κf²C at both ends):**
> "we formulate a novel problem to minimize a weighted sum of the energy consumption at both the users and the MEC server under a round-trip latency constraint, using a combination of data partitioning, transmit power control and CPU frequency scaling at both the user and server ends."

> "We formulate a novel optimization problem to minimize the system's energy consumption, including both the users and the MEC, subject to a latency requirement. Our aim is to explore the benefit of computation offloading to meet a hard latency constraint while minimizing the energy consumption at both the user terminals and the MEC servers."

> "The formulation accounts for energy consumption at both the users and MEC ends, compared to current literature considering only one side [11], [12], [13], [14], [15]."

**Energy formulas used (verbatim transcription).**
- UE local compute, Eq. (4): `E_LC = Σ_{i=1}^{K} κ_i c_i (u_i − s_i) f_{u,i}²`, and `t_{L,i} = c_i(u_i − s_i)/f_{u,i}`.
- **MEC server compute, Eq. (5):** `E_OC = Σ_{i=1}^{K} κ_m f_{mi}² d_m s_i` — with "`d_m` is the number of CPU cycles required to compute one bit at the MEC, the CPU frequency `f_{mi}` is the computational rate assigned to the i-th user's task by the MEC, and `κ_m` is a hardware dependent constant of the MEC server." This is exactly `κ f² C` on the server (C = `d_m s_i`).
- MEC compute time, Eq. (6): `T_2 = max{t_{M,i}},  t_{M,i} = d_m s_i / f_{mi}`.
- UE offload energy, Eq. (3): `E_OFF = Σ_{i=1}^{K} p_{li} s_i /(B r_{u,i})`.
- AP downlink (result) energy, Eq. (9): `E_DL = Σ_{i=1}^{K} P η_{li} μ s_i /(B r_{d,i})`.
- Uplink massive-MIMO rate, Eq. (1): `r_{u,i} = ν log₂(1 + SINR^ul_{li}/Γ₁)`, `SINR^ul_{li} = N γ^l_{li} p_{li} / σ²_{1,li}`.

**Parameters seen.** Per-tier CPU frequencies: UE `f_{u,i}`, edge server `f_{mi}` (per-user, virtualization); cycles/bit: UE `c_i`, server `d_m`; capacitance: UE `κ_i` ("effective switched capacitance"), server `κ_m` ("hardware dependent constant of the MEC server"); bandwidth `B`; AP average transmit power `P`; power coefficients `η_{li}`; result/data ratio `μ`; antennas `N`; latency `T_d`. **No numeric values** in the parts I read.

**Latency handling / penalties.** Hard round-trip deadline `T_d` decomposed into three phases (offload `T₁ = max_i t_{u,i}`, compute `T₂ = max_i t_{M,i}`, download `T₃ = max_i t_{d,i}`), with the user's local computation allowed to span phases II and III. No penalty term in the objective.

---

## CATEGORY B — objective EXPLICITLY EXCLUDES MEC server compute energy (device / UE / vehicle only)

*Honest caveat: for all three papers below the in-scope statement is the explicit, unambiguous **device-only objective function**. I did not find, in the open-access full texts I could read, the literal sentence "the energy consumption of the MEC server is not considered / negligible" in any of these three. The meta-statement that this convention is the common one is in the survey quoted in the Prevalence section. The server's CPU frequency and cycles still appear in these papers — but only inside **delay** expressions or capacity constraints, never inside the energy objective.*

### B1. Vu, Huynh, Hoang, Nguyen, Dutkiewicz — "Offloading Energy Efficiency with Delay Constraint for Cooperative Mobile Edge Computing Networks"

**Citation.** T. T. Vu, N. Van Huynh, D. T. Hoang, D. N. Nguyen, E. Dutkiewicz, "Offloading Energy Efficiency with Delay Constraint for Cooperative Mobile Edge Computing Networks," *Proc. IEEE GLOBECOM 2018*, pp. 1–6. DOI: 10.1109/GLOCOM.2018.8647856. Preprint: arXiv:1811.12686.
**URL read.** https://ar5iv.labs.arxiv.org/html/1811.12686 (open access)
**Year / OA.** 2018. Open access via arXiv.

**Verbatim scope quotes (device-only objective):**
> "we consider a joint offloading and resource allocation problem in which the total energy consumption of mobile devices is minimized."

> "the objective is to minimize the total energy consumption of all mobile devices and all delay constraints must be satisfied, i.e.,  (P1) min_{x_i, r_i} Σ_{i=1}^{N} E_i"

> "To minimize the total energy consumption for mobile users in the network and meet all tasks' delay requirements, we first formulate the joint task offloading and resource allocation optimization problem for all mobile users and edge nodes."

**Energy formulas used.**
- Local processing, Eq. (1): `E_i^l = v_i C_i`, and `T_i^l = C_i / f_i^l`, where `v_i` = "the consumed energy per CPU cycle".
- MEC-node processing, Eq. (2): `E_ij^f = E_ij^u + E_ij^d`, with `E_ij^u = e_ij^u D_i^i` and `E_ij^d = e_ij^d D_i^o`; `e_ij^u`, `e_ij^d` = device energy per unit of transmitted/received data.
- Cloud processing, Eq. (5): `E_ij^c = E_ij^f = E_ij^u + E_ij^d` (device transmit/receive energy again).
- No server-side term anywhere in `E_i`.

**Parameters seen.** Device processing rate `f_i^l`; task cycles `C_i`; device energy/cycle `v_i`; input/output data `D_i^i`, `D_i^o`; device energy per unit data `e_ij^u`, `e_ij^d`; MEC node resources `(R_j^u, R_j^d, F_j^f)` = uplink rate, downlink rate, CPU cycle rate; cloud link rate `r^{fc}`, cloud rate `f^c`. **No numeric values** in the text read.

**Latency handling / penalties.** Hard per-task deadline `T_i ≤ T_i^r` (no penalty). Server CPU rate `f_ij^f` appears **only in delay**: `T_ij^f = D_i^i/r_ij^u + D_i^o/r_ij^d + C_i/f_ij^f`, as does cloud rate in `T_ij^c = D_i^i/r_ij^u + D_i^o/r_ij^d + (D_i^i+D_i^o)/r^{fc} + C_i/f^c`. Constraints `Σ_i f_ij^f ≤ F_j^f`, `Σ_i r_ij^u ≤ R_j^u`, `Σ_i r_ij^d ≤ R_j^d`.

---

### B2. Chen, Yi, Alam, Nallanathan — "Dynamic Task Software Caching-Assisted Computation Offloading for Multi-Access Edge Computing"

**Citation.** Z. Chen, W. Yi, A. S. Alam, A. Nallanathan, "Dynamic Task Software Caching-Assisted Computation Offloading for Multi-Access Edge Computing," *IEEE Transactions on Communications*, vol. 70, no. 10, pp. 6950–6965, Oct. 2022. DOI: 10.1109/TCOMM.2022.3200109. Preprint: arXiv:2208.07151.
**URL read.** https://ar5iv.labs.arxiv.org/html/2208.07151 (open access)
**Year / OA.** 2022. Open access via arXiv.

**Verbatim scope quotes (users-only objective; server is a constraint, not an energy term):**
> "we formulate a joint task software caching update (TSCU) and computation offloading (COMO) problem to minimize users' energy consumption while guaranteeing delay constraints, where the limited cache size and computation capability of the MEC server, as well as the time-varying task demand of users are investigated."

> "We formulate a joint TSCU and COMO problem in a multi-channel wireless environment to minimize the average energy consumption of mobiles users over each time slot while satisfying the task execution delay tolerance."

**Energy formulas used (device side only).**
- Local computing, Eq. (3): `E_{k,f}^L = ζ (f_k^L)² S_f ≥ ζ S_f³/τ²`; with the deadline-optimal choice `f_k^L = S_f/τ`, so `E_{k,f}^L = ζ S_f³/τ²`; ζ = "the energy coefficient of mobile devices, determined by the chip architecture".
- Offloading (non-caching), Eq. (5): `E_{k,f,t}^O = p_k (I_f + D_f)/r_{k,t}` ("includes the transmit energy consumption of both input parameters and the corresponding software").
- Rate, Eq. (2): `r_{k,t} = (B/M) log(1 + p_k h_k /(Σ_{n≠k, α_{n,t}=α_{k,t}} p_n h_n + σ²))`.

**Parameters seen.** Device CPU capability `f_k^L = S_f/τ`; device energy coefficient `ζ`; task computation load `S_f`; transmit power `p_k`; bandwidth `B` split into `M` orthogonal channels; input size `I_f`, software size `D_f`; MEC cache size `C`; MEC CPU capability `f_C` with `f_C ≫ f_k^L`. **No numeric values** in the text read.

**Latency handling / penalties.** Hard per-slot deadline, no penalty: `T_{k,f,t}^O = S_f/f_C + (I_f + D_f)/r_{k,t} ≤ τ`. The MEC server's capability appears **only** as the delay term `S_f/f_C` and as the cache constraint `Σ_f b_f^(t) D_f ≤ C`.

---

### B3. You, Huang, Chae, Kim — "Energy-Efficient Resource Allocation for Mobile-Edge Computation Offloading"

**Citation.** C. You, K. Huang, H. Chae, B.-H. Kim, "Energy-Efficient Resource Allocation for Mobile-Edge Computation Offloading," *IEEE Transactions on Wireless Communications*, vol. 16, no. 3, pp. 1397–1411, Mar. 2017. DOI: 10.1109/TWC.2016.2633522. Preprint: arXiv:1605.08518.
**URL read.** https://ar5iv.labs.arxiv.org/html/1605.08518 (open access)
**Year / OA.** 2016 preprint, 2017 journal. Open access via arXiv. (Highly cited baseline: 1400+ Crossref citations.)

**Verbatim scope quotes (weighted-sum *mobile* energy, cloud energy excluded):**
> "The objective is to minimize the weighted sum mobile energy consumption:  Σ_{k=1}^{K} β_k (E_off,k + E_loc,k),  where the positive weight factors {β_k} account for fairness among mobiles."

> "for the TDMA MECO system with infinite or finite computation capacity, the optimal resource allocation is formulated as a convex optimization problem for minimizing the weighted sum mobile energy consumption under the constraint on computation latency."

> "the radio and computation resources were jointly allocated to minimize the mobile energy consumption under offloading latency constraints."

**Objective as printed (P1):**
min_{ℓ_k, t_k}  Σ_{k=1}^{K} β_k [ (t_k/h_k²) f(ℓ_k/t_k) + (R_k − ℓ_k) C_k P_k ]
s.t.  Σ_k t_k ≤ T,  Σ_k C_k ℓ_k ≤ F,  t_k ≥ 0,  m_k⁺ ≤ ℓ_k ≤ R_k

**Energy formulas used.**
- Local computing: `E_loc,k = (R_k − ℓ_k) C_k P_k`, with `C_k` = CPU cycles per bit and `P_k` = energy per cycle.
- Offloading, Eq. (2): `E_off,k = p_k t_k = (t_k/h_k²) f(ℓ_k/t_k)`, with `f(x) = N₀(2^{x/B} − 1)`.
- Rate, Eq. (1): `r_k = B log₂(1 + p_k h_k²/N₀)`.
- No MEC/cloud energy term exists in the objective.

**Parameters seen.** `C_k` (CPU cycles/bit, UE), `P_k` (energy per cycle, UE), `F_k` (UE CPU cycles/s), `R_k` (bits), bandwidth `B`, noise `N₀`, slot `T`, cloud capacity `F` (CPU cycles per slot), fairness weights `β_k`. **No numeric values** in the text read.

**Latency handling / penalties.** Hard constraints, no penalty: `C_k(R_k − ℓ_k)/F_k ≤ T`; cloud-load cap `Σ_k C_k ℓ_k ≤ F`; optional non-negligible cloud time `t_comp = (Σ_k ℓ_k C_k)/F′` folded into the latency constraint. The cloud's compute *energy* is never modelled — the cloud appears only through capacity `F`/`F′`. Note also: *"Cloud computing has small latency; the downloading consumes negligible mobile energy"*.

---

## Prevalence evidence

**The single most authoritative statement I found** is in a peer-reviewed survey (IEEE T-ITS 2024), which states the *exclude* convention is the typical one for grid-powered MEC/MCC servers, and cites ~22 papers for it, while separately listing the *include* camp:

> "Typically, in scenarios where tasks are offloaded to MCC or MEC servers with a continuous energy supply, researchers do not consider the energy consumption of these servers, as noted in several studies [60], [90], [61], [91], [75], [67], [92], [81], [128], [59], [68], [93], [69], [94], [70], [71], [72], [95], [62], [129], [18], [63]. However, there are many research works that consider the energy of edge and cloud servers in the optimization [79], [80], [78], [130], this is because cloud service providers and data centers need to optimize their energy usage to reduce their environmental impact and lower operating costs."

— S. Zhang, N. Yi, Y. Ma, "A Survey of Computation Offloading With Task Types," *IEEE Transactions on Intelligent Transportation Systems*, vol. 25, no. 8, pp. 8313–8333, Aug. 2024, DOI 10.1109/TITS.2024.3410896. Quoted from the open-access preprint: https://ar5iv.labs.arxiv.org/html/2401.01017 (arXiv:2401.01017).

The same survey tabulates the split by hand in its Table I ("Papers corresponding to each category", row *Optimization Objective → Energy Saving Maximization*):

> "1)  Energy consumption of UE only | [61]-[66] | [29], [56], [57], [59], [60], [67]-[74]
> 2)  Energy consumption of multiple entities (including various combinations of UE, relay node, edge or cloud server) | [8], [19], [75]–[78] | [79], [80]"

i.e. roughly **19 codes for UE-only vs 7 codes for multi-entity** in that table, and the prose list above gives **22 vs 4**. The same source also notes: *"energy consumption of cloud server in MCC will not be considered usually"* and classifies compute energy as *"energy consumption of local computing and RPN computing."*

Corroborating independent statement from an *include*-side paper (Malik & Vu, IEEE TWC 2020, URL as in A3):

> "Most existing works have considered energy minimization at only one side of the network, either the users [11][12][13], or the MEC-server when it is energy constrained, such as a UAV-MEC [14][15]."

**Summary.** Device-side-only energy objectives are the dominant convention for battery-powered-UE MEC offloading (the survey's counts ≈ 22 vs 4; Malik & Vu's reading is the same); server-side compute energy enters the objective mainly when (i) the edge node itself is energy-constrained (UAV/fog/EH nodes), (ii) the authors deliberately study *system-level* energy including the operator's cost, or (iii) the objective is defined as the *server's* energy only. For a V2V-assisted MEC paper, the defensible framing is therefore: state the scope explicitly, and note that including RSU/server compute energy is the minority-but-defensible choice used by system-level-energy papers such as Malik & Vu (TWC 2020) and Cang et al. (TCOM 2024).

---

## Unverified / not accessible

- **DAG-specific papers.** I could not obtain verifiable open-access full text for the DAG/dependency-aware offloading papers surfaced by search (IEEE TMC "DAG-ED" 2024, IEEE TMC 10301697, ACM TOIT 10.1145/3762993, Elsevier/ACM dependency-aware offloading). Retrieval failures: `stc.computer.org` PDF endpoint (fetch failed), MDPI (`403 Access Denied`), Wiley (`403`), Nature (`cross-origin redirect`), IEEE Xplore (paywalled). **NOT VERIFIED**: none of the DAG-specific papers listed above has an energy scope I can attest to from full text. The six verified papers are task- or data-partition-level, not DAG-level.
- **Explicit "negligible MEC server energy" sentence.** **NOT VERIFIED**: I could not retrieve an open-access full text containing the literal statement that MEC-server energy is ignored/negligible. The exclusion is verified here only through explicit device-only objective functions plus the survey's meta-statement.
- **Numeric parameter values** (frequencies in GHz, κ, C, bandwidth, transmit power) were not present in the sections of these papers that my fetches returned; none are reported above, by design.
