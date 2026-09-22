# Literature Verification — "Real-Time Energy-Aware Offloading in Vehicular Networks with MEC"

## 1. Full citation

**Haibo Zhang, Xiangyu Liu, Xia Bian, Yan Cheng, Shengting Xiang**, "A Resource Allocation Scheme for Real-Time Energy-Aware Offloading in Vehicular Networks with MEC," *Wireless Communications and Mobile Computing*, vol. 2022, Article ID 8138079, 17 pages, 2022. **DOI: 10.1155/2022/8138079**. Open Access (CC BY 4.0). Academic Editor: Omprakash Kaiwartya. Received 11 Jun 2021; accepted 05 Dec 2021; published 10 Feb 2022.

- DOI/canonical: https://doi.org/10.1155/2022/8138079
- Publisher (Cloudflare-protected, HTTP 403 to automated fetch): https://onlinelibrary.wiley.com/doi/10.1155/2022/8138079
- **Full text actually read** (archived OA HTML, all sections + Table 1 + Table 2 + appendix): https://web.archive.org/web/20240423015325/https://www.hindawi.com/journals/wcmc/2022/8138079/
- **Full text actually read** (archived OA PDF, 17 pp., text + equations extracted): http://web.archive.org/web/20220423103105if_/https://downloads.hindawi.com/journals/wcmc/2022/8138079.pdf (live gold-OA PDF: https://downloads.hindawi.com/journals/wcmc/2022/8138079.pdf)
- **Full text actually read** (archived JATS XML with MathML; used to confirm Eq. 14/15 forms): http://web.archive.org/web/20220803150306id_/https://downloads.hindawi.com/journals/wcmc/2022/8138079.xml
- Metadata cross-check: https://api.crossref.org/works/10.1155/2022/8138079 ; https://api.semanticscholar.org/graph/v1/paper/DOI:10.1155/2022/8138079

Title is a **close variant**, not exact: project says "real-time energy-aware offloading in vehicular networks with MEC"; actual title begins "A Resource Allocation Scheme for …". Same authors/venue confirmed.

**Not a DAG paper, and not a V2V/helper-vehicle offloading paper.** Task model is one *independent* task per V-UE, `A_{i,j} = {d_{i,j}, c_{i,j}, D^max_{i,j}}` (no precedence/DAG). The formulation has only two execution modes: **local V-UE** or **MEC-RSU**. No helper-vehicle compute tier exists; a V2V link appears only as a label in Figure 1 and is never used in any equation or in P1. If the MARGO project needs a DAG/V2V helper vehicle baseline, this paper cannot supply it.

## 2. Energy scope — V-UE only (CONFIRMED)

The energy in the objective is **only the V-UE's energy** (local-compute energy + uplink transmit energy). **No MEC/RSU server compute energy term, no downlink/receive energy term, no helper-vehicle energy term.** MEC compute time *does* enter latency (`c_{i,j}/f_C`) but MEC compute *energy* never enters the objective.

Verbatim proof (all from the archived full text URLs above):

> "The corresponding energy consumption of V-UE is expressed as $e^{C}_{i,j}=\sum_{n=1}^{N} a_{i,j,n}p_{i,j,n}\dfrac{d_{i,j}}{R_{i,j}}$." (Eq. 7 caption text)

> "Here, we assume that the time and energy consumption of computation result from the MEC server to the V-UE are neglected in this case because the size of output data is much smaller than the size of input data and the download data rate is very high in general, which is similar to the study [32]."

> "the transmission expenditure between the MEC server and the RSU is neglected [31, 32]."

> "it is critical to make an efficient offloading decision and study the trade-off between the energy consumption of vehicle units and the latency of the corresponding tasks."

> "we propose a real-time energy-aware offloading scheme to study the trade-off between the energy consumption and the task latency of the vehicle units"

> "The objective is to minimize the overhead of V-UEs" (immediately before P1, Eq. 13)

> "During the execution of a task, both latency and energy consumption will affect the V-UEs (i.e., battery energy limitation of V-UEs)."

## 3. Exact energy formulas (transcribed from full text)

**Local compute energy** (Eq. 2):
$$e^{L}_{i,j} = k\,(f_{i,\mathrm{loc}})^{2}\,c_{i,j}$$
> "where $k = 10^{-26}$ is a coefficient that depends on the chip architecture [28, 29]."

**Uplink transmit energy for offloading** (Eq. 7) — the *only* offload energy:
$$e^{C}_{i,j} = \sum_{n=1}^{N} a_{i,j,n}\,p_{i,j,n}\,\frac{d_{i,j}}{R_{i,j}}$$

**Uplink rate** (Eq. 3), $w = W/N$:
$$R_{i,j,n} = w\log_2\!\left(1+\frac{p_{i,j,n}h_{i,j,n}}{\sigma^{2}+I_{i,j,n}}\right)$$

**Interference** (Eq. 4):
$$I_{i,j,n}=\sum_{k=1}^{U_l}\sum_{l=1,\,l\neq j}^{M} a_{k,l,n}p_{k,l,n}h^{j}_{k,l,n}$$

**Latencies** (Eqs. 1, 6):
$$t^{L}_{i,j}=\frac{c_{i,j}}{f_{i,\mathrm{loc}}},\qquad t^{C}_{i,j}=\frac{d_{i,j}}{R_{i,j}}+\frac{c_{i,j}}{f_{C}}$$

- **No** downlink/receive energy term. **No** $k(f_C)^2$-style MEC server energy term. **No** fixed circuit/idle energy.

## 4. Parameter table (Table 2, "Simulation parameters")

Source: archived full text (Table 2) — https://web.archive.org/web/20240423015325/https://www.hindawi.com/journals/wcmc/2022/8138079/

| Parameter | Value | Note |
|---|---|---|
| Max transmit power $p^{\max}$ | **23 dBm** | maximum, not fixed; allocated $p_{i,j,n}\le p^{\max}$ is a variable |
| Bandwidth $W$ | **20 MHz** | per-channel $w=W/N$ |
| MEC server CPU freq. $f_C$ | **4 GHz/cycle** | "The CPU frequency of MEC servers $f_C$" |
| Total CPU cycles $c_{i,j}$ | **[0.1, 1] GHz** | total cycles per task (Table 1) |
| Min V-UE CPU freq. $f^{\min}_{l}$ | **0.2 GHz/cycle** | |
| Max V-UE CPU freq. $f^{\max}_{l}$ | **1 GHz/cycle** | |
| Noise power $\sigma^{2}$ | **−113 dBm** | |
| Total energy / battery $E^{\text{total}}$ | **100 J** | "The total energy consumption $E^{\text{total}}$" |
| Max latency $D^{\max}_{i,j}$ | **0.5 s** | hard deadline (C1) |
| Max iterations $I_d$ | 600 | |
| Step sizes $\mu_1,\mu_2,\mu_3$ | $2\times10^{-18},\ 10^{-5},\ 10^{-10}$ | |
| Power allocation precision $\varepsilon$ | $10^{-5}$ | |
| V-UE queue size $|Q_u|$ | 10 | |
| Cloudlet queue size $|Q^{c}_{i}|$ | 4 | |
| Distance states $H$ | 4 | |
| Decision period $T$ | 60 s | |
| Input data size $d_{i,j}$ | **[300, 1600] KB** | |
| $k$ (switched capacitance) | $10^{-26}$ | from body text Eq. (2), **not** in Table 2 |

Scenario: 5 RSUs, 10 channels/RSU, 25 V-UEs/RSU, RSU communication radius 250 m; highway settings follow 3GPP TR 36.885.

**Cycles per bit / computational intensity:** NOT GIVEN. $c_{i,j}$ is defined (Table 1) as "the total number of CPU cycle to accomplish the computation task" — a total cycle count, not cycles/bit. No cycles-per-bit ratio is reported.

## 5. Latency handling

Latency enters **twice**:
1. **Weighted-sum scalarization in the objective** (Eqs. 9–12): $O^{L}_{i,j}=w^{t}_{i,j}t^{L}_{i,j}+w^{e}_{i,j}e^{L}_{i,j}$; $O^{C}_{i,j}=w^{t}_{i,j}t^{C}_{i,j}+w^{e}_{i,j}e^{C}_{i,j}$. The weights come from a **residual-energy-rate-dependent** factor (Eq. 8): $w'_{i,j}=w_{i,j}r^{E}_{i,j}$ with $r^{E}_{i,j}=\big(E^{\text{total}}-[(1-s_{i,j})e^{L}_{i,j}+s_{i,j}e^{C}_{i,j}]\big)/E^{\text{total}}$, then $w^{t}_{i,j}=w'_{i,j}$, $w^{e}_{i,j}=1-w'_{i,j}$.
2. **Hard deadline constraint** C1: $s_{i,j}\!\left(\frac{d_{i,j}}{R_{i,j}}+\frac{c_{i,j}}{f_C}\right)+(1-s_{i,j})\frac{c_{i,j}}{f_{i,\mathrm{loc}}}\le D^{\max}_{i,j}$.

Objective **P1** (Eq. 13), minimize over $s,p,a,f$:
$$\sum_{i=1}^{U_l}\sum_{j=1}^{M}s_{i,j}\Big[w^{t}_{i,j}\Big(\tfrac{d_{i,j}}{R_{i,j}}+\tfrac{c_{i,j}}{f_C}\Big)+w^{e}_{i,j}\sum_{n=1}^{N}a_{i,j,n}p_{i,j,n}\tfrac{d_{i,j}}{R_{i,j}}\Big]+\sum_{i=1}^{U_j}\sum_{j=1}^{M}(1-s_{i,j})\Big[w^{t}_{i,j}\tfrac{c_{i,j}}{f_{i,\mathrm{local}}}+w^{e}_{i,j}k(f_{i,\mathrm{local}})^{2}c_{i,j}\Big]$$
Other constraints: C2 energy $\le E^{\max}_{i,j}$; C3 $f^{\min}_{\mathrm{local}}\le f_{i,\mathrm{local}}\le f^{\max}_{\mathrm{local}}$; C4 $0\le\sum_n a_{i,j,n}p_{i,j,n}\le p^{\max}$; C5 interference $\le I$; C6 $\sum_n a_{i,j,n}\le1$; C7/C8 binary $a$, $s$.

## 6. Penalties

**No penalty terms exist.** Full-text search for "penalt", "dropout/drop out", "violat", "infeasib", "outage", "failure" returns **zero** matches. The only deadline mechanism is the hard constraint C1. The DRL stage uses a reward/cost, not penalties:
$$R(s,a)=U(s,a)-C(s,a),\quad U(s,a)=\rho\,(O^{L}_{i,j}+O^{C}_{i,j}),\quad C(s,a)=\eta_1 E(s,a)+\eta_2 T(s,a)$$
with $E(s,a)=a_0e^{L}_{i,j}+a_ie^{C}_{i,j}$, $T(s,a)=a_0t^{L}_{i,j}+a_it^{C}_{i,j}$ (Eqs. 34–37). Deadline violation/dropout is not modelled or penalised.

## 7. Verdict per claimed value

| Claim | Verdict | Evidence |
|---|---|---|
| V-UE CPU frequency = 0.2–1 GHz | **CONFIRMED** | Table 2: $f^{\min}_{l}=0.2$ GHz/cycle, $f^{\max}_{l}=1$ GHz/cycle — [full text](https://web.archive.org/web/20240423015325/https://www.hindawi.com/journals/wcmc/2022/8138079/) |
| MEC server CPU frequency = 4 GHz | **CONFIRMED** | Table 2: $f_C=4$ GHz/cycle — same URL |
| $k$ (switched capacitance) = 1e-26 | **CONFIRMED** | Eq. (2): "where $k=10^{-26}$ is a coefficient that depends on the chip architecture" — same URL |
| UE transmit power = 23 dBm | **CORRECTED: 23 dBm is the MAXIMUM** ($p^{\max}$), not a fixed operating power. The actual $p_{i,j,n}\in[0,p^{\max}]$ is an optimisation variable (constraint C4). | Table 2 + C4 — same URL |
| UE battery capacity = 100 J | **CONFIRMED** | Table 2 $E^{\text{total}}=100$ J; body: "$E^{\text{total}}$ is the battery capacity in Joules [34]" — same URL |
| "optimises ONLY the V-UE battery" | **CONFIRMED** for energy scope (objective contains only V-UE local-compute + uplink energy; MEC/RSU compute energy absent). Caveat: it is not a pure-battery objective — it is a weighted sum of V-UE energy **and** latency, with residual-energy-dependent weights. | Eqs. 7, 9–13 + quotes §2 — same URL |

Every number above is sourced to the archived full text and cross-checked against the Crossref record https://api.crossref.org/works/10.1155/2022/8138079.

## 8. Ambiguity / other candidates

This is the exact-title match; no competing paper matches the full parameter set. Near-neighbours found but not the target: "New energy-aware based offloading optimization method for collaborative edge computing for the Internet of Vehicles" (Elsevier, https://www.sciencedirect.com/science/article/abs/pii/S1570870526001368) and "An energy-aware distributed federated soft actor-critic framework for intelligent task offloading in vehicular MEC networks" (Ad Hoc Networks, https://www.sciencedirect.com/science/article/abs/pii/S1570870525002914). Neither is open access and neither was read.
