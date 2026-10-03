# Energy Model Literature Review (v2 system model)

Status: primary-literature review, **not** a spec change. Supersedes nothing yet.
Date of review: **2026-10-03** (all web retrievals performed on this date).
Reviewer: delegated literature-review agent.
Target artifact: an energy model for MARGO that is **computable from scheduler events**
(per-task executed CPU work, per-hop radio service time, outages, retries/restarts).

> **Evidence discipline used throughout.** Every equation transcribed below was read in the
> original document text (publisher PDF or publisher/author HTML with embedded LaTeX). Where a
> document could not be read, this is stated explicitly and **no equation number, symbol value,
> or numerical parameter is reported for it**. Nothing in this file is inferred from a secondary
> source, a search-engine snippet, or a survey. All arithmetic that I performed myself (unit
> conversions, implied coefficients) is explicitly labelled **DERIVED BY REVIEWER**.

---

## 0. The MARGO formula set of record (the thing being audited)

Read from the repository (read-only inspection):

- `env/mec_offloaing_envs/scheduler/energy_model.py` (module docstring + `TierSpec`)
- `spec/frozen_experiment.yaml` (`energy_model:` block)
- `spec/OBJECTIVE_AND_ENERGY.md`, `spec/decisions/ADR-001-energy-scope.md`, `spec/PHASE5_ENERGY_LOCK.md`

```text
C_i            = workload_bytes * 8 * cycles_per_bit          # cycles (task property)
T_cpu(i, x)    = C_i / f_x                                    # s
E_cpu(i, x)    = kappa_x * C_i * f_x^2                        # J   == (kappa_x * f_x^3) * T_cpu
P_cpu(x)       = kappa_x * f_x^3                              # W
E_tx(hop)      = P_tx(source tier) * (active transmission service time)   # J
E_rx           = OPTIONAL; include_rx_energy default false
```

Frozen configuration values (`spec/frozen_experiment.yaml`, `energy_model:`):

| symbol | value | unit | recorded provenance in repo |
|---|---|---|---|
| `cycles_per_bit` | 300.0 | cycles/bit | "Zhao2018 Table I; sweep 300 / 500 / 1000" |
| `kappa_ue` (UE / vehicle) | 1.0e-27 | s²/cycle (SI, f in Hz) | "chosen inside the verified band (kappa 1e-28..1e-26, f 0.2..2.5 GHz); kappa=1e-27 matches the Gu2025/Liang2024 vehicle tier" |
| `f_ue` | 1.0e9 | Hz | same band |
| `kappa_helper` | 5.0e-27 | s²/cycle | "Liu et al., DCN 9(6):1399-1410, 2023, Table 1 (F_j in [1,2] GHz, gamma_j = 5e-27)" |
| `f_helper` | 1.5e9 | Hz | same source (midpoint of the stated band) |
| `kappa_mec` | 1.0e-27 | s²/cycle | "Liu et al., DCN 9(6), 2023, Table 1 (F_m = 10 GHz, gamma_m = 1e-27)" |
| `f_mec` | 10.0e9 | Hz | double-sourced with Zhao et al. arXiv:1807.02311 Table I |
| `ue_tx_w` | 1.0 | W (= 30 dBm) | "Liu2023 Table 1" |
| `mec_tx_w` | 3.162 | W (= 35 dBm) | "Zhao2018 Table I (separate provenance)" |
| `helper_tx_w` | 1.0 | W | "assumed equal to the UE class; no helper TX published" |
| `ue_rx_w`, `helper_rx_w` | null | W | "NOT literature-pinned; needed only if include_rx_energy" |
| `server_static_power_w` | null | W | sensitivity only |

Accounting scopes available: `requester` < `mobile` (UE+HELPER) < `system` (UE+HELPER+MEC).
Frozen `energy_scope` in the YAML is `system`; ADR-001 / PHASE5_ENERGY_LOCK state the paper scope
as `total_mobile_joules = UE + HELPER` with MEC compute excluded from the primary objective.

**Note on the scope conflict (not a literature finding, a repo observation).** The frozen YAML
sets `energy_scope: system` while ADR-001 and PHASE5_ENERGY_LOCK pin the paper scope to
`mobile`. The code has an explicit guard (`validate_frozen_energy_scope`) that only fires when
`energy.accounting_primary_scope` is also present. This review does not change it; it is flagged
because every source below differs in exactly this dimension.

---

## A. Comparison table

Abbreviations used in the table:
`CPU-dyn` = dynamic CPU energy (switched-capacitance / per-cycle form);
`CPU-static` = static/idle "keep-alive" power while not computing;
`TX` = transmitter radio energy; `RX` = receiver radio energy;
`MEC` = server-side compute energy included; `BH` = backhaul/fronthaul/propagation energy;
`FULL` = original document read in full text; `INACC` = inaccessible.

### A.1 Required comparison table (one row per source)

| # | Citation | Venue + year | Peer-reviewed vs preprint | Atomic vs DAG | Accounting scope | Energy components covered | Objective vs constraint | State / action definitions | Key omitted assumptions | Source equations / section | Accessibility | Applicability to MARGO |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 | Z. Gao, G. Luo, S. Zhan, B. Liu, L. Huang, H.-C. Chao, "ST-HO: Symmetry-Enhanced Energy-Efficient DAG Task Offloading Algorithm in Intelligent Transport System" | *Symmetry* (MDPI) **2024**, 16(2), 164; DOI 10.3390/sym16020164 | Peer-reviewed journal (MDPI, open access CC BY) | **DAG** (explicit subtask dependency graph `G`, `r_xy`, `V_local` for local-only subtasks) | **Vehicle/on-board only** for the objective (Eq. 16 charges the on-board device); MEC compute time appears only as `t_mec = C/f_mec` and never as MEC energy | `CPU-dyn` (as linear `p_local·t_local`), **`CPU-static` yes** (`p_basic·t_mec` while offloaded), `TX` (upload `p_up`, and download `p_down` attributed to the on-board device), **`RX` yes** (download, Eq. 9–10), `MEC` **no**, `BH` **no** | **Objective** = minimize system energy `min_S E(S)` (Eq. 17); delay is the constraint `t_total < t_local` | Decision: binary vector `S=[s_1..s_N]`, `s_x=0` MEC / `s_x=1` local (Eq. 3). No RL state/action; solved by SA+Tabu hybrid search | No switched-capacitance coefficient; constant per-cycle energy `p_local/f_local`; no server energy; no queueing/contention; no outage/retry energy; delay threshold is the all-local makespan; `R_vg`/`R_gv` assumed static per decision | Eq. (2)–(17); §2.3 "Energy Consumption Model" (§2.3.1/2.3.2/2.3.3), §2.4 "Problem Formulation"; parameters Table 1 (§4) | **FULL** (PDF, mdpi-res.com attachment) | **High** for the cost structure and for `p_basic` (idle) and `p_down` (RX); **DAG-with-per-location-dependency** structure is the closest published analogue to MARGO's multi-DAG scheduler. Its linear CPU power must be converted (see §C) |
| S2 | P. Parastar, G. Caso, J. A. Omaña Iglesias, A. Lutu, Ö. Alay, "Energy-Efficient Task Computation at the Edge for Vehicular Services" | IEEE/IFIP **NOMS 2025**, pp. 1–9; DOI 10.1109/NOMS57970.2025.11073636. Preprint arXiv:2511.18449 (23 Nov 2025) | **Peer-reviewed conference** (I read the arXiv v1 preprint text; the paper states it is published in NOMS 2025) | **Atomic** (one task per car per slot, no dependency edges) | **Both** (explicit objective on cars AND MEC nodes, weighted by `alpha`) | `CPU-dyn` (explicit `kappa` form, local **and** MEC compute), `CPU-static` **no**, `TX` (car uplink `p_c^tx`, MEC-node transmit `p_b^tx`), `RX` **no** (downlink latency and energy neglected by assumption), `MEC` **yes**, `BH` **partially** (a regressed transport latency `0.014·dist+1.225` is charged, but its energy is not broken out) | **Objective** (Eq. 10) `Min Σ_t Σ_k e_k(t)`; deadline + resource + battery are **constraints** (Eq. 11a) | State: per-MEC/node `{lat, lon, cpu_m(t), trans_m(t)}`, per-car `{b(t), lat_b, lon_b, mob_c, cpu_c, E_c, M_c, k_c}`. Action: MEC-node selection `map_m`, local-vs-offload `y_k`, resource fraction `f_k`, MEC selection `x_k^m`. Multi-agent PPO (LAPPO / MALAPPO) | Downlink latency/energy neglected; no outage/retry; no helper/V2V tier; `kappa` unit convention not stated (see §C flag U-2); transport latency regression is operator-specific | Eq. (3)–(12); §IV "System model", §IV-A (Eq. 6, 7), §IV-B (Eq. 8, 9 + inline), §IV-C (Eq. 10, 11a–11g); parameters in §VI | **FULL** (arXiv HTML v1 with embedded LaTeX; also PDF) | **Very high for the CPU term** — Eq. (7) is algebraically identical to MARGO's `E = kappa·C·f²`. Also the only source read here that prices **MEC compute** with the same `kappa` family, which is what MARGO's `system` scope needs |
| S3 | Y. Sun, J. Hu, et al. (author list not fully extracted), "A cooperative multi-agent optimization approach for task offloading in vehicular edge computing systems" (COMA2C) | *Discover Computing* (Springer) **2025**; DOI 10.1007/s10791-025-09887-6 | Peer-reviewed journal (gold open access, CC BY-NC-ND) | **Atomic** (per-slot per-VU tasks; `s_k(t)` input size, `c_k` cycles/bit; no dependency graph) | **Vehicle only** (the objective and battery model are per-VU; RSU compute energy is never priced) | `CPU-dyn` **yes** (Eq. 8, `eps·f²·c·s`), `CPU-static` **no**, `TX` **yes** (Eq. 9, `s/r · p`), `RX` **no**, `MEC` **no**, `BH` **no** | **Objective** = weighted latency+energy cost (Eq. 12, `eta_1 D + eta_2 T`); energy enters **also as a constraint** via the battery recursion (Eq. 11) and the battery band (Eq. 16) | State per VU: `S_k={S_task, S_gain, S_power, S_resource, S_battery}` with `S_task=[s_k,c_k,D_max]`. Actions: `(x_{m,k}, p_{m,k}, f_{m,k})` — discrete offload gate at 0.5 threshold plus continuous power and local-frequency fractions in `[0,1]`. Hierarchical master/client actor-critic with Q-networks | `eps` units not stated; objective as printed puts `T_k^total` in the second (energy-weighted) term while the reward in Eq. 22 uses `E_k^total` — an internal inconsistency in the paper (flagged in §B.3); no MEC energy; no RX | Eq. (1)–(23); §3.4 "Energy consumption model" (Eq. 8, 9, 10), §3.5 "Energy harvesting model" (Eq. 11), §4 objective/constraints (Eq. 12–18), §5 reward (Eq. 22) | **FULL** (publisher HTML with embedded LaTeX, retrieved through a text-rendering proxy after direct access was blocked) | **Very high.** Eq. (8) is MARGO's CPU term verbatim (`E = eps·f²·cycles`). Eq. (9) is MARGO's TX term verbatim (TX-only, no RX). Eq. (11) supplies a battery/energy-constraint pattern MARGO could adopt for a long-horizon constraint |
| S4 | W. M. H. Almuseelem, "Deep Reinforcement Learning-Enabled Computation Offloading: A Novel Framework to Energy Optimization and Security-Aware in Vehicular Edge-Cloud Computing Networks" | *Sensors* (MDPI) **2025**, 25(7), 2039; DOI 10.3390/s25072039; PubMed 40218550 / PMC11991124 | Peer-reviewed journal (MDPI, open access CC BY) | **Atomic** (task tuple `(beta_ij, sigma_ij, delta_ij)`; no dependencies) | **Vehicle only** (server energy never appears; Eq. 5 is the *vehicle's* transmit energy) | `CPU-dyn` **yes but in per-cycle form** (`E^L = delta·eta`, Eq. 3), `CPU-static` **no**, `TX` **yes** (Eq. 5, 9), `RX` **no** (output-data energy explicitly disregarded), `MEC` **no**, `BH` **no** (only propagation delay `varpi` for the cloud tier) | **Objective** `min_alpha ΣΣ E_ij` **and** energy **constraint** `C1: E_ij − E^L_ij ≤ 0` (Eq. 24) | Decision vectors: offloading `alpha_ijk ∈{0,1}`, security `tau_ij ∈{0,1}`, caching `upsilon_ij ∈{0,1}`, server resource `f^RSU_i`, `f^UAV_i`. Deep-Q-learning agent chooses `alpha` | `eta_i` (energy per CPU cycle) is **never given a numerical value** in the paper — the entire local-energy term is un-calibrated; no MEC compute; encryption energy is charged to the vehicle only; no RX; no outages | Eq. (2)–(24); §3.3 Communication Model, §3.4 Computation Model (§3.4.1 Eq. 3–4, §3.4.2 Eq. 5–12), §3.5 Security (Eq. 15–19), §3.6 caching (Eq. 22), §3.7 problem (Eq. 24) | **FULL** (PDF, mdpi-res.com attachment) | **Medium.** Useful as the per-cycle-energy alternative family (`E = cycles × J/cycle`) and as the only source here that models an *extra* per-cycle cost (encryption) — relevant if MARGO ever adds a crypto/reliability cost. Unusable for calibration (no `eta` value) |
| S5 | Y. Chen, Z. Huang, Y.-W. Zhang, W.-W. Fang, N. N. Xiong, "LyDRL: Lyapunov-guided Deep Reinforcement Learning for Stable Task Offloading in Connected Autonomous Vehicles" | *ACM Transactions on Autonomous and Adaptive Systems* **2025**; DOI 10.1145/3715333 | Peer-reviewed journal (ACM) | **UNVERIFIED** | **UNVERIFIED** | **UNVERIFIED** | **UNVERIFIED** | **UNVERIFIED** | **UNVERIFIED** | **UNVERIFIED** | **INACC** — see §B.5. `dl.acm.org` returns an anti-bot HTTP 403 challenge for the article page and the PDF, direct and through a text-rendering proxy; Semantic Scholar reports `isOpenAccess=false`, `openAccessPdf.status="CLOSED"`. Only title/authors/venue/year and part of the public abstract were verified, via the Semantic Scholar Graph API | **Not assessable.** A second search pass found no author copy, repository copy, or preprint. Do **not** cite its Lyapunov formulation in MARGO |
| S6 | Y. Liu, D. Li, H. Wu, Z. Sun, W. Qin, J. Li, H. Du, G. Sun, "Task Offloading and Resource Allocation for MEC-assisted Consumer Internet of Vehicle Systems" (JTOCRA) | arXiv:2508.15795v1 [cs.NI], **13 Aug 2025** (submitting author Z. Sun). No journal reference on the abstract page as of 2026-10-03 | **Preprint** (no journal reference listed) | **Atomic** (task `Psi_v^t = (l_v^t, mu_v^t, tau_v^t)`; no dependency edges) | **Both** (vehicle energy in Eq. 10/12 and an MEC-server energy term `eps·l·mu` in Eq. 12/14, plus separate energy budgets `E_v` and `E_m`) | `CPU-dyn` **yes** (Eq. 10, `gamma_v (F_v)^2 l mu`), `CPU-static` **no**, `TX` **yes** (Eq. 12, `p_v l / R`), `RX` **no** (result feedback neglected), `MEC` **yes** (per-cycle `eps` term), `BH` **no** | **Objective** = weighted cost `C_v = w_D D_v + w_E E_v` (Eq. 15) minimized over `K, F` (Eq. 16a); **constraints**: energy budgets `E_v ∈ [5,25] J`, `E_m = 1000 J` carried in the state | State (Eq. 17, 18): `{q_v, F_v, E_v, l_v, mu_v, tau_v, q_m, F_m, E_m}` (per-vehicle queues plus MEC-server queues). Action (Eq. 19): `{k_n, f_n}` = offload target + allocated server frequency. Multi-agent DDPG | `gamma_v` unit convention is ambiguous and internally inconsistent with Table II's "GHz" label (see §C flag U-1); queue/Kalman mobility model; no DAG; no RX; no outage energy | Eq. (9)–(14) for the delay/energy terms, Eq. (15)–(16a) objective, Eq. (17)–(19) state/action, Table II parameters, §III-D "Energy Consumption" (Eq. 10), §III-D2 (Eq. 12), §III-E (Eq. 14) | **FULL** (arXiv HTML with embedded LaTeX) | **High for the CPU term** (Eq. 10 is MARGO's `kappa·C·f²` with `C = l·mu`), **high for MEC-side per-cycle energy**, and it is the only 2025 source read here that carries **explicit per-vehicle and per-server energy budgets** — directly relevant if MARGO adds a long-horizon energy constraint. Its Table II is the *only* place in this review where a numeric `gamma` (1e-28) and a numeric per-cycle MEC energy (8.2e-28 J) are both printed |
| S7 | S. H. Ahmadpanah, "Semantic-Aware LLM Orchestration for Proactive Resource Management in Predictive Digital Twin Vehicular Networks" (SP-LLM) | arXiv:2508.09149v1, **Aug 2025** | **Preprint** | **Atomic** (task arrivals + queues; no dependency graph read) | **System** (`E_sys`, i.e. a system-level energy penalty) | Objective/decomposition only — the paper defines a drift-plus-penalty surrogate with an energy penalty term; no component-by-component CPU/TX breakdown was located in the text I read | **Lyapunov drift-plus-penalty with queue stability as a long-term objective** converted to per-slot decisions (Eq. 1); energy appears as a penalty weight inside the bound | Per-slot objective over `w(t), a(t)` conditioned on queue state `Q(t)`; `Delta_L(t)` = Lyapunov drift, `C_sys(t)` = operational cost | No explicit **virtual energy-deficit queue** was located; energy is folded into the penalty term via a time-varying weight. Math rendering in the HTML dropped some symbols | Eq. (1) and the following unnumbered drift-plus-penalty bound; §3.3.2 "Proactive Optimization Problem Formulation" | **Partial / FULL of the model section** (arXiv HTML; only the formulation section was read in detail) | **Medium–low as an energy model; medium as a constraint mechanism.** If MARGO ever needs a long-horizon energy constraint, this is an accessible example of the drift-plus-penalty transformation on a VEC problem. It supplies **no** energy equation usable as a base model |
| S8 | S. Liu, J. Tian, C. Zhai, T. Li, "Joint computation offloading and resource allocation in vehicular edge computing networks" | *Digital Communications and Networks* **2023**, 9(6), 1399–1410; DOI 10.1016/j.dcan.2022.12.002 | Peer-reviewed journal (gold open access CC BY-NC-ND) | **UNVERIFIED** | **UNVERIFIED** | **UNVERIFIED** | **UNVERIFIED** | **UNVERIFIED** | **UNVERIFIED** | **UNVERIFIED** | **INACC** (verification attempt failed: ScienceDirect `/pdfft` HTTP 403, publisher SciEngine returns a JavaScript shell only, text-rendering proxy blocked). Bibliographic metadata verified via Crossref (title, authors, journal, volume 9, issue 6, pages 1399–1410, Dec 2023, CC BY-NC-ND) | **This is the source MARGO's frozen config already pins `kappa_helper = 5e-27` and `kappa_mec = 1e-27` to.** Those two numbers are therefore **UNVERIFIED by this review**. Recommend treating them as "cited" not "verified" until someone reads Table 1 of the original. (Outside the requested 2024–2026 window in any case) |

### A.2 Supporting detail: venue provenance

| # | Verified bibliographic facts | How verified |
|---|---|---|
| S1 | *Symmetry* 2024, 16(2), 164; authors Zhibin Gao, Gaoyu Luo (corresponding), Shanhao Zhan, Bang Liu, Lianfen Huang, Han-Chieh Chao | Read from PDF page 1 header + Crossref/Semantic Scholar record |
| S2 | NOMS 2025 (IEEE/IFIP Network Operations and Management Symposium), pp. 1–9; DOI 10.1109/NOMS57970.2025.11073636; arXiv:2511.18449 [cs.NI] submitted 23 Nov 2025 | arXiv abstract page (shows the comment "Published in: IEEE/IFIP NOMS 2025" and the journal reference) |
| S3 | *Discover Computing* 2025; DOI 10.1007/s10791-025-09887-6; DBLP key `journals/ir/SunHGW25`; Semantic Scholar `isOpenAccess=true`, license CC BY-NC-ND | Semantic Scholar Graph API + publisher HTML |
| S4 | *Sensors* 2025, 25(7), 2039; DOI 10.3390/s25072039; PubMed 40218550; PMC11991124 | Semantic Scholar Graph API + PDF |
| S5 | *ACM TAAS* 2025; DOI 10.1145/3715333; authors Yanming Chen, Ziyang Huang, Yi-Wen Zhang, Wei-Wei Fang, Neal N. Xiong; `isOpenAccess=false` | Semantic Scholar Graph API (metadata only) |
| S6 | arXiv:2508.15795v1 [cs.NI], 13 Aug 2025; authors Yanheng Liu, Dalin Li, Hao Wu, Zemin Sun, Weihong Qin, Jun Li, Hongyang Du, Geng Sun | arXiv abstract page + arXiv HTML author span |
| S7 | arXiv:2508.09149v1; author Seyed Hossein Ahmadpanah | arXiv HTML author span |
| S8 | *Digital Communications and Networks* 9(6):1399–1410, Dec 2023; authors Shuang Liu, Jie Tian, Chao Zhai, Tiantian Li; license CC BY-NC-ND | Crossref REST API |

---

## B. Equation-by-equation detail (accessible sources only)

Units are quoted **exactly as the source states them**. Where a source does not state a unit, the
entry says `not stated in source` — I do not infer a unit into the transcription. Reviewer
inferences are in separate, explicitly labelled notes.

### B.1 S1 — ST-HO (Gao et al., *Symmetry* 2024, 16(2), 164)

Section: **§2.3 "Energy Consumption Model"**, §2.3.1 Local Computing, §2.3.2 MEC Computing,
§2.3.3 Results Transmission; §2.4 Problem Formulation.

**Link rate — Eq. (2)** (§2.3):

```text
R = B * log2( 1 + p*h / (B*N0) )
```

- `N0` — noise power spectral density (unit not stated at this point; the symbol is introduced as "the noise power spectral density").
- `B` — "the communication bandwidth between the vehicle and the RSU"; Table 1 gives `B = 10 MHz`.
- `p` — "the transmission power of the signal sender" (unit not stated in the text; Table 1 gives the corresponding concrete powers in W).
- `h` — channel gain.
- `R` — transmission rate; unit not stated explicitly, but the expression `B·log2(1+·)` with `B` in Hz gives bit/s.
- **Reviewer note:** the argument `p·h/(B·N0)` is only dimensionless if `h` is dimensionless and `N0` is W/Hz.

**Offloading decision — Eq. (3)** (§2.3):

```text
s_x = 0   if subtask x is computed at MEC
s_x = 1   if subtask x is computed at local
```

- `s_x ∈ {0,1}` — dimensionless binary decision identity.

**Local computing time — Eq. (4)** (§2.3.1):

```text
t_local_x = C_x / f_local
```

- `C_x` — **"M cycles"** as printed ("where C_x (M cycles) is the computation amount of subtask x"). Note: the text labels the unit as *M cycles*; the symbol itself is therefore in units of 10⁶ cycles.
- `f_local` — "the local computation capacity of the vehicle device"; Table 1 gives `f_local = 1 GHz`.
- `t_local_x` — seconds (not stated, but consistent with the other time equations).

**Local computing energy — Eq. (5)** (§2.3.1):

```text
e_local_x = p_local * t_local_x
```

- `p_local` — Table 1 gives `p_local = 1 W`.
- `e_local_x` — joules (unit not printed; follows from W × s).

**Idle/basic energy while offloaded — Eq. (6)** (§2.3.2):

```text
e_basic_x = p_basic * t_mec_x
```

- `t_mec_x = C_x / f_mec` (stated inline in §2.3.2, with `f_mec` = "the computational capability of MEC"; Table 1 gives `f_mec = 5 GHz`).
- `p_basic` — "the power of the onboard devices to maintain basic operation"; Table 1 gives `p_basic = 0.1 W`.
- **This is a static/idle power term that MARGO does not currently have.**

**Offload transmission energy — Eq. (7)** (§2.3.2):

```text
e_trans_x = p_up * t_trans_x
```

- `t_trans_x = W_x / R_vg` ("the transmission time of offloading subtask x to MEC server"), `R_vg` = "the communication transmission rate from the vehicle to the MEC server".
- `W_x` — **"KB"** as printed ("W_x (KB) represents the amount of data corresponding to the basic information that subtask x needs to provide").
- `p_up` — "the communication power when the onboard device is uploading"; Table 1 gives `p_up = 0.5 W`.

**Result upload (MEC→local boundary) — Eq. (8)** (§2.3.3):

```text
e_up_xy = p_up * t_up_xy        with   t_up_xy = d_xy / R_vg
```

**Result download (local→MEC boundary) — Eq. (9)** (§2.3.3):

```text
e_down_xy = p_down * t_down_xy  with   t_down_xy = d_xy / R_gv
```

- `p_down` — "the communication power of downloading"; Table 1 gives `p_down = 0.2 W`.
- `R_gv` — "the communication transmission rate from the MEC server to the vehicle".
- **This is the download/receive-side power. In Eq. (10) this energy is charged to *the onboard device*, so `p_down = 0.2 W` is the closest thing to a literature-pinned UE receive power found in this review.** (Caveat: the paper is loose about whose power `p_down` is; see §C flag U-5.)

**Per-edge transmission energy and time — Eq. (10), (11)** (§2.3.3):

```text
e_xy = (1 - s_x) * e_up_xy + s_x * e_down_xy        (10)
t_xy = (1 - s_x) * t_up_xy + s_x * t_down_xy        (11)
```

with `t_xy = e_xy = 0` when `s_x = s_y` (i.e. no boundary crossing).

**Completion / start times — Eq. (12), (13), (14), (15)** (§2.4):

```text
t_mcom_x  = t_exec_x + t_mstart_x                                            (12)
t_exec_x  = s_x * t_local_x + (1 - s_x) * t_edge_x                           (13)
            where t_edge_x = t_trans_x + t_mec_x     (stated inline, §2.3.2)
t_mstart_x = max_{(x,y) in G} [ r_xy * t_mcom_y + | s_x - s_y | * t_xy ]     (14)
t_total   = t_mcom_N - t_mstart_1                                            (15)
```

- `r_xy ∈ {0,1}` — dependency indicator defined in §2.2 ("we define a binary vector r_xy ∈ {0,1} to represent the dependency between subtasks").
- `d_xy` — data transferred between subtasks `x` and `y` (unit not stated at the point of use).
- **Transcription caveat:** the extracted text layer of the PDF garbles the absolute-value bars in Eq. (14) and (16) (they appear as `h`/`i` glyph fragments). I reconstructed `|s_x − s_y|` from the surrounding prose, which explicitly says `t_xy = e_xy = 0` when `s_x = s_y` and that the term accounts for "transferring calculation results when interdependent subtasks are computed at different locations".

**Total system energy — Eq. (16)** (§2.4):

```text
E(S) = SUM_{v_x in V} [ s_x * e_local_x + (1 - s_x) * e_edge_x ]
     + SUM_{(x,y) in G} | s_x - s_y | * e_xy
s.t.  s_x in {0,1}
      s_x = 1, for all v_x in V_local
      S = [s_1, s_2, ..., s_N]
```

with `e_edge_x = e_trans_x + e_basic_x` (stated in the sentence immediately preceding Eq. (16)).
`V_local` = the set of subtasks that "can only be computed locally" (§2.4).

**Optimization problem — Eq. (17)** (§2.4):

```text
min_S  E(S)
s.t.   t_total < t_local
       s_x in {0,1}
       s_x = 1, for all v_x in V_local
       S = [s_1, ..., s_N]
```

`t_local` is defined as "the time when all subtasks are executed locally, i.e. S = [1,1,...,1]".

**Simulation parameters — Table 1 ("Parameters for simulation"), §4:**

| parameter | value as printed |
|---|---|
| `T` (initial SA temperature) | 100 |
| `T_end` | 0.01 |
| `a` (cooling factor) | 0.992 |
| `f_local` | 1 GHz (cited to ref. [25]) |
| `f_mec` | 5 GHz |
| `p_local` | 1 W |
| `p_up` | 0.5 W |
| `p_down` | 0.2 W |
| `p_basic` | 0.1 W (cited to ref. [30]) |
| `B` | 10 MHz (cited to ref. [12]) |

**Not present in ST-HO:** any switched-capacitance coefficient, any frequency-dependent CPU power,
any MEC-server compute *energy*, any queueing/contention, any outage or retry energy, any
per-hop *service* time accounting (transmission time is `bytes / rate`, a nominal quantity, not an
observed service interval).

---

### B.2 S2 — Parastar et al. (NOMS 2025 / arXiv:2511.18449)

Sections: **§IV System model**, §IV-A "Latency and Energy with Local Task Computation",
§IV-B "Latency and Energy with Task Computation at MEC", §IV-C "Problem Formulation".
Equation numbers below are the numbers printed in the arXiv v1 text (verified in the HTML).

**Task definition** (§IV, inline, unnumbered):

```text
ci_k = data_k * cl_k
```

- `data_k` — task data size; §VI gives `data_k ∈ [200, 1000] Kbytes`.
- `cl_k` — computation intensity; §VI gives `cl_k = 25 cycles/bit`.
- `ci_k` — task computation in **cycles**.
- Note: the paper mixes Kbytes (data) with cycles/**bit** (intensity) without stating a
  conversion. See §C flag U-3.

**Latency, per task — Eq. (3)** (§IV, "Given the above notation…"):

```text
l_k(t) = y_k(t) * l_k^loc(t) + SUM_m x_k^m(t) * l_k^m(t)
```

**Weighted combined energy — Eq. (4)** (§IV):

```text
e_k(t) = e_k^c(t) + alpha * e_k^off(t)
```

- `alpha` — "balances the importance of local energy relative to MEC energy consumption";
  §VI: `alpha = 1` unless stated otherwise; the paper also sweeps `alpha ∈ {0 … 1}`.

**Local vs offloaded car energy — Eq. (5)** (§IV):

```text
e_k^c(t)   = y_k(t) * e_k^{c,loc}(t) + (1 - y_k(t)) * e_k^{c,off}(t)
e_k^off(t) = SUM_m x_k^m(t) * e_k^m(t)
```

- `y_k(t) = 1 - SUM_m x_k^m(t)` (stated inline) — local-execution indicator.
- `x_k^m(t) ∈ {0,1}` — offload-to-MEC-node-`m` indicator.

**Local computing latency — Eq. (6)** (§IV-A):

```text
l_k^loc(t) = ci_k / ( f_k(t) * cpu_c(t) )
```

- `f_k(t) ∈ (0,1]` — "the CPU frequency ratio allocated to the task" (dimensionless fraction).
- `cpu_c(t)` — the car's computational capacity; §VI gives `cpu_c ∈ [1,2] GHz`.
- `l_k^loc(t)` — seconds (not stated; consistent with `ci_k` in cycles and capacity in GHz ⇒ ns scale, and the latency thresholds are in **ms**).

**Local computing energy — Eq. (7)** (§IV-A) — **the key equation for MARGO**:

```text
e_k^{c,loc}(t) = ci_k * kappa * [ f_k(t) * cpu_c(t) ]^2
```

Verbatim surrounding text: *"As for the energy, we adopt a widely used model for energy
consumption (used, for example, in [20], [40], [41]), so that e_k^{c,loc}(t) is as follows: …
where, kappa = 10^-11 is the effective switched capacitance depending on the chip architecture
[40]."*

- `kappa = 10^-11` — **"effective switched capacitance"**; **the paper does not state a unit for
  `kappa`, nor a unit for `cpu_c` inside this equation** (Table/§VI says GHz). See §C flag U-2.
- The functional form is **identical to MARGO's**: `E = kappa · C · f²` with `C = ci_k` and
  effective frequency `f_k(t)·cpu_c(t)`.

**MEC-path latency decomposition — Eq. (8)** (§IV-B):

```text
l_k^m(t)      = l_k^{m,comm}(t) + l_k^{m,comp}(t)
l_k^{m,comm}(t) = l_k^air(t) + l_k^{m,ts}(t) + l_k^{m,tx}(t)
```

**MEC-path energy decomposition — Eq. (9)** (§IV-B):

```text
e_k^{c,off}(t) = e_k^air(t)
e_k^m(t)       = e_k^{m,tx}(t) + e_k^{m,comp}(t)
```

**Over-the-air energy — inline, unnumbered (§IV-B)**:

```text
e_k^air(t) = l_k^air(t) * p_c^tx
```

- `p_c^tx` — car transmit power; §VI: **`p_c^tx = 0.5 W`** (cited to refs [48], [49]).

**Transport latency — inline, unnumbered (§IV-B)**:

```text
l_k^{m,ts}(t) = 0.014 * dist_{b,m}(t) + 1.225
```

described as "a liner regression model derived from real-world traces captured on the network of
the MNO under study". `dist_{b,m}` in the paper's distance unit (km, given the 40×40 km² area).

**MEC-node transmission latency and energy — inline, unnumbered (§IV-B)**:

```text
l_k^{m,tx}(t) = data_k / trans_m(t)
e_k^{m,tx}(t) = l_k^{m,tx}(t) * p_b^tx
```

- `trans_m(t)` — MEC node's "data transmission rate"; §VI: `R_m ∈ {10, 25, 100} Gb/s`.
- `p_b^tx` — base-station/MEC transmit power; §VI: **`p_b^tx = 1 W`**.
- **This is a server-side transmit energy term — MARGO does not have one in the primary model.**

**MEC compute latency and energy — inline, unnumbered (§IV-B)**:

```text
l_k^{m,comp}(t) = ci_k / ( f_k(t) * cpu_m(t) )
e_k^{m,comp}(t) = ci_k * kappa * [ f_k(t) * cpu_m(t) ]^2
```

- `cpu_m(t)` — MEC-node available computational capacity; §VI: `CPU_m ∈ [40,60] GHz`.
- `f_k(t) ∈ (0,1]`, `cpu_m(t) ≤ CPU_m`.
- **Same `kappa = 10^-11` as the car.** This is the strongest published precedent for MARGO
  charging MEC compute with the *same functional form* as UE compute. (MARGO currently uses the
  same numeric `kappa` for UE and MEC but the same frequency — see §C.)

**Objective — Eq. (10)** (§IV-C):

```text
Min  SUM_t SUM_k e_k(t)
```

**Constraints — Eq. (11a)** (and following labels, §IV-C):

```text
l_k(t) <= l_k^max                              for all k, t
f_k(t) * y_k(t) <= 1                           for all k, t
SUM_k f_k(t) * x_k^m(t) <= 1                   for all m, t
e_k^c(t) <= E_c(t)                             for all c, t
y_k(t) + SUM_m x_k^m(t) = 1                    for all k, t
x_k^m(t) in {0,1}                              for all k, m, t
```

**Battery-state / energy-queue transition — inline, unnumbered (§IV):**

```text
E_c(t+1) = E_c(t) - e_k^c(t)
```

- `E_c(t)` — car battery energy level; §VI: `E_c ∈ [50,150] mJ` initial.
- This is the paper's local reserve constraint. It is a **one-step battery recursion, not a
  virtual queue** — no Lyapunov drift term is used.

**Reward — Eq. (12)** (§V):

```text
r_c(t) = -penalty                                 if m not in M_c  or  l_k > l_k^max  or  e_k^c > E_c
       = -[ e_k(t) / ( l_k(t) / l_k^max ) ]       otherwise
```

**Other §VI parameters relevant to a unit audit:**

| parameter | value as printed |
|---|---|
| `cpu_c` | [1, 2] GHz |
| `CPU_m` | [40, 60] GHz |
| `R_m` | {10, 25, 100} Gb/s |
| `E_c` (initial battery) | [50, 150] **mJ** |
| `data_k` | [200, 1000] Kbytes |
| `cl_k` | 25 cycles/bit |
| `l_k^max` | {10, 30, 50, 100} ms |
| `p_c^tx` | 0.5 W |
| `p_b^tx` | 1 W |
| `alpha` | 1 (swept 0…1) |

**Not present in Parastar:** CPU static/idle power; RX energy on any hop (the text says downlink
latency is neglected "assuming that the task execution result is minimal in size"); outages/retry
energy; V2V/helper tier; DAG dependencies.

---

### B.3 S3 — COMA2C (Sun et al., *Discover Computing* 2025)

Sections: §3.3 Time delay model, §3.4 Energy consumption model, §3.5 Energy harvesting model,
§4 problem formulation, §5 algorithm. Equation numbers are the printed numbers (verified against
the publisher HTML, which retains the LaTeX source).

**Uplink rate — Eq. (2)** (§3.2/3.3):

```latex
r_{k}^{t} = \frac{B^{e}}{|K|} \cdot \log_{2}\!\left(1 + \frac{p_{k}^{T} \cdot \rho \cdot (d_{k,0}^{t})^{-\alpha} \cdot |g_{k,0}^{t}|^{2}}{N_{0} \cdot (B^{e}/|K|)}\right)
```

- `B^e` — "total wireless bandwidth available for task offloading in the system, which is equally
  shared among the |K| task-generating VUs". §5: **20 MHz total**.
- `N_0` — "power spectral density of additive white Gaussian noise". §5: **N_0 = −174 dBm/Hz**.
- `p_k^T` — transmission power of VU k.
- Path loss model given in §5: `PL_dB(d) = 127 + 30 log d` (dB), `h(d) = 10^(−PL_dB(d)/10)`.

**Local delay — Eq. (3)**:

```latex
D_{k}^{\text{local}}(t) = \frac{s_{k}(t) \cdot c_{k}}{f_{k}(t)}
```

- `s_k(t)` — "the input data size of the task generated at time slot t".
- `c_k` — "the number of CPU cycles required per bit of data".
- `f_k(t)` — "the amount of local computational resources allocated by the VU".

**Transmission delay — Eq. (4)**: `D_k^tran(t) = s_k(t) / r_k^t`
**RSU compute delay — Eq. (5)**: `D_k^com(t) = n_active · s_k(t)·c_k / f_0`
**Offload delay — Eq. (6)**: `D_k^offload(t) = D_k^tran(t) + D_k^com(t)`
**Total delay — Eq. (7)**: `D_k^total(t) = (1−x_k)·D_k^local(t) + x_k·D_k^offload(t)`

**Local computation energy — Eq. (8)** (§3.4) — **the key equation for MARGO**:

```latex
E_{k}^{\text{local}}(t) = \varepsilon \cdot f_{k}^{2}(t) \cdot c_{k} \cdot s_{k}(t)
```

Verbatim surrounding text: *"When a VU k chooses to process the task locally at time slot t, the
energy consumption primarily results from CPU execution. Based on the classical CPU energy
consumption model, the local computation energy consumption can be expressed as: … where
`epsilon` is the energy coefficient related to the hardware characteristics of the device's
processor."*

- `varepsilon` — **unit not stated in the paper.** SYMBOL only; no numeric value is given anywhere
  in the paper (checked the simulation-setup section). See §C flag U-4.
- `f_k(t)` — same symbol as in Eq. (3); the paper does not state whether it is in Hz or GHz, nor
  whether it is a frequency or a resource-capacity quantity. Note Eq. (3) uses `f_k(t)` in the
  position of a *rate* (cycles/s) and Eq. (17) bounds it by `f_k^max`.
- This is **algebraically MARGO's `E = kappa · C · f²`**, with `C = c_k · s_k(t)` (cycles) and
  `kappa ≡ varepsilon`.

**Offloading energy — Eq. (9)** (§3.4):

```latex
E_{k}^{\text{offload}}(t) = \frac{s_{k}(t)}{r_{k}^{t}} \cdot p_{k}(t)
```

Verbatim: *"When a VU chooses to offload the task to the RSU, the primary source of energy
consumption is the wireless data transmission, while the preprocessing energy is neglected."*

- `p_k(t)` — VU transmission power; constrained by Eq. (14) to `0 ≤ p_k(t) ≤ p_k^max`.
- **This is exactly MARGO's TX term: power × active transmission duration. No RX term.**

**Total energy — Eq. (10)** (§3.4):

```latex
E_{k}^{\text{total}}(t) = (1 - x_{k}) \cdot E_{k}^{\text{local}}(t) + x_{k} \cdot E_{k}^{\text{offload}}(t)
```

**Battery recursion (energy-harvesting model) — Eq. (11)** (§3.5):

```latex
b_{k}(t+1) = \min\!\left(b_{k}^{\max},\ \max\!\left(0,\ b_{k}(t) - E_{k}^{\text{total}}(t) + e_{k}(t)\right)\right)
```

- `b_k(t)` — "remaining battery energy of VU k at time slot t".
- `e_k(t)` — "the amount of energy harvested from the environment at the beginning of slot t".

**Objective — Eq. (12)** (§4) — **quoted verbatim, including an apparent internal inconsistency**:

```latex
\arg \min_{\{x_k(t), f_k(t), p_k(t)\}} \mathbb{E}\left[\sum_{t}^{\mathbb{T}} \sum_{k=1}^{\mathbb{K}} (\eta_{1} D_{k}^{total} + \eta_{2} T_{k}^{total})\right]
```

- `eta_1`, `eta_2` — "two tunable weighting parameters"; §5: `eta_1 = eta_2 = 0.5`, and one
  experiment with `eta_1 = 1, eta_2 = 5`.
- **Reviewer observation:** the paper's prose says the cost "is jointly determined by its
  execution delay and energy consumption" and Eq. (22) uses `E_k^total` in the energy-weighted
  position, but Eq. (12) as printed writes `T_k^total` there. This looks like a typographical
  error in Eq. (12) (delay appearing twice), but **I am reporting the printed formula, not
  correcting it.** Do not cite Eq. (12) as an authority for the energy term; cite Eq. (8)–(10).

**Constraints — Eq. (13)–(18)** (§4):

```text
(13)  x_k in {0,1}
(14)  0 <= p_k(t) <= p_k^max
(15)  D_k^total(t) <= D_max
(16)  b_k^min <= b_k(t) <= b_k^max
(17)  0 <= f_k(t) <= f_k^max
(18)  SUM_k x_k <= K
```

**Reward — Eq. (22)** (§5):

```latex
r(t) = -(\eta_1 D_k^{total} + \eta_2 E_k^{total}) - (\eta_1 D_k^{p} + \eta_2 E_k^{p})
```

- `D_k^p` — "delay penalty, which is triggered if the task exceeds its maximum tolerable
  completion time"; `E_k^p` — "energy penalty, incurred when energy consumption violates the
  minimum battery constraint".

**State / action — Eq. (19), (20), (21)** (§5):

```text
(19)  S(t) = { S_k(t) }, and
      S_k(t) = { S_k^task(t), S_k^gain(t), S_k^power(t), S_k^resource(t), S_k^battery(t) }
      with S_k^task(t) = [s_k(t), c_k(t), D_max], S_k^power(t) = p_k^max,
           S_k^resource(t) = f_k^max, S_k^battery(t) from Eq. (11)
(20)  A_c(t) = x_{c,k}(t) <- theta_k(S_k(t))
(21)  A_m(t) = ( x_{m,k}(t), p_{m,k}(t), f_{m,k}(t) ) <- phi(S, A, S_bar, A_bar)
```

`x_{m,k} ≥ 0.5` ⇒ offload, else local; `p_k = max(p^min, p_{m,k}(t)·p_k^max)`;
`f_k = max(f^min, f_{c,k}(t)·f_k^max)`.

**Not present in COMA2C:** MEC/server compute energy; RX energy; queueing or outage/retry energy;
DAG dependencies; any numeric value of `varepsilon`; any static/idle server power.

---

### B.4 S4 — Sensors 2025 (Almuseelem)

Sections: §3.3 Communication Model, §3.4 Computation Model (§3.4.1 Local Processing,
§3.4.2 Remote Processing), §3.5 Security, §3.6 Task Caching, §3.7 Problem Formulation.
Equation numbers are as printed.

**Task tuple — inline (§3.3):** `(beta_ij, sigma_ij, delta_ij)` where

- `beta_ij` — "input data size" (β; MB in §5),
- `sigma_ij` — "output data size",
- `delta_ij` — "computational demand in CPU cycles required for task j of vehicle i".

Explicit modeling decision, quoted: *"following the approach outlined in [44], this study
disregards the energy and time consumption associated with transmitting output data."*

**Upload rate — Eq. (2)** (§3.3):

```text
R_ik = B_ik * log2( 1 + p_i * g^2 / (omega * B_ik) )
```

- `B_ik` — "the uplink bandwidth allocated to the vehicle";
- `p_i` — "the vehicle's transmission power";
- `g` — "the channel gain"; `omega` — "the noise power at the edge server".
- **Transcription note:** the PDF text layer renders this as `Rik = Biklog2(1+ pig2ωBik)`. The
  semantics (Shannon form with `g²` and `omega·B_ik`) are unambiguous from the surrounding prose,
  but the exact grouping of the fraction could not be resolved from the PDF text layer alone.
  Treat the exact placement of `g²` as **UNVERIFIED**.

**Local computation energy — Eq. (3)** (§3.4.1) — key equation:

```text
E^L_ij = delta_ij * eta_i
```

- `delta_ij` — computational demand in **CPU cycles**.
- `eta_i` — "represents the energy consumed per CPU cycle".
- **This is a per-cycle-energy model, not a switched-capacitance model.** It is the degenerate
  case `E = C · (kappa·f²)` at fixed frequency, so it is unit-compatible with MARGO **provided
  `eta_i = kappa_i · f_i²`**.
- **`eta_i` is never assigned a numerical value anywhere in the paper that I could find** — the
  local-energy term is un-calibrated in the simulation setup (§5.1 lists vehicles, tasks, data
  sizes, cycles/byte, server frequencies, transmit power and bandwidth, but not `eta`).

**Local computation time — Eq. (4)** (§3.4.1): `T^L_ij = delta_ij / f^L_i`
- `f^L_i` — "the processing capacity of vehicle i". §5.1: vehicle compute ∈ [0.5, 1.0] GHz.

**Remote (transmit) energy — Eq. (5)** (§3.4.2) — key equation:

```text
E^R_ij = p_i * T^Tran_ij
```

- `p_i` — vehicle transmit power; §5.1: **100 mW** for all vehicles.
- `T^Tran_ij = beta_ij / R_ik` (Eq. 9).
- Verbatim framing: *"the execution of vehicle tasks on remote servers … it is possible to
  precisely estimate the energy usage and time needed by the servers"* — but Eq. (5) is the
  **vehicle's** transmit energy; **no server compute energy term exists in this paper.**

**Remote delays — Eq. (6), (7), (8), (10), (11), (12):**

```text
T^RSU_ij  = T^Tran_ij + T^RSU_ex_ij          (6)
T^UAV_ij  = T^Tran_ij + T^UAV_ex_ij          (7)
T^CLO_ij  = T^Tran_ij + varpi + T^CLO_ex_ij  (8)
T^RSU_ex_ij = delta_ij / f^RSU_i             (10)
T^UAV_ex_ij = delta_ij / f^UAV_i             (11)
T^CLO_ex_ij = delta_ij / f^CLO_i             (12)
```

- `varpi` — "the propagation delay between the edge server (RSU or UAV) and the cloud server".
- `f^RSU_i`, `f^UAV_i`, `f^CLO_i` — "processing capacities available to vehicle i at the RSU,
  UAV, and Cloud server".
- §5.1: cloud = 500 GHz, RSU = 100 GHz, UAV = 50 GHz; RSU bandwidth 20 MHz.

**Server resource constraints — Eq. (13), (14):** `SUM alpha_ijk f^RSU_i <= F^RSU_c` and
`SUM alpha_ijk f^UAV_i <= F^UAV_c`.

**Encryption energy — Eq. (15)** (§3.5): `e^ENC_ij = ENC_ij * eta_i`
- `ENC_ij`, `DEC_ij` — "the processing cycles required to execute encryption and decryption
  operations" (cycles; no numeric value given).
- Charged with the **same `eta_i`** as ordinary local compute — i.e. security overhead is an
  additive per-cycle cost on the vehicle.

**Total secure communication energy — Eq. (18)** (§3.5):

```text
E^SEC_ij = ( tau_i * ( e^ENC_ij + E^R_ij ) + (1 - tau_i) * E^R_ijk ) )
```

- `tau_ij ∈ {0,1}` — "binary decision variable representing security decisions"; `tau_ij = 1`
  means the security layer encrypted the task data. (Note the paper writes `tau_i` here and
  `tau_ij` in the definition; and there is an unmatched closing parenthesis in the printed
  equation.)

**Per-task total energy — Eq. (22)** (§3.6):

```text
E_ij = alpha_ij0 * E^L_ij + SUM_{k=1}^{U+1} alpha_ijk * (1 - upsilon_ij) * E^SEC_ij
```

- `alpha_ijk ∈ {0,1}` — execution-location decision (`k=0` local, `k=1..K` RSUs, `k=K+1..K+U` UAVs, `k=U+1` cloud).
- `upsilon_ij ∈ {0,1}` — caching decision ("task j from vehicle i has been cached at the edge server").

**Problem — Eq. (24)** (§3.7):

```text
min_alpha  SUM_i SUM_j E_ij
s.t.  E_ij - E^L_ij <= 0            C1
      T_ij - T^L_ij <= 0            C2
      SUM_{k=0}^{K+1} alpha_ijk = 1 C3
      ... cache/storage/resource C4-C7 ...
      alpha_ijk, tau_ij, upsilon_ij in {0,1}   C8-C10
```

- `C1` is an **energy constraint**: total energy may not exceed the all-local energy
  (`E^L_ij`). So energy is **both** objective and constraint in this paper.
- §5.1: 100 vehicles, 3 tasks each, input data ∈ [0,10] MB, **500 CPU cycles/byte**, cloud 500
  GHz, RSU 100 GHz, UAV 50 GHz, vehicle compute ∈ [0.5, 1.0] GHz, transmit power **100 mW**,
  RSU bandwidth 20 MHz.

**Not present in S4:** MEC/UAV/cloud **compute energy**; RX energy; `eta_i` numeric value;
outage/retry energy; DAG dependencies; V2V.

---

### B.5 S5 — LyDRL (Chen et al., ACM TAAS 2025) — **INACCESSIBLE**

- DOI 10.1145/3715333. Publisher: ACM Transactions on Autonomous and Adaptive Systems, 2025.
- `https://dl.acm.org/doi/full/10.1145/3715333` → **HTTP 403** ("Performing security verification",
  Cloudflare-style bot challenge), both directly with a browser user-agent and through a
  text-rendering proxy.
- `https://dl.acm.org/doi/pdf/10.1145/3715333` → **HTTP 403**, same challenge.
- Semantic Scholar Graph API: `isOpenAccess: false`, `openAccessPdf: {url: "", status: "CLOSED"}`.
- A second search pass found no author copy, institutional-repository copy, or preprint.
- **Verified from public metadata only:** title, five authors (Yanming Chen, Ziyang Huang,
  Yi-Wen Zhang, Wei-Wei Fang, Neal N. Xiong), venue (ACM TAAS), year (2025). Part of the public
  abstract was retrievable through the Semantic Scholar API and describes CAV task offloading to
  a VEC server, with metaverse-class applications motivating the problem; the abstract text
  visible to me does **not** state the energy equation.
- **No equation, equation number, symbol value, or numerical parameter from this paper is
  reported anywhere in this review.** Anything about its Lyapunov formulation would be
  unverifiable. **Do not cite it.**
- One search-engine result exposed a fragment of Lyapunov drift algebra while I was looking for the
  paper. That fragment was **not** verified against the original and is therefore **deliberately
  not transcribed here.**

---

### B.6 S6 — JTOCRA (Liu et al., arXiv:2508.15795, 13 Aug 2025, preprint)

Sections: §III-D "Local Computing" (Eq. 9, 10), §III-D2 "Task Offloading" (Eq. 11, 12),
§III-E total cost (Eq. 13, 14, 15), §IV problem/state/action (Eq. 16a, 17, 18, 19).

**Task model — inline (§III-D):** `Psi_v^t = (l_v^t, mu_v^t, tau_v^t)`
- `l_v^t` — "data size of the task (in bit)";
- `mu_v^t` — "computation intensity of the task (in cycles/bit)";
- `tau_v^t` — "the deadline of the task".

**Local delay — Eq. (9):** `D_{v,v}^t = l_v^t * mu_v^t / F_v`

**Local energy — Eq. (10)** — key equation:

```latex
E_{v,v}^{t} = \gamma_{v} (F_{v})^{2} l_{v}^{t} \mu_{v}^{t}
```

Verbatim: *"where `gamma_i >= 0` represents the effective capacitance of the CPU, which depends
on the CPU chip architecture of vehicle v."*

- `F_v` — vehicle computing resources; Table II: **F_v ∈ [1,5] GHz**.
- `gamma_v` — effective capacitance; Table II: **`gamma_v = 10^-28`**.
- **Same functional form as MARGO's CPU term**, with `C = l_v^t · mu_v^t` (bits × cycles/bit = cycles).

**Offload delay — Eq. (11):** `D_{m,v}^t = l_v^t / R_{v,m}^t + l_v^t * mu_v^t / f_{m,v}^t`

**Offload energy — Eq. (12)** (§III-D2):

```latex
E_{m,v}^{t} = p_{v}^{t} l_{v}^{t} / R_{v,m}^{t} + \epsilon\, l_{v}^{t} \mu_{v}^{t}
```

Verbatim: *"Note that we neglect the delay for result feedback, as the size of results for most
mobile applications …"*.

- First term: TX energy over the uplink ⇒ `p_v^t · (l/R) = power × active transmission time`.
  **No RX term.**
- `epsilon` — Table II: **"The energy consumed per CPU cycle by MEC servers — 8.2 × 10^-28 J"**.
  Second term is the **MEC-server compute energy**.
- `p_v^t` — Table II: **P_v ∈ [10, 25] dBm**.

**Total delay / energy — Eq. (13), (14):**

```latex
D_{v}^{t} = k_{v,v}^{t} l_{v}^{t} \mu_{v}^{t} / F_{v} + \sum_{m} k_{v,m}^{t} ( l_{v}^{t}/R_{v,m}^{t} + l_{v}^{t}\mu_{v}^{t}/f_{m,v}^{t} )
E_{v}^{t} = k_{v,v}^{t} \gamma_{v} (F_{v})^{2} l_{v}^{t} \mu_{v}^{t} + \sum_{s} o_{v,s}^{t} ( p_{v}^{t} l_{v}^{t}/R_{v,m}^{t} + \epsilon\, l_{v}^{t} \mu_{v}^{t} )
```

- `k_{v,v}^t`, `k_{v,m}^t` — binary local/offload selection variables.

**Cost and objective — Eq. (15), (16a):**

```text
C_v^t = w_v^D D_v^t + w_v^E E_v^t                                  (15)
P:  min_{K, F} (1/T) SUM_t SUM_v C_v^t   s.t. k_{v,z}^t in {0,1}   (16a)
```

- Energy and delay are combined into a weighted cost (energy is part of the objective), and the
  per-agent energy budgets `E_v`, `E_m` appear in the state (Eq. 17, 18) — i.e. **energy is also
  a long-horizon budget**.

**State / action — Eq. (17), (18), (19):**

```text
(17)  s^t = { q_v^t, F_v, E_v, l_v^t, mu_v^t, tau_v^t, q_m, F_m, E_m }
(18)  o_n^t = { q_n^t, l_n^t, mu_n^t, tau_n^t, F_n, q_m^t }
(19)  a_n^t = { k_n^t, f_n^t }
```

**Table II (parameters) — verbatim values relevant to a unit audit:**

| symbol | description | default value |
|---|---|---|
| `F_v` | computing resources of vehicle v | [1, 5] GHz |
| `E_v` | energy constraint of vehicle v | [5, 25] J |
| `l_v^t` | task size | [1, 3] Mb |
| `mu_v^t` | computation intensity of tasks | [500, 1500] cycles/bit |
| `tau_v^t` | deadline of task | [1, 5] s |
| `E_m` | energy constraint of MEC server m | 1000 J |
| `F_m` | computing resources of MEC server m | [50, 100] GHz |
| `B_{v,m}` | bandwidth between vehicle and MEC server | 20 MHz |
| `P_v` | transmit power of vehicle v | [10, 25] dBm |
| `N_0` | noise power | −98 dBm |
| `gamma_v` | effective capacitance of the CPU for vehicle v | **10^-28** |
| `epsilon` | energy consumed per CPU cycle by MEC servers | **8.2 × 10^-28 J** |

**Not present in S6:** DAG dependencies; RX energy; CPU static/idle; outages/retries; V2V/helper
tier; any unit statement for `gamma_v`.

---

### B.7 S7 — SP-LLM (Ahmadpanah, arXiv:2508.09149, preprint)

Section read: **§3.3.2 "Proactive Optimization Problem Formulation"**. Only the formulation
section was read in depth.

**Per-slot Lyapunov drift-plus-penalty problem — Eq. (1):**

```text
min_{w(t), a(t)}  E[ Delta_L(t) + C_sys(t) | Q(t) ]
```

- `Delta_L(t)` — "the Lyapunov drift";
- `C_sys(t)` — "a penalty term representing the system's operational cost";
- `Q(t)` — queue state.

**Drift-plus-penalty bound — unnumbered, immediately following Eq. (1)** (transcribed from the
HTML; some symbols were dropped by the renderer and are marked `[…]`):

```text
E[...] <= B - SUM_{k=1}^{K} q_k(t) ( E[Z_hat_k(t)] - E[phi_k(t)] )
          + E[ beta_E(t) * E_hat_sys(t) - beta_Q(t) * U_hat_sys(t) + [...] ]
```

- `beta_E(t)` — a time-varying weight multiplying the predicted system energy `E_hat_sys(t)`.
- **Reviewer note:** energy enters as a **penalty weight inside the drift-plus-penalty bound**. I
  searched the full rendered text for an explicit *virtual energy-deficit queue* (the standard
  mechanism for a long-term energy constraint) and **did not find one**; the mechanism here is a
  weighted penalty, not a dedicated energy queue. Because the arXiv HTML renderer dropped some
  math, this should be treated as **UNVERIFIED-to-negative**, not as a confirmed absence.

**Not usable as a base energy model:** the paper supplies no CPU/TX energy breakdown with which to
compute joules from scheduler events.

---

### B.8 S8 — Liu et al. (DCN 2023) — **INACCESSIBLE (verification failed)**

- This is the paper MARGO's `frozen_experiment.yaml` cites for `kappa_helper = 5e-27`,
  `f_helper ∈ [1,2] GHz`, `kappa_mec = 1e-27`, `f_mec = 10 GHz`.
- Bibliographic identity **verified** via Crossref: "Joint computation offloading and resource
  allocation in vehicular edge computing networks", Shuang Liu, Jie Tian, Chao Zhai, Tiantian Li,
  *Digital Communications and Networks* **9**(6):1399–1410, December 2023,
  DOI 10.1016/j.dcan.2022.12.002, license CC BY-NC-ND 4.0, PII S2352864822002620.
- Full text **NOT read**: ScienceDirect `/pdfft` returned HTTP 403; the publisher's own SciEngine
  page returns a JavaScript-only shell (2,998 bytes); a text-rendering proxy was blocked by the
  same anti-bot layer.
- **Consequence:** the numbers `gamma_j = 5e-27`, `F_j ∈ [1,2] GHz`, `gamma_m = 1e-27`,
  `F_m = 10 GHz` are **UNVERIFIED by this review**, and `gamma`'s unit convention in that paper is
  therefore also unknown. MARGO's config already flags that the *vehicle* tier is not from this
  paper ("NOT imported from Liu2023, which is fully offloaded and has no UE-local tier") — that
  caveat is consistent with what I could see, but I could not confirm the MEC/helper numbers.
- **Recommendation:** mark these two rows of MARGO's provenance table as "cited, pending
  verification", or re-source them from an accessible paper.

---

## C. Unit audit

### C.1 The dimensional question that decides transplantability

MARGO's CPU term is `E = kappa · C · f²` with **`f` in Hz** and `C` in cycles. Therefore

```text
kappa has SI unit  J / (cycle · Hz^2)  ==  s^2 / cycle          [DERIVED BY REVIEWER]
per-cycle energy of the tier = kappa * f^2         [J/cycle]
dynamic power of the tier     = kappa * f^3         [W]
```

A paper that writes `E = γ · F² · C` is **directly transplantable only if its `F` is also in Hz.**
If the paper's `F` is numerically in **GHz**, the conversion is

```text
kappa_SI = gamma_paper * (1e-9)^2 = gamma_paper * 1e-18        [DERIVED BY REVIEWER]
```

Most VEC papers state the frequency in GHz in their parameter tables but do not state the unit
inside the energy equation. This is the single largest transplant hazard found in this review.

### C.2 Parameter unit audit table

| Parameter | Value as printed | Unit as printed in source | Source (exact location) | Compatible with MARGO formula set? | Notes |
|---|---|---|---|---|---|
| `p_local` (vehicle CPU) | 1 | W | S1 ST-HO, Table 1 (§4) | **Yes, after conversion.** MARGO needs `kappa`, not `p`. | **DERIVED BY REVIEWER:** `kappa_ue = p_local / f_local³ = 1 / (1e9)³ = 1.0e-27` with `f` in Hz. This is **numerically identical to MARGO's frozen `kappa_ue = 1e-27` at `f_ue = 1 GHz`.** Independent corroboration of MARGO's UE tier. |
| `f_local` | 1 | GHz | S1, Table 1 | Yes | Matches MARGO `f_ue = 1.0e9 Hz` exactly. |
| `f_mec` | 5 | GHz | S1, Table 1 | Yes (but ST-HO never prices MEC compute) | MARGO uses 10 GHz; 2× higher. |
| `p_up` (vehicle TX) | 0.5 | W | S1, Table 1 | **Yes** — directly usable as a TX power | MARGO `ue_tx_w = 1.0 W`. ST-HO is **2× lower** (0.5 W = 27 dBm). |
| `p_down` (download / receive) | 0.2 | W | S1, Table 1 + Eq. (9) | **Yes, if MARGO enables RX** | This is the **only literature-pinned receive/download power found in this review.** It contradicts the note in MARGO's config that "no literature-pinned p_rx exists" — see flag U-5. |
| `p_basic` (idle/static on-board) | 0.1 | W | S1, Table 1 + Eq. (6) | **Not in MARGO's primary model** | MARGO has `server_static_power_w` (sensitivity only) but **no UE static/idle power**. ST-HO charges this while the vehicle is offloaded. |
| `B` | 10 | MHz | S1, Table 1 | Yes (radio model) | MARGO's radio bandwidth is config-driven. |
| `kappa` | `10^-11` | **not stated** | S2 Parastar, §IV-A, Eq. (7) | **Compatible in form; INCOMPATIBLE in magnitude unless the GHz convention is applied** | **Flag U-2.** With `cpu_c` in GHz (as §VI states), `kappa_SI = 1e-11 × 1e-18 = 1.0e-29` **[DERIVED BY REVIEWER]** — i.e. **100× smaller than MARGO's `kappa_ue = 1e-27`.** Transplanting the literal `1e-11` into a formula with `f` in Hz would understate CPU energy by a factor of **10^18**. |
| `cpu_c` | [1, 2] | GHz | S2, §VI | Yes | Compatible with MARGO's UE frequency band. |
| `CPU_m` | [40, 60] | GHz | S2, §VI | Yes | MARGO `f_mec = 10 GHz` is below this band. |
| `R_m` (MEC transmission rate) | {10, 25, 100} | Gb/s | S2, §VI | Yes (rate units are unambiguous) | 10–100 Gb/s ⇒ 1.25–12.5 GB/s. |
| `E_c` (initial battery) | [50, 150] | **mJ** | S2, §VI | Yes, but note the scale | 50–150 mJ is a very small energy budget; consistent with the small `data_k` and `cl_k` in that paper. |
| `data_k` | [200, 1000] | **Kbytes** | S2, §VI | Yes | `cl_k` is in cycles/**bit** while `data_k` is in **bytes** — the paper does not state the ×8 conversion. **Flag U-3.** |
| `cl_k` | 25 | cycles/bit | S2, §VI | Yes | = 200 cycles/byte. **12× lower than MARGO's 300 cycles/bit.** |
| `p_c^tx` | 0.5 | **W** | S2, §VI | Yes | = 27 dBm. MARGO `ue_tx_w = 1.0 W` (30 dBm). |
| `p_b^tx` | 1 | **W** | S2, §VI | Yes — a **MEC-side** TX power | = 30 dBm. MARGO `mec_tx_w = 3.162 W` (35 dBm) is **3.2× higher**. |
| `varepsilon` | no numeric value given | **not stated** | S3 COMA2C, §3.4 Eq. (8) | **Form-compatible, value-incompatible** | **Flag U-4.** The paper never prints a value for `varepsilon`, and never states whether `f_k(t)` is Hz or GHz. **Do not transplant.** Form is identical to MARGO. |
| `B^e` | 20 | MHz | S3, §5 | Yes | |
| `N_0` | −174 | dBm/Hz | S3, §5 | Yes | Standard thermal noise PSD. |
| `delta_ij` | (no value given) | CPU cycles | S4 Sensors, §3.3 | Yes (task property) | |
| `eta_i` (vehicle, J/cycle) | **no numeric value given** | not stated | S4, §3.4.1 Eq. (3) and §3.5 Eq. (15) | **Value-incompatible (uncalibrated)** | §5.1 lists everything except `eta`. Local energy cannot be reproduced from this paper. |
| `p_i` (vehicle TX) | 100 | **mW** | S4, §5.1 | Yes | = 20 dBm = 0.1 W. **10× lower than MARGO's `ue_tx_w = 1.0 W`.** |
| compute intensity | 500 | **CPU cycles/byte** | S4, §5.1 | Yes, after conversion | **DERIVED BY REVIEWER:** 500 cycles/byte = **62.5 cycles/bit** (÷8). **4.8× lower than MARGO's 300 cycles/bit.** |
| vehicle compute | [0.5, 1.0] | GHz | S4, §5.1 | Yes | |
| `f^CLO` / `f^RSU` / `f^UAV` | 500 / 100 / 50 | GHz | S4, §5.1 | Yes | Not a MARGO tier. |
| `gamma_v` | `10^-28` | **not stated** | S6 arXiv:2508.15795, Table II; used in Eq. (10) | **Ambiguous — see flag U-1** | With `F_v` in **Hz**, `kappa_SI = 1.0e-28` (10× below MARGO `kappa_ue = 1e-27`), giving `gamma·F² = 1e-10 J/cycle` at 1 GHz and ≈`5 × 10^-2 J` for a 1 Mb / 500 cycles-per-bit task — physically plausible. With `F_v` in **GHz** (as Table II labels it), Eq. (10) yields ≈`5 × 10^-20 J` for the same task — implausible. |
| `F_v` | [1, 5] | GHz | S6, Table II | Yes | |
| `mu_v^t` | [500, 1500] | cycles/bit | S6, Table II | Yes | 1.7–5× MARGO's 300 cycles/bit. |
| `l_v^t` | [1, 3] | **Mb** (megabit) | S6, Table II | Yes | Units are explicit — unusually good practice. |
| `P_v` | [10, 25] | **dBm** | S6, Table II | Yes | = 0.01–0.32 W. **MARGO's 1.0 W (30 dBm) is above this range.** |
| `N_0` | −98 | dBm | S6, Table II | Yes | Different noise convention from S3/S4 (total noise power vs PSD) — do not mix. |
| `E_v` / `E_m` | [5, 25] J / 1000 J | **J** | S6, Table II | Yes | Per-vehicle and per-server energy budgets. |
| `epsilon` (MEC, J/cycle) | `8.2 × 10^-28` | **J** (per CPU cycle, per the description) | S6, Table II | **Arithmetically compatible as J/cycle; physically implausible as printed — flag U-6** | MARGO's per-cycle MEC energy is `kappa_mec·f_mec² = 1e-27 × 1e20 = 1e-7 J/cycle` **(100 nJ/cycle)**. The paper's `8.2e-28 J/cycle` is **~20 orders of magnitude smaller**. A real CPU is ~1e-9 J/cycle. **Do not transplant.** |
| `kappa_ue` | `1.0e-27` | SI (f in Hz) | **MARGO** frozen YAML | — | Implied power at 1 GHz = **1 W**; implied per-cycle energy = **1e-9 J/cycle**. |
| `kappa_helper` | `5.0e-27` | SI | **MARGO** frozen YAML, cited to S8 | **UNVERIFIED source (S8 inaccessible)** | Implied power at 1.5 GHz = 5e-27 × 3.375e27 = **16.9 W**. Per-cycle = 1.125e-8 J/cycle. |
| `kappa_mec` | `1.0e-27` | SI | **MARGO** frozen YAML, cited to S8 | **UNVERIFIED source (S8 inaccessible)** | Implied power at 10 GHz = 1e-27 × 1e30 = **1000 W (1 kW)**. Per-cycle = 1e-7 J/cycle. The YAML itself flags this. |
| `cycles_per_bit` | 300 | cycles/bit | **MARGO** frozen YAML, cited to Zhao et al. (2018) — not read in this review | In range, but **unsourced within 2024–2026** | Literature range observed in the accessible 2024–2026 sources: **25** (S2) / **62.5** (S4) / **500–1500** (S6) cycles/bit. 300 sits inside the span but is 4.8×–12× the two low values. |
| `ue_tx_w` | 1.0 | W | MARGO frozen YAML, cited to S8 | **UNVERIFIED source** | 30 dBm. Accessible sources: 20 dBm (S4), 27 dBm (S2), 10–25 dBm (S6). |
| `mec_tx_w` | 3.162 | W | MARGO frozen YAML, cited to Zhao et al. 2018 | UNVERIFIED by this review | 35 dBm. Accessible source: 30 dBm (S2 `p_b^tx`). |
| `helper_tx_w` | 1.0 | W | MARGO frozen YAML — **explicitly an assumption** | Assumption | Text: "assumed equal to the UE class; no helper TX published". Confirmed: **no source read here publishes a helper-V2V transmit power** (see §E). |
| `ue_rx_w`, `helper_rx_w` | null | W | MARGO frozen YAML — "NOT literature-pinned" | **Contradicted by S1** | S1 Table 1 gives `p_down = 0.2 W` (download power charged to the onboard device). See flag U-5. |

### C.3 Unit-incompatibility flags (papers whose quoted numbers must NOT be transplanted as-is)

**U-1 — S6 arXiv:2508.15795 `gamma_v = 10^-28` (internal GHz/Hz ambiguity).**
Table II labels `F_v` in **GHz**, but Eq. (10) uses `F_v` as a squared factor with
`gamma_v = 1e-28`. Interpreted with `F_v` in Hz, the pair is physically sensible
(`1e-28 × (1e9)² = 1e-10 J/cycle`, ~0.1 nJ/cycle at 1 GHz, and ~0.05 J for a 1 Mb task at 500
cycles/bit). Interpreted with `F_v` in GHz, the same numbers give ~`1e-19 J` per task — about
19 orders of magnitude below any real CPU. **The printed value is therefore only usable if the
frequency is re-inserted in Hz.** FLAG: unit-ambiguous.

**U-2 — S2 Parastar `kappa = 10^-11`.**
`kappa`'s unit is never stated, and `cpu_c` appears in the same expression with its GHz label in
§VI. If `cpu_c` is in GHz, `kappa_SI = 1e-11 × 1e-18 = 1e-29`. Transplanting `1e-11` literally
into `E = kappa·C·f²` with `f` in Hz would understate energy by **10^18**. FLAG: must convert.

**U-3 — S2 Parastar bytes-vs-bits.**
`data_k` is given in **Kbytes** but `cl_k` in **cycles/bit**, and the equations combine them
directly as `ci_k = data_k · cl_k`. The ×8 conversion is never stated. FLAG: unit convention
inferred, not stated.

**U-4 — S3 COMA2C `varepsilon`.**
No numeric value anywhere in the paper; `f_k(t)`'s unit (Hz vs GHz) is never stated and the symbol
is overloaded between a delay denominator (cycles/s position) and an energy factor. FLAG:
value-unusable and unit-ambiguous.

**U-5 — MARGO's own config claim "no literature-pinned p_rx exists" is too strong.**
S1 (ST-HO) Eq. (9) + Table 1 give `p_down = 0.2 W` and Eq. (10) charges it to the onboard device.
That is a usable (if loosely attributed) receive/download power anchor at the vehicle side. FLAG:
MARGO's justification for `include_rx_energy: false` should be restated as "RX is excluded because
it is not in the primary accounting scope", not "no literature value exists", **or** the sensitivity
sweep should include `ue_rx_w = 0.2 W` from S1. Full-text review of S1 did not resolve whether
`p_down` is the vehicle's receive power or the server's transmit power; the paper attributes the
resulting energy to the onboard device. Treat as **candidate anchor, medium confidence**.

**U-6 — S6 `epsilon = 8.2 × 10^-28 J` per MEC CPU cycle is physically implausible as printed.**
MARGO's own MEC per-cycle energy is `1e-7 J/cycle` (from `kappa_mec·f_mec²`); a realistic CPU is
~`1e-9 J/cycle`; the paper's figure is ~`8e-28 J/cycle`. Three different conventions cannot all be
right. FLAG: do not use S6's `epsilon` to calibrate MARGO's MEC tier; note instead that S6 is the
only accessible 2025 source that includes a **MEC-side per-cycle energy term at all**, which is
the structural point MARGO needs.

**U-7 — S1 ST-HO's CPU power is frequency-independent.**
`e_local = p_local · C/f_local`. This equals MARGO's `kappa·C·f²` only if `p_local = kappa·f³`.
ST-HO's parameters happen to satisfy this exactly at `f = 1 GHz` (see C.2), which is why the
numbers agree — but the *models* differ: ST-HO cannot represent a frequency change, MARGO can.
Also note ST-HO's `C_x` is printed as "**M cycles**" while the local time equation divides by
`f_local` in GHz; the µs/ns bookkeeping is easy to get wrong. FLAG: model form differs.

**U-8 — S4 Sensors uses a per-cycle energy model with no value.**
`E = delta · eta` is only convertible to MARGO's form via `eta_i = kappa_i · f_i²`. Without a
numeric `eta_i` the model is un-calibrated: you cannot reproduce the paper's local energy at all.
FLAG: not transplantable, but structurally compatible.

---

## D. Dated search log

All activity on **2026-10-03**. Access method in parentheses. "Opened" means I retrieved and read
the item; "searched" means the result list only.

### D.1 Queries run

| # | Query (as issued) | Engine | Outcome |
|---|---|---|---|
| Q1 | `ST-HO Symmetry-Enhanced Energy-Efficient DAG Task Offloading Algorithm Intelligent Transport System Symmetry 2024 energy model equation` | web_search | Located the DOAJ/Semantic Scholar records and the MDPI article; no full text in results |
| Q2 | `Parastar Energy-Efficient Task Computation at the Edge for Vehicular Services arXiv 2511.18449` | web_search | Located arXiv abs page + a Zenodo PDF (a *different* paper — not used) |
| Q3 | `Sun cooperative multi-agent optimization task offloading vehicular edge computing Discover Computing 2025 energy` | web_search | Located the Springer article page (paywall-redirect) |
| Q4 | `"ST-HO" Symmetry 2024 Gao Luo DAG task offloading energy consumption model kappa f^3 equation pdf` | web_search | Only index/metadata pages; no PDF mirror |
| Q5 | `LyDRL Lyapunov-guided Deep Reinforcement Learning Stable Task Offloading Connected Autonomous Vehicles ACM 2025 energy model` | web_search | Located `dl.acm.org` article and PDF endpoints and a Semantic Scholar record |
| Q6 | `Sensors 2025 25 2039 "Deep Reinforcement Learning-Enabled Computation Offloading" energy optimization security vehicular edge-cloud PMC full text` | web_search | Located PubMed 40218550, Semantic Scholar, and the `mdpi-res.com` direct PDF attachment URL |
| Q7 | `"Lyapunov" "task offloading" connected autonomous vehicles 2025 energy virtual queue constraint primary source` | web_search | No accessible LyDRL copy; surfaced unrelated VEC papers |
| Q8 | `energy-aware DAG task offloading vehicular edge computing 2025 per-task CPU energy consumption model switched capacitance kappa f^2 primary research article` | web_search | Surfaced candidate S6 (arXiv:2508.15795) and others |
| Q9 | `Lyapunov drift-plus-penalty virtual energy queue long-term energy constraint task offloading vehicular edge computing 2024 2025 primary paper` | web_search | Surfaced candidate S7 (arXiv:2508.09149) |
| Q10 | `V2V relay transmission energy consumption model vehicular edge computing 2024 2025 relay vehicle energy cost equation` | web_search | **No primary source with an explicit V2V/helper relay energy equation was found** (see §E) |
| Q11 | `"LyDRL" Lyapunov task offloading connected autonomous vehicles pdf 2025` | web_search | Only `dl.acm.org` (403) and NSTL metadata |
| Q12 | `TAAS 3715333 Lyapunov-guided deep reinforcement learning stable task offloading energy consumption model vehicle` | web_search | Same; no accessible copy |
| Q13 | `arxiv 2025 vehicular edge computing Lyapunov virtual energy queue drift plus penalty long-term energy constraint task offloading DAG` | web_search | Surfaced arXiv:2508.09149 (S7) and unrelated items |
| Q14 | `"Joint computation offloading and resource allocation in vehicular edge computing networks" Liu Tian Zhai Li pdf table 1 gamma 5e-27` | web_search | Located SciEngine/CNKI/Wanfang mirrors; none machine-readable |
| Q15 | Crossref REST API: journal ISSN 2352-8648, 2023, volume 9, page 1399 | api | Resolved S8's DOI 10.1016/j.dcan.2022.12.002 and authors |
| Q16 | Semantic Scholar Graph API for DOIs 10.1007/s10791-025-09887-6, 10.1145/3715333, 10.3390/sym16020164, 10.3390/s25072039, 10.1016/j.dcan.2022.12.002 | api | Open-access status, PDF links, PMC id, abstracts |

### D.2 Documents actually opened and read

| Item | URL | Method | Result | Depth |
|---|---|---|---|---|
| S1 ST-HO full text | `https://mdpi-res.com/d_attachment/symmetry/symmetry-16-00164/article_deploy/symmetry-16-00164.pdf` | curl (PDF) + pypdf text/layout extraction | 16 pages, 75 KB of text | **Full text read** (§2.1–§2.4, §3, §4 incl. Table 1) |
| S2 Parastar | `https://arxiv.org/abs/2511.18449`, `https://arxiv.org/pdf/2511.18449v1`, `https://arxiv.org/html/2511.18449v1` | curl + web_fetch + LaTeX extraction from HTML | 10 pages; HTML retains `alttext` LaTeX for every numbered equation | **Full text read** (§III–§VII; all equations (1)–(12) with printed numbers) |
| S3 COMA2C | `https://link.springer.com/article/10.1007/s10791-025-09887-6` | direct curl → Cloudflare challenge; **`https://r.jina.ai/…` with `x-respond-with: html`** → publisher HTML with `mathjax-tex` LaTeX spans | 478 KB HTML; all 23 equations recovered with printed numbers | **Full text read** (§1–§6; §3.3–§3.5 and §4 in detail) |
| S4 Sensors 2025 | `https://mdpi-res.com/d_attachment/sensors/sensors-25-02039/article_deploy/sensors-25-02039-v2.pdf` | curl (PDF) + pypdf | 29 pages, 87 KB of text | **Full text read** (§3.3–§3.7, §5.1; Eq. (2)–(24)) |
| S5 LyDRL | `https://dl.acm.org/doi/full/10.1145/3715333` and `/doi/pdf/10.1145/3715333` | curl (browser UA), curl via `r.jina.ai` | **HTTP 403 anti-bot challenge** at both endpoints, both methods | **INACCESSIBLE** — metadata only via Semantic Scholar API |
| S6 arXiv:2508.15795 | `https://arxiv.org/abs/2508.15795`, `https://arxiv.org/html/2508.15795v1` | curl + LaTeX extraction | 31 numbered equations recovered; Table II parameters read | **Full text read** for §III-D, §III-E, Table II |
| S7 arXiv:2508.09149 | `https://arxiv.org/html/2508.09149v1` | curl + text render | Eq. (1) and the drift-plus-penalty bound recovered | **Partial read** (§3.3.2 formulation) |
| S8 Liu DCN 2023 | `https://www.sciencedirect.com/science/article/pii/S2352864822002620/pdfft`, `https://www.sciengine.com/DCAN/doi/10.1016/j.dcan.2022.12.002` | curl (browser UA), `r.jina.ai` (incl. `x-engine: browser`) | ScienceDirect **403**; SciEngine returns a 3 KB JavaScript shell | **INACCESSIBLE** — bibliographic metadata only via Crossref |
| MDPI article pages (`/2073-8994/16/2/164`, `/1424-8220/25/7/2039`) | — | curl / web_fetch | **HTTP 403** (Akamai edge deny) | Rendered irrelevant by the `mdpi-res.com` attachments above |
| `pubmed.ncbi.nlm.nih.gov/40218550/` | — | curl | Page loaded but no PMC id in the raw HTML | Superseded by the Semantic Scholar `PubMedCentral: 11991124` field |

### D.3 Search that produced **no** usable primary source

- **Q10 (V2V relay energy)** returned no primary paper with an explicit V2V/helper relay energy
  equation that I could open. The results were either paywalled (`dl.acm.org`,
  `sciencedirect.com`), surveys, or about UAV/platooning rather than one-vehicle-as-helper. **This
  is a genuine gap, not a search failure:** see §E for the consequence.
- **Q9/Q13 (Lyapunov long-term energy)** produced one accessible example (S7) but it uses an
  energy *penalty weight*, not a dedicated virtual energy queue, and its math was partly dropped
  by the HTML renderer. The canonical mechanism (virtual energy-deficit queue) is exercised in
  S5 (LyDRL), which is **inaccessible**.

---

## E. Recommendation

### E.1 Recommended single base model

**Adopt `physical_v1` exactly as already implemented, with the CPU term and TX term as follows,
and with no DVFS action space:**

```text
C_executed(i, x) = executed_work_i * 8 * cycles_per_bit          # cycles, after retries/restarts
T_cpu(i, x)      = C_executed(i, x) / f_x                        # s
E_cpu(i, x)      = kappa_x * C_executed(i, x) * f_x^2            # J
                 = P_cpu(x) * T_cpu(i, x),  P_cpu(x) = kappa_x * f_x^3
E_tx(hop, src)   = P_tx(src) * t_active_service(hop)             # J, active transmission only
E_rx             = excluded from the primary model (optional accounting)
```

Why this exact form, on the evidence:

1. **Three independent accessible primary sources use it verbatim.**
   - S3 COMA2C Eq. (8): `E^local = varepsilon · f² · c · s` — the same product of a hardware
     coefficient, `f²`, and a cycle count.
   - S2 Parastar Eq. (7): `e^{c,loc} = ci_k · kappa · [f_k(t)·cpu_c(t)]²`, with `kappa` explicitly
     named "effective switched capacitance".
   - S6 Eq. (10): `E = gamma_v (F_v)² l_v μ_v`, with `gamma_v` explicitly named "effective
     capacitance of the CPU".
   None of the three is a survey; all three are 2024–2025 primary sources with a vehicular
   setting. Three independent uses of the same algebraic form is a strong base-model warrant.
2. **The TX term is equally well supported and is TX-only in every accessible source.**
   S3 Eq. (9), S2 inline `e^air = l^air · p_c^tx`, S4 Eq. (5), S6 Eq. (12) — all are
   `power × transmission duration`, and every one of them neglects the receive side. MARGO's
   `include_rx_energy: false` default is therefore the mainstream choice, not a gap.
3. **S1 ST-HO is the natural DAG companion, and it independently corroborates MARGO's UE
   coefficient.** ST-HO's Eq. (16) is a DAG cost with per-location dependency-crossing terms, and
   its Table 1 gives `p_local = 1 W` at `f_local = 1 GHz`, which converts to
   `kappa = 1e-27` with `f` in Hz — **numerically identical to MARGO's frozen `kappa_ue`.**
   That is the single most useful calibration agreement found in this review.
4. **The model is computable from scheduler events by construction.** `C_executed` is
   post-retry/post-restart cycle count; `t_active_service` is the radio service interval. Both are
   already produced by MARGO's scheduler. No paper read here does this — all of them use
   *nominal* task sizes, so MARGO must state that its energy is charged on **executed** work, not
   requested work.

**Adopt `energy_scope = system` for logging but keep the paper's primary objective explicit.**
S2 is the only accessible source that prices car energy *and* MEC compute with the same
coefficient family, and S6 does the same. That supports MARGO's `system` scope as the *loggable*
scope. However ADR-001/PHASE5 lock the *claimed* scope at `mobile`. The frozen YAML and the ADR
currently disagree; that is a repository decision, not a literature one, and is flagged in §0.

### E.2 Components that are OUT of scope

| Component | Why out of scope | Evidence |
|---|---|---|
| DVFS / frequency scaling action space | Not needed: frequencies are fixed and configuration-driven. All accessible sources that use `kappa·f²` treat `f` as an allocation variable, but MARGO's fixed-`f` case is a strict restriction, not a contradiction | S2 §IV-A (`f_k(t)∈(0,1]`), S3 Eq. (17), S6 Eq. (19) |
| MEC datacenter **static/idle** power (`P_idle`) | No accessible source models it; S1's `p_basic` is a *vehicle* idle term, not a server one; the only server idle discussion in the repo is the YAML's own sensitivity hook | S1 Eq. (6) is vehicle-side; no server-idle equation found in S1–S7 |
| Backhaul / cloud / fronthaul energy | Only S4 has a cloud tier and it charges only delay (`varpi`), not energy; S2 charges a regressed transport *latency* but not its energy | S4 Eq. (8); S2 §IV-B |
| UAV / cloud execution tiers as energy payers | Out of MARGO's three-location model | S4, S2 |
| Security / encryption energy as a per-cycle cost | S4's `e^ENC = ENC · eta_i` is the only such term found; it is un-calibrated (`eta_i` has no value) and out of MARGO's scope | S4 §3.5, Eq. (15) |
| Task caching energy | S4 §3.6 only changes the decision set, not the energy equation | S4 Eq. (22) |
| Energy harvesting / battery replenishment | Out of scope for MARGO (no harvester); S3 Eq. (11) includes `e_k(t)` | S3 Eq. (11) |
| UE/helper **RX** energy as a *primary* component | Every accessible primary source neglects RX. Keep it as optional accounting only | S2 §IV-B (explicit neglect), S3 §3.4, S4 §3.3 (explicit neglect), S6 §III-D2 (explicit neglect) |
| **V2V/helper relay radio energy as a *published* component** | **No primary source with a helper-V2V relay energy equation was found.** The HELPER tier's radio energy in MARGO is therefore *not* literature-backed and must be declared as an assumption | Search Q10 found no openable primary source; MARGO's own config says `helper_tx_w` is "assumed equal to the UE class; no helper TX published" |

### E.3 Parameters that lack a literature source and MUST be declared as transparent assumptions

Each row below is a value that MARGO uses (or needs) and for which **this review found no primary
source that states the value and its unit**. They must appear in the paper's assumption table with
the stated unit and the stated reasoning.

| # | Parameter | MARGO value / status | Why it must be declared an assumption |
|---|---|---|---|
| A-1 | `helper_tx_w` (helper V2V transmit power) | 1.0 W | Repo already says "assumed equal to the UE class; no helper TX published". **Confirmed by this review**: no accessible 2024–2026 primary source publishes a helper/relay V2V transmit power at all (Q10) |
| A-2 | `helper_rx_w` (UE receive power on the helper→UE hop) | null / disabled | No source publishes a helper-hop receive power. S1's `p_down = 0.2 W` is a *download* power on the vehicle↔RSU path, not a V2V-receive power (flag U-5) |
| A-3 | `ue_rx_w` (UE receive power on the MEC downlink) | null / disabled | MARGO's config asserts no literature value exists. **This review partially contradicts that**: S1 Table 1 gives `p_down = 0.2 W`, attributed to the onboard device in S1 Eq. (10). If MARGO keeps RX disabled, the justification must be *scope*, not *absence* |
| A-4 | Energy charged during **outage / backoff / retry-dead time** | not modelled anywhere in this review | No accessible source distinguishes active service time from outage time. Every source computes transmission energy from `bytes/rate`, i.e. a nominal active interval. MARGO's recommendation (charge **active service only**; outage time free) is a modelling assumption with **zero** literature precedent found |
| A-5 | Energy charged for **re-executed** work after a retry/restart | not modelled anywhere in this review | All accessible sources charge nominal task cycles once. MARGO's event-derived `C_executed` (which grows on restart) is a novel accounting choice and must be declared |
| A-6 | `f_helper = 1.5 GHz` | 1.5e9 Hz | Repo source is "F_j in [1,2] GHz"; 1.5 GHz is the **midpoint of a range**, i.e. a selection, not a published value. Also the cited paper (S8) is **unverified** here |
| A-7 | `kappa_ue = 1.0e-27` | 1.0e-27 s²/cycle | Repo says "chosen inside the verified band". S1 independently corroborates the value at 1 GHz (`p_local = 1 W`), but no single paper states `1e-27` in MARGO's unit convention |
| A-8 | `kappa_mec = 1.0e-27` at `f_mec = 10 GHz` | 1.0e-27 s²/cycle | Cited to S8, which is **INACCESSIBLE**. The implied dynamic power is **1 kW**, which the repo itself flags. Needs re-sourcing or an explicit assumption + sweep |
| A-9 | `kappa_helper = 5.0e-27` at `f_helper = 1.5 GHz` | 5.0e-27 s²/cycle | Cited to S8, **INACCESSIBLE**. Implied dynamic power **16.9 W** at 1.5 GHz — an implausibly high per-cycle cost for a vehicle-tier device |
| A-10 | `cycles_per_bit = 300` | 300 cycles/bit | Cited to Zhao et al. 2018 (not read here, and outside the 2024–2026 window). Observed 2024–2026 primary range is **25–1500 cycles/bit** (S2 = 25, S4 = 62.5, S6 = 500–1500), so 300 is inside the span but unsupported by any source read here |
| A-11 | `ue_tx_w = 1.0 W` (30 dBm) | 1.0 W | Cited to S8, **INACCESSIBLE**. Observed 2024–2026 primary range: 10–27 dBm (S6 = 10–25 dBm, S4 = 20 dBm, S2 = 27 dBm). MARGO is at or above the top of that range |
| A-12 | `mec_tx_w = 3.162 W` (35 dBm) | 3.162 W | Cited to Zhao et al. 2018 (not read here). Only accessible comparator is S2's `p_b^tx = 1 W` (30 dBm) |
| A-13 | MEC / helper **static or idle** power | `server_static_power_w = null` | No accessible source gives a server-idle power. S1's `p_basic = 0.1 W` is the only idle term found, and it is **vehicle-side** |

### E.4 Sensitivity sweeps required

Ordered by expected impact on the reported comparisons:

1. **`cycles_per_bit` ∈ {25, 62.5, 300, 500, 1000, 1500}** (cycles/bit). Anchors: S2 = 25,
   S4 = 62.5 (from 500 cycles/byte), S6 = 500–1500, MARGO = 300. This is the largest single
   multiplier on `E_cpu` and a 60× span exists inside the 2024–2026 primary literature alone.
2. **`kappa_ue` ∈ {1e-29, 1e-28, 1e-27, 1e-26}** (s²/cycle). Anchors: S2-implied 1e-29,
   S6 1e-28 (Hz reading), S1-implied 1e-27 = MARGO, repo's stated band up to 1e-26. Directly tests
   whether `total_mobile_joules` rankings are driven by the CPU coefficient.
3. **`kappa_mec` and `f_mec` jointly**, so that the *implied* MEC dynamic power spans a defensible
   range. The current pair implies **1 kW**. Sweep `kappa_mec ∈ {1e-28, 1e-27, 1e-26}` ×
   `f_mec ∈ {5e9, 1e10, 2e10}` and report the implied `P = kappa·f³` for each cell. (Note S2 prices
   MEC compute with the *same* `kappa` as the vehicle; S1 uses `f_mec = 5 GHz` — MARGO's 10 GHz is
   2× S1 and far below S2's `CPU_m ∈ [40,60] GHz`.)
4. **`f_helper` ∈ {1.0e9, 1.5e9, 2.0e9} Hz** × **`kappa_helper` ∈ {1e-27, 5e-27}**, reporting
   implied power (currently 16.9 W at the frozen pair). Anchors: S8's stated band `F_j ∈ [1,2] GHz`
   (unverified) and S1's `f_local = 1 GHz`.
5. **`ue_tx_w` ∈ {0.1, 0.5, 1.0, 3.162} W.** Anchors: S4 = 0.1 W, S2 = 0.5 W, MARGO = 1.0 W,
   MARGO MEC = 3.162 W. Tests whether offloading decisions are TX-dominated.
6. **`helper_tx_w` ∈ {0.1, 0.5, 1.0} W.** Assumption A-1. Because there is *no* published value,
   this sweep is the only defensible way to report a HELPER-inclusive energy metric.
7. **`include_rx_energy` with `ue_rx_w` ∈ {0.0, 0.2, 0.5} W.** Anchor for 0.2 W: S1 Table 1
   `p_down`. Tests whether excluding RX materially changes rankings (the mainstream literature says
   it does not; this is the check).
8. **`server_static_power_w` ∈ {0, 10, 50, 100} W** with `server_utilization_model` on/off. Pure
   assumption (A-13). Needed only for the *system*-scope number, and it is the item most likely to
   swamp the mobile-scope signal if reported carelessly next to `total_mobile_joules`.
9. **Outage/retry energy accounting**: (a) charge active service only [recommended default];
   (b) charge active + outage time; (c) charge active + re-executed CPU cycles on restart. Report
   all three. This is assumption A-4/A-5 and is the MARGO-specific behaviour that no source read
   here covers.

### E.5 One-line summary of the recommendation

Adopt MARGO's existing `physical_v1` model unchanged as the base
(`E_cpu = kappa_x · C_executed · f_x²`, `E_tx = P_tx · t_active_service`, RX optional and off by
default); it is the same model family as COMA2C Eq. (8) (2025, peer-reviewed), Parastar Eq. (7)
(2025, peer-reviewed conference) and arXiv:2508.15795 Eq. (10) (2025 preprint), it is corroborated
at the UE tier by ST-HO's `p_local = 1 W @ 1 GHz`, and it is the only family among the sources read
that can be computed directly from scheduler events. Declare A-1 … A-13 as transparent assumptions,
re-source or explicitly assume the S8-derived coefficients (A-8, A-9, A-11, A-12) since that paper
was unreadable, and run the §E.4 sweeps before any energy claim is made in the paper.

---

## Appendix: explicit UNVERIFIED / INACCESSIBLE register

| Item | Status |
|---|---|
| S5 LyDRL (ACM TAAS 2025, DOI 10.1145/3715333) full text, all equations, all parameters | **INACCESSIBLE** (HTTP 403 anti-bot on article and PDF endpoints; closed access per Semantic Scholar) |
| S8 Liu et al. DCN 2023 (DOI 10.1016/j.dcan.2022.12.002) full text, Table 1 | **INACCESSIBLE** (ScienceDirect 403; SciEngine JS-only shell). Bibliographic metadata verified via Crossref |
| `kappa_helper = 5e-27`, `kappa_mec = 1e-27`, `f_mec = 10 GHz` as *sourced values* | **UNVERIFIED** (inherit S8's inaccessibility) |
| `cycles_per_bit = 300` (Zhao et al. 2018) | **NOT READ** in this review (outside the requested window) |
| `ue_tx_w = 1.0 W`, `mec_tx_w = 3.162 W` as *sourced values* | **UNVERIFIED** (S8 / Zhao 2018 not read) |
| S4 Sensors Eq. (2) exact grouping of the `g²/(omega·B_ik)` fraction | **UNVERIFIED** (PDF text layer garbles the fraction) |
| S4 Sensors the true absence of an `eta_i` value anywhere in the paper | **High confidence but not exhaustive** (full text read; no value found in §3, §5.1, or the parameter tables) |
| S7 arXiv:2508.09149 whether an explicit *virtual energy-deficit queue* exists | **UNVERIFIED-to-negative** (no such queue found in the rendered text; the HTML renderer dropped some math) |
| S1 ST-HO attribution of `p_down` to the vehicle receiver vs the RSU transmitter | **UNVERIFIED** (paper attributes the resulting energy to the onboard device but does not name the payer of `p_down` unambiguously) |
| S1 ST-HO Eq. (14) and Eq. (16) absolute-value bars | **RECONSTRUCTED** from surrounding prose; the PDF text layer garbles the glyphs |
| S3 COMA2C Eq. (12) second term `T_k^total` vs `E_k^total` | **PRINTED AS `T_k^total`**; appears to be a typo given the prose and Eq. (22), but reported verbatim, not corrected |
| Whether any 2024–2026 primary source exists for HELPER/relay V2V energy | **NOT FOUND** (search Q10); treated as a genuine literature gap |
