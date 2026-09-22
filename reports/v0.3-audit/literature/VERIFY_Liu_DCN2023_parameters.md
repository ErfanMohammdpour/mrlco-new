# Verification — Liu et al., "Joint computation offloading and resource allocation in vehicular edge computing networks" (DCN 2023)

**Verdict: ALL SIX CLAIMED PARAMETERS CONFIRMED.** The paper does use SV 1–2 GHz, RSU 10 GHz,
γ_SV = 5×10⁻²⁷, γ_RSU = 1×10⁻²⁷, vehicle TX = 30 dBm, B = 10 MHz. It also **includes the MEC/RSU server
compute energy** in the objective (Eq. 10, Table 1 γ_m), and **latency enters twice**: as a weighted term in
the cost (ω_l = 0.5) *and* as a deadline constraint (C1: T_i ≤ T_i^max).

---

## 1. Full citation + URLs

Shuang Liu ᵃ, Jie Tian ᵃ, Chao Zhai ᵇ, Tiantian Li ᵃ, "Joint computation offloading and resource allocation in
vehicular edge computing networks," *Digital Communications and Networks*, vol. 9, no. 6, pp. 1399–1410,
Dec. 2023 (received 22 Nov 2021; accepted 2 Dec 2022; issue Dec 2023).
ᵃ School of Information Science and Engineering, Shandong Normal University, Jinan 250358, China;
ᵇ School of Information Science and Engineering, Shandong University, Qingdao 266237, China.

- DOI: https://doi.org/10.1016/j.dcan.2022.12.002
- Publisher (KeAi/Elsevier) version of record: https://www.sciencedirect.com/science/article/pii/S2352864822002620 — **HTTP 403 / CAPTCHA to every automated client from this network**
- **Society platform (source used here):** https://www.sciengine.com/DCAN/doi/10.1016/j.dcan.2022.12.002
- **Article data API used for Table 1 + all 39 equations:** https://www.sciengine.com/sci-open/api/v1/open/article/figAndTable?articleBaseId=BE6EF1BE63DF40219C71410642808C16
- OpenAlex: https://api.openalex.org/works/doi:10.1016/j.dcan.2022.12.002 · Crossref: https://api.crossref.org/works/10.1016/j.dcan.2022.12.002 · DOAJ: https://doaj.org/article/cb7970628997409a9061dfa500d89229

### Disambiguation
The exact title is **unique**. Near-miss titles that are *different papers* and must not be confused:
"Joint Computation Offloading and **Wireless** Resource Allocation in Vehicular Edge Computing Networks"
(Zhang, Liu, Gu, Liang, Chen; Chinacom 2021, Springer, DOI 10.1007/978-3-030-99200-2_29); "Joint **offloading
decision** and resource allocation in vehicular edge computing networks" (DCN 2023, DOI
10.1016/j.dcan.2023.03.006); "Joint Offloading and Resource Allocation in Vehicular Edge Computing **and
Networks**" (Dai et al., GLOBECOM 2018). The project's guess "likely IEEE/Springer, ~2018–2021" is **wrong on
venue/year**: it is **Elsevier/KeAi, Digital Communications and Networks, accepted Dec 2022 / issue Dec 2023**.

### Access status (explicit)
The **prose of the article body could not be read**. Only the abstract page is reachable from this network.
ScienceDirect serves a CAPTCHA to curl, to `r.jina.ai`, to CORS proxies, and even inside a real Chromium
session (Cloudflare "Are you a robot?"); the `pdfft`, `/pdf`, `pdf.sciencedirectassets.com` and
`ars.els-cdn.com` endpoints all fail. Sci-Hub reports the paper is not in its database and points back to the
OA publisher page. Wayback has only the abstract snapshot (2024-04-12; the PDF snapshots are 403). SciEngine
holds metadata, Table 1 and the **full MathML of all 39 numbered equations**, but not the prose. **Therefore
the energy-scope *sentences* below could not be quoted verbatim; the proof given is the published equation
plus the published Table 1 row, which is stronger evidence than prose anyway.**

---

## 2. Energy scope — INCLUDES the MEC/RSU server compute energy

Proof (published equation + published table row, both from
https://www.sciengine.com/sci-open/api/v1/open/article/figAndTable?articleBaseId=BE6EF1BE63DF40219C71410642808C16):

- Eq. (10) — V2R/RSU mode: `E_{i,m} = P_i · t^{up}_{i,m} + C_i (f^R_{i,m})² · γ_m`
  → the **second term is the RSU/MEC server's compute energy**, with γ_m the server's switched-capacitance constant.
- Eq. (15) — V2V/SV mode: `E_{i,j} = P_i · t^{up}_{i,j} + C_i (f^S_{i,j})² · γ_j`
  → the **service-vehicle's compute energy**.
- Table 1 row (verbatim): **"Switch capacitance constant of RSU server γ_m | 1 × 10⁻²⁷"**
- Table 1 row (verbatim): **"Switch capacitance constant of SV γ_j | 5 × 10⁻²⁷"**

Scope boundary detail: the model is **fully offloaded** — Eq. (5)/(C5) is
`Σ_m x_{i,m} + Σ_j y_{i,j} = 1`, i.e. each task goes to exactly one RSU or one SV. Consequently **there is no
task-vehicle (UE) local-compute energy term at all**, and **no separate UE-tier γ** appears in Table 1. The
energy scope is therefore: *task-vehicle uplink transmission energy + the compute energy of whichever node
executes (RSU server or service vehicle)*. Task-vehicle compute is excluded because it never computes.

---

## 3. Exact energy formulas (verbatim transcription of the published MathML)

Rates (path-loss model, κ = path-loss exponent, δ₀ = reference channel gain):
- Eq. (3): `r_{i,m}(t) = B log(1 + δ₀ P H_{i,m} / (σ² [d_{i,m}(t)]^κ))`
- Eq. (4): `r_{i,j}(t) = B log(1 + δ₀ P H_{i,j} / (σ² [d_{i,j}(t)]^κ))`

Execution / total latency:
- Eq. (8): `t^{exe}_{i,m} = C_i / f^R_{i,m}` ;  Eq. (9): `t_{i,m} = t^{up}_{i,m} + t^{exe}_{i,m}`
- Eq. (13): `t^{exe}_{i,j} = C_i / f^S_{i,j}` ; Eq. (14): `t_{i,j} = t^{up}_{i,j} + t^{exe}_{i,j}`

**Energy:**
- Eq. (10) V2R: `E_{i,m} = P_i t^{up}_{i,m} + C_i (f^R_{i,m})² γ_m`
- Eq. (15) V2V: `E_{i,j} = P_i t^{up}_{i,j} + C_i (f^S_{i,j})² γ_j`
- Eq. (23): `E_i = Σ_{m∈M} x_{i,m} E_{i,m} + Σ_{j∈J} y_{i,j} E_{i,j}`

Monetary expense (linear pricing on allocated CPU frequency):
- Eq. (11): `e_{i,m} = μ_{i,m} f^R_{i,m}` ; Eq. (16): `e_{i,j} = μ_{i,j} f^S_{i,j}` ; Eq. (24): `e_i = Σ x_{i,m} e_{i,m} + Σ y_{i,j} e_{i,j}`

Distinctive "transmission gain" term (mobility-induced rate surplus, **subtracted** from the objective):
- Eq. (17): `g_{i,m} = ∫_0^{t^{up}_{i,m}} ( r_{i,m}(t) − r̄_{i,m} ) dt` ; Eq. (18) analogous for j
- Eq. (19): `r̄_{i,m} = R_i / t^{up}_{i,m}` ; Eq. (20): `r̄_{i,j} = R_i / t^{up}_{i,j}`

Note: `C_i` is used as a **cycle count** in Eqs. (8)/(10)/(13)/(15) (`t^exe = C_i/f`, `E ⊃ C_i f²γ`), although
Table 1 labels it "Computation cycles of task C_i" with value **"[0.5, 2.5] GHz"** — an internal unit
inconsistency *in the paper's own Table 1*. Report it verbatim; do **not** cite a cycles/bit value for this paper.

---

## 4. Parameter table — Table 1 "Simulation parameters." (verbatim, all rows)

Source: https://www.sciengine.com/sci-open/api/v1/open/article/figAndTable?articleBaseId=BE6EF1BE63DF40219C71410642808C16
(article page https://www.sciengine.com/DCAN/doi/10.1016/j.dcan.2022.12.002)

| Parameter (as printed) | Value (as printed) |
|---|---|
| Data size of task R_i | [100, 2000] KB |
| Computation cycles of task C_i | **[0.5, 2.5] GHz** *(unit inconsistent with "cycles" — as printed)* |
| Vehicle transmission power P | **30 dBm** |
| Bandwidth B | **10 MHz** |
| Noise spectral density N₀ | −174 dBm/Hz |
| Path loss exponent κ | 4 |
| Channel power gain at a reference distance δ₀ | −30 dB |
| Service Vehicle resource F_j^max | **[1, 2] GHz** |
| RSU server resource F_m^max | **10 GHz** |
| Maximum latency T_i^max | [0.5, 2.5] s |
| Switch capacitance constant of SV γ_j | **5 × 10⁻²⁷** |
| Switch capacitance constant of RSU server γ_m | **1 × 10⁻²⁷** |
| Weight of delay ω_l | 0.5 |
| Weight of cost ω_e | 0.5 |

Not in Table 1 (so **NOT VERIFIED**): number of task vehicles I / SVs J / RSUs M (they are swept in
Figs. 3–4 "Impacts of the number of TVs / SVs"), the per-SV association cap Q (appears in C7 and Figs. 8–9),
and any noise *power* value (only the spectral density −174 dBm/Hz is given).

---

## 5. Latency / delay handling

**Latency enters twice — weighted objective term AND constraint.** Reproduced from the published equations:

- Objective (Eq. 21): `U_i = ω_l T_i + ω_e E_i + e_i − Σ_{m∈M} x_{i,m} g_{i,m} − Σ_{j∈J} y_{i,j} g_{i,j}`
  with Eq. (22) `T_i = Σ_m x_{i,m} t_{i,m} + Σ_j y_{i,j} t_{i,j}` and Eq. (23) `E_i` as above.
  Problem (Eq. 25): `P1: min_{X,Y,F^R,F^S} Σ_{i∈I} U_i`.
  → **weighted sum of delay and energy** (plus a monetary price, minus a transmission gain). Table 1 fixes
  ω_l = 0.5, ω_e = 0.5. Figure captions confirm the weighting is studied explicitly: Fig. 6 "Impacts of
  different weight of time delay and energy consumption under four schemes", Fig. 7 "Impacts of weight of
  delay on the average delay and energy of TVs".
- Constraint (Eq. 25, C1): `T_i ≤ T_i^max, ∀i∈I`, with T_i^max = [0.5, 2.5] s.
  Capacity constraints C2 `Σ_i f^R_{i,m} ≤ F_m^max`, C3 `Σ_i f^S_{i,j} ≤ F_j^max`;
  C4 binary x,y; C5 full-offload equality; C6 Σx ≤ 1, Σy ≤ 1; C7 `Σ_i y_{i,j} ≤ Q`.

---

## 6. Penalty terms

There is **no fixed deadline-violation / dropout penalty constant**. Deadline and capacity limits are handled
by **Lagrangian dual variables** (Eq. 33):

`L(F^R,F^S,α,β,ρ) = ω_l T_i + ω_e E_i + e_i − Σ_m x_{i,m} g_{i,m} − Σ_j y_{i,j} g_{i,j} + α_i(T_i^max − T_i) + β_m(F_m^max − Σ_i x_{i,m} f^R_{i,m}) + ρ_j(F_j^max − Σ_i y_{i,j} f^S_{i,j})`

Dual problem (Eq. 35): `max D(α,β,ρ) s.t. α,β,ρ ≻ 0`; duals updated by subgradient (Eq. 39):
`α_i(t_d+1) = [α_i(t_d) + Δ_{α_i}(T_i^max − T_i)]⁺`, and likewise for β_m, ρ_j. Gradients in Eqs. (36)–(37).
The **only hard feasibility cap on association** is C7 `Σ_i y_{i,j} ≤ Q` (max TVs served by one SV).
A separate penalty/violation constant does **not** exist in the published model.

---

## 7. Verification verdict per claimed parameter

| Claimed value | Verdict | Evidence (URL) |
|---|---|---|
| SV CPU 1–2 GHz | **CONFIRMED** — Table 1 "Service Vehicle resource F_j^max = [1, 2] GHz" | https://www.sciengine.com/sci-open/api/v1/open/article/figAndTable?articleBaseId=BE6EF1BE63DF40219C71410642808C16 |
| RSU CPU 10 GHz | **CONFIRMED** — Table 1 "RSU server resource F_m^max = 10 GHz" | same |
| γ_SV = 5×10⁻²⁷ | **CONFIRMED** — Table 1 "Switch capacitance constant of SV γ_j = 5 × 10⁻²⁷" | same |
| γ_RSU = 1×10⁻²⁷ | **CONFIRMED** — Table 1 "Switch capacitance constant of RSU server γ_m = 1 × 10⁻²⁷" | same |
| vehicle TX power 30 dBm | **CONFIRMED** — Table 1 "Vehicle transmission power P = 30 dBm" | same |
| bandwidth B = 10 MHz | **CONFIRMED** — Table 1 "Bandwidth B = 10 MHz" | same |

All six came from the same published Table 1; the article landing page carrying that table is
https://www.sciengine.com/DCAN/doi/10.1016/j.dcan.2022.12.002 and the DOI is
https://doi.org/10.1016/j.dcan.2022.12.002.

### Secondary check — Liang et al. arXiv:2411.10770 is NOT the source of these values
Read independently from the OA PDF (https://arxiv.org/pdf/2411.10770), Table II:

`C^exe_pk, C^exe_rj = 24 cycles/bit` · `f_pk = [1, 2.5] GHz` (parked vehicle) · `f_rj = [4, 6] GHz` (RSU) ·
`κ_v = 10⁻²⁷`, `κ_r = 10⁻²⁸` · `P_t = 0.28183815 W (= 24.5 dBm)` · `N₀ = 1.2589×10⁻¹³ W` · `W_b = 15 MB`
(unit printed as "MB") · `D_qi = [10, 30] MB` · `T_max^i = [100, 200] ms` · `d₀ = 100 m`, `δ = 2`,
`η = 1.63726×10⁻⁹`, `β = 10⁶ cycles`, `θ = 10×10⁶ cycles`, `p_th = 0.95`, `ϖ = 1 KB`, `α = 1`.

That is **1e-27 / 1e-28 and 1–2.5 / 4–6 GHz, 24.5 dBm, 15 (MB)** — it does **not** match 5e-27 / 1e-27 /
1–2 GHz / 10 GHz / 30 dBm / 10 MHz. So the project's numbers are **not** mis-attributed from Liang et al.
(a correction to the earlier hypothesis); they are genuinely Liu et al. Table 1. Liang's energy form is
also `E^exe = κ f² D φ C^exe` at both PV and RSU tiers (Eqs. 16, 20), i.e. the same `κCf²` family as Liu's
Eqs. (10)/(15).

---

## 8. Residual limitations
- The article **prose** (model narrative, notation table, any qualifying sentences) could **not** be read;
  the LaTeX/XML used here is the publisher platform's machine-readable rendering of the published article,
  and its Table 1 and equations are internally consistent with each other (γ_m/γ_j in Table 1 match Eqs.
  10/15; ω_l/ω_e in Table 1 match Eq. 21; T_i^max matches C1), so it is treated as faithful.
- No cycles-per-bit value exists in this paper; the "computation cycles" row is printed in GHz.
- `Q`, and the counts I/J/M, are not tabulated.
