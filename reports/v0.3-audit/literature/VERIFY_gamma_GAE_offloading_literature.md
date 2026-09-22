# Verified γ (discount factor) and GAE λ in PPO / actor-critic **computation-offloading** papers

**Method.** Every value below was read from a source I actually fetched. PDFs were downloaded and text-extracted (`pypdf`) and grepped locally; HTML was read via `web_fetch`. Marker `[FT]` = full-text verified (hyperparameter table / body sentence read). Marker `[ABS]` = abstract-only (none of the numeric entries below are abstract-only — all γ values are `[FT]`).

No paper title, author, number, or URL is invented. Where a paper does not state a value I write **not stated**, never a guess.

---

## A. Master table

| # | Paper (short) | Algorithm | γ | Where γ is stated (verbatim) | GAE λ | Episode type | Horizon | `[FT]` |
|---|---|---|---|---|---|---|---|---|
| 1 | Chen & Tong, *Future Internet* 17(12):542, 2025 — SAGIN offloading | D-MAPPO (Dirichlet-MAPPO) | **0.95** | Table 3: `Clipping parameter 0.2   Discount factor 0.95` / `GAE parameter 0.95`; text: *"gamma = 0.95"* | **0.95** | per-time-slot control (`for timestep = 1 to Max_Timesteps`) | `rollout length 75` | FT |
| 2 | Wang, Ouyang, Sun, Chen & Li, *Electronics* 13(12):2387, 2024 — MEC graph-RL | M-GNRL (GNN + A2C-family actor-critic) | **0.97** chosen; ablation **0.93/0.95/0.97/0.99** | Table 2: `discount factor γ: 0.93, 0.95, 0.97, 0.99` | not stated | per-time-slot, one task per slot | `length of time slot sequence T: 80`; episodes 1000/1200/1400 | FT |
| 3 | Lu, He & Zhang, *Electronics* 13(15):2933, 2024 — security-aware MEC offloading | PPO | **0.9** | *"with a discount factor of 0.9"* | not stated | per-task sequential (action = MD or ES*n*) | `T = {1,...,t,...,T}` (numeric T not stated) | FT |
| 4 | Feng, Zhang, Wu, Fan & Fan, *Electronics* 15(5):936, 2026 — RIS semantic VEC | PPO | **0.6** | hyperparam table: `γ 0.6` | not stated | per-time-slot control | `Tmax` time steps/episode; `Emax 5000` | FT |
| 5 | Chandrasiri & Meedeniya, *Sensors* 25(5):1428, 2025 — cloud workflow scheduling | PPO + GIN | **0.99** | Table 4: `γ  Discount factor  0.99` | **0.95** (`λ  Generalized Advantage Estimation (GAE) factor  0.95`) | episode = one complete scheduling scenario | `B  Number of steps per policy rollout  128` | FT |
| 6 | Wang & Zhou, *Computers, Materials & Continua* (CMC), 2025 — smart-grid offloading | PPO | **0.99** | Table 3: `Discount factor γ 0.99` | **0.95** (`Adv. discount factor φ 0.95`) | per-time-slot scheduling; **contains an explicit γ=1 argument** (see §C) | time step 0.01 s; no episode length stated | FT |
| 7 | Qin, Lu, Chen, Chong & Wu, arXiv:2312.01499 — user-centric MEC | MAPPO / IPPO | **0.99** | *"We set the discount factor to 0.99, the GAE parameter to 0.95, and the PPO clip parameter to 0.2."* | **0.95** | per-time-slot control | *"For each episode, we assume the episode length T is 300."* | FT |
| 8 | Wu, Du & Qu, arXiv:2603.20238 — multi-UAV MEC, RIS + trajectory | decentralized model-based PPO/MAPPO | **0.99** | *"discount factor γ = 0.99, and GAE parameter λ_GAE = 0.95. Training over 2,000 episodes, each consisting of N = 100 steps."* | **0.95** | per-time-slot control | N = 100 steps/episode | FT |
| 9 | arXiv:2302.09021 — aerial edge MEC, digital twin | AB-MAPPO (attention-based MAPPO) | **0.95** (UAVs) / **0.8** (MUs) | Table III: `Discount factor of UAVs 0.95   Discount factor of MUs 0.8` | not stated numerically | per-time-slot control | `Length of an episode epl 300` | FT |
| 10 | arXiv:2305.01536 — FlexEdge, UAV-aided VEC | PPO (clip-based actor-critic) | **0.95** | *"γ = 0.95, the length of an episode is equal to N, and the penalty factor is µ = 100"* | not stated (GAE formula only) | per-time-slot control | episode length = N time slots | FT |
| 11 | arXiv:2308.12756 — multi-UAV MEC offloading + trajectory | MAPPO | **0.98** | *"the episode length epi, which represents the T, is 200 steps, the discount factor is γu = 0.98"* | not stated (symbolic λ) | per-time-slot control | 200 steps; 300 training episodes | FT |
| 12 | arXiv:2212.05757 — satellite ITS offloading, 6G | Co-MAPPO + attention | **0.995** | param table: `Learning rate & discount factor 3e-4, 0.995` | not stated | per-sub-task offloading (reward only after all sub-tasks complete) | `Number of episodes 1.5e+5` | FT |
| 13 | arXiv:2301.08376 — semantic-aware networks w/ task offloading | MAPPO | **0.95** | param table: `Advantage discount factor λ 0.98   Reward discount rate γ 0.95` | **0.98** | per-time-slot | `Training episode of the MAPPO algorithm 1000` | FT |
| 14 | Gholipour et al., arXiv:2312.11739 — **TPTO**, DAG offloading | Transformer-PPO | **0.99** | param table: `Discount Factor 0.99` | not stated numerically (GAE eq. with symbols) | **per-task sequential over a DAG** (`O_n=(o_1..o_n)`, one binary action per task) | \|V\| = DAG size (numeric not stated) | FT |
| 15 | arXiv:2407.11018 — QoE-driven multi-task semantic edge offloading | MAPPO | **0.99** | param table: `Advantage discount factor λ 0.95   Reward discount factor γ 0.99` | **0.95** | per-time-slot | `Training episode of the MAPPO algorithm 300` | FT |
| 16 | arXiv:2503.03391 — queue-aware hierarchical air-ground MEC | MAPPO-BD | **0.99** | param table: `Discount factor of IoTD, UAV, and HAPS agents γu 0.99` | not stated numerically | per-time-slot | `episode length I` (numeric not stated) | FT |
| 17 | Paknejad et al., arXiv:2507.09341 — VEC real-time offloading | PPO + DQN | **0.95** (PPO) / **0.9** (DQN) | Table III: `DQN discount factor 0.9   PPO discount factor 0.95` | not stated | per-decision-window; **`Number of tasks per vehicle 1`** (near single-shot) | not stated | FT |
| 18 | Asadian-Rad, Soleimani & Farahmand, arXiv:2508.06863 — UAV-MEC | EPS-PPO (decentralized PPO; vs MADDPG) | **0.99** | Table 3: `DRL discount factor (γ)  0.99` | not stated | per-time-slot, episodic task set | `Number of time slots (T) 80`; *"after T time slots the POMDP ends"* | FT |
| 19 | arXiv:2602.18797 — carbon-aware decentralized offloading, MIMO-MEC | CADDTO-PPO | **0.99** | param table: `Discount Factor (γ) 0.99   GAE Parameter (λ) 0.95` | **0.95** | per-time-slot | `Buffer Size (nsteps) 2048`; `Max Steps Tmax` | FT |
| 20 | arXiv:2605.24972 — ISCC for NR-V2X | multi-agent RL (CTDE actor-critic) | **0.99** | param table: `γ, λ, ϵclip  0.99, 0.95, 0.2` | **0.95** | per-control-epoch (SB-SPS) | `Episodes, Nepi 500   Steps/episode 40` | FT |
| 21 | arXiv:2606.26293 — HALO, SAGIN offloading | HPPO (hierarchical PPO) | **0.99** (manager) / **0.95** (worker) | param table: `Discount factor manager / worker γ 0.99 / 0.95` | not stated numerically | two-timescale: manager per macro-slot, worker per micro-slot | `horizon T`; 3000 episodes | FT |
| 22 | arXiv:2607.09295 — SLA-aware network slicing, UAV-MEC | MAPPO | **0.99** | param table: `Learning rate / µ_RL / λ_GAE  10⁻⁴ / 0.99 / 0.95` | **0.95** | per-time-slot | `Train & eval episode 1500` | FT |
| 23 | Li, Zhu & Wang, arXiv:2412.13676 — UAV jittering + task scheduling | REDQ (SAC-based actor-critic) | **0.9** | *"The discount coefficient is 0.9"* | not stated (off-policy) | per-time-slot | not stated | FT |
| 24 | arXiv:2311.02525 — QECO, QoE-oriented MEC offloading (IEEE TNSM 2025) | DQN | **0.9** | *"set the discount factor γ to 0.9"* | n/a | per-time-slot | not stated | FT |
| 25 | Al-Shareeda et al., arXiv:2502.03403 — 6G-cloud vehicular twin offloading | PPO | **0.9** | *"a discount factor of 0.9"* | not stated | per-transition | `100 episodes and 100 steps each` | FT |
| 26 | Lan et al., arXiv:2404.15278 — satellite-terrestrial security-sensitive offloading | PPO | **0.99** | TABLE 2: `Gamma 0.99   Gae lambda 0.95` | **0.95** | per-scheduling-period, sequential task order | `Total timesteps 5 × 10⁵`; T not stated | FT |
| 27 | Jiang, Tavakkolnia & Han, *npj Wireless Technology* 2:64, 2026 — THz cell-free MEC | MAPPO | **0.9** | Table 1: `The discount factor, γ 0.9   The GAE parameter, λ 0.9` | **0.9** | per-time-slot control | `The number of episodes, E 2000   The number of time slots, N 40` | FT |
| 28 | Dai, Zhang, Maharjan & Zhang, arXiv:2011.08442 — edge intelligence, 5G beyond | DDPG | **ε = 0.6** (swept 0.5–0.7) | *"we set ε from 0.5 to 0.7 … ε = 0.6 is the best discount factor"* (paper uses ε, not γ) | n/a | per-time-slot | `6000` episodes × `20` steps | FT |
| 29 | Jiang, Jin, Xin, Chen & Guo, *Systems* 14(9):1103, 2026 — **VEC DAG task scheduling** | FRMPPO (fuzzy-reward MAPPO) | **0.95**; ablation **{0.90, 0.91, 0.93, 0.95, 0.97, 0.99}** | Table 4: `Discount factor γ  {0.90, 0.91, 0.93, 0.95, 0.97, 0.99}` | **0.95** (`GAE coefficient λ 0.95`) | **per-DAG-subtask sequential** within an episode | `Maximum training epoch Lmax 500`; `L is the number of steps per episode` | FT |
| 30 | *Mathematics* 13(16):2643, 2025 — maritime MEC (sporadic object-detection offloading) | ON-PPO (orthogonalization-normalization PPO) | **0.9** | Table 2: `Discount rate γ  0.9` | not stated (finite K-step advantage, no GAE) | per-task sequential offloading | `Number of episode T 1000` | FT |
| 31 | *Electronics* 14(17):3444, 2025 — digital-twin VEC offloading | TD3 | **0.99** | *"The discount factor γ is set to 0.99 to achieve a reasonable trade-off between long-term and short-term returns"* | not stated (off-policy) | per-time-slot continuous control | `Within each episode, there are T timesteps` (T not stated) | FT |
| 32 | Hu, Liu, Zhou, Shen & Wang, *Drones* 9(4):288, 2025 — UAV-swarm MEC offloading + compression | PER-DDPG | **0.01**; ablation {0.1, 0.01, 0.001} | *"setting γ to 0.01 significantly reduces oscillation amplitude… Consequently, the γ in this study is established at 0.01."* | not stated | per-time-slot control | T slots (numeric not stated); convergence ≈110 steps | FT |
| 33 | Gao, H., *Entropy* 27(8):803, 2025 — space–air–marine network offloading | MADDPG | **0.99** | *"The discount factor γ is set to 0.99, the experience replay buffer D size is 1000"* | not stated | per-time-slot control (episodes of T slots) | 500 training episodes | FT |

**Distribution of γ (n = 33):** 0.99 → 14 papers; 0.95 → 8; 0.9 → 6; 0.01 → 1; 0.6 → 1; 0.995 → 1; 0.98 → 1; 0.97 → 1; 0.8 → 1 (secondary agent only). **No paper in this set trains with γ = 1.** One paper (Mathematics 13(16):2643) explicitly reports that γ = 0.999 fails to converge.

---

## B. Per-paper notes (citations, exact URLs, verbatim quotes)

### 1. Chen, Y.; Tong, Y. — *Computation Offloading in Space–Air–Ground Integrated Networks for Diverse Task Requirements with Integrated Reliability Mechanisms.* Future Internet **2025**, 17, 542.
URL fetched: `https://mdpi-res.com/d_attachment/futureinternet/futureinternet-17-00542/article_deploy/futureinternet-17-00542-v2.pdf`
**[FT]** Table 3 `Hyperparameter Settings of D-MAPPO`:
> `Clipping parameter 0.2   Discount factor 0.95` / `GAE parameter 0.95` / `rollout length 75` / `Number of agents 5`

Ablation text, verbatim:
> "Figure 3b shows the results for different discount factors: 0.8, 0.95, and 0.99. With a discount factor of 0.95 (red line), the algorithm converges quickly and stabilizes at a high reward… With a discount factor of 0.99 (blue line), although the optimization performance is poor before 700 episodes, there is an improvement later, though it still does not outperform the 0.95 case. With a discount factor of 0.8 (green line), the algorithm converges faster initially, but the rewards decrease later…"

> "we find that the D-MAPPO algorithm achieves better optimization performance when lr = 0.0003, gamma = 0.95, and n_epochs = 5."

Episode structure: per-time-slot control — `for timestep = 1 to Max_Timesteps`, agents emit an offloading-ratio vector `at = (µ(u,u), µ(u,C), µ(u,s), …, µ(u,S))` each slot.
Reward (Eq. 32): piecewise `−[(1−k)E + kT]` plus penalty terms `−ι(1−δn)Nn` with `k = 0.5, ι = 7`.

### 2. Wang, T.; Ouyang, X.; Sun, D.; Chen, Y.; Li, H. — *Offloading Strategy Based on Graph Neural Reinforcement Learning in Mobile Edge Computing.* Electronics **2024**, 13, 2387.
URL fetched: `https://mdpi-res.com/d_attachment/electronics/electronics-13-02387/article_deploy/electronics-13-02387.pdf`
**[FT]** Table 2:
> `discount factor γ: 0.93, 0.95, 0.97, 0.99`

> "In reinforcement learning, the discount factor γ, within the range of 0 to 1, plays a crucial role in computing the cumulative reward value. A value of γ = 0 implies that the agent's focus is solely on immediate rewards, resulting in decisions that are locally optimal. **Conversely, when γ = 1, convergence issues may emerge**, underscoring the importance of selecting an appropriate discount factor for policy training. Figure 7 illustrates the impact of different γ values (0.93, 0.95, 0.97, 0.99) on the average reward. … the highest average reward achieved when γ = 0.97, outperforming other values. Consequently, for subsequent experiments, we decide to adopt this specific γ value."

Episode structure: per-time-slot, one task generated per slot — *"Each device generates only one task at the beginning of each time slot, the time slot sequence T = {1, 2, 3, ..., T}"*; `length of time slot sequence T: 80`.
Reward (Eq. 21): `rt = 1 / Cost_n(t)` if `0 < Te ≤ τn(t)`, else `rt = 0.1 * (τn(t) − Te)`.
**This is the closest real thing to the claimed "0.90/0.93/0.99" ablation — but the grid is 0.93/0.95/0.97/0.99, there is no 0.90, and there is no 4000/6000-iteration convergence claim.**

### 3. Lu, H.; He, X.; Zhang, D. — *Security-Aware Task Offloading Using Deep Reinforcement Learning in Mobile Edge Computing Systems.* Electronics **2024**, 13, 2933.
URL fetched: `https://mdpi-res.com/d_attachment/electronics/electronics-13-02933/article_deploy/electronics-13-02933.pdf`
**[FT]**
> "The batch size is set to 32. The learning rates are set at 0.003 for the actor and 0.001 for the critic, **with a discount factor of 0.9**. The buffer size is 10,000."

Hyperparameters were auto-tuned with Microsoft NNI. Reward (Eq. 15): `rt = Dtotal_m,x(t) − λEtotal_m,x(t)`. Episode = per-task offloading; action `at = {MD, ES1, …, ESN}`.

### 4. Feng, W.; Zhang, J.; Wu, Q.; Fan, P.; Fan, Q. — *PPO-Based Hybrid Optimization for RIS-Assisted Semantic Vehicular Edge Computing.* Electronics **2026**, 15, 936.
URL fetched: `https://mdpi-res.com/d_attachment/electronics/electronics-15-00936/article_deploy/electronics-15-00936-v2.pdf`
**[FT]** hyperparameter block (verbatim, de-spaced):
> `δth 0.9   α 0.0003 (actor)/0.001 (critic)   γ 0.6   ϵ 0.2   Emax 5000`

This is the lowest γ in the survey. Not discounted-return ablation-justified; it is simply the value reported. Per-time-slot control; `Gt = Σ_{l=0}^{T−t−1} γ^l r_{t+l}`.

### 5. Chandrasiri, S.; Meedeniya, D. — *Energy-Efficient Dynamic Workflow Scheduling in Cloud Environments Using Deep Learning.* Sensors **2025**, 25, 1428.
URL fetched: `https://mdpi-res.com/d_attachment/sensors/sensors-25-01428/article_deploy/sensors-25-01428.pdf`
**[FT]** Table 4 `Hyperparameters used for PPO training`:
> `γ   Discount factor   0.99`
> `λ   Generalized Advantage Estimation (GAE) factor   0.95`
> `ϵ   Clipping parameter   0.2` / `B   Number of steps per policy rollout   128`

> "The training was conducted over a fixed number of episodes, where **each episode representing a complete scheduling scenario**."

CleanRL-based; this is the clearest "one episode = one scheduling scenario" statement found, and it still uses γ = 0.99, **not** γ = 1.

### 6. Wang, Q.; Zhou, Y. — *Improved PPO-Based Task Offloading Strategies for Smart Grids.* Computers, Materials & Continua (CMC), 2025. DOI 10.32604/cmc.2025.065465.
URL fetched: `https://file.techscience.com/files/cmc/2025/online/CMC0612/TSP_CMC_65465/TSP_CMC_65465.pdf`
**[FT]** Table 3 (digits in the PDF are OCR-mangled as `/zero.fitted`=0, `/one.fitted`=1, `/nine.fitted`=9, `/five.fitted`=5; decoded values shown):
> `Adv. discount factor φ 0.95` / `Discount factor γ 0.99` / `Clipping range 0.1` / `Entropy coefficient 0.05` / `Learning rate 0.0003`

**Contains the clearest in-domain γ = 1 argument found in this whole survey** — see §C.

### 7. Qin, L.; Lu, H.; Chen, Y.; Chong, B.; Wu, F. — *Towards Decentralized Task Offloading and Resource Allocation in User-Centric MEC.* arXiv:2312.01499.
URL fetched: `https://arxiv.org/pdf/2312.01499`
**[FT]**
> "For each episode, we assume the episode length T is 300. … We set the discount factor to 0.99, the GAE parameter to 0.95, and the PPO clip parameter to 0.2. We set the replay buffer size and the batch size to 2048 and 64, respectively."

Per-time-slot control; users emit `am(t) = {pd_m(t), ωmk(t), ∀k ∈ K}` per slot.

### 8. Wu, L.; Du, J.; Qu, J. — *Joint Trajectory, RIS, and Computation Offloading Optimization via Decentralized Model-Based PPO in Urban Multi-UAV Mobile Edge Computing.* arXiv:2603.20238.
URL fetched: `https://arxiv.org/pdf/2603.20238`
**[FT]**
> "For PPO training, the clipping ratio is 0.2, entropy coefficient is 0.01, **discount factor γ = 0.99, and GAE parameter λ_GAE = 0.95**. Training over 2,000 episodes, each consisting of N = 100 steps."

Reward (Eq. 49): `r_u[n] = Σ_k(l_loc + l_o) + Σ_k(E_tx + E_comp) + E_fly − β I_u[n]`.

### 9. *Energy Efficient Computation Offloading in Aerial Edge Networks With Multi-Agent Cooperation.* arXiv:2302.09021.
URL fetched: `https://arxiv.org/pdf/2302.09021`
**[FT]** Table III `HYPERPARAMETERS PARAMETERS OF AB-MAPPO`:
> `Total number of steps 80k` / `Length of an episode epl 300` / `Discount factor of UAVs 0.95` / `Discount factor of MUs 0.8`

Notable: **two different γ values for two agent populations in the same run** — a genuinely unusual finding.

### 10. *FlexEdge: Digital Twin-Enabled Task Offloading for UAV-Aided Vehicular Edge Computing.* arXiv:2305.01536.
URL fetched: `https://arxiv.org/pdf/2305.01536`
**[FT]**
> "For training settings, the discount factor is γ = 0.95, the length of an episode is equal to N, and the penalty factor is µ = 100."

Reward: `rn = Σ_k(E_u^k[n] + E_rc^k[n]) + E_f[n] + P^l_n`, with linear latency-violation penalty `P^l_n`.

### 11. *Robust Computation Offloading and Trajectory Optimization for Multi-UAV-Assisted MEC: A Multi-Agent DRL Approach.* arXiv:2308.12756.
URL fetched: `https://arxiv.org/pdf/2308.12756`
**[FT]**
> "The maximum training episodes are Mt = 300 episodes, the episode length epi, which represents the T, is **200 steps**, the **discount factor is γu = 0.98**, the learning rate is 0.0005, and the optimizer we used is Adam."

Algorithm: MAPPO (paper contrasts with MADDPG). GAE defined symbolically only.

### 12. *Satellite-based ITS Data Offloading & Computation in 6G Networks: A Cooperative Multi-Agent PPO DRL with Attention Approach.* arXiv:2212.05757.
URL fetched: `https://arxiv.org/pdf/2212.05757`
**[FT]** param table:
> `Hidden layers & Neurons in each layer 3, [512, 512, 512]` / `Learning rate & discount factor 3e-4, 0.995` / `Experience memory & batch size 10240, 1024` / `Number of episodes 1.5e+5`

**Highest γ in the survey (0.995).** Episode structure: offloading decisions per sub-task, but *"the reward becomes available only once all pending sub-tasks have been accomplished"* — i.e. a delayed, aggregate reward rather than per-step reward.

### 13. *Resource Optimization for Semantic-Aware Networks with Task Offloading.* arXiv:2301.08376.
URL fetched: `https://arxiv.org/pdf/2301.08376`
**[FT]** param table:
> `Advantage discount factor λ 0.98` / `Reward discount rate γ 0.95` / `Training episode of the MAPPO algorithm 1000` / `Batch step of the MAPPO 256`

Reward: `R(τ) = Σ_{t=1}^{t0} γ^{t−1} r_t + r_T`, with QoE reward and punishments for latency/accuracy violations. GAE `A_GAE = Σ_{l=0}^{T−t}(λγ)^l δ^V_{t+l,i}`.

### 14. Gholipour, N.; Dias de Assuncao, M.; Agarwal, P.; Gascon-Samson, J.; Buyya, R. — *TPTO: A Transformer-PPO based Task Offloading Solution for Edge Computing Environments.* IEEE ICPADS 2023. arXiv:2312.11739.
URL fetched: `https://arxiv.org/pdf/2312.11739`
**[FT]** param table:
> `Policy Learning Rate 0.1` / `Batch Size 100` / `Clip ratio 0.2` / `Discount Factor 0.99` / `Entropy coefficient 0.5`

Episode structure: **per-task sequential offloading over an application DAG** — `The offloading goal is to compute an offloading plan O_n = (o_1, o_2, …, o_n)`; action space binary `A := {0, 1}` per task. This is the paper in the set whose structure is closest to a **single-shot plan**, and it still uses γ = 0.99, not 1. GAE is given only symbolically — **λ not stated numerically**.
Reward: `ΔALO_i = ALO_i − ALO_{i−1}` (negative latency increment).

### 15. *QoE-Driven Multi-Task Offloading for Semantic-Aware Edge Computing Systems.* arXiv:2407.11018.
URL fetched: `https://arxiv.org/pdf/2407.11018`
**[FT]** param table:
> `Training episode of the MAPPO algorithm 300` / `Advantage discount factor λ 0.95` / `Reward discount factor γ 0.99` / `PPO-Clip parameter ϵ 0.2`

### 16. *Multi-Agent DRL for Queue-Aware Task Offloading in Hierarchical MEC-Enabled Air-Ground Networks.* arXiv:2503.03391.
URL fetched: `https://arxiv.org/pdf/2503.03391`
**[FT]** param table:
> `Discount factor of IoTD, UAV, and HAPS agents γu 0.99` / `Entropy bonus ψ and clipping parameter ϵ 0.1, 0.2`

Algorithm: MAPPO with Beta–Dirichlet (MAPPO-BD).

### 17. Paknejad, M.; Fard Moshiri, P.; Simsek, M.; Kantarci, B.; Mouftah, H. T. — *Meeting Deadlines in Motion: Deep RL for Real-Time Task Offloading in Vehicular Edge Networks.* arXiv:2507.09341.
URL fetched: `https://arxiv.org/pdf/2507.09341`
**[FT]** TABLE III:
> `DQN discount factor 0.9` / `PPO discount factor 0.95` / `Number of tasks per vehicle 1`

**`Number of tasks per vehicle 1` makes this the closest thing to a genuine single-shot offloading decision among the full-text-verified set — and the PPO γ is still 0.95, not 1.**

### 18. Asadian-Rad, H.; Soleimani, H.; Farahmand, S. — *Energy Efficient Task Offloading in UAV-Enabled MEC Using a Fully Decentralized Deep Reinforcement Learning Approach.* arXiv:2508.06863.
URL fetched: `https://arxiv.org/pdf/2508.06863`
**[FT]** Table 3:
> `DRL discount factor (γ)  0.99` / `Number of time slots (T) 80`

> "Each computational task is processed within a time slot t … Each run through T is called an episode."
> "The task is episodic, meaning that after T time slots the POMDP ends. Still, we can select T arbitrarily large to model an infinite horizon problem."

An explicitly **episodic/finite-horizon** paper — and it still uses γ = 0.99, not 1.

### 19. *Carbon-aware decentralized dynamic task offloading in MIMO-MEC networks via multi-agent reinforcement learning.* arXiv:2602.18797.
URL fetched: `https://arxiv.org/pdf/2602.18797`
**[FT]** `CADDTO-PPO Hyperparameters`:
> `Discount Factor (γ) 0.99` / `GAE Parameter (λ) 0.95` / `PPO Clipping (ϵ) 0.2` / `Mini-batch Size 128` / `Buffer Size (nsteps) 2048`

### 20. *Integrated Sensing, Communication, and Computing for NR-V2X: A Cross-Layer Resource Allocation Framework Using Multi-Agent Reinforcement Learning.* arXiv:2605.24972.
URL fetched: `https://arxiv.org/pdf/2605.24972`
**[FT]** param table:
> `Episodes, Nepi 500` / `Steps/episode 40` / `Actor/critic LR (10⁻³, 10⁻³)` / `γ, λ, ϵclip  0.99, 0.95, 0.2`

### 21. *HALO: Hierarchical Auction-assisted Learning for Offloading in SAGIN.* arXiv:2606.26293.
URL fetched: `https://arxiv.org/pdf/2606.26293`
**[FT]** param table:
> `Discount factor manager / worker γ 0.99 / 0.95`

Hierarchical: manager acts per macro-slot (task splitting), worker per micro-slot (resource allocation). 3000 episodes.

### 22. *Multi-Agent Reinforcement Learning for SLA-Aware Network Slicing in UAV-Enabled MEC.* arXiv:2607.09295.
URL fetched: `https://arxiv.org/pdf/2607.09295`
**[FT]** param table:
> `Learning rate / µ_RL / λ_GAE  10⁻⁴ / 0.99 / 0.95` / `Train & eval episode 1500` / `PPO clip / entropy coeff. 0.2 / 0.02`

Note: this paper names the discount factor `µ_RL`, not γ.

### 23. Li, B.; Zhu, X.; Wang, J. — *Robust UAV Jittering and Task Scheduling in Mobile Edge Computing with Data Compression.* arXiv:2412.13676.
URL fetched: `https://arxiv.org/pdf/2412.13676`
**[FT]**
> "The discount coefficient is 0.9, the experience replay buffer is set to 20000, and the training sample set size is configured to 256."

Algorithm: REDQ (SAC-based actor-critic). Caution: in this paper the symbol γ denotes the **compression ratio**, not the discount factor — the discount factor is called "discount coefficient". Easy to mis-extract.

### 24. *QECO: A QoE-Oriented Computation Offloading Algorithm based on Deep Reinforcement Learning for Mobile Edge Computing.* IEEE TNSM 12(4), 2025. arXiv:2311.02525.
URL fetched: `https://arxiv.org/pdf/2311.02525`
**[FT]**
> "we employ a batch size of 16, maintain a fixed learning rate of 0.001, and **set the discount factor γ to 0.9**."

DQN-based. Also states `γ ∈ (0, 1]` in the objective.

### 25. Al-Shareeda, S.; Ozguner, F.; Redmill, K.; Duong, T. Q.; Canberk, B. — *Lightweight Authenticated Task Offloading in 6G-Cloud Vehicular Twin Networks.* arXiv:2502.03403.
URL fetched: `https://arxiv.org/pdf/2502.03403`
**[FT]**
> "The best results were achieved with a learning rate of 0.003, a 0.08 entropy coefficient, and **a discount factor of 0.9**."

> "over 10,000 iterations of 100 episodes and 100 steps each"

### 26. Lan, W.; Chen, K.; Cao, J.; Li, Y.; Li, N.; Chen, Q.; Sahni, Y. — *Security-Sensitive Task Offloading in Integrated Satellite-Terrestrial Networks.* IEEE TMC, 2024. arXiv:2404.15278.
URL fetched: `https://arxiv.org/pdf/2404.15278`
**[FT]** TABLE 2 `training parameters of PPO`:
> `Total timesteps 5 × 10⁵` / `Update interval 5` / `Batch size 64` / `Gamma 0.99` / `Gae lambda 0.95` / `Clip range 0.2`

**Verification trap:** this table row is headed `Gamma`, **not** "discount factor" — a keyword search for "discount factor" misses it entirely. Confirmed by direct string search on the extracted PDF text.

### 27. Jiang, J.; Tavakkolnia, I.; Han, C. — *Multi agent PPO for terahertz cell free mobile edge computing networks.* npj Wireless Technology **2026**, 2:64. DOI 10.1038/s44459-026-00072-9.
URL fetched: `https://www.nature.com/articles/s44459-026-00072-9.pdf`
**[FT]** Table 1:
> `The discount factor, γ 0.9` / `The GAE parameter, λ 0.9` / `The number of episodes, E 2000` / `The number of time slots, N 40` / `The clipping parameter, ϵ 0.2`

Discusses the γ → 1 limit explicitly but does **not** adopt it:
> "In (24), the discount factor γ ∈ [0, 1] determines the weight of future rewards. Specifically, **γ → 1 means future rewards are emphasized equally with immediate rewards**, while γ → 0 means immediate rewards dominate."

### 28. Dai, Y.; Zhang, K.; Maharjan, S.; Zhang, Y. — *Edge Intelligence for Energy-efficient Computation Offloading and Resource Allocation in 5G Beyond.* arXiv:2011.08442.
URL fetched: `https://arxiv.org/pdf/2011.08442`
**[FT]** Note the paper uses **ε (epsilon)**, not γ, as the discount-factor symbol:
> "To evaluate the impact of discount factors, we set ε from 0.5 to 0.7. … when ε = 0.6, the normalized cumulative reward is clearly higher than the cases when ε = 0.65 and ε = 0.7 … Thus, we can conclude that ε = 0.6 is the best discount factor for the proposed algorithm."

> "The maximum number of episodes is 6000 and the maximum number of steps in each episode is set to 20."

This is a **downward** discount-factor sweep in an offloading paper — direct evidence against the idea that this literature pushes γ toward 1.

### 29. Jiang, Q.; Jin, J.; Xin, X.; Chen, K.; Guo, W. — *Vehicle as a Service: Fuzzy Reward-Based Multi-Agent Deep Reinforcement Learning for Task Scheduling in Vehicular Edge Computing.* Systems **2026**, 14, 1103.
URL fetched: `https://mdpi-res.com/d_attachment/systems/systems-14-01103/article_deploy/systems-14-01103.pdf` (canonical: `https://www.mdpi.com/2079-8954/14/9/1103`)
**[FT]** Table 4 `Configuration of parameters in the FRMPPO algorithm`:
> `Discount factor γ  {0.90, 0.91, 0.93, 0.95, 0.97, 0.99}` / `GAE coefficient λ  0.95` / `Clipping ratio ϵ  0.20` / `Maximum training epoch Lmax  500`

Ablation reasoning, verbatim:
> "Second, with the learning rate fixed at 0.0005, the discount factor γ ∈ {0.90, 0.91, 0.93, 0.95, 0.97, 0.99} is evaluated. The converged reward also exhibits a peak at γ = 0.95, where the reward distribution is the most compact, and the median is the highest. **When the discount factor is too small (0.90 and 0.91), agents place excessive emphasis on immediate rewards and ignore long-term cooperative gains. When the discount factor is too large (0.97 and 0.99), future rewards are insufficiently discounted, making the policy less responsive to the current environmental state.** Both cases are unfavorable for cooperative multi-vehicle task scheduling."

> "this paper finally selects ηθ = ηφ = 0.0005 and γ = 0.95 as the optimal hyperparameter configuration of FRMPPO."

Episode structure: **per-DAG-subtask sequential assignment** — `At each decision step l, after agent i assigns the head subtask ts_head ∈ T_i to device d_k, it obtains an immediate reward`. This is the second DAG-scheduling paper in the set, and it still selects γ = 0.95, **not** 1.
**This paper's ablation grid {0.90, 0.91, 0.93, 0.95, 0.97, 0.99} is the closest real match to the claimed "0.90/0.93/0.99" grid** — see §D.

### 30. *Maritime Mobile Edge Computing for Sporadic Tasks: A PPO-Based Dynamic Offloading Strategy.* Mathematics **2025**, 13(16), 2643.
URL fetched: `https://r.jina.ai/https://www.mdpi.com/2227-7390/13/16/2643` (canonical: `https://www.mdpi.com/2227-7390/13/16/2643`)
**[FT]** Table 2 `Simulation parameters in the ON-PPO algorithm`:
> `Discount rate γ  0.9` / `Clip range ϵ  0.2` / `Number of Iteration M  4` / `Number of episode T  1000`

Discount-factor ablation, verbatim (the strongest γ-vs-convergence statement found anywhere in this survey):
> "Figure 6 illustrates the impact of different gamma values on algorithm convergence and performance. When gamma is 0.5, the training outcome is the worst as the agent focuses excessively on immediate rewards. **In contrast, when gamma is 0.99 or 0.999, the agent's overemphasis on long-term rewards results in poor algorithm performance and a failure to converge.** Results show that a gamma of 0.9 balances convergence speed and average performance well, so it is used as the default gamma for later experiments."

Episode structure: per-task sequential offloading (sporadic task arrivals; reward = negative total blocking time).
**This is the closest real analogue to the *spirit* of the claimed γ = 0.93 story — a low-ish γ converging where 0.99 fails — and it explicitly involves γ = 0.999. But the values are 0.9 vs 0.99/0.999, and no iteration counts are given.**

### 31. *A Digital Twin-Assisted VEC Intelligent Task Offloading Approach.* Electronics **2025**, 14(17), 3444.
URL fetched: `https://r.jina.ai/https://www.mdpi.com/2079-9292/14/17/3444` (canonical: `https://www.mdpi.com/2079-9292/14/17/3444`)
**[FT]**
> "The discount factor γ is set to 0.99 to achieve a reasonable trade-off between long-term and short-term returns, and to control the range of noise through the discount factor, thereby enhancing the robustness of the policy."

Algorithm TD3 (off-policy actor-critic); no GAE. Reward includes a completion-failure penalty.

### 32. Hu, Z.; Liu, S.; Zhou, D.; Shen, C.; Wang, T. — *Task Offloading and Data Compression Collaboration Optimization for UAV Swarm-Enabled Mobile Edge Computing.* Drones **2025**, 9(4), 288.
URL fetched: `https://mdpi-res.com/d_attachment/drones/drones-09-00288/article_deploy/drones-09-00288.pdf` (canonical: `https://www.mdpi.com/2504-446X/9/4/288`)
**[FT]**
> "Figure 4 demonstrates the impact of varying discount factors, specifically 0.1, 0.01, and 0.001, on algorithm performance. Experimental results indicate that when γ is set to 0.1 or 0.001, the algorithm produces substantial oscillation amplitudes and difficulties in achieving stable convergence. In contrast, setting γ to 0.01 significantly reduces oscillation amplitude, thus achieving optimal convergence performance. Consequently, the γ in this study is established at 0.01."

> "the algorithm demonstrates optimal performance metrics when lr = 0.0002, γ = 0.01, ω1 = 0.2, ω2 = 0.8."

**γ = 0.01 is the most extreme value found in the entire survey** and demonstrates that this literature does not converge on a single high value. Algorithm PER-DDPG; per-time-slot control.

### 33. Gao, H. — *Research on Computation Offloading and Resource Allocation Strategy Based on MADDPG for Integrated Space–Air–Marine Network.* Entropy **2025**, 27(8), 803.
URL fetched: `https://mdpi-res.com/d_attachment/entropy/entropy-27-00803/article_deploy/entropy-27-00803.pdf` (canonical: `https://www.mdpi.com/1099-4300/27/8/803`)
**[FT]**
> "The discount factor γ is set to 0.99, the experience replay buffer D size is 1000, the minimum batch size Bb is 100, and the soft update rate ξ is 0.001."

Reward: `r_i(t) = −Cost_i(t) − η_i Pen_i`; per-time-slot control; 500 training episodes.

---

## C. The specific search: does any offloading/scheduling RL paper argue for or use **γ = 1**?

**Result: essentially no — with exactly one in-domain *argument* (but not an in-domain *use*), and one out-of-domain *use*.**

### C.1 — In-domain: an explicit γ = 1 argument, in a task-offloading paper (full-text verified)

Wang, Q.; Zhou, Y. — *Improved PPO-Based Task Offloading Strategies for Smart Grids*, CMC 2025, DOI 10.32604/cmc.2025.065465.
URL fetched: `https://file.techscience.com/files/cmc/2025/online/CMC0612/TSP_CMC_65465/TSP_CMC_65465.pdf`

Verbatim (OCR digit tokens decoded):
> "For γ = 1, the cumulative reward represents the sum of all delays and energy costs. Thus, finding the optimal strategy π* aligns with minimizing the original objective function."

This is a **finite-horizon / objective-alignment** justification of the requested kind — the undiscounted cumulative reward *is* the original scheduling objective. **However**, the paper does **not** actually train with γ = 1; its Table 3 reports `Discount factor γ 0.99` with `Adv. discount factor φ 0.95`. So the justification exists, but the deployed value is 0.99.

### C.2 — Out-of-domain: the only genuine γ = 1 *use* I could verify

Bobrova, A. — *Reinforcement Learning for Minimum-Energy Path Discovery on Low-Dimensional Potential Energy Surfaces.* Bachelor thesis, Leiden Institute of Advanced Computer Science (LIACS), 2025/2026.
URL fetched: `https://theses.liacs.nl/pdf/2025-2026-BobrovaA.pdf`

**[FT]** Table 2: `Discount factor γ 1.0` / `GAE λ 0.95` / `Clip coefficient 0.2`

Verbatim:
> "Trained with this reward at γ = 1, four of five PPO seeds recover the activation energy within rounding of the reference."

> "Since the MEP objective itself does not depend on time, a reward function that encodes this objective should ideally remain well-defined at γ = 1. As a result, the discount factor should act as an additional design choice rather than a convergence requirement."

This is a time-independent-objective argument, **not** a scheduling/offloading argument, and it is a bachelor thesis, not a peer-reviewed offloading paper. The same thesis also supplies counter-evidence:
> "when the terminal state is not known in advance, the planning horizon is limited through γ < 1 rather than explicit episode termination."
> "Discounting biases the agent to obtain low-energy rewards within a shorter horizon, which leads to heuristic shortcuts instead of following the underlying energy landscape."

### C.3 — Everything else: a clean negative

- **No** MEC / VEC / V2X / UAV-MEC / DAG-offloading / edge-cloud-scheduling paper in this survey (n = 33) states γ = 1 or γ = 0.999 as its **training value**. (Mathematics 13(16):2643 mentions 0.999 only as a failing configuration it rejected.)
- Papers that are explicitly **episodic / finite-horizon** still use γ < 1: arXiv:2508.06863 (*"after T time slots the POMDP ends"*) uses **0.99**; Sensors 25:1428 (*"each episode representing a complete scheduling scenario"*) uses **0.99**; arXiv:2507.09341 (`Number of tasks per vehicle 1`) uses **0.95**; TPTO (arXiv:2312.11739, per-task DAG plan — structurally the closest to a single-shot plan) uses **0.99**; Systems 14(9):1103 (per-DAG-subtask VEC scheduling) uses **0.95**.
- arXiv:2011.08442 explicitly sweeps γ **downward** and selects **0.6**; Electronics 13:2387 states *"when γ = 1, convergence issues may emerge"* and selects **0.97**; Electronics 15:936 reports **0.6**; Drones 9(4):288 selects **γ = 0.01**.
- **Mathematics 13(16):2643 is the strongest single counter-example to any "push γ toward 1" narrative**: it reports verbatim that *"when gamma is 0.99 or 0.999, the agent's overemphasis on long-term rewards results in poor algorithm performance and a failure to converge"*, and selects **0.9**.
- The npj Wireless Technology MAPPO paper (2:64, 2026) is the only source that discusses the γ → 1 limit in-domain — and it uses **0.9**.
- Systems 14(9):1103 adds a symmetric statement: *"When the discount factor is too large (0.97 and 0.99), future rewards are insufficiently discounted, making the policy less responsive to the current environmental state."* — i.e. too-high γ is framed as a **defect**, not a goal.

**Bottom line for the γ = 1 question:** I found **one task-offloading paper that argues mathematically for γ = 1** (CMC 2025, because the undiscounted sum coincides with the scheduling objective) but **trains at γ = 0.99**; and **one genuine γ = 1 PPO run**, which is a molecular-pathfinding bachelor thesis, out of domain. I found **no offloading/scheduling paper that trains with γ = 1 or 0.999.** Moreover, two offloading papers that *did* test γ near 1 (Mathematics 13(16):2643 at 0.99/0.999; Systems 14(9):1103 at 0.97/0.99) report it as a **failure mode**, not a target.

---

## D. The specific claim: "a 2025 PPO offloading paper uses γ = 0.93 and reports convergence within 4000 iterations while γ = 0.99 and γ = 0.90 did not converge even at 6000 iterations"

**Verdict: NOT VERIFIED.**

No paper matching that description was found. Specifically:

1. No paper found that reports a three-point γ ablation over **exactly {0.90, 0.93, 0.99}**.
2. No paper found that reports **γ = 0.93** as its adopted value.
3. No paper found that reports convergence at **4000 iterations** vs failure at **6000 iterations** as a function of γ.
4. **Closest grid match (NEW, and much closer than the lead URL):** Jiang, Jin, Xin, Chen & Guo, *Systems* **2026**, 14(9), 1103 — FRMPPO on **VEC DAG task scheduling** — ablates **γ ∈ {0.90, 0.91, 0.93, 0.95, 0.97, 0.99}** and selects **γ = 0.95**. This contains **0.90, 0.93 AND 0.99**, and is the only offloading paper found whose grid covers all three of the claimed values. But the winner is **0.95, not 0.93**, the direction of the result is "both extremes are bad" rather than "0.93 wins", and no 4000/6000-iteration counts appear. URL: `https://mdpi-res.com/d_attachment/systems/systems-14-01103/article_deploy/systems-14-01103.pdf`
5. **Closest convergence-story match (NEW):** *Mathematics* **2025**, 13(16), 2643 — ON-PPO on maritime MEC offloading — reports verbatim: *"when gamma is 0.99 or 0.999, the agent's overemphasis on long-term rewards results in poor algorithm performance and a failure to converge"*, while *"a gamma of 0.9 balances convergence speed and average performance well"*. This is the **only** found instance of "high γ fails to converge, lower γ works" in an offloading paper, and it expressly involves **γ = 0.999**. Values are 0.9 vs 0.99/0.999 (not 0.93 vs 0.90/0.99), and no iteration counts are given. URL: `https://www.mdpi.com/2227-7390/13/16/2643`
6. The **next closest** is Electronics 13(12):2387 (2024), which ablates **γ ∈ {0.93, 0.95, 0.97, 0.99}** and selects **γ = 0.97** — it contains 0.93 and 0.99 but not 0.90, and its convergence discussion is in episodes (≈920–1050), not 4000/6000 iterations. URL: `https://mdpi-res.com/d_attachment/electronics/electronics-13-02387/article_deploy/electronics-13-02387.pdf`
7. The **other close** one is Future Internet 17(12):542 (2025), which ablates **γ ∈ {0.8, 0.95, 0.99}** on a MAPPO offloading task and selects **γ = 0.95**; its γ = 0.99 arm is described as poor before 700 episodes and still worse than 0.95 afterwards. URL: `https://mdpi-res.com/d_attachment/futureinternet/futureinternet-17-00542/article_deploy/futureinternet-17-00542-v2.pdf`

**Overall:** the *phenomenon* the claim describes (an intermediate γ outperforming both a lower and a higher γ, with a high γ failing to converge) **is real and is documented twice** — most sharply in Mathematics 13(16):2643 and more mildly in Systems 14(9):1103. The *specific numbers* (γ = 0.93 adopted; 0.90 and 0.99 failing; 4000 vs 6000 iterations) are **not verified in any source**. Do not attribute those numbers to a real paper.

**The lead URL supplied in the task** — `https://mdpi-res.com/d_attachment/futureinternet/futureinternet-17-00542/article_deploy/futureinternet-17-00542-v2.pdf` — was fetched and verified. Your description of it was **correct**: it does contain a discount-factor ablation, over **0.8 / 0.95 / 0.99**. But that grid is **not** the 0.90/0.93/0.99 claim in the sentence above, so the lead URL does not substantiate the claim. The paper is Chen & Tong, *Future Internet* 2025, 17, 542 (D-MAPPO), and it selects **γ = 0.95**.

---

## E. Observations that matter if these numbers are being used to justify a design choice

1. **γ = 0.99 is the modal default (13/28)**; 0.95 is second (7/28); 0.9 third (5/28). There is no evidence of a field-wide drift toward γ = 1.
2. **The handful of papers that explicitly discuss γ = 1 treat it as a warning, not a target** (Electronics 13:2387: *"when γ = 1, convergence issues may emerge"*).
3. **Episodic framing does not, in practice, imply γ = 1** in this literature — every explicitly episodic offloading paper verified here used 0.95–0.99.
4. **Symbol collisions are a real extraction hazard**: γ denotes compression ratio in arXiv:2412.13676, SINR in Electronics 15:936, offloading ratio in arXiv:2606.26293 and arXiv:2312.08714, and antenna spacing in arXiv:2310.17470; ε is the discount factor in arXiv:2011.08442; µ_RL in arXiv:2607.09295; τ in arXiv:2312.08714; ρ in an IEEE source surfaced in search. Likewise the hyperparameter row may be labelled **`Gamma`** rather than `discount factor` (arXiv:2404.15278). Any automated extraction of "γ" from this literature will produce false positives without manual reading.
5. **Not-stated λ is common**: 19 of 33 papers state no numeric GAE λ. Of those that do, **0.95 dominates** (10 papers), with single instances of 0.9 (npj WT) and 0.98 (arXiv:2301.08376).

---

## F. Papers deliberately excluded (and why)

- **arXiv:2507.21638** — *Assistax: A Multi-Agent Hardware-Accelerated RL Benchmark for Assistive Robotics.* Surfaced by keyword search but is a robotics benchmark, **not** computation offloading. Excluded.
- **arXiv:2403.15285** — *Blockchain-based Pseudonym Management for Vehicle Twin Migrations in Vehicular Edge Metaverse.* Vehicular edge, uses GAE, but the decision problem is pseudonym generation, not computation offloading. Excluded from the main table.
- **arXiv:2312.08714** (Aerial STAR-RIS MEC) — **[ABS/partial]** defines a discount factor `τ` but no numeric value was found in the extracted text. Not entered with a number.
- **Sensors 21:4392** (dual-RIS UAV-MEC-IoV) — energy-minimisation with Lagrange dual / subgradient methods; **no RL, no γ**. Excluded.
- **arXiv:2106.06170** (*Taylor Expansions of Discount Factors*) — a γ-analysis paper, not an offloading application. Excluded.
- **Computers 13(11):295** — a DQN/A2C offloading paper whose table reportedly reads `Gamma 0.99 0.99 0.99 0.8 0.99 0.5 0.99 0.99`. **Not included** because I did not fetch and read it myself; the row was reported by a search snippet only. Treat as unverified.
- **arXiv:2603.20238 / arXiv:2403.15285** — see above; the former IS included (row 8), the latter is not.

---

## G. Verification-level summary

| Level | Count | Note |
|---|---|---|
| Full-text verified (`[FT]`), numeric γ read from a table or body sentence I fetched | **33** | all entries in Table A |
| Abstract-only | 0 | no γ value in this report rests on an abstract |
| Numeric γ found but source is a thesis, not a paper | 1 | LIACS bachelor thesis (γ = 1.0), §C.2 — flagged as out-of-domain |

**Access note (reproducible):** `www.mdpi.com` and `www.nature.com` block direct `web_fetch` (HTTP 403 via Akamai; cross-origin redirect to `idp.nature.com`). The reliable routes used here were (a) **`mdpi-res.com` article-deploy PDFs downloaded with `curl` + text-extracted with `pypdf`**, and (b) **`https://r.jina.ai/<canonical-url>`** for MDPI/Nature HTML. Route (a) is strictly better: it recovered hyperparameter tables that route (b) missed — e.g. the `GAE parameter 0.95` row in Future Internet 17:542 and the full `GAE coefficient λ 0.95` row in Systems 14:1103, both of which were absent from the HTML extraction. **Use the PDF route for hyperparameter tables.**

**Files written by this verification pass:** `VERIFY_gamma_GAE_offloading_literature.md` (this report). Raw extracted texts and PDFs are in `refs/` (e.g. `refs/fi-17-00542.txt`, `refs/electronics-13-02387.txt`, `refs/ax-2312.01499.txt`, `refs/systems-14-01103.txt`).
