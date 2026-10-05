### S1. trajectory sample

| itr | objective | latency | reward | requester J | mobile J | system J | lam_te | viol_te | penalty | local | mec | v2v | approx_kl | grad_norm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.53 | 0.32 | 0.599 | 0.191 | 2.52 | 63.9 | 0 | 0.00277 | 0 | 0.129 | 0.436 | 0.435 | 1.29 | 16.4 |
| 1 | 0.332 | 0.499 | 0.376 | 0.49 | 0.593 | 86.6 | 0 | 0.0543 | 0 | 0.836 | 0.045 | 0.119 | 0.483 | 5.11 |
| 2 | 0.194 | 0.537 | 0.221 | 0.541 | 0.813 | 85.2 | 0 | 0.00759 | 0 | 0.9 | 0.0423 | 0.0578 | 0.155 | 4.29 |
| 5 | 0.214 | 0.469 | 0.244 | 0.481 | 0.687 | 72.1 | 0 | 0.0107 | 0 | 0.871 | 0.0948 | 0.0343 | 0.144 | 2.96 |
| 10 | 0.573 | 0.221 | 0.653 | 0.22 | 0.766 | 71.9 | 0.000678 | 0.098 | 0 | 0.546 | 0.349 | 0.104 | 0.125 | 2.35 |
| 25 | 1.39 | 0.161 | 1.58 | 0.113 | 0.229 | 98.3 | 0.117 | 0.341 | 0.0341 | 0.405 | 0.562 | 0.0323 | 0.353 | 7.83 |
| 50 | 0.205 | 0.477 | 0.234 | 0.481 | 0.702 | 79 | 0.264 | 0.0325 | 0.00895 | 0.481 | 0.492 | 0.0265 | 0.256 | 2.46 |
| 75 | 1.45 | 0.22 | 1.64 | 0.216 | 0.451 | 109 | 0.455 | 0.232 | 0.103 | 0.459 | 0.485 | 0.0558 | 0.105 | 2.64 |
| 100 | 0.726 | 0.241 | 0.818 | 0.24 | 0.539 | 72.5 | 0.575 | 0.0929 | 0.0533 | 0.497 | 0.433 | 0.0698 | 0.0608 | 1.91 |
| 150 | 1.17 | 0.186 | 1.33 | 0.183 | 0.704 | 108 | 0.851 | 0.183 | 0.154 | 0.454 | 0.456 | 0.0905 | 0.109 | 1.31 |
| 200 | 0.705 | 0.187 | 0.8 | 0.207 | 0.483 | 83.2 | 1.13 | 0.104 | 0.116 | 0.525 | 0.423 | 0.0517 | 0.0218 | 0.656 |
| 250 | 0.597 | 0.246 | 0.675 | 0.256 | 1.25 | 74.7 | 1.23 | 0.0432 | 0.0533 | 0.497 | 0.343 | 0.16 | 0.0649 | 0.935 |
| 300 | 0.655 | 0.326 | 0.744 | 0.349 | 1.13 | 84.9 | 1.18 | 0.0331 | 0.0393 | 0.569 | 0.299 | 0.132 | 0.0503 | 1.32 |
| 350 | 0.569 | 0.261 | 0.643 | 0.249 | 1.36 | 69.8 | 1.06 | 0.0248 | 0.0262 | 0.462 | 0.316 | 0.223 | 0.0216 | 0.777 |
| 400 | 0.835 | 0.178 | 0.944 | 0.13 | 1.46 | 88.1 | 1.08 | 0.112 | 0.121 | 0.339 | 0.378 | 0.283 | 0.0166 | 0.721 |
| 450 | 1.11 | 0.199 | 1.25 | 0.171 | 1.47 | 91.2 | 1.19 | 0.0962 | 0.114 | 0.398 | 0.36 | 0.242 | 0.037 | 0.702 |
| 475 | 0.907 | 0.214 | 1.03 | 0.158 | 1.54 | 83.2 | 1.24 | 0.0854 | 0.106 | 0.336 | 0.349 | 0.314 | 0.0252 | 0.815 |
| 499 | 0.76 | 0.206 | 0.856 | 0.198 | 1.22 | 89.4 | 1.31 | 0.101 | 0.132 | 0.446 | 0.352 | 0.201 | 0.0372 | 0.79 |

### S2. validation cadence table

| itr | firm_miss | high_miss | ghard | ghigh | tardy_H | tardy_M | q_latency | k0_regret | k3_regret | paired_k3 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.22 | 0.374 | 0.405 | 0.825 | 150 | 0 | 0.267 | 0.243 | 0.239 | -0.00481 |
| 50 | 0.0235 | 0.04 | 0 | 0.14 | 16 | 0 | 0.0883 | 0.0524 | 0.0598 | +0.00739 |
| 100 | 0.0247 | 0.042 | 0 | 0.145 | 16.8 | 0 | 0.0894 | 0.0574 | 0.0609 | +0.00356 |
| 150 | 0.0182 | 0.031 | 0 | 0.105 | 12.4 | 0 | 0.084 | 0.0499 | 0.0556 | +0.0057 |
| 200 | 0.0285 | 0.0485 | 0 | 0.165 | 19.4 | 0 | 0.0922 | 0.06 | 0.0637 | +0.00376 |
| 250 | 0.0324 | 0.055 | 0 | 0.175 | 22 | 0 | 0.0966 | 0.0573 | 0.0682 | +0.0108 |
| 300 | 0.0447 | 0.076 | 0 | 0.25 | 30.4 | 0 | 0.101 | 0.073 | 0.0726 | -0.000436 |
| 350 | 0.0112 | 0.019 | 0 | 0.09 | 7.6 | 0 | 0.0863 | 0.0626 | 0.0578 | -0.0048 |
| 400 | 0.0256 | 0.0425 | 0 | 0.14 | 17 | 0.4 | 0.0844 | 0.0546 | 0.0559 | +0.00134 |
| 450 | 0.0312 | 0.0525 | 0 | 0.15 | 21 | 0.2 | 0.0895 | 0.0563 | 0.061 | +0.00472 |

### S3. objective | first | last | min | max | mean | quartiles | slope

| metric | first | last | min | max | mean | q1 | q2 | q3 | q4 | slope/iter |
|---|---|---|---|---|---|---|---|---|---|---|
| `training/objective_d` | 0.53 | 0.76 | -0.011 | 1.79 | 0.7982 | 0.8096 | 0.8147 | 0.7224 | 0.846 | +2.41e-05 |
| `training/objective_l` | 0.599 | 0.856 | -0.0173 | 2.04 | 0.9034 | 0.9208 | 0.9217 | 0.8163 | 0.9546 | +9.08e-06 |
| `Average reward` | 0.599 | 0.856 | -0.0173 | 2.04 | 0.9034 | 0.9208 | 0.9217 | 0.8163 | 0.9546 | +9.08e-06 |
| `Average latency` | 0.32 | 0.206 | 0.154 | 0.537 | 0.2206 | 0.2384 | 0.1953 | 0.2441 | 0.2047 | -5.17e-05 |
| `Average greedy latency` | 0.0561 | 0.0743 | 0.0305 | 0.0915 | 0.06085 | 0.06116 | 0.05994 | 0.06029 | 0.062 | +3.24e-06 |
| `MaxReturn` | 2.7 | 1.91 | 0.0855 | 4.93 | 2.884 | 2.989 | 2.894 | 2.858 | 2.794 | -0.000382 |
| `MinReturn` | -0.188 | -0.01 | -0.428 | 0.0456 | -0.08035 | -0.0885 | -0.1183 | -0.05927 | -0.05536 | +0.000123 |
| `StdReturn` | 0.854 | 0.723 | 0.0458 | 1.6 | 0.9383 | 0.961 | 0.9561 | 0.8899 | 0.9464 | -5.38e-05 |

### S4. energy

| metric | first | last | min | max | mean | q1 | q2 | q3 | q4 | slope/iter |
|---|---|---|---|---|---|---|---|---|---|---|
| `energy/requester_joules` | 0.191 | 0.198 | 0.0914 | 0.541 | 0.2123 | 0.2324 | 0.2059 | 0.2373 | 0.1734 | -0.000118 |
| `energy/mobile_joules` | 2.52 | 1.22 | 0.216 | 2.52 | 0.996 | 0.5783 | 0.6622 | 1.299 | 1.444 | +0.00249 |
| `energy/system_joules` | 63.9 | 89.4 | 42.9 | 133 | 82.74 | 85.33 | 83.18 | 79.29 | 83.17 | -0.00836 |
| `Average energy` | 1280 | 1790 | 857 | 2660 | 1655 | 1707 | 1664 | 1585 | 1664 | -0.167 |

### S5. constraint / dual

| metric | first | last | min | max | mean | q1 | q2 | q3 | q4 | slope/iter |
|---|---|---|---|---|---|---|---|---|---|---|
| `constraint/lambda_to` | 0 | 1.31 | 0 | 1.31 | 0.9251 | 0.3504 | 1.03 | 1.155 | 1.164 | +0.00209 |
| `constraint/lambda_ue` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | +0 |
| `constraint/penalty` | 0 | 0.132 | 0 | 0.199 | 0.07876 | 0.04906 | 0.1098 | 0.06178 | 0.09442 | +7.62e-05 |
| `constraint/updates` | 1 | 500 | 1 | 500 | 250.5 | 63 | 188 | 313 | 438 | +1 |
| `constraint/total_ene` | 0.00277 | 0.101 | 0.00277 | 0.36 | 0.09728 | 0.1444 | 0.1104 | 0.0532 | 0.08113 | -0.000196 |
| `constraint/ue_energy` | 0.00719 | 0 | 0 | 0.00719 | 0.0001745 | 0.0005143 | 2.627e-05 | 0.00014 | 1.763e-05 | -1.21e-06 |
| `constraint/mean_sign` | -0.399 | -0.537 | -0.771 | -0.108 | -0.4945 | -0.4786 | -0.492 | -0.4408 | -0.5664 | -0.00016 |

### S6. PPO

| metric | first | last | min | max | mean | q1 | q2 | q3 | q4 | slope/iter |
|---|---|---|---|---|---|---|---|---|---|---|
| `policy/approx_kl` | 1.29 | 0.0372 | 0.00719 | 1.29 | 0.06052 | 0.1161 | 0.03956 | 0.05254 | 0.03392 | -0.000204 |
| `policy/clip_fraction` | 0.658 | 0.0925 | 0.0563 | 0.658 | 0.1254 | 0.1799 | 0.1076 | 0.1196 | 0.09443 | -0.000212 |
| `policy/entropy_valid` | 0.954 | 0.262 | 0.119 | 0.954 | 0.3406 | 0.3668 | 0.3176 | 0.3777 | 0.3002 | -0.000119 |
| `policy/grad_norm` | 16.4 | 0.79 | 0.382 | 16.4 | 1.428 | 2.841 | 1.129 | 0.9124 | 0.8315 | -0.00545 |
| `policy/policy_loss_mean` | 0.377 | 0.0126 | -0.0141 | 0.377 | 0.01277 | 0.02729 | 0.005743 | 0.009777 | 0.008271 | -4.81e-05 |
| `policy/value_loss_mean` | 2.13 | 0.0652 | 0.0115 | 2.13 | 0.1014 | 0.1903 | 0.08326 | 0.06586 | 0.06635 | -0.000338 |
| `critic/value_abs_max` | 2.47 | 2.36 | 0.171 | 3.09 | 2.084 | 2.165 | 2.14 | 1.941 | 2.091 | -0.000284 |

### S7. action mix

| metric | first | last | min | max | mean | q1 | q2 | q3 | q4 | slope/iter |
|---|---|---|---|---|---|---|---|---|---|---|
| `action_fraction/local` | 0.129 | 0.446 | 0.129 | 0.9 | 0.4639 | 0.4778 | 0.4929 | 0.4851 | 0.3996 | -0.000196 |
| `action_fraction/mec` | 0.436 | 0.352 | 0.0423 | 0.627 | 0.3839 | 0.4562 | 0.4235 | 0.316 | 0.34 | -0.000347 |
| `action_fraction/v2v` | 0.435 | 0.201 | 0.0168 | 0.435 | 0.1522 | 0.06585 | 0.08357 | 0.199 | 0.2603 | +0.000542 |

### S8. throughput

| metric | first | last | min | max | mean | q1 | q2 | q3 | q4 | slope/iter |
|---|---|---|---|---|---|---|---|---|---|---|
| `EnvExecTime` | 8.98 | 8.57 | 7.76 | 10.6 | 8.73 | 8.702 | 8.66 | 8.719 | 8.838 | +0.000461 |
| `PolicyExecTime` | 0.385 | 0.666 | 0.229 | 1.26 | 0.379 | 0.4231 | 0.3653 | 0.3494 | 0.3783 | -0.000119 |

### S9. correlations with the optimised objective (Pearson over 500 iterations)

| metric | r |
|---|---|| `training/objective_l` | +1.000 |
| `Average reward` | +1.000 |
| `energy/system_joules` | +0.874 |
| `Average energy` | +0.874 |
| `constraint/mean_sign` | -0.750 |
| `StdReturn` | +0.659 |
| `constraint/total_ene` | +0.602 |
| `MaxReturn` | +0.581 |
| `constraint/penalty` | +0.578 |
| `critic/value_abs_max` | +0.358 |
| `energy/requester_joules` | -0.303 |
| `Average latency` | -0.268 |
| `EnvExecTime` | -0.266 |
| `MinReturn` | +0.230 |
| `policy/value_loss_mean` | +0.214 |
| `constraint/ue_energy` | -0.208 |
| `policy/entropy_valid` | -0.208 |
| `policy/clip_fraction` | -0.206 |

### S10. first-quarter -> last-quarter change (sorted by |%|)

| metric | q1 mean | q4 mean | change |
|---|---|---|---|| `constraint/updates` | 63 | 438 | +595.2% |
| `action_fraction/v2v` | 0.065846 | 0.26032 | +295.3% |
| `constraint/lambda_to` | 0.35042 | 1.1642 | +232.2% |
| `energy/mobile_joules` | 0.57832 | 1.4444 | +149.7% |
| `constraint/ue_energy` | 0.00051428 | 1.763e-05 | -96.6% |
| `constraint/penalty` | 0.049063 | 0.094419 | +92.4% |
| `policy/approx_kl` | 0.11607 | 0.033921 | -70.8% |
| `policy/grad_norm` | 2.8409 | 0.8315 | -70.7% |
| `policy/policy_loss_mean` | 0.027291 | 0.0082708 | -69.7% |
| `policy/value_loss_mean` | 0.19025 | 0.066351 | -65.1% |
| `policy/clip_fraction` | 0.17995 | 0.094427 | -47.5% |
| `constraint/total_ene` | 0.14442 | 0.081126 | -43.8% |
| `MinReturn` | -0.088505 | -0.055362 | +37.4% |
| `action_fraction/mec` | 0.45621 | 0.33998 | -25.5% |
| `energy/requester_joules` | 0.23244 | 0.17344 | -25.4% |
| `constraint/mean_sign` | -0.47863 | -0.56638 | -18.3% |
| `policy/entropy_valid` | 0.36685 | 0.30021 | -18.2% |
| `action_fraction/local` | 0.47784 | 0.3996 | -16.4% |
| `Average latency` | 0.23845 | 0.20467 | -14.2% |
| `PolicyExecTime` | 0.42314 | 0.37828 | -10.6% |
| `MaxReturn` | 2.9893 | 2.7941 | -6.5% |
| `training/objective_d` | 0.80959 | 0.84595 | +4.5% |
| `training/objective_l` | 0.92079 | 0.95463 | +3.7% |
| `Average reward` | 0.92079 | 0.95463 | +3.7% |
| `critic/value_abs_max` | 2.1648 | 2.091 | -3.4% |
| `energy/system_joules` | 85.327 | 83.169 | -2.5% |

### S11. run.json

```json
{
  "evidence": {
    "checkpoint": {
      "path": "/out/ckpt/v2_run.ckpt",
      "promotable": true,
      "restored_from": null,
      "resume_state_path": "/out/ckpt/v2_resume_state.json",
      "status": "completed_iterations=500",
      "was_resume": false
    },
    "config": {
      "background_dags": 2,
      "background_policy": "all_mec",
      "budget_fractions": {
        "total_energy": 0.5,
        "ue_energy": 1.0
      },
      "constraints": true,
      "crn": {
        "protocol_id": "automotive_crn_gumbel_v1",
        "r_select": 2,
        "s_select": 2
      },
      "device": "gpu",
      "gpu_allow_growth": false,
      "gpu_fraction": 0.03,
      "inter_op": 2,
      "intra_op": 8,
      "iterations": 500,
      "link_regime": "moderate",
      "mec_workers": 1,
      "obs_version": "automotive_v2_obs_v1",
      "r_select": 2,
      "reliability": false,
      "reward_mode": "latency_only",
      "s_select": 2,
      "seed": 0,
      "v2_constraints": {
        "budget_fractions": {
          "total_energy": 0.5,
          "ue_energy": 1.0
        },
        "enabled": true,
        "lambdas": {
          "total_energy": 0.0,
          "ue_energy": 0.0
        },
        "names": [
          "ue_energy",
          "total_energy"
        ],
        "objectives": {
          "primary": "constrained_latency",
          "weighted_ablation": "w_T*T/T_ref + w_E*E/E_ref"
        },
        "spec_sha256": "e71942341c22a8b85ab5ade96a3d0c0d4eeebb06c2559e59f400325a5b3a6b43"
      },
      "v2_system": {
        "background_dags": 2,
        "background_policy": "all_mec",
        "budget_fractions": {
          "total_energy": 0.5,
          "ue_energy": 1.0
        },
        "constraints_enabled": true,
        "helper_busy_s": 0.0,
        "helper_contact_cv": 0.5,
        "helper_contact_mean_s": 2.0,
        "link_regime": "moderate",
        "mec_workers": 1,
        "reliability": false
      }
    },
    "constraints": {
      "lambdas": {
        "total_energy": 1.305534390180372,
        "ue_energy": 0.0
      },
      "names": [
        "ue_energy",
        "total_energy"
      ],
      "spec_sha256": "e71942341c22a8b85ab5ade96a3d0c0d4eeebb06c2559e59f400325a5b3a6b43",
      "status": {
        "deadline": "not_configured",
        "helper_energy": "not_configured",
        "mec_task_fraction": "not_configured",
        "total_energy": "active",
        "ue_energy": "active",
        "v2v_airtime": "not_configured",
        "v2v_task_fraction": "not_configured"
      },
      "updates": 500
    },
    "device": {
      "cuda_visible_devices": "<unset>",
      "requested": "gpu"
    },
    "error": null,
    "iteration_measurements": {
      "mean_requester_joules": 0.1978596539478312,
      "mean_system_joules": 89.4277885011633,
      "mean_violation": 0.10121981881050089
    },
    "iterations_completed": 500,
    "iterations_executed": 500,
    "iterations_requested": 500,
    "iterations_started": 500,
    "lambda_broadcast": {
      "targets": 11,
      "values": {
        "total_energy": 1.305534390180372,
        "ue_energy": 0.0
      }
    },
    "last_completed_iteration": 499,
    "last_rollout": {
      "background_joules": 33.443497318233305,
      "constraint_penalty": 0.0,
      "constraint_signed": {
        "total_energy": -0.15405366556305,
        "ue_energy": -0.23787168947006926
      },
      "constraint_violations": {
        "total_energy": 0.0,
        "ue_energy": 0.0
      },
      "energy_constraint": "configured",
      "makespan_s": 0.1850290318755861,
      "mobile_joules": 1.03673487845024,
      "n_helper_tasks": 6,
      "n_high_tasks": 0,
      "n_mec_tasks": 5,
      "n_medium_tasks": 0,
      "n_tasks": 20,
      "requester_joules": 0.19855595207929927,
      "system_joules": 39.1379796736324,
      "unmodeled_components": [
        "receiver/static radio power (include_rx_energy=False in the frozen config)",
        "CPU idle/leakage power between tasks",
        "M
```
