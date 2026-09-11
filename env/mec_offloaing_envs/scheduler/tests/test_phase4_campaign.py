#!/usr/bin/env python3
"""Phase 4 campaign-contract tests. Numpy/stdlib only. No GPU. No training."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spec.phase4_campaign import (
    DIAG200_ITERS,
    DIAG200_METHOD_ID,
    DIAG500_ITERS,
    DIAG500_METHOD_ID,
    DIAG_ITERS,
    DIAG_LAT_ITERS,
    DIAG_LAT_METHOD_ID,
    DIAG_METHOD_ID,
    DIAG_POMO_ITERS,
    DIAG_POMO_METHOD_ID,
    DIAG_BC_ITERS,
    DIAG_BC_METHOD_ID,
    DIAG_BCONLY_EPOCHS,
    DIAG_BCONLY_METHOD_ID,
    DIAG_BCCONT_MAX_EPOCHS,
    DIAG_BCCONT_METHOD_ID,
    DIAG_KLPPO_ITERS,
    DIAG_KLPPO_KL_COEF,
    DIAG_KLPPO_METHOD_ID,
    DIAG_KLPPO_WARMUP_ITERS,
    DIAG_BCUNSEEN_METHOD_ID,
    DIAG_BCFEW_METHOD_ID,
    DIAG_BCSS_METHOD_ID,
    DIAG_BCSS_EPOCHS,
    DIAG_H2_METHOD_ID,
    DIAG_H2_N_GRAPHS,
    DIAG_MOTIF_METHOD_ID,
    DIAG_MOTIF_N_TRAIN,
    DIAG_TWOPT_METHOD_ID,
    DIAG_TWOPT_MAX_H2_ROUNDS,
    DIAG_BC2OPT_METHOD_ID,
    DIAG_PAIR_METHOD_ID,
    DIAG_PAIR_TOP_K,
    DIAG_PAIR_ORACLE_N_VAL,
    DIAG_PAIR_KSWEEP_METHOD_ID,
    DIAG_PAIR_K_SWEEP,
    DIAG_PAIR_RANKER_METHOD_ID,
    DIAG_PAIR_RANKER_K,
    DIAG_PAIR_SEQ_METHOD_ID,
    DIAG_PAIR_SEQ_BESTIMP_K,
    DIAG_CAVIA_METHOD_ID,
    DIAG_CAVIA_ENERGY_METHOD_ID,
    DIAG_CAVIA_STRONG_METHOD_ID,
    DIAG_CAVIA_Z_DIM,
    DIAG_CAVIA_INNER_STEPS,
    DIAG_CAVIA_INNER_LR,
    DIAG_CAVIA_STRONG_INNER_STEPS,
    DIAG_CAVIA_STRONG_INNER_LR,
    DIAG_PAIRSUP_METHOD_ID,
    DIAG_PAIRSUP_EPOCHS,
    DIAG_PAIRSUP_LAMBDA,
    DIAG_PAIRSUP_A_VAL_T,
    DIAG_PAIRSUP_A_TEST_T,
    DIAG_CAVIA_BCCONT_METHOD_ID,
    DIAG_CAVIA_BCCONT_IDENTITY_T_VAL_LO,
    DIAG_CAVIA_BCCONT_IDENTITY_T_VAL_HI,
    DIAG_PAIRFRAC_METHOD_ID,
    DIAG_PAIRFRAC_N_GRAPHS,
    DIAG_REWRITE_METHOD_ID,
    DIAG_REWRITE_EPOCHS,
    DIAG_REWRITE_LAMBDA,
    DIAG_REWRITE_K,
    DIAG_REWRITE_A_VAL_T,
    DIAG_REWRITE_A_TEST_T,
    DIAG_ORACLE_DIST_METHOD_ID,
    DIAG_ORACLE_DIST_EPOCHS,
    DIAG_ORACLE_DIST_Z_DIM,
    DIAG_ORACLE_DIST_N_DIST,
    DIAG_ORACLE_DIST_A_VAL_T,
    DIAG_ORACLE_DIST_A_TEST_T,
    DIAG_ORACLE_DIST_TRAIN_REF_T,
    DIAG_ORACLE_DIST_IDENTITY_T_VAL_LO,
    DIAG_ORACLE_DIST_IDENTITY_T_VAL_HI,
    DIAG_BINARY_LAT_METHOD_ID,
    DIAG_BINARY_LAT_ITERS,
    DIAG_BINARY_LAT_VOCAB,
    DIAG_ENC_METHOD_ID,
    DIAG_BOK_METHOD_ID,
    DIAG_EAS_METHOD_ID,
    DIAG_EAS_INST_METHOD_ID,
    GPU_ENV,
    GPU_FLAG,
    K_REPORT,
    META_TEST_IDS,
    META_TRAIN_IDS,
    OUTER_ITERS,
    PARENT_SHA,
    PARALLEL_PROBE_METHOD_ID,
    PROBE_ITERS,
    PROBE_METHOD_ID,
    SEEDS,
    VALIDATION_IDS,
    GpuPermissionError,
    campaign_jobs,
    diag200_run_dir,
    diag500_run_dir,
    diag_lat_run_dir,
    diag_pomo_run_dir,
    diag_bc_run_dir,
    diag_bconly_run_dir,
    diag_bccont_run_dir,
    diag_klppo_run_dir,
    diag_bcunseen_run_dir,
    diag_bcfew_run_dir,
    diag_bcss_run_dir,
    diag_h2_run_dir,
    diag_motif_run_dir,
    diag_twopt_run_dir,
    diag_bc2opt_run_dir,
    diag_pair_run_dir,
    diag_pair_ksweep_run_dir,
    diag_pair_ranker_run_dir,
    diag_pair_seq_run_dir,
    diag_cavia_run_dir,
    diag_cavia_energy_run_dir,
    diag_cavia_strong_run_dir,
    diag_pairsup_run_dir,
    diag_cavia_bccont_run_dir,
    diag_pairfrac_run_dir,
    diag_rewrite_run_dir,
    diag_oracle_dist_run_dir,
    diag_binary_lat_run_dir,
    diag_encoder_run_dir,
    diag_bestofk_run_dir,
    diag_eas_run_dir,
    diag_eas_inst_run_dir,
    diag_expert_profiles_run_dir,
    diag_bc_profiles_run_dir,
    diag_eas_profiles_run_dir,
    diag_ctx_profiles_run_dir,
    diag_bok_profiles_run_dir,
    diag_expert_energy_run_dir,
    diag_bc_energy_run_dir,
    diag_eas_energy_run_dir,
    diag_run_dir,
    eval_json_path,
    parallel_probe_run_dir,
    probe_run_dir,
    provenance_template,
    require_gpu_permission,
    seed_run_dir,
    write_plan,
)
from spec.split_loader import (
    meta_test_distribution_ids,
    meta_train_distribution_ids,
    validation_distribution_ids,
)


class TestCampaignFreeze(unittest.TestCase):
    def test_seeds_and_budget(self):
        self.assertEqual(SEEDS, (0, 1, 2, 3, 4))
        self.assertEqual(OUTER_ITERS, 3500)
        self.assertEqual(PROBE_ITERS, 5)
        self.assertEqual(DIAG_ITERS, 1000)
        self.assertEqual(DIAG200_ITERS, 200)
        self.assertEqual(DIAG500_ITERS, 500)
        self.assertEqual(DIAG_LAT_ITERS, 50)
        self.assertEqual(DIAG_POMO_ITERS, 50)
        self.assertEqual(DIAG_BC_ITERS, 50)
        self.assertEqual(DIAG_BCONLY_EPOCHS, 40)
        self.assertEqual(DIAG_BCCONT_MAX_EPOCHS, 80)
        self.assertEqual(DIAG_KLPPO_ITERS, 50)
        self.assertEqual(DIAG_KLPPO_WARMUP_ITERS, 5)
        self.assertEqual(DIAG_KLPPO_KL_COEF, 0.1)
        self.assertNotEqual(PROBE_METHOD_ID, "margo_v0.1_primary")
        self.assertNotEqual(DIAG_METHOD_ID, "margo_v0.1_primary")
        self.assertNotEqual(DIAG200_METHOD_ID, "margo_v0.1_primary")
        self.assertNotEqual(PARALLEL_PROBE_METHOD_ID, "margo_v0.1_primary")
        self.assertNotEqual(DIAG500_METHOD_ID, "margo_v0.1_primary")
        self.assertNotEqual(DIAG_LAT_METHOD_ID, "margo_v0.1_primary")
        self.assertNotEqual(DIAG_POMO_METHOD_ID, "margo_v0.1_primary")
        self.assertNotEqual(DIAG_BC_METHOD_ID, "margo_v0.1_primary")
        self.assertNotEqual(DIAG_BCONLY_METHOD_ID, "margo_v0.1_primary")
        self.assertNotEqual(DIAG_BCCONT_METHOD_ID, "margo_v0.1_primary")
        self.assertNotEqual(DIAG_KLPPO_METHOD_ID, "margo_v0.1_primary")
        self.assertNotEqual(DIAG_BCUNSEEN_METHOD_ID, "margo_v0.1_primary")
        self.assertEqual(DIAG_BCSS_EPOCHS, 40)
        self.assertEqual(DIAG_H2_N_GRAPHS, 200)
        self.assertEqual(DIAG_MOTIF_N_TRAIN, 200)
        self.assertEqual(DIAG_TWOPT_MAX_H2_ROUNDS, 8)
        self.assertNotEqual(DIAG_BCFEW_METHOD_ID, "margo_v0.1_primary")
        self.assertNotEqual(DIAG_BCSS_METHOD_ID, "margo_v0.1_primary")
        self.assertNotEqual(DIAG_H2_METHOD_ID, "margo_v0.1_primary")
        self.assertNotEqual(DIAG_MOTIF_METHOD_ID, "margo_v0.1_primary")
        self.assertNotEqual(DIAG_TWOPT_METHOD_ID, "margo_v0.1_primary")
        self.assertNotEqual(DIAG_BC2OPT_METHOD_ID, "margo_v0.1_primary")
        self.assertNotEqual(DIAG_PAIR_METHOD_ID, "margo_v0.1_primary")
        self.assertEqual(DIAG_PAIR_TOP_K, 20)
        self.assertEqual(DIAG_PAIR_ORACLE_N_VAL, 100)
        self.assertEqual(DIAG_PAIR_K_SWEEP, (10, 20, 50))
        self.assertNotEqual(DIAG_PAIR_KSWEEP_METHOD_ID, "margo_v0.1_primary")
        self.assertEqual(DIAG_PAIR_RANKER_K, (10, 20))
        self.assertNotEqual(DIAG_PAIR_RANKER_METHOD_ID, "margo_v0.1_primary")
        self.assertEqual(DIAG_PAIR_SEQ_BESTIMP_K, (20, 50))
        self.assertNotEqual(DIAG_PAIR_SEQ_METHOD_ID, "margo_v0.1_primary")
        self.assertEqual(DIAG_CAVIA_Z_DIM, 32)
        self.assertEqual(DIAG_CAVIA_INNER_STEPS, 20)
        self.assertEqual(DIAG_CAVIA_INNER_LR, 5.0e-4)
        self.assertEqual(DIAG_CAVIA_STRONG_INNER_STEPS, 50)
        self.assertEqual(DIAG_CAVIA_STRONG_INNER_LR, 1.0e-2)
        self.assertNotEqual(DIAG_CAVIA_METHOD_ID, "margo_v0.1_primary")
        self.assertNotEqual(DIAG_CAVIA_ENERGY_METHOD_ID, "margo_v0.1_primary")
        self.assertNotEqual(DIAG_CAVIA_STRONG_METHOD_ID, "margo_v0.1_primary")
        self.assertNotEqual(DIAG_CAVIA_METHOD_ID, DIAG_CAVIA_ENERGY_METHOD_ID)
        self.assertNotEqual(DIAG_CAVIA_METHOD_ID, DIAG_CAVIA_STRONG_METHOD_ID)
        self.assertNotEqual(DIAG_PAIRSUP_METHOD_ID, "margo_v0.1_primary")
        self.assertEqual(DIAG_PAIRSUP_EPOCHS, 40)
        self.assertEqual(DIAG_PAIRSUP_LAMBDA, 0.5)
        self.assertEqual(DIAG_PAIRSUP_A_VAL_T, 575.0)
        self.assertEqual(DIAG_PAIRSUP_A_TEST_T, 557.0)
        self.assertNotEqual(DIAG_CAVIA_BCCONT_METHOD_ID, "margo_v0.1_primary")
        self.assertEqual(DIAG_CAVIA_BCCONT_IDENTITY_T_VAL_LO, 568.0)
        self.assertEqual(DIAG_CAVIA_BCCONT_IDENTITY_T_VAL_HI, 582.0)
        self.assertEqual(DIAG_BINARY_LAT_ITERS, 50)
        self.assertEqual(DIAG_BINARY_LAT_VOCAB, 2)
        self.assertNotEqual(DIAG_BINARY_LAT_METHOD_ID, "margo_v0.1_primary")
        self.assertEqual(K_REPORT, (0, 3))
        self.assertEqual(PARENT_SHA, "0c776924b49da6c66c511c12a8cde70be732e25d")

    def test_split_matches_loader(self):
        self.assertEqual(list(META_TRAIN_IDS), meta_train_distribution_ids())
        self.assertEqual(list(VALIDATION_IDS), validation_distribution_ids())
        self.assertEqual(list(META_TEST_IDS), meta_test_distribution_ids())

    def test_jobs_cover_all_test_dists(self):
        jobs = campaign_jobs()
        trains = [j for j in jobs if j["kind"] == "train"]
        evals = [j for j in jobs if j["kind"] == "meta_test_eval"]
        self.assertEqual(len(trains), 5)
        self.assertEqual(len(evals), 5 * 5 * 2)
        for seed in SEEDS:
            for dist_id in META_TEST_IDS:
                for k in K_REPORT:
                    match = [
                        j
                        for j in evals
                        if j["seed"] == seed and j["distribution_id"] == dist_id and j["k_steps"] == k
                    ]
                    self.assertEqual(len(match), 1)
                    self.assertTrue(match[0]["artifact"].endswith("dist_%d_k%d.json" % (dist_id, k)))

    def test_unknown_seed_rejected(self):
        with self.assertRaises(ValueError):
            seed_run_dir(99)

    def test_gpu_blocked_without_flag_or_env(self):
        with self.assertRaises(GpuPermissionError):
            require_gpu_permission(False, environ={})
        with self.assertRaises(GpuPermissionError):
            require_gpu_permission(True, environ={})
        with self.assertRaises(GpuPermissionError):
            require_gpu_permission(True, environ={GPU_ENV: "0"})
        require_gpu_permission(True, environ={GPU_ENV: "1"})

    def test_execute_cli_blocked(self):
        from spec.phase4_campaign import main

        with self.assertRaises(GpuPermissionError):
            main(["--execute-train", "--seed", "0"])
        with self.assertRaises(GpuPermissionError):
            main(["--gpu-smoke", "--seed", "0"])
        with self.assertRaises(GpuPermissionError):
            main(["--learning-probe", "--seed", "0"])
        with self.assertRaises(GpuPermissionError):
            main(["--diagnostic-1k", "--seed", "0"])
        with self.assertRaises(GpuPermissionError):
            main(["--diagnostic-200", "--seed", "0"])
        with self.assertRaises(GpuPermissionError):
            main(["--parallel-probe", "--seed", "0"])
        with self.assertRaises(GpuPermissionError):
            main(["--diagnostic-500", "--seed", "0"])
        with self.assertRaises(GpuPermissionError):
            main(["--diagnostic-latency-tmec", "--seed", "0"])
        with self.assertRaises(GpuPermissionError):
            main(["--diagnostic-pomo-tmec", "--seed", "0"])
        with self.assertRaises(GpuPermissionError):
            main(["--diagnostic-bc-greedy-tmec", "--seed", "0"])
        with self.assertRaises(GpuPermissionError):
            main(["--diagnostic-bc-only-eval", "--seed", "0"])
        with self.assertRaises(GpuPermissionError):
            main(["--diagnostic-bc-continue", "--seed", "0"])
        with self.assertRaises(GpuPermissionError):
            main(["--diagnostic-kl-bc-ppo", "--seed", "0"])
        with self.assertRaises(GpuPermissionError):
            main(["--diagnostic-bc-unseen", "--seed", "0"])
        with self.assertRaises(GpuPermissionError):
            main(["--diagnostic-bc-fewshot", "--seed", "0"])
        with self.assertRaises(GpuPermissionError):
            main(["--diagnostic-bc-scheduled", "--seed", "0"])
        with self.assertRaises(GpuPermissionError):
            main(["--diagnostic-bc-2opt", "--seed", "0"])
        with self.assertRaises(GpuPermissionError):
            main(["--diagnostic-pair-head", "--seed", "0"])
        with self.assertRaises(GpuPermissionError):
            main(["--diagnostic-pair-ksweep", "--seed", "0"])
        with self.assertRaises(GpuPermissionError):
            main(["--diagnostic-pair-ranker", "--seed", "0"])
        with self.assertRaises(GpuPermissionError):
            main(["--diagnostic-pair-seq", "--seed", "0"])
        with self.assertRaises(GpuPermissionError):
            main(["--diagnostic-cavia", "--seed", "0"])
        with self.assertRaises(GpuPermissionError):
            main(["--diagnostic-cavia-energy", "--seed", "0"])
        with self.assertRaises(GpuPermissionError):
            main(["--diagnostic-cavia-strong", "--seed", "0"])
        with self.assertRaises(GpuPermissionError):
            main(["--diagnostic-pairsup", "--seed", "0"])
        with self.assertRaises(GpuPermissionError):
            main(["--diagnostic-cavia-bccont", "--seed", "0"])
        with self.assertRaises(GpuPermissionError):
            main(["--diagnostic-rewrite-mec", "--seed", "0"])
        with self.assertRaises(GpuPermissionError):
            main(["--diagnostic-binary-lat", "--seed", "0"])
        with self.assertRaises(GpuPermissionError):
            main(["--diagnostic-encoder", "--seed", "0"])
        with self.assertRaises(GpuPermissionError):
            main(["--diagnostic-bestofk", "--seed", "0"])
        with mock.patch.dict(os.environ, {GPU_ENV: "1"}, clear=False):
            with self.assertRaises(GpuPermissionError):
                main(["--execute-train", "--seed", "0"])
            with self.assertRaises(GpuPermissionError):
                main(["--gpu-smoke", "--seed", "0"])
            with self.assertRaises(GpuPermissionError):
                main(["--learning-probe", "--seed", "0"])
            with self.assertRaises(GpuPermissionError):
                main(["--diagnostic-1k", "--seed", "0"])
            with self.assertRaises(GpuPermissionError):
                main(["--diagnostic-200", "--seed", "0"])
            with self.assertRaises(GpuPermissionError):
                main(["--parallel-probe", "--seed", "0"])
            with self.assertRaises(GpuPermissionError):
                main(["--diagnostic-500", "--seed", "0"])
            with self.assertRaises(GpuPermissionError):
                main(["--diagnostic-latency-tmec", "--seed", "0"])
            with self.assertRaises(GpuPermissionError):
                main(["--diagnostic-pomo-tmec", "--seed", "0"])
            with self.assertRaises(GpuPermissionError):
                main(["--diagnostic-bc-greedy-tmec", "--seed", "0"])
            with self.assertRaises(GpuPermissionError):
                main(["--diagnostic-bc-only-eval", "--seed", "0"])
            with self.assertRaises(GpuPermissionError):
                main(["--diagnostic-bc-continue", "--seed", "0"])
            with self.assertRaises(GpuPermissionError):
                main(["--diagnostic-kl-bc-ppo", "--seed", "0"])
            with self.assertRaises(GpuPermissionError):
                main(["--diagnostic-bc-unseen", "--seed", "0"])
            with self.assertRaises(GpuPermissionError):
                main(["--diagnostic-bc-fewshot", "--seed", "0"])
            with self.assertRaises(GpuPermissionError):
                main(["--diagnostic-bc-scheduled", "--seed", "0"])
            with self.assertRaises(GpuPermissionError):
                main(["--diagnostic-bc-2opt", "--seed", "0"])
            with self.assertRaises(GpuPermissionError):
                main(["--diagnostic-pair-head", "--seed", "0"])
            with self.assertRaises(GpuPermissionError):
                main(["--diagnostic-pair-ksweep", "--seed", "0"])
            with self.assertRaises(GpuPermissionError):
                main(["--diagnostic-pair-ranker", "--seed", "0"])
            with self.assertRaises(GpuPermissionError):
                main(["--diagnostic-pair-seq", "--seed", "0"])
            with self.assertRaises(GpuPermissionError):
                main(["--diagnostic-cavia", "--seed", "0"])
            with self.assertRaises(GpuPermissionError):
                main(["--diagnostic-cavia-energy", "--seed", "0"])
            with self.assertRaises(GpuPermissionError):
                main(["--diagnostic-cavia-strong", "--seed", "0"])
            with self.assertRaises(GpuPermissionError):
                main(["--diagnostic-pairsup", "--seed", "0"])
            with self.assertRaises(GpuPermissionError):
                main(["--diagnostic-cavia-bccont", "--seed", "0"])
            with self.assertRaises(GpuPermissionError):
                main(["--diagnostic-rewrite-mec", "--seed", "0"])
            with self.assertRaises(GpuPermissionError):
                main(["--diagnostic-binary-lat", "--seed", "0"])
            with self.assertRaises(GpuPermissionError):
                main(["--diagnostic-bestofk", "--seed", "0"])

    def test_provenance_protocol_fields(self):
        row = provenance_template(0)
        self.assertEqual(row["outer_update_method"], "mrlco_first_order_mean_pseudogradient")
        self.assertEqual(row["hyperparameter_provenance.policy"], "fixed_literature_derived_defaults")
        self.assertEqual(row["k_steps"], 3)
        self.assertFalse(row["gpu_requested"])
        self.assertFalse(row["gpu_finished"])
        self.assertFalse(row["paper_result"])
        self.assertFalse(row["optimization_claim"])
        self.assertEqual(row["units_time"], "seconds")
        self.assertEqual(row["units_energy"], "joules")

    def test_plan_write(self):
        plan = write_plan()
        self.assertEqual(plan["gpu"]["require_cli_flag"], GPU_FLAG)
        self.assertTrue(plan["gpu"]["require_chat_approval"])
        self.assertEqual(plan["status"], "IN PROGRESS")

    def test_train_driver_not_imported_by_campaign(self):
        source = (ROOT / "spec" / "phase4_campaign.py").read_text()
        self.assertNotIn("import tensorflow", source)
        self.assertNotIn("from meta_trainer import", source)
        self.assertIn("require_gpu_permission", source)

    def test_eval_path_layout(self):
        run_dir = seed_run_dir(0)
        path = eval_json_path(run_dir, 12, 3)
        self.assertEqual(path.name, "dist_12_k3.json")
        self.assertIn("meta_test", str(path))

    def test_probe_and_diag_dirs_are_not_primary(self):
        from spec.phase4_campaign import smoke_run_dir

        primary = seed_run_dir(0)
        smoke = smoke_run_dir(0)
        probe = probe_run_dir(0)
        diag = diag_run_dir(0)
        diag200 = diag200_run_dir(0)
        parallel_probe = parallel_probe_run_dir(0)
        diag500 = diag500_run_dir(0)
        diag_lat = diag_lat_run_dir(0)
        diag_pomo = diag_pomo_run_dir(0)
        diag_bc = diag_bc_run_dir(0)
        diag_bconly = diag_bconly_run_dir(0)
        diag_bccont = diag_bccont_run_dir(0)
        diag_klppo = diag_klppo_run_dir(0)
        diag_bcunseen = diag_bcunseen_run_dir(0)
        diag_bcfew = diag_bcfew_run_dir(0)
        diag_bcss = diag_bcss_run_dir(0)
        diag_h2 = diag_h2_run_dir(0)
        diag_motif = diag_motif_run_dir(0)
        diag_twopt = diag_twopt_run_dir(0)
        diag_bc2opt = diag_bc2opt_run_dir(0)
        diag_pair = diag_pair_run_dir(0)
        diag_pair_ksweep = diag_pair_ksweep_run_dir(0)
        diag_pair_ranker = diag_pair_ranker_run_dir(0)
        diag_pair_seq = diag_pair_seq_run_dir(0)
        diag_cavia = diag_cavia_run_dir(0)
        diag_cavia_energy = diag_cavia_energy_run_dir(0)
        diag_cavia_strong = diag_cavia_strong_run_dir(0)
        diag_pairsup = diag_pairsup_run_dir(0)
        diag_cavia_bccont = diag_cavia_bccont_run_dir(0)
        diag_pairfrac = diag_pairfrac_run_dir(0)
        diag_rewrite = diag_rewrite_run_dir(0)
        diag_oracle = diag_oracle_dist_run_dir(0)
        diag_binary = diag_binary_lat_run_dir(0)
        diag_enc = diag_encoder_run_dir(0, "meanagg", "triple")
        diag_bok = diag_bestofk_run_dir(0)
        diag_eas = diag_eas_run_dir(0, "lastlayer", "pg_il")
        diag_eas_inst = diag_eas_inst_run_dir(0, "lastlayer", "pg", 32)
        diag_expert_prof = diag_expert_profiles_run_dir(0)
        diag_bc_prof = diag_bc_profiles_run_dir(0)
        diag_eas_prof = diag_eas_profiles_run_dir(0)
        diag_ctx_prof = diag_ctx_profiles_run_dir(0)
        diag_bok_prof = diag_bok_profiles_run_dir(0)
        diag_expert_energy = diag_expert_energy_run_dir(0)
        diag_bc_energy = diag_bc_energy_run_dir(0)
        diag_eas_energy = diag_eas_energy_run_dir(0)
        self.assertIn("margo_v0.1_primary", str(primary))
        self.assertIn("gpu_smoke", str(smoke))
        self.assertIn("margo_v0.1_learning_probe", str(probe))
        self.assertIn("margo_v0.1_diag_1k", str(diag))
        self.assertIn("margo_v0.1_diag_200", str(diag200))
        self.assertIn("margo_v0.1_parallel_probe", str(parallel_probe))
        self.assertIn("margo_v0.1_diag_500_parallel", str(diag500))
        self.assertIn("margo_v0.1_diag_latency_tmec", str(diag_lat))
        self.assertIn("margo_v0.1_diag_pomo_tmec", str(diag_pomo))
        self.assertIn("margo_v0.1_diag_bc_greedy_tmec", str(diag_bc))
        self.assertIn("margo_v0.1_diag_bc_only_eval", str(diag_bconly))
        self.assertIn("margo_v0.1_diag_bc_continue", str(diag_bccont))
        self.assertIn("margo_v0.1_diag_kl_bc_ppo", str(diag_klppo))
        self.assertIn("margo_v0.1_diag_bc_unseen", str(diag_bcunseen))
        self.assertIn("margo_v0.1_diag_bc_fewshot", str(diag_bcfew))
        self.assertIn("margo_v0.1_diag_bc_scheduled", str(diag_bcss))
        self.assertIn("margo_v0.1_diag_hamming2_expert", str(diag_h2))
        self.assertIn("margo_v0.1_diag_motif_expert", str(diag_motif))
        self.assertIn("margo_v0.1_diag_2opt_expert", str(diag_twopt))
        self.assertIn("margo_v0.1_diag_bc_2opt", str(diag_bc2opt))
        self.assertIn("margo_v0.1_diag_pair_head", str(diag_pair))
        self.assertIn("margo_v0.1_diag_pair_ksweep", str(diag_pair_ksweep))
        self.assertIn("margo_v0.1_diag_pair_ranker", str(diag_pair_ranker))
        self.assertIn("margo_v0.1_diag_pair_seq", str(diag_pair_seq))
        self.assertIn("margo_v0.2_diag_cavia_frozen", str(diag_cavia))
        self.assertIn("margo_v0.2_diag_cavia_energy", str(diag_cavia_energy))
        self.assertIn("margo_v0.2_diag_cavia_strong", str(diag_cavia_strong))
        self.assertIn("margo_v0.2_diag_pairsup", str(diag_pairsup))
        self.assertIn("margo_v0.2_diag_cavia_bccont", str(diag_cavia_bccont))
        self.assertIn("margo_v0.2_diag_pairfrac_mec", str(diag_pairfrac))
        self.assertIn("margo_v0.2_diag_rewrite_mec", str(diag_rewrite))
        self.assertIn("margo_v0.2_diag_oracle_dist", str(diag_oracle))
        self.assertIn("margo_v0.2_diag_binary_lat", str(diag_binary))
        self.assertIn("margo_v0.3_diag_encoder", str(diag_enc))
        self.assertIn("margo_v0.3_diag_bestofk", str(diag_bok))
        self.assertIn("margo_v0.3_diag_eas", str(diag_eas))
        self.assertIn("margo_v0.3_diag_eas_inst", str(diag_eas_inst))
        self.assertIn("margo_v0.3_expert_profiles", str(diag_expert_prof))
        self.assertIn("margo_v0.3_bc_profiles", str(diag_bc_prof))
        self.assertIn("margo_v0.3_eas_profiles", str(diag_eas_prof))
        self.assertIn("margo_v0.3_ctx_profiles", str(diag_ctx_prof))
        self.assertIn("margo_v0.3_bok_profiles", str(diag_bok_prof))
        self.assertIn("margo_v0.3_expert_energy", str(diag_expert_energy))
        self.assertIn("margo_v0.3_bc_energy", str(diag_bc_energy))
        self.assertIn("margo_v0.3_eas_energy", str(diag_eas_energy))
        self.assertEqual(
            len(
                {
                    str(primary),
                    str(smoke),
                    str(probe),
                    str(diag),
                    str(diag200),
                    str(parallel_probe),
                    str(diag500),
                    str(diag_lat),
                    str(diag_pomo),
                    str(diag_bc),
                    str(diag_bconly),
                    str(diag_bccont),
                    str(diag_klppo),
                    str(diag_bcunseen),
                    str(diag_bcfew),
                    str(diag_bcss),
                    str(diag_h2),
                    str(diag_motif),
                    str(diag_twopt),
                    str(diag_bc2opt),
                    str(diag_pair),
                    str(diag_pair_ksweep),
                    str(diag_pair_ranker),
                    str(diag_pair_seq),
                    str(diag_cavia),
                    str(diag_cavia_energy),
                    str(diag_cavia_strong),
                    str(diag_pairsup),
                    str(diag_cavia_bccont),
                    str(diag_pairfrac),
                    str(diag_rewrite),
                    str(diag_oracle),
                    str(diag_binary),
                    str(diag_enc),
                    str(diag_bok),
                    str(diag_eas),
                    str(diag_eas_inst),
                    str(diag_expert_prof),
                    str(diag_bc_prof),
                    str(diag_eas_prof),
                    str(diag_ctx_prof),
                    str(diag_bok_prof),
                    str(diag_expert_energy),
                    str(diag_bc_energy),
                    str(diag_eas_energy),
                }
            ),
            45,
        )

    def test_kl_anchor_matches_aggregator_uids(self):
        from spec.kl_bc_anchor import canonical_weight_slot, pair_ckpt_to_vars

        a = canonical_weight_slot(
            "bc_frozen/encoder/meanaggregator_45_vars/neigh_weights:0", "bc_frozen"
        )
        b = canonical_weight_slot(
            "core_policy/encoder/meanaggregator_3_vars/neigh_weights:0", "core_policy"
        )
        self.assertEqual(a, b)

        class _V(object):
            def __init__(self, name):
                self.name = name

        loaded = {
            "core_policy/encoder/meanaggregator_3_vars/neigh_weights:0": [1.0],
            "core_policy/encoder/meanaggregator_4_vars/neigh_weights:0": [2.0],
        }
        dst = [
            _V("bc_frozen/encoder/meanaggregator_45_vars/neigh_weights:0"),
            _V("bc_frozen/encoder/meanaggregator_46_vars/neigh_weights:0"),
        ]
        paired = pair_ckpt_to_vars(loaded, dst, "core_policy", "bc_frozen")
        self.assertEqual([p[1] for p in paired], [[1.0], [2.0]])

    def test_fewshot_encoder_freeze_and_mix(self):
        from spec.bc_fewshot import CE_STEPS, action_mix, encoder_frozen_adapt_vars, is_encoder_var_name

        self.assertEqual(CE_STEPS, 3)
        self.assertTrue(is_encoder_var_name("validation_policy/encoder/meanaggregator_3_vars/neigh_weights:0"))
        self.assertFalse(is_encoder_var_name("validation_policy/decoder/lstm_cell/kernel:0"))
        self.assertFalse(is_encoder_var_name("validation_policy/output_projection/kernel:0"))

        class _V(object):
            def __init__(self, name):
                self.name = name

        frozen, adapt = encoder_frozen_adapt_vars(
            [
                _V("validation_policy/encoder/dense/kernel:0"),
                _V("validation_policy/decoder/lstm_cell/kernel:0"),
                _V("validation_policy/output_projection/kernel:0"),
            ]
        )
        self.assertEqual([v.name for v in frozen], ["validation_policy/encoder/dense/kernel:0"])
        self.assertEqual(len(adapt), 2)
        mix = action_mix({"actions": [[1, 1, 0], [1, 2, 1]]})
        self.assertAlmostEqual(mix["mec_frac"], 4.0 / 6.0)
        self.assertAlmostEqual(mix["local_frac"], 1.0 / 6.0)
        self.assertAlmostEqual(mix["v2v_frac"], 1.0 / 6.0)

    def test_scheduled_sampling_mix(self):
        import numpy as np
        from spec.bc_scheduled import SS_EPOCHS, mix_decoder_inputs, ss_eps

        self.assertEqual(SS_EPOCHS, DIAG_BCSS_EPOCHS)
        self.assertAlmostEqual(ss_eps(0, 40), 0.8)
        self.assertAlmostEqual(ss_eps(39, 40), 0.3)
        gold = np.array([[9, 1, 2], [9, 0, 1]], dtype=np.int32)
        pred = np.array([[8, 2, 0], [8, 1, 2]], dtype=np.int32)
        all_gold = mix_decoder_inputs(gold, pred, 1.0, np.random.RandomState(0))
        self.assertEqual(all_gold.tolist(), [[0, 1, 2], [0, 0, 1]])
        all_pred = mix_decoder_inputs(gold, pred, 0.0, np.random.RandomState(0))
        self.assertEqual(all_pred.tolist(), [[0, 2, 0], [0, 1, 2]])

    def test_hamming2_neighbors_and_verdict(self):
        import numpy as np
        from spec.hamming2_probe import classify_mean_delta, count_h2_neighbors, stratified_graph_picks

        self.assertEqual(count_h2_neighbors([1] * 20), 760)
        self.assertEqual(classify_mean_delta(3.0), "ood_not_teacher")
        self.assertEqual(classify_mean_delta(18.0), "mixed")
        self.assertEqual(classify_mean_delta(38.0), "weak_expert")
        rng = np.random.RandomState(0)
        picks = stratified_graph_picks((1, 3, 4), 5, 10, rng)
        self.assertEqual(len(picks), 5)
        self.assertEqual(len({(d, g) for d, g in picks}), 5)

    def test_hamming2_cli_does_not_need_gpu(self):
        from spec.phase4_campaign import main

        with mock.patch("spec.phase4_train_driver.run_diagnostic_hamming2_expert", return_value=None) as fn:
            rc = main(["--diagnostic-hamming2-expert", "--seed", "0"])
        self.assertEqual(rc, 0)
        fn.assert_called_once()

    def test_motif_cc_homophily_and_verdict(self):
        from spec.motif_audit import (
            classify_motif,
            connected_components,
            decoder_acts_to_task_order,
            edge_homophily,
            local_neighbor_frac,
            undirected_adj,
        )

        succ = [{1}, {2}, set(), set()]
        adj = undirected_adj(succ)
        act = [0, 0, 1, 0]
        n_cc, sizes = connected_components([0, 1, 3], adj)
        self.assertEqual(n_cc, 2)
        self.assertEqual(sorted(sizes), [1, 2])
        homo, n_same, n_ll, n_mm, n_vv = edge_homophily(act, [(0, 1), (1, 2)])
        self.assertAlmostEqual(homo, 0.5)
        self.assertEqual((n_same, n_ll, n_mm, n_vv), (1, 1, 0, 0))
        self.assertAlmostEqual(local_neighbor_frac(act, adj), 2.0 / 3.0)
        self.assertEqual(classify_motif(5.0, 1.2, 0.8), "region_clustered")
        self.assertEqual(classify_motif(5.0, 4.5, 0.2), "scattered_support")
        self.assertEqual(classify_motif(5.0, 2.5, 0.4), "mixed_structure")
        self.assertEqual(decoder_acts_to_task_order([2, 0, 1], [1, 0, 2], 3), [0, 2, 1])

    def test_motif_cli_does_not_need_gpu(self):
        from spec.phase4_campaign import main

        with mock.patch("spec.phase4_train_driver.run_diagnostic_motif_expert", return_value=None) as fn:
            rc = main(["--diagnostic-motif-expert", "--seed", "0"])
        self.assertEqual(rc, 0)
        fn.assert_called_once()

    def test_twopt_pair_synergy_and_h1(self):
        from spec.twopt_expert import iterate_2opt

        def pair_only(actions):
            if int(actions[0]) == 0 and int(actions[1]) == 0:
                return 7.0
            return 10.0

        out = iterate_2opt([1, 1, 1], 10.0, pair_only)
        self.assertEqual(out["actions"][:2], [0, 0])
        self.assertEqual(out["h1_moves"], 0)
        self.assertEqual(out["h2_moves"], 1)
        self.assertAlmostEqual(out["t"], 7.0)
        self.assertAlmostEqual(out["t_after_h1"], 10.0)
        self.assertAlmostEqual(out["t_after_first_h2"], 7.0)

        def count_ones(actions):
            return float(sum(1 for a in actions if int(a) == 1))

        out_h1 = iterate_2opt([1, 1, 0], 2.0, count_ones)
        self.assertEqual(out_h1["actions"], [0, 0, 0])
        self.assertGreaterEqual(out_h1["h1_moves"], 2)
        self.assertEqual(out_h1["h2_moves"], 0)
        self.assertAlmostEqual(out_h1["t"], 0.0)

    def test_align_greedy_pred_pads_short_time(self):
        from spec.bc_greedy_mec import align_greedy_pred

        pred, trunc = align_greedy_pred(np.ones((4, 12), dtype=np.int32), 20)
        self.assertTrue(trunc)
        self.assertEqual(pred.shape, (4, 20))
        self.assertTrue(np.all(pred[:, 12:] == 1))
        same, trunc2 = align_greedy_pred(np.zeros((2, 20), dtype=np.int32), 20)
        self.assertFalse(trunc2)
        self.assertEqual(same.shape, (2, 20))

    def test_greedy_end_token_outside_actions(self):
        src = (ROOT / "policies" / "meta_seq2seq_policy.py").read_text()
        self.assertIn("end_token=int(vocab_size)", src)
        self.assertNotRegex(src, r"(?m)^\s*end_token=2,?\s*$")
        self.assertNotRegex(src, r"(?m)^\s*end_token=3,?\s*$")

    def test_binary_greedy_never_emits_v2v(self):
        from env.mec_offloaing_envs.scheduler.greedy import (
            BINARY_ACTIONS,
            greedy_from_mec_plan,
            greedy_plan,
            resolve_greedy_actions,
        )
        from env.mec_offloaing_envs.scheduler.resources import ResourceConfig
        import types

        self.assertEqual(resolve_greedy_actions(None), (0, 1, 2))
        self.assertEqual(resolve_greedy_actions(BINARY_ACTIONS), (0, 1))
        self.assertEqual(DIAG_BINARY_LAT_METHOD_ID, "margo_v0.2_diag_binary_lat")
        self.assertEqual(DIAG_BINARY_LAT_VOCAB, 2)
        self.assertEqual(DIAG_ENC_METHOD_ID, "margo_v0.3_diag_encoder")
        self.assertEqual(DIAG_BOK_METHOD_ID, "margo_v0.3_diag_bestofk")
        self.assertEqual(DIAG_EAS_METHOD_ID, "margo_v0.3_diag_eas")
        self.assertEqual(DIAG_EAS_INST_METHOD_ID, "margo_v0.3_diag_eas_inst")
        from spec.phase4_campaign import (
            DIAG_BC_ENERGY_METHOD_ID,
            DIAG_BC_PROFILES_METHOD_ID,
            DIAG_BOK_PROFILES_METHOD_ID,
            DIAG_CTX_PROFILES_METHOD_ID,
            DIAG_EAS_ENERGY_METHOD_ID,
            DIAG_EAS_PROFILES_METHOD_ID,
            DIAG_EXPERT_ENERGY_METHOD_ID,
            DIAG_EXPERT_PROFILES_METHOD_ID,
        )

        self.assertEqual(DIAG_EXPERT_PROFILES_METHOD_ID, "margo_v0.3_expert_profiles")
        self.assertEqual(DIAG_BC_PROFILES_METHOD_ID, "margo_v0.3_bc_profiles")
        self.assertEqual(DIAG_EAS_PROFILES_METHOD_ID, "margo_v0.3_eas_profiles")
        self.assertEqual(DIAG_CTX_PROFILES_METHOD_ID, "margo_v0.3_ctx_profiles")
        self.assertEqual(DIAG_BOK_PROFILES_METHOD_ID, "margo_v0.3_bok_profiles")
        self.assertEqual(DIAG_EXPERT_ENERGY_METHOD_ID, "margo_v0.3_expert_energy")
        self.assertEqual(DIAG_BC_ENERGY_METHOD_ID, "margo_v0.3_bc_energy")
        self.assertEqual(DIAG_EAS_ENERGY_METHOD_ID, "margo_v0.3_eas_energy")

        class TG:
            task_number = 2
            task_list = [
                types.SimpleNamespace(processing_data_size=1048576, transmission_data_size=458752),
                types.SimpleNamespace(processing_data_size=1048576, transmission_data_size=458752),
            ]
            pre_task_sets = [set(), {0}]
            edge_set = [[0, 0, 1048576, 458752, 1, 1, 1048576]]
            prioritize_sequence = [0, 1]

        cfg = ResourceConfig.from_frozen_yaml()
        plan, _ = greedy_plan(TG(), cfg, actions=BINARY_ACTIONS)
        self.assertTrue({a for _, a in plan} <= {0, 1})
        plan2, _ = greedy_from_mec_plan(TG(), cfg, actions=BINARY_ACTIONS)
        self.assertTrue({a for _, a in plan2} <= {0, 1})

    def test_pair_motif_diamond_and_verdict(self):
        from spec.pair_head import (
            classify_pair_verdict,
            joint_id,
            motif_pairs,
            predict_joint,
            split_joint,
            train_joint_softmax,
        )

        succ = [{1, 2}, {3}, {3}, set()]
        pre = [set(), {0}, {0}, {1, 2}]
        pairs = { (p["i"], p["j"]): p for p in motif_pairs(succ, pre, [0, 1, 2, 3]) }
        self.assertEqual(pairs[(1, 2)]["sibling"], 1)
        self.assertEqual(pairs[(1, 2)]["join"], 1)
        self.assertEqual(pairs[(0, 1)]["direct"], 1)
        self.assertEqual(split_joint(joint_id(1, 0)), (1, 0))
        self.assertEqual(classify_pair_verdict(582.0, 470.0, oracle_t=450.0), "decoder_interaction")
        self.assertEqual(classify_pair_verdict(582.0, 575.0, oracle_t=570.0), "encoder_insufficient")
        self.assertEqual(
            classify_pair_verdict(582.0, 575.0, oracle_t=450.0),
            "pair_search_works_ranker_fails",
        )
        from spec.pair_head import PAIR_K_SWEEP

        self.assertEqual(PAIR_K_SWEEP, (10, 20, 50))
        rng = np.random.RandomState(0)
        y = np.arange(90) % 9
        x = np.zeros((90, 9), dtype=np.float64)
        x[np.arange(90), y] = 4.0
        x += rng.randn(90, 9) * 0.05
        model = train_joint_softmax(x, y, rng, epochs=25, lr=0.2, batch=30)
        pred, _ = predict_joint(model, x)
        self.assertGreater(float(np.mean(pred == y)), 0.9)

    def test_pair_ranker_ridge_and_verdict(self):
        from spec.pair_ranker import (
            PAIR_RANKER_K,
            classify_ranker_verdict,
            pearson_corr,
            predict_ridge,
            recall_at_k,
            train_weighted_ridge,
        )

        self.assertEqual(PAIR_RANKER_K, (10, 20))
        rng = np.random.RandomState(0)
        true_w = rng.randn(8)
        x = rng.randn(400, 8)
        y = np.maximum(x.dot(true_w), 0.0)
        model = train_weighted_ridge(x, y, lam=1e-3, pos_weight=5.0)
        pred = predict_ridge(model, x)
        self.assertGreater(pearson_corr(pred, y), 0.85)
        rec = recall_at_k(y, pred, k=20)
        rec_rand = recall_at_k(y, rng.randn(400), k=20)
        self.assertGreater(rec, rec_rand)
        self.assertEqual(classify_ranker_verdict(458.0, 440.0, 438.0), "delta_ranker_fits")
        self.assertEqual(classify_ranker_verdict(458.0, 456.0, 454.0), "one_pass_gain_rank_insufficient")
        self.assertEqual(classify_ranker_verdict(458.0, 455.0, 438.0), "delta_labels_not_enough")

    def test_pair_seq_verdict(self):
        from spec.pair_seq import SEQ_BESTIMP_K, classify_seq_verdict

        self.assertEqual(SEQ_BESTIMP_K, (20, 50))
        self.assertEqual(classify_seq_verdict(457.8, 457.0, 438.0), "seq_top20_closes_oracle_gap")
        self.assertEqual(classify_seq_verdict(457.8, 457.0, 448.0), "seq_helps_need_more_pairs")
        self.assertEqual(classify_seq_verdict(457.8, 448.0, 456.0), "multipass_helps")
        self.assertEqual(classify_seq_verdict(457.8, 456.0, 456.0), "seq_on_top20_not_enough")

    def test_cavia_phase1_contract(self):
        from spec.cavia_objective import (
            CAVIA_INNER_LR,
            CAVIA_INNER_STEPS,
            CAVIA_STRONG_INNER_LR,
            CAVIA_STRONG_INNER_STEPS,
            CAVIA_Z_DIM,
            classify_cavia_verdict,
        )

        self.assertEqual(DIAG_CAVIA_Z_DIM, CAVIA_Z_DIM)
        self.assertEqual(DIAG_CAVIA_INNER_STEPS, CAVIA_INNER_STEPS)
        self.assertEqual(DIAG_CAVIA_INNER_LR, CAVIA_INNER_LR)
        self.assertEqual(DIAG_CAVIA_STRONG_INNER_STEPS, CAVIA_STRONG_INNER_STEPS)
        self.assertEqual(DIAG_CAVIA_STRONG_INNER_LR, CAVIA_STRONG_INNER_LR)
        self.assertEqual(classify_cavia_verdict(581.8, 581.8, 0.20, identity_t=581.8), "cavia_no_gain")
        self.assertEqual(classify_cavia_verdict(581.8, 560.0, 0.20, identity_t=581.8), "cavia_helps")
        from spec.pair_sup import PAIRSUP_A_VAL_T, PAIRSUP_EPOCHS, PAIRSUP_LAMBDA, classify_pairsup_verdict

        self.assertEqual(DIAG_PAIRSUP_EPOCHS, PAIRSUP_EPOCHS)
        self.assertEqual(DIAG_PAIRSUP_LAMBDA, PAIRSUP_LAMBDA)
        self.assertEqual(DIAG_PAIRSUP_A_VAL_T, PAIRSUP_A_VAL_T)
        self.assertEqual(classify_pairsup_verdict(575.0, 0.20), "pairsup_no_gain")
        self.assertEqual(classify_pairsup_verdict(560.0, 0.20), "pairsup_helps")
        from spec.cavia_objective import (
            CAVIA_BCCONT_IDENTITY_T_VAL_HI,
            CAVIA_BCCONT_IDENTITY_T_VAL_LO,
        )

        self.assertEqual(DIAG_CAVIA_BCCONT_IDENTITY_T_VAL_LO, CAVIA_BCCONT_IDENTITY_T_VAL_LO)
        self.assertEqual(DIAG_CAVIA_BCCONT_IDENTITY_T_VAL_HI, CAVIA_BCCONT_IDENTITY_T_VAL_HI)
        from spec.rewrite_mec import REWRITE_A_VAL_T, REWRITE_EPOCHS, REWRITE_K, REWRITE_LAMBDA, classify_rewrite_verdict

        self.assertEqual(DIAG_REWRITE_EPOCHS, REWRITE_EPOCHS)
        self.assertEqual(DIAG_REWRITE_LAMBDA, REWRITE_LAMBDA)
        self.assertEqual(DIAG_REWRITE_K, REWRITE_K)
        self.assertEqual(DIAG_REWRITE_A_VAL_T, REWRITE_A_VAL_T)
        self.assertEqual(DIAG_REWRITE_METHOD_ID, "margo_v0.2_diag_rewrite_mec")
        self.assertEqual(classify_rewrite_verdict(575.0, 0.20), "rewrite_no_gain")
        self.assertEqual(classify_rewrite_verdict(560.0, 0.20), "rewrite_helps")
        from spec.oracle_dist import (
            ORACLE_A_VAL_T,
            ORACLE_EPOCHS,
            ORACLE_IDENTITY_T_VAL_HI,
            ORACLE_IDENTITY_T_VAL_LO,
            ORACLE_N_DIST,
            ORACLE_Z_DIM,
            classify_oracle_verdict,
        )

        self.assertEqual(DIAG_ORACLE_DIST_EPOCHS, ORACLE_EPOCHS)
        self.assertEqual(DIAG_ORACLE_DIST_Z_DIM, ORACLE_Z_DIM)
        self.assertEqual(DIAG_ORACLE_DIST_N_DIST, ORACLE_N_DIST)
        self.assertEqual(DIAG_ORACLE_DIST_A_VAL_T, ORACLE_A_VAL_T)
        self.assertEqual(DIAG_ORACLE_DIST_IDENTITY_T_VAL_LO, ORACLE_IDENTITY_T_VAL_LO)
        self.assertEqual(DIAG_ORACLE_DIST_IDENTITY_T_VAL_HI, ORACLE_IDENTITY_T_VAL_HI)
        self.assertEqual(DIAG_ORACLE_DIST_METHOD_ID, "margo_v0.2_diag_oracle_dist")
        self.assertEqual(classify_oracle_verdict(575.0, 0.20), "oracle_no_gain")
        self.assertEqual(classify_oracle_verdict(560.0, 0.20), "oracle_helps")

    def test_twopt_cli_does_not_need_gpu(self):
        from spec.phase4_campaign import main

        with mock.patch("spec.phase4_train_driver.run_diagnostic_2opt_expert", return_value=None) as fn:
            rc = main(["--diagnostic-2opt-expert", "--seed", "0"])
        self.assertEqual(rc, 0)
        fn.assert_called_once()

    def test_pairfrac_cli_does_not_need_gpu(self):
        from spec.phase4_campaign import main

        with mock.patch("spec.phase4_train_driver.run_diagnostic_pairfrac_mec", return_value=None) as fn:
            rc = main(["--diagnostic-pairfrac-mec", "--seed", "0"])
        self.assertEqual(rc, 0)
        fn.assert_called_once()
        self.assertEqual(DIAG_PAIRFRAC_METHOD_ID, "margo_v0.2_diag_pairfrac_mec")
        self.assertEqual(DIAG_PAIRFRAC_N_GRAPHS, 200)

    def test_status_not_closed(self):
        text = (ROOT / "spec" / "PHASE4_STATUS.md").read_text()
        self.assertRegex(text, r"(?m)^Status:\s*IN PROGRESS\b")
        self.assertNotRegex(text, r"^Status:\s*CLOSED\b")


if __name__ == "__main__":
    unittest.main()
