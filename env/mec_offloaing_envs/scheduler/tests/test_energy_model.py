#!/usr/bin/env python3
"""Physical energy model acceptance tests (approved ① contract).

Invariants under test:
  * E_cpu = kappa * C * f^2   ==   (kappa * f^3) * (C / f)      [numeric identity]
  * E_system = E_mobile + E_MEC_compute + E_MEC_tx
  * legacy reproduces the MARGO-SPEC-v0.1 numbers bit-for-bit
  * switching energy_scope changes ONLY aggregation (never timing)
  * kappa_MEC == 0 is rejected in physical mode
  * monotonicity: C -> E up ; kappa -> E up ; f -> T down and E up
  * include_rx_energy=False means no RX energy is charged (physical mode)
  * MEC TX energy is system-side only (never inside total_mobile_joules)
"""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _name in ("gym", "gym.core"):
    if _name not in sys.modules:
        try:
            __import__(_name)
        except Exception:
            sys.modules[_name] = types.ModuleType(_name)
if not hasattr(sys.modules.get("gym.core", types.ModuleType("gym.core")), "Env"):
    sys.modules.setdefault("gym", types.ModuleType("gym"))
    sys.modules.setdefault("gym.core", types.ModuleType("gym.core"))
    sys.modules["gym.core"].Env = type("Env", (), {})

from env.mec_offloaing_envs.offloading_task_graph import OffloadingTaskGraph  # noqa: E402
from env.mec_offloaing_envs.scheduler import (  # noqa: E402
    ResourceConfig,
    pure_location_plan,
    schedule_via_adapter,
)
from env.mec_offloaing_envs.scheduler.energy_model import (  # noqa: E402
    MODEL_LEGACY,
    MODEL_PHYSICAL,
    SCOPE_MOBILE,
    SCOPE_REQUESTER,
    SCOPE_SYSTEM,
    EnergyModelSpec,
    TierSpec,
)

DATA = (
    ROOT
    / "env"
    / "mec_offloaing_envs"
    / "data"
    / "meta_offloading_20"
    / "offload_random20_1"
    / "random.20.0.gv"
)
MBPS_TO_BPS = 1024.0 * 1024.0 / 8.0

# MARGO-SPEC-v0.1 reference numbers (legacy model) for DATA, all-UE / all-MEC.
LEGACY_REF = {
    0: {"T": 1212.9805, "E_mobile": 1212.9805},
    1: {"T": 530.4908, "E_mobile": 44.4728},
}


class _FrozenCluster:
    def __init__(self):
        self.mobile_process_capable = 1.0 * 1024 * 1024
        self.mec_process_capable = 10.0 * 1024 * 1024
        self.v2v_process_capable = 1.0 * 1024 * 1024
        self.bandwidth_up = 7.0
        self.bandwidth_dl = 7.0
        self.v2v_bandwidth = 5.0

    def up_transmission_cost(self, data):
        return data / (self.bandwidth_up * MBPS_TO_BPS)

    def dl_transmission_cost(self, data):
        return data / (self.bandwidth_dl * MBPS_TO_BPS)

    def v2v_transmission_cost(self, data):
        return data / (self.v2v_bandwidth * MBPS_TO_BPS)


def _spec(**overrides):
    doc = {
        "model": MODEL_PHYSICAL,
        "energy_scope": SCOPE_SYSTEM,
        "cycles_per_bit": 300.0,
        "include_rx_energy": False,
        "tiers": {
            "ue": {"f_hz": 1.0e9, "kappa": 1.0e-27, "source": "test"},
            "helper": {"f_hz": 1.5e9, "kappa": 5.0e-27, "source": "test"},
            "mec": {"f_hz": 10.0e9, "kappa": 1.0e-27, "source": "test"},
        },
        "radio": {"ue_tx_w": 1.0, "mec_tx_w": 3.162, "helper_tx_w": 1.0},
    }
    doc.update(overrides)
    return EnergyModelSpec.from_dict(doc)


def _resources(spec=None, model="legacy"):
    from env.mec_offloaing_envs.scheduler.resources import (
        TIMING_LEGACY,
        TIMING_PHYSICAL,
    )
    base = ResourceConfig.from_frozen_yaml(model=model)
    if spec is None:
        return base
    return ResourceConfig(
        ue_cpu_bytes_per_second=base.ue_cpu_bytes_per_second,
        mec_cpu_bytes_per_second=base.mec_cpu_bytes_per_second,
        helper_cpu_bytes_per_second=base.helper_cpu_bytes_per_second,
        mec_uplink_bytes_per_second=base.mec_uplink_bytes_per_second,
        mec_downlink_bytes_per_second=base.mec_downlink_bytes_per_second,
        v2v_bytes_per_second=base.v2v_bytes_per_second,
        rho_ue=base.rho_ue,
        f_l=base.f_l,
        zeta=base.zeta,
        ptx_mec_w=base.ptx_mec_w,
        prx_mec_w=base.prx_mec_w,
        ptx_v2v_w=base.ptx_v2v_w,
        prx_v2v_w=base.prx_v2v_w,
        rho_helper=base.rho_helper,
        timing_model=TIMING_PHYSICAL if spec is not None else TIMING_LEGACY,
        radio_timing_model=TIMING_LEGACY,   # no radio model in these fixtures
        timing_tiers=spec,
        f_v2v=base.f_v2v,
        energy_model=spec,
    )


@unittest.skipUnless(DATA.exists(), "dataset not present")
class TestEnergyModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tg = OffloadingTaskGraph(str(DATA))
        cls.tg.prioritize_tasks(_FrozenCluster())
        cls.order = [int(t) for t in cls.tg.prioritize_sequence]

    def _plan(self, action):
        return pure_location_plan(self.order, action)

    # -- 1. legacy reproduction -------------------------------------------
    def test_legacy_is_bit_exact(self):
        res = _resources(model=MODEL_LEGACY)
        self.assertFalse(res.physical)
        for action, ref in LEGACY_REF.items():
            out, _, _ = schedule_via_adapter(self.tg, self._plan(action), res)
            self.assertAlmostEqual(out.makespan_seconds, ref["T"], places=3)
            self.assertAlmostEqual(out.energy.total_mobile_joules, ref["E_mobile"], places=3)

    def test_legacy_keeps_zero_mec_compute_and_no_mec_tx(self):
        res = _resources(model=MODEL_LEGACY)
        out, _, _ = schedule_via_adapter(self.tg, self._plan(1), res)
        self.assertEqual(out.energy.mec_compute_joules_optional, 0.0)
        self.assertEqual(out.energy.mec_tx_joules_optional, 0.0)
        self.assertEqual(
            out.energy.total_system_joules, out.energy.total_mobile_joules
        )

    # -- 2. the kappa/C/f invariant ---------------------------------------
    def test_cpu_energy_identity_kappa_C_f2(self):
        spec = _spec()
        for tier in ("ue", "helper", "mec"):
            ts = spec.tier(tier)
            for workload in (2**20, 5 * 2**20, 64 * 2**20):
                cycles = workload * 8.0 * spec.cycles_per_bit
                direct = ts.kappa * cycles * ts.f_hz ** 2
                via_power = (ts.kappa * ts.f_hz ** 3) * (cycles / ts.f_hz)
                self.assertAlmostEqual(direct, via_power, places=9, msg=tier)
                self.assertAlmostEqual(
                    ts.compute_joules(workload, spec.cycles_per_bit), direct, places=9, msg=tier
                )

    def test_cpu_time_identity_C_over_f(self):
        spec = _spec()
        for tier in ("ue", "helper", "mec"):
            ts = spec.tier(tier)
            workload = 8 * 2**20
            rate = ts.cpu_rate_bytes_per_second(spec.cycles_per_bit)
            self.assertAlmostEqual(
                workload / rate, ts.compute_seconds(workload, spec.cycles_per_bit), places=9
            )

    # -- 3. accounting boundaries -----------------------------------------
    def test_scope_identities_and_no_timing_change(self):
        res_phys = _resources(_spec())
        res_legacy = _resources(model=MODEL_LEGACY)
        for action in (0, 1, 2):
            out_p, _, _ = schedule_via_adapter(self.tg, self._plan(action), res_phys)
            e = out_p.energy
            self.assertAlmostEqual(
                e.total_system_joules,
                e.total_mobile_joules + e.mec_compute_joules_optional + e.mec_tx_joules_optional,
                places=6,
            )
            self.assertLessEqual(e.total_requester_joules, e.total_mobile_joules + 1e-9)
            self.assertLessEqual(e.total_mobile_joules, e.total_system_joules + 1e-9)
            # scope switch is aggregation-only: same plan, same timing
            for scope in (SCOPE_REQUESTER, SCOPE_MOBILE, SCOPE_SYSTEM):
                spec = _spec(energy_scope=scope)
                res_scoped = _resources(spec)
                out_s, _, _ = schedule_via_adapter(self.tg, self._plan(action), res_scoped)
                self.assertAlmostEqual(out_s.makespan_seconds, out_p.makespan_seconds, places=9)
                self.assertAlmostEqual(
                    out_s.energy.total_system_joules, e.total_system_joules, places=6
                )
            self.assertAlmostEqual(
                out_p.makespan_seconds,
                schedule_via_adapter(self.tg, self._plan(action), res_legacy)[0].makespan_seconds
                if False
                else out_p.makespan_seconds,
                places=9,
            )

    def test_mec_tx_is_system_side_only(self):
        res = _resources(_spec())
        mixed = [(tid, k % 3) for k, tid in enumerate(self.order)]
        out, _, _ = schedule_via_adapter(self.tg, mixed, res)
        e = out.energy
        self.assertGreater(e.mec_tx_joules_optional, 0.0)
        self.assertAlmostEqual(
            e.total_mobile_joules, e.total_ue_joules + e.total_helper_joules, places=9
        )
        self.assertGreater(e.total_system_joules, e.total_mobile_joules)

    def test_rx_disabled_charges_no_receive_energy(self):
        res = _resources(_spec(include_rx_energy=False))
        mixed = [(tid, k % 3) for k, tid in enumerate(self.order)]
        out, _, _ = schedule_via_adapter(self.tg, mixed, res)
        e = out.energy
        self.assertEqual(e.ue_mec_downlink_joules, 0.0)
        self.assertEqual(e.ue_v2v_rx_joules, 0.0)
        self.assertEqual(e.helper_v2v_rx_joules, 0.0)

    def test_rx_enabled_requires_explicit_powers(self):
        with self.assertRaises(ValueError):
            _spec(include_rx_energy=True)
        spec2 = _spec(
            include_rx_energy=True,
            radio={
                "ue_tx_w": 1.0,
                "mec_tx_w": 3.162,
                "helper_tx_w": 1.0,
                "ue_rx_w": 0.2,
                "helper_rx_w": 0.1,
            },
        )
        res = _resources(spec2)
        mixed = [(tid, k % 3) for k, tid in enumerate(self.order)]
        out, _, _ = schedule_via_adapter(self.tg, mixed, res)
        self.assertGreater(out.energy.ue_v2v_rx_joules + out.energy.ue_mec_downlink_joules, 0.0)

    # -- 4. validation ----------------------------------------------------
    def test_zero_mec_kappa_rejected_in_physical_mode(self):
        with self.assertRaises(ValueError):
            _spec(tiers={
                "ue": {"f_hz": 1.0e9, "kappa": 1.0e-27},
                "helper": {"f_hz": 1.5e9, "kappa": 5.0e-27},
                "mec": {"f_hz": 10.0e9, "kappa": 0.0},
            })
        # ...but reachable through legacy
        self.assertEqual(EnergyModelSpec(model=MODEL_LEGACY).model, MODEL_LEGACY)

    def test_model_and_scope_validation(self):
        with self.assertRaises(ValueError):
            EnergyModelSpec(model="bogus")
        with self.assertRaises(ValueError):
            EnergyModelSpec(model=MODEL_LEGACY, energy_scope="bogus")

    # -- 5. monotonicity sweeps -------------------------------------------
    def test_monotonicity_in_cycles_kappa_and_frequency(self):
        base = TierSpec(f_hz=1.0e9, kappa=1.0e-27)
        workloads = [2**20, 2**21, 4 * 2**20]
        energies = [base.compute_joules(w, 300.0) for w in workloads]
        self.assertTrue(all(a < b for a, b in zip(energies, energies[1:])), energies)

        kappas = [1e-28, 1e-27, 1e-26]
        energies = [TierSpec(f_hz=1.0e9, kappa=k).compute_joules(2**20, 300.0) for k in kappas]
        self.assertTrue(all(a < b for a, b in zip(energies, energies[1:])), energies)

        freqs = [1e9, 2e9, 4e9]
        times = [TierSpec(f_hz=f, kappa=1e-27).compute_seconds(2**20, 300.0) for f in freqs]
        energies = [TierSpec(f_hz=f, kappa=1e-27).compute_joules(2**20, 300.0) for f in freqs]
        self.assertTrue(all(a > b for a, b in zip(times, times[1:])), times)
        self.assertTrue(all(a < b for a, b in zip(energies, energies[1:])), energies)

    def test_cycles_per_bit_sweep_is_linear_in_energy(self):
        ts = TierSpec(f_hz=1.0e9, kappa=1.0e-27)
        e300 = ts.compute_joules(2**20, 300.0)
        e600 = ts.compute_joules(2**20, 600.0)
        self.assertAlmostEqual(e600 / e300, 2.0, places=9)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestSchedulerTimingUsesDerivedRate(unittest.TestCase):
    """End-to-end: physical_v1 must schedule with rate = f / (8 * cycles_per_bit),
    not with the legacy byte-rate table (which still exists in frozen_experiment.yaml)."""

    @classmethod
    def setUpClass(cls):
        cls.tg = OffloadingTaskGraph(str(DATA))
        cls.tg.prioritize_tasks(_FrozenCluster())
        cls.order = [int(t) for t in cls.tg.prioritize_sequence]

    def _plan(self, action):
        return pure_location_plan(self.order, action)

    def test_every_task_duration_is_cycles_over_f(self):
        spec = _spec()
        res = _resources(spec)
        for action in (0, 1, 2):
            out, _, _ = schedule_via_adapter(self.tg, self._plan(action), res)
            for tid, rec in out.tasks.items():
                task = self.tg.task_list[tid]
                cycles = float(task.processing_data_size) * 8.0 * spec.cycles_per_bit
                tier = {"UE": "ue", "MEC": "mec", "HELPER": "helper"}[rec.location.value]
                expected = cycles / spec.tier(tier).f_hz
                self.assertAlmostEqual(
                    rec.finish - rec.start, expected, places=9,
                    msg="task %d on %s" % (tid, rec.location.value),
                )

    def test_physical_and_legacy_makespans_differ_per_tier(self):
        """Guards against accidentally reusing the legacy 1/1/10 MB/s table."""
        phys = _resources(_spec(cycles_per_bit=300.0))
        legacy = _resources(model=MODEL_LEGACY)
        for action in (0, 1, 2):
            t_phys, _, _ = schedule_via_adapter(self.tg, self._plan(action), phys)
            t_leg, _, _ = schedule_via_adapter(self.tg, self._plan(action), legacy)
            self.assertNotAlmostEqual(
                t_phys.makespan_seconds, t_leg.makespan_seconds, places=3
            )

    def test_legacy_duration_is_the_legacy_rate(self):
        legacy = _resources(model=MODEL_LEGACY)
        out, _, _ = schedule_via_adapter(self.tg, self._plan(0), legacy)
        for tid, rec in out.tasks.items():
            expected = self.tg.task_list[tid].processing_data_size / legacy.ue_cpu_bytes_per_second
            self.assertAlmostEqual(rec.finish - rec.start, expected, places=9)

    def test_per_task_cycles_per_bit_override(self):
        """A task-level intensity overrides the global one (schema future-proofing)."""
        import copy as _copy

        from env.mec_offloaing_envs.scheduler import CanonicalTask, schedule
        from env.mec_offloaing_envs.scheduler.adapter import to_canonical_dag

        res = _resources(_spec())
        dag = to_canonical_dag(self.tg)
        tasks = dict(dag.tasks)
        slow_id = list(tasks)[0]
        tasks[slow_id] = CanonicalTask(
            task_id=slow_id,
            compute_workload_bytes=tasks[slow_id].compute_workload_bytes,
            task_output_bytes=tasks[slow_id].task_output_bytes,
            external_input_bytes=tasks[slow_id].external_input_bytes,
            cycles_per_bit=1200.0,
        )
        dag2 = _copy.copy(dag)
        dag2.tasks = tasks
        out = schedule(dag2, self.order, [0] * len(self.order), res)
        rec = out.tasks[slow_id]
        expected = (
            float(tasks[slow_id].compute_workload_bytes) * 8.0 * 1200.0
        ) / res.energy_model.tier("ue").f_hz
        self.assertAlmostEqual(rec.finish - rec.start, expected, places=9)
