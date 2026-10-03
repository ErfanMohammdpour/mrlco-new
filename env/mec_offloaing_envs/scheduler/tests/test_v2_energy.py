#!/usr/bin/env python3
"""v2 energy ledger: hand-checkable physics, boundaries, attribution and provenance.

The ledger is priced by the SAME frozen `physical_v1` implementation as v1
(`env/mec_offloaing_envs/scheduler/energy_model.py`), so every number here can be
reproduced by hand from the frozen constants:

    ue:     f = 1.0 GHz, kappa = 1e-27  -> P = kappa f^3 =        1.000 W
    helper: f = 1.5 GHz, kappa = 5e-27  -> P = kappa f^3 =       16.875 W
    mec:    f = 10  GHz, kappa = 1e-27  -> P = kappa f^3 =     1000.000 W
    ue_tx_w = 1.0 W, mec_tx_w = 3.162 W, helper_tx_w = 1.0 W, include_rx_energy = False
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spec.automotive_training.v2.energy import (  # noqa: E402
    UNMODELED, frozen_energy_spec, reference_ranges_from_plans, schedule_energy,
)
from spec.automotive_training.v2.helper_model import HelperState  # noqa: E402
from spec.automotive_training.v2.shared_scheduler import (  # noqa: E402
    V2ComputeSpec, V2DAGSpec, V2LinkSpec, V2TaskSpec, schedule_shared,
)

LINK = V2LinkSpec(20e6, 20e6, 10e6)
COMPUTE = V2ComputeSpec(10e6, 1, (1e6,), (4e6,))
P_UE, P_HELPER, P_MEC = 1.0, 16.875, 1000.0
#: a scoped ReferenceRanges requires a 64-hex scheduler fingerprint
SHA = "a" * 64


def _task(tid=0, compute=6e6, out=2e6, preds=(), root=True, sink=True, external=1000.0,
          crit="MEDIUM"):
    return V2TaskSpec(tid, compute, out, tuple(preds), root, sink, crit, None, 0,
                      external_input_bytes=external)


def _one(compute=6e6, out=2e6, external=1000.0):
    return V2DAGSpec("d", 0, [_task(compute=compute, out=out, external=external)])


HELPER_LONG = {0: HelperState(0, 4e6, contact_end_s=100.0, predicted_contact_end_s=100.0)}


class TestFrozenModelProvenance(unittest.TestCase):
    def test_the_frozen_configuration_is_the_physical_model(self):
        spec = frozen_energy_spec()
        self.assertEqual(spec.model, "physical_v1")
        self.assertEqual(spec.energy_scope, "system")
        self.assertAlmostEqual(spec.cycles_per_bit, 300.0)
        self.assertFalse(spec.include_rx_energy)

    def test_implied_dynamic_powers_match_the_hand_values(self):
        spec = frozen_energy_spec()
        self.assertAlmostEqual(spec.tiers["ue"].implied_dynamic_power_w, P_UE, places=9)
        self.assertAlmostEqual(spec.tiers["helper"].implied_dynamic_power_w, P_HELPER, places=9)
        self.assertAlmostEqual(spec.tiers["mec"].implied_dynamic_power_w, P_MEC, places=6)

    def test_unmodeled_components_are_declared(self):
        joined = " ".join(UNMODELED)
        for token in ("receiver", "idle", "static", "backhaul", "battery"):
            self.assertIn(token, joined)


class TestHandCalculableEnergy(unittest.TestCase):
    def test_all_mec_cpu_energy_is_power_times_duration(self):
        res = schedule_shared([_one()], {"d": {0: 1}}, link=LINK, compute=COMPUTE)
        led = schedule_energy(res)
        # 6 MB at 10 MB/s = 0.6 s of MEC CPU at 1000 W
        self.assertAlmostEqual(led.per_tier_cpu_seconds["mec"], 0.6, places=9)
        self.assertAlmostEqual(led.breakdown.mec_compute_joules_optional, 600.0, places=6)

    def test_all_mec_radio_energy_uses_active_service_and_the_transmitter_tier(self):
        res = schedule_shared([_one()], {"d": {0: 1}}, link=LINK, compute=COMPUTE)
        led = schedule_energy(res)
        b = led.breakdown
        # UE uploads 1000 B at 20 MB/s = 5e-5 s at 1 W -> the REQUESTER pays its own TX
        self.assertAlmostEqual(b.ue_mec_uplink_joules, 5e-5 * 1.0, places=12)
        # the RSU sends 2 MB at 20 MB/s = 0.1 s at 3.162 W -> system-side, not requester
        self.assertAlmostEqual(b.mec_tx_joules_optional, 0.1 * 3.162, places=9)
        self.assertAlmostEqual(led.requester_joules, 5e-5, places=12)
        self.assertAlmostEqual(led.system_joules, 600.0 + 0.1 * 3.162 + 5e-5, places=6)

    def test_all_ue_is_purely_requester_energy(self):
        res = schedule_shared([_one()], {"d": {0: 0}}, link=LINK, compute=COMPUTE)
        led = schedule_energy(res)
        # 6 MB at 1 MB/s = 6 s at 1 W
        self.assertAlmostEqual(led.requester_joules, 6.0, places=9)
        self.assertAlmostEqual(led.mobile_joules, 6.0, places=9)
        self.assertAlmostEqual(led.system_joules, 6.0, places=9)

    def test_all_helper_splits_between_requester_and_helper(self):
        dag = V2DAGSpec("d", 0, [_task()], helper_id=0)
        res = schedule_shared([dag], {"d": {0: 2}}, link=LINK, compute=COMPUTE,
                              helper_states=HELPER_LONG)
        led = schedule_energy(res)
        b = led.breakdown
        self.assertAlmostEqual(b.helper_compute_joules, 1.5 * P_HELPER, places=9)
        self.assertAlmostEqual(b.ue_v2v_tx_joules, 0.0001 * 1.0, places=12)
        self.assertAlmostEqual(b.helper_v2v_tx_joules, 0.2 * 1.0, places=9)
        self.assertAlmostEqual(led.requester_joules, 0.0001, places=12)
        self.assertAlmostEqual(led.mobile_joules, led.system_joules, places=9)
        self.assertAlmostEqual(led.mobile_joules,
                               1.5 * P_HELPER + 0.0001 + 0.2, places=9)

    def test_work_form_equals_duration_form_under_a_physical_allocation(self):
        # make the UE rate equal f/(8*cpb) = 1e9/2400 bytes/s, i.e. the allocation that is
        # CONSISTENT with the frequency in the energy model
        physical_rate = 1e9 / (8 * 300.0)
        compute = V2ComputeSpec(10e6, 1, (physical_rate,), (4e6,))
        res = schedule_shared([_one()], {"d": {0: 0}}, link=LINK, compute=compute)
        led = schedule_energy(res)
        self.assertAlmostEqual(led.cpu_duration_form_j, led.cpu_work_form_j, places=6)
        # E = kappa * C * f^2 with C = bytes * 8 * cycles_per_bit
        expected = 1e-27 * (6e6 * 8 * 300.0) * (1e9 ** 2)
        self.assertAlmostEqual(led.cpu_work_form_j, expected, places=6)

    def test_work_over_duration_reports_the_frozen_rate_mismatch(self):
        res = schedule_shared([_one()], {"d": {0: 1}}, link=LINK, compute=COMPUTE)
        led = schedule_energy(res)
        # scheduled 10 MB/s vs physical f/(8*cpb) = 4.1667 MB/s -> exactly 2.4
        self.assertAlmostEqual(led.cpu_work_over_duration, 10e6 / (10e9 / 2400.0), places=6)

    def test_more_workers_do_not_change_the_energy_of_the_same_work(self):
        # two INDEPENDENT DAGs on the shared MEC pool: 1 server serialises them, 2 servers run
        # them in parallel. The work (and therefore the energy) is identical either way:
        # occupancy is a queueing property, never a reduction of the physical clock frequency.
        dags = [V2DAGSpec("d0", 0, [_task()]), V2DAGSpec("d1", 0, [_task()])]
        plans = {"d0": {0: 1}, "d1": {0: 1}}
        one = schedule_shared(dags, plans, link=LINK, compute=COMPUTE)
        two = schedule_shared(dags, plans, link=LINK,
                              compute=V2ComputeSpec(10e6, 2, (1e6,), (4e6,)))
        e1 = schedule_energy(one).per_tier_cpu_seconds["mec"]
        e2 = schedule_energy(two).per_tier_cpu_seconds["mec"]
        self.assertAlmostEqual(e1, 1.2, places=9)
        self.assertAlmostEqual(e2, 1.2, places=9, msg="occupancy is not a power reduction")
        self.assertLess(two.makespan_s, one.makespan_s)


class TestBoundariesAndAttribution(unittest.TestCase):
    def test_boundaries_are_nested_and_never_overwritten(self):
        tasks = [_task(tid=0, compute=1e6, out=1e6, root=True, sink=False),
                 _task(tid=1, compute=1e6, out=2e6, preds=(0,), root=False, sink=True)]
        dag = V2DAGSpec("d", 0, tasks, helper_id=0)
        res = schedule_shared([dag], {"d": {0: 0, 1: 2}}, link=LINK, compute=COMPUTE,
                              helper_states=HELPER_LONG)
        led = schedule_energy(res)
        self.assertLessEqual(led.requester_joules, led.mobile_joules + 1e-12)
        self.assertLessEqual(led.mobile_joules, led.system_joules + 1e-12)
        self.assertAlmostEqual(led.system_joules,
                               led.breakdown.total_system_joules, places=12)
        self.assertAlmostEqual(led.mobile_joules,
                               led.breakdown.total_mobile_joules, places=12)
        self.assertAlmostEqual(led.requester_joules,
                               led.breakdown.total_requester_joules, places=12)

    def test_same_location_transfer_prices_to_zero_radio_energy(self):
        tasks = [_task(tid=0, compute=1e6, out=2e6, root=True, sink=False, external=0.0),
                 _task(tid=1, compute=1e6, out=2e6, preds=(0,), root=False, sink=True,
                       external=0.0)]
        dag = V2DAGSpec("d", 0, tasks)
        res = schedule_shared([dag], {"d": {0: 0, 1: 0}}, link=LINK, compute=COMPUTE)
        led = schedule_energy(res)
        for name in ("ue_mec_uplink_joules", "ue_mec_downlink_joules", "ue_v2v_tx_joules",
                     "ue_v2v_rx_joules", "helper_v2v_tx_joules", "helper_v2v_rx_joules",
                     "mec_tx_joules_optional"):
            self.assertEqual(getattr(led.breakdown, name), 0.0)
        self.assertAlmostEqual(led.system_joules, 2 * 1e6 / 1e6 * P_UE, places=9)

    def test_receiver_energy_is_off_and_reported_as_unmodeled(self):
        res = schedule_shared([_one()], {"d": {0: 1}}, link=LINK, compute=COMPUTE)
        led = schedule_energy(res)
        self.assertEqual(led.breakdown.ue_mec_downlink_joules, 0.0)
        self.assertEqual(led.breakdown.ue_v2v_rx_joules, 0.0)
        self.assertIn("receiver", " ".join(led.as_dict()["unmodeled"]))

    def test_background_energy_is_separated_and_sums_to_the_total(self):
        dags = [_one()]
        for b in range(2):
            dags.append(V2DAGSpec("d_bg%d" % b, 1 + b, [_task()]))
        plans = {d.dag_id: {0: 1} for d in dags}
        res = schedule_shared(dags, plans, link=LINK, compute=COMPUTE)
        led = schedule_energy(res, background_dag_ids=("d_bg0", "d_bg1"))
        self.assertGreater(led.background_joules, 0.0)
        self.assertGreater(led.foreground_joules, 0.0)
        self.assertAlmostEqual(led.foreground_joules + led.background_joules,
                               led.system_joules, places=9)
        for name in ("mec_compute_joules_optional", "ue_mec_uplink_joules",
                     "mec_tx_joules_optional"):
            self.assertAlmostEqual(
                getattr(led.foreground, name) + getattr(led.background, name),
                getattr(led.breakdown, name), places=12)


class TestFailureAndWastedEnergy(unittest.TestCase):
    def test_wasted_remote_work_and_the_local_restart_are_both_paid(self):
        # contact 1.6 s: the helper computes 1.5 s and the result cannot come back, so the
        # whole task restarts locally. Total CPU energy = helper attempt + UE restart.
        dag = V2DAGSpec("d", 0, [_task()], helper_id=0)
        hel = {0: HelperState(0, 4e6, contact_end_s=1.6, predicted_contact_end_s=1.84)}
        res = schedule_shared([dag], {"d": {0: 2}}, link=LINK, compute=COMPUTE,
                              helper_states=hel)
        self.assertEqual(res.queue_stats["helper_contact_failures"], 1)
        led = schedule_energy(res)
        self.assertAlmostEqual(led.breakdown.helper_compute_joules, 1.5 * P_HELPER, places=9)
        self.assertAlmostEqual(led.breakdown.ue_local_cpu_joules, 6.0 * P_UE, places=9)
        self.assertAlmostEqual(led.system_joules,
                               1.5 * P_HELPER + 6.0 * P_UE + 0.0001, places=9)
        self.assertAlmostEqual(led.per_tier_cpu_seconds["helper"], 1.5, places=9)
        self.assertAlmostEqual(led.per_tier_cpu_seconds["ue"], 6.0, places=9)

    def test_no_energy_is_free_after_a_contact_failure(self):
        dag = V2DAGSpec("d", 0, [_task()], helper_id=0)
        hel = {0: HelperState(0, 4e6, contact_end_s=1.6, predicted_contact_end_s=1.84)}
        res = schedule_shared([dag], {"d": {0: 2}}, link=LINK, compute=COMPUTE,
                              helper_states=hel)
        led = schedule_energy(res)
        self.assertGreater(led.requester_joules, 0.0)
        self.assertGreater(led.mobile_joules, 0.0)
        self.assertTrue(all(math.isfinite(v) for k, v in led.breakdown.as_dict().items()
                            if not k.startswith("total_")))


class TestReferenceRanges(unittest.TestCase):
    def test_reference_ranges_come_from_v2_pure_location_plans(self):
        dag = V2DAGSpec("d", 0, [_task()], helper_id=0)
        plans = {
            "all_ue": schedule_shared([dag], {"d": {0: 0}}, link=LINK, compute=COMPUTE),
            "all_mec": schedule_shared([dag], {"d": {0: 1}}, link=LINK, compute=COMPUTE),
            "all_helper": schedule_shared([dag], {"d": {0: 2}}, link=LINK, compute=COMPUTE,
                                          helper_states=HELPER_LONG),
        }
        refs = reference_ranges_from_plans(plans, scheduler_config_sha256=SHA)
        self.assertEqual(refs.energy_scope, "system")
        self.assertEqual(refs.scheduler_config_sha256, SHA)
        self.assertAlmostEqual(refs.E_ue, 6.0, places=9)
        self.assertAlmostEqual(refs.E_mec, 600.0 + 0.1 * 3.162 + 5e-5, places=6)
        self.assertGreater(refs.E_mec, refs.E_ue)
        self.assertAlmostEqual(refs.L_ue, 6.0, places=9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
