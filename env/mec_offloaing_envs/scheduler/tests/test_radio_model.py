#!/usr/bin/env python3
"""④ RADIO_MODEL_V1 tests + the mandated ③ re-run under the new radio rates.

Radio changes transfer_time -> changes deadline feasibility, so after switching
the radio model every ③ guarantee is re-verified here:

    LB admissibility · mask soundness · suffix feasibility · terminal potential
"""

from __future__ import annotations

import itertools
import math
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

from env.mec_offloaing_envs.scheduler import (  # noqa: E402
    CanonicalDAG,
    CanonicalTask,
    ReferenceRanges,
    ResourceConfig,
    schedule,
)
from env.mec_offloaing_envs.scheduler.feasibility import transfer_lower_bound  # noqa: E402
from env.mec_offloaing_envs.scheduler.model import Location  # noqa: E402
from env.mec_offloaing_envs.scheduler.objective import (  # noqa: E402
    LATENCY_REF_ALL_UE,
    ObjectiveSpec,
    evaluate_plan_objective,
)
from env.mec_offloaing_envs.scheduler.radio import (  # noqa: E402
    LINK_V2I_DL,
    LINK_V2I_UL,
    LINK_V2V,
    RADIO_LEGACY,
    RADIO_PHYSICAL,
    RATE_MODEL_ETA,
    RATE_MODEL_SINR,
    RadioLinkSpec,
    RadioModelSpec,
    effective_rate_bps,
    transfer_time_seconds,
)
from env.mec_offloaing_envs.scheduler.suffix import (  # noqa: E402
    ObjectiveContext,
    SuffixContext,
    TaskDeadline,
    construct_fastest_feasible_suffix,
    dag_lower_bound_ready,
    evaluate_action,
    evaluate_all_actions,
    suffix_potential,
)

WORKLOAD = 4_166_700
BIG_OUTPUT = 4_166_700
SMALL = 2048


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _radio_doc(model=RADIO_PHYSICAL, bw=10.0e6, eta=1.5, v2v_eta=None):
    return {
        "model": model,
        "links": {
            LINK_V2I_UL: {"bandwidth_hz": bw, "rate_model": RATE_MODEL_ETA,
                          "spectral_efficiency": eta, "source": "test"},
            LINK_V2I_DL: {"bandwidth_hz": bw, "rate_model": RATE_MODEL_ETA,
                          "spectral_efficiency": eta, "source": "test"},
            LINK_V2V: {"bandwidth_hz": bw, "rate_model": RATE_MODEL_ETA,
                       "spectral_efficiency": eta if v2v_eta is None else v2v_eta,
                       "assumption": True, "source": "test"},
        },
    }


def _resources(radio_model="legacy", energy_model="legacy"):
    """ResourceConfig with selectable radio/energy model (both default legacy)."""
    from env.mec_offloaing_envs.scheduler.energy_model import MODEL_PHYSICAL, EnergyModelSpec

    base = ResourceConfig.from_frozen_yaml(model=energy_model)
    radio = (
        RadioModelSpec.from_dict(_radio_doc())
        if radio_model == RADIO_PHYSICAL
        else RadioModelSpec(model=RADIO_LEGACY)
    )
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
        f_v2v=base.f_v2v,
        energy_model=base.energy_model,
        radio_model=radio,
    )


def _chain(n=3, deadlines=None, dtype="hard", heavy_first=True):
    deadlines = deadlines or {}
    tasks = [
        CanonicalTask(
            task_id=i,
            compute_workload_bytes=WORKLOAD if (i == 0 and heavy_first) else SMALL,
            task_output_bytes=BIG_OUTPUT if i == 0 else SMALL,
            deadline_s=deadlines.get(i),
            deadline_type=dtype if i in deadlines else "none",
        )
        for i in range(n)
    ]
    edges = [(i, i + 1, BIG_OUTPUT if i == 0 else SMALL) for i in range(n - 1)]
    return CanonicalDAG.from_records(tasks, edges)


def _refs(dag, res, order):
    metrics = {}
    for action in (0, 1, 2):
        out = schedule(dag, order, [action] * len(order), res)
        metrics[action] = (out.makespan_seconds, out.energy.total_mobile_joules)
    return ReferenceRanges(
        L_ue=metrics[0][0], L_mec=metrics[1][0], L_helper=metrics[2][0],
        E_ue=metrics[0][1], E_mec=metrics[1][1], E_helper=metrics[2][1],
    )


# ---------------------------------------------------------------------------
# radio model unit tests
# ---------------------------------------------------------------------------
class TestRadioModel(unittest.TestCase):
    def test_legacy_rates_are_byte_exact(self):
        res = _resources(radio_model="legacy")
        cfg = ResourceConfig.from_frozen_yaml(model="legacy")
        for hop in ("MEC_UL", "MEC_DL", "V2V"):
            self.assertAlmostEqual(res.hop_rate(hop), cfg.hop_rate(hop), places=9)
        self.assertAlmostEqual(res.hop_rate("MEC_UL"), 917_504.0, places=6)
        self.assertAlmostEqual(res.hop_rate("V2V"), 655_360.0, places=6)

    def test_eta_rate_is_bandwidth_times_efficiency(self):
        spec = RadioModelSpec.from_dict(_radio_doc(bw=10.0e6, eta=1.5))
        for link in (LINK_V2I_UL, LINK_V2I_DL, LINK_V2V):
            self.assertAlmostEqual(spec.link(link).effective_rate_bps, 15.0e6, places=3)
            self.assertAlmostEqual(
                spec.link(link).effective_rate_bytes_per_second, 15.0e6 / 8.0, places=6
            )

    def test_no_hidden_one_bit_per_hz(self):
        """B/8 is NOT a rate: the documented conversion must include eta."""
        b = 10.0e6
        eta = 3.0
        spec = RadioModelSpec.from_dict(_radio_doc(bw=b, eta=eta))
        rate_bytes = spec.rate_bytes_per_second(LINK_V2I_UL)
        self.assertAlmostEqual(rate_bytes, b * eta / 8.0, places=6)
        self.assertNotAlmostEqual(rate_bytes, b / 8.0, places=3)
        # doubling eta doubles the rate; doubling B doubles the rate
        self.assertAlmostEqual(
            RadioModelSpec.from_dict(_radio_doc(bw=b, eta=2 * eta)).rate_bytes_per_second(LINK_V2I_UL),
            2.0 * rate_bytes, places=6,
        )
        self.assertAlmostEqual(
            RadioModelSpec.from_dict(_radio_doc(bw=2 * b, eta=eta)).rate_bytes_per_second(LINK_V2I_UL),
            2.0 * rate_bytes, places=6,
        )

    def test_sinr_form_matches_formula(self):
        spec = RadioModelSpec.from_dict(
            {
                "model": RADIO_PHYSICAL,
                "links": {
                    name: {"bandwidth_hz": 10.0e6, "rate_model": RATE_MODEL_SINR,
                           "sinr_db": 10.0, "source": "test"}
                    for name in (LINK_V2I_UL, LINK_V2I_DL, LINK_V2V)
                },
            }
        )
        expected = 10.0e6 * math.log2(1.0 + 10.0)
        self.assertAlmostEqual(spec.rate_bytes_per_second(LINK_V2I_UL), expected / 8.0, places=3)
        self.assertAlmostEqual(effective_rate_bps(10.0e6, sinr_db=10.0), expected, places=3)

    def test_monotonicity(self):
        rates = [
            effective_rate_bps(bw, eta=1.5) for bw in (5.0e6, 10.0e6, 20.0e6)
        ]
        self.assertEqual(rates, sorted(rates))
        etas = [effective_rate_bps(10.0e6, eta=e) for e in (1.0, 1.5, 2.0, 4.0)]
        self.assertEqual(etas, sorted(etas))
        times = [transfer_time_seconds(n, 1.875e6) for n in (1024, 2048, 4096)]
        self.assertEqual(times, sorted(times))

    def test_validation_and_assumptions(self):
        with self.assertRaises(ValueError):
            RadioLinkSpec(bandwidth_hz=10e6, rate_model=RATE_MODEL_ETA)  # no eta
        with self.assertRaises(ValueError):
            RadioLinkSpec(bandwidth_hz=10e6, rate_model=RATE_MODEL_SINR)  # no sinr
        with self.assertRaises(ValueError):
            RadioLinkSpec(bandwidth_hz=0.0, rate_model=RATE_MODEL_ETA, spectral_efficiency=1.5)
        with self.assertRaises(ValueError):
            RadioLinkSpec(bandwidth_hz=10e6, rate_model="bogus", spectral_efficiency=1.5)
        with self.assertRaises(ValueError):
            RadioModelSpec(model="bogus")
        with self.assertRaises(ValueError):
            RadioModelSpec.from_dict({"model": RADIO_PHYSICAL, "links": {}})
        with self.assertRaises(KeyError):
            RadioModelSpec.from_dict(_radio_doc()).link("nope")
        self.assertEqual(RadioModelSpec.from_dict(_radio_doc()).assumptions(), [LINK_V2V])

    def test_frozen_yaml_radio_block_loads_with_provenance(self):
        spec = RadioModelSpec.from_frozen_yaml()
        self.assertEqual(spec.model, RADIO_PHYSICAL)
        for link in (LINK_V2I_UL, LINK_V2I_DL, LINK_V2V):
            self.assertTrue(spec.link(link).source not in ("", "unspecified"))
        # the assumed V2V link is flagged, not hidden
        self.assertIn(LINK_V2V, spec.assumptions())


# ---------------------------------------------------------------------------
# ③ re-run under the physical radio model
# ---------------------------------------------------------------------------
class TestSuffixGuaranteesUnderPhysicalRadio(unittest.TestCase):
    def setUp(self):
        self.res_legacy = _resources(radio_model="legacy", energy_model="legacy")
        self.res_phys = _resources(radio_model=RADIO_PHYSICAL, energy_model="legacy")

    def _ctx(self, dag, res, order, deadlines=None, index=0, decisions=None, prefix=None):
        dl = {
            int(t): TaskDeadline(deadline_s=float(d), deadline_type="hard")
            for t, d in (deadlines or {}).items()
        }
        obj = ObjectiveContext(
            latency_ref_s=_refs(dag, res, order).L_ue, deadlines=dl
        )
        return SuffixContext(
            dag=dag, resources=res, order=order, objective=obj, index=index,
            decisions=decisions or {}, prefix_finish=prefix or {},
        )

    def test_transfer_lower_bound_uses_the_new_rate(self):
        nbytes = 1_875_000
        phys = transfer_lower_bound(nbytes, Location.MEC, Location.UE, self.res_phys)
        self.assertAlmostEqual(phys, 1.0, places=6)          # 1.875e6 B/s -> 1 s
        legacy = transfer_lower_bound(nbytes, Location.MEC, Location.UE, self.res_legacy)
        self.assertGreater(legacy, phys)                     # legacy is slower

    def test_lb_admissible_under_physical_radio(self):
        dag = _chain(3)
        order = [0, 1, 2]
        ctx = self._ctx(dag, self.res_phys, order)
        for action in (0, 1, 2):
            lb = dag_lower_bound_ready(ctx, action)
            for suffix in itertools.product((0, 1, 2), repeat=2):
                out = schedule(dag, order, [action, suffix[0], suffix[1]], self.res_phys)
                for tid in order:
                    achieved = out.tasks[tid].availability_seconds
                    self.assertLessEqual(
                        lb[tid], achieved + 1e-9 * max(1.0, abs(achieved)),
                        msg="LB above achieved: action %d suffix %s task %d" % (action, suffix, tid),
                    )

    def test_mask_soundness_under_physical_radio(self):
        dag = _chain(3, deadlines={1: 0.5})
        order = [0, 1, 2]
        for index in (0,):
            ctx = self._ctx(dag, self.res_phys, order, deadlines={1: 0.5}, index=index)
            for ev in evaluate_all_actions(ctx):
                if ev.mask:
                    continue
                for suffix in itertools.product((0, 1, 2), repeat=2):
                    out = schedule(dag, order, [ev.action, suffix[0], suffix[1]], self.res_phys)
                    self.assertGreater(
                        out.tasks[1].tardiness_s, 0.0,
                        msg="masked action %d rescued by suffix %s" % (ev.action, suffix),
                    )

    def test_suffix_feasibility_and_potential_under_physical_radio(self):
        dag = _chain(3, deadlines={2: 50.0}, dtype="soft")
        order = [0, 1, 2]
        ctx = self._ctx(dag, self.res_phys, order, index=0)
        s = construct_fastest_feasible_suffix(ctx, 1)
        self.assertTrue(s.found, s.reason)
        self.assertGreater(s.estimated_latency_s, 0.0)
        realized = schedule(dag, order, [a for _t, a in s.plan], self.res_phys)
        self.assertAlmostEqual(
            suffix_potential(realized, ctx.objective), s.estimated_J, places=9
        )
        # terminal potential equals the actual objective under the new radio
        refs = _refs(dag, self.res_phys, order)
        spec = ObjectiveSpec(latency_ref=LATENCY_REF_ALL_UE, energy_budget_j=1e12)
        actual = evaluate_plan_objective(realized, refs, spec)
        self.assertAlmostEqual(s.estimated_J, actual.J, places=6)

    def test_radio_switch_changes_feasibility_but_not_soundness(self):
        """A deadline can flip between models; the mask must stay sound in both."""
        dag = _chain(3, deadlines={1: 3.0})
        order = [0, 1, 2]
        results = {}
        for name, res in (("legacy", self.res_legacy), ("physical", self.res_phys)):
            ctx = self._ctx(dag, res, order, deadlines={1: 3.0}, index=0)
            evals = evaluate_all_actions(ctx)
            results[name] = [e.mask for e in evals]
            for ev in evals:
                if ev.mask:
                    continue
                for suffix in itertools.product((0, 1, 2), repeat=2):
                    out = schedule(dag, order, [ev.action, suffix[0], suffix[1]], res)
                    self.assertGreater(
                        out.tasks[1].tardiness_s, 0.0,
                        msg="%s: masked action %d rescued" % (name, ev.action),
                    )
        # the faster radio may keep more actions open, never fewer
        self.assertGreaterEqual(
            sum(results["physical"]), sum(results["legacy"])
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
