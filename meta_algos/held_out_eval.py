"""Held-out support-adapt / query-report. Copies core into a scratch policy."""

from __future__ import annotations

import numpy as np

from meta_algos.variable_io import assign_trainable
from spec.eval_protocol import query_metrics_from_samples, require_sliced_task
from spec.split_loader import support_query_tasks


class HeldOutQueryEvaluator:
    """Adapt on support only, report query only. Does not mutate source_policy."""

    def __init__(self, env, policy, sampler, processor, ppo, source_policy, ppo_batch_size=20):
        self.env = env
        self.policy = policy
        self.sampler = sampler
        self.processor = processor
        self.ppo = ppo
        self.source_policy = source_policy
        self.ppo_batch_size = int(ppo_batch_size)
        if getattr(env, "distribution_ids", None) is None:
            raise ValueError("held-out env must expose distribution_ids")

    def _activate(self, task):
        self.env.set_task(require_sliced_task(task))

    def evaluate_one(self, env_index, distribution_id, k_steps, sess=None):
        k_steps = int(k_steps)
        if k_steps not in (0, 3):
            raise ValueError("held-out k_steps must be 0 or 3")
        assign_trainable(self.source_policy, self.policy, sess=sess)
        support_task, query_task = support_query_tasks(env_index, distribution_id)
        if k_steps > 0:
            self._activate(support_task)
            support_paths = self.sampler.obtain_samples(log=False, log_prefix="")
            support_data = self.processor.process_samples(support_paths, log=False, log_prefix="")
            self.ppo.UpdatePPOTarget(support_data, batch_size=self.ppo_batch_size, k_steps=k_steps)
        self._activate(query_task)
        query_paths = self.sampler.obtain_samples(log=False, log_prefix="query_")
        query_data = self.processor.process_samples(query_paths, log=False, log_prefix="query_")
        greedy = self.env.greedy_solution_for_current_task()
        metrics = query_metrics_from_samples(query_data)
        metrics["k_steps"] = k_steps
        metrics["distribution_id"] = int(distribution_id)
        # P1 baseline: all-MEC latency under the SAME scheduler/config (baseline
        # anchor, not a replay of the rollout), so the pilot can report the gap.
        all_mec = getattr(self.env, "all_mec_latency_for_current_task", None)
        if callable(all_mec):
            metrics["query_all_mec_latency"] = float(np.mean(all_mec()))
        # E4.2: give the plan-level objective channel its per-graph input from the
        # SAME rollout results (env stores them; no second scheduler run).
        plans = getattr(self.env, "validation_plan_payload", None)
        if callable(plans):
            payload = plans()
            if payload:
                metrics["validation_per_graph_plans"] = list(payload)
                identities = getattr(self.env, "validation_plan_identities", None)
                if callable(identities):
                    metrics["validation_plan_identities"] = identities()
        # P1/P1b read-only evaluation dashboard: action mix, entropy and the
        # plan summary computed from the SAME query samples/results.
        try:
            from spec.pilot_metrics import action_fractions, flatten_actions, plan_summary
        except Exception:
            action_fractions = flatten_actions = plan_summary = None
        if action_fractions is not None:
            for name, value in action_fractions(flatten_actions([query_data])).items():
                metrics["query_action_fraction/%s" % name] = float(value)
            acc = query_data.get("mask_accumulator") if hasattr(query_data, "get") else None
            if acc is not None:
                try:
                    from env.mec_offloaing_envs.scheduler.mask_metrics import rates

                    metrics["query_entropy_valid"] = float(rates(acc)["policy/entropy_valid"])
                except Exception:
                    pass
            payload = metrics.get("validation_per_graph_plans")
            identities = metrics.get("validation_plan_identities") or []
            if payload and identities and len(payload) == len(identities):
                rows = [
                    plan_summary(res, ident.get("edges"), len(ident.get("order", [])))
                    for (res, _refs), ident in zip(payload, identities)
                ]
                for key in rows[0]:
                    metrics["query_%s" % key] = sum(r[key] for r in rows) / float(len(rows))
        if self.env.resource_cluster.use_energy:
            _, greedy_latency, greedy_energy = greedy
            metrics["query_greedy_latency"] = float(np.mean(greedy_latency))
            metrics["query_greedy_energy"] = float(np.mean(greedy_energy))
        else:
            _, greedy_latency = greedy
            metrics["query_greedy_latency"] = float(np.mean(greedy_latency))
        return metrics

    def evaluate_all(self, k_steps, sess=None):
        rows = []
        for env_index, dist_id in enumerate(self.env.distribution_ids):
            rows.append(self.evaluate_one(env_index, dist_id, k_steps, sess=sess))
        composite = float(np.mean([row["validation_query_composite_objective"] for row in rows]))
        latency = float(np.mean([row["query_mean_latency"] for row in rows]))
        # The criterion (discounted return) is aggregated on the SAME rows; the
        # legacy undiscounted mean stays as a labelled companion only.
        objective = float(np.mean([row["query_discounted_return"] for row in rows]))
        legacy_sum = float(np.mean([row["query_legacy_undiscounted_sum"] for row in rows]))
        out = {
            "query_discounted_return": objective,
            "query_legacy_undiscounted_sum": legacy_sum,
            "validation_query_composite_objective": composite,
            "query_mean_latency": latency,
            "k_steps": int(k_steps),
            "n_distributions": len(rows),
            "per_distribution": rows,
        }
        # hoist the per-graph plans so the trainer's objective channel sees them
        plans = []
        identities = []
        for row in rows:
            plans.extend(row.get("validation_per_graph_plans", ()))
            identities.extend(row.get("validation_plan_identities", ()))
        if plans:
            out["validation_per_graph_plans"] = plans
        if identities:
            out["validation_plan_identities"] = identities
        if all("query_mean_energy" in row for row in rows):
            out["query_mean_energy"] = float(np.mean([row["query_mean_energy"] for row in rows]))
        if all("query_all_mec_latency" in row for row in rows):
            out["query_all_mec_latency"] = float(
                np.mean([row["query_all_mec_latency"] for row in rows])
            )
        if all("query_greedy_latency" in row for row in rows):
            out["query_greedy_latency"] = float(
                np.mean([row["query_greedy_latency"] for row in rows])
            )
        dashboard_keys = [
            k for k in rows[0]
            if k.startswith("query_action_fraction/")
            or k == "query_entropy_valid"
            or k in (
                "query_task_fraction/local", "query_task_fraction/mec", "query_task_fraction/v2v",
                "query_co_location_rate", "query_cross_location_edges", "query_total_edges",
                "query_utilization_mean", "query_utilization_max",
            )
        ]
        for key in dashboard_keys:
            if all(key in row for row in rows):
                out[key] = float(np.mean([row[key] for row in rows]))
        return out
