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
        # E4.2: give the plan-level objective channel its per-graph input from the
        # SAME rollout results (env stores them; no second scheduler run).
        payload = getattr(self.env, "validation_plan_payload", None)
        if callable(payload):
            plans = payload()
            if plans:
                metrics["validation_per_graph_plans"] = list(plans)
                identities = getattr(self.env, "validation_plan_identities", None)
                if callable(identities):
                    metrics["validation_plan_identities"] = identities()
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
        out = {
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
        return out
