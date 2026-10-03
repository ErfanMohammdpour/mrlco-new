#!/usr/bin/env python3
"""Stronger deterministic placement search for headroom analysis (analysis only).

This is NOT a global optimum and must never be reported as one. It is a bounded
multi-start coordinate descent plus iterated local search whose evaluation budget is
recorded, so results are labelled **candidate-search lower bound**.

Budget accounting: every call to `objective` is counted; the search stops at
`budget` evaluations or when no single-token move improves the incumbent.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Sequence


@dataclass
class SearchResult:
    plan: dict
    objective: float
    evaluations: int
    starts: int
    improving_moves: int
    convergence: list = field(default_factory=list)
    label: str = "candidate-search lower bound"


def _coordinate_descent(plan: dict, tokens: Sequence[int], objective: Callable,
                        counter: list, convergence: list, rng: random.Random,
                        max_moves: int = 200) -> tuple:
    best = dict(plan)
    best_value = objective(best)
    counter[0] += 1
    convergence.append(best_value)
    moves = 0
    improved = True
    order = list(tokens)
    while improved and moves < max_moves:
        improved = False
        rng.shuffle(order)
        for tok in order:
            current = best[tok]
            for action in (0, 1, 2):
                if action == current:
                    continue
                trial = dict(best)
                trial[tok] = action
                value = objective(trial)
                counter[0] += 1
                if value < best_value - 1e-12:
                    best, best_value, improved = trial, value, True
                    moves += 1
                    convergence.append(value)
                    break
    return best, best_value, moves


def stronger_search(tokens: Sequence[int], objective: Callable, *,
                    starts: int = 4, seed: int = 0, budget: int = 4000,
                    ils_rounds: int = 2) -> SearchResult:
    """Multi-start coordinate descent + a few iterated-local-search kicks."""
    rng = random.Random(int(seed))
    counter = [0]
    convergence: list = []
    best_plan, best_value = None, float("inf")
    moves_total = 0

    def wrap(plan: dict) -> float:
        if counter[0] >= budget:
            return float("inf")
        return float(objective(plan))

    for start in range(max(1, int(starts))):
        if counter[0] >= budget:
            break
        if start == 0:
            plan = {tok: 1 for tok in tokens}          # all-MEC start
        else:
            plan = {tok: rng.choice((0, 1, 2)) for tok in tokens}
        plan, value, moves = _coordinate_descent(plan, tokens, wrap, counter, convergence, rng)
        moves_total += moves
        if value < best_value:
            best_plan, best_value = plan, value
    # iterated local search: kick a few tokens and re-descend
    for _ in range(max(0, int(ils_rounds))):
        if counter[0] >= budget or best_plan is None:
            break
        kicked = dict(best_plan)
        for tok in rng.sample(list(tokens), k=min(3, len(tokens))):
            kicked[tok] = rng.choice((0, 1, 2))
        kicked, value, moves = _coordinate_descent(kicked, tokens, wrap, counter, convergence, rng)
        moves_total += moves
        if value < best_value:
            best_plan, best_value = kicked, value
    if best_plan is None:
        best_plan = {tok: 1 for tok in tokens}
        best_value = objective(best_plan)
        counter[0] += 1
    return SearchResult(plan=best_plan, objective=best_value, evaluations=counter[0],
                        starts=int(starts), improving_moves=moves_total,
                        convergence=convergence)
