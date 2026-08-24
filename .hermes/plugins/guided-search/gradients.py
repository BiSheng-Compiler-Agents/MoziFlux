"""Transition-derived gradient estimates for guided MAP-Elites."""

from __future__ import annotations

import math
from itertools import product
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

ORDINAL_DIMENSIONS = ("algorithm", "engine", "memory", "dispatch")


def estimate_gradients(
    transitions: Iterable[dict[str, Any]],
    *,
    decay: float = 0.95,
    parent: dict[str, Any] | None = None,
    elites: Iterable[dict[str, Any]] = (),
    gradient_weights: tuple[float, float, float] = (0.4, 0.4, 0.2),
) -> dict[str, dict[str, float]]:
    """Calculate and combine all gradient components."""
    records = list(transitions)
    if records and all(
            record.get("created_at") is not None for record in records):
        records = sorted(
            records,
            key=lambda record: (
                str(record["created_at"]),
                int(record.get("id") or 0),
            ),
        )
    elif records and all(record.get("id") is not None for record in records):
        records = sorted(records, key=lambda record: int(record["id"]))
    weighted_delta: dict[str, float] = defaultdict(float)
    transition_weights: dict[str, float] = defaultdict(float)
    positive_improvements: dict[str, list[int]] = defaultdict(list)
    negative_improvements: dict[str, list[int]] = defaultdict(list)
    total = len(records)
    for index, transition in enumerate(records):
        age = total - index - 1
        temporal = decay**age
        delta_fitness = float(transition.get("delta_fitness") or 0.0)
        transition_parent = transition.get("parent", {})
        child = transition.get("child", {})
        improved = int(bool(transition.get("improved", delta_fitness > 0)))
        for dimension in ORDINAL_DIMENSIONS:
            movement = (int(child.get(dimension, 0)) -
                        int(transition_parent.get(dimension, 0)))
            if movement == 0:
                continue
            direction = 1.0 if movement > 0 else -1.0
            weighted_delta[dimension] += temporal * delta_fitness * direction
            transition_weights[dimension] += temporal
            target = positive_improvements if movement > 0 else negative_improvements
            target[dimension].append(improved)

    exploration = {dimension: 0.0 for dimension in ORDINAL_DIMENSIONS}
    if parent is not None:
        mechanism = str(parent.get("mechanism", "standard_triton"))
        occupied: dict[tuple[int, ...], float] = {}
        for elite in elites:
            if str(elite.get("mechanism")) != mechanism:
                continue
            coordinate = tuple(
                int(elite.get(dimension, 0))
                for dimension in ORDINAL_DIMENSIONS)
            quality = elite.get("hardware_score")
            if quality is None:
                quality = elite.get("simulation_score")
            occupied[coordinate] = max(occupied.get(coordinate, 0.0),
                                       float(quality or 0.0))
        max_quality = max(occupied.values(), default=0.0)
        origin = tuple(
            int(parent.get(dimension, 0)) for dimension in ORDINAL_DIMENSIONS)
        for cell in product(range(4), repeat=len(ORDINAL_DIMENSIONS)):
            distance = sum(
                abs(cell[index] - origin[index])
                for index in range(len(origin)))
            if distance == 0:
                continue
            if cell in occupied:
                potential = (max(0.0, 1.0 - occupied[cell] / max_quality)
                             if max_quality > 0 else 0.0)
            else:
                potential = 1.0
            for index, dimension in enumerate(ORDINAL_DIMENSIONS):
                exploration[dimension] += (potential *
                                           (cell[index] - origin[index]) /
                                           distance**2)
        scale = max((abs(value) for value in exploration.values()),
                    default=0.0)
        if scale > 0:
            exploration = {
                dimension: value / scale
                for dimension, value in exploration.items()
            }

    result: dict[str, dict[str, float]] = {}
    for dimension in ORDINAL_DIMENSIONS:
        support = len(positive_improvements[dimension]) + len(
            negative_improvements[dimension])
        confidence = support / (support + 3.0)
        fitness = (weighted_delta[dimension] / transition_weights[dimension]
                   if transition_weights[dimension] else 0.0)
        positive_rate = (sum(positive_improvements[dimension]) /
                         len(positive_improvements[dimension])
                         if positive_improvements[dimension] else 0.0)
        negative_rate = (sum(negative_improvements[dimension]) /
                         len(negative_improvements[dimension])
                         if negative_improvements[dimension] else 0.0)
        rate = positive_rate - negative_rate
        fitness_component = math.tanh(fitness)
        exploration_component = float(exploration[dimension])
        combined = (confidence * (gradient_weights[0] * fitness_component +
                                  gradient_weights[1] * rate) +
                    gradient_weights[2] * exploration_component)
        result[dimension] = {
            "fitness": fitness,
            "fitness_normalized": fitness_component,
            "improvement_rate": rate,
            "exploration": exploration_component,
            "confidence": confidence,
            "combined": combined,
            "support": float(support),
        }
    return result
