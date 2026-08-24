"""Parent and target selection policies for sparse MAP-Elites archives."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import math
import random
from typing import Any, Iterable

from .gradients import estimate_gradients
from .descriptors import annotate_target

_DIMENSIONS = ("algorithm", "engine", "memory", "dispatch")
_MECHANISMS = (
    "standard_triton",
    "compiler_managed",
    "extension_hinted",
    "manual_extension",
    "native_hybrid",
)


def _gradient_inputs_for_parent(
        parent: dict[str, Any],
        transitions: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    inputs = []
    for transition in transitions:
        parent_json = transition.get("parent_json")
        child_json = transition.get("child_json")
        if not parent_json or not child_json:
            continue
        if any(
                parent_json.get(dimension) != parent.get(dimension)
                for dimension in (*_DIMENSIONS, "mechanism")):
            continue
        inputs.append({
            "parent": parent_json,
            "child": child_json,
            "delta_fitness": transition.get("delta_fitness"),
            "improved": (transition.get("delta_fitness") or 0) > 0,
            "created_at": transition.get("created_at"),
            "id": transition.get("id"),
        })
    return inputs


def select_parent(
    candidates: Iterable[dict[str, Any]],
    transitions: Iterable[dict[str, Any]],
    *,
    generation: int,
    run_id: int = 0,
    archive_revision: int = 0,
) -> tuple[dict[str, Any] | None, str]:
    """Sample a parent using its KernelFoundry combined-gradient magnitude."""
    eligible = sorted(
        (c for c in candidates if c.get("correct") and c.get("safe")),
        key=lambda candidate: int(candidate["id"]),
    )
    if not eligible:
        return None, "seed"
    transitions = list(transitions)
    weighted: list[tuple[dict[str, Any], float, dict[str, dict[str,
                                                               float]]]] = []
    for candidate in eligible:
        gradients = estimate_gradients(
            _gradient_inputs_for_parent(candidate, transitions),
            parent=candidate,
            elites=eligible,
            gradient_weights=(0.4, 0.4, 0.2),
        )
        magnitude = math.sqrt(
            sum(float(data["combined"])**2 for data in gradients.values()))
        weighted.append((candidate, 0.05 + magnitude, gradients))
    seed_material = (
        f"{run_id}:{generation}:{archive_revision}:" +
        ",".join(str(candidate["id"]) for candidate, _, _ in weighted))
    seed = int.from_bytes(
        hashlib.sha256(seed_material.encode("utf-8")).digest()[:8], "big")
    rng = random.Random(seed)
    total_weight = sum(weight for _, weight, _ in weighted)
    threshold = rng.random() * total_weight
    cumulative = 0.0
    selected = weighted[-1]
    for item in weighted:
        cumulative += item[1]
        if threshold <= cumulative:
            selected = item
            break
    candidate, weight, gradients = selected
    parent = dict(candidate)
    parent["_selection_gradients"] = gradients
    parent["_selection_gradient_magnitude"] = weight - 0.05
    parent["_selection_probability"] = weight / total_weight
    return parent, "gradient_weighted"


def select_target(
    parent: dict[str, Any] | None,
    transitions: Iterable[dict[str, Any]],
    elites: Iterable[dict[str, Any]] = (),
    *,
    generation: int,
) -> dict[str, Any]:
    """Choose an ordinal direction or categorical mechanism edge."""
    transitions = list(transitions)
    elites = list(elites)
    gradients = (parent.get("_selection_gradients") if parent else None)
    if not isinstance(gradients, dict):
        gradients = estimate_gradients(
            _gradient_inputs_for_parent(parent, transitions) if parent else [],
            parent=parent,
            elites=elites,
            gradient_weights=(0.4, 0.4, 0.2),
        )
    supported = [(dimension, data) for dimension, data in gradients.items()
                 if data["support"] > 0 or abs(data["exploration"]) > 0]
    if supported:
        dimension, data = max(supported,
                              key=lambda item: abs(item[1]["combined"]))
        direction = 1 if data["combined"] >= 0 else -1
        confidence = data["confidence"]
    else:
        dimension = _DIMENSIONS[(generation - 1) % len(_DIMENSIONS)]
        direction = 1
        confidence = 0.0

    parent_coordinate = None
    mechanism_target = None
    if parent:
        parent_coordinate = [
            int(parent["algorithm"]),
            int(parent["engine"]),
            int(parent["memory"]),
            int(parent["dispatch"]),
            str(parent["mechanism"]),
        ]
        current = int(parent[dimension])
        if not 0 <= current + direction <= 3:
            direction *= -1
        # Periodically explore an implementation mechanism without pretending
        # the categorical lane has a signed gradient.
        if generation % 5 == 0:
            current_mechanism = str(parent["mechanism"])
            edge_values: dict[str, list[float]] = defaultdict(list)
            for transition in transitions:
                parent_json = transition.get("parent_json") or {}
                child_json = transition.get("child_json") or {}
                child_mechanism = child_json.get("mechanism")
                delta = transition.get("delta_fitness")
                if (parent_json.get("mechanism") == current_mechanism
                        and child_mechanism and delta is not None):
                    edge_values[str(child_mechanism)].append(float(delta))
            if edge_values:
                mechanism_target = max(
                    edge_values,
                    key=lambda mechanism: (sum(edge_values[mechanism]) / len(
                        edge_values[mechanism])),
                )
            else:
                index = (_MECHANISMS.index(current_mechanism)
                         if current_mechanism in _MECHANISMS else 0)
                mechanism_target = _MECHANISMS[(index + 1) % len(_MECHANISMS)]

    semantics = annotate_target(
        parent_coordinate,
        dimension,
        direction,
        mechanism_target,
    )
    return {
        "dimension":
        dimension,
        "direction":
        direction,
        "confidence":
        confidence,
        "mechanism_target":
        mechanism_target,
        "parent_coordinate":
        parent_coordinate,
        "gradient":
        gradients.get(dimension, {}),
        "gradients":
        gradients,
        "parent_gradient_magnitude":
        (float(parent.get("_selection_gradient_magnitude", 0.0))
         if parent else 0.0),
        "parent_selection_probability":
        (float(parent.get("_selection_probability", 1.0)) if parent else 1.0),
        "gradient_weights": {
            "fitness": 0.4,
            "improvement_rate": 0.4,
            "exploration": 0.2,
        },
        **semantics,
    }
