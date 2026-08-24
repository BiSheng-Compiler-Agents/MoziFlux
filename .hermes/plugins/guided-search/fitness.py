"""Fitness functions for simulation and physical-NPU measurements."""

from __future__ import annotations

import math
import re
import json
from pathlib import Path
from collections.abc import Sequence
from typing import Any, Mapping


def _positive(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return number


def simulation_fitness(candidate: Mapping[str, Any],
                       baseline: Mapping[str, Any]) -> float:
    """Return normalized useful-work throughput relative to an equal-work probe."""
    candidate_contract = candidate.get("probe_contract")
    baseline_contract = baseline.get("probe_contract")
    if (candidate_contract is not None and baseline_contract is not None
            and candidate_contract != baseline_contract):
        raise ValueError("probe contract mismatch")
    candidate_rate = _positive(
        candidate.get("useful_work"), "candidate useful_work") / _positive(
            candidate.get("wall_cycles"), "candidate wall_cycles")
    baseline_rate = _positive(
        baseline.get("useful_work"), "baseline useful_work") / _positive(
            baseline.get("wall_cycles"), "baseline wall_cycles")
    return candidate_rate / baseline_rate


def trace_wall_cycles(evidence: Mapping[str, Any]) -> float:
    """Derive wall cycles from a trusted cannsim Chrome trace result."""
    trace = evidence.get("trace_json")
    if isinstance(trace, str) and trace.strip():
        events = json.loads(trace)
    elif isinstance(trace, list):
        events = trace
    else:
        path = evidence.get("trace_local_path")
        if not path:
            raise ValueError("cannsim evidence has no trace content or path")
        events = json.loads(Path(str(path)).read_text(encoding="utf-8"))
    durations = [event for event in events if event.get("ph") == "X"]
    if not durations:
        raise ValueError("cannsim trace is scaffold-only (no duration events)")
    start = min(float(event["ts"]) for event in durations)
    end = max(
        float(event["ts"]) + float(event.get("dur", 0)) for event in durations)
    return _positive(end - start, "trace wall cycles")


def hardware_fitness(*, candidate_ms: Sequence[float],
                     pytorch_acl_ms: Sequence[float]) -> float:
    """Weighted geometric-mean speedup using PyTorch/ACL as the sole reference."""
    if not candidate_ms or len(candidate_ms) != len(pytorch_acl_ms):
        raise ValueError(
            "candidate and PyTorch/ACL measurements must be non-empty and aligned"
        )
    logs = []
    for candidate, reference in zip(candidate_ms, pytorch_acl_ms):
        speedup = _positive(reference, "PyTorch/ACL latency") / _positive(
            candidate, "candidate latency")
        logs.append(math.log(speedup))
    return math.exp(sum(logs) / len(logs))


def hardware_fitness_from_benchmark(bench_output: str) -> float:
    """Parse canonical perf-report output and score Optimized vs PyTorch/ACL.

    Headers use two-or-more spaces between columns while method names contain
    single spaces. Data labels are contractually whitespace-free.
    """
    lines = bench_output.splitlines()
    candidate_ms: list[float] = []
    reference_ms: list[float] = []
    methods: list[str] | None = None
    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            continue
        columns = [
            part.strip() for part in re.split(r"\s{2,}", stripped)
            if part.strip()
        ]
        if "PyTorch / ACL" in columns and "Optimized Triton" in columns:
            methods = columns[1:] if columns[0] in {"label", "N"} else columns
            continue
        if methods is None:
            continue
        tokens = stripped.split()
        if not tokens or not tokens[0].isdigit():
            continue
        n_values = len(methods)
        if len(tokens) < n_values + 1:
            continue
        values = tokens[-n_values:]
        try:
            parsed = [float(value) for value in values]
        except ValueError:
            continue
        reference_ms.append(parsed[methods.index("PyTorch / ACL")])
        candidate_ms.append(parsed[methods.index("Optimized Triton")])
    if not candidate_ms:
        raise ValueError(
            "remote benchmark output has no canonical PyTorch / ACL and "
            "Optimized Triton measurements")
    return hardware_fitness(
        candidate_ms=candidate_ms,
        pytorch_acl_ms=reference_ms,
    )
