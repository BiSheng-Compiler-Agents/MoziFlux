"""Bounded prompt packets for active guided-search attempts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .descriptors import describe_coordinate


def render_attempt_prompt(
    *, run: dict[str, Any], attempt: dict[str, Any], parent: dict[str, Any] | None = None
) -> str:
    target = json.loads(attempt.get("target_json") or "{}")
    parent_coordinate = target.get("parent_coordinate")
    hermes_home = Path(os.environ.get("HERMES_HOME", "~/.hermes")).expanduser()
    artifact_dir = (
        hermes_home / "guided-search" / "artifacts"
        / str(run["id"]) / str(attempt["id"])
    )
    promotion_domain = str(run.get("promotion_domain", "simulation"))
    promotion_rule = (
        "remote_verify is available: only physical-NPU results relative to "
        "PyTorch/ACL may replace elites. Cannsim is diagnostic only."
        if promotion_domain == "hardware"
        else "remote_verify is unavailable: valid equal-work cannsim results may "
             "replace elites and select the final winner."
    )
    if attempt.get("kind") == "baseline":
        evaluator_action = (
            "Run remote_verify correctness and PyTorch/ACL-relative benchmarking "
            "for the unchanged baseline, then submit hardware evidence."
            if promotion_domain == "hardware"
            else "Run the unchanged baseline through the fixed-work cannsim probe, "
                 "then submit source-bound simulation evidence using that same "
                 "baseline source hash and probe contract."
        )
        lines = [
            "[GUIDED SEARCH - BASELINE CALIBRATION]",
            f"RUN: {run['id']}",
            f"ATTEMPT: {attempt['id']}",
            "KIND: baseline evaluation (not part of the mutation budget)",
            f"PHASE: {attempt['phase']}",
            f"ELITE PROMOTION DOMAIN: {promotion_domain}",
            f"SEARCH ARTIFACT DIR: {artifact_dir}",
            "The archive must contain one authoritative baseline elite before "
            "mutation search begins.",
            "Call guided_search_checkout_parent exactly once, do not modify the "
            "materialized source, and evaluate it unchanged.",
            evaluator_action,
            "If baseline correctness/evaluation fails or this attempt is abandoned, "
            "the run terminates as failed_baseline_evaluation.",
        ]
        if attempt.get("last_error"):
            lines.append(f"LAST ERROR: {attempt['last_error']}")
        if parent:
            lines.append(
                f"BASELINE SOURCE HASH: {parent.get('source_hash', 'unknown')}")
        return "\n".join(lines)
    parent_semantics = (
        describe_coordinate(parent_coordinate) if parent_coordinate else [])
    try:
        parent_evidence = json.loads(
            (parent or {}).get("descriptor_evidence_json") or "[]")[:10]
    except (json.JSONDecodeError, TypeError):
        parent_evidence = []
    gradient = target.get("gradient") or {}
    gradients = target.get("gradients") or {}
    gradient_weights = target.get("gradient_weights") or {}
    lines = [
        "[GUIDED SEARCH - ASCEND MAP-ELITES]",
        f"RUN: {run['id']}",
        f"ATTEMPT: {attempt['id']}",
        f"GENERATION: {attempt['generation']} / BUDGET {run['budget']}",
        f"PHASE: {attempt['phase']}",
        f"PARENT CANDIDATE: {attempt.get('parent_candidate_id') or 'baseline seed'}",
        f"PARENT CELL (stored coordinate): {parent_coordinate}",
        "PARENT BEHAVIOR:",
        *[
            f"  - {item['label']}: {item['bin']} — {item['name']}: "
            f"{item.get('description', item.get('guidance', ''))}"
            for item in parent_semantics
        ],
        "PARENT CLASSIFICATION EVIDENCE:",
        *[
            f"  - {item.get('dimension')}:{item.get('rule')} line "
            f"{item.get('line', 0)} — {item.get('detail', '')}"
            for item in parent_evidence
        ],
        "PARENT GRADIENT SAMPLING:",
        f"  Combined-gradient magnitude: "
        f"{float(target.get('parent_gradient_magnitude', 0.0)):.3f}",
        f"  Parent sampling probability: "
        f"{float(target.get('parent_selection_probability', 1.0)):.3f}",
        "FULL SIGNED GRADIENT VECTOR:",
        *[
            f"  - {dimension}: {float((gradients.get(dimension) or {}).get('combined', 0.0)):+.3f} "
            f"[fitness={float((gradients.get(dimension) or {}).get('fitness_normalized', 0.0)):+.3f}, "
            f"rate={float((gradients.get(dimension) or {}).get('improvement_rate', 0.0)):+.3f}, "
            f"explore={float((gradients.get(dimension) or {}).get('exploration', 0.0)):+.3f}]"
            for dimension in ("algorithm", "engine", "memory", "dispatch")
        ],
        "PRIMARY GRADIENT-TO-PROMPT HINT (not a mandatory target cell):",
        f"  Dimension: {target.get('dimension_label', target.get('dimension', 'unknown'))}",
        f"  Current: bin {target.get('current_bin')} — {target.get('current_name')}: "
        f"{target.get('current_description', '')}",
        f"  Target: bin {target.get('target_bin')} — {target.get('target_name')}: "
        f"{target.get('target_description', '')}",
        f"  Direction: {target.get('direction', 0):+d} means "
        f"{target.get('direction_meaning', 'an empirical archive movement')}; "
        "a higher bin is not inherently better.",
        f"  Direction evidence confidence: {float(target.get('confidence', 0.0)):.3f}; "
        f"parent selection policy: {target.get('selection_mode', 'unknown')}.",
        "  Combined gradient: "
        f"fitness={float(gradient.get('fitness_normalized', 0.0)):+.3f} "
        f"(w={float(gradient_weights.get('fitness', 0.4)):.1f}), "
        f"improvement_rate={float(gradient.get('improvement_rate', 0.0)):+.3f} "
        f"(w={float(gradient_weights.get('improvement_rate', 0.4)):.1f}), "
        f"exploration={float(gradient.get('exploration', 0.0)):+.3f} "
        f"(w={float(gradient_weights.get('exploration', 0.2)):.1f}), "
        f"combined={float(gradient.get('combined', 0.0)):+.3f}.",
        f"  Suggested mutation behavior: {target.get('mutation_guidance', '')}",
        f"OBJECTIVE: {attempt['mutation_objective']}",
        f"ELITE PROMOTION DOMAIN: {promotion_domain}",
        f"SEARCH ARTIFACT DIR: {artifact_dir}",
        "",
        "Keep this gradient-sampled parent and focused hint sticky until the current candidate has a "
        "conclusive compile, correctness, and evaluation result.",
        "If PHASE is checkout, call guided_search_checkout_parent exactly once before "
        "editing. It materializes the selected SQL parent and refuses to overwrite a "
        "candidate after the attempt enters editing.",
        "Do not create final profile_kernels.py, Optimizations.md, "
        "performance_report.md, review.md, or results.txt during SEARCH.",
        "Temporary compile/cannsim/profile harnesses may be created only under the "
        "SEARCH ARTIFACT DIR; they are not kernel-sandbox deliverables.",
        "Optimizations may use standard Triton, compiler-managed BishengIR/NPUOptions, "
        "lightweight Ascend hints, manual AL/BL extensions, or native hybrid dispatch. "
        "Prefer the least fragile mechanism supported by evidence; do not force extensions.",
        promotion_rule,
        "Cannsim evidence always requires equal mathematical work, [HOST] PASS, "
        "and a non-scaffold trace.",
        "For simulation promotion, run source-bound parent/baseline cannsim probes "
        "under the same fixed-work probe_contract. Submit the contract identifier and "
        "baseline source_content_hash; do not invent useful_work or cycle values. "
        "Wall cycles are derived from the traces and equal work is implied by the contract.",
        "After evaluation, call guided_search_submit_candidate with the exact candidate "
        "path and structured result. Do not call kernel_status advance during SEARCH.",
    ]
    if target.get("mechanism_target"):
        lines.extend((
            f"MECHANISM TARGET: {target['mechanism_target']} — "
            f"{target.get('mechanism_target_label', target['mechanism_target'])}",
            f"MECHANISM GUIDANCE: {target.get('mechanism_guidance', '')}",
            "Mechanism is categorical, not a higher/lower quality level.",
        ))
    error = attempt.get("last_error")
    if error:
        lines.extend(("", f"LAST ERROR: {error}"))
    if parent:
        lines.extend((
            "",
            f"SELECTED PARENT SOURCE HASH: {parent.get('source_hash', 'unknown')}",
            "Use guided_search_checkout_parent to materialize the source; it is "
            "omitted from per-request middleware context to keep requests bounded.",
        ))
    return "\n".join(lines)
