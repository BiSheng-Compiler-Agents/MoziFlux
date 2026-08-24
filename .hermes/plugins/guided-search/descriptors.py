"""Static Ascend MAP-Elites descriptor classification."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

from .source_features import DescriptorEvidence, extract_source_features

_NPU_OPTIONS = {
    "allow_fp8e4nv",
    "allowed_dot_input_precisions",
    "arch",
    "auto_blockify_size",
    "auto_tile_and_bind_subblock",
    "auto_vectorize_v2_max_fused_ops_num",
    "bisheng_options",
    "cluster_dims",
    "code_motion",
    "compile_mode",
    "compile_on_910_95",
    "debug",
    "default_dot_input_precision",
    "deprecated_fp8_dtypes",
    "disable_auto_inject_block_sync",
    "disable_fma",
    "disable_size_align_for_cast",
    "disable_tightly_coupled_buffer_reuse",
    "enable_auto_bind_sub_block",
    "enable_auto_vectorize_v2",
    "enable_bishengir_simt_optimization",
    "enable_buffer_insert_optimization",
    "enable_cce_vf_auto_sync",
    "enable_cce_vf_remove_membar",
    "enable_costmodel_backend",
    "enable_cross_if_fusion",
    "enable_cube_block_merge",
    "enable_drop_unit_dims",
    "enable_dynamic_cv_pipeline",
    "enable_flatten",
    "enable_fp_fusion",
    "enable_hivm_auto_cv_balance",
    "enable_mask_fallback_conversion",
    "enable_mixed_cv",
    "enable_nd2nz_on_vector",
    "enable_persistent",
    "enable_preload",
    "enable_select_analysis",
    "enable_simt_reorder_instruction",
    "enable_sync_block_lock",
    "enable_ub_refine_opt",
    "enable_ubuf_saving",
    "enable_vf_fusion",
    "enable_vf_operand_substitution",
    "enable_warp_specialization",
    "extern_libs",
    "force_simt_only",
    "force_simt_template",
    "has_auto_blockify_blacklist_op",
    "hfusion_enable_multiple_consumer_fusion",
    "inject_barrier_all",
    "inject_block_all",
    "inter_cache_num",
    "intra_cache_num",
    "kernel_name",
    "limit_auto_multi_buffer_buffer",
    "limit_auto_multi_buffer_of_local_buffer",
    "limit_auto_multi_buffer_only_for_local_buffer",
    "llvm_version",
    "load_cache_num",
    "max_num_imprecise_acc_default",
    "mix_mode",
    "multibuffer",
    "num_buffers_warp_spec",
    "num_consumer_groups",
    "num_ctas",
    "num_stages",
    "num_warps",
    "ops_reorder",
    "optimize_dynamic_offset",
    "optimize_epilogue",
    "parallel_mode",
    "prevec_max_fused_ops_num",
    "reg_dec_producer",
    "reg_inc_consumer",
    "sanitize_overflow",
    "set_workspace_multibuffer",
    "shared_mem_dynamic_size",
    "simt_stack_limit",
    "storage_align",
    "stream",
    "superblock_factor",
    "supported_fp8_dtypes",
    "sync_solver",
    "tile_mix_cube_loop",
    "tile_mix_vector_loop",
    "unit_flag",
    "vf_fusion_mode",
    "vf_merge_level",
    "warp_size",
}
NPU_OPTIONS = frozenset(_NPU_OPTIONS)

# Generic launch/metadata fields are valid NPUOptions but do not by themselves
# mean that the implementation relies on compiler-managed optimization.
_NON_MANAGED_NPU_OPTIONS = {
    "allowed_dot_input_precisions",
    "arch",
    "bisheng_options",
    "cluster_dims",
    "compile_on_910_95",
    "debug",
    "default_dot_input_precision",
    "deprecated_fp8_dtypes",
    "extern_libs",
    "kernel_name",
    "llvm_version",
    "max_num_imprecise_acc_default",
    "num_buffers_warp_spec",
    "num_consumer_groups",
    "num_ctas",
    "num_stages",
    "num_warps",
    "reg_dec_producer",
    "reg_inc_consumer",
    "sanitize_overflow",
    "stream",
    "supported_fp8_dtypes",
    "warp_size",
}
_COMPILER_OPTIONS = _NPU_OPTIONS - _NON_MANAGED_NPU_OPTIONS

_MEMORY_TILING_OPTIONS = {
    "auto_tile_and_bind_subblock",
    "enable_auto_bind_sub_block",
    "enable_drop_unit_dims",
    "enable_flatten",
    "enable_nd2nz_on_vector",
    "storage_align",
}
_MEMORY_REUSE_OPTIONS = {
    "disable_tightly_coupled_buffer_reuse",
    "enable_ub_refine_opt",
}
_MEMORY_PIPELINE_OPTIONS = {
    "enable_buffer_insert_optimization",
    "enable_preload",
    "enable_ubuf_saving",
    "inter_cache_num",
    "intra_cache_num",
    "limit_auto_multi_buffer_buffer",
    "limit_auto_multi_buffer_of_local_buffer",
    "limit_auto_multi_buffer_only_for_local_buffer",
    "load_cache_num",
    "multibuffer",
    "set_workspace_multibuffer",
}
_COORDINATION_OPTIONS = {
    "disable_auto_inject_block_sync",
    "enable_cce_vf_auto_sync",
    "enable_cce_vf_remove_membar",
    "enable_cube_block_merge",
    "enable_dynamic_cv_pipeline",
    "enable_hivm_auto_cv_balance",
    "enable_mixed_cv",
    "enable_sync_block_lock",
    "inject_barrier_all",
    "inject_block_all",
    "sync_solver",
}
_DISPATCH_OPTIONS = {
    "auto_blockify_size",
    "enable_persistent",
    "superblock_factor",
    "tile_mix_cube_loop",
    "tile_mix_vector_loop",
}


@dataclass(frozen=True)
class BehaviorDescriptor:
    algorithm: int
    engine: int
    memory: int
    dispatch: int
    mechanism: str
    evidence: tuple[DescriptorEvidence, ...] = ()

    def coordinate(self) -> tuple[int, int, int, int, str]:
        return (
            self.algorithm,
            self.engine,
            self.memory,
            self.dispatch,
            self.mechanism,
        )


DESCRIPTOR_SCHEMA: dict[str, dict[str, Any]] = {
    "algorithm": {
        "label": "algorithmic structure",
        "bins": {
            0: {
                "name":
                "direct_single_stage",
                "description":
                ("One direct stage with no reachable fusion, online recurrence, or "
                 "multi-kernel decomposition."),
                "guidance":
                ("Use a direct formulation; remove unnecessary fusion or complex "
                 "reformulation when its overhead exceeds its benefit."),
            },
            1: {
                "name":
                "local_fusion",
                "description":
                ("Locally fused operations or a fused pointwise/reduction epilogue."
                 ),
                "guidance":
                ("Fuse adjacent pointwise, reduction, or epilogue work to avoid "
                 "intermediate materialization while preserving exact semantics."
                 ),
            },
            2: {
                "name":
                "loop_or_reduction_reformulation",
                "description":
                ("Online, scan, chunked, or loop-carried reduction reformulation."
                 ),
                "guidance":
                ("Explore a single-pass, online, scan, folding, or other structural "
                 "rewrite that reduces repeated work or full-tensor intermediates."
                 ),
            },
            3: {
                "name":
                "multi_stage_decomposition",
                "description":
                ("Sequential partial/finalize or producer/consumer kernel stages."
                 ),
                "guidance":
                ("Explore a correctness-preserving multi-stage decomposition when an "
                 "atomic, oversized, or serial stage limits performance."),
            },
        },
    },
    "engine": {
        "label": "Ascend execution-engine behavior",
        "bins": {
            0: {
                "name":
                "single_engine",
                "description":
                "Single-engine execution: Vector-only or mandatory Cube-only.",
                "guidance":
                ("Keep a single engine when no cross-engine handoff is useful; all "
                 "dot/BMM/matmul work remains on Cube."),
            },
            1: {
                "name":
                "implicit_mixed",
                "description":
                "Reachable Cube work plus a Vector stage without explicit partitioning.",
                "guidance":
                ("Fuse a necessary Vector reduction/epilogue around Cube work before "
                 "introducing explicit scheduling machinery."),
            },
            2: {
                "name":
                "explicit_serial_handoff",
                "description":
                ("Explicit Cube/Vector scopes or synchronization without buffering overlap."
                 ),
                "guidance":
                ("Partition work across Cube and Vector engines with a simple, correct "
                 "serial handoff before introducing overlap."),
            },
            3: {
                "name":
                "overlapped_cv_pipeline",
                "description":
                ("Mixed Cube/Vector execution with explicit or compiler-managed "
                 "coordination and potential overlap."),
                "guidance":
                ("Explore safe Cube/Vector overlap or coordinated handoff using scopes, "
                 "events, auto scheduling, or CV balancing; validate event accounting."
                 ),
            },
        },
    },
    "memory": {
        "label": "memory hierarchy and dataflow",
        "bins": {
            0: {
                "name":
                "direct_access",
                "description":
                "No explicit tiling, reuse, buffering, or pipelining detected.",
                "guidance":
                ("Use simple direct access when reuse machinery costs more than it saves; "
                 "minimize temporary storage and data movement."),
            },
            1: {
                "name":
                "contiguous_or_accumulator_reuse",
                "description":
                ("Contiguous/aligned access or a numerical accumulator reuses values on chip."
                 ),
                "guidance":
                ("Carry an accumulator across repeated work so partial results stay on chip "
                 "instead of being reloaded or rematerialized."),
            },
            2: {
                "name":
                "explicit_tiling_or_local_buffer",
                "description":
                ("Explicit block pointers, local allocation, tiling, or register blocking."
                 ),
                "guidance":
                ("Use explicit tiling/local buffers or register blocking to increase reuse "
                 "without exceeding UB/L1/register limits."),
            },
            3: {
                "name":
                "pipelined_buffering",
                "description":
                ("Multibuffer, preload, UB-saving, workspace buffering, or ping-pong dataflow."
                 ),
                "guidance":
                ("Explore preload/multibuffer or ping-pong dataflow to overlap movement and "
                 "compute; check UB pressure, alignment, synchronization, and queue depth."
                 ),
            },
        },
    },
    "dispatch": {
        "label": "work decomposition and dispatch",
        "bins": {
            0: {
                "name":
                "fixed_direct",
                "description":
                "One fixed direct launch policy.",
                "guidance":
                ("Prefer a simple launch/grid when specialization or persistent dispatch "
                 "adds overhead without enough parallel benefit."),
            },
            1: {
                "name":
                "tuned_or_swizzled_direct",
                "description":
                "Autotuned or GROUP_M/N-swizzled direct launch.",
                "guidance":
                ("Choose an explicit block/tile decomposition that balances parallelism, "
                 "tail masking, and per-program work."),
            },
            2: {
                "name":
                "persistent_or_decomposed",
                "description":
                ("Persistent/grid-stride traversal or sequential multi-stage launches."
                 ),
                "guidance":
                ("Explore persistent or grid-stride decomposition, auto-blockification, "
                 "or diagonal scheduling while respecting grid and resource limits."
                 ),
            },
            3: {
                "name":
                "specialized_multi_path",
                "description":
                ("Autotuned, persistent-specialized, direct, or native multi-path dispatch."
                 ),
                "guidance":
                ("Use shape-aware/autotuned or multi-path dispatch only when branches are "
                 "fully covered and dispatch overhead is justified."),
            },
        },
    },
}

MECHANISM_SCHEMA: dict[str, dict[str, str]] = {
    "standard_triton": {
        "label":
        "standard Triton",
        "guidance":
        "Use portable Triton primitives and ordinary launch configuration.",
    },
    "compiler_managed": {
        "label":
        "compiler-managed Triton",
        "guidance":
        ("Use BishengIR/NPUOptions such as auto scheduling, CV balancing, flattening, "
         "vectorization, preload, or UB-saving without manual pipeline ownership."
         ),
    },
    "extension_hinted": {
        "label":
        "Ascend extension hinted",
        "guidance":
        ("Use lightweight Ascend compile hints, casts, indexing, gather/scatter, or "
         "multibuffer hints while leaving scheduling mostly compiler managed."
         ),
    },
    "manual_extension": {
        "label":
        "manual Ascend extension scheduling",
        "guidance":
        ("Use explicit AL/BL scopes, allocation, sub-vector binding, and synchronization; "
         "prove event balance, alignment, and resource safety."),
    },
    "native_hybrid": {
        "label":
        "native/ACL hybrid dispatch",
        "guidance":
        ("Use ACL/torch_npu/native dispatch for justified shape regimes while preserving "
         "the public API and covering every fallback path."),
    },
}


def describe_coordinate(
    coordinate: tuple[int, int, int, int, str] | list[Any],
) -> list[dict[str, Any]]:
    """Return semantic records for a stored coordinate."""
    values = list(coordinate)
    records: list[dict[str, Any]] = []
    for index, dimension in enumerate(
        ("algorithm", "engine", "memory", "dispatch")):
        value = int(values[index])
        spec = DESCRIPTOR_SCHEMA[dimension]
        bin_spec = spec["bins"][value]
        records.append({
            "dimension": dimension,
            "label": spec["label"],
            "bin": value,
            **bin_spec,
        })
    mechanism = str(values[4])
    mechanism_spec = MECHANISM_SCHEMA.get(
        mechanism, {
            "label": mechanism,
            "guidance": "Treat this implementation mechanism as categorical.",
        })
    records.append({
        "dimension": "mechanism",
        "label": "implementation mechanism",
        "bin": mechanism,
        "name": mechanism,
        **mechanism_spec,
    })
    return records


def annotate_target(
    parent_coordinate: tuple[int, int, int, int, str] | list[Any] | None,
    dimension: str,
    direction: int,
    mechanism_target: str | None = None,
) -> dict[str, Any]:
    """Translate a numeric target into explicit mutation semantics."""
    spec = DESCRIPTOR_SCHEMA[dimension]
    result: dict[str, Any] = {
        "dimension_label":
        spec["label"],
        "direction_meaning":
        ("move toward the next higher behavioral bin"
         if direction > 0 else "move toward the next lower behavioral bin"),
    }
    if parent_coordinate is not None:
        index = ("algorithm", "engine", "memory", "dispatch").index(dimension)
        current_bin = int(list(parent_coordinate)[index])
        target_bin = max(0, min(3, current_bin + int(direction)))
        current_spec = spec["bins"][current_bin]
        target_spec = spec["bins"][target_bin]
        transition_guidance = (
            "Introduce behavior characteristic of the target bin. "
            if direction > 0 else
            "Simplify or remove behavior characteristic of the current bin, "
            "then realize the target bin. ") + target_spec["guidance"]
        result.update({
            "current_bin": current_bin,
            "current_name": current_spec["name"],
            "current_description": current_spec["description"],
            "target_bin": target_bin,
            "target_name": target_spec["name"],
            "target_description": target_spec["description"],
            "mutation_guidance": transition_guidance,
        })
    else:
        target_bin = 1 if direction > 0 else 0
        target_spec = spec["bins"][target_bin]
        result.update({
            "current_bin": None,
            "current_name": "unknown_seed",
            "current_description": "No classified parent is available yet.",
            "target_bin": target_bin,
            "target_name": target_spec["name"],
            "target_description": target_spec["description"],
            "mutation_guidance": target_spec["guidance"],
        })
    if mechanism_target:
        mechanism_spec = MECHANISM_SCHEMA[mechanism_target]
        result.update({
            "mechanism_target_label": mechanism_spec["label"],
            "mechanism_guidance": mechanism_spec["guidance"],
        })
    return result


def classify_candidate(
    source: str,
    launch_options: Mapping[str, Any] | None = None,
    environment: Mapping[str, Any] | None = None,
) -> BehaviorDescriptor:
    """Classify public-entrypoint-reachable optimization structure."""
    features = extract_source_features(
        source,
        launch_options or {},
        npu_options=NPU_OPTIONS,
        compiler_managed_options=_COMPILER_OPTIONS,
        memory_tiling_options=_MEMORY_TILING_OPTIONS,
        memory_reuse_options=_MEMORY_REUSE_OPTIONS,
        memory_pipeline_options=_MEMORY_PIPELINE_OPTIONS,
        coordination_options=_COORDINATION_OPTIONS,
        dispatch_options=_DISPATCH_OPTIONS,
        environment=environment or {},
    )

    algorithm = (3 if features.multi_stage else 2 if features.online_recurrence
                 or features.scan else 1 if features.fused_kernel else 0)
    engine = (3 if features.overlapped_cv_pipeline else
              2 if features.explicit_cv_handoff else
              1 if features.cube_ops and features.vector_ops else 0)
    memory = (
        3 if features.buffered_pipeline else 2 if features.explicit_tiling else
        1 if features.contiguous_access or features.accumulator_reuse else 0)
    dispatch = (3 if features.multi_path else 2 if features.persistent
                or features.multi_stage else 1 if features.tuned_direct else 0)

    if features.native_call:
        mechanism = "native_hybrid"
    elif features.manual_extension:
        mechanism = "manual_extension"
    elif features.extension_hint:
        mechanism = "extension_hinted"
    elif features.compiler_managed:
        mechanism = "compiler_managed"
    else:
        mechanism = "standard_triton"

    return BehaviorDescriptor(
        algorithm=algorithm,
        engine=engine,
        memory=memory,
        dispatch=dispatch,
        mechanism=mechanism,
        evidence=tuple(features.evidence),
    )


def candidate_fingerprint(
    source: str,
    launch_options: Mapping[str, Any] | None = None,
    environment: Mapping[str, Any] | None = None,
    *,
    target: str = "",
    compiler: str = "",
) -> str:
    payload = {
        "source": source.replace("\r\n", "\n").strip(),
        "launch_options": dict(sorted((launch_options or {}).items())),
        "environment": dict(sorted((environment or {}).items())),
        "target": target,
        "compiler": compiler,
    }
    encoded = json.dumps(payload,
                         sort_keys=True,
                         separators=(",", ":"),
                         default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
