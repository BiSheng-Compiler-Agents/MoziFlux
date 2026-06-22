# Triton Operator Static Code Review Report

## Basic Information
- Operator Name: SELU
- Code File: `opt_27_SELU_.py`

## Host Side

| Check Item | Level | Status | Issue | Suggestion |
|------------|-------|--------|-------|------------|
| Grid/Core Type | P0 | ✅ | Pure elementwise kernel uses vector-core style 1D program grid; no hardcoded physical core count. | Keep direct grid below 65535 and persistent fallback for larger tensors. |
| Block Configuration | P1-P2 | ✅ | `BLOCK_SIZE=8192` is constexpr at launch and used by both kernels. | Re-tune only if hardware benchmark shows UB pressure on lower-memory targets. |
| Dispatch Coverage | P0 | ✅ | Direct and persistent paths both exist and are covered by `profile_kernels.py` shapes. | Preserve both paths; do not make persistent unconditional. |
| Parameter Validation | P2 | ✅ | Host validates NPU device, dtype, autograd, and empty tensors. | None. |

## Device Side

| Check Item | Level | Status | Issue | Suggestion |
|------------|-------|--------|-------|------------|
| Mask Completeness | P0 | ✅ | All `tl.load` and `tl.store` calls use `mask=`; loads provide `other=0.0`. | None. |
| Data Type Compliance | P0-P1 | ✅ | No `tl.dot`, atomics, permutes, or unsupported integer tensor APIs. | None. |
| Precision Handling | P1 | ✅ | SELU math upcasts to fp32 and stores back to input dtype. | Maintain `rtol=1e-3, atol=1e-3` checks for fp16/bf16/fp32. |
| Code Patterns | P0-P2 | ✅ | No `break`, early `return` inside JIT loops, tensor indexing, or third-party calls in kernels. | None. |

## Performance Hazards

| Code Feature | Location | Suggestion |
|--------------|----------|------------|
| Persistent loop adds JUMPC/VF overhead | `_selu_persistent_kernel` | Correctly gated to `n_tiles > 65535`; do not use for smaller shapes. |
| SELU requires `tl.exp` | device kernels | RVECEX remains material due to activation math; no safe algebraic removal without changing semantics. |
| Input contiguity copy | `ModelNew.forward` | Required to preserve general input layout support; callers can pass contiguous tensors to avoid copy. |

## Summary

### P0 Critical (Must Fix)
- None.

### P1 Severe (Strongly Recommended to Fix)
- None.

### P2 Suggestion (Optimization Items)
- Optional future tuning: compare `BLOCK_SIZE=4096/8192/16384` on hardware for smaller direct shapes; current `8192` wins on cannsim-normalized throughput and original-shape launchability.
