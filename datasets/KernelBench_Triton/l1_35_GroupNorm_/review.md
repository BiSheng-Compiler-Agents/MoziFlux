# Triton Operator Static Code Review Report

## Basic Information
- Operator Name: GroupNorm
- Code File: `opt_35_GroupNorm_.py`

## Host Side

| Check Item | Level | Status | Issue | Suggestion |
|---|---|---|---|---|
| Grid/Core Type | P0 | ✅ | Vector/reduction kernels do not use `tl.dot`; grids are 1D. Stats grid is capped through adaptive `num_parts`; apply has a persistent fallback for `N*C > 65535`. | None. |
| Block Configuration | P1-P2 | ✅ | `BLOCK_SIZE` and `BLOCK_HW` are `tl.constexpr`; `BLOCK_HW=1024` fits UB for fp32 load/compute/store. | None. |
| Parameter Validation | P2 | ✅ | Preserves baseline checks for NPU tensor, no autograd, rank >= 3, and `C % groups == 0`. | None. |

## Device Side

| Check Item | Level | Status | Issue | Suggestion |
|---|---|---|---|---|
| Mask Completeness | P0 | ✅ | All boundary `tl.load`/`tl.store` operations use masks. Scalar parameter loads are in-bounds by construction. | None. |
| Data Type Compliance | P0-P1 | ✅ | No unsupported `tl.dot`, atomics, tensor indexing, `break`, or early `return`. | None. |
| Precision Handling | P1 | ✅ | Reductions upcast input to fp32 and accumulate sum/sum-square in fp32; output stores to `y_flat` with input dtype. | None. |
| Code Patterns | P0-P2 | ✅ | Uses tensor accumulators for partial stats; persistent apply loop iterates over tiles, not elements. | None. |

## Performance Hazards

| Code Feature | Location | Suggestion |
|---|---|---|
| Three Triton launches for optimized path | partial stats + finalize + apply | Acceptable for medium/large GroupNorm; small shape is slower than baseline1 due launch overhead. |
| Persistent apply fallback has integer division/modulo | `_groupnorm_apply_persistent_kernel` | Only used beyond direct grid limit; direct path is used for benchmark shape. |
| Baseline2/PyTorch ACL remains much faster on full benchmark | hardware report | Further Triton gains likely require a lower-level contiguous DMA/vectorization strategy beyond this patch. |

## Summary

### P0 Critical (Must Fix)
- None.

### P1 Severe (Strongly Recommended to Fix)
- None.

### P2 Suggestion (Optimization Items)
- Consider a small-shape dispatch back to baseline-style stats to avoid the extra partial/finalize launch overhead.
- Investigate larger vectorized GM transactions for the apply phase; cannsim still shows memory wait/SCALARLDST pressure.
