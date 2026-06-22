# Triton Operator Static Code Review Report

## Basic Information
- Operator Name: reverse cumsum
- Code File: `opt_91_cumsum_reverse.py`

## Host Side
| Check Item | Level | Status | Issue | Suggestion |
|------------|-------|--------|-------|------------|
| Grid/Core Type | P0 | ✅ | Production path uses ACL cumsum; retained fallback Triton kernel is pure vector/scan and grid-capped. | None. |
| Block Configuration | P1-P2 | ✅ | Fallback `BLOCK_N` is constexpr and UB use is below 192 KB. | None. |
| Parameter Validation | P2 | ✅ | Validates NPU tensor, dtype, ndim, dim range, and empty scan dimension. | None. |

## Device Side
| Check Item | Level | Status | Issue | Suggestion |
|------------|-------|--------|-------|------------|
| Mask Completeness | P0 | ✅ | Fallback Triton `tl.load` and `tl.store` calls use masks; production ACL path has no custom OOB access. | None. |
| Data Type Compliance | P0-P1 | ✅ | No `tl.dot`, atomics, unsupported tensor indexing, `break`, or in-loop `return` in fallback kernel. | None. |
| Precision Handling | P1 | ✅ | Fallback upcasts to fp32 before scan/reduction; ACL path matches PyTorch reference in unit tests. | None. |
| Code Patterns | P0-P2 | ✅ | Fallback uses `tl.range` and persistent row loop; production path removes scalar-limited custom scan. | None. |

## Performance Hazards
| Code Feature | Location | Suggestion |
|---|---|---|
| Retained fallback custom scan remains scalar-limited | `_rcumsum_lastdim_kernel_opt` | Do not route production traffic to it unless ACL is unavailable. |
| Two `torch.flip` temporaries | production ACL path | Accepted because remote hardware shows 26.47x speedup vs Baseline Triton1 at target. |

## Summary

### P0 Critical (Must Fix)
- None.

### P1 Severe (Strongly Recommended to Fix)
- None.

### P2 Suggestion (Optimization Items)
- If a future Ascend Triton vector prefix primitive becomes available, replace the retained fallback kernel and re-benchmark against ACL.
