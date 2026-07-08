# Triton Operator Static Code Review Report

## Basic Information
- Operator Name: Conv3d_Softmax_MaxPool_MaxPool
- Code File: `opt_6_Conv3d_Softmax_MaxPool_MaxPool.py`

## Host Side
| Check Item | Level | Status | Issue | Suggestion |
|------------|-------|--------|-------|-------------|
| Grid/Core Type | P0 | ✅ | Vector softmax/pool kernels use vector-style elementwise/reduction work; no `tl.dot`/Cube mismatch. Grid is data-derived, not hardcoded to a physical core count. | Keep direct grid below `_MAX_GRID`; persistent path handles overflow. |
| Block Configuration | P1-P2 | ✅ | `BLOCK_C`, `BLOCK_OW`, and `K` are constexpr launch meta-parameters. | Keep `C <= 64` Triton guard unless UB budget is retuned. |
| Parameter Validation | P2 | ✅ | Preserves baseline constructor and input contract. Wide channels route to ACL fallback instead of failing. | None. |

## Device Side
| Check Item | Level | Status | Issue | Suggestion |
|------------|-------|--------|-------|-------------|
| Mask Completeness | P0 | ✅ | All `tl.load` and `tl.store` calls use masks. | None. |
| Data Type Compliance | P0-P1 | ✅ | Kernel uses fp32 reductions for softmax; no unsupported `tl.dot` dtypes. | None. |
| Precision Handling | P1 | ✅ | Softmax subtracts per-column max before `tl.exp`; accumulation is fp32; output stores to destination dtype. | Keep tolerance at `1e-3` in profiling. |
| Code Patterns | P0-P2 | ✅ | No Python tensor indexing inside JIT, no `break`/`return` in loops, persistent path loops over tiles. | None. |

## Performance Hazards
| Code Feature | Location | Suggestion |
|--------------|----------|-------------|
| `permute(...).contiguous()` materializes the full conv output | `_softmax_then_two_pools_fused_triton` | Validate end-to-end hardware latency; ACL fallback can be selected if materialization dominates. |
| Persistent path is rarely used for target shape | `_softmax_pool2_clast_persistent_kernel` | Unit test forces persistent dispatch by lowering `_MAX_GRID`; benchmark direct path for target. |

## Summary

### P0 Critical (Must Fix)
- None found.

### P1 Severe (Strongly Recommended to Fix)
- None found.

### P2 Suggestion (Optimization Items)
- Use remote hardware results to decide whether channel-last fused Triton or native ACL post-op is faster at the target shape.
