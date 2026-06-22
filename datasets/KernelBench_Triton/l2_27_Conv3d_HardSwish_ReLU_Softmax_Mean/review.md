# Triton Operator Static Code Review Report

## Basic Information
- Operator Name: Conv3d_HardSwish_ReLU_Softmax_Mean
- Code File: opt_27_Conv3d_HardSwish_ReLU_Softmax_Mean.py

## Host Side
| Check Item | Level | Status | Issue | Suggestion |
|------------|-------|--------|-------|-------------|
| Grid/Core Type | P0 | ✅ | Pure vector/reduction kernels; no `tl.dot` core mismatch. Partial grid is `N * n_tiles`; reduce grid is `N * C`. | Keep grid guards before launch. |
| Block Configuration | P1-P2 | ✅ | `BLOCK_C` and `BLOCK_S` are constexpr; target `C=16`, `BLOCK_S=512` is UB-safe. | Re-tune `BLOCK_S` only with hardware proof. |
| Parameter Validation | P2 | ✅ | Preserves `ModelNew(in_channels, out_channels, kernel_size, bias=True)`. Wide/grid-overflow regimes route to ACL instead of raising. | None. |

## Device Side
| Check Item | Level | Status | Issue | Suggestion |
|------------|-------|--------|-------|-------------|
| Mask Completeness | P0 | ✅ | All pointer loads/stores have masks or exact grids. | None. |
| Data Type Compliance | P0-P1 | ✅ | Reductions and softmax are computed in fp32; stores cast back to output dtype. | None. |
| Precision Handling | P1 | ✅ | Softmax subtracts per-spatial max before `exp`; spatial partials are accumulated in fp32. | None. |
| Code Patterns | P0-P2 | ✅ | No tensor indexing, no `break`/loop `return`, no atomics in loops. | None. |

## Performance Hazards
| Code Feature | Location | Suggestion |
|--------------|----------|-------------|
| Two-launch post-op | Partial + reduce kernels | Necessary to expose spatial tile parallelism; benchmark against ACL for very small shapes. |
| ACL multi-tile dispatch | `_post_ops_acl` | Hardware-selected fastest path for `n_tiles > 1`; tiny Triton and wide-channel ACL paths are both unit-tested. |

## Summary

### P0 Critical (Must Fix)
- None.

### P1 Severe (Strongly Recommended to Fix)
- None.

### P2 Suggestion (Optimization Items)
- If future hardware shows tiny single-tile Triton is also slower than ACL, remove that dispatch path and keep the cannsim probe only as diagnostic code.
