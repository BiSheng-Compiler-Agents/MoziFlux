# Triton Operator Static Code Review Report

## Basic Information
- Operator Name: Gemm_GroupNorm_Hardtanh
- Code File: opt_30_Gemm_GroupNorm_Hardtanh.py

## Host Side
| Check Item | Level | Status | Issue | Suggestion |
|------------|-------|--------|-------|-------------|
| Grid/Core Type | P0 | ✅ | Vector-only kernels; direct path uses 1D tile grid, persistent path caps at 65,535. | None. |
| Block Configuration | P1-P2 | ✅ | `BLOCK_SIZE` and `GROUP_BLOCK` are constexpr; target live tile is `4*512` elements. | Keep `GROUP_BLOCK` bounded by UB budget. |
| Parameter Validation | P2 | ✅ | Preserves baseline `C % G` assertion and NPU-only contract. | Add dtype assertions only if the public baseline contract is tightened. |

## Device Side
| Check Item | Level | Status | Issue | Suggestion |
|------------|-------|--------|-------|-------------|
| Mask Completeness | P0 | ✅ | All loads/stores have masks and `other=` where applicable. | None. |
| Data Type Compliance | P0-P1 | ✅ | No `tl.dot`; reductions upcast to fp32. | None. |
| Precision Handling | P1 | ✅ | Mean/variance computed in fp32 and clamp applied in fp32 before store. | None. |
| Code Patterns | P0-P2 | ✅ | No tensor indexing, early returns, atomics, or third-party calls inside kernels. | None. |

## Performance Hazards
| Code Feature | Location | Suggestion |
|--------------|----------|-------------|
| Persistent loop | `_groupnorm_hardtanh_groupblock_persistent_kernel` | Only used when `total_tiles > 65535`; direct path avoids loop overhead at target. |
| 2D tile offsets | group-block kernel | Kept to contiguous per-group channel spans; GROUP_BLOCK capped to avoid UB overflow. |

## Summary

### P0 Critical (Must Fix)
- None found.

### P1 Severe (Strongly Recommended to Fix)
- None found.

### P2 Suggestion (Optimization Items)
- Future tuning could benchmark `GROUP_BLOCK=2` vs `4` on hardware for very small `Cg`, but target `Cg=512` is UB-safe with `4`.
