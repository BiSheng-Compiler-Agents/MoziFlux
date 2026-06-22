# Triton Operator Static Code Review Report

## Basic Information
- Operator Name: BMM_InstanceNorm_Sum_ResidualAdd_Multiply
- Code File: opt_28_BMM_InstanceNorm_Sum_ResidualAdd_Multiply.py

## Host Side
| Check Item | Level | Status | Issue | Suggestion |
|------------|-------|--------|-------|-------------|
| Grid/Core Type | P0 | ✅ | Pure vector/reduction kernels use row-grid vector dispatch; no `tl.dot` in the custom kernel. Direct path is used for `B <= 65535`; persistent path caps larger grids. | Keep direct path for normal batch sizes. |
| Block Configuration | P1-P2 | ✅ | `BLOCK` is `tl.constexpr` and derived from `features`. No matrix BLOCK constraints apply to the post-linear vector kernel. | Guard very large feature widths if future shapes exceed UB. |
| Parameter Validation | P2 | ✅ | Shape, dtype, device, and bias checks are preserved from the baseline host interface. | None. |

## Device Side
| Check Item | Level | Status | Issue | Suggestion |
|------------|-------|--------|-------|-------------|
| Mask Completeness | P0 | ✅ | All `tl.load` and `tl.store` operations are masked. | None. |
| Data Type Compliance | P0-P1 | ✅ | No unsupported `tl.dot`, atomics, transpose, or third-party calls. | None. |
| Precision Handling | P1 | ✅ | Row reductions upcast inputs to FP32 and use `inv_F` multiplication instead of division. | Keep tolerance at `rtol=1e-3, atol=1e-3` in hardware tests. |
| Code Patterns | P0-P2 | ✅ | No `break`, `return` in JIT loops, tensor indexing, or atomics. Persistent loop iterates rows by runtime `n_programs`. | None. |

## Performance Hazards
| Code Feature | Location | Suggestion |
|--------------|----------|-------------|
| `F.linear` dominates target workload | Host `forward()` | This correctly uses mature ACL GEMM; do not rewrite as vector loops. |
| Row width equals 8192 | `_rownorm_addmul_*` | Single-pass UB-resident row normalization is appropriate; reassess if `out_features` grows much larger. |
| cannsim sub-kernel PUSHQ bottleneck | trace summaries | Hardware full-shape timing is required because grid-level dispatch effects are not visible at `grid=1`. |

## Summary

### P0 Critical (Must Fix)
- None.

### P1 Severe (Strongly Recommended to Fix)
- None.

### P2 Suggestion (Optimization Items)
- Consider an ACL-only post-op candidate if hardware shows the custom row kernel is slower for small/irregular shapes.
