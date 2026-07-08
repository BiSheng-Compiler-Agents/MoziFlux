# Triton Operator Static Code Review Report

## Basic Information
- Operator Name: Matmul_GELU_Softmax
- Code File: `opt_99_Matmul_GELU_Softmax.py`

## Host Side
| Check Item | Level | Status | Issue | Suggestion |
|------------|-------|--------|-------|-------------|
| Grid/Core Type | P0 | ✅ | Triton epilogue is vector/reduction only and launches one row per program; GEMM is routed to ACL. | None. |
| Block Configuration | P1-P2 | ✅ | `BLOCK_N` is constexpr and power-of-two; Triton path limited to N <= 512 to stay within UB headroom. | Retune threshold if hardware changes. |
| Parameter Validation | P2 | ✅ | Shape, dtype, and device consistency checks are preserved from the baseline. | None. |
| Dispatch Coverage | P0 | ✅ | Small shape exercises Triton epilogue; medium/target exercise ACL fallback. | Keep both paths in profile tests. |

## Device Side
| Check Item | Level | Status | Issue | Suggestion |
|------------|-------|--------|-------|-------------|
| Mask Completeness | P0 | ✅ | All `tl.load`/`tl.store` calls have masks. | None. |
| Data Type Compliance | P0-P1 | ✅ | No `tl.dot`; reductions upcast to fp32. | None. |
| Precision Handling | P1 | ✅ | GELU is exact erf form; softmax subtracts row max before exp. | None. |
| Code Patterns | P0-P2 | ✅ | No tensor indexing, atomics, break/return inside device loops, or third-party calls. | None. |

## Performance Hazards
| Code Feature | Location | Suggestion |
|--------------|----------|-------------|
| ACL fallback for medium/large rows | `matmul_gelu_softmax` | Intentional: remote hardware shows ACL is faster than Triton epilogue for 2048/8192-wide rows. |
| `torch.empty_like(z)` warning on NPU internal format | Triton small-row path | Warning is benign in verification; if it becomes noisy, allocate with `torch.empty(z.shape, device=z.device, dtype=z.dtype)`. |
| Comparison baseline preskip | `profile_kernels.py` | Preskip avoids hanging on row-wise vector GEMM at target size while preserving optimized correctness tests. |

## Summary

### P0 Critical (Must Fix)
- None.

### P1 Severe (Strongly Recommended to Fix)
- None.

### P2 Suggestion (Optimization Items)
- Consider replacing `empty_like` with explicit `torch.empty` to avoid the NPU internal-format warning.
- If a future custom Cube GEMM is added, compare it against ACL on hardware before routing target shapes away from ACL.
