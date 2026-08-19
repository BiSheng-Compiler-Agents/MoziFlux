# Triton Operator Static Code Review Report

## Basic Information
- Operator Name: Matmul_MaxPool_Sum_Scale
- Code File: `opt_55_Matmul_MaxPool_Sum_Scale.py`

## Host Side
| Check Item | Level | Status | Issue | Suggestion |
|------------|-------|--------|-------|-------------|
| Grid/Core Type | P0 | ✅ | `tl.dot` fallback uses AI-core style capped 1D grid; production uses ACL. | None. |
| Block Configuration | P1-P2 | ✅ | `BLOCK_M=16`, `BLOCK_P=16`, `BLOCK_K=64`; Cube K/N dimensions are multiples of 16. | None. |
| Parameter Validation | P2 | ✅ | Preserves baseline constructor and NPU requirement; `kernel_size != 2` safely uses ACL path. | None. |
| Dispatch Coverage | P0 | ✅ | `profile_kernels.py` tests optimized ACL path and forced Triton fallback. | Keep fallback tests if dispatch thresholds change. |

## Device Side
| Check Item | Level | Status | Issue | Suggestion |
|------------|-------|--------|-------|-------------|
| Mask Completeness | P0 | ✅ | All pointer `tl.load`/`tl.store` operations in fallback have masks. | None. |
| Data Type Compliance | P0-P1 | ✅ | `tl.dot` operands are fp32 tensors and accumulators are fp32. | None. |
| Precision Handling | P1 | ✅ | Pair maxima and reductions use fp32; remote max_abs <= 9.54e-7 for fallback tests. | None. |
| Code Patterns | P0-P2 | ✅ | No tensor indexing, `break`, `return` in JIT loops, or atomics. | None. |

## Performance Hazards
| Code Feature | Location | Suggestion |
|--------------|----------|-------------|
| Production path delegates to ACL | `ModelNew.forward` | Appropriate for the large standard GEMM/pool/sum chain; cannsim fallback data is not claimed as production full-shape latency. |
| Fallback stores partials then reduces | `forward_triton_fallback` | Acceptable for correctness and cannsim coverage; for production custom-Triton-only use, consider fusing more reduction work per tile. |
| `Baseline Triton2` comparison unavailable | `profile_kernels.py` runtime | Read-only reference fails MLIR compilation; kept visible as `inf` rather than hiding the provider. |

## Summary

### P0 Critical (Must Fix)
- None.

### P1 Severe (Strongly Recommended to Fix)
- None.

### P2 Suggestion (Optimization Items)
- If future requirements disallow ACL production dispatch, tune the two-stage fallback further and benchmark a custom-Triton-only default path.
