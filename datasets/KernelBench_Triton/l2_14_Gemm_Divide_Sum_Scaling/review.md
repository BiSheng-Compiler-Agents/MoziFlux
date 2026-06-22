# Triton Operator Static Code Review Report

## Basic Information
- Operator Name: Gemm_Divide_Sum_Scaling
- Code File: opt_14_Gemm_Divide_Sum_Scaling.py

## Host Side
| Check Item | Level | Status | Issue | Suggestion |
|------------|-------|--------|-------|-------------|
| Grid/Core Type | P0 | ✅ | Production path uses ACL GEMV; diagnostic Triton fallback uses `tl.dot` and `num_aicore`. | None. |
| Block Configuration | P1-P2 | ✅ | Fallback BLOCK_M=128, BLOCK_K=64, BLOCK_N=16 are Cube-aligned. | None. |
| Parameter Validation | P2 | ✅ | Checks NPU tensors, inference mode, rank-2 input, and K/input_size match. | None. |
| Dispatch Coverage | P0 | ✅ | `forward` production ACL path and `forward_triton` diagnostic fallback are covered in `profile_kernels.py`. | None. |

## Device Side
| Check Item | Level | Status | Issue | Suggestion |
|------------|-------|--------|-------|-------------|
| Mask Completeness | P0 | ✅ | Every `tl.load` and `tl.store` in the fallback has a mask. | None. |
| Data Type Compliance | P0-P1 | ✅ | GEMV fallback uses fp32/fp16-compatible `tl.dot` inputs and fp32 accumulator. | None. |
| Precision Handling | P1 | ✅ | Reduction is represented as fp32 summed weight plus fp32 GEMV for default fp32 inputs. | Keep rtol/atol 1e-3 gating in profiler. |
| Code Patterns | P0-P2 | ✅ | No tensor indexing, loop `return`, unsupported atomics, or `.cg` cache modifier in optimized fallback. | None. |

## Performance Hazards
| Code Feature | Location | Suggestion |
|--------------|----------|-------------|
| Triton fallback pads GEMV N=1 to N=16 | `forward_triton` / `_gemv_cube_kernel` | Keep it diagnostic-only unless hardware profiling proves it beats ACL. |
| Cached summed weight assumes inference semantics | `_effective_sum_weight` | Correctly invalidates on `weight._version`; avoid using this module for training. |

## Summary

### P0 Critical (Must Fix)
- None.

### P1 Severe (Strongly Recommended to Fix)
- None.

### P2 Suggestion (Optimization Items)
- Keep production dispatch on ACL GEMV; the cannsim fallback trace is slower for the bounded tile.
