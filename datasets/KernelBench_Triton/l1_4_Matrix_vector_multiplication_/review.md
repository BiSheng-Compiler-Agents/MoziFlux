# Triton Operator Static Code Review Report

## Basic Information
- Operator Name: Matrix-vector multiplication
- Code File: `opt_4_Matrix_vector_multiplication_.py`

## Host Side
| Check Item | Level | Status | Issue | Suggestion |
|------------|-------|--------|-------|------------|
| Grid/Core Type | P0 | ✅ | Uses vector-core count for a vector reduction kernel and caps grid to tile count. | None |
| Block Configuration | P1-P2 | ✅ | `BLOCK_M=64`, `BLOCK_K=512`; fp32 A tile is 128 KiB plus small B/acc buffers. | Keep cannsim coverage for larger block changes. |
| Parameter Validation | P2 | ✅ | Preserves baseline validation for 2D inputs, `B.shape=(K,1)`, dtype/device checks. | None |
| Dispatch Coverage | P0 | ✅ | Single dispatch path covers small/non-power-of-two/benchmark shapes via masks. | Unit tests in `profile_kernels.py` cover the path. |

## Device Side
| Check Item | Level | Status | Issue | Suggestion |
|------------|-------|--------|-------|------------|
| Mask Completeness | P0 | ✅ | All `tl.load` and `tl.store` operations are masked. | None |
| Data Type Compliance | P0-P1 | ✅ | Inputs are fp16/bf16/fp32; reduction upcasts tiles to fp32 before accumulation. | None |
| Precision Handling | P1 | ✅ | Accumulator is fp32 and output stores back to destination dtype. | Validate hardware tolerance with `profile_kernels.py --test`. |
| Code Patterns | P0-P2 | ✅ | No tensor indexing, no return/break in JIT loops, no atomics. | None |

## Performance Hazards
| Code Feature | Location | Suggestion |
|--------------|----------|------------|
| Vector GEMV rather than Cube GEMV | `_gemv_vector_opt_kernel` | Cube path was tested and regressed for `N=1`; keep vector path unless future hardware trace shows padded `tl.dot` wins. |
| Large fp32 tile | `BLOCK_K=512` | Safe in cannsim; if adding more live tensors, re-check UB budget. |

## Summary

### P0 Critical (Must Fix)
- None found.

### P1 Severe (Strongly Recommended to Fix)
- None found.

### P2 Suggestion (Optimization Items)
- Re-run physical NPU profiling when `remote_verify` is available; local cannsim cannot provide end-to-end hardware latency.
