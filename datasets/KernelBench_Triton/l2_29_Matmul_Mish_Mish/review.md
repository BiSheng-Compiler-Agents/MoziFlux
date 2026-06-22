# Triton Operator Static Code Review Report

## Basic Information
- Operator Name: Matmul_Mish_Mish
- Code File: `opt_29_Matmul_Mish_Mish.py`

## Host Side
| Check Item | Level | Status | Issue | Suggestion |
|------------|-------|--------|-------|-------------|
| Grid/Core Type | P0 | ✅ | Triton fallback uses 1D grid capped by `min(total_tiles, _num_aicore(), 65535)` and queries `num_aicore` for `tl.dot`. | None. |
| Block Configuration | P1-P2 | ✅ | `BLOCK_M/N/K=64` are multiples of 16; all are `tl.constexpr`. | None. |
| Parameter Validation | P2 | ✅ | Checks device, rank, shape compatibility, dtype compatibility, bias shape/device/dtype, and empty output. | None. |
| Dispatch Coverage | P0 | ✅ | Production ACL path and `force_triton=True` fallback are covered in `profile_kernels.py`. | None. |

## Device Side
| Check Item | Level | Status | Issue | Suggestion |
|------------|-------|--------|-------|-------------|
| Mask Completeness | P0 | ✅ | All `tl.load` and `tl.store` operations use masks and safe `other=` values. | None. |
| Data Type Compliance | P0-P1 | ✅ | `tl.dot` consumes floating operands and accumulates fp32; no unsupported atomics, permutes, or indexing. | None. |
| Precision Handling | P1 | ✅ | Matmul accumulator is fp32; output is cast back to destination dtype after the fused Mish epilogue. | Keep tolerance check at rtol/atol 1e-3. |
| Code Patterns | P0-P2 | ✅ | No `break`, no early `return` inside Triton loops, no Python tensor indexing in JIT. | None. |

## Performance Hazards
| Code Feature | Location | Suggestion |
|--------------|----------|-------------|
| Production path delegates to ACL rather than custom Triton for GEMM | `matmul_mish_mish()` | Accepted for this standard large GEMM; fallback Triton kernel remains traced and tested. |
| Fused fallback Mish epilogue is vector-exp heavy | `_mish_twice()` | Expected cost of Mish twice; production path is preferred for target latency. |
| `Baseline Triton1` target grid would exceed Ascend cap | `profile_kernels.py` guard | Pre-skip target baseline timing to avoid NPU context poisoning. |

## Summary

### P0 Critical (Must Fix)
- None.

### P1 Severe (Strongly Recommended to Fix)
- None.

### P2 Suggestion (Optimization Items)
- Consider a fully custom Triton large-GEMM path only if future hardware measurements show ACL is slower than a tuned Cube kernel for this exact model regime.
