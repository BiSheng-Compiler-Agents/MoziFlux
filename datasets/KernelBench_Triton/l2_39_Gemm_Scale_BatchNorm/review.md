# Triton Operator Static Code Review Report

## Basic Information
- Operator Name: Gemm + Scale + BatchNorm
- Code File: `opt_39_Gemm_Scale_BatchNorm.py`

## Host Side

| Check Item | Level | Status | Issue | Suggestion |
|------------|-------|--------|-------|------------|
| Grid/Core Type | P0 | ✅ | `tl.dot` kernel uses AI-core count (`num_aicore`) and caps grid to `65535`. | Keep the fallback only as a last resort for compile-only environments. |
| Block Configuration | P1-P2 | ✅ | `BLOCK_M=128`, `BLOCK_N=128`, `BLOCK_K=32`, all multiples of 16. | Good for Cube alignment and UB budget. |
| Parameter Validation | P2 | ✅ | Preserves baseline NPU/dtype/eval/autograd guards. | No new shape-specific rejection was added. |
| Dispatch Coverage | P0 | ✅ | Single fused dispatch path handles arbitrary `M,K,N` through masks and constexpr recompilation. | Profile unit test covers small, medium, irregular, and required shapes. |

## Device Side

| Check Item | Level | Status | Issue | Suggestion |
|------------|-------|--------|-------|------------|
| Mask Completeness | P0 | ✅ | All `tl.load` and `tl.store` operations have masks. | None. |
| Data Type Compliance | P0-P1 | ✅ | `tl.dot` consumes floating tensors and accumulates in FP32. | Keep input cast to FP32 to match baseline semantics. |
| Precision Handling | P1 | ✅ | GEMM accumulator is FP32; BatchNorm affine is computed/stored in FP32. | Verify on hardware with `profile_kernels.py --test`. |
| Code Patterns | P0-P2 | ✅ | No Python slicing/indexing inside JIT, no `break`/`return` inside loops, no atomics. | None. |

## Performance Hazards

| Code Feature | Location | Suggestion |
|--------------|----------|------------|
| Cached tensors depend on parameter versions | Host cache | Current cache key includes data pointer, shape/dtype/device, and `_version`; keep this if training support is ever added. |
| Full-shape latency not measured on physical NPU | `performance_report.md` | Run `remote_verify` or `profile_kernels.py --test --bench` on Ascend hardware. |
| FP32 GEMM is faithful but expensive | Fused kernel | If tolerance allows in a future task, test FP16/BF16 input/weight path separately; do not change precision silently. |

## Summary

### P0 Critical (Must Fix)
- None found.

### P1 Severe (Strongly Recommended to Fix)
- None found; `remote_verify` unit tests passed for Optimized Triton on all profiled dispatch paths.

### P2 Suggestion (Optimization Items)
- Consider hardware autotuning `BLOCK_M/N/K` after physical NPU latency is available.
