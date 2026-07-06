# Triton Operator Static Code Review Report

## Basic Information
- Operator Name: Gemm_LogSumExp_LeakyReLU_LeakyReLU_GELU_GELU
- Code File: opt_64_Gemm_LogSumExp_LeakyReLU_LeakyReLU_GELU_GELU.py

## Host Side

| Check Item | Level | Status | Issue | Suggestion |
|------------|-------|--------|-------|-------------|
| Grid/Core Type | P0 | ✅ | Direct fallback uses one vector program per row; persistent fallback caps programs at `_MAX_PROGRAMS=65535`. | Keep direct path for normal batches; use persistent only when needed. |
| Block Configuration | P1-P2 | ✅ | `BLOCK_N` is constexpr and capped at 1024. | Re-profile if changing `BLOCK_N`; row LSE is PUSHQ/MTE dominated. |
| Parameter Validation | P2 | ✅ | Host preserves original NPU-only input expectation and constructor signature. | None. |
| Dispatch Coverage | P0 | ✅ | Production ACL path plus direct/persistent Triton fallbacks are exposed. | `profile_kernels.py` force-tests both fallback paths. |

## Device Side

| Check Item | Level | Status | Issue | Suggestion |
|------------|-------|--------|-------|-------------|
| Mask Completeness | P0 | ✅ | All row loads use `mask=offs < N`; stores write one valid output per row. | None. |
| Data Type Compliance | P0-P1 | ✅ | Reductions upcast loaded values to fp32; no unsupported `tl.dot`/integer casts. | None. |
| Precision Handling | P1 | ✅ | LogSumExp uses online max/sum; GELU remains exact erf-based. | Keep exact GELU unless tolerance policy changes. |
| Code Patterns | P0-P2 | ✅ | No `break`/`return` in kernels; no tensor indexing; persistent loop iterates rows with capped program count. | None. |

## Performance Hazards

| Code Feature | Location | Suggestion |
|--------------|----------|-------------|
| Fallback row kernel is PUSHQ/front-end dominated | `_rowwise_lse_leaky_gelu2_*` | Keep CANN/ACL as production path for standard large row `logsumexp` chain unless hardware proves custom Triton faster. |
| Large GEMM output is still materialized before LogSumExp | `ModelNew.forward` | A fully fused GEMM+online-LSE kernel would require multi-stage partial reductions and separate proof of fp32 numerical equivalence. |

## Summary

### P0 Critical (Must Fix)
- None found.

### P1 Severe (Strongly Recommended to Fix)
- None found.

### P2 Suggestion (Optimization Items)
- Future work: investigate true fused GEMM+row LogSumExp if CANN/ACL production path is insufficient on hardware.
