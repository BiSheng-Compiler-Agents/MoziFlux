# Triton Operator Static Code Review Report

## Basic Information
- Operator Name: Gemm_GroupNorm_Min_BiasAdd
- Code File: `opt_75_Gemm_GroupNorm_Min_BiasAdd.py`

## Host Side

| Check Item | Level | Status | Issue | Suggestion |
|---|---|---|---|---|
| Grid/Core Type | P0 | ✅ | Production path uses ACL/CANN ops; fallback Triton path is vector/reduction-only and does not use `tl.dot`. | Keep fallback vector-only; do not launch it for large production shapes unless benchmarked. |
| Block Configuration | P1-P2 | ✅ | Fallback uses `BLOCK_N=16`, `BLOCK_C=16`; `C`, `N`, and strides are constexpr for compiler specialization. | The fallback is diagnostic/forced-test only; ACL dispatch remains default. |
| Dispatch Coverage | P0 | ✅ | Production ACL path and fallback path are both present. | `profile_kernels.py` includes a forced tiny fallback test. |
| Grid Limit | P0 | ✅ | Fallback checks both grid dimensions against `_MAX_GRID = 65535`. | Keep the guard before every fallback launch. |
| Parameter Validation | P2 | ✅ | Bias is reshaped and validated against `C`; fallback rejects non-2D normalized inputs. | No additional runtime shape restrictions beyond the original constructor contract. |

## Device Side

| Check Item | Level | Status | Issue | Suggestion |
|---|---|---|---|---|
| Mask Completeness | P0 | ✅ | Fallback `tl.load`/`tl.store` use masks for `N` and `C` boundaries. | Maintain masks if block sizes change. |
| Data Type Compliance | P0-P1 | ✅ | Fallback uses fp32 reductions after `.to(tl.float32)` and no unsupported `tl.dot` dtype. | Keep min reduction in fp32. |
| Precision Handling | P1 | ✅ | Production path matches PyTorch/ACL formula: `F.linear -> F.group_norm -> min(dim=1) -> bias add`. | Unit test threshold is `max_abs <= 2e-3`. |
| Code Patterns | P0-P2 | ✅ | No tensor indexing assignment, no atomics, no `break`, no return inside Triton control loops. | The fallback loop over `C` is scalar/MTE-heavy and should remain non-production unless improved. |

## Performance Hazards

| Code Feature | Location | Suggestion |
|---|---|---|
| Fallback recomputes the row minimum for each channel tile | `_min_bias_direct_kernel` | Accept for forced-test/cannsim diagnostics only; use production ACL dispatch for benchmark shapes. |
| Output layout `[1, C, N, 1]` makes per-row channel stores strided by `N` | Fallback store path | Prefer ACL broadcast add for production; if custom path is needed, use a two-phase contiguous row-min + channel-major fill. |
| Baseline reference file is not imported | `profile_kernels.py` | Intentional sandbox compliance: Baseline Triton2 remains parser-visible as `SKIP_UNAVAILABLE`. |

## Summary

### P0 Critical (Must Fix)
- None found.

### P1 Severe (Strongly Recommended to Fix)
- None found for the production path.

### P2 Suggestion (Optimization Items)
- Improve or remove the diagnostic fallback if production ever needs to disable ACL dispatch.
- Consider a two-phase Triton fallback for min and fill if native `torch.min`/broadcast add underperforms on future hardware.
