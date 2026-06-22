# Triton Operator Static Code Review Report

## Basic Information
- Operator Name: LayerNorm
- Code File: `opt_40_LayerNorm.py`

## Host Side
| Check Item | Level | Status | Issue | Suggestion |
|------------|-------|--------|-------|-------------|
| Grid/Core Type | P0 | ✅ | Vector/reduction kernels use 1-D grids; no `tl.dot`/AI-core mismatch. | None. |
| Block Configuration | P1-P2 | ✅ | `BLOCK_SIZE` is constexpr; reduction uses 16384 and apply uses 8192; reduction finalize caps partial tile count to a safe 8192. | Keep persistent path for `total_tasks > 65535`. |
| Parameter Validation | P2 | ✅ | Preserves baseline NPU-only behavior and handles empty tensors by returning `empty_like`. | None. |
| Dispatch Coverage | P0 | ✅ | Direct and persistent partial/apply paths exist; `profile_kernels.py` includes `persistent_dispatch` unit coverage. | None. |

## Device Side
| Check Item | Level | Status | Issue | Suggestion |
|------------|-------|--------|-------|-------------|
| Mask Completeness | P0 | ✅ | All pointer `tl.load`/`tl.store` operations have masks, except scalar mean/rstd/partial writes with in-range program IDs. | None. |
| Data Type Compliance | P0-P1 | ✅ | No unsupported `tl.dot`, atomics, tensor indexing, `break`, or in-loop returns. | None. |
| Precision Handling | P1 | ✅ | Input, weight, bias, partial sums, mean, and variance compute in FP32; output stores to destination dtype. | None. |
| Code Patterns | P0-P2 | ✅ | Uses private partials instead of atomics; persistent loops iterate over tiles/tasks, not elements. | None. |

## Performance Hazards
| Code Feature | Location | Suggestion |
|--------------|----------|-------------|
| Three kernel launches plus partial GM buffer | partial/finalize/apply path | Required for `M=4,194,304`; a single program cannot fit the row in UB. |
| `torch.nn.functional.layer_norm` fallback for `num_tiles > 8192` | host path | Correctness fallback for extreme normalized shapes; not expected for benchmark regime. |

## Summary

### P0 Critical (Must Fix)
- None.

### P1 Severe (Strongly Recommended to Fix)
- None.

### P2 Suggestion (Optimization Items)
- Hardware benchmark should confirm full-grid atomic-removal benefit because grid=1 cannsim cannot expose atomic contention.
