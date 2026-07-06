# Triton Operator Static Code Review Report

## Basic Information
- Operator Name: ConvTranspose3d_Swish_GroupNorm_HardSwish
- Code File: `opt_60_ConvTranspose3d_Swish_GroupNorm_HardSwish.py`

## Host Side
| Check Item | Level | Status | Issue | Suggestion |
|------------|-------|--------|-------|------------|
| Grid/Core Type | P0 | ✅ | Production path uses ACL. Triton fallback uses 1D grids and persistent cap `_MAX_GRID=65535`. | Keep direct/persistent gate by tile count. |
| Block Configuration | P1-P2 | ✅ | `_REDUCE_BLOCK=8192`, `_APPLY_BLOCK=4096`; fallback unit-tested direct and persistent. | If enabling fallback by default, retune on hardware because cannsim microprobe is scalar-heavy. |
| Parameter Validation | P2 | ✅ | Constructor and `run_operator` interface preserved. | No new unsupported runtime guard added. |
| Base reference handling | P2 | ✅ | `base_*.py` was not read per sandbox rule; profiler keeps Baseline Triton2 visible as skipped. | Continue not reading/modifying reference files. |

## Device Side
| Check Item | Level | Status | Issue | Suggestion |
|------------|-------|--------|-------|------------|
| Mask Completeness | P0 | ✅ | All fallback `tl.load`/`tl.store` operations have `mask=`. | Maintain zero-tolerance OOB masking. |
| Data Type Compliance | P0-P1 | ✅ | Reductions and normalization use FP32; no unsupported `tl.dot` or int64 tensor ops. | Keep FP32 stats for GroupNorm. |
| Precision Handling | P1 | ✅ | Swish/GroupNorm/HardSwish semantics match reference; remote max_abs is 0 on production path. | Use algebraic HardSwish, not unsupported `F.hardswish` kernel. |
| Code Patterns | P0-P2 | ✅ | No `break`/`return` inside JIT loops; persistent loop iterates over tiles, not elements. | None. |

## Performance Hazards
| Code Feature | Location | Suggestion |
|--------------|----------|------------|
| Fallback grouped reduction has scalar index decode (`//`, `%`) | `_swish_group_reduce_parts_3d` | Keep fallback disabled by default unless future hardware results justify it. |
| Production path launches multiple ACL kernels | `_acl_post` | Acceptable: remote latency matches PyTorch/ACL and beats baseline custom Triton on comparable shapes. |
| Baseline default benchmark preskipped | `profile_kernels.py` | Preskip avoids poisoning NPU context from oversized/slow comparison launch; optimized default is still tested and timed. |

## Summary

### P0 Critical (Must Fix)
- None found.

### P1 Severe (Strongly Recommended to Fix)
- None found.

### P2 Suggestion (Optimization Items)
- If fallback is ever made production-default, reduce scalar divisions in `_swish_group_reduce_parts_3d` or specialize for common `group_size/H/W` values.
