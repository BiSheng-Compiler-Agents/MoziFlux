# Triton Operator Static Code Review Report

## Basic Information
- Operator Name: Conv2d_Min_Add_Multiply
- Code File: `opt_31_Conv2d_Min_Add_Multiply.py`

## Host Side
| Check Item | Level | Status | Issue | Suggestion |
|------------|-------|--------|-------|-------------|
| Grid/Core Type | P0 | ✅ | Pure vector epilogue uses vector-style element kernels; no `tl.dot`/Cube mismatch. Direct grid is computed from runtime shape, not hardcoded. | Keep direct path below `_MAX_PROGRAMS`; persistent path covers overflow. |
| Block Configuration | P1-P2 | ✅ | `BLOCK_HW` is `tl.constexpr` and fixed at 8192 for the epilogue. | Re-tune 4096/8192/16384 on hardware if future shapes change dtype or live-buffer pressure. |
| Parameter Validation | P2 | ✅ | Validates NPU tensor, 4D epilogue input, non-empty handling, and bias channel count. | No additional shape guards beyond baseline semantics. |

## Device Side
| Check Item | Level | Status | Issue | Suggestion |
|------------|-------|--------|-------|-------------|
| Mask Completeness | P0 | ✅ | All `tl.load`/`tl.store` operations use masks where boundary offsets are possible. Persistent bias load is valid because `tile_id < total_tiles` by loop construction. | Keep masks on HW tail stores. |
| Data Type Compliance | P0-P1 | ✅ | No `tl.dot`, atomics, unsupported tensor indexing, or third-party calls inside kernels. | N/A |
| Precision Handling | P1 | ✅ | Elementwise min/add/multiply preserves input dtype behavior and matches baseline/PyTorch semantics. No reduction precision issue. | N/A |
| Code Patterns | P0-P2 | ✅ | No `return`/`break` in JIT control flow; persistent loop iterates over tiles, not elements. | If adopting int64 offsets for >2GiB outputs, benchmark the overhead first. |

## Performance Hazards
| Code Feature | Location | Suggestion |
|--------------|----------|-------------|
| Persistent path has extra loop/control overhead | `_min_bias_scale_persistent_kernel` | Correctly gated to only `total_tiles > 65535`; unit-test-only persistent coverage is included in `profile_kernels.py`. |
| `x.contiguous()` may copy if conv output layout changes | `_apply_min_bias_scale` | Current Conv2d output is contiguous; retain for safe pointer arithmetic. |
| Bias conversion each forward | `bias_1d.to(...).contiguous()` | Necessary for dtype/device safety; can be revisited only if profiling shows host overhead. |

## Summary

### P0 Critical (Must Fix)
- None found.

### P1 Severe (Strongly Recommended to Fix)
- None found.

### P2 Suggestion (Optimization Items)
- Hardware-tune `BLOCK_HW` after remote verification; cannsim scalar probes do not represent the full 8192-element tile benefit.
