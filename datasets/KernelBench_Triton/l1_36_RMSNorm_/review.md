# Triton Operator Static Code Review Report

## Basic Information
- Operator Name: RMSNorm
- Code File: `opt_36_RMSNorm_.py`

## Host Side
| Check Item | Level | Status | Issue | Suggestion |
|------------|-------|--------|-------|-------------|
| Grid/Core Type | P0 | ✅ | Pure vector/reduction kernel, no `tl.dot`; 1D grid capped at 65535. | Keep persistent tile loop for target shape. |
| Block Configuration | P1-P2 | ✅ | `BLOCK_HW` and `BLOCK_C` are `tl.constexpr`; `BLOCK_C` bounded to 64/128/256. | Tune `BLOCK_HW` on hardware if medium shapes matter. |
| Parameter Validation | P2 | ✅ | Validates NPU device, 4D NCHW, contiguous input, and feature dimension. | None. |
| Dispatch Coverage | P0 | ✅ | Persistent path covers both normal and `total_tiles > 65535` cases. | Profile includes small/irregular/medium/target. |

## Device Side
| Check Item | Level | Status | Issue | Suggestion |
|------------|-------|--------|-------|-------------|
| Mask Completeness | P0 | ✅ | Every `tl.load`/`tl.store` has channel and HW masks plus `other=0.0` on loads. | None. |
| Data Type Compliance | P0-P1 | ✅ | No unsupported atomics, dot, tensor indexing, slicing, return/break in loops. | None. |
| Precision Handling | P1 | ✅ | Input is upcast to FP32 before `x*x` reduction; output preserves input dtype. | None. |
| Code Patterns | P0-P2 | ✅ | Uses `tl.range` persistent loop; no Python tensor indexing in kernel. | None. |

## Performance Hazards
| Code Feature | Location | Suggestion |
|--------------|----------|-------------|
| NCHW channel-strided loads | `_rmsnorm_nchw_hw_kernel` | Intentional tradeoff to remove full-tensor NHWC permutes; target hardware improves 2.159x vs PyTorch/ACL. |
| Two reads of input | reduction pass + store pass | Required unless storing the full `(BLOCK_C, BLOCK_HW)` tile; current UB-safe approach is acceptable. |
| Medium shape slower than baseline2 | `medium` benchmark | Consider a small-shape direct 2D-grid path if optimizing medium shapes, while retaining persistent path for target. |

## Summary

### P0 Critical (Must Fix)
- None.

### P1 Severe (Strongly Recommended to Fix)
- None.

### P2 Suggestion (Optimization Items)
- Optional future tuning: add a direct non-persistent path for medium shapes if the benchmark target changes away from the oversized 112×64×512×512 case.
