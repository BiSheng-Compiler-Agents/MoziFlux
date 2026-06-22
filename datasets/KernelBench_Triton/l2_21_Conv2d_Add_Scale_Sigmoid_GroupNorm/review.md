# Triton Operator Static Code Review Report

## Basic Information
- Operator Name: Conv2d_Add_Scale_Sigmoid_GroupNorm
- Code File: opt_21_Conv2d_Add_Scale_Sigmoid_GroupNorm.py

## Host Side
| Check Item | Level | Status | Issue | Suggestion |
|------------|-------|--------|-------|-------------|
| Grid/Core Type | P0 | ✅ | Elementwise epilogue uses 1D Vector Core launches; no hardcoded physical core count. | Keep direct/persistent routing by tile count. |
| Block Configuration | P1-P2 | ✅ | `BLOCK_HW=4096` is constexpr and UB-safe for fp32 elementwise sigmoid. | Retune only with cannsim/hardware. |
| Parameter Validation | P2 | ✅ | Input device, autograd, and dtype are validated; module parameter devices follow `Module.to(...)`. | None. |
| Dispatch Coverage | P0 | ✅ | Direct path covers `total_tiles <= 65535`; persistent path covers larger valid shapes. | Profile tests include exact persistent target. |

## Device Side
| Check Item | Level | Status | Issue | Suggestion |
|------------|-------|--------|-------|-------------|
| Mask Completeness | P0 | ✅ | All vector `tl.load`/`tl.store` operations over data pointers are masked; scalar bias/scale loads are in-bounds by row-channel mapping. | None. |
| Data Type Compliance | P0-P1 | ✅ | No `tl.dot`, atomics, permute/trans, or unsupported dtype APIs. | None. |
| Precision Handling | P1 | ✅ | Epilogue upcasts input, bias, and scale to fp32 before sigmoid math. | Store cast is handled by output pointer dtype. |
| Code Patterns | P0-P2 | ✅ | No returns/breaks in kernels, no tensor indexing/slicing, no atomics. | None. |

## Performance Hazards
| Code Feature | Location | Suggestion |
|--------------|----------|-------------|
| Persistent loop adds control overhead | `_bias_scale_sigmoid_row_persistent` | Correctly gated only when `total_tiles > 65535`; keep direct path for legal shapes. |
| GroupNorm remains separate ACL op | `ModelNew.forward` | Acceptable: mature ACL op; fusing GroupNorm would require a separate reduction kernel and correctness risk. |
| Scalar setup still visible in cannsim | row-tiled kernels | Dominant baseline scalar storm is removed; remaining setup is small vs DMA/vector work. |

## Summary

### P0 Critical (Must Fix)
- None found.

### P1 Severe (Strongly Recommended to Fix)
- None found.

### P2 Suggestion (Optimization Items)
- Optional future work: hardware-test a fused GroupNorm epilogue only if ACL GroupNorm becomes the measured bottleneck.
