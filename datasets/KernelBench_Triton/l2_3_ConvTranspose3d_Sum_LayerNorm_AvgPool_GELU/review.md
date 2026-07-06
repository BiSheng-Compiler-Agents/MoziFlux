# Triton Operator Static Code Review Report

## Basic Information
- Operator Name: ConvTranspose3d + Sum + LayerNorm + AvgPool3d + GELU
- Code File: `opt_3_ConvTranspose3d_Sum_LayerNorm_AvgPool_GELU.py`

## Host Side

| Check Item | Level | Status | Issue | Suggestion |
|------------|-------|--------|-------|------------|
| Grid/Core Type | P0 | ✅ | Tiny Triton kernels are vector/reduction kernels and use normal Triton launch; medium/default path avoids custom grid overflow. | Keep default path on ACL unless Triton grid is restructured below 65,535 CTAs. |
| Block Configuration | P1-P2 | ✅ | `BLOCK_SIZE`/`BLOCK_W` are constexpr and power-of-two bounded to 1024. | No P1 blocker. |
| Dispatch Coverage | P0 | ✅ | Tiny Triton and ACL fallback paths are both covered in `profile_kernels.py`. | Keep threshold tests in profiling when changing `_TRITON_POST_MAX_NUMEL`. |
| Parameter Validation | P2 | ✅ | Inherited baseline NPU and LayerNorm-shape guards remain. | No new unsupported-shape guard was added. |

## Device Side

| Check Item | Level | Status | Issue | Suggestion |
|------------|-------|--------|-------|------------|
| Mask Completeness | P0 | ✅ | All `tl.load` and `tl.store` operations use masks. | None. |
| Data Type Compliance | P0-P1 | ✅ | Reductions upcast input values to fp32 before mean/variance and pooling accumulation. | None. |
| Precision Handling | P1 | ✅ | LayerNorm variance and AvgPool accumulation are fp32; remote max error <= 1e-3. | None. |
| Code Patterns | P0-P2 | ✅ | No tensor indexing, `break`, or returns inside Triton loops. | None. |

## Performance Hazards

| Code Feature | Location | Suggestion |
|--------------|----------|------------|
| Scalar decode in AvgPool3d/GELU Triton kernel | Tiny fallback `_avgpool3d_gelu_kernel` | Acceptable only for tiny path; medium/default use ACL. |
| `tl.erf` GELU in custom Triton path | Tiny fallback `_avgpool3d_gelu_kernel` | Keep exact GELU for correctness; ACL path handles larger tensors. |
| One-row cannsim probe does not represent full dispatch latency | `cannsim_*_ln` harness | Use hardware `remote_verify` numbers for production dispatch decisions. |

## Summary

### P0 Critical (Must Fix)
- None.

### P1 Severe (Strongly Recommended to Fix)
- None.

### P2 Suggestion (Optimization Items)
- Future work: if a fully custom Triton path is needed for medium/default shapes, rewrite the epilogue with grid-capped persistent scheduling or multi-launch tiling instead of restoring the baseline grid.
