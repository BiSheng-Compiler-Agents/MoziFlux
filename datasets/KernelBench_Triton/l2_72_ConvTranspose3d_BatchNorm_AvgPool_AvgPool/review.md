# Triton Operator Static Code Review Report

## Basic Information
- Operator Name: ConvTranspose3d_BatchNorm_AvgPool_AvgPool
- Code File: `opt_72_ConvTranspose3d_BatchNorm_AvgPool_AvgPool.py`

## Host Side
| Check Item | Level | Status | Issue | Suggestion |
|------------|-------|--------|-------|-------------|
| Grid/Core Type | P0 | ✅ | Production path uses ACL `avg_pool3d`; Triton fallback uses vector-style code and caps dispatch at `_MAX_GRID=65535`. | Keep direct path below cap and persistent path only above cap. |
| Block Configuration | P1-P2 | ✅ | `ROWS_PER_CTA=8`, `BLOCK_W=32`; output tile is 256 lanes with masks for `OW<32`. | Tune `ROWS_PER_CTA` on hardware if a different row-block size wins. |
| Parameter Validation | P2 | ✅ | No new runtime shape guards were added; dimensions come from runtime tensor shape and compile as constexpr meta. | Keep `x.contiguous()` before launching because pointer math assumes contiguous NCDHW. |

## Device Side
| Check Item | Level | Status | Issue | Suggestion |
|------------|-------|--------|-------|-------------|
| Mask Completeness | P0 | ✅ | Every `tl.load` and `tl.store` has `mask=mask`. | None. |
| Data Type Compliance | P0-P1 | ✅ | No `tl.dot`; loads are converted to fp32 for accumulation and stored to output dtype. | None. |
| Precision Handling | P1 | ✅ | 64-way average accumulates in fp32 before multiplying by `1/64`. | Preserve fp32 accumulation for fp16/bf16 inputs. |
| Code Patterns | P0-P2 | ✅ | No tensor indexing, break/return in kernels, atomics, or unmasked memory operations. | None. |

## Performance Hazards
| Code Feature | Location | Suggestion |
|--------------|----------|-------------|
| Scalar row decomposition (`//`/`%`) | row-block direct/persistent kernels | Current cost is amortized across 8 rows; if hardware profiling shows scalar pressure, tune `ROWS_PER_CTA` or specialize default `OH/OW`. |
| Persistent fallback | `_avg_pool3d_k4s4_rowblock_persistent_kernel` | Unit test forces this path by lowering `_MAX_GRID`; benchmark production path should remain direct at default shape. |
| `x.contiguous()` copy | host wrapper | BatchNorm output is expected contiguous; keep the call for safety, but hardware profiling can check whether it is a no-op. |

## Summary

### P0 Critical (Must Fix)
- None found.

### P1 Severe (Strongly Recommended to Fix)
- None found.

### P2 Suggestion (Optimization Items)
- Consider hardware autotuning `ROWS_PER_CTA` in `{4, 8, 16}` after remote verification.
- If persistent fallback is performance-critical for larger shapes, add a dedicated large-shape hardware benchmark; cannsim micro-probes do not measure full FFTS dispatch behavior.
