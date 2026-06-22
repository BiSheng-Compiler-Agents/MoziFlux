# Triton Operator Static Code Review Report

## Basic Information
- Operator Name: cumsum_exclusive
- Code File: `opt_92_cumsum_exclusive.py`

## Host Side
| Check Item | Level | Status | Issue | Suggestion |
|------------|-------|--------|-------|------------|
| Grid/Core Type | P0 | ✅ | Vector/scan kernel uses vector-core-style 1D grid; no `tl.dot`/Cube mismatch. | Keep 1D grid. |
| Grid Cap | P0 | ✅ | Fallback launches `min(rows, 65535)` programs. | Covered by unit-only fallback tests. |
| Block Configuration | P1-P2 | ✅ | `BLOCK_SIZE` is `tl.constexpr`; dispatch uses 128/256/512 according to `N`. | No change. |
| Parameter Validation | P2 | ✅ | Preserves baseline NPU, 2D, dim=1/-1, dtype checks. | No new restrictive guard added. |

## Device Side
| Check Item | Level | Status | Issue | Suggestion |
|------------|-------|--------|-------|------------|
| Mask Completeness | P0 | ✅ | Tile `tl.load` and shifted `tl.store` use `mask=col < n_cols_in`; zero-column store is within the row loop. | No P0 mask issue. |
| Data Type Compliance | P0-P1 | ✅ | Inputs are loaded as fp16/bf16/fp32 and upcast to fp32 for prefix accumulation. | No unsupported dtype use. |
| Precision Handling | P1 | ✅ | Fallback uses fp32 prefix state before storing to output dtype; production ACL reference uses `torch.cumsum`. | No change. |
| Code Patterns | P0-P2 | ✅ | No `break`, Python tensor indexing inside JIT, unmasked OOB access, atomics, or hardcoded target shape. | No P0/P1 issue. |

## Performance Hazards
| Code Feature | Location | Suggestion |
|--------------|----------|------------|
| `tl.cumsum` lowers to scalar-heavy scan on Ascend | `_exclusive_cumsum_vectorized_kernel` | Kept as fallback only; production dispatch uses ACL cumsum. |
| Non-contiguous `y[:, 1:]` host assignment | `ModelNew.forward` production path | Acceptable because ACL cumsum dominates baseline; benchmark verifies real hardware latency. |

## Summary

### P0 Critical (Must Fix)
- None found.

### P1 Severe (Strongly Recommended to Fix)
- None found.

### P2 Suggestion (Optimization Items)
- The Triton fallback remains scalar-limited by Ascend scan lowering; keep production traffic on ACL unless a true vector prefix primitive becomes available.
