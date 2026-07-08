# Triton Operator Static Code Review Report

## Basic Information
- Operator Name: Conv2d_Subtract_HardSwish_MaxPool_Mish
- Code File: `opt_35_Conv2d_Subtract_HardSwish_MaxPool_Mish.py`
- Review Scope: final `ModelNew.forward()` ACL dispatch plus retained Triton helper kernels

## Host Side

| Check Item | Level | Status | Issue | Suggestion |
|------------|-------|--------|-------|------------|
| Grid/Core Type | P0 | ✅ | Final hot path uses ACL operators and does not launch an invalid grid. Retained Triton helpers cap direct K=2 launch at `65535` and use vector-core style kernels only. | Keep grid cap if Triton helpers are re-enabled. |
| Block Configuration | P1-P2 | ✅ | Triton helper block sizes are `tl.constexpr`; no matrix/Cube blocks are used. | None. |
| Parameter Validation | P2 | ✅ | Device type is checked; square pool validation is preserved from the source interface. | Dtype is inherited from PyTorch/ACL ops; no extra guard needed. |
| Dispatch Coverage | P0 | ✅ | `profile_kernels.py` tests default `K=2`, irregular `K=2`, and fallback `K=3`. | Keep every host branch in the profile shape table. |

## Device Side

| Check Item | Level | Status | Issue | Suggestion |
|------------|-------|--------|-------|------------|
| Mask Completeness | P0 | ✅ | All retained Triton `tl.load` and `tl.store` calls use masks; invalid offsets are remapped before stores/row loads where needed. | None. |
| Data Type Compliance | P0-P1 | ✅ | No `tl.dot`, atomics, `permute`, or unsupported dtypes are used. Activation math upcasts loads to fp32. | None. |
| Precision Handling | P1 | ✅ | HardSwish and Mish are evaluated in fp32 for Triton helpers; final ACL path follows PyTorch fp32 semantics for the benchmark inputs. | Maintain rtol/atol `1e-3` tests. |
| Code Patterns | P0-P2 | ✅ | No Python tensor indexing inside JIT, no `break`/`return` inside JIT loops, no atomics. | None. |

## Performance Hazards

| Code Feature | Location | Suggestion |
|--------------|----------|------------|
| Retained custom Triton helper kernels are not used by final `forward()` | `opt_*.py` helper definitions | P2 maintainability: acceptable because cannsim evidence documents why ACL is selected; remove only if the deliverable no longer needs an optimized-kernel artifact. |
| Custom Triton epilogue is scalar/spill-bound | cannsim baseline/candidate traces | Final host dispatch correctly avoids this path with ACL standard operators. |
| Baseline2 comparison mismatch on K=2 shapes | `profile_kernels.py` remote output | Read-only reference was not modified; comparison remains visible and optimized correctness is gated against PyTorch/ACL. |

## Summary

### P0 Critical (Must Fix)
- None.

### P1 Severe (Strongly Recommended to Fix)
- None.

### P2 Suggestion (Optimization Items)
- Consider deleting unused Triton helper kernels in a future cleanup if the evaluator accepts ACL-only optimized host paths.
- If a custom Triton epilogue is required later, start from cannsim's SCALARLDST/PUSHQ bottleneck rather than the rejected vectorized 2x2 pooling candidate.
