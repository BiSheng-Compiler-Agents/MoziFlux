---
name: triton-operator-code-review
description: Static review of Triton operator code quality (Host+Device side) for Ascend NPU. Identify potential bugs, API misuse, and performance hazards. Focuses only on static code analysis. Keywords: code review, static analysis.
---

# Triton Operator Static Code Review (Ascend NPU)

## Review Principles

- **Ascend-specific constraints first**: Focus on hardware differences, do not check general Triton knowledge
- **Mask zero tolerance**: Ascend has zero tolerance for out-of-bounds access

### Severity Levels

| Level | Meaning | Typical Issues |
|-------|---------|-----------------|
| **P0** | Guaranteed crash or incorrect results | Missing mask, core type mismatch, atomic loop deadlock |
| **P1** | High probability of precision/functionality issues | Reduction without upcasting, Softmax without max subtraction |
| **P2** | Performance/maintainability | Redundant memory access, unaligned BLOCK |

## Reference Resource Loading

| Phase | Load | Do Not Load |
|-------|------|--------------|
| Phase 1: Host Side | [`ascend-triton-api-constraints.md`](references/ascend-triton-api-constraints.md) | dtype-matrix |
| Phase 2: Device Side | [`ascend-api-dtype-matrix.md`](references/ascend-api-dtype-matrix.md) | test-patterns |
| Item-by-item Check | [`code-review-checklist.md`](references/code-review-checklist.md) | — |
| Reference Official Implementation | [`ascend-test-patterns.md`](references/ascend-test-patterns.md) | — |
| Confirm API Signature/Parameters | [`triton-api-reference.md`](../triton-operator-shared/references/triton-api-reference.md) | — |

**Load Trigger**: When needing to confirm the signature, parameters, or available data types of a tl/libdevice API, fully read `triton-api-reference.md`.

## Review Workflow

### Phase 1: Host Side

| Check Item | Identification Method | Level |
|------------|----------------------|-------|
| Hardcoded core count | Literals like `grid = (20,)` | P0 |
| Core type mismatch | Kernel containing `tl.dot` using `num_vectorcore` | P0 |
| Matrix multiplication degraded to element-wise | No `tl.dot`, using Vector Core element-wise multiply-add for matmul/GEMV | P0 |
| BLOCK_SIZE not `tl.constexpr` | Declaration check | P1 |
| Matrix operation BLOCK not multiple of 16 | Numeric check | P2 |

**Core Type**: Contains `tl.dot` → AI Core (`num_aicore`); element-wise/reduction/activation → Vector Core (`num_vectorcore`). Note: Matrix multiplication operators (including GEMV) must use `tl.dot` + AI Core, as Cube Core throughput is far higher than Vector Core.

### Phase 2: Device Side

**Mask Completeness (P0)**: All `tl.load`/`tl.store` must have `mask=` or use `make_block_ptr`.

**Data Types (P0-P1)**: `tl.dot` input only supports int8/fp16/fp32/bf16; `dot_scaled` has conditional support (bf16/fp16 lhs/rhs, int8 scale, fp32 out); `permute`/`trans` do not support int64.

**Precision Handling (P1)**: FP16/BF16 must `.to(tl.float32)` before reduction; Softmax must subtract max.

**Reduction Operator Single Pass Check (P1)**: For operators like GroupNorm/LayerNorm/RMSNorm, check if the same data is loaded multiple times (compute statistics first, then load again for normalization). When D ≤ UB, use single pass: after one load, use `tl.sum(x, 1)` to complete all computation within UB. Two-pass can be 5x slower.

**Control Flow (P0)**: Triton IR uses MLIR structured control flow (`scf.for`/`scf.while`) and does not support early exit.
- ❌ `return` inside `for/while` loops — Compilation error: "Cannot have return statements inside while or for" (including returns in child functions)
- ❌ `break` inside `for` loop — Compilation error: "unsupported AST node type: Break"

**Tensor Indexing (P0)**: Triton tensors do not support Python-style `[]` subscript operations.
- ❌ `tensor[i] = val` — AssertionError
- ❌ `val = tensor[i]` — AssertionError
- ❌ `tensor[i:j]` slicing — Compilation error

**Code Patterns**:
- ❌ `for ... : tl.atomic_cas/or/xor/and/xchg(...)` — May deadlock (P0)
- ❌ Using `tl.atomic_add` return value in multi-core kernel (P0)
- ❌ `import numpy` inside kernel (P0)
- ⚠️ `tensor[i].item()` in host hot path — Triggers CPU-NPU synchronization (P2)
- ⚠️ Multiple `for ... : tl.load(x_ptr + ...)` loops in reduction operators — Two-pass anti-pattern (P1)
- ⚠️ Post-dot operations not using `tl.parallel(bind_sub_block=True)` — Element-wise operations after `tl.dot` without parallel sharding (P2)
- ⚠️ Integer type conversion overflow — `tl.cast` to int8/int16 without specifying `overflow_mode` (P1)

### Phase 3: Performance Hazards (P2)

- Multiple `tl.load` of same pointer → Redundant GM access
- `tl.arange(0, N) * stride` (stride > 1) → Non-contiguous memory access
- Direct block mapping with `pid` without loop → Load imbalance
- Using broadcast stride for auxiliary tensors (cos/sin, etc.) inside kernel (`b * stride0 + n * stride1 + ...`) → Suggest host-side expand+contiguous with unified `row * D + col` offset
- Weight/bias not pre-expanded in reduction operators, using `weight_ptr + ch_idx` gather inside kernel → Suggest host-side expansion to contiguous layout

## Anti-Pattern Checklist (NEVER)

### Host
- ❌ Hardcoded core count / Using `num_vectorcore` for matrix multiplication / BLOCK_SIZE not `tl.constexpr`
- ❌ Matrix multiplication (including GEMV) implemented with Vector Core element-wise multiply-add — Must use `tl.dot` + AI Core

### Device
- ❌ `tl.load`/`tl.store` without mask / `tl.dot` input int32/int16/int64 / `dot_scaled`
- ❌ `atomic_or/xor/and/xchg/cas` inside for loop / Third-party libraries inside kernel
- ❌ FP16/BF16 reduction without upcasting to FP32 / Softmax without max subtraction
- ❌ Integer truncation cast without specifying `overflow_mode`
- ❌ `return` / `break` inside `for/while` loops — Triton structured control flow does not support early exit (including return in child functions)
- ❌ `tensor[i]` indexing operations (read/assign/slice) — Use `tl.where` / `tl.gather` / `tl.extract_slice` as alternatives

## Output

Output the report following the [`code-review-report-template.md`](references/code-review-report-template.md) format.