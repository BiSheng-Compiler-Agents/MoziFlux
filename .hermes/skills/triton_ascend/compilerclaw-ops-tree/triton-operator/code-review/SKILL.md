# Triton Operator Static Code Review [LEAF NODE]

Static review of Triton operator code quality (Host+Device side) for Ascend NPU.
Identify potential bugs, API misuse, and performance hazards.
Keywords: code review, static analysis.

## Review Principles

- **Ascend-specific constraints first**: Focus on hardware differences, do not check general Triton knowledge
- **Mask zero tolerance**: Ascend has zero tolerance for out-of-bounds access

### Severity Levels

| Level | Meaning | Typical Issues |
|-------|---------|-----------------| 
| **P0** | Guaranteed crash or incorrect results | Missing mask, core type mismatch, atomic loop deadlock |
| **P1** | High probability of precision/functionality issues | Reduction without upcasting, Softmax without max subtraction |
| **P2** | Performance/maintainability | Redundant memory access, unaligned BLOCK |

---

## Review Workflow

### Phase 1: Host Side

| Check Item | Identification Method | Level |
|------------|----------------------|-------|
| Hardcoded core count | Literals like `grid = (20,)` | P0 |
| Core type mismatch | Kernel containing `tl.dot` using `num_vectorcore` | P0 |
| Matrix multiplication degraded to element-wise | No `tl.dot`, using Vector Core element-wise multiply-add for matmul/GEMV | P0 |
| Shape-specific kernel branch with no fallback | `if C == 16: fast_kernel` with no else clause — other shapes crash or produce wrong results | P0 |
| Untested dispatch path | Fast path and generic path exist but only one has unit tests | P0 |
| New runtime guards not in baseline | `if out_channels > 256: raise` or `if sum_dim != 1: raise` when baseline had no such constraint | P0 |
| BLOCK_SIZE not `tl.constexpr` | Declaration check | P1 |
| Matrix operation BLOCK not multiple of 16 | Numeric check | P2 |

**Core Type**: Contains `tl.dot` → AI Core (`num_aicore`); element-wise/reduction/activation → Vector Core (`num_vectorcore`). Matrix multiplication operators (including GEMV) must use `tl.dot` + AI Core.

```python
import triton.runtime.driver as driver
device = torch.npu.current_device()
# Contains tl.dot
num_aicore = driver.active.utils.get_device_properties(device)["num_aicore"]
# Does not contain tl.dot
num_vectorcore = driver.active.utils.get_device_properties(device)["num_vectorcore"]
```

### Phase 2: Device Side

**Mask Completeness (P0)**: All `tl.load`/`tl.store` must have `mask=` or use `make_block_ptr`.

```python
# ❌ Missing mask
x = tl.load(x_ptr + offsets)

# ✅ With mask
x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

# ✅ make_block_ptr (handles boundaries automatically)
block_ptr = tl.make_block_ptr(
    base=ptr, shape=(M, N), strides=(stride_m, stride_n),
    offsets=(pid_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))
data = tl.load(block_ptr)
```

**Data Types (P0-P1)**:
- `tl.dot` input only supports int8/fp16/fp32/bf16
- `dot_scaled` — conditionally supported (bf16/fp16 lhs/rhs, int8 scale, fp32 out)
- `permute`/`trans` do not support int64

**tl.dot dtype matrix:**
| int8 | int16 | int32 | int64 | fp16 | fp32 | bf16 |
|------|-------|-------|-------|------|------|------|
| ✓ | × | × | × | ✓ | ✓ | ✓ |

Accumulator: `tl.float32` for floating point, `tl.int32` for int8.

**Precision Handling (P1)**:
- FP16/BF16 must `.to(tl.float32)` before reduction
- Softmax must subtract max: `tl.exp(x - max_x)`
- Convert back to target precision before output

```python
# Matrix Multiplication Precision Pattern:
if dtype == "int8":
    accumulator_type = tl.int32
else:
    accumulator_type = tl.float32
accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)
for k in range(0, tl.cdiv(K, BLOCK_K)):
    a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
    accumulator = tl.dot(a, b, accumulator, out_dtype=acc_dtype)
c = accumulator.to(c_ptr.dtype.element_ty)
```

**Reduction Operator Single Pass Check (P1)**: For GroupNorm/LayerNorm/RMSNorm, check if the
same data is loaded multiple times. When D ≤ UB, use single pass: after one load, use
`tl.sum(x, 1)` to complete all computation within UB. Two-pass can be 5× slower.

**Control Flow (P0)**: Triton IR uses MLIR structured control flow (`scf.for`/`scf.while`).
- ❌ `return` inside `for/while` loops — Compilation error: "Cannot have return statements inside while or for" (including returns in child functions)
- ❌ `break` inside `for` loop — Compilation error: "unsupported AST node type: Break"

```python
# ❌ return inside loop
for i in range(N):
    if cond:
        return val

# ✅ Alternative: tl.where mask (recommended)
for i in range(N):
    result = tl.where(active_mask, compute(i), result)
```

**Tensor Indexing (P0)**: Triton tensors do not support Python-style `[]` subscript operations.
- ❌ `tensor[i] = val` — AssertionError
- ❌ `val = tensor[i]` — AssertionError
- ❌ `tensor[i:j]` slicing — Compilation error

```python
# ❌ Index assignment
local_vector[i] = 123.0  # AssertionError

# ✅ tl.where as alternative
idx = tl.arange(0, BLOCK_SIZE)
local_vector = tl.where(i == idx, 123.0, local_vector)

# ❌ Indexing to get value
val = tensor[i]

# ✅ tl.gather
val = tl.gather(tensor, index, axis=0)

# ❌ Slicing
subx = x[1:3]

# ✅ al.extract_slice
import triton.language.extra.cann.extension as al
subx = al.extract_slice(x, offsets=(1,), sizes=(2,), strides=(1,))
```

**Code Patterns (P0)**:
- ❌ `for ... : tl.atomic_cas/or/xor/and/xchg(...)` — May deadlock
- ❌ Using `tl.atomic_add` return value in multi-core kernel
- ❌ `import numpy` inside kernel
- ⚠️ `tensor[i].item()` in host hot path — Triggers CPU-NPU synchronization (P2)
- ⚠️ Multiple `for ... : tl.load(x_ptr + ...)` loops in reduction operators — Two-pass anti-pattern (P1)
- ⚠️ Post-dot operations not using `tl.parallel(bind_sub_block=True)` — Element-wise ops after `tl.dot` without parallel sharding (P2)
- ⚠️ Integer type conversion overflow — `tl.cast` to int8/int16 without specifying `overflow_mode` (P1)

### Phase 3: Performance Hazards (P2)

- Multiple `tl.load` of same pointer → Redundant GM access
- `tl.arange(0, N) * stride` (stride > 1) → Non-contiguous memory access
- Direct block mapping with `pid` without loop → Load imbalance
- Using broadcast stride for auxiliary tensors (cos/sin, etc.) inside kernel → Suggest host-side expand+contiguous
- Weight/bias not pre-expanded in reduction operators → Suggest host-side expansion to contiguous layout

---

## BLOCK_SIZE Constraints

| Check Item | How to Identify in Code |
|------------|-------------------------|
| BLOCK_SIZE not constexpr | Function parameter missing `: tl.constexpr` declaration |
| Matrix BLOCK not multiple of 16 | Literals like `BLOCK_M=100`, `BLOCK_N=50` |
| BLOCK_K not aligned | Not calculated according to `kalign = 32 // dtype_bytes` |

```python
# BLOCK_K alignment (from official test cases)
dtype_bytes = torch.tensor(0, dtype=eval('torch.' + dtype)).element_size()
kalign = 32 // dtype_bytes
BLOCK_K = min(max(K, kalign), 32)
```

---

## Atomic Operations Constraints

| Code Pattern | Issue |
|--------------|-------|
| `for ...: tl.atomic_cas/or/xor/and/xchg(...)` | Not supported in loops |
| `ret = tl.atomic_add(...)` and using `ret` in multi-core kernel | Multi-core add + saving intermediate results not supported |

**Atomic dtype matrix:**

| Op | int8 | int16 | int32 | uint32 | int64 | fp16 | fp32 | bf16 |
|----|------|-------|-------|--------|-------|------|------|------|
| atomic_add | ✓ | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ |
| atomic_cas | × | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × |
| atomic_max/min | ✓ | ✓ | ✓ | × | × | ✓ | ✓ | ✓ |

---

## Precision Validation Reference Tolerances

Criteria: **MERE < threshold AND MARE < 10 × threshold**

| Data Type | Threshold | MERE Upper Bound | MARE Upper Bound |
|-----------|-----------|------------------|------------------|
| float16 | 2⁻¹⁰ ≈ 9.77e-4 | 9.77e-4 | 9.77e-3 |
| bfloat16 | 2⁻⁷ ≈ 7.81e-3 | 7.81e-3 | 7.81e-2 |
| float32 | 2⁻¹³ ≈ 1.22e-4 | 1.22e-4 | 1.22e-3 |
| Integer/bool | exact match | exact match | exact match |

---

## Anti-Pattern Checklist (NEVER)

### Host
- ❌ Hardcoded core count / Using `num_vectorcore` for matrix multiplication / BLOCK_SIZE not `tl.constexpr`
- ❌ Matrix multiplication (including GEMV) with Vector Core element-wise multiply-add — Must use `tl.dot` + AI Core

### Device
- ❌ `tl.load`/`tl.store` without mask / `tl.dot` input int32/int16/int64
- ❌ `atomic_or/xor/and/xchg/cas` inside for loop / Third-party libraries inside kernel
- ❌ FP16/BF16 reduction without upcasting to FP32 / Softmax without max subtraction
- ❌ Integer truncation cast without specifying `overflow_mode`
- ❌ `return` / `break` inside `for/while` loops (including return in child functions)
- ❌ `tensor[i]` indexing operations (read/assign/slice)

---

## Output Format

Output the review report in this format:

```markdown
# Triton Operator Static Code Review Report

## Basic Information
- Operator Name: [Name]
- Code File: [Path]

## Host Side
| Check Item | Level | Status | Issue | Suggestion |
|------------|-------|--------|-------|-------------|
| Grid/Core Type | P0 | ✅/❌ | | |
| Block Configuration | P1-P2 | ✅/❌ | | |
| Parameter Validation | P2 | ✅/❌ | | |

## Device Side
| Check Item | Level | Status | Issue | Suggestion |
|------------|-------|--------|-------|-------------|
| Mask Completeness | P0 | ✅/❌ | | |
| Data Type Compliance | P0-P1 | ✅/❌ | | |
| Precision Handling | P1 | ✅/❌ | | |
| Code Patterns | P0-P2 | ✅/❌ | | |

## Performance Hazards
| Code Feature | Location | Suggestion |
|--------------|----------|-------------|

## Summary

### P0 Critical (Must Fix)
- [List all P0 issues]

### P1 Severe (Strongly Recommended to Fix)
- [List all P1 issues]

### P2 Suggestion (Optimization Items)
- [List all P2 issues]
```

## Constraints
- Focus on static analysis only — do not run the code
- Ascend-specific constraints have priority over general Triton knowledge
- Always check mask completeness first — it's the most common P0 issue
- Check for two-pass reduction pattern — extremely common P1 performance issue

---

## Full API & Constraint References

These files are part of this tree and contain the authoritative API and constraint details:

- **`../../shared/references/ascend-test-patterns.md`** (169 lines) — Official test case patterns; contains patterns NOT in the review workflow above:
  - Section 3: FlashAttention pattern — `tl.make_block_ptr` + `tl.advance` for QK/V iteration, large HEAD_DIM slicing with `extension.extract_slice/insert_slice`
  - Section 4: Rotary Embedding — composite 3D mask `(token_range < num_tokens) & (head_range < num_heads)`, multi-dim offsets with `stride_state_n/h/d`
  - Section 5: Reduction — `tl.reduce` with custom combine function (`_reduce_combine`), dim as constexpr, conditional store by dim
  - Section 6: Precision validation pattern with `torch_reference` wrapper + exact tolerance values (FP16 rtol/atol 1e-3, FP32 1e-4, integers exact)

- **`../../shared/references/ascend-triton-api-constraints.md`** (256 lines) — Constraint details for review:
  - Section 1: Masking — `make_block_ptr` + `tl.advance` as alternative to explicit mask
  - Section 2: BLOCK_SIZE constraints with `kalign = 32 // dtype_bytes` formula
  - Section 6: Specific op constraints with test-case enablement status (dot_scaled conditions, sort 1D-5D, gather multi-axis, permute 3D (2,1,0) compatibility)
  - Section 9: Full tensor indexing constraints with `tl.full` alternative, `al.extract_slice` for slicing
  - Section 10: All high-performance APIs with constraints: `tl.parallel(bind_sub_block=True)`, `tl.load(care_padding=False)`, `tl.cast(overflow_mode=...)`, `sync_block_set/wait`, `index_select_simd`

- **`../../shared/references/ascend-api-dtype-matrix.md`** (120 lines) — Complete dtype support matrix:
  - `tl.dot` supported types (int8/fp16/fp32/bf16 only — not int32/int64)
  - `tl.arange` int32 only
  - `permute`/`trans`: no int64, 3D (2,1,0) compatibility note
  - `gather`: axes 0-4 supported
  - Atomic ops matrix: `atomic_add/cas/max/min` by dtype
  - Scan ops (`associative_scan`, `cumsum`/`cumprod`) dtype support
  - Precision validation tolerance table: MERE/MARE thresholds for FP16/BF16/FP32/int

- **`../../shared/references/code-review-checklist.md`** (66 lines) — Full static review checklist:
  - Host Interface Design: input shape assert, dtype validation, device consistency `x.device == y.device`, boundary/empty input handling
  - Grid: 1D recommended, size reasonable
  - Performance hazards: Sufficient data reuse, contiguous memory access, no redundant loads

- **`../../shared/references/code-review-report-template.md`** (35 lines) — Output report format template

- **`../../shared/references/triton-api-reference.md`** — Complete Triton-Ascend API reference for verifying correct API usage during review:
  - Section 4 (AL extension): All op signatures, parameter types, error conditions for `al.sync_block_set/wait`, `al.scope`, `al.cast`, `al.fixpipe`, `al.copy`, `@register_custom_op` field requirements, `al.parallel` bind_sub_block constraint (max 2 on 910B), memory ops (`al.index_put`, `al.gather_out_to_ub`, `al.scatter_ub_to_out`, `al.index_select_simd`) — all with exact error types raised on misuse
  - Section 5 (BL extension): `bl.alloc`, `bl.subview` 32-byte alignment rules — common source of silent misalignment bugs
  - Section 7 (NPUOptions): Compiler flag interactions — e.g. `sync_solver` must be True when using `al.sync_block_set/wait`; `force_simt_only` vs `parallel_mode` conflicts; `enable_sync_block_lock` requirement
  - Hardware notes: 910_95-only APIs that silently fail on other hardware

- **`../../shared/references/ascend-terminology.md`** — Hardware terminology for identifying constraint violations:
  - HIVM IR mapping — helps identify which Triton ops map to which hardware pipeline and where conflicts arise
  - UB constraints, alignment rules
