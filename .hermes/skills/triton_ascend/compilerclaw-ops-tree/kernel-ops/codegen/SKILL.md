# Triton Kernel Code Generation [LEAF NODE]

Generate and optimize Triton kernel code for Ascend NPU operators. Use when implementing a
new Triton operator (from a design doc or direct requirements), or when optimizing an existing
one for performance. Covers the full loop: code generation → cannsim bottleneck analysis →
episode-driven optimization → episode recording.

## Core Principles

**Compute Logic → Tiling Strategy → Code Implementation → cannsim Profiling → Optimization**
The order must not be reversed. Never optimize blindly — always profile first.

## Scope Rule

> Cover ALL shapes the kernel signature accepts — not just the benchmark shape.
> Read the kernel's **actual parameter list** to understand what varies at runtime (N, C, H, W, dtype).
> Never infer test shapes from other benchmark directories.
>
> Before writing any kernel, answer:
> - What are the runtime parameters and their ranges?
> - Does per-program startup cost dominate for small inputs? → persistent grid
> - Does tile count create excessive sync stalls for large inputs? → tune BLOCK size
> - Profile at least: one small case, one large case, one non-power-of-2 case

---

## Workflow

### Step 1: Retrieve Past Episodes

```python
episode_retrieve(query="<kernel_type> <bottleneck or technique>", target="ascend950", limit=5)
```

Examples:
```python
episode_retrieve(query="matmul low cube utilization dot pad static range", target="ascend950")
episode_retrieve(query="softmax wide rows MTE stall online reduction", target="ascend950")
episode_retrieve(query="norm scalar overhead two-pass single-pass reduction", target="ascend950")
episode_retrieve(query="elementwise large N FFTS dispatch persistent grid", target="ascend950")
episode_retrieve(query="relu fp16 maximum VEC slow fp32 RVECEX upcast", target="ascend950")
```

Episodes IDs 12–40 cover 30 patterns tagged by `kernel_name`:
`matmul`, `softmax`, `rmsnorm`, `layernorm`, `gelu`, `relu`, `maxpool`, `elementwise`,
`hingeloss`, `cross_entropy`, `embedding`, `general`

Browse the DB:
```bash
sqlite3 ~/CompilerClaw/episodes.db \
  "SELECT id, kernel_name, substr(observation,1,80) FROM episodes ORDER BY id DESC"
```

---

### Step 2: Understand Requirements → Confirm Compute Logic

Extract mathematical formulas, input/output specifications, constraints.
Understand the hardware architecture before tiling design:

**Ascend NPU Architecture:**
- Each AI Core: 1 Cube Core (matrix multiply, L1 Buffer 1MB) + 2 Vector Cores (UB 192KB)
- GM (DDR) → UB via MTE2 → Vector Core → MTE3 → GM
- GM → L1 → Cube Core → L1 → GM (for matrix ops)

**Core type selection:**
```python
import triton.runtime.driver as driver
props = driver.active.utils.get_device_properties("npu")
num_cube_cores   = props["num_aicore"]     # for kernels with tl.dot
num_vector_cores = props["num_vectorcore"] # for pure vector kernels
```

---

### Step 3: Implement Tiling Strategy

**Inter-Core Partitioning — two rules:**
1. `grid = number of physical cores` (never more, never multidimensional)
2. Each core's intra-core loop processes multiple tiles (load balancing)

```python
grid = (max(1, min(core_num, triton.cdiv(total_tiles, 1))),)

# Inside kernel:
pid = tl.program_id(0)
num_core = tl.num_programs(0)
for tile_idx in tl.range(pid, total_tiles, num_core):
    ...
```

**UB budget:**
```python
safe_BLOCK = int((196608 - 32) / (num_buffers × dtype_size) × 0.65)
# For 3 fp32 buffers: 3 × BLOCK × 4 ≤ 32KB → BLOCK ≤ 2048
```

**Operator type → core type → template:**
| Operator Type | Core | Template |
|---|---|---|
| Reduction (norm, softmax, loss) | Vector | Template 1 |
| GEMM / Attention | AI Core (Cube) | Templates 2, 6 |
| Activation / Loss / Index / MoE | Vector | Templates 3–5, 7–8 |
| Convolution | AI Core | Template 9 |

---

### Step 4: Generate Kernel Code

**Universal rules — apply to every kernel without exception:**

1. Add alignment hints to every offset array:
   ```python
   offsets = pid * BLOCK + tl.arange(0, BLOCK)
   tl.multiple_of(offsets, 16)
   tl.max_contiguous(offsets, BLOCK)
   ```

2. Always use `tl.range` (not Python `range`) for inner loops.

3. Always upcast to fp32 before `tl.maximum`, `tl.sum`, or `tl.dot` accumulation.

4. Cap any grid dimension to 65535 (FFTS hard limit — silent crash above, no error).

5. `tl.dot` is the ONLY instruction that activates the Cube engine. Even GEMV must use `tl.dot` with B padded to BLOCK_N=16.

**Template 1 — Reduction Operators (GroupNorm, LayerNorm, RMSNorm, Softmax):**
```python
@triton.jit
def group_norm_kernel(x_ptr, y_ptr, M, D, eps, rows_per_prog, MBLOCK, NBLOCK):
    pid = tl.program_id(0)
    row_start = pid * rows_per_prog
    col_off = tl.arange(0, NBLOCK)
    col_mask = col_off < D
    for mb in range(0, rows_per_prog, MBLOCK):
        row_idx = row_start + mb + tl.arange(0, MBLOCK)
        tot_off = row_idx[:, None] * D + col_off[None, :]
        tot_mask = (row_idx < M)[:, None] & col_mask[None, :]
        x = tl.load(x_ptr + tot_off, mask=tot_mask, other=0.0).to(tl.float32)
        # Single pass: one load, all computation within UB
        mean = tl.sum(x, 1) / D
        var = tl.sum(x * x, 1) / D - mean * mean
        y = (x - mean[:, None]) * tl.rsqrt(var[:, None] + eps)
        tl.store(y_ptr + tot_off, y, mask=tot_mask)
```

**Template 2 — GEMM:**
```python
@triton.jit
def matmul_kernel(mat_a, mat_b, mat_c, M, N, K, num_cores,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                  BLOCK_THRESHOLD: tl.constexpr = 4):
    pid = tl.program_id(axis=0)
    NUM_BLOCKS_M = triton.cdiv(M, BLOCK_M)
    NUM_BLOCKS_N = triton.cdiv(N, BLOCK_N)
    NUM_BLOCKS = NUM_BLOCKS_M * NUM_BLOCKS_N
    blocks_per_core = triton.cdiv(NUM_BLOCKS, num_cores)
    block_start = pid * blocks_per_core
    block_end = triton.minimum(block_start + blocks_per_core, NUM_BLOCKS)
    for block_idx in range(block_start, block_end):
        # Diagonal scheduling for large matrices (L2 cache hit rate)
        if NUM_BLOCKS_M >= BLOCK_THRESHOLD and NUM_BLOCKS_N >= BLOCK_THRESHOLD:
            task_m_idx = block_idx % NUM_BLOCKS_M
            task_n_idx = (block_idx // NUM_BLOCKS_M) % NUM_BLOCKS_N
        else:
            task_m_idx = block_idx // NUM_BLOCKS_N
            task_n_idx = block_idx % NUM_BLOCKS_N
        m_start = task_m_idx * BLOCK_M
        n_start = task_n_idx * BLOCK_N
        mat_c_block = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            mat_a_block = tl.load(mat_a + ..., mask=..., other=0.0, care_padding=False)
            al.compile_hint(mat_a_block, "dot_pad_only_k")
            mat_b_block = tl.load(mat_b + ..., mask=..., other=0.0, care_padding=False)
            al.compile_hint(mat_b_block, "dot_pad_only_k")
            mat_c_block = tl.dot(mat_a_block, mat_b_block, mat_c_block)
        tl.store(mat_c + ..., mat_c_block.to(tl.bfloat16), mask=...)
```

**Template 3 — Activation:**
```python
@triton.jit
def activation_kernel(a_ptr, b_ptr, c_ptr, stride, n_rows, n_cols: tl.constexpr,
                      rows_per_core: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row_start = pid * rows_per_core
    row_end = tl.minimum(row_start + rows_per_core, n_rows)
    for row_idx in range(row_start, row_end):
        for i in range(0, n_cols, BLOCK_SIZE):
            offsets = tl.arange(0, BLOCK_SIZE) + i
            mask = offsets < n_cols
            a = tl.load(a_ptr + row_idx * stride + offsets, mask=mask, other=0).to(tl.float32)
            b = tl.load(b_ptr + row_idx * stride + offsets, mask=mask, other=0)
            c = a * tl.sigmoid(a) * b  # SwiGLU pattern
            tl.store(c_ptr + row_idx * stride + offsets, c.to(b.dtype), mask=mask)
```

---

### Step 5: Generate Correctness Test

Write a smoke test (single shape × single dtype) in the host file to verify the kernel
compiles, runs, and produces correct results via `cannsim_remote_run`.

---

### Step 6: Profile with cannsim

After the kernel runs correctly, **always profile before optimizing**:

```python
result = cannsim_remote_run(local_dir="...", run_script="run_kernel.sh", gen_report=True)
```

Read trace using aggregate script:
```bash
python <tree_root>/kernel-ops/simulation/scripts/aggregate_trace.py \
    /path/to/report/trace_core0.json
```

Identify the bottleneck:

| cannsim Metric | Bottleneck | First Action |
|---|---|---|
| `aiv_scalar > 80%` | Scalar-bound | Switch to single-pass + tl.sum(x,1); check Python scalar loop-carried state |
| `aiv_mte2 > 50%` | Memory-bound | Contiguous access, expand+contiguous for aux tensors, increase BLOCK |
| `aiv_vec > 50%` | VEC-bound | fp16 tl.maximum → upcast to fp32 (fp16 → slow VEC, fp32 → RVECEX) |
| `aic_cube_ratio < 50%` | Low Cube util | Check BLOCK multiples of 16; add `al.compile_hint(a, "dot_pad_only_k")` |
| `SCALARLDST high` | Scalar register spill | Replace Python float/int loop accumulators with `tl.zeros([1], tl.float32)` |
| PUSHQ high, many VF fences | tl.zeros inside loop | Hoist before `tl.range`; use `num_stages=2` |
| `instr.bin < 1 MB`, `~28 cycles`, 100% SCALAR | UB overflow (silent!) | Reduce BLOCK; fp32 safe limit = 2048 |

---

### Step 7: Optimize Based on Bottleneck

Retrieve relevant episodes first, then apply patterns. Priority order:

1. Block/Grid size tuning (highest ROI — block size alone can give 7.5×)
2. Memory access contiguity (2D broadcast → expand+contiguous on host)
3. Single-pass reduction (replace two-pass; tl.sum(x,1) stays in UB)
4. Compiler hints (`al.compile_hint`, `tl.multiple_of`, `tl.max_contiguous`)
5. Persistent grid for small-HW or large-N kernels
6. Double-buffering (`al.multibuffer(tensor, 2)`)
7. Cube-Vector pipeline sync (`al.sync_block_set/wait`) for mixed kernels

```python
import triton.language.extra.cann.extension as al

# dot_pad_only_k: pad only K dimension, reducing UB usage 30-50%
al.compile_hint(acc, "dot_pad_only_k")

# Double buffering
a = al.multibuffer(a, size=2)  # only size=2 supported

# Cast with overflow control
y = al.cast(x, tl.int8, overflow_mode="saturate")
```

---

### Step 8: Record New Patterns as Episodes

**After every successful optimization, write an episode.** This is mandatory.

```python
episode_write(
    kernel_name="softmax",
    target="ascend950",
    observation="Baseline: one-row-per-program, aiv_scalar=85%, 3605 µs.",
    thoughts="Scalar overhead from 2D broadcast. expand().contiguous() eliminates broadcast stride.",
    action="Host-side: cos_flat = cos.expand(shape).contiguous().reshape(rows, D). Kernel: uniform row*D+col offset.",
    result="3605 µs → 752 µs (4.8×). aiv_mte2 now bottleneck. Contiguity > data reuse count.",
)
```

---

## Anti-Pattern Checklist (NEVER)

- Write code without confirming compute logic first
- Optimize without cannsim profiling — never guess the bottleneck
- Ignore UB size (192KB) / grid > physical core count / grid dimension > 65535
- Reduce in FP16 / matrix multiply with BLOCK not multiple of 16
- Use third-party libraries or element-wise compute inside kernel
- GEMV with Vector Core element-wise multiply-add — use `tl.dot` + pad B to BLOCK_N=16
- Two-pass reduction (load x multiple times) — use single-pass `tl.sum(x, 1)` in UB
- Python scalar loop-carried state inside `tl.range` — use `tl.zeros([1], tl.float32)` accumulators
- `tl.zeros([N], dtype)` inside `tl.range` body — hoist before loop, use `num_stages=2`
- `tl.maximum` on fp16 — always upcast to fp32 first
- `num_stages > 2` — only 1 or 2 on Ascend
- `return` / `break` inside `for`/`while` loops — use `tl.where` mask instead
- `tensor[i]` indexing — use `tl.where`, `tl.gather`, `al.extract_slice` instead
- Deliver a kernel that only works for the benchmark shape
- Skip writing an episode after a successful optimization

## Common Pitfalls Quick Reference

| Symptom | Cause | Fix |
|---|---|---|
| `aiv_scalar > 80%` | Two-pass / Python scalar accumulators | Single-pass; `tl.zeros([1], fp32)` accumulators |
| `SCALARLDST` high | Python float/int in `tl.range` loop | `tl.zeros([1], tl.float32)` instead |
| PUSHQ / VF fences high | `tl.zeros` inside loop body | Hoist before `tl.range`; `num_stages=2` |
| `instr.bin < 1MB`, 100% SCALAR | Silent UB overflow | Reduce BLOCK; fp32 limit = BLOCK ≤ 2048 |
| `aic_cube_ratio < 50%` | M/N/K not ×16, or no pad hint | Multiples of 16; `compile_hint("dot_pad_only_k")` |
| `coreDim > UINT16_MAX` crash | Grid dimension > 65535 | `min(grid, 65535)` + persistent loop |
| Output all zeros, kernel "completes" | Silent UB overflow | Check `instr.bin` size; reduce BLOCK |
| `addptrRes.hasOneUse()` assertion | Same loaded pointer used for two loads | Load into separate variables |
| `num_stages=1` + hoisted `tl.zeros` crash | bishengir register conflict | Use `num_stages=2` |

## Constraints
- Always call `episode_retrieve` before writing any kernel code
- Always call `episode_write` after every successful optimization
- Use `cannsim_remote_run` with `gen_report=True` for all profiling
- Cover ALL shapes — never deliver a kernel that only works for the benchmark shape

---

## Full Kernel Templates (Templates 4–12)

The codegen workflow above covers Templates 1–3. The following templates are in:
**`../../shared/references/templates.md`** — all 12 templates with full runnable code:

- **Template 2 (extended)**: Full `@triton.autotune` GEMM with complete pointer-offset math, `care_padding=False`, post-dot parallelization via `al.parallel(bind_sub_block=True)` + `al.extract_slice`, double buffering via `al.multibuffer`, `al.cast(overflow_mode=...)` usage
- **Template 4**: CrossEntropy loss — online softmax algorithm, `ignore_index` handling, FP32 reduction throughout
- **Template 5**: Index Transformation (ConvertIndex/RoPE) — token index remapping with block-table pointer, nested tile loops
- **Template 6**: Attention — QK^T kernel with `tl.make_block_ptr` + `tl.advance`; Cube-Vector signal sync (`al.sync_block_set/wait`, `event_id` 0–15)
- **Template 7**: MoE gating (FusedGDNgating) — softplus/sigmoid activation, 2D program grid
- **Template 8**: Post-processing (ExpandBatchToTokens) — cumulative token expansion kernel
- **Template 9**: Convolution — state management, sliding window, continuous batching flag
- **Template 10**: Contiguous Expand Pattern (RoPE) — host-side `expand().contiguous().reshape()`, uniform `row * D + col` offset in kernel
- **Template 11**: Embedding/Gather — `tl.gather`, `index_select_simd` (dim < ndim-1 constraint), `index_put` scatter write
- **Template 12**: Sort/Flip — `tl.sort` last-dim-only constraint, descending sort via ascending + flip

Also in **`../../shared/references/hardware-architecture.md`** (214 lines):
- `TRITON_ALL_BLOCKS_PARALLEL=1` env var for merged grid distribution
- INT64 avoidance: Vector ADD/CMP do not support int64 → cast to FP32 for comparisons
- Platform differences table: A2/A3 vs 910_95 (`multibuffer` default, `auto_bind_sub_block`, FP8, `sync_solver`, `inject_block_all`, `overflow_mode="saturate"`)
- Full GM/UB/L1 memory hierarchy diagram + Cube-Vector collaboration data path
- `coreDim > UINT16_MAX` solutions: increase BLOCK, set `TRITON_ALL_BLOCKS_PARALLEL=1`, use intra-core loops

---

## Full API & Compiler References

These files are part of this tree and must be consulted when using any API beyond the patterns above:

- **`../../shared/references/triton-api-reference.md`** — Complete Triton-Ascend API reference:
  - Section 1–3: `triton`, `triton.language` (tl), `triton.testing` — all ops, shapes, dtypes
  - Section 4 (AL extension): Full enumerations (CORE, PIPE, MODE, FixpipeDMAMode, SYNC_IN_VF), all ops: `al.copy`, `al.fixpipe`, `al.debug_barrier`, `al.sync_block_set/wait`, `al.scope`, `al.custom`/`@register_custom_op`, math ops (`al.atan2`, `al.isfinited`, `al.finitef`), auxiliary ops (`al.parallel`, `al.compile_hint`, `al.multibuffer`), vector ops (`al.insert_slice`, `al.extract_slice`, `al.get_element`, `al.sort`, `al.flip`, `al.cast`), memory ops (`al.index_put`, `al.gather_out_to_ub`, `al.scatter_ub_to_out`, `al.index_select_simd`)
  - Section 5 (BL extension): `bl.alloc`, `bl.to_buffer`, `bl.to_tensor`, `bl.subview`, `buffer_type`, `buffer` — explicit on-chip memory management
  - Section 7 (NPUOptions): All 70+ compiler flags passed as kernel launch kwargs: execution model (`compile_mode`, `parallel_mode`, `force_simt_only`), parallelism (`num_warps`, `num_stages`, `auto_blockify_size`), multi-buffering (`multibuffer`, `enable_ubuf_saving`, `enable_preload`), synchronization (`sync_solver`, `unit_flag`, `inject_barrier_all`, `enable_sync_block_lock`), vectorization/fusion (`enable_hivm_auto_cv_balance`, `enable_vf_fusion`, `vf_merge_level`, `add_auto_scheduling`), shape/layout (`enable_nd2nz_on_vector`, `enable_drop_unit_dims`, `enable_flatten`), SIMT-specific, precision/numerics (`default_dot_input_precision`, `allow_fp8e4nv`), debug (`debug`, `bisheng_options`)
  - Hardware notes: which APIs are 910_95-only, int width defaults, `sub_vec_num()` behavior

- **`../../shared/references/tiling-strategies.md`** — Detailed tiling methodology:
  - All 3 inter-core patterns (batch dim, feature dim, row-wise) with full code
  - Load balancing (round-up vs dynamic allocation)
  - Full intra-core UB space calculation walkthrough (buffer inventory, loop amount, 32B alignment)
  - Fixed vs dynamic buffer allocation strategies
  - 512B alignment for matrix ops, optimal BLOCK sizes table
  - Operator case studies: LayerNorm, Softmax, MatMul with UB budgets
  - Diagonal grid scheduling for large matrices
  - Common errors: UB overflow, alignment, load imbalance, precision loss
  - 2D tiling UB budget formula (offset arrays + mask arrays overhead)

- **`../../shared/references/ascend-terminology.md`** — Ascend-specific hardware terminology:
  - AI Core structure (Cube + Vector cores), memory hierarchy (GM/UB/L1), UB constraints
  - HIVM IR mapping: Triton constructs → HIVM IR → pipeline (PIPE_S/V/M/MTE1/2/3)
  - Memory alignment rules, key hardware rules (upcast, accumulator dtype, GM access)
