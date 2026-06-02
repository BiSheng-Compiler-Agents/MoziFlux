---
name: triton-operator-code-gen
description: >
  Generate and optimize Triton kernel code for Ascend NPU operators. Use when
  implementing a new Triton operator (from a design doc or direct requirements),
  or when optimizing an existing one for performance. Covers the full loop:
  code generation → cannsim bottleneck analysis → episode-driven optimization
  → episode recording.
tags: [triton, ascend, npu, optimization, kernelbench, performance, codegen]
metadata:
  hermes:
    requires_tools:
      - cannsim_remote_run
    related_skills:
      - triton-ascend-cannsim
      - kernel-episode-memory
      - triton-ascend-kernel-profiling
      - triton-operator-performance-optim
---

# Triton Operator Code Generation & Optimization (Ascend NPU)

## Core Principles

**Compute Logic → Tiling Strategy → Code Implementation → cannsim Profiling → Optimization**
The order must not be reversed. Never optimize blindly — always profile first.

## Scope Rule (enforced by user)

> Cover ALL shapes the kernel signature accepts — not just the benchmark shape.
> Read the kernel's **actual parameter list** to understand what varies at runtime (N, C, H, W, dtype).
> Profile across representative shapes and design dispatch logic that handles all of them.
> Never infer test shapes from other benchmark directories.
>
> Before writing any kernel, answer:
> - What are the runtime parameters and their ranges?
> - Does per-program startup cost dominate for small inputs? → persistent grid
> - Does tile count create excessive sync stalls for large inputs? → tune BLOCK size
> - Profile at least: one small case, one large case, one non-power-of-2 case
>
> **Do NOT deliver a kernel that only works for the benchmark shape.**

---

## Workflow

### Step 1: Retrieve Past Episodes

Before writing any kernel code, query the episode database for relevant patterns:

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

Episodes (IDs 12–40): 30 patterns from 198 KernelBench pairs, tagged by kernel_name:
`matmul`, `softmax`, `rmsnorm`, `layernorm`, `gelu`, `relu`, `maxpool`, `elementwise`,
`hingeloss`, `cross_entropy`, `embedding`, `general`

Browse the DB directly:
```bash
sqlite3 ~/CompilerClaw/episodes.db \
  "SELECT id, kernel_name, substr(observation,1,80) FROM episodes ORDER BY id DESC"
```

---

### Step 2: Understand Requirements → Confirm Compute Logic

Extract mathematical formulas, input/output specifications, constraints.
Load [`references/hardware-architecture.md`](references/hardware-architecture.md) before tiling design.

---

### Step 3: Implement Tiling Strategy

If a design doc (output of design skill) exists, follow its tiling strategy. Otherwise design here.

**Inter-Core Partitioning — two rules:**
1. `grid = number of physical cores` (never more, never multidimensional)
2. Each core's intra-core loop processes multiple tiles (load balancing)

```python
from triton.runtime import driver
props = driver.active.utils.get_device_properties("npu")
core_num = props["num_aicore"]      # for kernels with tl.dot (Cube)
core_num = props["num_vectorcore"]  # for pure vector kernels

grid = (max(1, min(core_num, triton.cdiv(total_tiles, 1))),)

# Inside kernel:
pid = tl.program_id(0)
num_core = tl.num_programs(0)
for tile_idx in tl.range(pid, total_tiles, num_core):
    ...
```

**UB budget:** `safe_BLOCK = int((196608 - 32) / (num_buffers × dtype_size) × 0.65)`

**Operator type → core type → template:**
| Operator Type | Core | Template |
|---|---|---|
| Reduction (norm, softmax, loss) | Vector | Template 1 |
| GEMM / Attention | AI Core (Cube) | Templates 2, 6 |
| Activation / Loss / Index / MoE | Vector | Templates 3–5, 7–8 |
| Convolution | AI Core | Template 9 |

Load [`references/templates.md`](references/templates.md) before kernel implementation.

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

---

### Step 5: Generate Correctness Test

Write a smoke test (single shape × single dtype) in the host file to verify the kernel compiles, runs, and produces correct results via `cannsim_remote_run`. Comprehensive validation is handled by the precision-eval skill.

---

### Step 6: Profile with cannsim

After the kernel runs correctly, **always profile before optimizing**. Never guess the bottleneck.

```python
# Run via cannsim_remote_run tool — gen_report=True produces trace_core0.json
cannsim_remote_run(local_dir="...", run_script="run_kernel.sh", gen_report=True)
```

Read the cannsim trace to identify the bottleneck:

| cannsim Metric | Bottleneck | First Action |
|---|---|---|
| `aiv_scalar > 80%` | Scalar-bound | Check two-pass pattern; switch to single-pass + tl.sum(x,1); check Python scalar loop-carried state (spills to SCALARLDST) |
| `aiv_mte2 > 50%` | Memory-bound | Contiguous access, expand+contiguous for aux tensors, increase BLOCK |
| `aiv_vec > 50%` | VEC-bound | fp16 tl.maximum → upcast to fp32 (fp16 routes to slow VEC unit, fp32 to RVECEX) |
| `aic_cube_ratio < 50%` | Low Cube util | Check BLOCK multiples of 16; add `tl.compile_hint(a, "dot_pad_only_k")` before tl.dot |
| `SCALARLDST high` | Scalar register spill | Replace Python float/int loop accumulators with `tl.zeros([1], tl.float32)` |
| PUSHQ high, many VF fences | tl.zeros inside loop | Hoist `zero = tl.zeros([BLOCK], tl.float32)` before `tl.range`; use `num_stages=2` |
| `instr.bin < 1 MB`, `~28 cycles`, 100% SCALAR | UB overflow (silent!) | BLOCK_HW too large; fp32 safe limit is 2048 elements (3 buffers × 2048×4 = 24KB < 32KB UB) |

Load [`references/optimization-patterns.md`](../triton-operator-performance-optim/references/optimization-patterns.md) and
[`triton-api-reference.md`](../triton-operator-shared/references/triton-api-reference.md) (§7 NPUOptions) for full pattern details.

---

### Step 7: Optimize Based on Bottleneck

Retrieve relevant episodes first (`episode_retrieve`), then apply patterns. Priority order:

1. Block/Grid size tuning (highest ROI — block size alone can give 7.5×)
2. Memory access contiguity (2D broadcast → expand+contiguous on host)
3. Single-pass reduction (replace two-pass; tl.sum(x,1) stays in UB)
4. Compiler hints (`tl.compile_hint`, `tl.multiple_of`, `tl.max_contiguous`)
5. Persistent grid for small-HW or large-N kernels
6. Double-buffering (`al.multibuffer(tensor, 2)`)
7. Cube-Vector pipeline sync (`al.sync_block_set/wait`) for mixed kernels

---

### Step 8: Record New Patterns as Episodes

**After every successful optimization, write an episode.** This is mandatory — it grows the shared knowledge base that benefits all future optimizations.

```python
episode_write(
    kernel_name="softmax",          # operation type
    target="ascend950",
    observation="Baseline: one-row-per-program, aiv_scalar=85%, 3605 µs. "
                "Per-row loop with integer division for row offset.",
    thoughts="Scalar overhead from 2D broadcast row_off[:,None]+col_off[None,:]. "
             "Root cause: non-contiguous access pattern prevents DMA merging. "
             "Tried 2D tiling (improved MTE but scalar jumped to 85%), "
             "tried incremental pointer (5× regression from branch divergence). "
             "expand().contiguous() on host eliminates broadcast stride entirely.",
    action="Host-side: cos_flat = cos.expand(shape).contiguous().reshape(rows, D). "
           "Kernel: uniform row*D+col offset for all tensors. "
           "Eliminated per-row integer division inside kernel.",
    result="3605 µs → 752 µs (4.8×). aiv_mte2 now bottleneck (memory BW). "
           "30% gap to torch_npu (550 µs). Contiguity > data reuse count.",
)
```

**When to write an episode:**
- Any optimization that changed performance by >10%
- Any compiler crash or silent failure with a discovered workaround
- Any pattern that surprised you (good or bad)
- Any shape-dependent dispatch trick

**Episode quality checklist:**
- `observation`: includes the cannsim metric that was the bottleneck (e.g. `aiv_scalar=85%`)
- `thoughts`: includes what alternatives were considered and why rejected
- `action`: includes the exact code change (one-liner or snippet)
- `result`: includes before/after latency numbers and the key insight

---

## Reference Files

| File | When to Load |
|---|---|
| [`references/hardware-architecture.md`](references/hardware-architecture.md) | Step 2 — before tiling design |
| [`references/templates.md`](references/templates.md) | Step 3 — before kernel implementation |
| [`../triton-operator-performance-optim/references/optimization-patterns.md`](../triton-operator-performance-optim/references/optimization-patterns.md) | Step 6/7 — hardware constraints, pitfalls G1–G12, RoPE/GroupNorm case studies |
| [`../triton-operator-shared/references/triton-api-reference.md`](../triton-operator-shared/references/triton-api-reference.md) | Step 3/4 — full tl.* API, AL/BL extensions, §7 NPUOptions compiler flags |

**Reference kernel pairs** (baseline + optimized + perf) in `../triton-operator-performance-optim/references/`:
```
l1_1_Square_matrix_multiplication_/   ← dot_pad_only_k, GROUP_M swizzle        5.5×
l1_19_ReLU/                           ← persistent kernel, propagate_nan
l1_23_Softmax/                        ← multi-row blocked online softmax
l1_26_GELU_/                          ← make_block_ptr + tl.advance, tl.math.tanh
l1_36_RMSNorm_/                       ← 2D NCHW grid, no transpose              45×
l1_41_Max_Pooling_1D/                 ← 2D vectorized load, no-index fast path
l1_100_HingeLoss/                     ← two-phase reduction, no atomic_add       2.4×
l2_9_Matmul_Subtract_Multiply_ReLU/  ← block size tuning only                  7.5×
l2_18_Matmul_Sum_Max_AvgPool_.../     ← tl.static_range K unroll                587×
l2_76_Gemm_Add_ReLU/                  ← weight pre-transpose, torch fallback    2×
```

---

## Anti-Pattern Checklist (NEVER)

- ❌ Write code without confirming compute logic first
- ❌ Optimize without cannsim profiling — never guess the bottleneck
- ❌ Ignore UB size (192KB) / grid > physical core count / grid dimension > 65535
- ❌ Reduce in FP16 / matrix multiply with BLOCK not multiple of 16
- ❌ Use third-party libraries or element-wise compute inside kernel
- ❌ GEMV with Vector Core element-wise multiply-add — use `tl.dot` + pad B to BLOCK_N=16
- ❌ Two-pass reduction (load x multiple times) — use single-pass `tl.sum(x, 1)` in UB
- ❌ Python scalar loop-carried state inside `tl.range` — use `tl.zeros([1], tl.float32)` accumulators
- ❌ `tl.zeros([N], dtype)` inside `tl.range` body — hoist before loop, use `num_stages=2`
- ❌ `tl.maximum` on fp16 — always upcast to fp32 first (fp16 → slow VEC unit, fp32 → RVECEX)
- ❌ `num_stages > 2` — MTE hardware prefetches; `num_stages=1` or `2` only
- ❌ `num_warps` / `num_stages` as a performance knob — silently ignored on Ascend
- ❌ Not using diagonal scheduling for large matrix multiplication (L2 cache thrashing)
- ❌ `return` / `break` inside `for`/`while` loops — use `tl.where` mask instead
- ❌ `tensor[i]` indexing — use `tl.where`, `tl.gather`, `al.extract_slice` instead
- ❌ Deliver a kernel that only works for the benchmark shape
- ❌ Skip writing an episode after a successful optimization

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
