---
name: codegen
description: Generate and optimize Triton kernel code for Ascend NPU operators.
tags: [triton, ascend, codegen]
---

# Triton Kernel Code Generation [LEAF NODE]

> **Skill content policy**: This SKILL.md contains only general, reusable knowledge for any agent/user/platform. Per-kernel case studies, session-specific trace data, and dated findings go in `references/` or episodes — not here. Keep concise and task-focused.

Generate and optimize Triton kernel code for Ascend NPU operators. Covers the full loop:
code generation → correctness verification → cannsim profiling → optimization.

**Compute Logic → Tiling Strategy → Code Implementation → Verify → Profile → Optimize**
The order must not be reversed.

## Scope Rule

> Cover ALL shapes the kernel signature accepts — not just the benchmark shape.
> Read the kernel's **actual parameter list**, problem name, docstring/comments, and `get_inputs()` to understand what varies at runtime and what regime is required (large-K, small-K, tall-skinny, irregular, etc.).
> Preserve the operator's stated regime in code, tiling, tests, and profiling: large-K kernels must be optimized/tested with large K; small-K with small K; tall-skinny matmul with `M >> N` or `N >> M`; irregular kernels with non-power-of-two/boundary dimensions.
> Never infer test shapes from other benchmark directories or add unrelated square/control shapes as headline benchmarks.

---

## Workflow

### Step 1: Retrieve Past Episodes

```python
episode_retrieve(query="<kernel_type> <bottleneck or technique>", target="ascend950", limit=5)
```

Browse the DB:
```bash
sqlite3 ~/CompilerClaw/episodes.db \
  "SELECT id, kernel_name, substr(observation,1,80) FROM episodes ORDER BY id DESC"
```

### Step 2: Understand Requirements → Confirm Compute Logic

Extract mathematical formulas, input/output specifications, constraints.
Understand the hardware architecture before tiling design:

- Each AI Core: 1 Cube Core (matrix multiply, L1 Buffer 1MB) + 2 Vector Cores (UB 192KB)
- GM (DDR) → UB via MTE2 → Vector Core → MTE3 → GM
- GM → L1 → Cube Core → L1 → GM (for matrix ops)

Core type selection:
```python
import triton.runtime.driver as driver
props = driver.active.utils.get_device_properties("npu")
num_cube_cores   = props["num_aicore"]     # for kernels with tl.dot
num_vector_cores = props["num_vectorcore"] # for pure vector kernels
```

### Step 3: Implement Tiling Strategy

**Inter-Core Partitioning:**
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

**Operator type → core type:**
| Operator Type | Core |
|---|---|
| Reduction (norm, softmax, loss) | Vector |
| GEMM / Attention | AI Core (Cube) |
| Activation / Loss / Index / MoE | Vector |
| Convolution | AI Core |

For detailed tiling methodology, see `../../shared/references/tiling-strategies.md`.

### Step 4: Generate Kernel Code

**Universal rules — apply to every kernel:**

1. Add alignment hints to every offset array:
   ```python
   offsets = pid * BLOCK + tl.arange(0, BLOCK)
   tl.multiple_of(offsets, 16)
   tl.max_contiguous(offsets, BLOCK)
   ```

2. Always use `tl.range` (not Python `range`) for inner loops.

3. `tl.dot` is the ONLY instruction that activates the Cube engine. Even GEMV must use `tl.dot` with B padded to BLOCK_N=16.

4. Cap any grid dimension to 65535 (FFTS hard limit — silent crash above).

**For complete runnable templates (Templates 1–12), see:**
**`../../shared/references/templates.md`** — includes:
- Template 1: Reduction (GroupNorm, LayerNorm, RMSNorm, Softmax) — single-pass
- Template 2: GEMM — full `@triton.autotune`, diagonal scheduling, `al.parallel`, `al.multibuffer`
- Template 3: Activation — element-wise with 2D tiling
- Template 4: CrossEntropy — online softmax, `ignore_index`
- Template 5: Index Transformation / RoPE — block-table pointer, nested tile loops
- Template 6: Attention — QK^T with `tl.make_block_ptr` + `tl.advance`, Cube-Vector sync
- Template 7: MoE gating — softplus/sigmoid, 2D program grid
- Template 8: Post-processing — cumulative token expansion
- Template 9: Convolution — state management, sliding window
- Template 10: Contiguous Expand (RoPE) — host-side `expand().contiguous().reshape()`
- Template 11: Embedding/Gather — `tl.gather`, `index_select_simd`, `index_put`
- Template 12: Sort/Flip — `tl.sort` last-dim-only, descending via ascending + flip

### Step 5: Verify Correctness

Write a smoke test (single shape × single dtype) to verify the kernel compiles and produces correct results. Use the simulation skill for cannsim workflow and host template. Use the profiling skill for hardware verification.

### Step 6: Profile & Optimize

After the kernel runs correctly, profile with cannsim and optimize based on bottleneck. See the **optimization skill** for the full workflow: cannsim trace analysis, bottleneck identification, optimization patterns, and episode recording.

---

## Anti-Pattern Checklist (NEVER)

- Write code without confirming compute logic first
- Use Python `range` for inner loops — use `tl.range`
- Forget alignment hints (`tl.multiple_of`, `tl.max_contiguous`)
- GEMV with Vector Core element-wise multiply-add — use `tl.dot` + pad B to BLOCK_N=16
- `return` / `break` inside `for`/`while` loops — use `tl.where` mask instead
- `tensor[i]` indexing — use `tl.where`, `tl.gather`, `al.extract_slice` instead
- `tl.zeros([N], dtype)` inside `tl.range` body — hoist before loop, use `num_stages=2`
- Python scalar loop-carried state inside `tl.range` — use `tl.zeros([1], tl.float32)` accumulators
- `num_stages > 2` — only 1 or 2 on Ascend
- Hardcode grid as `ceil(M/128)` in ModelNew when autotune has different BLOCK configs — use `ceil(M/min_BLOCK_M)`
- Autotune `key=['param']` but kernel signature is missing `param: tl.constexpr` — every key name MUST appear as a `tl.constexpr` parameter in EVERY kernel decorated by that autotune, including persistent kernels (causes "No valid triton configs" RuntimeError)
- Persistent kernel iterating over elements instead of tiles — use `n_tiles = cdiv(n_elements, BLOCK_SIZE)` then `range(pid, n_tiles, n_programs)`, NOT `range(pid, n_elements, n_programs)`
- Deliver a kernel that only works for the benchmark shape
- Skip writing an episode after a successful optimization

## Constraints

- Always call `episode_retrieve` before writing any kernel code
- Always call `episode_write` after every successful optimization
- Cover ALL shapes — never deliver a kernel that only works for the benchmark shape
- For profiling: use `cannsim_remote_run` with `gen_report=True` (see simulation + optimization skills)
- For hardware verification: use `remote_verify` (see profiling skill)

---

## Reference files

### Shared references (in `../../shared/references/`)

Agents MUST read the relevant shared reference file(s) with `read_file` before implementing code that depends on them; this list is an index, not a substitute for the full docs.

- **`templates.md`** — Read before starting a new kernel or selecting a skeleton. Contains all 12 runnable kernel templates and expected host-launch patterns.
- **`optimization-patterns.md`** — Read before applying performance idioms beyond the selected template: core rules, reference kernels, pitfalls G1-G7, case studies.
- **`triton-api-reference.md`** — Read before using `al.*`, `bl.*`, `NPUOptions`, `tl.dot` options, synchronization, custom ops, or compiler flags. Use it to verify exact signatures and unsupported APIs.
- **`tiling-strategies.md`** — Read before choosing/changing `BLOCK_*`, grid shape, swizzle/grouping, persistent-grid loops, reduction decomposition, or UB/L1 budgets.
- **`ascend-terminology.md`** — Read when hardware terms appear in traces/reviews: AI Core, Cube/Vector, GM/UB/L1, HIVM IR, pipeline stages, alignment.
- **`hardware-architecture.md`** — Read when choosing core type, reasoning about platform differences, `TRITON_ALL_BLOCKS_PARALLEL`, INT64 hazards, or memory hierarchy constraints.
