---
name: optimization
description: Optimize Ascend NPU-native Triton operator performance. Use when diagnosing bottlenecks from cannsim traces and applying optimization patterns. Covers UB overflow detection, Cube utilization improvement, tiling strategy design, and pattern application.
tags: [triton, ascend, optimization, performance]
---

# Triton Kernel Performance Optimization [LEAF NODE]

> **Skill content policy**: This SKILL.md contains only general, reusable knowledge for any agent/user/platform. Per-kernel case studies, session-specific trace data, and dated findings go in `references/` or episodes — not here. Keep concise and task-focused.

Optimize Ascend NPU-native Triton operator performance. Use when diagnosing bottlenecks
from cannsim traces and applying optimization patterns. Covers UB overflow detection,
Cube utilization improvement, tiling strategy design, and pattern application.

## Bottom Lines (Not to be Broken)

1. **Precision**: After optimization, rtol=1e-3, atol=1e-3 must align with PyTorch-NPU. Roll back if not met.
2. **Generalization**: Support all original input shapes and dtypes; do not hardcode specific sizes.

**Performance Ratio Definition**: `Ratio = torch_npu time / Triton time`. Ratio > 1.0 means Triton is faster.
**Priority**: Correctness > Generalization > Performance.

### Generalization Rules (mandatory — violations are P0)

- **Never read shapes from benchmark files, perf reports, or other operator files** when writing the optimized kernel. The source of truth is the original kernel's function signature, problem name, docstring/comments, `get_inputs()`, and mathematical definition.
- **Preserve the operator's stated regime while optimizing**. Large-K kernels should use/block/tune for large K; small-K kernels for small K; tall-skinny matmul for `M >> N` or `N >> M`; irregular kernels for non-power-of-two/boundary dimensions. Do not optimize for unrelated square/control shapes and then claim success.
- **Multiple kernel variants dispatched by the host are fine** (e.g. a fast no-mask path for power-of-2 C alongside a general masked path for all other C). What is not allowed is a variant that only handles specific shapes and leaves other shapes broken or unhandled.
- **Every dispatch path must be correct and tested** — if you write a C=16 fast path and a generic path, both must have unit tests. Never ship an untested code path.
- **Never add new runtime guards that the baseline did not have** (e.g. `if out_channels > 256: raise`, `if sum_dim != 1: raise`). If the baseline kernel accepted a parameter freely, the optimized kernel must too.
- **Do not compare invalid kernels**: benchmark speedups are meaningless unless both providers produce the same outputs. If an editable baseline/input kernel fails correctness, investigate and fix the root cause before trusting latency comparisons; if a non-editable/golden provider cannot run, keep it as `inf` with explicit rationale.
- **Cover all parameter combinations in unit tests inside the required regime**: small/large/non-power-of-2 sizes for every free dimension in the signature, while preserving the problem's shape class. Never test only the benchmark shape.

---

## Optimization Workflow

### Phase 0: Algorithm Review

Review the algorithm itself before optimization. An inefficient algorithm has inherent
limitations no matter how much you optimize.

### Phase 1: Retrieve Past Episodes

Before writing any code, query episodes for relevant patterns:

```python
episode_retrieve(query="<kernel_type> <bottleneck>", target="ascend950", limit=5)
```

### Phase 2: Hierarchical Evaluation

1. **Quick Screening**: If real NPU hardware available, measure end-to-end with `time.time()`.
2. **Precise Diagnosis**: Use `cannsim_remote_run(gen_report=True)` with a **sub-kernel host**
   (grid=1, M=BLOCK_M, K=1×BLOCK_K — see simulation skill Rule 1). Then run:
   ```bash
   python <tree_root>/kernel-ops/simulation/scripts/aggregate_trace.py \
       /path/to/report/trace_core0.json
   ```

### Phase 3: Bottleneck Optimization

| Bottleneck | Optimization Focus |
|------------|---------------------|
| Memory-Bound (MTE2 > 50%) | Vectorized memory access, UB cache reuse, operator fusion |
| Compute-Bound (aiv_vec > 50%) | Cube adaptation, block size tuning |
| Scalar-Bound (aiv_scalar > 80%) | Single-pass reduction, eliminate div/mod, tl.zeros accumulators |
| Latency-Bound | Increase parallelism, reduce synchronization |

**Four Fundamental Moves** (in order):
1. Block/Grid Size Tuning
2. Contiguous Memory Access
3. UB Reuse (single-pass)
4. Compile-time Constants

---

## Hardware Constraints Quick Reference

| Resource | Value | Notes |
|---|---|---|
| Unified Buffer (UB) | 192 KB | Per AI Core. All active tensors must fit simultaneously. |
| L1 Buffer | 1 MB | Cube core only. |
| 32-byte alignment | Required | All load/store addresses and buffer starts. |
| Cube granularity | 16×16 minimum | BLOCK_M, BLOCK_N, BLOCK_K must all be multiples of 16. |
| UB safety factor | ×0.65–0.8 | Compiler adds ~15–35% overhead beyond estimates. |
| Max grid (1D) | `num_aicore` or `num_vectorcore` | Exceeding adds host scheduling overhead. |

**UB estimation:**
```python
def estimate_ub_bytes(BLOCK_M, BLOCK_N, BLOCK_K, dtype_size=2):
    a_tile = BLOCK_M * BLOCK_K * dtype_size
    b_tile = BLOCK_K * BLOCK_N * dtype_size
    acc    = BLOCK_M * BLOCK_N * 4            # FP32 accumulator
    masks  = BLOCK_M * BLOCK_N                # mask array
    return a_tile + b_tile + acc + masks

AVAILABLE_UB = int(192 * 1024 * 0.65)  # ~128 KB usable
# For 2D tiling: also add offset arrays: 2 * ROWS * COLS * 4 (int32)
```

`next_power_of_2` trap: `triton.next_power_of_2(48) = 64`, which may exceed UB.
Use floor rounding: `1 << (n.bit_length() - 1)`.

---

## Core Optimization Rules

### Rule 1: Block Sizing

- Vector ops: BLOCK_SIZE 1024–2048 for FP16 to fit UB with headroom
- Matrix ops: BLOCK_M/N/K multiples of 16. Safe starting point: M=128, N=256, K=64–256
- Use `@triton.autotune` to sweep block sizes
- Grid: 1D preferred, equal to physical core count

**CRITICAL: Grid must cover the smallest autotune BLOCK size.** When the ModelNew host
code hardcodes `ceil(M / 128)` for the grid but autotune selects `BLOCK_M=64` or `256`,
the grid provides too few programs → tiles go unwritten → silent wrong output.

**Fix pattern:** Use a conservative grid that covers the smallest BLOCK in your configs.
Extra programs are harmless (all masks evaluate to false — no-op).

```python
smallest_m = 64  # smallest BLOCK_M across all autotune configs
smallest_n = 64  # smallest BLOCK_N across all autotune configs
grid_m = triton.cdiv(M, smallest_m)
grid_n = triton.cdiv(N, smallest_n)
grid = (grid_m * grid_n, B)
```

For batched matmul with broadcasting (B≠1), the safest approach is a per-batch
sequential loop calling a 2D kernel with conservative grid. This avoids batch-stride
issues in the 3D kernel when expanded/contiguous tensors have unexpected strides.

### Rule 2: Memory Access Contiguity (highest-impact rule)

Contiguous access enables the compiler to merge many small transactions into large-block DMA.
Non-contiguous patterns force many tiny MTE instructions.

```python
# Good: contiguous, compiler can merge into large DMA
offsets = block_start + tl.arange(0, BLOCK_SIZE)

# Bad: non-contiguous
offsets = block_start + tl.arange(0, BLOCK_SIZE) * stride

# Bad: 2D broadcast creates non-contiguous pattern
off = row_off[:, None] + col_off[None, :]

# Good: host-side expand+contiguous eliminates broadcast stride
cos_flat = cos.expand(x_shape).contiguous().reshape(total_rows, D)
```

For auxiliary tensors (cos/sin, etc.), always prefer host-side `expand().contiguous()`
over in-kernel broadcasting. Contiguity > data reuse count.

### Rule 3: Single Pass Over Multi-Pass

Loading the same data multiple times multiplies scalar overhead. When the whole row fits in UB:

```python
# Bad: multiple loads of x
# Good: load once, all computation in UB
x = tl.load(x_ptr + tot_off, mask=mask, other=0.0).to(tl.float32)
sum_x  = tl.sum(x, 1)
mean   = sum_x / D
var    = tl.sum(x * x, 1) / D - (mean ** 2)
x_norm = (x - mean[:, None]) * tl.rsqrt(var + eps)[:, None]
tl.store(y_ptr + tot_off, x_norm * w + b, mask=mask)
```

NBLOCK can be as large as 8192 when D ≤ UB capacity / num_buffers.

**`care_padding=False`** — skip padding check for ~5–10% free speedup:
```python
x = tl.load(ptr + offsets, mask=mask, other=0.0, care_padding=False)
```
Safe when padding values do not affect downstream computation.

### Rule 4: Operator Fusion

Eliminating intermediate GM round-trips transforms memory-bound → compute-bound:
```python
# Before: 2 GM round-trips
# After: 1 GM round-trip (load x → relu → softmax → store w)
x = tl.load(x_ptr + offsets, mask=mask)
w = tl.softmax(tl.where(x > 0, x, 0.0).to(tl.float32))
tl.store(y_ptr + offsets, w.to(tl.float16), mask=mask)
```

### Rule 4b: Post-Linear Contiguous Epilogues

For `nn.Linear`/GEMM followed by pure elementwise post-ops, check whether the GEMM output is contiguous. If yes, flatten the epilogue to one 1D contiguous kernel with a large block (e.g. 4096 elements) instead of launching `(rows, col_blocks)`; keep a persistent fallback only when `cdiv(n_elements, BLOCK_SIZE) > 65535`. For scalar divide epilogues, compute the reciprocal once on the host and use vector multiply (`x * scale`) instead of `x / divisor`. When cannsim traces use different block sizes, report normalized elements/cycle or cycles/element, not only absolute wall cycles. See `references/post-linear-contiguous-epilogue-tiling.md`.

### Rule 5: Precision Rules

```python
# All reductions: upcast to FP32 first
x_fp32 = x.to(tl.float32)
mean = tl.sum(x_fp32, axis=-1) / D

# Matrix multiply: FP16 load, FP32 accumulate, FP16 store
acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
acc += tl.dot(a_fp16, b_fp16)
tl.store(c_ptr + ..., acc.to(tl.float16))
```

### Rule 6: Compile-Time Constants

```python
def kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
    for i in tl.static_range(0, BLOCK_SIZE):  # static unroll
        ...
mask = offsets < n_elements
x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
```

### Rule 7: Intra-Core Tiling When UB Is Tight

```python
for sub_start in range(0, BLOCK_SIZE, SUB_BLOCK_SIZE):
    offsets = start + sub_start + tl.arange(0, SUB_BLOCK_SIZE)
    mask = offsets < n_elements
    x_chunk = tl.load(x_ptr + offsets, mask=mask)
    tl.store(y_ptr + offsets, process(x_chunk), mask=mask)
```

### Rule 8: Two-Path Dispatch for Elementwise Kernels (MANDATORY)

**The trap:** "use a persistent grid" is a common elementwise recommendation.
It is **only a win when the natural tile count exceeds the FFTS grid cap (65535)**.
Below that threshold it is a regression.

| n_tiles (BLOCK=4096) | n_elements | Best path | Why |
|---|---|---|---|
| ≤ 65,535 | ≤ 268,431,360 (~256M) | **direct** (one program per tile) | Fastest dispatch; JUMPC overhead has no upside |
| > 65,535 | > 256M | **persistent** (work-stealing while loop) | Direct would crash (coredim > UINT16_MAX) or saturate FFTS |

**Implementation:**

```python
_MAX_PROGRAMS = 65535  # Ascend FFTS grid cap

class ModelNew(nn.Module):
    def forward(self, x):
        n_tiles = triton.cdiv(x.numel(), BLOCK_SIZE)
        if n_tiles > _MAX_PROGRAMS:
            n_programs = _MAX_PROGRAMS
            _kernel_persistent[(n_programs,)](..., n_programs=n_programs, BLOCK_SIZE=BLOCK_SIZE)
        else:
            _kernel_direct[(n_tiles,)](..., BLOCK_SIZE=BLOCK_SIZE)
```

**Routing threshold:** Use `cdiv(n_elements, BLOCK_SIZE) > _MAX_PROGRAMS`, NOT a raw element count.
If using `@triton.autotune` with multiple `BLOCK_SIZE` configs, the routing threshold
MUST use the **smallest** BLOCK in the configs.

**Sub-kernel trace is identical for direct vs persistent** at grid=1 — the
benefit is purely at full-shape FFTS dispatch level.

**ACL fallback can beat persistent for standard epilogues.** When a UB-safe
Triton tiling for a standard operation chain (e.g. clamp/softmax/pool) exceeds
the 65,535 grid cap, benchmark a CANN/ACL fallback before writing a persistent
loop. Persistent dispatch fixes legality, but CANN's native implementation can
be faster at target scale; cannsim sub-kernel traces will not show this full-shape
dispatch/implementation win.

**`BLOCK_SIZE` conflict with `@triton.autotune`:** When calling an autotuned
kernel, NEVER pass `BLOCK_SIZE=<int>` as a keyword argument. This raises
`ValueError: Conflicting meta-parameter: BLOCK_SIZE`. The autotune decorator
clears non-config kwargs before setting config values, so explicit BLOCK_SIZE
creates a conflict. Only pass `grid` and non-constexpr parameters:

```python
# WRONG
_kernel_autotuned[grid](x, y, n, BLOCK_SIZE=256)

# CORRECT
_kernel_autotuned[grid](x, y, n)
```

The grid must be conservative (cover smallest BLOCK in configs); autotune
selects the actual `BLOCK_SIZE` at compile time.

**Persistent path must use a SEPARATE `@triton.jit` function, NOT the same one
decorated with `@triton.autotune`.** Reasons:

1. `@triton.autotune` compiles multiple variants with different `BLOCK_SIZE`.
2. The persistent kernel's `n_programs` is a runtime arg, not `tl.constexpr`.
3. The work loop `for tile_id in range(pid, n_tiles, n_programs)` needs
   `n_programs` from Python launch time.

Pattern:
```python
@triton.jit
def _kernel_persistent(x_ptr, y_ptr, n_elements, n_programs, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    n_tiles = tl.cdiv(n_elements, BLOCK_SIZE)
    for tile_id in range(pid, n_tiles, n_programs):
        offs = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        # ... compute and store ...
```

n_programs is passed at launch: `_kernel_persistent[(grid_n,)](x, y, n, grid_n, BLOCK_SIZE=4096)`

---

## Ascend-Specific Compiler Hints

```python
import triton.language.extra.cann.extension as al

# dot_pad_only_k: pad only K dimension, reducing UB usage 30-50%
al.compile_hint(acc, "dot_pad_only_k")

# Double buffering (overlaps DMA with compute)
a = al.multibuffer(a, size=2)  # only size=2 supported

# Type conversion with overflow control
y = al.cast(x, tl.float16, fp_downcast_rounding="rtne")
y = al.cast(x, tl.int8, overflow_mode="saturate")
```

**Cube-Vector pipeline sync:**
```python
# Cube side: signal Vector after tl.dot completes
al.sync_block_set(sender="cube", receiver="vector", event_id=0)  # event_id range 0-15

# Vector side: wait before processing accumulator output
al.sync_block_wait(sender="cube", receiver="vector", event_id=0)
```

**Multi-Vector Core post-dot parallelism:**
```python
SUB_BLK_M: tl.constexpr = BLOCK_M // 2
for s in al.parallel(0, 2, bind_sub_block=True):
    sub = al.extract_slice(acc, (s * SUB_BLK_M, 0), (SUB_BLK_M, BLOCK_N), (1, 1))
    sub = tl.where(sub > 0.0, sub, 0.0).to(tl.float16)
    tl.store(c_ptr + ..., sub)
```

**`num_warps` / `num_stages` on Ascend**: silently ignored. Use `NPUOptions` instead.

**`al.multibuffer` pitfalls** — `al.multibuffer(tensor, size=2)` is a side-effect hint only.
Do NOT reassign its return. Call `al.compile_hint` BEFORE `al.multibuffer` on the same tensor.
When using `tl.dot(a, b, acc)` in-place accumulation, prefer it over `al.multibuffer` —
larger gain, eliminates UB overflow risk.

### Max reductions and staged masks

- For row-wise max/online-softmax paths, A/B explicit NaN semantics:
  `tl.max(x, axis, propagate_nan=True)` and
  `tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)`. On Ascend these
  semantics can select a different reduction lowering; require correctness and
  same-run hardware evidence rather than treating the flags as performance-neutral.
- If a causal or structural predicate is rebuilt as broadcasted row/column
  indices in every loop iteration, benchmark a cached host mask and contiguous
  mask-tile load. Include mask creation in end-to-end measurements, even when a
  filtered device-kernel profile excludes it.
- Do not pass a GM mask pointer directly into an outlined SIMD helper when the
  helper's argument is inferred as UB. Load the complete tile in the caller and
  pass the UB-resident tensor through the outlined helper graph; otherwise
  BiSheng may fail with a GM/UB `memref` operand-type mismatch.

---

## Compounding Effects — Matmul Optimizations Are Multiple

When applied together, matmul optimization effects **compound multiplicatively** because
they free different pipeline resources:

| Optimization | Frees | Pipeline |
|-------------|-------|----------|
| `tl.dot(a, b, acc)` in-place | 64 KB UB temp + RVEC LD/ST/ADD | RVECEX, RVECST, RVECLD |
| `tl.range` vs `while` | Dynamic exit checks + SET_INTRA_BLOCKI | FLOWCTRL, PUSHQ |
| Native dtype loads (no fp32 upcast) | VCVT_F2F ops + UB bandwidth | RVECEX, MTE3 |
| `compile_hint("dot_pad_only_k")` | Cube M/N padding cycles | CUBE |

When applying a cocktail of matmul optimizations, expect 4–8× total improvement.

---

## Hardware-Specific Optimization

### Cube (AI Core)
- BLOCK_M/N/K multiples of 16 (512B / element size requirement)
- Accumulator in FP32
- Use diagonal scheduling for large matrices (L2 cache hit rate)

**Diagonal scheduling:**
```python
BLOCK_THRESHOLD: tl.constexpr = 4
for block_idx in range(pid, NUM_BLOCKS_M * NUM_BLOCKS_N, tl.num_programs(0)):
    if NUM_BLOCKS_M >= BLOCK_THRESHOLD and NUM_BLOCKS_N >= BLOCK_THRESHOLD:
        task_m = block_idx % NUM_BLOCKS_M
        task_n = (block_idx // NUM_BLOCKS_M) % NUM_BLOCKS_N
    else:
        task_m = block_idx // NUM_BLOCKS_N
        task_n = block_idx % NUM_BLOCKS_N
```

### UB (Vector Core)
- Total buffer size < 192KB including offset/mask/index arrays
- Single-value buffer 32B aligned

### Grid
- 1D Grid ≤ number of physical cores
- Intra-core loop processes multiple tiles
- Grid > 65535 causes silent crash

---

## Common Bottleneck Quick Reference

| cannsim Metric | Bottleneck | Typical Optimization |
|---------------|------------|---------------------|
| aiv_scalar > 80% | Scalar Bound | Check two-pass / per-row loop; change to single-pass `tl.sum(x,1)` |
| aiv_mte2 > 50% | Memory Bound | Contiguous memory access, expand+contiguous, increase BLOCK |
| aiv_vec > 50% | Compute Bound | Algorithm optimization, reduce redundant computation |
| aic_cube_ratio < 50% | Low Cube Utilization | Check alignment (512B), BLOCK multiples of 16, `compile_hint('dot_pad_only_k')` |

---

### Two-phase reduction: tile-count threshold

Two-phase reduction (private partial sums + single reduce) is the canonical fix for
`tl.atomic_add` serialization. **It has a tile-count sweet spot — it regresses on
small n_tiles.** Each two-phase call is 2 launches (+~0.92 µs overhead). Atomic
serialization at n_tiles < 64 costs less. Only deploy two-phase when n_tiles > 64.
Rule: `NUM_PARTS = min(32, n_tiles)` — never launch more programs than tiles.
See `references/two_phase_reduction_threshold.md` for full data.

---

## Anti-Pattern Checklist (NEVER)

- Make optimization decisions based solely on single-scale data
- Optimize kernel directly when end-to-end target is not met (use cannsim first to confirm bottleneck)
- Sacrifice precision for performance / hardcode that breaks generalization
- Reduce directly in FP16 / matrix multiplication with BLOCK not multiple of 16
- BLOCK_SIZE exceeding UB (192KB) / non-contiguous memory access
- Assuming `mask=` alone makes padded lanes safe when pointer arithmetic has already produced invalid addresses; remap padded offsets first (e.g. `safe_hw = tl.where(offs_hw < H * W, offs_hw, 0)`) before deriving `h/w` or strided pointers
- Use `tensor.item()` in hot path (triggers CPU-NPU synchronization)
- Use if branches inside loops to modify variables (Triton compiles to masked operations, catastrophic performance degradation)
- Calculate UB only for data buffer in 2D tiling (must include offset/mask/index arrays)
- Use precomputed offset tensors for 2D broadcasting (triggers compiler addptr multi-user assertion)
- Use broadcast stride to access auxiliary tensors inside kernel — change to host-side expand+contiguous
- Two-pass mode for reduction operators — use single pass, compute everything within UB after one load
- Not using diagonal scheduling for large matrices (L2 cache thrashing, must enable above BLOCK_THRESHOLD)
- **ALWAYS gate persistent-grid dispatch on `cdiv(n, BLOCK_SIZE) > 65535` (see Rule 8). Applying persistent grid unconditionally is a regression at the bench shape.**

---

## Verification Checklist

- [ ] Precision aligns with PyTorch-NPU (rtol=1e-3, atol=1e-3)
- [ ] Non-aligned dimensions and boundaries pass
- [ ] Performance tests cover small/medium/large sizes
- [ ] grid ≤ number of physical cores, BLOCK_SIZE is compile-time constant
- [ ] Buffer < 192KB, all load/store have masks
- [ ] Reduction upcast to FP32, matrix multiplication BLOCK multiples of 16
- [ ] Is the reduction operator single-pass? (Required when D ≤ UB)
- [ ] Diagonal scheduling enabled for large matrices
- [ ] Elementwise kernel uses two-path dispatch (direct + persistent) — see Rule 8

## Constraints

- **Skill name disambiguation**: When loading skills by short name (e.g., `skill_view(name='optimization')`), plugin-registered skills may cause ambiguity (4+ matches). Always use the fully qualified name: `triton_ascend/compilerclaw-ops-tree/kernel-ops/optimization`.
- Always use `episode_retrieve` before applying any optimization pattern
- Always use `episode_write` after every successful optimization
- Precision (rtol=1e-3, atol=1e-3) is non-negotiable — roll back if not met
- Always verify with cannsim before declaring optimization complete
- For elementwise kernels, ALWAYS implement two-path dispatch (Rule 8) — never apply persistent grid unconditionally
- Prefer `@triton.testing.perf_report` for benchmarks. If `do_bench` is unavailable or unreliable, use the `time.perf_counter` + `torch.*.synchronize()` fallback (see profiling skill).
- For hardware verification after optimization, use `remote_verify` (see profiling skill).

---
## Reference files

### Local references
- `../prefetching/references/stage2-performance-model.md` — FlashAttention Cube/Vector scheduling, outlined row-softmax lowering, explicit max semantics, cached causal-mask attribution, and GM-to-UB helper-boundary rules
- `references/kernel-status-flag-timing.md` — When to set kernel_status flags (verified/recorded) without them being reset by pipeline advances
- `references/trace_comparison_methodology.md` — Sub-kernel trace comparison methodology
- `references/two_phase_reduction_threshold.md` — Two-phase reduction tile-count sweet spot
- `references/grid-autotune-mismatch.md` — Grid/autotune BLOCK size mismatch pitfalls
- `references/grid-cap-acl-fallback.md` — Hybrid dispatch when small Triton epilogues are useful but full/default shapes exceed Ascend grid cap and ACL standard operators are faster/safer
- `references/large-aligned-gemm-acl-dispatch.md` — Large aligned GEMM production ACL dispatch with tested Triton fallback; keep cannsim fallback claims separate from hardware production latency.
- `references/matmul-pair-maxpool-sum-acl-fallback.md` — Linear + adjacent-pair MaxPool1d + row-sum + scale: ACL production dispatch with Cube-based pair-max partial-sum Triton fallback; includes normalized cannsim reporting when baseline microprobes must be shrunk.
- `references/convtranspose-pool-reduction-hybrid.md` — ConvTranspose + channel-min/height-sum/GELU/bias hybrid dispatch when nested Triton reductions trigger BiSheng collapse failures; includes tiny epilogue and profiling guidance
- `references/convtranspose3d-layernorm-pool-acl-hybrid.md` — ConvTranspose3d + LayerNorm + AvgPool3d + GELU hybrid dispatch: keep tiny Triton path, route medium/default standard post ops to ACL, and use one-row cannsim micro-probes when full row blocks are too slow.
- `references/convtranspose3d-batchnorm-avgpool-acl-dispatch.md` — ConvTranspose3d + BatchNorm3d + repeated AvgPool3d: prefer ACL production pooling when hardware wins, keep row-blocked Triton direct/persistent fallbacks tested, and report cannsim by normalized cycles/output.
- `references/convtranspose3d-maxpool-sum-acl-dispatch.md` — ConvTranspose3d + two MaxPool3d stages + channel sum: use ACL production dispatch when hardware beats custom atomic fallback, keep force-tested Triton direct/persistent fallback, and normalize cannsim cycles per channel-output.
- `references/convtranspose3d-swish-groupnorm-hardswish-acl.md` — ConvTranspose3d + Swish + GroupNorm + HardSwish: route standard post chain to ACL/PyTorch, use algebraic HardSwish if native op fails, keep direct+persistent Triton fallback tested but disabled unless hardware wins. + channel sum: use ACL production dispatch when hardware beats custom atomic fallback, keep force-tested Triton direct/persistent fallback, and normalize cannsim cycles per channel-output.
- `references/convtranspose3d-swish-groupnorm-hardswish-acl.md` — ConvTranspose3d + Swish + GroupNorm + HardSwish: route standard post chain to ACL/PyTorch, use algebraic HardSwish if native op fails, keep direct+persistent Triton fallback tested but disabled unless hardware wins.
- `references/convtranspose3d-softmax-sigmoid-acl-dispatch.md` — ConvTranspose3d + softmax(dim=1) + sigmoid hybrid dispatch: single-pass tiny Triton epilogue, ACL fallback when `N*D*H*W` exceeds grid cap, and comparison-provider pre-skips.
- `references/convtranspose3d-relu-groupnorm-acl-dispatch.md` — ConvTranspose3d + ReLU + GroupNorm: prefer ACL/native post chain over custom two-pass Triton group reduction; optional ReLU fallback must have direct+persistent dispatch.
- `references/convtranspose-channel-plane-tanh-epilogue.md` — ConvTranspose2d + per-channel bias/subtract + tanh epilogue: keep ConvTranspose on ACL, retile NCHW epilogue by `(N,C)` plane to remove scalar div/mod, use direct+persistent grid-cap dispatch, and promote large offsets to int64.
- `references/groupnorm-multigroup-ub-tiling.md`
- `references/conv3d-acl-channel-segment-epilogue.md` — Conv3d + per-channel pointwise epilogues: keep Conv3d on ACL, tile contiguous NCDHW by `(N,C)` plane to scalarize channel parameters, and unit-test direct/persistent plus generic-C dispatch.
- `references/conv3d-activation-bias-channel-epilogue.md` — Conv3d + ReLU/LeakyReLU/GELU/Sigmoid + per-channel BiasAdd: channel-segment epilogue, exact LeakyReLU-after-ReLU no-op, direct+persistent grid-cap dispatch, and forced-persistent profiling requirements.
- `references/conv3d-activation-softmax-mean-acl-dispatch.md` — Conv3d + activation + softmax/mean
- `references/standard_conv_activation_pool_acl_dispatch.md` — Decision rule for Conv2d + standard activation/pool chains: when cannsim shows scalar/spill-bound custom epilogues, test ACL host dispatch before deeper Triton fusion.
- `references/conv2d-gelu-gap-acl-dispatch.md` — Conv2d + exact GELU + GlobalAvgPool: remove the custom Triton epilogue launch, route to `F.gelu(...).mean((-2,-1))`, and keep baseline columns parser-visible with neutral skips when exact baseline timing is unsafe.
- `references/conv2d-avgpool-sigmoid-sum-acl-dispatch.md` — Conv2d + AvgPool2d + Sigmoid + Sum: remove scalar/MTE-heavy custom pooling/reduction epilogues, route standard chain to ACL, and keep Baseline Triton2 parser-visible when sandbox forbids reading `base_*.py`.
- `references/conv-mish-tanh-epilogue.md` — Conv + Mish + Tanh epilogue pattern: keep ACL convolution, use one-exp stable `tanh(softplus(x))`, `tl.math.tanh`, direct+persistent dispatch, and forced-persistent unit testing.
- `references/relu-hardswish-algebraic-epilogue.md` — ReLU + HardSwish exact algebraic rewrite: replace separate `tl.maximum` ReLU with a piecewise `tl.where` form to reduce RVEC/PUSHQ work; benchmark small-shape fallback if needed.
- `references/leakyrelu-branch-select-epilogue.md` — Divide/scale + LeakyReLU pointwise epilogues: prefer `tl.where(y >= 0, y, y * slope)` over `tl.minimum` algebra when cannsim shows RVEC/PUSHQ inflation, and force-test persistent fallback.
- `references/conv-mish-batchnorm-tile-dispatch.md` — Conv2d + Mish + BatchNorm pattern: keep Conv/BN on ACL, use large-tile direct Mish dispatch with persistent fallback, normalize cannsim cycles when BLOCK changes, and force-test persistent fallback.
- `references/conv2d-batchnorm-scaling-affine-fold.md` — Conv2d + BatchNorm2d + scalar scale: fold scale into BN affine parameters, cache no-grad affine/eval folds safely, and benchmark cached paths under `torch.no_grad()`.

- `references/gemm-swish-groupnorm-acl-dispatch.md` — GEMM + Swish/bias + GroupNorm ACL production dispatch with legal Triton fallback/chunking; separate cannsim fallback traces from hardware dispatch results.
- `references/gemm-bias-relu-contiguous-weight.md` — GEMM + bias/activation pattern: transpose PyTorch `[N,K]` weights to contiguous `[K,N]`, use in-place `tl.dot(a,b,acc)`, and normalize cannsim cycles when `BLOCK_N` changes.
- `references/post-linear-contiguous-epilogue-tiling.md` — GEMM/Linear followed by pure elementwise epilogue: flatten contiguous output, reduce epilogue program count, use direct+persistent grid-cap dispatch, and compare cannsim traces by normalized elements/cycle when block sizes differ.
- `references/constant-singleton-softmax-fill.md` — Constant `(B,1)` softmax output: fill-ones direct/persistent dispatch, forced persistent unit testing, and normalized cannsim elements/cycle reporting.
- `references/algebraic-dead-gemm-zero-output.md` — Algebraically eliminate dead GEMM/reduction/activation chains that become exact zero (e.g. `GELU(max - max)`), with direct+persistent zero-fill dispatch and sandbox-safe Baseline Triton2 profiling.
- `references/post-linear-groupnorm-acl-dispatch.md` — Linear/GEMM followed by GroupNorm + activation/scale: prefer ACL/CANN production dispatch when custom Triton uses narrow per-group GEMM tiles; keep/test a chunked Triton epilogue fallback and normalize cannsim cycles per group.
- `references/post-linear-swish-direct-sigmoid.md` — Linear/GEMM + Swish/SILU + scale epilogues: test fp32 direct sigmoid against branch-stable sigmoid, keep shape-aware block sizes, and force-test persistent fallback.
- `references/exact-sigmoid-epilogue-tiling.md` — Exact sigmoid/residual epilogues: normalize cannsim by elements when changing block size, A/B exp2 rewrites, and force-test grid-cap persistent fallback.
- `references/pushq_vf_bottleneck_matmul.md` — PUSHQ/VF bottleneck patterns in matmul
- `references/row_scale_diagonal_matmul.md` — Row-scale diagonal matmul patterns
- `references/small-gemm-activation-reduction-dot-split.md` — Small dense linear + activation + hidden-sum + global-reduction pattern: replace vector broadcast GEMM with row-tiled `tl.dot`, then reduce row sums.
- `references/gemm-sigmoid-sum-logits-epilogue.md` — Large GEMM + sigmoid + hidden-sum pattern: compute logits with Cube `tl.dot`, then run a vector bias/sigmoid/row-sum epilogue when direct post-dot fusion hits compiler broadcast issues.

### Shared references (in `../../shared/references/`)

Agents MUST read the relevant shared reference file(s) with `read_file` before coding or reviewing patterns that depend on them; mentions in this SKILL.md are only an index, not the full procedure.

- **`optimization-patterns.md`** — Read before applying non-trivial performance patterns. Contains all core rules, reference kernel implementations (GEMM with diagonal scheduling + `al.parallel`, LayerNorm, Online Softmax, Flash Attention), pitfalls G1-G7 (Python-style slices, return/break in loops, integer comparison in `tl.where`, multi-dim grid, Cube never activated, FP16 overflow, `.item()` sync), and detailed case studies. The patterns in this skill are a subset; consult this doc for full details.
- **`references/gemm-scale-batchnorm-cached-transpose.md`** — GEMM+scale+BatchNorm cached `[K,N]` weight transpose pattern for `nn.Linear` kernels whose baseline reads PyTorch `[N,K]` weights with strided N access.
- **`triton-api-reference.md`** — Read when using or changing any non-basic Triton-Ascend API: `al.*`, `bl.*`, `NPUOptions`, `tl.dot` options, synchronization, buffer management, custom ops, or compiler flags. Section 4: AL extension (`al.compile_hint`, `al.multibuffer`, `al.cast`, `al.sync_block_set/wait`, `al.parallel`, `al.extract_slice`, `al.insert_slice`, etc.). Section 5: BL extension. Section 7: NPUOptions.
- **`tiling-strategies.md`** — Read before changing `BLOCK_*`, grid shape, swizzle/grouping, persistent-grid loops, reductions, or any schedule that affects UB/L1 usage. Includes inter-core patterns, intra-core UB budget calculation with alignment, operator case studies (LayerNorm, Softmax, MatMul), diagonal scheduling, and common errors (UB overflow, alignment, load imbalance, precision loss).
- **`ascend-terminology.md`** — Read when interpreting cannsim traces or hardware terms: AI Core (Cube + Vector), GM/UB/L1, UB constraints, alignment rules, HIVM IR mapping table, pipeline stages.
- **`hardware-architecture.md`** — Read when choosing core type, reasoning about memory hierarchy/platform differences, using `TRITON_ALL_BLOCKS_PARALLEL`, or avoiding INT64/hardware-specific hazards.

### Per-kernel examples (in `references/`)
- `references/l1_19_ReLU/` — ReLU
- `references/l1_23_Softmax/` — Softmax
- `references/l1_26_GELU_/` — GELU
- `references/l1_36_RMSNorm_/` — RMSNorm
- `references/l1_41_Max_Pooling_1D/` — MaxPool1D
- `references/l1_100_HingeLoss/` — HingeLoss
- `references/l1_1_Square_matrix_multiplication_/` — Square MatMul
- `references/l2_9_Matmul_Subtract_Multiply_ReLU/` — Matmul+Sub+Mul+ReLU
- `references/l2_18_Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp/` — Complex matmul fusion
- `references/l2_76_Gemm_Add_ReLU/` — Gemm+Add+ReLU
