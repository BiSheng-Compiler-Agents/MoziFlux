---
name: triton-operator-performance-optim
description: Optimize Ascend NPU-native Triton operator performance. Solve UB overflow, improve Cube utilization, design tiling strategies. 
tags: [triton, ascend, npu, optimization, kernelbench, performance]
metadata:
  hermes:
    related_skills:
      - triton-ascend-cannsim
      - kernel-episode-memory
      - triton-ascend-kernel-profiling
      - triton-operator-code-gen

---

# Triton Operator Performance Optimization (Ascend NPU)

## References

- **`references/al_parallel_epilogue_subkernel.md`** — A/B cannsim test of `al.parallel` for
  simple GEMM epilogues. Key: al.parallel HURTS for bias+activation fused onto a single tile
  (20296 vs 19828 wall_cycles, l2_9 June 2026). Use al.parallel only when post-dot work is
  substantial. Also documents MTE3 as inherent sub-kernel bottleneck — don't chase it.

- **`references/relu_fp16_vs_fp32_maximum_trace.md`** — cannsim trace evidence for the fp16 vs fp32
  tl.maximum routing difference (l1_19_ReLU June 2026). Exact instruction names (WAIT_FLAG_VEC,
  RV_VCMP_NE, RV_VMAXS, RV_VSEL, RV_VCVT_F2F), cycle counts, and instruction mapping table.
  Use when debugging WAIT_FLAG_VEC stalls in elementwise kernels.

- **`references/relu_elementwise_dispatch_floor.md`** — Full l1_19_ReLU optimization session record
  (June 2026). Cannsim sub-kernel traces for direct (BS=256) and persistent (BS=4096) paths,
  fixed per-program SCALARLDST overhead breakdown (~3641 cy), hardware benchmark table (torch vs
  baseline vs optimized across all shapes), root cause of why optimized doesn't beat baseline, and
  conclusion that standalone elementwise Triton cannot beat torch ACL at small-medium N.

## Required Output Files (produce ALL at session end)

Every optimization session must deliver these five files. User will ask for missing ones:

```
l<N>_<KernelName>/
├── opt_<N>_<KernelName>.py    ← full kernel + ModelNew + get_inputs() + get_init_inputs()
├── profile_kernels.py         ← three-way benchmark (torch_ref/baseline/optimized) + unit test
├── Optimizations.md           ← numbered baseline issues + per-optimization explanation + pitfalls
└── performance_report.md      ← trace tables + cyc/elem comparison + hardware latency (TBD)
```

opt_*.py must be a complete drop-in: @triton.jit kernel + dispatch function + ModelNew class.
profile_kernels.py: use `@perf_report`, `do_bench(return_mode="mean")`, plain colour styles (no hex).
If baseline has no ModelNew, dispatch the bare @triton.jit kernel directly in _run_baseline.
performance_report.md: include hardware latency section marked TBD if no NPU run has been done.

---

1. **Precision**: After optimization, rtol=1e-3, atol=1e-3 must align with PyTorch-NPU. Roll back if not met.
2. **Generalization**: Support all original input shapes and dtypes; do not hardcode specific sizes.

**Performance Ratio Definition**: `Ratio = torch_npu time / Triton time` (reciprocal of time). Ratio > 1.0 means Triton is faster.

**Priority**: Correctness > Generalization > Performance.

### Generalization Rules (mandatory — violations are P0)

- **Read the actual baseline `.py` in the kernel's own directory to determine what constraints exist.** Do NOT infer constraints from any other file in the dataset — perf reports, kernelbench_z references, bench scripts, or prior optimized versions may carry guards introduced by a previous optimization pass. None of them are the ground truth. Only the baseline kernel file in the target directory is.
- **Multiple kernel variants dispatched by the host are fine** (e.g. fast no-mask path for power-of-2 C, masked path for all others). What is not allowed is a variant that handles specific shapes and leaves others broken or unhandled.
- **Every dispatch path must be correct and tested.** If there is a fast path and a generic path, both must have unit tests. Never ship an untested code path.
- **Never add new runtime guards the baseline did not have** (e.g. `if out_channels > 256: raise`, `if sum_dim != 1: raise`). If the baseline accepted a parameter freely, the optimized kernel must too.
- **Cover all parameter combinations in unit tests**: small/large/non-power-of-2 for every free dimension. Never test only the benchmark shape.

## Optimization Workflow

### Phase 0: Algorithm Review

Review the algorithm itself before optimization. An inefficient algorithm has inherent limitations no matter how much you optimize.

### Phase 1: Hierarchical Evaluation

1. **Quick Screening**: If real NPU hardware is available, measure end-to-end with `time.time()` (cover small/medium/large sizes). Done if target is met.
2. **Precise Diagnosis**: Use `cannsim` to measure kernel-side time when target is not met, identify the real bottleneck. Checkout `triton-ascend-cannsim` skill for more details. 

### Phase 2: Bottleneck Optimization

| Bottleneck | Optimization Focus |
|------------|---------------------|
| Memory-Bound | Vectorized memory access, UB cache reuse, operator fusion |
| Compute-Bound | Cube adaptation, block size tuning |
| Latency-Bound | Increase parallelism, reduce synchronization |

**Four Fundamental Moves** (in order): Block/Grid Size → Contiguous Memory Access → UB Reuse → Compile-time Constants

**Retrieve patterns before writing code:**
```python
episode_retrieve(query="<kernel_type> <bottleneck>", target="ascend950", limit=5)
# Examples:
# episode_retrieve(query="matmul cube utilization dot pad static range", target="ascend950")
# episode_retrieve(query="softmax wide rows MTE online reduction", target="ascend950")
# episode_retrieve(query="norm scalar overhead two-pass single-pass", target="ascend950")
# episode_retrieve(query="elementwise FFTS dispatch persistent grid", target="ascend950")
```

**Load**: [`optimization-patterns.md`](references/optimization-patterns.md), [`ascend-terminology.md`](../triton-operator-shared/references/ascend-terminology.md)

### Phase 3: Hardware Specialization

- **Cube**: BLOCK_M/N/K are multiples of 16, accumulator in FP32
- **UB**: Total buffer size < 192KB, single-value buffer 32B aligned
- **Grid**: 1D Grid ≤ number of physical cores, intra-core loop processes multiple rows
- **Diagonal Scheduling**: Use diagonal grid scheduling for large matrices (above BLOCK_THRESHOLD) to improve L2 cache hit rate
- **Multi-Vector Core**: Use `tl.parallel(bind_sub_block=True)` to distribute post-dot operations to 2 vector cores

**Load**: [`optimization-patterns.md`](references/optimization-patterns.md) (§2.5 compile hints & inline APIs), [`tiling-strategies.md`](../triton-operator-shared/references/tiling-strategies.md), [`triton-api-reference.md`](../triton-operator-shared/references/triton-api-reference.md) (§4 AL ext, §5 BL ext, §7 NPUOptions compiler flags)

### Phase 4: Advanced Optimization (As Needed)

Operator fusion, Double Buffer (`al.multibuffer(tensor, 2)` or `al.compile_hint(tensor, "multi_buffer", 2)`)

### Phase 5: Verification (MANDATORY)

Precision + Generalization + Performance + End-to-end regression

## Reference Files

| File | Contents |
|---|---|
| [`optimization-patterns.md`](references/optimization-patterns.md) | §1 Hardware constraints table (UB/L1/alignment/Cube/core count/UB formula), §2 Core optimization rules (contiguity, single-pass, fusion, precision, constexpr, intra-core tiling), §2.5 Ascend-specific hints & APIs (al.compile_hint table, al.multibuffer, al.cast, sync_block_set/wait, num_warps note), §3 Reference kernels (GEMM/LayerNorm/Softmax/FlashAttn), §4 General pitfalls G1–G12, RoPE case study (8 pitfalls + perf table), GroupNorm case study |
| [`../triton-operator-shared/references/triton-api-reference.md`](../triton-operator-shared/references/triton-api-reference.md) | §1 triton top-level (@triton.jit, cdiv, compile), §2 tl.* full kernel language (all 80+ ops across 15 subsections), §3 triton.testing (do_bench, perf_report, Benchmark, assert_close, hardware query fns), §4 AL extension (al.*) all ops + enums, §5 BL extension (bl.*) alloc/to_buffer/to_tensor/subview, §6 module structure, §7 NPUOptions — all 50+ bishengir-compile flags with defaults and flag names |

- ❌ **Delete or replace a reference document without running a content audit first.** When restructuring reference files (merging, deleting, rewriting), enumerate every named item in the originals and verify each one is present in the new file before deleting anything. Use `execute_code` to automate the diff — do not rely on memory or manual scanning. Missing content is silent.

## Anti-Pattern Checklist (NEVER)

- ❌ **Write output files outside the kernel's own directory.** When optimizing a kernel in `/path/to/kernel_dir/`, ALL output files (`opt_*.py`, `profile_kernels.py`, `Optimizations.md`, `performance_report.md`, `review.md`) must be written ONLY inside that directory. Never write to a separate evaluation harness (e.g. `~/KernelGen/`), CompilerClaw datasets directory, or any temp path — even if those contain related-looking files. The user's request names exactly one directory; that is the sole output target.
- ❌ Make optimization decisions based solely on single-scale data
- ❌ Optimize kernel directly when end-to-end target is not met (use cannsim first to confirm bottleneck)
- ❌ Sacrifice precision for performance / hardcode that breaks generalization
- ❌ Reduce directly in FP16 / matrix multiplication with BLOCK not multiple of 16
- ❌ BLOCK_SIZE exceeding UB (192KB) / non-contiguous memory access
- ❌ Use `tensor.item()` in hot path (triggers CPU-NPU synchronization)
- ❌ **Use if branches inside loops to modify variables** (Triton compiles to masked operations, catastrophic performance degradation)
- ❌ **Use `if full/else` branching inside @triton.jit to pick masked vs mask-free load paths** — on Ascend this compiles to STI_XN_IMM / LD_XD_XN_IMM scalar register spills costing ~2,900 cycles/tile, far more than just using a uniform masked load on every tile. A single `mask = offsets < n_elements; tl.load(..., mask=mask)` path is always faster.
- ❌ **Use `tl.make_block_ptr` for simple 1D elementwise kernels** — adds SCALARLDST overhead (ST_XD_XN_IMM/LD_XD_XN_IMM) from block pointer struct construction. For 1D kernels use manual `offsets = start + tl.arange(0, BLOCK)` + explicit mask. `make_block_ptr` is worthwhile for 2D tiling with strided access (GELU, RoPE) but not for flat elementwise.
- ❌ **Use `cache_modifier=".cg"` on Ascend** — this CUDA L2-bypass hint causes Triton's `compile()` to silently produce no .npubin (empty _triton_dump, no exception). Drop unconditionally for Ascend kernels.
- ❌ **Calculate UB only for data buffer in 2D tiling** (must include offset/mask/index arrays)
- ❌ **Use precomputed offset tensors for 2D broadcasting** (triggers compiler addptr multi-user assertion)
- ❌ **Use broadcast stride to access auxiliary tensors (cos/sin, etc.) inside kernel** — change to host-side expand+contiguous. Expand extra memory < performance loss from non-contiguous access
- ❌ **Two-pass mode for reduction operators** (loading same data multiple times to compute statistics and normalization separately) — use single pass, compute everything within UB after one load, see Pitfall 8
- ✅ **Host-side scalar collapse for fused bias+sum**: when the operator computes `sum(x[b,:] + bias[:])`, pre-compute `bias_sum = bias.sum().item()` on the host and pass it as a scalar kernel arg. Inside the kernel: `total = tl.sum(vals.to(fp32)) + bias_sum`. This is always valid because `sum(x+b) = sum(x) + sum(b)` and eliminates a separate elementwise Add GPU op (~7 µs on real hardware). Works for any bias shape as long as the entire bias is added before summing.

- ❌ **Two kernels with the same name when comparing with msprof** — msprof aggregates by OP Type, same names will be mixed together
- Not using diagonal scheduling for large matrices (L2 cache thrashing, must enable above BLOCK_THRESHOLD)
- **Using `tl.compile_hint` instead of `al.compile_hint`** — `triton.language` has no
  `compile_hint` attribute. The real API is `al.compile_hint` from
  `triton.language.extra.cann.extension`. A compile-time shim that no-ops the call lets
  the kernel compile but the file then crashes at import time with
  `AttributeError: module 'triton.language' has no attribute 'compile_hint'`. Always
  `import triton.language.extra.cann.extension as al` and use `al.compile_hint(t, hint)`.
- **Blindly applying an episode pattern from one kernel to another** — the l1_2
  Standard matmul episode (v2 win 9.2× over v1) was specifically because its v1 used
  *dynamic* K range (`tl.range`); v2 added `tl.static_range` + `al.multibuffer` together,
  with `static_range` being the load-bearing optimization (SET_INTRA_BLOCKI 8→2 events).
  When the same v2 pattern was applied to l1_1 Square matmul (whose v1 already has
  `tl.static_range`), the multibuffer produced -0.6% at K=2 (within noise) and +4.6%
  regression at K=4 (MTE2 busy 3548→15602, new CRITICAL events from multibuffer sync
  overhead). **Always verify with a sub-kernel trace before adopting an episode pattern
  in a new kernel.**
- ❌ **Running full-size cannsim (e.g. 4096×4096) for bottleneck diagnosis** — large matrices with small blocks (32×32) generate 16000+ programs and take hours in cannsim. Always use the smallest matrix that exercises the same code path (256×256 or 512×512) for cannsim trace analysis. Reserve full-size for real hardware benchmarking.
- ❌ **Concluding a persistent-grid optimization failed because the sub-kernel trace is identical** — FFTS dispatch savings are invisible at sub-kernel scale (grid=1, 1 loop iteration). Sub-kernel traces only show per-tile instruction mix. For persistent-grid kernels, estimate the dispatch saving analytically: `saved_programs × 1150_cy × 0.40_ns`. Full-shape hardware measurement is required to confirm. A flat sub-kernel trace is expected, not a failure signal.

- ❌ **Applying persistent grid unconditionally to all shapes** — hardware-verified (l1_19_ReLU June 2026): persistent while-loop adds SCALAR overhead (while-condition check + `tile_id += n_programs`) on EVERY tile. When `n_tiles <= MAX_PROGRAMS` (65535), no FFTS savings are possible but the loop overhead is still paid — 1.35–1.43× SLOWER than direct dispatch at small N. The only shape where persistent helped was where n_tiles JUST exceeded 65535 (N=4M, 1.42× speedup). **Rule: use persistent kernel only when `cdiv(n_elements, MIN_BLOCK_SIZE) > 65535`**, where `MIN_BLOCK_SIZE` is the *smallest* BLOCK_SIZE in the autotune configs. Below that threshold use a direct (non-looping) kernel. Implement as two separate `@triton.jit` functions with a host-side dispatch branch:
  ```python
  MIN_BLOCK = 256   # smallest autotune config — determines the safe threshold
  MAX_PROGRAMS = 65535

  # Threshold uses MIN_BLOCK, not max BLOCK.
  # Reason: autotune tries ALL configs including BLOCK_SIZE=256.
  # If cdiv(n, 256) > 65535, the direct kernel's grid would exceed UINT16_MAX
  # (Ascend FFTS hard limit) causing a runtime crash — even if cdiv(n, 4096) is fine.
  # Hardware-verified crash: N=16M, cdiv(16M,4096)=4096 (fine), cdiv(16M,256)=65536 (CRASH).
  n_tiles_at_min_block = triton.cdiv(n, MIN_BLOCK)
  if n_tiles_at_min_block > MAX_PROGRAMS:
      _kernel_persistent[grid](...)    # while tile_id < n_elements; grid capped at 65535
  else:
      # Safe: cdiv(n, BLOCK_SIZE) <= cdiv(n, MIN_BLOCK) <= MAX_PROGRAMS for ALL configs
      _kernel_direct[direct_grid](...) # simple pid * BLOCK_SIZE, no cap needed
  ```

- ❌ **Using exact n_elements as the autotune key for large-N kernels** — autotune caches per unique key value. With `key=["n_elements"]` and n_elements=1,610,612,736, autotune runs all configs at that exact size on the first call (5 × full-tensor trials inside do_bench warmup). This caused 6.33× slowdown on the benchmark shape (l1_19_ReLU June 2026). **Rule: use a bucketed key** — `key=["n_elements_pow2"]` where `n_elements_pow2 = 1 << (n-1).bit_length()` (next power-of-2). This collapses all n_elements values to O(log N) ≈ 30 distinct cache entries. Never use an exact continuous integer (n_elements, N*C*H*W) as an autotune key.

- ❌ **Incorrect `_run_baseline` in profile_kernels.py for large-N kernels without a loop.** When the baseline kernel has no persistent loop and the benchmark runner caps its grid at 65535, it silently processes only `65535 × BLOCK_SIZE` elements — leaving the rest uninitialized. At the l1_19_ReLU bench shape (1.6B elements, BLOCK_SIZE=4096): `65535 × 4096 = 268M covered`, **83% of elements skipped**. The baseline appears 6.3× faster because it does wrong computation. **Fix: chunk at `65535 × MIN_BLOCK_SIZE` — NOT max BLOCK_SIZE.** Autotune tries all configs including the smallest; using max BLOCK_SIZE as the chunk bound still crashes when autotune picks a smaller config (hardware-verified: `65535 * 4096` chunk + autotune BLOCK_SIZE=256 → `cdiv(chunk, 256) = 1,048,320` → UINT16_MAX crash):
  ```python
  MIN_BLOCK = 256   # smallest BLOCK_SIZE in autotune configs — determines safe bound
  MAX_ELEMS = 65535 * MIN_BLOCK  # 16,776,960: guarantees cdiv(chunk, ANY_BLOCK) <= 65535
  if n <= MAX_ELEMS:
      _baseline_model(x)   # single launch, safe for all autotune configs
  else:
      x_flat = x.view(-1); y_flat = torch.empty_like(x_flat)
      for start in range(0, n, MAX_ELEMS):
          end = min(start + MAX_ELEMS, n)
          y_flat[start:end].copy_(_baseline_model(x_flat[start:end]))
      return y_flat.view_as(x)
  ```
  **Invariant**: `cdiv(chunk, BLOCK_SIZE) <= cdiv(chunk, MIN_BLOCK) <= 65535` for ALL autotune configs, since `BLOCK_SIZE >= MIN_BLOCK` always.

- ⚠️ **`tl.constexpr` for autotune-key-only args reduces SCALARLDST load count but not wall cycles.** Passing an autotune key like `n_elements_pow2` as `tl.constexpr` instead of `i32` removes its `LD_XD_XN_IMM` from the args-struct load sequence (verified: 3 loads → 1 load, 476 SCALARLDST busy-cy saved on a 2910-cy tile). However, the compiler compensates by emitting extra `LDP_XI_XJ_XN` (scalar register-file loads, parallel pipeline) to carry the same information. Net wall_cycles effect: zero (2910 → 2915, within noise). The constexpr form is still cleaner and avoids wasting a runtime arg slot, but do not expect a measurable speedup from this change alone.

- ❌ **Expecting cannsim sub-kernel traces to predict hardware latency for small N.** For elementwise kernels at small N (< ~50M elements), the hardware latency is dominated by the Python→CANN JIT dispatch stack (~40ms fixed cost), not per-tile compute. `torch.nn.functional` ops (pre-compiled ACL) pay ~1-2ms for the same dispatch. Cannsim per-tile cycles (e.g. 2910 cy = 1.16 µs per tile) are accurate for *tile-level* analysis but are meaningless as a hardware latency predictor when dispatch overhead is 40ms and compute is 0.001ms. **When the target is to beat torch on a standalone elementwise op: the only viable path is kernel fusion with an adjacent op.** There is no tile-level optimization that closes a 40ms dispatch gap.

## Checklist

- [ ] Precision aligns with PyTorch-NPU (rtol=1e-3, atol=1e-3)
- [ ] Non-aligned dimensions and boundaries pass
- [ ] Performance tests cover small/medium/large sizes
- [ ] grid ≤ number of physical cores, BLOCK_SIZE is compile-time constant
- [ ] Buffer < 192KB, all load/store have masks
- [ ] Reduction upcast to FP32, matrix multiplication BLOCK multiples of 16
- [ ] Is the reduction operator single-pass? (Required when D ≤ UB)
- [ ] Diagonal scheduling enabled for large matrices

## Common Bottleneck Quick Reference

| cannsim Metric | Bottleneck | Typical Optimization |
|---------------|------------|----------------------|
| aiv_scalar > 80% | Scalar Bound | Check two-pass / per-row loop + tl.where accumulation, change to single-pass |
| aiv_mte2 > 50% | Memory Bound | Contiguous memory access, expand+contiguous, increase BLOCK |
| aiv_vec > 50% | Compute Bound | Algorithm optimization, reduce redundant computation |
| aic_cube_ratio < 50% | Low Cube Utilization | Check alignment (512B / element size), whether BLOCK is multiple of 16, use compile_hint('dot_pad_only_k') |