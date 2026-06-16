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

## Bottom Lines (Not to be Broken)

1. **Precision**: After optimization, rtol=1e-3, atol=1e-3 must align with PyTorch-NPU. Roll back if not met.
2. **Generalization**: Support all original input shapes and dtypes; do not hardcode specific sizes.

**Performance Ratio Definition**: `Ratio = torch_npu time / Triton time` (reciprocal of time). Ratio > 1.0 means Triton is faster.

**Priority**: Correctness > Generalization > Performance.

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

- ❌ Make optimization decisions based solely on single-scale data
- ❌ Optimize kernel directly when end-to-end target is not met (use cannsim first to confirm bottleneck)
- ❌ Sacrifice precision for performance / hardcode that breaks generalization
- ❌ Reduce directly in FP16 / matrix multiplication with BLOCK not multiple of 16
- ❌ BLOCK_SIZE exceeding UB (192KB) / non-contiguous memory access
- ❌ Use `tensor.item()` in hot path (triggers CPU-NPU synchronization)
- ❌ **Use if branches inside loops to modify variables** (Triton compiles to masked operations, catastrophic performance degradation)
- ❌ **Calculate UB only for data buffer in 2D tiling** (must include offset/mask/index arrays)
- ❌ **Use precomputed offset tensors for 2D broadcasting** (triggers compiler addptr multi-user assertion)
- ❌ **Use broadcast stride to access auxiliary tensors (cos/sin, etc.) inside kernel** — change to host-side expand+contiguous. Expand extra memory < performance loss from non-contiguous access
- ❌ **Two-pass mode for reduction operators** (loading same data multiple times to compute statistics and normalization separately) — use single pass, compute everything within UB after one load, see Pitfall 8
- ❌ **Two kernels with the same name when comparing with msprof** — msprof aggregates by OP Type, same names will be mixed together
- ❌ **Not using diagonal scheduling for large matrices** (L2 cache thrashing, must enable above BLOCK_THRESHOLD)

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
