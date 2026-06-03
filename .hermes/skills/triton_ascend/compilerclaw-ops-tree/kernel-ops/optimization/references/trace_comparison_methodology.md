# Sub-Kernel Trace Comparison Methodology

How to set up and interpret baseline-vs-optimized cann sim traces for any kernel.
This pattern was consolidated from l1_1, l1_2, l1_3, l2_8, and l2_9 optimizations
(June 2026). It is the standard way to measure per-tile optimization impact.

## Prerequisites

- A working sub-kernel host for the baseline kernel (simulation/SKILL.md Rule 1)
- A working sub-kernel host for the optimized kernel (may differ only in .npubin)
- `cannsim_local_run` or `cannsim_remote_run` configured
- `aggregate_trace.py` from `kernel-ops/simulation/scripts/`

## Workflow

### Step 1: Create Two Identical C++ Hosts

The two host launchers must differ ONLY in the `.npubin` they load. Use the same
sub-kernel dimensions (M=BLOCK_M, N=BLOCK_N, K=2×BLOCK_K, grid=1×1×1), same
random seed, same correctness-check code.

```cpp
// Baseline dir:
//   compile_kernel.py   — baseline kernel (no optimizations)
//   test_kernel.cpp     — same C++ host as optimized (identical code)
//   run_kernel.sh       — same shell wrapper

// Optimized dir:
//   compile_kernel.py   — optimized kernel
//   test_kernel.cpp     — IDENTICAL to baseline/test_kernel.cpp
//   run_kernel.sh       — IDENTICAL to baseline/run_kernel.sh
```

If the baseline already has a C++ host, copy it verbatim to the optimized dir.
The only change between the two runs is the Python kernel compiled into the .npubin.

### Step 2: Compile with the Same Constexpr Block Sizes

Both kernels must use the same `BLOCK_M`, `BLOCK_N`, `BLOCK_K` constexpr values.
The sub-kernel host must pass the same runtime `SUB_M`, `SUB_N`, `SUB_K`.

```python
# compile_kernel.py (both baseline and optimized)
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 64
GROUP_M = 8
```

```cpp
// test_kernel.cpp (both hosts)
constexpr int SUB_M = 128;
constexpr int SUB_N = 128;
constexpr int SUB_K = 128;  // 2 * BLOCK_K
```

### Step 3: Run Both Through Cannsim

```python
# Baseline
result_bl = cannsim_local_run(
    local_dir="path/to/baseline_cannsim_work",
    run_script="run_kernel.sh",
    binary_name="test_kernel",
    gen_report=True,
)

# Optimized
result_opt = cannsim_local_run(
    local_dir="path/to/opt_cannsim_work",
    run_script="run_kernel.sh",
    binary_name="test_kernel",
    gen_report=True,
)
```

### Step 4: Aggregate Both Traces

```bash
python scripts/aggregate_trace.py /path/to/baseline/report/trace_core0.json
python scripts/aggregate_trace.py /path/to/optimized/report/trace_core0.json
```

Read the two `trace_summary.txt` outputs side by side.

## Reading the Comparison

### wall_cycles — Primary Metric

The single most important number. A reduction means the per-tile instruction mix
was improved. For sub-kernel scale (1 tile, 2 K-iterations), even 3–5% is meaningful
because it compounds across all tiles and K-iterations at full scale.

**Full-shape wall time estimate:**
```
sub_kernel_savings = baseline_wall - optimized_wall  # cycles per tile
total_tiles = BATCH × ceil(M/BLOCK_M) × ceil(N/BLOCK_N)
total_k_iters = ceil(K / BLOCK_K)
full_shape_savings = sub_kernel_savings × total_tiles × (total_k_iters - 2)  # sub-kernel uses 2 iters
```

### Pipeline busy_cyc — What Changed and What Didn't

| Type | Meaning | What to check |
|------|---------|---------------|
| **Decreased** | That pipeline is doing less work | Improvement from your optimization |
| **Unchanged** | The optimization didn't affect that pipeline | Expected — e.g. CUBE is identical because same BLOCK sizes |
| **Increased** | Regression or new overhead | Investigate — could be from added instructions |

**Common unchanged pipelines (should be identical in baseline vs optimized):**
- `RVECST` / `RVECLD` — same number of store/load instructions for same BLOCK sizes
- `CUBE` — same matrix multiply work for same tile dimensions
- `MTE1` — same L1→L0A/B movement

**Pipelines that should decrease with specific optimizations:**
| Optimization | Pipelines affected |
|-------------|-------------------|
| Mask hoisting | SCALARLDST ops count, ST_XD_XN_IMM total |
| care_padding=False | MTE2 busy_cyc, VEC busy_cyc |
| multibuffer | MTE2 busy_cyc (overlap), WAIT_FLAG_MTE2/MTE3 |
| compile_hint(dot_pad_only_k) | UB pressure reduction (indirect — MTE2, VEC) |

### Top Instructions — Attribution

Each instruction has a pipeline owner. The delta tells you what changed:

| Instruction | Pipeline | Optimization Signal |
|-------------|----------|---------------------|
| `ST_XD_XN_IMM` | SCALARLDST | Scalar spill reduction (mask hoisting, constexpr elimination) |
| `LD_XD_XN_IMM` | SCALARLDST | Fewer scalar loads (hoisted masks) |
| `WAIT_FLAG_VEC` | MTE3 | Memory write stalls reduced (multibuffer overlap) |
| `WAIT_FLAG_MTE2` | VEC | Memory read stalls reduced (multibuffer, care_padding) |
| `WAIT_FLAG_MTE3` | VEC | Memory write stalls on vector side (multibuffer) |
| `SET_INTRA_BLOCKI` | FLOWCTRL | Structural cost — NOT reducible from user code |
| `RV_VSTI` | RVECST | Vector stores — scales with BLOCK area, not optimization |

### BOTTLENECK Annotation

The pipeline with highest busy_cyc is annotated `← BOTTLENECK`. This tells you
which pipeline constrains overall throughput. A successful optimization may either:
1. Reduce the bottleneck pipeline's busy_cyc directly
2. Leave it unchanged but reduce secondary bottlenecks (improving overall wall cycles)

### CRITICAL Annotation

An instruction is `← CRITICAL` if its per-event avg_cyc ≥ 25% of wall_cycles,
OR its total_cyc dominates. These are the high-leverage targets:
- `WAIT_FLAG_*` → memory/compute overlap optimization (multibuffer, pipelining)
- `ST_XD_XN_IMM` (avg high) → scalar spill (constexpr hoisting, mask hoisting)
- `SET_INTRA_BLOCKI` (avg high) → structural, accept

## Pitfalls

1. **Sub-kernel K-iteration count matters.** With K=2×BLOCK_K (2 iterations), the
   per-tile trace underrepresents multibuffer benefit. Multibuffer pipelines next-iteration
   DMA with current-iteration compute — with only 2 iters, there's 1 overlap opportunity.
   With 16 iters, there are 15 overlaps. The sub-kernel delta is an underestimate.

2. **FFTS dispatch savings are invisible.** If the optimization changes the grid
   (e.g. persistent grid), the per-tile trace is identical. See Rule 8 in the
   optimization skill for the full analysis.

3. **Do not compare across different BLOCK sizes.** Different BLOCK_M/N/K produce
   different instruction mixes. Always compare baseline vs optimized at the SAME
   block size. The autotune selection is tested on real hardware, not in cannsim.

4. **wall_cycles delta in single digits (1–5%) is meaningful at sub-kernel scale.**
   A 3.7% per-tile improvement across 100K+ tile iterations is a large absolute gain.
   Do not dismiss small percentages without projecting to full scale.

5. **BUILD EACH KERNEL SEPARATELY** — do not reuse the same .npubin across baseline
   and optimized directories. The compile step embeds the kernel IR. Even if the
   Python kernel source differs by one line, compile it fresh for each directory.
