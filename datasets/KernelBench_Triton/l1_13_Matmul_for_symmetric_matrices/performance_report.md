# Performance Report — `l1_13_Matmul_for_symmetric_matrices`

## Environment

| Parameter | Value |
|-----------|-------|
| **Target** | Ascend950 (cannsim SOC version) |
| **Compiler Arch** | Ascend910_9589 |
| **Data Type** | FP32 |
| **Layout** | Row-major (M×K · K×N = M×N) |
| **cannsim version** | CANN 9.0.0 |
| **Simulation mode** | CA mode, ref_period=0.40ns, self_period=0.61ns |

## Sub-Kernel Trace Comparison

Both traces use grid=1×1×1 (single tile, single core) for direct per-tile comparison.
The baseline uses 32×32 tiles (1,024 output elements); the optimized uses 128×128 tiles
(16,384 output elements — 16× more work per tile). Both use 2 K-loop iterations.

### Baseline Configuration

- Kernel: `_symmetric_matmul_kernel` (dynamic K loop)
- BLOCK: 32×32×32
- Grid: 2D (pid_m, pid_n)
- Masks: re-computed inside K loop
- No compile_hint, multibuffer, or care_padding

### Optimized Configuration

- Kernel: `_symmetric_matmul_kernel_opt` (K as constexpr, range K loop)
- BLOCK: 128×128×32 (autotune with 9 configs)
- Grid: 1D with GROUP_M=8 swizzle
- Masks: hoisted (m_mask, n_mask outside K loop)
- `al.compile_hint("dot_pad_only_k")` on A and B
- `al.multibuffer(a, size=2)` and `al.multibuffer(b, size=2)`
- `care_padding=False` on all loads

### Pipeline Busy Cycles

| Pipeline Unit | Baseline (cyc) | Optimized (cyc) | Change |
|---------------|:--------------:|:---------------:|:------:|
| wall_cycles (max end) | 12,559 | 20,655 | +64.5% *(16× more elements)* |
| 01_SCALAR | 9,788 | 8,821 | **-9.9%** |
| 02_SCALARLDST | 48,602 | 48,785 | +0.4% |
| 03_MTE1 | 470 | 2,742 | +483% |
| 04_MTE2 | 9,682 | 7,171 | **-25.9%** |
| 05_VEC | 10,422 | 4,423 | **-57.6%** |
| 06_CUBE | 640 | 4,542 | **+710%** |
| 07_MTE3 | 8,470 | 10,216 | +20.6% |
| 10_PUSHQ | 4,328 | 9,220 | +113% |
| 13_RVECLD | 4,608 | 18,432 | +300% |
| 14_RVECST | 6,456 | 25,946 | +302% |
| 15_FLOWCTRL | 5,364 | 9,489 | +76.9% |
| 17_FIXP | 664 | 3,273 | +393% |

### Per-Element Performance

| Metric | Baseline | Optimized | Improvement |
|--------|:-------:|:---------:|:----------:|
| Output elements / tile | 1,024 | 16,384 | 16× |
| wall_cycles | 12,559 | 20,655 | +64.5% |
| **cyc/elem** | **12.26** | **1.26** | **9.73× speedup** |

### Key Instruction Analysis

| Instruction | Baseline | Optimized | Note |
|-------------|:--------:|:---------:|:----:|
| SET_INTRA_BLOCKI count | 8 | 8 | Same K iterations (both use range) |
| RV_VLDI count | 512 | 2,048 | 4× from larger tiles |
| RV_VSTI count | 512 | 2,048 | 4× from larger tiles |
| ST_XD_XN_IMM (SCALARLDST) | 39,362 cyc | 43,528 cyc | +10.6% (larger tile, more output) |
| MMAD (CUBE) | 518 cyc | 4,230 cyc | 8.2× more Cube work |
| WAIT_FLAG_MTE2@VEC | 5,211 cyc | 2,187 cyc | **-58.0%** (multibuffer effect) |
| WAIT_FLAG_VEC@MTE3 | 8,168 cyc | 8,991 cyc | +10.1% (larger output tile) |

### Bottleneck Distribution

**Baseline** (`% of wall_cycles`, not directly comparable due to parallel pipelines):
- FLOWCTRL: 5,364 (dominant sync overhead from 8 SET_INTRA_BLOCKI events)
- SCALARLDST: 48,602 (scalar register spill from dynamic K-loop ptr arithmetic)
- CUBE: 640 (5.1% utilization — severely underutilized)
- MTE2: 9,682 (memory load, waiting on VEC)
- VEC: 10,422 (waiting on MTE2 and MTE3)

**Optimized**:
- CUBE: 4,542 (22.0% utilization — 7.1× more Cube work)
- MTE3: 10,216 (output store — inherent bottleneck, 49.5% of wall)
- PUSHQ: 9,220 (VF sync from multibuffer)
- FLOWCTRL: 9,489 (SET_INTRA_BLOCKI from 8 events)
- SCALARLDST: 48,785 (output store address computation)

### Hardware Latency Estimate

Real NPU hardware latency cannot be accurately predicted from sub-kernel cannsim
traces alone due to:
1. **FFTS dispatch overhead**: ~1,150 cycles per program × number of programs
2. **Grid scheduling**: GROUP_M swizzle benefits multi-tile L2 reuse (not visible at grid=1)
3. **Host launch latency**: Python→CANN dispatch adds ~40ms fixed overhead

**TBD** — requires real Ascend NPU hardware measurement via `profile_kernels.py`:

```bash
python profile_kernels.py --bench
```

Expected real-hardware trends:
- **Small shapes** (M=N=K≤256): Grid dispatch overhead dominates. Baseline with
  2D grid may be competitive; optimized with GROUP_M swizzle has slightly more
  launch overhead but better per-tile compute.
- **Medium shapes** (512≤M=N=K≤2048): Optimized kernel benefits from CUBE
  efficiency and GROUP_M swizzle. Expected 3-8× over baseline.
- **Large shapes** (M=N=K≥4096): Optimized kernel with diagonal scheduling
  avoids L2 cache thrashing. Expected 5-10× over baseline.

### Correctness Validation

The cannsim simulation includes a host-side correctness check comparing the
kernel output against a CPU reference matmul:

| Configuration | Result |
|--------------|:------:|
| Baseline (32×32×32, M=32 N=32 K=64) | ✅ PASS (tol=1e-3 rel) |
| Optimized (128×128×32, M=128 N=128 K=64) | ✅ PASS (tol=1e-3 rel) |

Both baseline and optimized kernels produce numerically correct results within
the expected tolerance (rtol=1e-3, atol=1e-3).
