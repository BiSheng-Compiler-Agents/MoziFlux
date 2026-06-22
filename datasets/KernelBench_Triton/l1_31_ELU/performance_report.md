# Performance Report

## Cannsim setup

- Baseline trace: `/tmp/cannsim_local/cannsim_baseline_1782327657/cannsim_20260624190117_test_kernel/report/trace_core0.json`
- Optimized trace: `/tmp/cannsim_local/cannsim_optimized_1782327802/cannsim_20260624190342_test_kernel/report/trace_core0.json`
- Sub-kernel: fp32 ELU, `N=8192`, `grid=(1,)`, `BLOCK_SIZE=8192`.
- Note: the primary full-shape optimization is legal capped dispatch; sub-kernel cannsim does not measure FFTS grid-overflow avoidance.

## Cannsim summary

| Kernel | wall cycles | hw time (ns, 0.4ns/cycle) | x events | Bottleneck |
|---|---:|---:|---:|---|
| Baseline direct | 4758 | 1903.2 | 2170 | MTE3 / `WAIT_FLAG_VEC` |
| Optimized direct | 4724 | 1889.6 | 2172 | MTE3 / `WAIT_FLAG_VEC` |
| Delta | -34 (-0.71%) | -13.6 | +2 | Same bottleneck |

## Pipeline table

| Pipeline | Baseline busy cyc | Optimized busy cyc | Delta |
|---|---:|---:|---:|
| MTE3 | 2974 | 2930 | -44 |
| SCALAR | 1782 | 1792 | +10 |
| SCALARLDST | 1732 | 1744 | +12 |
| PUSHQ | 1322 | 1271 | -51 |
| RVECEX | 1273 | 1224 | -49 |
| RVECLD | 1084 | 1086 | +2 |
| MTE2 | 1067 | 1069 | +2 |
| VEC | 1061 | 1063 | +2 |
| RVECST | 934 | 940 | +6 |
| FLOWCTRL | 7 | 7 | 0 |

## Top instruction costs

| Instruction | Pipe | Baseline total cyc | Optimized total cyc | Note |
|---|---|---:|---:|---|
| `WAIT_FLAG_VEC` | MTE3 | 2356 | 2311 | dominant stall |
| `RV_VMULS` | RVECEX | 3072 | 2048 | optimized trace emits fewer VMULS |
| `RV_VEXP` | RVECEX | 2048 | 2048 | ELU exp cost is intrinsic |
| `LD_XD_XN_IMM` | SCALARLDST | 1699 | 1711 | scalar load/setup |
| `VF` | PUSHQ | 1314 | 1265 | queue pressure slightly lower |
| `STI_XN_IMM` | SCALAR | 1213 | 1226 | scalar store/setup |

## Hardware verification

Remote directory: `/home/s00929845/kernel_verify/l1_31_ELU_1782328239`.

| Shape | PyTorch / ACL ms | Baseline Triton1 ms | Baseline Triton2 ms | Optimized Triton ms | Result |
|---|---:|---:|---:|---:|---|
| direct_1M | 0.009581 | 0.010896 | 0.011142 | 0.010920 | PASS |
| direct_irregular | 0.009582 | 0.011070 | 0.011152 | 0.011122 | PASS |
| persistent_original | 8.923337 | inf (skipped grid overflow) | 9.190184 | 9.377321 | PASS |

## Interpretation

- Correctness passed for optimized direct and original oversized paths (`UNIT_TEST PASS`).
- The editable baseline cannot launch on the original shape because `coreDim` would exceed 65535; optimized Triton is the legal Triton dispatch for that path.
- Sub-kernel trace shows similar per-tile work because the ELU math and tile size are intentionally unchanged; the optimization is mainly full-shape launch legality plus 64-bit offset correctness.
