# Performance Report

## Measurement setup

- cannsim: `cannsim_local_run(gen_report=True)`, grid=(1,), B=1, D=4096, fp32 inputs.
- Baseline trace: `/tmp/cannsim_local/l1_97_cosine_baseline_v2/.../trace_core0.json`.
- Optimized trace: `/tmp/cannsim_local/l1_97_cosine_opt_contig/.../trace_core0.json`.
- Cycle-to-time conversion: `cycles * 0.4 ns`.
- Hardware: `remote_verify(run_test=True, run_bench=True)` on Ascend NPU.

## Cannsim trace comparison

| Kernel | wall_cycles | HW-equivalent time (ns) | x_events | i_events | Bottleneck |
|---|---:|---:|---:|---:|---|
| Baseline Triton row kernel | 5,854 | 2,341.6 | 967 | 31 | PUSHQ 2,336 cycles |
| Optimized contiguous row kernel | 5,893 | 2,357.2 | 949 | 29 | PUSHQ 2,519 cycles |

## Pipeline utilization

| Pipeline | Baseline busy_cyc | Optimized busy_cyc | Delta |
|---|---:|---:|---:|
| PUSHQ | 2,336 | 2,519 | +183 |
| SCALAR | 1,960 | 1,909 | -51 |
| SCALARLDST | 1,855 | 1,860 | +5 |
| MTE2 | 1,168 | 1,067 | -101 |
| VEC | 705 | 1,031 | +326 |
| RVECEX | 625 | 625 | 0 |
| RVECLD | 510 | 510 | 0 |
| MTE3 | 334 | 333 | -1 |
| RVECST | 48 | 48 | 0 |

## Top instructions

| Kernel | Top instruction | Pipe | Count | Total cycles | Avg cycles |
|---|---|---|---:|---:|---:|
| Baseline | RV_VCADD | RVECEX | 192 | 4,224 | 22 |
| Baseline | LDP_XI_XJ_XN | SCALAR | 5 | 2,464 | 493 |
| Baseline | VF | PUSHQ | 5 | 2,316 | 463 |
| Optimized | RV_VCADD | RVECEX | 192 | 4,224 | 22 |
| Optimized | VF | PUSHQ | 5 | 2,499 | 500 |
| Optimized | LD_XD_XN_IMM | SCALARLDST | 6 | 2,225 | 371 |

## Hardware benchmark

| label | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 (ms) | Optimized Triton (ms) | Speedup vs Baseline Triton1 |
|---|---:|---:|---:|---:|---:|
| small_4x128 | 0.022747 | 0.004150 | 0.004110 | 0.004162 | 0.997x |
| irregular_129x1000 | 0.036510 | 0.009404 | 0.006385 | 0.009424 | 0.998x |
| exact_128x4096 | 0.044070 | 0.012052 | 0.010857 | 0.012005 | 1.004x |

Correctness: all baseline1, baseline2, optimized, and optimized persistent-dispatch tests passed with `max_diff=0`; `UNIT_TEST PASS`.
