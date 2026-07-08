# Performance Report

## cannsim setup

- Baseline simulated kernel: `_swish_scale_kernel`, `BLOCK_SIZE=8192`, `N=8192`, `grid=(1,)`
- Optimized simulated kernel: `_swish_scale_direct`, `BLOCK_SIZE=8192`, `N=8192`, `grid=(1,)`
- Tool: `cannsim_local_run(..., gen_report=True)` with `trace_core0.json`
- Trace paths:
  - Baseline: `/tmp/cannsim_local/l2_59_swish_baseline/cannsim_20260630001141_test_kernel/report/trace_core0.json`
  - Optimized: `/tmp/cannsim_local/l2_59_swish_optimized_8192/cannsim_20260630001536_test_kernel/report/trace_core0.json`

## cannsim trace summary

| Metric | Baseline | Optimized | Delta |
|---|---:|---:|---:|
| Elements per sub-kernel | 8192 | 8192 | same |
| wall_cycles | 4627 | 4041 | -12.7% |
| hardware latency (cycles × 0.4 ns) | 1850.8 ns | 1616.4 ns | -234.4 ns |
| cycles / element | 0.565 | 0.493 | -12.7% |
| trace events (`x_events`) | 1640 | 998 | -39.1% |
| bottleneck pipeline | MTE3 | MTE3 | unchanged |

## Pipeline utilization

| Pipeline | Baseline busy cycles | Optimized busy cycles | Delta |
|---|---:|---:|---:|
| MTE3 | 2840 | 2264 | -20.3% |
| SCALAR | 1785 | 1775 | -0.6% |
| SCALARLDST | 1745 | 1735 | -0.6% |
| MTE2 | 1064 | 1068 | +0.4% |
| VEC | 1058 | 1062 | +0.4% |
| PUSHQ | 1186 | 608 | -48.7% |
| RVECEX | 1141 | 562 | -50.7% |
| RVECLD | 971 | 503 | -48.2% |
| RVECST | 691 | 519 | -24.9% |

## Top instruction changes

| Instruction | Baseline total cycles / count | Optimized total cycles / count | Impact |
|---|---:|---:|---|
| RV_VDIV | 4352 / 256 | 2176 / 128 | one sigmoid divide path removed |
| RV_VEXP | 2048 / 128 | 2048 / 128 | same exp count at same tile size |
| RV_VMULS | 2048 / 256 | 2048 / 256 | same multiply count |
| WAIT_FLAG_VEC | 2231 / 1 | 1657 / 1 | lower vector wait after reduced vector work |
| VF (PUSHQ) | 1182 / 1 | not in top-12 | queue pressure reduced |

## Remote hardware benchmark

`remote_verify` result: `test_passed=true`, `bench_passed=true`.

| Shape | PyTorch / ACL ms | Baseline Triton1 ms | Optimized Triton ms | Optimized vs Baseline |
|---|---:|---:|---:|---:|
| small_4x1024x1024 | 0.090410 | 0.173191 | 0.168977 | 1.025x |
| medium_16x4096x4096 | 0.387783 | 1.163948 | 1.163936 | 1.000x |
| default_128x32768x32768 | 23.343973 | 68.357521 | 68.049301 | 1.005x |

## Correctness

| Provider/path | Result |
|---|---|
| Baseline Triton1 all profile shapes | PASS, max_abs ≤ 0.00195312 |
| Optimized direct path all profile shapes | PASS, max_abs = 5.96046e-08 |
| Optimized forced persistent path | PASS, max_abs = 5.96046e-08 |
| Baseline Triton2 | SKIP_UNAVAILABLE because sandbox forbids reading `base_*.py` |
