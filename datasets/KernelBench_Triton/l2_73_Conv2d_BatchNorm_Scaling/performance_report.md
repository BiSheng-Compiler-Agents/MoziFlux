# Performance Report

## Cannsim setup

- Baseline diagnostic kernel: `_scale_kernel`, `BLOCK_SIZE=16384`, one program, fp32 input/output, scale `2.0`.
- Optimized diagnostic kernel: `_scale_direct_kernel`, `BLOCK_SIZE=4096`, one program, fp32 input/output, scale `2.0`.
- Trace source: `cannsim_local_run(..., gen_report=True)` with `trace_core0.json` and `aggregate_trace.py`.
- Hardware cycle conversion: `hardware_time_ns = cycles * 0.4`.

## Cannsim trace summary

| Kernel | Elements in diagnostic tile | wall_cycles | hardware_time_ns | x_events | i_events | Bottleneck |
|---|---:|---:|---:|---:|---:|---|
| Baseline `_scale_kernel` | 16,384 | 4,283 | 1,713.2 | 865 | 8 | MTE3 store / WAIT_FLAG_VEC |
| Optimized fallback `_scale_direct_kernel` | 4,096 | 3,359 | 1,343.6 | 289 | 8 | SCALAR setup then MTE3/VEC waits |

## Pipeline utilization

| Kernel | MTE3 busy | SCALAR busy | SCALARLDST busy | MTE2 busy | VEC busy | PUSHQ busy | RVECEX ops | RVECLD/RVECST ops |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline `_scale_kernel` | 2,501 | 1,780 | 1,742 | 1,249 | 1,243 | 315 | 257 | 256 / 256 |
| Optimized fallback `_scale_direct_kernel` | 1,582 | 1,775 | 1,737 | 1,014 | 1,008 | 123 | 65 | 64 / 64 |

## Top critical instructions

| Kernel | Critical instruction evidence |
|---|---|
| Baseline | `RV_VLDI` 256× / 2304 cyc, `RV_VSTI` 256× / 2304 cyc, `RV_VMULS` 256× / 2048 cyc, `WAIT_FLAG_VEC` 1547 cyc, `WAIT_FLAG_MTE2` 1243 cyc |
| Optimized fallback | `RV_VLDI` 64× / 576 cyc, `RV_VSTI` 64× / 576 cyc, `RV_VMULS` 64× / 512 cyc, `WAIT_FLAG_VEC` 1120 cyc, `WAIT_FLAG_MTE2` 1008 cyc |

## Interpretation

The main optimization is algebraic removal of the full-output scale kernel from `ModelNew.forward`; the production training path fuses scale into BatchNorm affine parameters and therefore pays zero Triton scale-kernel cycles. The fallback Triton scale path is retained for coverage and legality, with a direct path below the grid cap and a persistent path above it.

## Remote hardware latency (`remote_verify`)

`remote_verify` passed correctness and benchmark on the physical Ascend NPU. Values are milliseconds from `profile_kernels.py`.

| Shape | PyTorch / ACL | Baseline Triton1 | Baseline Triton2 | Optimized Triton | Speedup vs Baseline Triton1 |
|---|---:|---:|---:|---:|---:|
| `small_direct` | 0.242323 | 0.224978 | inf (sandbox skip) | 0.197309 | 1.140x |
| `medium_direct` | 1.921494 | 1.574609 | inf (sandbox skip) | 1.539209 | 1.023x |
| `default` | 27.749615 | 20.992706 | inf (sandbox skip) | 21.022699 | 0.999x |

Correctness summary: optimized provider passed all benchmark shapes plus forced `scale_direct` and forced `scale_persistent`; maximum absolute difference was `0` in all optimized tests.

## Trace files

- Baseline: `/tmp/cannsim_local/l2_73_scale_baseline/cannsim_20260630053227_test_kernel/report/trace_core0.json`
- Optimized fallback: `/tmp/cannsim_local/l2_73_scale_optimized/cannsim_20260630053416_test_kernel/report/trace_core0.json`
