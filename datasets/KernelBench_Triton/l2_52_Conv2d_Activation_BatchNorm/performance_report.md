# Performance Report

## Cannsim trace methodology

- Baseline sub-kernel: `_act_softplus_tanh_mul_kernel`, `BLOCK_SIZE=4096`, one 4096-element activation tile.
- Optimized sub-kernel: `_mish_direct_kernel`, `BLOCK_SIZE=8192`, one 8192-element activation tile.
- Trace source: `cannsim_local_run(..., gen_report=True)` and `aggregate_trace.py`.
- Cycle conversion: `hardware_time_ns = cycles * 0.4`.

## Cannsim trace summary

| Kernel | Elements / program | wall cycles | normalized cycles / 4096 elems | hw time / program | normalized hw time / 4096 elems | x events | bottleneck |
|---|---:|---:|---:|---:|---:|---:|---|
| Baseline direct | 4096 | 3952 | 3952.0 | 1.581 us | 1.581 us | 1004 | MTE3 `WAIT_FLAG_VEC` |
| Optimized direct | 8192 | 4817 | 2408.5 | 1.927 us | 0.963 us | 1900 | MTE3 `WAIT_FLAG_VEC` |

Normalized cannsim result: 39.1% fewer cycles per 4096 elements for the optimized tile.

## Pipeline utilization

| Pipeline | Baseline busy cycles | Optimized busy cycles | Notes |
|---|---:|---:|---|
| MTE3 | 2165 | 3038 | Optimized moves twice as many elements; per-element MTE3 wait is lower. |
| SCALAR | 1785 | 1777 | Scalar setup is essentially flat despite 2x elements. |
| PUSHQ | 721 | 1359 | Queue cost grows sublinearly vs elements. |
| RVECEX | 676 | 1314 | Vector math scales with element count. |
| MTE2 | 1013 | 1084 | Load setup mostly amortized by larger tile. |
| RVECST | 497 | 1007 | Store work scales with element count. |

## Top instruction costs

| Kernel | Instruction | Pipe | Count | Total cycles | Avg cycles |
|---|---|---|---:|---:|---:|
| Baseline | WAIT_FLAG_VEC | MTE3 | 1 | 1713 | 1713 |
| Baseline | RV_VDIV | RVECEX | 64 | 1088 | 17 |
| Baseline | RV_VEXP | RVECEX | 64 | 1024 | 16 |
| Optimized | WAIT_FLAG_VEC | MTE3 | 1 | 2422 | 2422 |
| Optimized | RV_VDIV | RVECEX | 128 | 2176 | 17 |
| Optimized | RV_VEXP | RVECEX | 128 | 2048 | 16 |

## Remote hardware benchmark

Remote `profile_kernels.py --test --bench` passed; latency values are milliseconds.

| label | PyTorch / ACL | Baseline Triton1 | Baseline Triton2 | Optimized Triton | Speedup vs Baseline Triton1 |
|---|---:|---:|---:|---:|---:|
| direct_tiny | 0.264229 | 0.199950 | 0.213662 | 0.210264 | 0.951x |
| direct_irregular | 0.422897 | 0.267303 | 0.279676 | 0.274420 | 0.974x |
| persistent_default | 111.227417 | 42.782085 | 39.964504 | 40.374432 | 1.060x |

Primary default-shape hardware latency improved from 42.782085 ms to 40.374432 ms vs the editable baseline. Small direct shapes regress slightly because the 8192 tile adds more per-program work where launch count is already low; the default KernelBench shape benefits from fewer programs.
