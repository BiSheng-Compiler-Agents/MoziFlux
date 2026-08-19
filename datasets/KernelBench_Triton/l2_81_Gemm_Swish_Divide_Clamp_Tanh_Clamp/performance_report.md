# Performance Report

## Cannsim setup

Both traces simulate the fused elementwise epilogue as a sub-kernel with `grid=(1,)`, using fp32 buffers and correctness checks in the C++ host. Baseline uses the input kernel's `BLOCK=1024`; optimized uses the production `BLOCK=4096`. `cannsim_local_run(..., gen_report=True)` generated both `trace_core0.json` files.

- Baseline trace: `/tmp/cannsim_local/l2_81_epilogue_baseline/cannsim_20260630071316_test_kernel/report/trace_core0.json`
- Optimized trace: `/tmp/cannsim_local/l2_81_epilogue_opt/cannsim_20260630071503_test_kernel/report/trace_core0.json`

## Trace summary

| Kernel | Elements in trace | Wall cycles | HW latency ns (`cycles*0.4`) | Cycles/elem | Elems/cycle | Bottleneck |
|---|---:|---:|---:|---:|---:|---|
| Baseline epilogue | 1,024 | 3,989 | 1,595.6 | 3.8955 | 0.2567 | MTE3 / `WAIT_FLAG_VEC` |
| Optimized epilogue | 4,096 | 4,986 | 1,994.4 | 1.2173 | 0.8215 | MTE3 / `WAIT_FLAG_VEC` |

Normalized throughput improved **3.20x** (`0.8215 / 0.2567`). Absolute sub-kernel cycles are higher because the optimized trace processes 4x as many elements per program.

## Pipeline utilization

| Kernel | MTE3 busy | SCALAR busy | MTE2 busy | VEC busy | PUSHQ busy | RVECEX busy | RVECLD/RVECST busy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 2,202 | 1,785 | 967 | 961 | 882 | 833 | 144 / 144 |
| Optimized | 3,189 | 1,795 | 1,017 | 1,011 | 1,745 | 1,696 | 568 / 573 |

## Estimated full target epilogue cost

For the target `(batch_size, out_features) = (1024, 8192)`, there are 8,388,608 epilogue elements.

| Kernel | Tile size | Tiles | Estimated cycles | Estimated HW latency ms |
|---|---:|---:|---:|---:|
| Baseline epilogue | 1,024 | 8,192 | 32,677,888 | 13.071 |
| Optimized epilogue | 4,096 | 2,048 | 10,211,328 | 4.085 |

This estimate is derived from sub-kernel trace cycles per tile and is intended for epilogue comparison only; the full model also includes the backend GEMM/linear call.

## Correctness evidence

Both cannsim C++ hosts reported `[HOST] PASS` after comparing every simulated element against the fp32 reference formula over non-zero positive and negative inputs.
