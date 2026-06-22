# Performance Report

## Cannsim sub-kernel traces

All cannsim runs used `grid=(1,1,1)` sub-kernel hosts and generated `trace_core0.json` via `cannsim_local_run(gen_report=True)`.

| Kernel | Trace path | Wall cycles | Time (cycles*0.4ns) | Bottleneck | Top critical instruction |
|---|---:|---:|---:|---|---|
| Baseline single tile | `/tmp/cannsim_local/l1_34_instancenorm_baseline/.../trace_core0.json` | 5087 | 2.035 us | PUSHQ 2888 cy | `VF` 2860 cy |
| Optimized partial | `/tmp/cannsim_local/l1_34_instancenorm_opt_partial/.../trace_core0.json` | 2280 | 0.912 us | MTE2 1013 cy | `RV_VCADD` 1408 cy / `MOV_SRC_TO_DST_ALIGNv2` 998 cy |
| Optimized finalize | `/tmp/cannsim_local/l1_34_instancenorm_opt_finalize/.../trace_core0.json` | 5759 | 2.304 us | PUSHQ 3447 cy | `VF` 3419 cy |
| Optimized apply | `/tmp/cannsim_local/l1_34_instancenorm_opt_apply_v2/.../trace_core0.json` | 3063 | 1.225 us | SCALARLDST 1682 cy | `LDP_XI_XJ_XN` 1433 cy |

### Pipeline breakdown

| Kernel | PUSHQ | MTE2 | VEC | MTE3 | SCALAR | SCALARLDST | RVECEX |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 2888 | 1011 | 975 | 810 | 739 | 641 | 323 |
| Opt partial | 228 | 1013 | 969 | 460 | 597 | 526 | 182 |
| Opt finalize | 3447 | 889 | 0 | 645 | 630 | 276 | 127 |
| Opt apply | 101 | 835 | 832 | 1305 | 560 | 1682 | 54 |

## Hardware verification

`remote_verify(run_test=True, run_bench=True)` passed correctness for baseline1, baseline2, and optimized on all profiler test shapes.

| Shape | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 (ms) | Optimized Triton (ms) | Opt vs Baseline1 |
|---|---:|---:|---:|---:|---:|
| small_direct_8x64x32x32 | 0.012078 | 0.016165 | 0.021014 | 0.017068 | 0.95x |
| medium_direct_16x64x128x128 | 0.152844 | 0.156459 | 0.156452 | 0.158587 | 0.99x |
| large_direct_32x64x128x128 | 0.302964 | 0.257354 | 0.257532 | 0.274182 | 0.94x |

The direct fallback is correctness-preserving and near baseline on measured hardware. The persistent oversized-grid path is implemented and cannsim-validated by sub-kernels; full 112x64x512x512 remote benchmarking exceeded the verification timeout in earlier attempts, so the hardware table reports bounded profiler shapes.
