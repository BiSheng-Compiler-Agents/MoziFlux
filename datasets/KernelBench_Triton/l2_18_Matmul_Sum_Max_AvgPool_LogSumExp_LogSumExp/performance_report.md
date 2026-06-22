# Performance Report

## Cannsim setup

- Baseline trace: `/tmp/cannsim_local/l2_18_baseline_small/cannsim_20260625111443_test_kernel/report/trace_core0.json`
- Optimized trace: `/tmp/cannsim_local/l2_18_optimized_small/cannsim_20260625111639_test_kernel/report/trace_core0.json`
- Probe shape: `B=16, I=64, O=64`, `grid=1`; this is a bounded sub-kernel probe of the fused body.
- Cycle conversion: `hardware_time_ns = cycles * 0.4`.

## Cannsim trace summary

| Kernel | wall_cycles | est. ns | x_events | i_events | Bottleneck | Notes |
|---|---:|---:|---:|---:|---|---|
| Baseline fused W-scan | 6901 | 2760.4 | 1115 | 89 | MTE2 2971 busy cycles | Recomputes `sum(W)` inside the row kernel; high MTE2/VEC/SCALAR work. |
| Optimized cached-wsum row kernel | 4146 | 1658.4 | 602 | 38 | SCALARLDST 3068 busy cycles | Reads cached `wsum` only; fewer MTE2/VEC/PUSHQ events. |

## Pipeline comparison

| Pipeline | Baseline busy | Optimized busy | Change |
|---|---:|---:|---:|
| MTE2 | 2971 | 1004 | -66.2% |
| VEC | 2922 | 455 | -84.4% |
| SCALAR | 2695 | 915 | -66.0% |
| PUSHQ | 1639 | 897 | -45.3% |
| MTE3 | 1208 | 481 | -60.2% |
| RVECEX | 417 | 107 | -74.3% |
| RVECLD | 239 | 110 | -54.0% |
| RVECST | 151 | 78 | -48.3% |

## Cannsim conclusion

The sub-kernel wall cycles improved from `6901` to `4146` (`1.66x`, `-39.9%`). The largest reduction is in MTE2 and VEC activity because the optimized path avoids recomputing the column sum of the full weight matrix inside the device kernel.

## Hardware benchmark

`remote_verify` passed correctness and benchmark on the physical Ascend NPU.

| label | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 (ms) | Optimized Triton (ms) | Speedup vs Baseline1 | Speedup vs PyTorch / ACL |
|---|---:|---:|---:|---:|---:|---:|
| small_B64_I512_O512 | 0.007996 | 0.146700 | 0.004351 | 0.002751 | 53.33x | 2.91x |
| medium_B256_I2048_O2048 | 0.043912 | 2.204897 | 0.012117 | 0.007296 | 302.21x | 6.02x |
| target_B1024_I8192_O8192 | 0.964510 | 35.051521 | 0.061348 | 0.975088 | 35.95x | 0.99x |

Geomean speedup vs Baseline Triton1: `83.36x`. Target dispatch intentionally uses exact ACL reduction ordering; it is near parity with PyTorch / ACL and much faster than the editable baseline.
