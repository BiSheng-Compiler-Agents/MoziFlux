# Performance Report: 30_Softsign

## Verification summary

- cannsim baseline sub-kernel: PASS (`/tmp/cannsim_local/softsign_baseline/cannsim_20260624185144_test_kernel/report/trace_core0.json`)
- cannsim optimized sub-kernel: PASS (`/tmp/cannsim_local/softsign_optimized/cannsim_20260624185327_test_kernel/report/trace_core0.json`)
- Remote hardware unit test: `UNIT_TEST PASS`

## Cannsim trace comparison

| Kernel | Elements per tile | Wall cycles | Normalized cycles / 1024 elts | Bottleneck pipe | Hardware time per tile (cycles * 0.4ns) |
|---|---:|---:|---:|---|---:|
| Baseline direct | 1024 | 3202 | 3202.0 | SCALAR (1773 cyc) | 1280.8 ns |
| Optimized direct | 8192 | 3779 | 472.4 | MTE3 (2002 cyc) | 1511.6 ns |

Normalized cannsim throughput improvement: `3202.0 / 472.4 = 6.78x` per 1024 elements.

### Baseline pipeline table

| Pipeline | Ops | Busy cycles | Notes |
|---|---:|---:|---|
| SCALAR | 80 | 1773 | bottleneck |
| MTE3 | 2 | 1427 | store/WAIT_FLAG_VEC |
| SCALARLDST | 1 | 1221 | critical scalar load |
| MTE2 | 3 | 977 | GM -> UB movement |
| VEC | 1 | 971 | WAIT_FLAG_MTE2 |
| RVECEX | 49 | 59 | vector compute |

Top baseline critical instructions: `LDP_XI_XJ_XN` 1432 cyc, `LD_XD_XN_IMM` 1221 cyc, `STI_XN_IMM` 1220 cyc, `WAIT_FLAG_VEC` 1065 cyc, `WAIT_FLAG_MTE2` 971 cyc.

### Optimized pipeline table

| Pipeline | Ops | Busy cycles | Notes |
|---|---:|---:|---|
| MTE3 | 2 | 2002 | bottleneck after larger tile |
| SCALAR | 82 | 1775 | mostly fixed overhead over 8x elements |
| SCALARLDST | 1 | 1219 | fixed launch/setup cost |
| MTE2 | 3 | 1071 | GM -> UB movement |
| VEC | 1 | 1065 | WAIT_FLAG_MTE2 |
| RVECEX | 385 | 283 | vector compute over 8x elements |

Top optimized critical instructions: `RV_VDIV` 2176 cyc over 128 vector lanes, `WAIT_FLAG_VEC` 1383 cyc, `LD_XD_XN_IMM` 1219 cyc, `STI_XN_IMM` 1218 cyc, `WAIT_FLAG_MTE2` 1065 cyc.

## Hardware latency (`remote_verify`)

| Shape | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 (ms) | Optimized Triton (ms) |
|---|---:|---:|---:|---:|
| direct_1M | 0.018948 | 0.021786 | 0.012599 | 0.010449 |
| direct_irregular | 0.019075 | 0.021924 | 0.012678 | 0.010436 |
| persistent_original | 30.967613 | inf (grid overflow skipped) | 8.843431 | 9.186557 |

Optimized speedup vs editable baseline on safe direct shapes: 2.09x (`direct_1M`) and 2.10x (`direct_irregular`). On the original oversized shape, optimized uses the legal persistent path; editable baseline is skipped because it would exceed `coreDim <= 65535`, and optimized is 0.96x vs read-only baseline2.
