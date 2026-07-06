# Performance Report

## Verification summary

- `cannsim_local_run` completed for both baseline and optimized sub-kernel hosts.
- Remote hardware verification completed: `UNIT_TEST PASS`; optimized direct and forced-persistent dispatch paths passed.
- Reference `base_*.py` was not read because the active sandbox explicitly marks reference files as `DO NOT read`; `profile_kernels.py` keeps the Baseline Triton2 column parser-visible as `inf`/`SKIP_REFERENCE_SANDBOX`.

## Cannsim trace comparison (grid=1, N=1024 zero-fill micro-probe)

Trace paths:
- Baseline: `/tmp/cannsim_local/l2_80_baseline_zero/cannsim_20260630070324_test_kernel/report/trace_core0.json`
- Optimized: `/tmp/cannsim_local/l2_80_opt_zero/cannsim_20260630070500_test_kernel/report/trace_core0.json`

| Metric | Baseline | Optimized | Delta |
|---|---:|---:|---:|
| Wall cycles | 5,524 | 5,503 | -0.38% |
| Hardware time (cycles × 0.4 ns) | 2,209.6 ns | 2,201.2 ns | -8.4 ns |
| Trace events (`ph=X`) | 98 | 98 | 0 |
| SCALAR busy cycles | 1,749 | 1,722 | -1.54% |
| MTE3 busy cycles | 1,080 | 1,079 | -0.09% |
| SCALARLDST busy cycles | 486 | 477 | -1.85% |
| PUSHQ busy cycles | 399 | 402 | +0.75% |
| RVECST busy cycles | 144 | 144 | 0 |
| RVECEX busy cycles | 12 | 12 | 0 |
| MTE2 busy cycles | 5 | 5 | 0 |

## Dominant cannsim instructions

| Rank | Baseline instruction | Cycles | Optimized instruction | Cycles |
|---:|---|---:|---|---:|
| 1 | LDP_XI_XJ_XN | 972 | LDP_XI_XJ_XN | 954 |
| 2 | MOV_SRC_TO_DST_ALIGNv2 | 686 | MOV_SRC_TO_DST_ALIGNv2 | 682 |
| 3 | DC_PRELOAD_XN_IMM | 500 | DC_PRELOAD_XN_IMM | 491 |
| 4 | LD_XD_XN_IMM | 486 | LD_XD_XN_IMM | 477 |
| 5 | VF | 395 | VF | 398 |
| 6 | WAIT_FLAG_VEC | 394 | WAIT_FLAG_VEC | 397 |
| 7 | RV_VSTI | 144 | RV_VSTI | 144 |

Interpretation: the sub-kernel is already a minimal contiguous zero store; cannsim shows a small scalar/setup reduction from explicit block-valued zero + alignment hints. The production-level win is dominated by algebraically skipping the dead GEMM and keeping only a tiny zero-fill launch.

## Remote hardware latency (`profile_kernels.py --test --bench`)

| label | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 (ms) | Optimized Triton (ms) | Speedup vs Baseline Triton1 |
|---|---:|---:|---:|---:|---:|
| small | 0.017975 | 0.010542 | inf | 0.009561 | 1.103x |
| irregular | 0.019502 | 0.010467 | inf | 0.010449 | 1.002x |
| target | 0.019579 | 0.010670 | inf | 0.010731 | 0.994x |

Target optimized hardware latency: **0.010731 ms**.
