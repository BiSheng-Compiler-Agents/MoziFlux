# Performance Report

## Methodology
- Tool: `cannsim_local_run(gen_report=True)` with `trace_core0.json`.
- Probe: post-conv epilogue only, fp32, one spatial tile (`BLOCK_SIZE=2048`), one channel segment (`C=32`).
- Baseline trace: `cannsim_local_run` hit the known unsafe-early-exit wrapper condition, but `instr.bin` and `log_ca/` were produced; `cannsim report -e ... -n 0` recovered `trace_core0.json` successfully.
- Hardware latency conversion: `cycles * 0.4 ns`.

## Trace summary

| Kernel | wall cycles | hardware latency | x_events | i_events | bottleneck |
|---|---:|---:|---:|---:|---|
| Baseline flat epilogue | 28,975 | 11.590 us | 9,874 | 892 | SCALAR (24,843 busy cycles) |
| Optimized channel-segment epilogue | 3,724 | 1.490 us | 1,590 | 11 | MTE3 (2,164 busy cycles) |
| Improvement | **7.78x faster** | **-87.1%** | -83.9% | -98.8% | scalar-bound -> store/wait-bound |

## Pipeline utilization

| Pipeline | Baseline busy cycles | Optimized busy cycles | Change |
|---|---:|---:|---:|
| SCALAR | 24,843 | 613 | -97.5% |
| SCALARLDST | 23,083 | 1,106 | -95.2% |
| MTE2 | 1,009 | 1,023 | +1.4% |
| MTE3 | 0 | 2,164 | store wait now visible |
| PUSHQ | 372 | 1,417 | +1,045 |
| RVECEX | 8 | 1,365 | activation math now visible |
| RVECST | 40 | 288 | +248 |
| RVECLD | 0 | 288 | +288 |

## Top instruction changes

| Instruction / event | Baseline | Optimized | Interpretation |
|---|---:|---:|---|
| `DIV` | 5,322 cycles | eliminated from top list | per-lane channel division removed |
| `REM` | 5,316 cycles | eliminated from top list | per-lane channel modulo removed |
| `SIGNEXT` | 10,656 cycles | eliminated from top list | fewer per-lane scalar index conversions |
| `ST_XD_XN_IMM` | 25,190 cycles | not in top list | scalar spill storm removed |
| `RV_VADDS`/`RV_VMUL*` | not dominant | 7,488 combined cycles | actual activation math becomes visible |
| `WAIT_FLAG_VEC@MTE3` | not dominant | 1,780 single event | optimized kernel is store/wait bound after scalar removal |

## Remote hardware verification

`remote_verify(run_test=True, run_bench=True)` completed successfully.

| Shape | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 (ms) | Optimized Triton (ms) | Optimized vs PyTorch |
|---|---:|---:|---:|---:|---:|
| small | 0.206402 | 0.361933 | inf (sandbox skip) | 0.144910 | 1.42x faster |
| medium | 0.494866 | 2.919591 | inf (sandbox skip) | 0.444751 | 1.11x faster |
| default | 109.769173 | inf (grid-cap pre-skip) | inf (sandbox skip) | 102.081329 | 1.08x faster |

Correctness: optimized passed all shapes (`max_abs <= 4.148483e-05`) and forced persistent dispatch (`max_abs=3.671646e-05`).

## Notes
- The cannsim probe intentionally excludes `nn.Conv3d`, which is an ACL/native primitive and not the custom Triton bottleneck.
- The cannsim values are sub-kernel hardware-cycle estimates for the epilogue; the remote table is the measured end-to-end hardware latency for `Conv3d + post-ops`.
