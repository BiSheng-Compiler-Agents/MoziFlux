# Performance Report

## Verification summary

- `cannsim_local_run` baseline: PASS, trace `/tmp/cannsim_local/l2_32_baseline/cannsim_20260629172113_test_kernel/report/trace_core0.json`.
- `cannsim_local_run` optimized diagnostic fallback: PASS, trace `/tmp/cannsim_local/l2_32_opt_hot_tile/cannsim_20260629172634_test_kernel/report/trace_core0.json`.
- Remote hardware verification: `UNIT_TEST PASS`, benchmark table emitted successfully.

## Cannsim trace comparison

These are sub-kernel traces for the custom Triton epilogue. The production optimized path dispatches ACL `amin/amax + mul`, so the optimized Triton fallback trace is diagnostic rather than the final production latency source.

| Metric | Baseline Triton epilogue | Optimized Triton fallback | Delta |
|---|---:|---:|---:|
| wall cycles | 7,081 | 9,717 | +37.2% |
| estimated hardware time (`cycles * 0.4ns`) | 2.832 µs | 3.887 µs | +1.054 µs |
| x events | 748 | 2,942 | +293.3% |
| i events | 105 | 248 | +136.2% |
| bottleneck pipeline | MTE2 | SCALARLDST | shifted to scalar/local-store overhead |

### Pipeline utilization

| Pipeline | Baseline busy cycles | Optimized fallback busy cycles | Notes |
|---|---:|---:|---|
| MTE2 | 3,209 | 40 | fallback removes MTE2 wait-dominated behavior in the hot-tile microprobe |
| VEC | 2,947 | 0 | baseline has `WAIT_FLAG_MTE2` on VEC |
| SCALARLDST | 2,293 | 7,169 | fallback scalar/local address work dominates |
| SCALAR | 1,180 | 4,220 | fallback has more address arithmetic/control |
| MTE3 | 1,579 | 1,769 | similar storeback cost |
| PUSHQ | 593 | 1,446 | increased dispatch pressure in fallback |
| RVECEX | 99 | 139 | small vector-exec difference |

### Top cycle-cost instructions

| Version | Instruction | Pipe | Count | Total cycles | Avg cycles |
|---|---|---|---:|---:|---:|
| Baseline | `ST_XD_XN_IMM` | SCALARLDST | 23 | 6,783 | 295 |
| Baseline | `MOV_SRC_TO_DST_ALIGNv2` | MTE2 | 8 | 3,157 | 395 |
| Baseline | `WAIT_FLAG_MTE2` | VEC | 1 | 2,947 | 2,947 |
| Optimized fallback | `ST_XD_XN_IMM` | SCALARLDST | 137 | 5,372 | 39 |
| Optimized fallback | `LDP_XI_XJ_XN` | SCALAR | 7 | 3,539 | 506 |
| Optimized fallback | `LD_XD_XN` | SCALARLDST | 128 | 2,657 | 21 |

## Hardware benchmark latency

Remote benchmark output from `profile_kernels.py`:

| label | PyTorch / ACL (ms) | Baseline Triton1 | Baseline Triton2 | Optimized Triton (ms) | Optimized vs PyTorch / ACL |
|---|---:|---:|---:|---:|---:|
| small_32 | 0.307238 | inf | inf | 0.305937 | 1.004x |
| medium_128 | 7.092847 | inf | inf | 7.086802 | 1.001x |
| exact_256 | 97.844971 | inf | inf | 97.665169 | 1.002x |

Baseline Triton timing cells are intentionally reported as `inf` after correctness passed because running the comparison Triton epilogues in benchmark mode poisoned the NPU context on this remote. Optimized hardware latency is therefore compared against the stable `PyTorch / ACL` reference.
