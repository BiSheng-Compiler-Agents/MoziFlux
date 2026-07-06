# Performance Report

## Cannsim setup

- Session: `kernelbench-l2_61_ConvTranspose3d_ReLU_GroupNorm`
- Baseline microprobe: original `_relu_groupnorm_kernel`, one `(N,G)` group program, `N=1,C=128,D=2,H=2,W=2,G=8`, `BLOCK_SIZE=128`, `NUM_TILES=1`.
- Optimized microprobe: `_relu_direct_kernel`, one contiguous ReLU tile, `BLOCK_SIZE=4096`.
- Tool: `cannsim_local_run(..., gen_report=True)` with report traces:
  - Baseline: `/tmp/cannsim_local/l2_61_baseline/cannsim_20260630005822_test_kernel/report/trace_core0.json`
  - Optimized: `/tmp/cannsim_local/l2_61_opt/cannsim_20260630010033_test_kernel/report/trace_core0.json`
- Hardware conversion: `cycles * 0.4 ns`.

## Cannsim trace summary

| Provider | Wall cycles | Est. latency (ns) | x_events | i_events | Bottleneck |
|---|---:|---:|---:|---:|---|
| Baseline Triton epilogue | 10,489 | 4,195.6 | 2,122 | 219 | `04_MTE2` (5,834 busy cycles) |
| Optimized Triton fallback | 3,554 | 1,421.6 | 684 | 8 | `07_MTE3` (1,777 busy cycles) |

Speedup for simulated custom work: `10489 / 3554 = 2.95x`.

## Pipeline utilization

### Baseline

| Pipeline | Ops | Busy cycles | Lane sum | Lanes |
|---|---:|---:|---:|---:|
| `04_MTE2` | 83 | 5,834 | 33,772 | 9 |
| `05_VEC` | 34 | 5,792 | 31,662 | 8 |
| `07_MTE3` | 32 | 5,033 | 18,495 | 5 |
| `10_PUSHQ` | 90 | 4,089 | 5,138 | 2 |
| `02_SCALARLDST` | 182 | 3,347 | 12,544 | 7 |
| `01_SCALAR` | 1,021 | 3,338 | 8,028 | 10 |

### Optimized fallback

| Pipeline | Ops | Busy cycles | Lane sum | Lanes |
|---|---:|---:|---:|---:|
| `07_MTE3` | 2 | 1,777 | 1,777 | 1 |
| `01_SCALAR` | 91 | 1,775 | 3,496 | 9 |
| `02_SCALARLDST` | 1 | 1,216 | 1,216 | 1 |
| `04_MTE2` | 3 | 1,018 | 2,013 | 2 |
| `05_VEC` | 1 | 1,012 | 1,012 | 1 |
| `10_PUSHQ` | 2 | 328 | 328 | 1 |

## Top instruction costs

| Provider | Instruction | Pipe | Count | Total cycles | Avg cycles |
|---|---|---|---:|---:|---:|
| Baseline | `WAIT_FLAG_VEC` | MTE2 | 33 | 27,954 | 847 |
| Baseline | `WAIT_FLAG_VEC` | MTE3 | 16 | 16,374 | 1,023 |
| Baseline | `WAIT_FLAG_MTE2` | VEC | 17 | 15,850 | 932 |
| Baseline | `WAIT_FLAG_MTE3` | VEC | 16 | 15,113 | 945 |
| Optimized | `LDP_XI_XJ_XN` | SCALAR | 3 | 1,438 | 479 |
| Optimized | `WAIT_FLAG_VEC` | MTE3 | 1 | 1,325 | 1,325 |
| Optimized | `LD_XD_XN_IMM` | SCALARLDST | 1 | 1,216 | 1,216 |
| Optimized | `WAIT_FLAG_MTE2` | VEC | 1 | 1,012 | 1,012 |

## Remote hardware latency (`remote_verify`)

`remote_verify` passed correctness and benchmark. Latencies are milliseconds.

| label | PyTorch / ACL | Baseline Triton1 | Baseline Triton2 | Optimized Triton |
|---|---:|---:|---:|---:|
| tiny | 0.303248 | 0.518374 | inf | 0.303469 |
| medium | 0.851347 | 0.999444 | inf | 0.863194 |
| target | 62.851131 | inf | inf | 53.102151 |

- Target optimized vs PyTorch/ACL reference: `1.18x` faster (`62.851131 / 53.102151`).
- Tiny optimized vs Baseline Triton1: `1.71x` faster.
- Medium optimized vs Baseline Triton1: `1.16x` faster.
- `Baseline Triton2` was kept parser-visible but skipped because the sandbox forbids reading `base_*.py`.
- Baseline Triton1 target timing was pre-skipped to bound runtime; optimized correctness was still tested on the target shape.
