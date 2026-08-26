# Performance Report

## Cannsim setup

- Baseline probe: one `MaxPool3d(kernel=6,stride=6)` channel tile, `H2=1`, `W2=16`.
- Optimized Triton fallback probe: one fused 8-channel pool+sum tile, `C_BLOCK=8`, `H2=1`, `W2=16`.
- Both were run through `cannsim_local_run`; the wrapper reported `UNSAFE EARLY EXIT`, but `instr.bin` was present and `cannsim report -n 0` successfully generated `trace_core0.json` for both runs.
- Cannsim hardware time conversion: `cycles * 0.4 ns`.

## Cannsim trace comparison

| Kernel | Work represented | wall_cycles | cannsim hardware latency | x_events | dominant lane-sum bottleneck |
|---|---:|---:|---:|---:|---|
| Baseline pool | 1 channel × 16 outputs | 29,597 | 11.839 us | 2,397 | MTE2 / VEC wait |
| Optimized Triton fallback | 8 channels × 16 outputs | 28,219 | 11.288 us | 6,033 | MTE2 data movement |

## Normalized fallback throughput

| Metric | Baseline per channel-output | Optimized fallback per channel-output | Improvement |
|---|---:|---:|---:|
| wall cycles | 1,849.8 | 220.5 | 8.39× |
| MTE2 lane-sum cycles | 11,956.1 | 1,513.3 | 7.90× |
| VEC lane-sum cycles | 10,167.9 | 361.6 | 28.12× |
| PUSHQ lane-sum cycles | 305.4 | 38.8 | 7.88× |

## Pipeline details

### Baseline recovered trace

| Pipeline | events | lane-sum cycles |
|---|---:|---:|
| MTE2 | 194 | 191,298 |
| VEC | 64 | 162,686 |
| RVECEX | 1,155 | 7,058 |
| PUSHQ | 136 | 4,887 |
| SCALAR | 654 | 3,583 |
| RVECLD | 128 | 1,216 |
| RVECST | 65 | 585 |

Top instructions: `WAIT_FLAG_VEC` (166,532 cycles), `WAIT_FLAG_MTE2` (162,686), `MOV_SRC_TO_DST_ALIGNv2` (24,441), `VF` (4,603).

### Optimized fallback recovered trace

| Pipeline | events | lane-sum cycles |
|---|---:|---:|
| MTE2 | 364 | 193,704 |
| VEC | 36 | 46,284 |
| RVECEX | 3,927 | 24,138 |
| SCALARLDST | 9 | 6,746 |
| RVECLD | 576 | 5,364 |
| PUSHQ | 76 | 4,963 |
| SCALAR | 677 | 3,669 |
| RVECST | 296 | 2,682 |

Top instructions: `MOV_SRC_TO_DST_ALIGNv2` (143,130 cycles), `WAIT_FLAG_VEC` (50,389), `WAIT_FLAG_MTE2` (46,284), `RV_VLDI` (5,364).

## Remote hardware verification

`remote_verify` passed correctness and benchmark. Production optimized dispatch uses ACL for the standard post-op chain and keeps the Triton direct/persistent kernels as tested fallbacks.

| label | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 (ms) | Optimized Triton (ms) |
|---|---:|---:|---:|---:|
| tiny_direct | 0.915956 | 4.066240 | inf | 0.908945 |
| irregular_direct | 2.711759 | 12.766814 | inf | 2.709504 |
| default_direct | 113.671989 | inf (pre-skipped) | inf | 113.599976 |

Correctness:

- Optimized production path: PASS on tiny, irregular, and default shapes (`max_abs=0`).
- Optimized Triton fallback direct path: PASS (`max_abs=9.53674e-07`).
- Optimized Triton fallback persistent path: PASS (`max_abs=1.90735e-06`).
