# Performance Report

## Environment

- cannsim target: Ascend950
- cannsim sub-kernel: `M=128`, `N=256`, `K=32`, grid `(1,1,1)`
- Hardware verification: `remote_verify`, real Ascend NPU
- Correctness: `UNIT_TEST PASS` for small, non-power-of-two, and K=64 cases

## Cannsim Trace Summary

| Kernel | Trace path | wall cycles | hardware time (ns, cycles × 0.4) | x_events | i_events | bottleneck |
|---|---:|---:|---:|---:|---:|---|
| Baseline Triton | `/tmp/cannsim_local/cannsim_baseline_1782155585/.../trace_core0.json` | 18100 | 7240.0 | 4309 | 159 | FLOWCTRL |
| Optimized Triton | `/tmp/cannsim_local/cannsim_opt_simple_1782155777/.../trace_core0.json` | 18587 | 7434.8 | 4444 | 164 | FLOWCTRL |

Sub-kernel cannsim shows the optimized 128×256 tile is roughly neutral/slightly slower at single-tile scale; the hardware win comes from fixing the full-shape launch path, where the baseline benchmark config hits the Ascend `coreDim` limit.

### Pipeline Utilization

| Kernel | FLOWCTRL | MTE3 | PUSHQ | RVECST | RVECLD | FIXP | CUBE | SCALARLDST |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 9405 | 9196 | 8072 | 7943 | 6313 | 5987 | 4384 | 2750 |
| Optimized | 9432 | 9236 | 8086 | 7943 | 6313 | 5640 | 4382 | 3327 |

### Top Instructions

| Kernel | Instruction | Pipe | Count | Total cycles | Avg cycles |
|---|---|---|---:|---:|---:|
| Baseline | ST_XD_XN_IMM | SCALARLDST | 61 | 27869 | 457 |
| Baseline | WAIT_FLAG_VEC | MTE3 | 2 | 10537 | 5268 |
| Baseline | SET_INTRA_BLOCKI | FLOWCTRL | 5 | 9614 | 1923 |
| Baseline | VF | PUSHQ | 2 | 8064 | 4032 |
| Optimized | ST_XD_XN_IMM | SCALARLDST | 76 | 48452 | 638 |
| Optimized | WAIT_FLAG_VEC | MTE3 | 2 | 10624 | 5312 |
| Optimized | SET_INTRA_BLOCKI | FLOWCTRL | 5 | 9625 | 1925 |
| Optimized | VF | PUSHQ | 2 | 8078 | 4039 |

## Hardware Latency

| label | PyTorch / ACL (ms) | Baseline Triton (ms) | Optimized Triton (ms) | Optimized vs Baseline |
|---|---:|---:|---:|---:|
| small | 0.005127 | 0.032006 | 0.028369 | 1.13× faster |
| nonpow2 | 0.007040 | 0.030309 | 0.025906 | 1.17× faster |
| medium | 0.027049 | 0.026731 | 0.026246 | 1.02× faster |
| benchmark | 5.838677 | inf (`coreDim=65536`) | 5.667795 | baseline fails; opt runs |

## Notes

- The required benchmark shape is `M=N=32768, K=64`.
- Baseline hardware benchmark fails with `KernelLaunch failed because value 65536 for parameter coreDim is invalid`.
- Optimized Triton completes the benchmark path at `5.667795 ms` and passes all unit tests.
