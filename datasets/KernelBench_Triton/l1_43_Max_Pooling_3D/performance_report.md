# Performance Report: Max Pooling 3D

## Cannsim setup
- Baseline trace: `/tmp/cannsim_local/l1_43_maxpool3d_baseline2/cannsim_20260624233115_test_kernel/report/trace_core0.json`
- Optimized trace: `/tmp/cannsim_local/l1_43_maxpool3d_opt2/cannsim_20260624233545_test_kernel/report/trace_core0.json`
- Sub-kernel shape: `N=C=1, D=16, H=16, W=128`, `K=3`, `stride=2`, `padding=1`, `dilation=3`, one output-W tile.
- Cycle-to-time conversion: `cycles * 0.4 ns`.

## Trace summary

| Kernel | wall cycles | est. hardware time | x_events | i_events | Bottleneck |
|---|---:|---:|---:|---:|---|
| Baseline | 7,215 | 2.886 us | 1,583 | 118 | MTE2 |
| Optimized | 6,001 | 2.400 us | 1,020 | 44 | MTE2 |
| Delta | -16.8% | -16.8% | -35.6% | -62.7% | MTE2 remains |

## Pipeline comparison

| Pipeline | Baseline busy | Optimized busy | Delta |
|---|---:|---:|---:|
| MTE2 | 3,797 | 4,147 | +9.2% |
| VEC | 3,011 | 2,148 | -28.7% |
| SCALARLDST | 2,426 | 1,788 | -26.3% |
| SCALAR | 1,485 | 855 | -42.4% |
| MTE3 | 1,470 | 1,207 | -17.9% |
| PUSHQ | 1,324 | 1,056 | -20.2% |

## Top instruction comparison

| Kernel | Top instruction | total cycles | Note |
|---|---|---:|---|
| Baseline | `ST_XD_XN_IMM @ SCALARLDST` | 26,834 | Scalar store overhead from 2D/autotuned launch form. |
| Baseline | `WAIT_FLAG_MTE2 @ VEC` | 3,011 | Vector lane waits on GM/UB movement. |
| Optimized | `MOV_SRC_TO_DST_ALIGNv2 @ MTE2` | 24,798 | GM→UB remains dominant. |
| Optimized | `WAIT_FLAG_MTE2 @ VEC` | 2,148 | Reduced wait vs baseline. |


## Remote hardware verification

`remote_verify` passed correctness and benchmark after bounding the oversized full-target case:

| label | PyTorch / ACL (ms) | Baseline Triton1 | Baseline Triton2 | Optimized Triton (ms) |
|---|---:|---:|---:|---:|
| direct_small | 0.007699 | inf | inf | 0.012074 |
| nonpow_medium | 0.013871 | inf | inf | 0.056284 |
| persistent_synthetic | 0.050119 | inf | inf | 2.333691 |

Correctness:
- `optimized direct_small PASS max_err=0`
- `optimized nonpow_medium PASS max_err=0`
- `optimized persistent_synthetic PASS max_err=0`

Baseline Triton columns were preserved but intentionally skipped as `inf` to avoid long compilation/invalid-grid poisoning after earlier verifier timeouts. Full source target `(16,32,128,128,128)` exceeded the 900s verifier window; exact full-target hardware latency is not available.
