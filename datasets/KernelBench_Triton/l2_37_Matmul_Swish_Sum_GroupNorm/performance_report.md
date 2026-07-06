# Performance Report

## Cannsim sub-kernel traces

Trace source paths:
- Baseline: `/tmp/cannsim_local/l2_37_baseline/cannsim_20260629184410_test_kernel/report/trace_core0.json`
- Optimized Triton fallback body: `/tmp/cannsim_local/l2_37_opt/cannsim_20260629184555_test_kernel/report/trace_core0.json`

Cannsim hosts used one row tile.  Baseline processes one group (`C=64,G=1`); optimized fallback processes four groups (`C=256,G=4`), so normalized cycles/group are shown separately.

| Kernel body | Groups/program | Wall cycles | Hardware time (cycles × 0.4 ns) | Normalized cycles/group | Main bottleneck |
|---|---:|---:|---:|---:|---|
| Baseline one-group epilogue | 1 | 5,675 | 2.270 µs | 5,675 | PUSHQ 2,189 cycles; SCALAR 1,934; SCALARLDST 1,873 |
| Optimized multi-group fallback | 4 | 3,476 | 1.390 µs | 869 | SCALAR 1,844; SCALARLDST 1,741; MTE3 1,630 |

| Metric | Baseline | Optimized fallback | Change |
|---|---:|---:|---:|
| Wall cycles per program | 5,675 | 3,476 | 1.63× fewer |
| Normalized cycles/group | 5,675 | 869 | 6.53× fewer |
| PUSHQ busy cycles | 2,189 | 374 | 5.85× fewer |
| MTE2 busy cycles | 1,430 | 970 | 1.47× fewer |
| RVECEX busy cycles | 197 | 328 | higher total because 4 groups are processed |

## Remote hardware benchmark

`remote_verify` correctness and benchmark passed at `/home/s00929845/kernel_verify/l2_37_Matmul_Swish_Sum_GroupNorm_1782760139`.

| label | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 (ms) | Optimized Triton (ms) |
|---|---:|---:|---:|---:|
| small_direct | 0.234468 | 0.520867 | 0.525333 | 0.232915 |
| medium_direct | 2.125985 | 8.319883 | 8.521790 | 2.114749 |
| irregular_direct | 0.560840 | 2.841111 | 2.898768 | 0.558606 |
| largeC_direct | 0.502078 | 1.166801 | 1.205967 | 0.515538 |
| persistent_large | 56.856991 | inf (grid guard) | inf (grid guard) | 56.826000 |
| target | 226.366318 | inf (grid guard) | inf (grid guard) | 226.384262 |

Hardware latency at the required target shape is **226.384262 ms** for `Optimized Triton` (ACL production dispatch).  The editable/read-only baseline Triton paths are not legal at target scale because `B*G = 2,097,152 > 65,535`.
