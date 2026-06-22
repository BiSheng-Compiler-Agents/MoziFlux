# performance_report

## Scope

Kernel: `31_Conv2d_Min_Add_Multiply.py` → `opt_31_Conv2d_Min_Add_Multiply.py`.
The main convolution is ACL-backed in both versions; cannsim traces cover the fused Triton epilogue only.

## Cannsim traces

Full 1024+/4096+ element epilogue probes did not become instr-stable within the plugin safety window, so the reported traces are scalar micro-probes (`BLOCK=1`, grid=1) of the same baseline and optimized address-generation bodies. They are diagnostic for instruction mix, not full-tile hardware latency.

| Provider | Trace path | Probe | wall_cycles | Hardware ns (`cycles*0.4`) | Bottleneck |
|---|---|---:|---:|---:|---|
| Baseline Triton1 epilogue | `/tmp/cannsim_local/l2_31_conv2d_min_add_multiply_baseline_scalar/cannsim_20260625142557_test_kernel/report/trace_core0.json` | 1 element | 2875 | 1150.0 | SCALAR 1827 busy cycles |
| Optimized Triton epilogue | `/tmp/cannsim_local/l2_31_conv2d_min_add_multiply_optimized_scalar/cannsim_20260625142741_test_kernel/report/trace_core0.json` | 1 element | 3746 | 1498.4 | SCALARLDST 2592 busy cycles |

## Pipeline table

| Provider | SCALAR | SCALARLDST | MTE2 | MTE3 | VEC | PUSHQ | RVECEX |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline scalar probe | 1827 | 1720 | 874 | 1058 | 847 | 77 | 31 |
| Optimized scalar probe | 1819 | 2592 | 666 | 702 | 0 | 899 | 153 |

## Interpretation

The scalar probe is intentionally too small to expose the production optimization: halving the exact-shape launch count (`63,504` → `32,768`) and replacing vectorized per-element channel division with one scalar channel/bias selection per 8192-element HW tile. At scalar scale, the optimized tile scheduler overhead dominates; at production tile scale, the direct path processes contiguous HW spans and avoids the baseline per-element `offs // hw` / `% channels` pattern.

## Hardware latency

Remote verification directory: `/home/s00929845/kernel_verify/l2_31_Conv2d_Min_Add_Multiply_1782397854`.

| label | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 | Optimized Triton (ms) | Speedup vs Baseline1 | Ratio vs ACL |
|---|---:|---:|---:|---:|---:|---:|
| small_32 | 0.067529 | 0.211297 | inf (read-only skip) | 0.054967 | 3.84x | 1.23x |
| medium_64 | 0.178058 | 2.152540 | inf (read-only skip) | 0.173181 | 12.43x | 1.03x |
| exact_128 | 7.207358 | 71.593088 | inf (read-only skip) | 4.529304 | 15.81x | 1.59x |

Correctness: all optimized direct-shape tests and the unit-test-only persistent dispatch shape passed with `max_diff=0`; `UNIT_TEST PASS`.
