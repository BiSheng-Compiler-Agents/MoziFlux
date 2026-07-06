# Performance Report

## Verification summary

- `cannsim_local_run` baseline trace: `/tmp/cannsim_local/l2_60_baseline/cannsim_20260630004407_test_kernel/report/trace_core0.json`
- `cannsim_local_run` optimized fallback trace: `/tmp/cannsim_local/l2_60_optimized_cmp/cannsim_20260630004843_test_kernel/report/trace_core0.json`
- Remote hardware verification: `UNIT_TEST PASS`, benchmark pass.
- Hardware time conversion for cannsim: `cycles * 0.4 ns`.

## Cannsim trace comparison (one 8x8 Swish reduction micro-tile)

| Kernel trace | Events | Wall cycles | HW latency (ns) | Dominant pipes by busy cycles | Top instructions/stalls |
|---|---:|---:|---:|---|---|
| Baseline `_swish_reduce_3d` | 382 | 3,335 | 1,334.0 | SCALAR 45.9%, SCALARLDST 24.4%, VEC 9.9%, RVECEX 7.3%, MTE2 5.2% | `LD_XD_XN_IMM` 3287, `LDP_XI_XJ_XN` 2986, `STI_XN_IMM` 1273, `WAIT_FLAG_MTE2` 689 |
| Optimized Triton fallback `_swish_group_reduce_parts_3d` | 2,057 | 6,853 | 2,741.2 | SCALAR 55.7%, SCALARLDST 28.7%, PUSHQ 8.1%, MTE3 3.8%, RVECEX 2.3% | `LDP_XI_XJ_XN` 3440, `ST_XD_XN_IMM` 2604, `LD_XD_XN_IMM` 2029, `SIGNEXT` 1864 |

Interpretation: the generic grouped fallback removes atomics in the full algorithm but is scalar-index heavy in a one-channel microprobe. The production optimization therefore dispatches the post chain to ACL primitives; the fallback remains only as a correctness-covered legal Triton path.

## Remote hardware latency

| Label | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 | Optimized Triton (ms) | Speedup vs Baseline Triton1 |
|---|---:|---:|---:|---:|---:|
| tiny_triton_path | 0.371594 | 0.592973 | inf (read-prohibited) | 0.372314 | 1.59x |
| medium_acl_path | 1.992639 | 3.963478 | inf (read-prohibited) | 2.004239 | 1.98x |
| default_required | 166.034973 | inf (preskipped) | inf (read-prohibited) | 165.882019 | n/a |

Optimized correctness:

```text
TEST Optimized Triton tiny_triton_path: PASS max_abs=0
TEST Optimized Triton medium_acl_path: PASS max_abs=0
TEST Optimized Triton default_required: PASS max_abs=0
TEST Optimized Triton forced_triton_direct: PASS max_abs=1.43051e-06
TEST Optimized Triton forced_triton_persistent: PASS max_abs=1.43051e-06
UNIT_TEST PASS
```
