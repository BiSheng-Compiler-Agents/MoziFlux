# Performance Report

## Cannsim trace / simulation

`cannsim_local_run` was invoked on the baseline Triton kernel. The realistic `Cin=32,K=5,BLOCK_COUT=64,BLOCK_T=128` sub-kernel failed BiSheng compilation (`clang frontend exit 139`); the reduced `Cin=32,K=5,BLOCK_COUT=16,BLOCK_T=16` probe also exceeded VF stack (`75136 > 6144`). A final `Cin=1,K=1,BLOCK_COUT=16,BLOCK_T=16` micro-probe compiled and cannsim executed the device task, but the simulator wrapper exited with a teardown segfault before `trace_core0.json` was emitted.

| Path | Probe | cannsim result | Cycles / latency | Trace availability |
|---|---:|---|---:|---|
| Baseline Triton | `Cin=1,K=1,BLOCK_COUT=16,BLOCK_T=16,grid=1` | device task finished; wrapper segfault | 248 cycles ≈ 99.2 ns | unavailable (`instr.bin`/`trace_core0.json` not emitted) |
| Optimized | ACL dispatch, no custom Triton launch | custom Triton kernel removed | 0 cycles | not applicable |

## Hardware latency (`remote_verify`)

Correctness: `UNIT_TEST PASS`; optimized max diff was 0 on all tested shapes. Baseline Triton providers were pre-skipped as `inf` to avoid launching the scalar custom convolution path; optimized latency is compared to the PyTorch/ACL reference.

| Shape | PyTorch / ACL (ms) | Optimized Triton (ms) | ACL-dispatch speedup |
|---|---:|---:|---:|
| small_L256 | 0.169719 | 0.008266 | 20.532x |
| medium_L4096 | 0.162389 | 0.038915 | 4.173x |
| irregular_L8193 | 0.187609 | 0.137660 | 1.363x |
| exact_L131072 | 28.536316 | 27.095871 | 1.053x |

Geomean speedup vs PyTorch / ACL reference: `3.330x`. Exact target latency: `27.095871 ms` optimized vs `28.536316 ms` PyTorch / ACL.
