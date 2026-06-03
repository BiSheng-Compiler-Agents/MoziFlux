# Cannsim Sub-Kernel Timeout Mitigation

## Issue

Even with a proper sub-kernel host (grid=1, M=BLOCK_M, K=2*BLOCK_K), the
full 32-core Ascend950PR camodel must boot all 32 AI cores. On machines with
<32 GB RAM or <16 CPU cores, this creates two problems:

1. **Camodel boot time**: ~12s for 32-core initialization (248 cycles)
2. **Kernel simulation timeout**: The kernel launch may never complete within
   a 600s timeout because the simulated NPU runs at ~0.02-0.15 KHz on
   underpowered machines

## Diagnostic

After a cannsim record run that appears to finish (showing "all tasks are
finished"), search cannsim.log for these markers:

```
[HOST] Kernel completed        ← YES: kernel finished
[HOST] PASS                    ← YES: correctness verified
```

If you see `[HOST] Launching kernel...` followed by DRVSTUB_LOG lines but
NO `[HOST] Kernel completed`, the kernel did NOT finish execution. The
simulation was insufficient.

## Mitigations

1. **Reduce sub-kernel further**: K=1*BLOCK_K (single iteration, not 2)
   instead of K=2*BLOCK_K. This halves the simulated instruction count.

2. **Use smaller blocks**: BLOCK_M=BLOCK_N=64 instead of 128. This reduces
   per-iteration work by 4x.

## Verified On

- l1_11 4D Tensor Matrix Multiplication (June 2026)
- Sub-kernel: grid=(1,1), M=128, N=128, K=64 (2*BLOCK_K=64)
- Machine: >600s without trace, despite kernel compilation and binary registration working correctly
