# `RT_DEV_BINARY_MAGIC_ELF` vs `RT_DEV_BINARY_MAGIC_ELF_AIVEC` — Empirical Comparison

Both magic constants work for Triton kernels. Tested with same kernel, same cannsim config (matmul, BLOCK_M=N=K=128, grid=1x1x1, FP16, Ascend950).

## Results

| Metric | `ELF` (0x43554245) | `ELF_AIVEC` (0x41415246) |
|---|---|---|
| Cycles | 419 | 419 |
| Sim time | 2.71s | 2.74s |
| Trace events | 23,127 | 11,604 |
| Bottleneck | identical | identical |
| RVECLD events | 8,192 | 4,096 |
| RVECST events | 8,192 | 4,096 |
| RVECSU events | 3,872 | 1,936 |

## Conclusion

- **Execution is identical** — same cycles, same instructions, same bottleneck.
- **Trace verbosity differs** — `ELF` emits ~2× more events (extra vector pipeline state annotations).
- **Recommendation**: use `ELF_AIVEC` for Triton kernels. Same actionable data, less noise.

## All 4 Magic Constants

| Constant | Value | Hardware Unit |
|---|---|---|
| `RT_DEV_BINARY_MAGIC_ELF` | `0x43554245` ("CUBE") | AI Core (scalar/vector) |
| `RT_DEV_BINARY_MAGIC_ELF_AIVEC` | `0x41415246` ("AARF") | AI Vector (SIMD) — **Triton kernels** |
| `RT_DEV_BINARY_MAGIC_ELF_AICPU` | `0x41415243` ("AARC") | AI CPU (control core) |
| `RT_DEV_BINARY_MAGIC_ELF_AICUBE` | `0x41494343` ("AICC") | AI Cube (matrix unit) |

Source: `runtime/kernel.h` and `rt_external_kernel.h` in CANN 9.0.0.
