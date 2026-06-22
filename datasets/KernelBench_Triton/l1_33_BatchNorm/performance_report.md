# Performance Report

## Correctness / hardware verification

Remote Ascend verification: `UNIT_TEST PASS`.

| Shape / path | Baseline Triton1 | Baseline Triton2 | Optimized Triton | Max optimized error |
|---|---:|---:|---:|---:|
| `small_8x64x32x32` train/direct | PASS | PASS | PASS | `9.53674e-07` |
| `medium_16x64x128x128` train/direct | SKIP (`coreDim`) | SKIP (`coreDim`) | PASS | `8.34465e-07` |
| `target_64x64x512x512` train/persistent | SKIP (`coreDim`) | SKIP (`coreDim`) | PASS | `1.43051e-06` |
| `eval_dispatch_4x64x17x19` eval/direct | PASS | PASS | PASS | `0` |

## Hardware latency (`profile_kernels.py`, ms)

| label | PyTorch / ACL | Baseline Triton1 | Baseline Triton2 | Optimized Triton |
|---|---:|---:|---:|---:|
| `small_8x64x32x32` | 0.009189 | 0.731568 | 0.275070 | 0.038568 |
| `medium_16x64x128x128` | 0.162908 | inf (`coreDim`) | inf (`coreDim`) | 0.474645 |
| `target_64x64x512x512` | 8.659007 | inf (`coreDim`) | inf (`coreDim`) | 15.909076 |

## cannsim sub-kernel traces

Sub-kernel simulations use one program on core0. Baseline reduces one `W=512` row; optimized reduces one `H*W` block of 2048 contiguous elements. Hardware time uses `cycles * 0.4 ns`.

| Kernel | Work per program | wall_cycles | ns/program | cycles/element | Dominant pipeline |
|---|---:|---:|---:|---:|---|
| Baseline `_bn_row_reduce_nhw_store` | 512 elems | 3373 | 1349.2 | 6.59 | SCALAR (1906 busy cycles) |
| Optimized `_bn_reduce_hw_block` | 2048 elems | 3487 | 1394.8 | 1.70 | SCALAR (1820 busy cycles) |

### Pipeline utilization

| Kernel | SCALAR | SCALARLDST | MTE2 | VEC | MTE3 | RVECEX | RVECLD | PUSHQ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline busy cycles | 1906 | 1781 | 1012 | 569 | 390 | 57 | 18 | 104 |
| Optimized busy cycles | 1820 | 1738 | 995 | 951 | 463 | 182 | 134 | 228 |

### Top critical instructions

| Kernel | Instruction | Pipe | Count | Total cycles | Avg cycles |
|---|---|---:|---:|---:|---:|
| Baseline | `LD_XD_XN_IMM` | SCALARLDST | 7 | 2759 | 394 |
| Baseline | `LDP_XI_XJ_XN` | SCALAR | 5 | 2504 | 501 |
| Baseline | `MOV_SRC_TO_DST_ALIGNv2` | MTE2 | 2 | 1544 | 772 |
| Optimized | `LD_XD_XN_IMM` | SCALARLDST | 5 | 2205 | 441 |
| Optimized | `RV_VCADD` | RVECEX | 64 | 1408 | 22 |
| Optimized | `MOV_SRC_TO_DST_ALIGNv2` | MTE2 | 1 | 982 | 982 |

## Interpretation

The optimized reduction does 4x more useful work per program at similar wall cycles, improving sub-kernel reduction efficiency from `6.59` to `1.70` cycles/element (~3.87x). End-to-end target latency is `15.909076 ms`; the input/golden Triton paths cannot be timed at medium/target because they exceed Ascend `coreDim <= 65535` and abort.
