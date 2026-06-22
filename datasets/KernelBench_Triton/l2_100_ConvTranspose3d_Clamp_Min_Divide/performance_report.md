# Performance Report

## Shapes

- Target input: `(16, 64, 24, 48, 48)` fp32.
- ConvTranspose3d output: `(16, 128, 47, 95, 95)`.
- Output elements: `868,710,400`; fp32 output bytes: ~3.47 GB.
- Baseline epilogue tiles: `ceil(868,710,400 / 8192) = 106,044` > Ascend `coreDim` limit `65,535`.

## cannsim sub-kernel traces

Both traces use the same BLOCK_SIZE=8192 one-tile clamp/divide epilogue probe. This preserves per-tile vector/MTE behavior; the full-shape benefit is dispatch legality and is therefore not visible at grid=1.

| Path | Kernel | Probe | wall cycles | HW latency (cycles × 0.4 ns) | Bottleneck |
|---|---|---:|---:|---:|---|
| Baseline Triton1 | direct clamp/divide | 8192 fp32 elems, grid=1 | 4,108 | 1.643 us | MTE3 store/wait (`busy_cyc=2311`) |
| Optimized Triton | persistent clamp/divide | 8192 fp32 elems, grid=1, one loop iter | 4,145 | 1.658 us | MTE3 store/wait (`busy_cyc=2320`) |

### Pipeline table

| Path | MTE3 busy | SCALAR busy | SCALARLDST busy | MTE2 busy | VEC busy | RVECEX busy | PUSHQ busy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline Triton1 | 2,311 | 1,798 | 1,740 | 1,065 | 1,059 | 631 | 676 |
| Optimized Triton | 2,320 | 1,497 | 1,773 | 1,063 | 1,055 | 631 | 676 |

## Interpretation

The per-tile epilogue remains memory-store bound; optimized grid=1 is intentionally near-identical because it performs the same clamp/divide work plus a persistent loop guard. The actual target optimization is that Baseline Triton1 would launch 106,044 programs and violate Ascend's launch cap, while Optimized Triton caps the launch at 65,535 and loops over remaining tiles in-kernel.

## Hardware latency (`remote_verify`)

Correctness passed on all optimized shapes, including the target path: `UNIT_TEST PASS`, `TEST optimized target_B16_D24 PASS max_diff=0.000000e+00`.

| label | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 (ms) | Optimized Triton (ms) | PyTorch/Optimized |
|---|---:|---:|---:|---:|---:|
| small_B1_D4 | 0.012527 | 0.010799 | inf | 0.011085 | 1.130x |
| medium_B2_D8 | 0.036932 | 0.033107 | inf | 0.033013 | 1.119x |
| large_B4_D12 | 0.138519 | 0.131433 | inf | 0.131202 | 1.056x |
| target_B16_D24 | 20.476652 | inf (grid_guard) | inf | 16.333014 | 1.254x |

Target result: optimized hardware latency is `16.333014 ms`; PyTorch/ACL reference is `20.476652 ms`; Baseline Triton1 is not launch-legal at the target shape because its epilogue grid requires 106,044 programs.
