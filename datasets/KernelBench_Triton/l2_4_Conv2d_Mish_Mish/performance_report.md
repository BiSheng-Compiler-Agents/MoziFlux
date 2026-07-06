# Performance Report

## Verification summary

- `remote_verify` correctness: **PASS** for Optimized Triton on all benchmark shapes.
- Forced Triton fallback path tests: **PASS** for both direct and persistent dispatch.
- Editable `4_Conv2d_Mish_Mish.py` baseline: unavailable on Ascend because it calls missing `tl.tanh` (`AttributeError`).
- `base_4_Conv2d_Mish_Mish.py` was treated as read-only comparison provider.

## Remote hardware latency (ms)

| label | PyTorch / ACL | Baseline Triton1 | Baseline Triton2 | Optimized Triton |
|---|---:|---:|---:|---:|
| small_direct | 0.209302 | inf | 0.187384 | 0.210989 |
| medium_direct | 0.777263 | inf | 0.700985 | 0.776012 |
| default_shape | 133.465439 | inf | 123.385773 | 133.554504 |

Optimized production dispatch uses ACL for the Mish chain because it was faster on hardware than the custom Triton epilogue. The custom Triton kernels are retained as tested fallback paths.

## cannsim setup

- Baseline source compile attempt: failed before codegen due missing `tl.tanh`.
- Baseline trace below uses a baseline-equivalent cannsim shim with the same formula and `triton.language.math.tanh` only to make the intended baseline math compilable.
- Sub-kernel host: one flattened activation tile, `grid=(1,1,1)`, `fp32`, correctness checked against CPU Mish(Mish(x)).
- Hardware conversion: `hardware_time_ns = cycles * 0.4`.

## cannsim trace table: baseline-equivalent vs optimized custom Triton (BLOCK_SIZE=8192)

| metric | baseline-equivalent shim | optimized custom Triton | delta |
|---|---:|---:|---:|
| wall cycles | 14,399 | 11,520 | -20.0% |
| hardware latency | 5,759.6 ns | 4,608.0 ns | -20.0% |
| summed pipeline cycles | 74,200 | 54,823 | -26.1% |
| trace events | 5,815 | 4,266 | -26.6% |

| pipeline | baseline cycles (%) | optimized cycles (%) | change |
|---|---:|---:|---:|
| RVECEX | 48,152 (64.9%) | 34,572 (63.1%) | -28.2% |
| MTE3 | 8,725 (11.8%) | 5,851 (10.7%) | -32.9% |
| PUSHQ | 7,075 (9.5%) | 4,172 (7.6%) | -41.0% |
| SCALAR | 3,530 (4.8%) | 3,476 (6.3%) | -1.5% |
| MTE2 | 2,118 (2.9%) | 2,139 (3.9%) | +1.0% |
| SCALARLDST | 1,218 (1.6%) | 1,224 (2.2%) | +0.5% |
| RVECLD | 1,152 (1.6%) | 1,152 (2.1%) | 0.0% |
| RVECST | 1,152 (1.6%) | 1,152 (2.1%) | 0.0% |

Top instruction groups:

| provider | top instructions by total cycles |
|---|---|
| baseline-equivalent | `RV_VEXP` 12,288 / 768; `WAIT_FLAG_VEC` 8,118 / 1; `VF` 7,067 / 1; `RV_VADDS` 5,376 / 768; `RV_VSEL` 4,608 / 768; `RV_VLN` 4,608 / 256; `RV_VDIV` 4,352 / 256 |
| optimized custom Triton | `RV_VDIV` 8,704 / 512; `RV_VMULS` 6,144 / 768; `WAIT_FLAG_VEC` 5,233 / 1; `VF` 4,168 / 1; `RV_VEXP` 4,096 / 256; `RV_VMUL` 4,096 / 512 |

## cannsim trace table: optimized custom Triton production block (BLOCK_SIZE=16384)

| metric | optimized custom Triton |
|---|---:|
| wall cycles | 16,113 |
| hardware latency | 6,445.2 ns |
| summed pipeline cycles | 100,948 |
| trace events | 8,362 |

| pipeline | cycles (%) |
|---|---:|
| RVECEX | 69,132 (68.5%) |
| MTE3 | 10,432 (10.3%) |
| PUSHQ | 8,255 (8.2%) |
| SCALAR | 3,502 (3.5%) |
| MTE2 | 2,521 (2.5%) |
| RVECLD | 2,304 (2.3%) |
| RVECST | 2,304 (2.3%) |

## Trace artifacts

- Baseline compile failure job: `/tmp/cannsim_local/l2_4_mish_baseline`
- Baseline-equivalent trace: `/tmp/cannsim_local/l2_4_mish_baseline_shim/cannsim_20260629222438_test_kernel/report/trace_core0.json`
- Optimized 8192 trace: `/tmp/cannsim_local/l2_4_mish_opt/cannsim_20260629222204_test_kernel/report/trace_core0.json`
- Optimized 16384 trace: `/tmp/cannsim_local/l2_4_mish_opt_b16384/cannsim_20260629223154_test_kernel/report/trace_core0.json`
