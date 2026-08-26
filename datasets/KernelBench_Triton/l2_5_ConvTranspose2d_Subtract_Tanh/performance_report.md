# Performance Report

## Verification Summary

- `remote_verify` correctness: **PASS** for optimized direct, irregular, default, and forced-persistent paths.
- Baseline Triton1 is not a valid timing comparison in this environment: it uses unsupported `tl.tanh`; the default shape also exceeds the Ascend grid cap.
- Baseline Triton2 (`base_*.py`) was not read or executed per sandbox instruction.

## Cannsim Trace Results

`cannsim_local_run` was used for the supplied baseline, a diagnostic baseline probe, and the optimized epilogue.

| Path | Probe | Status | wall cycles | Hardware latency (`cycles * 0.4ns`) | Notes |
|---|---:|---|---:|---:|---|
| Baseline as written | 8192 elems | **compile failed** | n/a | n/a | `AttributeError: triton.language has no attribute 'tanh'` |
| Baseline diagnostic probe (`tl_tanh`, flat c_idx) | 1024 elems | PASS | 16,666 | 6.666 µs | Used to quantify baseline flat-index scalar bottleneck after only the API typo is corrected. |
| Optimized plane-tiled epilogue | 4096 elems | PASS | 4,152 | 1.661 µs | Direct one-plane tile; default production shape uses same body through persistent dispatch. |

Normalized throughput:

| Path | Elements | Cycles/element | ns/element |
|---|---:|---:|---:|
| Baseline diagnostic probe | 1,024 | 16.28 | 6.51 |
| Optimized plane-tiled epilogue | 4,096 | 1.01 | 0.41 |

## Cannsim Pipeline Tables

### Baseline diagnostic probe (`cannsim_baseline_probe_small`)

| Pipeline | Ops | Busy cycles | Window | Comment |
|---|---:|---:|---|---|
| SCALARLDST | 2,051 | 15,698 | `[3899,19660]` | Bottleneck |
| SCALAR | 15,512 | 15,171 | `[3879,20541]` | Per-element div/mod/index overhead |
| PUSHQ | 5 | 1,033 | `[4402,20161]` | Dispatch/vector-function overhead |
| MTE2 | 3 | 998 | `[5645,19702]` | GM load |
| MTE3 | 2 | 845 | `[19690,20536]` | GM store |
| RVECEX | 131 | 135 | `[4908,20142]` | Vector tanh work is not the bottleneck |

Top instructions: `ST_XD_XN_IMM` 35,281 total cycles; `AND`/`SHR` each ~12.3k total cycles; `JUMPC x1027`.

### Optimized plane-tiled epilogue (`cannsim_optimized`)

| Pipeline | Ops | Busy cycles | Window | Comment |
|---|---:|---:|---|---|
| SCALARLDST | 6 | 2,582 | `[3914,6541]` | Remaining setup/scalar load-store |
| SCALAR | 114 | 1,813 | `[3894,8042]` | Greatly reduced vs flat probe |
| MTE3 | 2 | 1,474 | `[6562,8037]` | Output store |
| MTE2 | 3 | 1,023 | `[5657,6681]` | Input/bias load |
| PUSHQ | 3 | 910 | `[6554,7585]` | Vector-function dispatch |
| RVECEX | 542 | 491 | `[7075,7566]` | `tanh`/arithmetic work |

Top instructions: `LD_XD_XN_IMM` 2,188 total cycles; `STI_XN_IMM` 1,227; `RV_VDIV` 1,088; `RV_VEXP` 1,024.

## Remote Hardware Latency

| Label | PyTorch / ACL (ms) | Baseline Triton1 | Baseline Triton2 | Optimized Triton (ms) | Speedup vs PyTorch / ACL |
|---|---:|---:|---:|---:|---:|
| direct_1x64x16x16 | 0.328616 | inf | inf | 0.285827 | 1.150x |
| irregular_2x64x17x19 | 0.370532 | inf | inf | 0.342336 | 1.082x |
| default_32x64x256x256 | 261.502228 | inf | inf | 245.712112 | 1.064x |

Remote optimized max absolute error was `<= 1.19209e-07` across tested shapes.
