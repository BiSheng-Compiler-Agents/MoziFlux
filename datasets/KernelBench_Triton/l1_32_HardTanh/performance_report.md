# Performance Report: HardTanh

## Simulation setup

- Tool: `cannsim_local_run(gen_report=True)` on `Ascend950`.
- Baseline sub-kernel: direct HardTanh, `BLOCK_SIZE=4096`, `N=4096`, grid=1.
- Optimized sub-kernel: persistent HardTanh body, `BLOCK_SIZE=8192`, `N=8192`, `n_programs=1`, grid=1.
- Cycle conversion: `hardware_time_ns = cycles * 0.4`.

## cannsim trace comparison

| Kernel | Elements / simulated program | wall_cycles | ns / program | cycles / element | ns / element | Bottleneck |
|---|---:|---:|---:|---:|---:|---|
| Baseline direct | 4096 | 3412 | 1364.8 | 0.8330 | 0.3332 | SCALAR (1809 busy cycles) |
| Optimized persistent | 8192 | 4126 | 1650.4 | 0.5037 | 0.2015 | MTE3 (1976 busy cycles) |

Normalized throughput improved by `1.65x` (`0.8330 / 0.5037` cycles/element), a `39.5%` cycle-per-element reduction. Raw per-program cycles are higher because the optimized simulated program processes twice as many elements.

## Pipeline utilization from trace summaries

| Pipeline | Baseline busy cycles | Optimized busy cycles | Note |
|---|---:|---:|---|
| SCALAR | 1809 | 1815 | Fixed launch/setup overhead; amortized across 2x elements in optimized tile. |
| SCALARLDST | 1706 | 1746 | Mostly fixed scalar load/store setup. |
| MTE2 | 1014 | 1075 | GM -> UB movement scales mildly with larger tile. |
| VEC | 1008 | 1071 | Clamp vector work scales with elements. |
| MTE3 | 1618 | 1976 | UB -> GM store dominates optimized trace. |
| RVECEX | 141 | 270 | Vector compare/select lane work doubles with elements. |
| RVECLD | 122 | 250 | Local vector load lane work doubles with elements. |
| RVECST | 134 | 262 | Local vector store lane work doubles with elements. |

## Hardware benchmark (`remote_verify`)

`profile_kernels.py` correctness: `UNIT_TEST PASS`; optimized direct and persistent dispatch paths both passed.

| label | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 (ms) | Optimized Triton (ms) | Notes |
|---|---:|---:|---:|---:|---|
| direct_1M | 0.009643 | 0.010547 | 0.011315 | 0.011343 | Direct path correctness/perf check. |
| direct_irregular | 0.009731 | 0.010399 | 0.011017 | 0.011249 | Non-power-ish shape coverage. |
| original_persistent | 8.919091 | inf | inf | 8.982212 | Baseline direct grid would exceed `65535`; optimized persistent path is legal and correct. |

For the original shape, optimized Triton runs at `8.982212 ms` versus PyTorch / ACL at `8.919091 ms` (`torch/opt = 0.993x`). Baseline Triton1 cannot be launched for the original shape because `ceil(1,610,612,736 / 4096) = 393216 > 65535`.

## Trace artifacts

- Baseline trace: `/tmp/cannsim_local/hardtanh_baseline_4096/cannsim_20260624191459_test_kernel/report/trace_core0.json`
- Optimized persistent trace: `/tmp/cannsim_local/hardtanh_opt_persistent_8192/cannsim_20260624192306_test_kernel/report/trace_core0.json`
