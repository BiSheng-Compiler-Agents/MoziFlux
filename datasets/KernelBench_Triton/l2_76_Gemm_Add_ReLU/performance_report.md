# Performance Report — Gemm_Add_ReLU

## Trace Methodology

Cannsim was run with sub-kernel hosts as required for cycle-level diagnosis:

- Baseline: `BLOCK_M=64`, `BLOCK_N=128`, `BLOCK_K=64`, grid `(1,1,1)`
- Optimized: `BLOCK_M=64`, `BLOCK_N=256`, `BLOCK_K=64`, grid `(1,1,1)`
- Inputs: fp16 all-ones A/B, fp16 bias=1, fp32 output; expected output `K + 1 = 65`
- Trace files:
  - Baseline: `/tmp/cannsim_local/l2_76_baseline/cannsim_20260630060011_test_kernel/report/trace_core0.json`
  - Optimized: `/tmp/cannsim_local/l2_76_optimized/cannsim_20260630060243_test_kernel/report/trace_core0.json`

Because the optimized diagnostic tile computes twice as many output elements, absolute wall cycles are not directly comparable; normalized cycles per output element are the primary comparison.

## Cannsim Summary

| Kernel | Probe Tile | Output Elements | wall_cycles | ns @ 0.4 ns/cyc | cycles/element | Normalized Speedup |
|---|---:|---:|---:|---:|---:|---:|
| Baseline Triton1 | 64×128×64 | 8,192 | 10,464 | 4,185.6 | 1.277 | 1.00× |
| Optimized Triton | 64×256×64 | 16,384 | 11,721 | 4,688.4 | 0.715 | 1.79× |

## Pipeline Utilization: Baseline

| Pipeline | ops | busy_cyc | Notes |
|---|---:|---:|---|
| MTE3 | 6 | 5,931 | Bottleneck |
| PUSHQ | 15 | 4,200 | Queue/instruction dispatch pressure |
| FLOWCTRL | 12 | 3,948 | Dynamic loop/control overhead |
| SCALARLDST | 122 | 3,182 | Scalar load/store pressure |
| MTE2 | 17 | 3,044 | GM/L1/UB loads |
| VEC | 5 | 2,960 | Vector wait/execution |
| RVECST | 1,408 | 2,687 | Vector stores |
| RVECLD | 1,666 | 2,435 | Vector loads |
| CUBE | 4 | 322 | Matrix engine active |

Top critical instructions: `ST_XD_XN_IMM` total 36,155 cycles, `RV_VSTI` total 15,950 cycles, `RV_VLDI` total 15,279 cycles, `WAIT_FLAG_VEC` on MTE3 total 6,428 cycles.

## Pipeline Utilization: Optimized

| Pipeline | ops | busy_cyc | Notes |
|---|---:|---:|---|
| PUSHQ | 14 | 7,184 | Bottleneck |
| MTE3 | 6 | 7,157 | Store/wait pressure, but over 2× elements |
| MTE2 | 16 | 6,044 | GM/L1/UB loads |
| RVECST | 2,564 | 6,010 | Vector stores for larger tile |
| FLOWCTRL | 12 | 5,868 | Control over larger tile |
| RVECLD | 3,332 | 5,206 | Vector loads for larger tile |
| VEC | 6 | 3,888 | Vector wait/execution |
| SCALARLDST | 150 | 3,451 | Scalar load/store pressure |
| CUBE | 4 | 514 | Matrix engine active |

Top critical instructions: `ST_XD_XN_IMM` total 109,132 cycles, `RV_VLDI` total 31,204 cycles, `RV_VSTI` total 30,416 cycles, `VF` total 7,214 cycles.

## Interpretation

The optimized trace increases absolute wall cycles by 12.0% while doing 100% more output work in the diagnostic tile. Normalized by output elements, the optimized kernel reduces cycle cost from 1.277 to 0.715 cycles/element (1.79× improvement). The remaining bottleneck is PUSHQ/MTE3, so further work should focus on instruction dispatch/store pressure rather than increasing Cube math density alone.

## Hardware Latency

`remote_verify` ran `profile_kernels.py` on physical Ascend NPU hardware. Correctness passed for the optimized path on all profiler shapes.

| Shape | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 | Optimized Triton (ms) | Optimized vs Baseline1 |
|---|---:|---:|---:|---:|---:|
| small_acl_fallback | 0.275525 | 0.288268 | inf (sandbox skip) | 0.273165 | 1.06× |
| large_triton_irregular | 0.627758 | 1.818688 | inf (sandbox skip) | 1.323332 | 1.37× |
| target_1024x8192x8192 | 11.457041 | 84.633667 | inf (sandbox skip) | 39.893669 | 2.12× |

Optimized target-shape latency is 39.893669 ms versus 84.633667 ms for the editable baseline Triton provider and 11.457041 ms for PyTorch/ACL. Baseline Triton2 is parser-visible but skipped because the sandbox explicitly forbids reading `base_*.py`.

For sub-kernel cycle conversion, cannsim hardware-equivalent latency is:

| Kernel | cannsim cycles | Hardware-equivalent time |
|---|---:|---:|
| Baseline sub-kernel | 10,464 | 4.1856 µs |
| Optimized sub-kernel | 11,721 | 4.6884 µs |
