# Performance Report

## Scope

- Baseline input: `28_BMM_InstanceNorm_Sum_ResidualAdd_Multiply.py`
- Optimized file: `opt_28_BMM_InstanceNorm_Sum_ResidualAdd_Multiply.py`
- cannsim sub-kernel: post-`F.linear` row normalization + residual multiply, `B=1`, `F=8192`, `grid=1`
- Cycle-to-time conversion: `cycles * 0.4 ns`

## cannsim trace comparison

| Metric | Baseline | Optimized | Delta |
|---|---:|---:|---:|
| wall_cycles | 7,612 | 7,693 | +1.06% |
| estimated tile time | 3.045 us | 3.077 us | +0.032 us |
| x_events | 2,096 | 2,096 | 0 |
| i_events | 38 | 38 | 0 |
| bottleneck | PUSHQ | PUSHQ | unchanged |

## Pipeline utilization

| Pipeline | Baseline busy_cyc | Optimized busy_cyc | Delta |
|---|---:|---:|---:|
| PUSHQ | 3,803 | 3,789 | -0.37% |
| SCALAR | 1,972 | 1,975 | +0.15% |
| SCALARLDST | 1,925 | 1,928 | +0.16% |
| MTE2 | 1,270 | 1,274 | +0.31% |
| MTE3 | 1,310 | 1,312 | +0.15% |
| VEC | 1,125 | 1,218 | +8.27% |
| RVECEX | 1,164 | 1,164 | 0 |
| RVECLD | 1,062 | 1,062 | 0 |
| RVECST | 422 | 422 | 0 |

## Top instruction comparison

| Instruction | Baseline | Optimized | Note |
|---|---:|---:|---|
| RV_VCADD total_cyc | 5,632 | 5,632 | row reduction tree unchanged |
| RV_VMUL count | 515 | 515 | arithmetic body intentionally kept parity with baseline |
| VF/PUSHQ total_cyc | 3,775 | 3,761 | small dispatch reduction |
| MOV_SRC_TO_DST_ALIGNv2 total_cyc | 2,415 | 2,512 | sub-kernel MTE noise/slight regression |

## Hardware latency (`remote_verify`)

| label | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 (ms) | Optimized Triton (ms) | Opt vs Baseline1 |
|---|---:|---:|---:|---:|---:|
| small_128x512x512 | 0.036256 | 0.012789 | 0.011210 | 0.012830 | 0.997x |
| irregular_257x768x640 | 0.048489 | 0.026827 | 0.023185 | 0.026670 | 1.006x |
| target_1024x8192x8192 | 6.507082 | 6.423610 | 6.415543 | 6.423433 | 1.00003x |

Correctness: all optimized benchmark shapes passed; persistent dispatch path `persistent_70000x32x32` passed with max diff `7.152557e-07`. The optimization is primarily a legality/generalization improvement for large batch counts while preserving target latency parity.
