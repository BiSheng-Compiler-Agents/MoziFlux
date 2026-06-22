# Performance Report

## Scope

- Operator: Conv2d + InstanceNorm2d(no affine/running stats) + divide
- Baseline file: `17_Conv2d_InstanceNorm_Divide.py`
- Optimized file: `opt_17_Conv2d_InstanceNorm_Divide.py`
- cannsim probe: one contiguous fp32 `(N,C)` plane, `HW=1024`, `grid=(1,)`; target Conv2d remains ACL/PyTorch in both versions.

## cannsim trace comparison

| Metric | Baseline direct | Optimized direct | Delta |
|---|---:|---:|---:|
| wall_cycles | 7,241 | 7,209 | -32 (-0.44%) |
| hardware time (`cycles * 0.4 ns`) | 2.896 us | 2.884 us | -0.012 us |
| x_events | 481 | 481 | 0 |
| i_events | 43 | 43 | 0 |
| bottleneck | PUSHQ | PUSHQ | unchanged |

## Pipeline utilization

| Pipeline | Baseline busy_cyc | Optimized busy_cyc | Delta |
|---|---:|---:|---:|
| PUSHQ | 3,821 | 3,787 | -34 |
| SCALARLDST | 2,027 | 2,031 | +4 |
| SCALAR | 1,985 | 1,989 | +4 |
| MTE2 | 977 | 970 | -7 |
| VEC | 940 | 933 | -7 |
| MTE3 | 696 | 697 | +1 |
| RVECEX | 264 | 264 | 0 |
| RVECLD | 146 | 146 | 0 |
| RVECST | 111 | 111 | 0 |
| FLOWCTRL | 7 | 7 | 0 |

## Top instruction comparison

| Instruction | Baseline | Optimized | Notes |
|---|---:|---:|---|
| `VF` total cycles | 3,785 | 3,751 | dominant PUSHQ cost, slightly lower |
| `LD_XD_XN_IMM` total cycles | 2,737 | 2,748 | essentially unchanged |
| `LDP_XI_XJ_XN` total cycles | 2,032 | 2,043 | essentially unchanged |
| `DC_PRELOAD_XN_IMM` total cycles | 1,029 | 1,036 | essentially unchanged |
| `MOV_SRC_TO_DST_ALIGNv2` MTE2 | 967 | 960 | unchanged GM load pattern |
| `WAIT_FLAG_MTE2` | 940 | 933 | memory wait unchanged |

## Hardware latency (`remote_verify`)

| label | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 (ms) | Optimized Triton (ms) |
|---|---:|---:|---:|---:|
| tiny_16 | 0.043774 | 0.075459 | 0.075713 | 0.075720 |
| small_32 | 0.106094 | 0.173408 | 0.173227 | 0.173053 |
| medium_64 | 0.758092 | 0.698614 | 0.699718 | 0.699217 |
| exact_128 | 6.382802 | 4.285429 | 4.296395 | 4.290766 |

Correctness: `UNIT_TEST PASS`. The synthetic `persistent_rows` shape passes optimized correctness while baseline providers are parser-visible and skipped with `grid_guard` because their direct `N*out_channels` launch would exceed 65,535.

## Interpretation

The target shape is already served by the fast direct fused Triton path; the optimized direct path is intentionally near-identical and measures within run noise of baseline. The main improvement is dispatch legality/generalization: large valid row counts route to a capped persistent kernel instead of invalid direct grids, while target-shape latency is preserved.
