# Performance Report

## Methodology

Simulation used `cannsim_local_run(gen_report=True)` with sub-kernel C++ hosts and real `trace_core0.json` reports. The main comparison uses the same logical sub-kernel shape for both versions: `M=64, K=512, grid=1`; baseline keeps `BLOCK_K=128` (4 loop iterations), optimized uses `BLOCK_K=512` (1 loop iteration).

Hardware latency on physical NPU was not available in this local sandbox; the table includes cannsim cycle-derived estimates using `cycles * 0.4 ns`.

## Trace Summary

| Kernel | Sub-kernel | wall_cycles | est. latency (µs) | x_events | i_events | Bottleneck |
|---|---:|---:|---:|---:|---:|---|
| Baseline Triton | M=64,K=512,BLOCK_K=128 | 7,573 | 3.029 | 3,567 | 69 | MTE2 |
| Optimized Triton | M=64,K=512,BLOCK_K=512 | 5,597 | 2.239 | 3,383 | 90 | MTE3 |

Speedup from cannsim wall cycles: `7573 / 5597 = 1.35x`.

## Pipeline Breakdown

| Pipeline | Baseline busy_cyc | Baseline % wall | Optimized busy_cyc | Optimized % wall | Change |
|---|---:|---:|---:|---:|---:|
| MTE2 | 5,028 | 66.4% | 1,951 | 34.9% | -61.2% |
| VEC | 5,009 | 66.1% | 1,933 | 34.5% | -61.4% |
| MTE3 | 4,425 | 58.4% | 3,703 | 66.2% | -16.3% |
| SCALAR | 2,225 | 29.4% | 1,892 | 33.8% | -15.0% |
| SCALARLDST | 2,029 | 26.8% | 1,738 | 31.1% | -14.3% |
| PUSHQ | 1,976 | 26.1% | 1,627 | 29.1% | -17.7% |
| RVECEX | 1,224 | 16.2% | 1,570 | 28.1% | +28.3% |
| RVECLD | 1,260 | 16.6% | 1,537 | 27.5% | +22.0% |
| RVECST | 1,047 | 13.8% | 520 | 9.3% | -50.3% |
| FLOWCTRL | 7 | 0.1% | 7 | 0.1% | 0.0% |

## Top Critical Instructions

| Kernel | Instruction | Pipe | Count | Total cycles | Avg cycles |
|---|---|---|---:|---:|---:|
| Baseline | ST_XD_XN_IMM | SCALARLDST | 32 | 39,507 | 1,235 |
| Baseline | WAIT_FLAG_MTE2 | VEC | 4 | 11,945 | 2,986 |
| Baseline | RV_VLDI | RVECLD | 1,032 | 9,996 | 10 |
| Baseline | WAIT_FLAG_VEC | MTE2 | 5 | 8,199 | 1,640 |
| Optimized | RV_VLDI | RVECLD | 1,024 | 9,856 | 10 |
| Optimized | RV_VMUL | RVECEX | 512 | 4,096 | 8 |
| Optimized | RV_VADD | RVECEX | 576 | 4,032 | 7 |
| Optimized | MOV_SRC_TO_DST_ALIGNv2 | MTE2 | 2 | 3,872 | 1,936 |

## Additional cannsim checks

| Variant | Result |
|---|---|
| Baseline K=128 sub-kernel | PASS, `wall_cycles=3443`, bottleneck SCALAR |
| Padded Cube GEMV K=128 | PASS, `wall_cycles=11009`, slower due N=1 padded to N=16 |
| Padded Cube GEMV K=512 | Compile failed: UB overflow (`2359296 bits` required vs `2031616` available) |
