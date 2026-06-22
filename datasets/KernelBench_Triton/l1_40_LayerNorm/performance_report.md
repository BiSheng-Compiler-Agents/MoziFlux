# Performance Report

## Cannsim setup

Sub-kernel simulation used `cannsim_local_run(gen_report=True)`. The baseline trace uses one 8192-element atomic reduction tile from `_layernorm_sums_kernel`; the optimized trace uses one 16384-element private partial-reduction tile from `_layernorm_partial_kernel`. Trace summaries were generated from `trace_core0.json` using `aggregate_trace.py`.

## Cannsim trace comparison

| Kernel | trace_core0.json | Elements/tile | wall cycles | HW time (cycles × 0.4 ns) | x_events | i_events | bottleneck |
|---|---:|---:|---:|---:|---:|---:|---|
| Baseline `_layernorm_sums_kernel` | `/tmp/cannsim_local/l1_40_ln_baseline/cannsim_20260624221122_test_kernel/report/trace_core0.json` | 8192 | 4060 | 1.624 µs | 932 | 21 | SCALAR 1842 cyc |
| Optimized `_layernorm_partial_kernel` | `/tmp/cannsim_local/l1_40_ln_optimized_b16k/cannsim_20260624221809_test_kernel/report/trace_core0.json` | 16384 | 4961 | 1.984 µs | 1662 | 11 | SCALAR 1812 cyc |

### Pipeline utilization

| Kernel | SCALAR | SCALARLDST | MTE2 | VEC | PUSHQ | RVECEX | RVECLD | MTE3 | FLOWCTRL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline 8K atomic tile | 1842 | 1738 | 1068 | 1024 | 732 | 686 | 629 | 440 | 7 |
| Optimized 16K private tile | 1812 | 1742 | 1274 | 1227 | 1405 | 1359 | 1288 | 491 | 7 |

### Top critical instructions

| Kernel | Critical instructions |
|---|---|
| Baseline | `RV_VCADD` 5632 total cyc; `MOV_SRC_TO_DST_ALIGNv2` MTE2 1055 cyc; `WAIT_FLAG_MTE2` 1024 cyc |
| Optimized | `RV_VCADD` 11264 total cyc; `VF` PUSHQ 1401 cyc; `MOV_SRC_TO_DST_ALIGNv2` MTE2 1259 cyc |

## Interpretation

The optimized tile processes 2× as many elements for only 1.22× wall cycles (`4060 → 4961`), reducing reduction tiles per target row from 512 to 256. Full-grid benefit comes from fewer reduction programs/partials and removing atomic updates into the 16 row accumulators.

## Hardware verification (`remote_verify`)

Correctness: `UNIT_TEST PASS`; direct target path and optimized persistent dispatch path both passed.

| label | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 (ms) | Optimized Triton (ms) | Speedup vs Baseline1 | Speedup vs Baseline2 |
|---|---:|---:|---:|---:|---:|---:|
| batch1_target_norm | 0.699865 | 0.093521 | 0.141362 | 0.090661 | 1.032× | 1.559× |
| target | 1.040866 | 0.779223 | 0.753424 | 0.686316 | 1.135× | 1.098× |

Optimized target latency is 0.686316 ms, 1.14× faster than the editable baseline and 1.52× faster than PyTorch / ACL.
