# Performance Report

## cannsim setup

- Tool: `cannsim_local_run(gen_report=True)`
- SoC: Ascend950
- Probe: one contiguous FP32 epilogue tile, `n_elements=8192`, `grid=(1,)`
- Baseline kernel probe: original fused divide + LeakyReLU algebra at `BLOCK_SIZE=8192`
- Optimized kernel probe: `tl.where` LeakyReLU algebra, no `.cg`, `care_padding=False`

## Trace summary

| Metric | Baseline | Optimized | Change |
|---|---:|---:|---:|
| wall cycles | 4287 | 3773 | -12.0% |
| hardware latency (`cycles * 0.4 ns`) | 1714.8 ns | 1509.2 ns | 1.136x faster |
| trace x_events | 1665 | 871 | -47.7% |
| trace i_events | 9 | 8 | -11.1% |
| bottleneck pipe | MTE3 | MTE3 | unchanged |

## Pipeline utilization

| Pipeline | Baseline busy cycles | Optimized busy cycles | Notes |
|---|---:|---:|---|
| MTE3 | 2474 | 1990 | Store-side wait reduced after the vector instruction stream shrank. |
| SCALAR | 1796 | 1798 | Effectively unchanged; not the optimization target. |
| SCALARLDST | 1755 | 1695 | Slightly reduced. |
| MTE2 | 1066 | 1075 | Effectively unchanged load movement. |
| VEC | 1041 | 1069 | Wait remains similar. |
| PUSHQ | 848 | 321 | Fewer vector instructions to enqueue. |
| RVECEX | 802 | 275 | Large reduction from removing `tl.minimum`/add form. |
| RVECLD | 762 | 253 | Fewer vector local-buffer load events. |
| RVECST | 743 | 264 | Fewer vector local-buffer store events. |

## Top instruction changes

| Instruction / group | Baseline | Optimized | Interpretation |
|---|---:|---:|---|
| `RV_VMINS` | 256 instructions / 1536 cycles | removed | Replaced by compare/select LeakyReLU form. |
| `RV_VMULS` | 128 instructions / 1024 cycles | 256 instructions / 2048 cycles | Extra multiply is cheaper than the removed min/add sequence overall. |
| `RV_VCMP_GE` | absent from top list | 128 instructions / 768 cycles | New branch-select predicate. |
| `RV_VSEL` | absent from top list | 128 instructions / 768 cycles | Selects positive vs negative path. |
| `WAIT_FLAG_VEC @ MTE3` | 1869 cycles | 1371 cycles | Store-side wait reduced by shorter vector stream. |

## Hardware latency

Remote hardware verification was run with `remote_verify(run_test=True, run_bench=True)` and passed correctness for the optimized direct path plus a forced persistent-path unit test.

| label | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 (ms) | Optimized Triton (ms) | Notes |
|---|---:|---:|---:|---:|---|
| small | 0.148259 | 0.162637 | inf | 0.152058 | Optimized is 1.07x faster than Baseline Triton1. |
| irregular | 0.775144 | 0.646248 | inf | 0.616833 | Optimized is 1.05x faster than Baseline Triton1. |
| default | 22.871266 | inf | inf | 0.029918 | Baseline Triton1 benchmark was pre-skipped after runtime failure; optimized correctness passed. |

Baseline Triton2 is parser-visible but skipped because the sandbox explicitly marks `base_*.py` reference files as `DO NOT read`. The `default` optimized latency is the remote profiler output recorded in `results.txt`; correctness for that shape reported `max_abs=0` against PyTorch / ACL.
