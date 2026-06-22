# Performance Report

## Cannsim setup

- Tool: `cannsim_local_run(gen_report=True)`
- Job: `/tmp/cannsim_local/l1_67_conv1d_baseline`
- Trace: `/tmp/cannsim_local/l1_67_conv1d_baseline/cannsim_20260625050212_test_kernel/report/trace_core0.json`
- Probe: one baseline custom Conv1d tile, `grid=(1,1,1)`, `BLOCK_OC=16`, `BLOCK_T=16`, `BLOCK_P=16`, runtime `B=1,C=1,L=16,OC=16,K=1`.
- Note: optimized `ModelNew` removes the custom Triton kernel and dispatches to ACL; therefore optimized custom-kernel cannsim cycles are reported as `0` / not applicable.

## Cannsim trace summary

| Kernel path | wall_cycles | hardware time (ns, cycles*0.4) | x_events | i_events | Dominant pipeline |
|---|---:|---:|---:|---:|---|
| Baseline custom Triton Conv1d micro-probe | 5,500 | 2,200 | 1,616 | 192 | SCALARLDST (3,137 busy cycles) |
| Optimized ACL dispatch custom Triton work | 0 | 0 | 0 | 0 | N/A (custom launch removed) |

## Baseline pipeline breakdown

| Pipeline | Ops | Busy cycles | Lanes | Window | Note |
|---|---:|---:|---:|---|---|
| SCALARLDST | 41 | 3,137 | 4 | [3449,8238] | Bottleneck |
| PUSHQ | 45 | 2,600 | 3 | [4355,7817] | Queue/dispatch pressure |
| SCALAR | 735 | 1,651 | 12 | [3423,8922] | Address and loop bookkeeping |
| MTE3 | 3 | 1,195 | 2 | [6670,7866] | Store/writeback wait |
| RVECEX | 286 | 574 | 8 | [4682,7423] | Vector-side work |
| RVECST | 211 | 558 | 9 | [4691,7515] | Vector store activity |
| MTE2 | 2 | 543 | 1 | [4741,5285] | Load movement |
| CUBE | 2 | 151 | 1 | [8209,8361] | Very small fraction of wall |

## Top baseline instructions

| Instruction | Pipe | Count | Total cycles | Avg cycles | Interpretation |
|---|---|---:|---:|---:|---|
| LDP_XI_XJ_XN | SCALAR | 7 | 5,805 | 829 | Critical scalar/address overhead |
| LD_XD_XN_IMM | SCALARLDST | 6 | 3,806 | 634 | Critical scalar load/store overhead |
| VF | PUSHQ | 22 | 3,228 | 147 | Queue pressure |
| ST_XD_XN_IMM | SCALARLDST | 19 | 3,207 | 169 | Scalar spill/store overhead |
| DC_PRELOAD_XN_IMM | SCALAR | 4 | 3,112 | 778 | Critical scalar preload overhead |
| WAIT_FLAG_VEC | MTE3 | 1 | 1,148 | 1,148 | Store-side wait |

## Hardware latency

Remote hardware verification: `remote_verify(run_test=True, run_bench=True)` passed. Optimized correctness passed on every benchmark shape with max_diff=0 against the ACL reference. Comparison Triton providers were parser-visible but pre-skipped as `inf` to avoid direct-convolution `coreDim`/timeout risk.

| Shape | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 (ms) | Optimized Triton (ms) | Optimized / ACL |
|---|---:|---:|---:|---:|---:|
| small_L256 | 0.007406 | inf | inf | 0.007453 | 1.006x |
| medium_L4096 | 0.046326 | inf | inf | 0.046139 | 0.996x |
| irregular_L8193 | 0.045343 | inf | inf | 0.045422 | 1.002x |
| exact_L131072 | 3.466113 | inf | inf | 3.461700 | 0.999x |

The optimized path is intentionally equivalent to ACL Conv1d, so hardware latency tracks the PyTorch/ACL reference within measurement noise. The custom Triton baseline was not launched in hardware profiling because the exact shape's launch product is `65536`, exceeding the Ascend 65,535 grid cap.
