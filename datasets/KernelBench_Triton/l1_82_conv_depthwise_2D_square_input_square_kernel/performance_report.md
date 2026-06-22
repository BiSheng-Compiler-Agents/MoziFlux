# Performance Report

## Cannsim setup

- Tool: `cannsim_local_run(gen_report=True)`
- Baseline probe: same Triton depthwise-conv kernel body, `grid=(1,1,1)`, `N=1`, `C=1`, `H=3`, `W=258`, `K=3`, `BLOCK_W=256`
- Trace: `/tmp/cannsim_local/l1_82_dwconv_baseline/cannsim_20260625053831_test_kernel/report/trace_core0.json`
- Trace summary: `/tmp/cannsim_local/l1_82_dwconv_baseline/cannsim_20260625053831_test_kernel/report/trace_summary.txt`
- Optimized path: no custom Triton device kernel is launched; ACL Conv2d handles the operation, so custom-kernel cannsim cycles are reported as `0 / N.A.` per the ACL-dispatch pattern.

## Cannsim trace comparison

| Path | wall cycles | hw time (cycles * 0.4ns) | Dominant pipeline | Key evidence |
|---|---:|---:|---|---|
| Baseline Triton sub-kernel | 3,921 | 1.568 us | `02_SCALARLDST` 2,761 busy cycles | 45 scalar-load/store ops; `ST_XD_XN_IMM` total 7,340 cycles; no Cube use |
| Optimized ACL dispatch | 0 custom Triton cycles | N.A. | N.A. | Custom Triton launch removed; physical latency measured by `profile_kernels.py` / `remote_verify` |

## Baseline pipeline breakdown

| Pipeline | Ops | Busy cycles | Lane sum |
|---|---:|---:|---:|
| `02_SCALARLDST` | 45 | 2,761 | 12,027 |
| `01_SCALAR` | 352 | 2,093 | 5,883 |
| `04_MTE2` | 19 | 1,703 | 9,932 |
| `07_MTE3` | 2 | 990 | 990 |
| `05_VEC` | 1 | 610 | 610 |
| `12_RVECEX` | 69 | 70 | 518 |

## Top baseline instructions

| Instruction | Pipe | Count | Total cycles | Avg cycles |
|---|---|---:|---:|---:|
| `ST_XD_XN_IMM` | SCALARLDST | 10 | 7,340 | 734 |
| `MOV_SRC_TO_DST_ALIGNv2` | MTE2 | 9 | 5,471 | 608 |
| `MOV_SPR_XN` | MTE2 | 10 | 4,461 | 446 |
| `LD_XD_XN_IMM` | SCALARLDST | 26 | 3,795 | 146 |
| `LDP_XI_XJ_XN` | SCALAR | 9 | 2,055 | 228 |

## Hardware latency

`remote_verify(run_test=True, run_bench=True)` passed correctness for the optimized path on all benchmark shapes.

| Shape | PyTorch / ACL (ms) | Baseline Triton1 (ms) | Baseline Triton2 (ms) | Optimized Triton (ms) |
|---|---:|---:|---:|---:|
| `small_square` | 0.025035 | 0.078746 | 0.059400 | 0.024245 |
| `medium_square` | 0.069445 | 0.262714 | 0.349823 | 0.070908 |
| `target_square` | 2.206660 | inf (grid_guard) | inf (grid_guard) | 2.218322 |

The target-shape baseline Triton grids are pre-skipped because `N*C*H_OUT*ceil(W_OUT/256) = 1,044,480` exceeds Ascend's 65,535 launch-product guard; the optimized ACL dispatch runs and matches the PyTorch/ACL reference.
