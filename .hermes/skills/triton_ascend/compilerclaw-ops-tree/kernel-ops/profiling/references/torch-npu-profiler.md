# torch_npu.profiler Reference

Measures device kernel and operator latency on Ascend NPU hardware. Use instead of `do_bench` when you need per-op device durations from `kernel_details.csv`, or when `do_bench` timing is unreliable.

## Basic Usage

```python
import torch
import torch_npu
from torch_npu.profiler import (
    ProfilerActivity,
    profile,
    schedule,
    tensorboard_trace_handler,
    _ExperimentalConfig,
    ProfilerLevel,
    AiCMetrics,
    ExportType,
)

out_dir = "/tmp/prof_output"
total_steps = wait + warmup + active

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
    schedule=schedule(wait=wait, warmup=warmup, active=active, repeat=1),
    on_trace_ready=tensorboard_trace_handler(out_dir),
    record_shapes=True,
    experimental_config=_ExperimentalConfig(
        profiler_level=ProfilerLevel.Level1,
        aic_metrics=AiCMetrics.PipeUtilization,
        export_type=[ExportType.Text],
    ),
) as prof:
    torch.npu.synchronize()           # ensure NPU idle before first step
    for step in range(total_steps):
        op()
        torch.npu.synchronize()       # separate launches for clean per-step attribution
        prof.step()
torch.npu.synchronize()               # ensure last step completes before parsing
```

---

## `profile()` Context Manager

```python
profile(
    activities,          # Iterable[ProfilerActivity] — which activities to record
    schedule,            # schedule() object — controls wait/warmup/active phasing
    on_trace_ready,      # callback when trace is ready (tensorboard_trace_handler or custom)
    record_shapes,       # bool — record input tensor shapes (useful for debugging)
    profile_memory,      # bool — track memory allocation events
    with_stack,          # bool — record Python stack traces
    with_flops,          # bool — estimate FLOPs per op
    with_modules,        # bool — record nn.Module hierarchy
    experimental_config, # _ExperimentalConfig — NPU-specific configuration
    use_cuda,            # Optional[bool] — ignored on Ascend
)
```

### ProfilerActivity

| Member | Description |
|--------|-------------|
| `ProfilerActivity.CPU` | Host-side CPU events |
| `ProfilerActivity.NPU` | Device-side NPU kernel events |

Use `[ProfilerActivity.CPU, ProfilerActivity.NPU]` for full host+device profiling.

---

## `schedule(wait, active, warmup, repeat, skip_first)`

Controls which steps are profiled. Steps advance via `prof.step()` calls.

| Parameter | Type | Description |
|-----------|------|-------------|
| `wait` | int | Number of initial steps to skip (no recording) |
| `warmup` | int | Steps to warm up the profiler (no recording, prepare internals) |
| `active` | int | Steps that actually record data — this is what appears in output |
| `repeat` | int | Number of wait+warmup+active cycles; 0 = repeat until script ends |
| `skip_first` | int | Additional initial steps to skip before the first cycle |

The profiler steps through: `[skip_first] [wait × repeat] [warmup × repeat] [active × repeat]`.

For profiling a single op, use: `schedule(wait=1, warmup=1, active=3, repeat=1)` → total steps = 5.

### ProfilerAction (step states)

| Action | Meaning |
|--------|---------|
| `NONE` | No profiling action |
| `RECORD` | Currently recording |
| `RECORD_AND_SAVE` | Recording complete, writing data |
| `WARMUP` | Warming up profiler internals |

---

## `tensorboard_trace_handler(dir_name, worker_name, analyse_flag, async_mode)`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dir_name` | `None` | Output directory for trace data. If `None`, uses `os.getcwd()` (or `ASCEND_WORK_PATH` env var if set). A timestamped subdirectory `<hostname>_<pid>_<timestamp>_ascend_pt/` is created inside it. |
| `worker_name` | `None` | Name prefix for worker-specific files. If `None`, uses `socket.gethostname()`. Final subdirectory name: `<worker_name>_<pid>_<timestamp>_ascend_pt/` |
| `analyse_flag` | `True` | Automatically parse trace data into CSV after recording |
| `async_mode` | `False` | Parse trace data asynchronously after context exits |

After profiling completes and `analyse_flag=True`, the handler runs CANN parsing to generate CSV outputs. Parsing is synchronous by default (the `with profile(...)` exit blocks until CSV files are written).

---

## `_ExperimentalConfig` — NPU-Specific Options

```python
_ExperimentalConfig(
    profiler_level,        # str or int — profiling granularity
    aic_metrics,           # str or int — AICore metric collection type
    l2_cache,              # bool — collect L2 cache hit/miss stats
    msprof_tx,             # bool — enable msprof TX (legacy)
    mstx,                  # bool — enable MSTX markers
    data_simplification,   # bool — simplify output data (default True, reduces size)
    record_op_args,        # bool — record operator input/output args
    op_attr,               # bool — record operator attributes
    gc_detect_threshold,   # float — GC detection threshold in ms
    export_type,           # str or list — output format(s)
    host_sys,              # list — host system metrics to collect
    mstx_domain_include,   # list — MSTX domains to include
    mstx_domain_exclude,   # list — MSTX domains to exclude
    sys_io,                # bool — collect system I/O metrics
    sys_interconnection,   # bool — collect system interconnect metrics
)
```

### `profiler_level`

Controls granularity of device profiling. Higher levels collect more data but add overhead.

| Level | String | Description | kernel_details columns |
|-------|--------|-------------|----------------------|
| `Level0` | `"Level0"` | Basic timeline only (default). Minimal overhead. | 9 |
| `Level1` | `"Level1"` | Timeline + AICore metrics (pipe utilization, MAC ratio, scalar ratio, etc.). Required for `aic_metrics`. | 28–49 (varies by metric) |
| `Level2` | `"Level2"` | Full detail: timeline + AICore metrics + memory bandwidth + extra device-level stats. Highest overhead. | 34–49 (varies by metric) |
| `Level_none` | `"Level_none"` | No device-level profiling. | — |

**Use `Level1`** for kernel duration + pipeline breakdown. Use `Level0` if you only need timing.

### `aic_metrics`

Controls which AICore pipeline metrics are collected. Only effective when `profiler_level` is `Level1` or higher. Not all metrics are supported on all hardware — unsupported ones are silently reset to `ACL_AICORE_PIPE_UTILIZATION` (default) with a warning.

| Enum | String | Description |
|------|--------|-------------|
| `AiCMetrics.AiCoreNone` | `"ACL_AICORE_NONE"` | No AICore metrics |
| `AiCMetrics.PipeUtilization` | `"ACL_AICORE_PIPE_UTILIZATION"` | Pipeline utilization (default). Breaks down MAC, scalar, MTE1/2/3, fixpipe ratios. |
| `AiCMetrics.ArithmeticUtilization` | `"ACL_AICORE_ARITHMETIC_UTILIZATION"` | Arithmetic unit utilization |
| `AiCMetrics.Memory` | `"ACL_AICORE_MEMORY_BANDWIDTH"` | Memory bandwidth utilization |
| `AiCMetrics.MemoryAccess` | `"ACL_AICORE_MEMORY_ACCESS"` | Memory access patterns |
| `AiCMetrics.MemoryL0` | `"ACL_AICORE_L0B_AND_WIDTH"` | L0 buffer usage and bandwidth |
| `AiCMetrics.MemoryUB` | `"ACL_AICORE_MEMORY_UB"` | Unified Buffer usage and bandwidth |
| `AiCMetrics.L2Cache` | `"ACL_AICORE_L2_CACHE"` | L2 cache hit/miss statistics |
| `AiCMetrics.ResourceConflictRatio` | `"ACL_AICORE_RESOURCE_CONFLICT_RATIO"` | Resource conflict statistics |

### `export_type`

Controls output format. Can be a string or list of formats.

| Enum | String | Description |
|------|--------|-------------|
| `ExportType.Text` | `"text"` | Human-readable text output (default). Generates CSV files in `ASCEND_PROFILER_OUTPUT/`. |
| `ExportType.Db` | `"db"` | SQLite database output (in `PROF_*/msprof_*.db`). Can be queried with msprof tools. |

To export both: `export_type=[ExportType.Text, ExportType.Db]`.

---

## Output Directory Structure

```
<dir_name>/
  <worker_name>/                           # e.g. tritondev_1_2119348_20260722233341791_ascend_pt/
    profiler_info.json                     # profiler config, device info, version
    profiler_metadata.json                 # run metadata
    logs/                                  # parser log files (for debugging)
    ASCEND_PROFILER_OUTPUT/                # ← main results
      kernel_details.csv                   # per-kernel device timings
      operator_details.csv                 # per-op host+device timings
      op_statistic.csv                     # aggregated op-level stats
      api_statistic.csv                    # host-side API call stats
      step_trace_time.csv                  # per-step timeline breakdown
      trace_view.json                      # full trace for TensorBoard / Perfetto
      analyse.done                         # marker file (0 bytes, signals parsing complete)
    FRAMEWORK/
      torch.op_mark                        # PyTorch op markers
      torch.op_range                       # PyTorch op range events
    PROF_000001_<id>/
      msprof_*.db                          # SQLite database (if export_type includes 'db')
      device_0/
        sample.json                        # raw device sampling data
        data/                              # raw device trace slices
      host/
        sample.json                        # raw host sampling data
        data/                              # raw host trace slices (API events, etc.)
```

---

## Key Output Files

### `kernel_details.csv` — Per-Kernel Device Durations

This is the primary file for measuring kernel latency. Each row is one kernel launch. The number and type of columns depend on `profiler_level` and `aic_metrics`.

#### Base columns (all levels)

Present in every `kernel_details.csv` regardless of `profiler_level`:

| Column | Description |
|--------|-------------|
| `Step Id` | Profiler step number |
| `Device_id` | NPU device index |
| `Name` | Kernel name (e.g. `aclnnFlashAttentionScoreV4_FlashAttentionScore_FlashAttentionScore`) |
| `Type` | Kernel type (`FlashAttentionScore`, etc.) |
| `Accelerator Core` | `MIX_AIC`, `AIC`, `AIV`, etc. |
| `Start Time(us)` | Device start timestamp (microseconds) |
| `Duration(us)` | **Device kernel duration (microseconds)** — primary metric |
| `Wait Time(us)` | Time waiting in queue before launch |
| `Block Num` | Number of AIC blocks |

#### Level1 columns (extended metadata)

Added when `profiler_level >= Level1`:

| Column | Description |
|--------|-------------|
| `Model ID` | Model identifier |
| `Task ID` | Task identifier |
| `Stream ID` | NPU stream identifier |
| `OP State` | `static` or `dynamic` |
| `Mix Block Num` | Number of mixed blocks (AIC + AIV) |
| `HF32 Eligible` | Whether HF32 precision is applicable |
| `Input Shapes` | Semicolon-separated input tensor shapes |
| `Input Data Types` | Semicolon-separated input dtypes |
| `Input Formats` | Semicolon-separated input layouts (NCHW, ND, etc.) |
| `Output Shapes` | Semicolon-separated output tensor shapes |
| `Output Data Types` | Semicolon-separated output dtypes |
| `Output Formats` | Semicolon-separated output layouts |
| `Context ID` | Execution context ID |
| `aicore_time(us)` | Total AICore execution time |
| `aic_total_cycles` | Total AICore cycles |
| `aiv_time(us)` | Total AIV (Vector) execution time |
| `aiv_total_cycles` | Total AIV cycles |

#### Metric-specific columns

These vary by `aic_metrics` setting. See the table below for what each metric adds.

---

### Metric-Specific Columns by `aic_metrics`

#### `PipeUtilization` (default, 49 total columns)

Pipeline time breakdown — which hardware unit is busy during execution.

| Column | Unit | Description |
|--------|------|-------------|
| `aic_mac_time(us)` | us | Matrix multiply (MAC) time |
| `aic_mac_ratio` | 0–1 | MAC time / aicore_time |
| `aic_scalar_time(us)` | us | Scalar unit time |
| `aic_scalar_ratio` | 0–1 | Scalar time / aicore_time |
| `aic_mte1_time(us)` | us | MTE1 (L1→L0) data movement time |
| `aic_mte1_ratio` | 0–1 | MTE1 time / aicore_time |
| `aic_mte2_time(us)` | us | MTE2 (GM→L1/UB) data movement time |
| `aic_mte2_ratio` | 0–1 | MTE2 time / aicore_time |
| `aic_mte3_time(us)` | us | MTE3 (UB→GM) data movement time |
| `aic_mte3_ratio` | 0–1 | MTE3 time / aicore_time |
| `aic_fixpipe_time(us)` | us | Fixpipe (L0C→UB) time |
| `aic_fixpipe_ratio` | 0–1 | Fixpipe time / aicore_time |
| `aic_icache_miss_rate` | 0–1 | AICore ICache miss rate |
| `aiv_vec_time(us)` | us | Vector execution time |
| `aiv_vec_ratio` | 0–1 | Vector time / aiv_time |
| `aiv_scalar_time(us)` | us | AIV scalar time |
| `aiv_scalar_ratio` | 0–1 | AIV scalar / aiv_time |
| `aiv_mte2_time(us)` | us | AIV MTE2 time |
| `aiv_mte2_ratio` | 0–1 | AIV MTE2 / aiv_time |
| `aiv_mte3_time(us)` | us | AIV MTE3 time |
| `aiv_mte3_ratio` | 0–1 | AIV MTE3 / aiv_time |
| `aiv_icache_miss_rate` | 0–1 | AIV ICache miss rate |
| `cube_utilization(%)` | % | Cube (MAC) utilization percentage |

**Use for:** identifying whether a kernel is compute-bound (high MAC ratio), memory-bound (high MTE2 ratio), or scalar-bound (high scalar ratio).

#### `ArithmeticUtilization` (29 total columns)

Arithmetic precision and FLOP count.

| Column | Unit | Description |
|--------|------|-------------|
| `aic_mac_fp16_ratio` | 0–1 | Ratio of FP16 MAC operations |
| `aic_mac_int8_ratio` | 0–1 | Ratio of INT8 MAC operations |
| `aic_cube_fops` | count | Total cube FLOP count |

**Use for:** checking precision utilization (fp16 vs int8 mix) and FLOP count.

#### `MemoryBandwidth` (34 total columns, Level2 only)

Memory bandwidth at each hierarchy level. **Note:** On some hardware this metric is not supported and may fail to generate `kernel_details.csv`.

| Column | Unit | Description |
|--------|------|-------------|
| `aic_l1_read_bw(GB/s)` | GB/s | L1 buffer read bandwidth |
| `aic_l1_write_bw(GB/s)` | GB/s | L1 buffer write bandwidth |
| `aic_main_mem_read_bw(GB/s)` | GB/s | GM (main memory) read bandwidth |
| `aic_main_mem_write_bw(GB/s)` | GB/s | GM (main memory) write bandwidth |
| `aiv_ub_read_bw(GB/s)` | GB/s | UB read bandwidth |
| `aiv_ub_write_bw(GB/s)` | GB/s | UB write bandwidth |
| `aiv_main_mem_read_bw(GB/s)` | GB/s | AIV main memory read bandwidth |
| `aiv_main_mem_write_bw(GB/s)` | GB/s | AIV main memory write bandwidth |

**Use for:** checking memory-bound kernels — if main memory BW is saturated, further compute optimization won't help.

#### `MemoryAccess` (49 total columns)

Same columns as `PipeUtilization`. Falls back to `PipeUtilization` on hardware that doesn't support MemoryAccess natively.

#### `L0B_AND_WIDTH` (33 total columns)

L0 buffer (L0A/L0B/L0C) read/write bandwidth.

| Column | Unit | Description |
|--------|------|-------------|
| `aic_l0a_read_bw(GB/s)` | GB/s | L0A (A-matrix) read bandwidth |
| `aic_l0a_write_bw(GB/s)` | GB/s | L0A write bandwidth |
| `aic_l0b_read_bw(GB/s)` | GB/s | L0B (B-matrix) read bandwidth |
| `aic_l0b_write_bw(GB/s)` | GB/s | L0B write bandwidth |
| `aic_l0c_read_bw_cube(GB/s)` | GB/s | L0C (accumulator) read bandwidth via Cube |
| `aic_l0c_write_bw_cube(GB/s)` | GB/s | L0C write bandwidth via Cube |
| `aiv_l0c_read_bw(GB/s)` | GB/s | AIV L0C read bandwidth |

**Use for:** checking if L0 buffer bandwidth is the bottleneck in matmul/conv kernels.

#### `MemoryUB` (34 total columns)

Unified Buffer read/write bandwidth for vector and scalar operations.

| Column | Unit | Description |
|--------|------|-------------|
| `aic_ub_read_bw_scalar(GB/s)` | GB/s | AIC UB read bandwidth (scalar) |
| `aic_ub_write_bw_scalar(GB/s)` | GB/s | AIC UB write bandwidth (scalar) |
| `aic_fixp2ub_write_bw(GB/s)` | GB/s | Fixpipe→UB write bandwidth |
| `aiv_ub_read_bw_vector(GB/s)` | GB/s | AIV UB read bandwidth (vector) |
| `aiv_ub_write_bw_vector(GB/s)` | GB/s | AIV UB write bandwidth (vector) |
| `aiv_ub_read_bw_scalar(GB/s)` | GB/s | AIV UB read bandwidth (scalar) |
| `aiv_ub_write_bw_scalar(GB/s)` | GB/s | AIV UB write bandwidth (scalar) |
| `aiv_fixp2ub_write_bw(GB/s)` | GB/s | AIV Fixpipe→UB write bandwidth |

**Use for:** checking if UB bandwidth is saturated in vector-heavy kernels.

#### `L2Cache` (38 total columns)

L2 cache hit/miss/victim counts for both AIC and AIV.

| Column | Unit | Description |
|--------|------|-------------|
| `aic_read_local_l2_hit` | count | AIC L2 read hits |
| `aic_read_local_l2_miss` | count | AIC L2 read misses |
| `aic_read_local_l2_victim` | count | AIC L2 read victims |
| `aic_write_local_l2_hit` | count | AIC L2 write hits |
| `aic_write_local_l2_miss` | count | AIC L2 write misses |
| `aic_write_local_l2_victim` | count | AIC L2 write victims |
| `aiv_read_local_l2_hit` | count | AIV L2 read hits |
| `aiv_read_local_l2_miss` | count | AIV L2 read misses |
| `aiv_read_local_l2_victim` | count | AIV L2 read victims |
| `aiv_write_local_l2_hit` | count | AIV L2 write hits |
| `aiv_write_local_l2_miss` | count | AIV L2 write misses |
| `aiv_write_local_l2_victim` | count | AIV L2 write victims |

**Use for:** checking if L2 cache thrashing is causing memory-bound behavior.

#### `ResourceConflictRatio` (28 total columns)

Resource conflict statistics — vector bank conflicts and resource conflicts.

| Column | Unit | Description |
|--------|------|-------------|
| `aiv_vec_bank_cflt_ratio` | 0–1 | Vector bank conflict ratio |
| `aiv_vec_resc_cflt_ratio` | 0–1 | Vector resource conflict ratio |

**Use for:** checking if vector resource conflicts are degrading performance.

---

### `operator_details.csv` — Per-Op Host + Device Timings

Each row is one PyTorch/ACL/Triton operator (may contain multiple kernel launches).

| Column | Description |
|--------|-------------|
| `Name` | Operator name (e.g. `npu::npu_fusion_attention`) |
| `Input Shapes` | Multiline input tensor shapes |
| `Call Stack` | Python call stack (if `with_stack=True`) |
| `Host Self Duration(us)` | Host-side self time (excludes child ops) |
| `Host Total Duration(us)` | Host-side total time (includes child ops) |
| `Device Self Duration(us)` | Device self time |
| `Device Total Duration(us)` | Device total time |
| `Device Self Duration With AICore(us)` | Device self time on AICore |
| `Device Total Duration With AICore(us)` | Device total time on AICore |

### `op_statistic.csv` — Aggregated Op Stats

| Column | Description |
|--------|-------------|
| `Device_id` | NPU device index |
| `OP Type` | Operation type name |
| `Core Type` | `MIX_AIC`, `AIC`, `AIV`, etc. |
| `Count` | Number of invocations in active steps |
| `Total Time(us)` | Total duration |
| `Min Time(us)` | Minimum duration |
| `Avg Time(us)` | Average duration |
| `Max Time(us)` | Maximum duration |
| `Ratio(%)` | Percentage of total device time |

### `api_statistic.csv` — Host API Call Stats

| Column | Description |
|--------|-------------|
| `Device_id` | `host` or device ID |
| `Level` | `acl`, etc. |
| `API Name` | ACL API name |
| `Time(us)` | Total time |
| `Count` | Number of calls |
| `Avg(us)` | Average time |
| `Min(us)` | Minimum time |
| `Max(us)` | Maximum time |
| `Variance` | Time variance |

### `step_trace_time.csv` — Per-Step Timeline

| Column | Description |
|--------|-------------|
| `Device_id` | NPU device index |
| `Step` | Step number |
| `Computing` | Computing time (us) |
| `Communication(Not Overlapped)` | Communication time not overlapped with compute |
| `Overlapped` | Communication time overlapped with compute |
| `Free` | Idle time |
| `Stage` | Total stage time |
| `Bubble` | Pipeline bubble time |
| `Preparing` | Preparation overhead |

### `trace_view.json` — Full Trace Events

Chrome Trace Format (TensorBoard / Perfetto compatible). Contains all host and device events with timestamps. Load in `chrome://tracing` or TensorBoard's Profiler tab.

---

## Recommended Configurations

### Minimal overhead — timing only
```python
experimental_config=_ExperimentalConfig(profiler_level="Level0")
```
9 columns in `kernel_details.csv`. Just `Duration(us)`, `Wait Time(us)`, `Block Num`.

### Standard — timing + pipeline breakdown
```python
experimental_config=_ExperimentalConfig(
    profiler_level=ProfilerLevel.Level1,
    aic_metrics=AiCMetrics.PipeUtilization,
    export_type=[ExportType.Text],
)
```
49 columns. Adds MAC/scalar/MTE/fixpipe ratios, cube utilization, input/output shapes.

### Arithmetic analysis — FLOP count + precision mix
```python
experimental_config=_ExperimentalConfig(
    profiler_level=ProfilerLevel.Level1,
    aic_metrics=AiCMetrics.ArithmeticUtilization,
    export_type=[ExportType.Text],
)
```
29 columns. Adds `aic_mac_fp16_ratio`, `aic_mac_int8_ratio`, `aic_cube_fops`.

### Memory analysis — L0/L1/UB/GM bandwidth
```python
experimental_config=_ExperimentalConfig(
    profiler_level=ProfilerLevel.Level2,
    aic_metrics=AiCMetrics.Memory,
    export_type=[ExportType.Text],
)
```
34 columns. Adds L1 read/write, GM read/write, UB read/write bandwidth in GB/s.
**Note:** May not work on all hardware — `MemoryAccess` was observed to reset to default on some machines.

### L2 cache analysis
```python
experimental_config=_ExperimentalConfig(
    profiler_level=ProfilerLevel.Level1,
    aic_metrics=AiCMetrics.L2Cache,
    export_type=[ExportType.Text],
)
```
38 columns. Adds L2 hit/miss/victim counts for both AIC and AIV.

### Full detail — all metrics
```python
experimental_config=_ExperimentalConfig(
    profiler_level=ProfilerLevel.Level2,
    aic_metrics=AiCMetrics.PipeUtilization,
    export_type=[ExportType.Text, ExportType.Db],
    record_op_args=True,
    l2_cache=True,
)
```

---

## Pitfalls

- **Sync between steps:** Always call `torch.npu.synchronize()` after each op and before `prof.step()` to ensure clean per-step device attribution. Without sync, queued launches can inflate reported durations.
- **Sync at profiler start/end:** Call `torch.npu.synchronize()` once before entering the `with profile(...)` block and once after exiting, to ensure the NPU is idle before recording starts and that the last step completes before data is written.
- **Level1/Level2 overhead:** Collecting AICore metrics adds non-trivial overhead. For accurate wall-clock timing comparison, use `Level0`; use `Level1+` for pipeline analysis.
- **Parsing time:** After the profiler context exits, CANN parsing takes 2–4 seconds to generate CSV files. This is blocking by default (set `async_mode=True` in `tensorboard_trace_handler` to make it non-blocking).
- **`analyse_flag=False`** skips CSV generation entirely. Only trace_view.json will be available.
- **`data_simplification=True`** (default) reduces output size by skipping some intermediate data. Set to `False` for complete data.
- **Output directory reuse:** Each profiling run creates a unique timestamped subdirectory under `dir_name`. Old data is not overwritten but accumulates. Clean up with `shutil.rmtree(dir_name, ignore_errors=True)` when done.
- **Large kernels and OOM:** If the profiled op uses a lot of memory, profiling may OOM. Clear tensors and call `torch.npu.empty_cache()` before each profiler session.
- **`Duration(us)` is device-side time** from the NPU hardware timestamp, not wall-clock. It excludes host dispatch overhead. Use `operator_details.csv` `Host Total Duration(us)` for the full host+device time.
- **`MemoryAccess` not supported on all hardware:** On some Ascend hardware, `ACL_AICORE_MEMORY_ACCESS` is silently reset to the default metric. Check the profiler warning logs (`logs/`) if you get the same columns as `PipeUtilization` when you expected memory access columns.
- **`MemoryBandwidth` (Level2) may fail:** On some machines, `ACL_AICORE_MEMORY_BANDWIDTH` with Level2 produces a "Failed to get acl to npu flow events" error and no `kernel_details.csv` is generated. Use `PipeUtilization` as fallback.
