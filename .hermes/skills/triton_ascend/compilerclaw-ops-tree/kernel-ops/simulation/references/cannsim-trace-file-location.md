# Cannsim Trace File Location

## Problem

After running `cannsim_local_run` with `gen_report=True`, the `trace_core0.json` file is NOT in the job root directory. It's in the `report/` subdirectory.

## Correct Paths

```
/tmp/cannsim_local/<job_name>/report/trace_core0.json
/tmp/cannsim_local/<job_name>/report/trace_summary.txt  (after running aggregate_trace.py)
```

NOT:
```
/tmp/cannsim_local/<job_name>/trace_core0.json  ← WRONG
```

## Diagnosis

If you can't find the trace:
```bash
find /tmp/cannsim_local/ -name "trace_core0.json" 2>/dev/null
```

## Cannsim Output Too Large

`cannsim_local_run` output can exceed 200K characters (236 KB+), exceeding tool result size limits. When this happens:
1. Read trace files directly from disk via `read_file` or terminal
2. Use `aggregate_trace.py` to generate a compact `trace_summary.txt`
3. Read the summary instead of the full trace
