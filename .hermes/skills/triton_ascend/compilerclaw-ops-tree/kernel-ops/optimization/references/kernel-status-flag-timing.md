# Kernel Status Flag Timing

## Problem

Setting `kernel_status(action='set', key='recorded', value=True)` before the pipeline has advanced to the `record` stage causes the flag to be reset when the agent later calls `advance`.

## Correct Timing

| Flag | Set when stage is | Do NOT set earlier |
|------|------------------|-------------------|
| `verified=True` | `verify` | Setting during `optimize` gets reset |
| `recorded=True` | `record` | Setting during `verify` gets reset |

## Pipeline Flow

```
optimize → verify → record → done
              ↑          ↑
          set verified  set record here
          HERE          HERE
```

## Symptom

Agent sets `recorded=True` during the optimize or verify stage. Then when `advance` is called to move from verify→record, the flag is reset to False. The pipeline shows `recorded=False` even though the agent previously set it.

## Fix

Only set status flags when the pipeline is at the matching stage. If you're not sure what stage the pipeline is at, call `kernel_status(action='read')` first.
