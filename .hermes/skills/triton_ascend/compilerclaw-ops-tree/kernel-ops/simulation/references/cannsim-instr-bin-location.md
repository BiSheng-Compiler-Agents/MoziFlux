# cannsim instr.bin location and report path detection

cannsim writes `instr.bin` and `log_ca/` to its **current working directory** (CWD),
while creating a separate `cannsim_<ts>_<bin>/` subdirectory for bookkeeping
(`.soc-version`, `report/`). The two layouts occur in practice:

| Layout | Where instr.bin lands | When it happens |
|--------|----------------------|-----------------|
| **Job root** (common) | `<job_dir>/instr.bin` + `<job_dir>/log_ca/` | Plugin runs with `cwd=job_dir` and no `-o` flag |
| **Experiment subdir** | `<exp_dir>/instr.bin` + `<exp_dir>/log_ca/` | Manual runs from within the exp dir, or with `-o` |

## Detection pattern

```python
def _find_instr_bin(cwd: str) -> str | None:
    """Locate instr.bin — checks both job root and experiment subdir."""
    candidates = [os.path.join(cwd, "instr.bin")]
    exp = _find_experiment_dir(cwd)  # newest cannsim_*/ subdir
    if exp:
        candidates.append(os.path.join(exp, "instr.bin"))
    present = [p for p in candidates if os.path.isfile(p)]
    if not present:
        return None
    present.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return present[0]
```

## Report `-e` path

`cannsim report -e <dir>` needs the directory that **contains** `log_ca/`,
not the experiment subdir:

```python
report_src = None
for cand in (job_dir, exp_subdir):
    if cand and os.path.isdir(os.path.join(cand, "log_ca")):
        report_src = cand
        break
if not report_src:
    report_src = exp_subdir or job_dir  # fallback
```

## Early-kill interaction

The early-kill optimization in cannsim-local fires on the `all tasks are finished!`
marker + a grace period. But `instr.bin` is serialized during the atexit/teardown
phase **after** the marker. A blind `time.sleep(3); kill` races that write.

Fix: `_wait_for_instr_bin()` polls for a size-stable instr.bin (non-empty,
same size across two consecutive polls) before killing. Falls back to a blind
kill after a configurable ceiling (default 120s).

See `cannsim-hung-teardown-recovery.md` in the kernel-ops/simulation references
for the full forensic analysis and recovery commands.
