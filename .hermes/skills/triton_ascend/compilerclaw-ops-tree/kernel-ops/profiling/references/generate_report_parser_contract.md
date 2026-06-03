# generate_report.py parser contract

`results.txt` (captured stdout of `python profile_kernels.py`) is consumed by
`generate_report.py`. The parser is strict — a profile that runs fine on
hardware can still produce **0 parsed records** if the table format is wrong.

## Hard requirements for the benchmark table to parse

1. **`x_names` MUST be a single shape column: `x_names=["label"]`.**
   The parser treats column 0 as the shape and EVERY remaining column as a
   method. If you use `x_names=["batch","M","N","K"]`, M/N/K get misread as
   "methods" and nothing matches the alias table → 0 records. Encode
   multi-dim shapes as ONE label string, e.g. `"b2-m128-n128-k64"`.

2. **The table header's first token must be `label` or `N`.**
   The parser only starts reading a table when
   `stripped.split()[0] in ("label", "N")`. A header starting with `batch`,
   `Shape`, `M`, etc. is never recognized. Hand-rolled `print()` tables with
   a `Shape`/`N`/`Baseline`/`Optimized` header also fail — use
   `@perf_report`, not a manual table.

3. **Method names must match the alias table exactly:**
   - `"PyTorch / ACL"` — reference
   - `"Baseline Triton1"` — input kernel (`<N>_<name>.py`)
   - `"Baseline Triton2"` — golden reference (`base_<N>_<name>.py`)
   - `"Optimized Triton"` — optimized kernel (`opt_<N>_<name>.py`)

   `"Triton Opt"` is NOT recognized. `"Baseline Triton"` is only recognized
   in the 3-line (single baseline) format.

4. **Geomean speedup needs both `Baseline Triton1` AND `Optimized Triton`.**
   A 2-line profile (just optimized + reference) parses for the runtime
   plots but produces no speedup bars.

5. **Label strings MUST contain NO spaces.** The parser splits each data row
   on whitespace. A label with an embedded space (e.g. `"B=1 small"`) shifts
   every column right by one and the row is discarded. Use hyphens/underscores:
   `"B1-small"`, `"b2-m128-n128-k64"`.

## Quick self-check

After writing a profile, run its `results.txt` (or a captured sample) through
`generate_report.py --source-dir <kernel>` and confirm `recs > 0` with the
expected method names.
