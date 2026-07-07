# Elementwise GELU 2D Block-Pointer Probe

When optimizing standalone GELU/activation kernels on 2D tensors, do not assume that replacing a flat 1D sigmoid/exp formulation with `tl.math.tanh` is sufficient. A 1D tanh variant can regress because the instruction lowering and per-tile overhead may increase even though the math is cleaner.

## Reusable pattern

1. Preserve the exact approximate-GELU formula and FP32 intermediate math:
   ```python
   u = x32 * (0.7978845608028654 + 0.035677408136300125 * x32 * x32)
   y = (0.5 * x32 * (1.0 + tl.math.tanh(u))).to(x.dtype)
   ```
2. For 2D inputs, try a row-major block-pointer tile before accepting a flat 1D tile:
   ```python
   x_block_ptr = tl.make_block_ptr(
       base=x_ptr, shape=(rows, cols), strides=(cols, 1),
       offsets=(pid_row * BLOCK_ROWS, 0),
       block_shape=(BLOCK_ROWS, BLOCK_COLS), order=(1, 0),
   )
   x = tl.load(x_block_ptr)  # even path
   ```
3. Keep an even no-boundary path plus a boundary-checked fallback. Add a persistent row-tile path only when `ceil(rows / BLOCK_ROWS) > 65535`.
4. If cannsim hosts process different element counts per program, compare **cycles per element**, not raw wall cycles. State the normalization in `performance_report.md`.

## Interpretation

For memory-bound elementwise activations, MTE3 store wait may remain the bottleneck. A valid improvement can show higher raw per-program cycles if the optimized program processes more elements; normalized cycles/element and remote hardware latency decide whether to keep it.
