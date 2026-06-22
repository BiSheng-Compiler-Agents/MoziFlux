"""
Optimized Softmax kernel for Ascend NPUs — KernelBench l1_23.

Optimizations applied vs. baseline (23_Softmax.py):

1. Pattern 6 — Multi-row blocked softmax (online max/sum)
   BLOCK_M rows per program with a chunked column loop (BLOCK_N columns per
   iteration). Uses numerically-stable online max/sum update so N can exceed
   SRAM capacity.  The baseline loads the entire row in one shot which is
   limited by the Ascend Unified Buffer size.

2. Pattern 11 — tl.multiple_of + tl.max_contiguous on column offsets
   Signals to Ascend MTE that loads are 128-byte aligned and contiguous,
   enabling burst transfers instead of scalar/strided loads.

3. Pattern 19 — eviction_policy="evict_first" on streaming loads
   Both passes read each cache line only once.  Marking them evict_first
   keeps L1 free for other data and avoids unnecessary evictions.

4. Pattern 20 — tl.range instead of Python range for inner loops
   Generates Ascend-native loop instructions that bishengir can pipeline.

5. Launch parameters: BLOCK_M=2, BLOCK_N=2048, num_warps=8, num_stages=1
   num_stages=1 is correct for Ascend — MTE has hardware prefetch;
   software double-buffering adds overhead.

cannsim results (Ascend950, 64 rows x 2048 cols, float32):
  Baseline:  248 cycles = 99.2 ns  (1 row/program, whole-row load)
  Optimized: 248 cycles = 99.2 ns  (2 rows/program, BLOCK_N=2048=N → single iteration)

At larger N (e.g. 393,216 cols from the KernelBench reference shape), the
chunked loop is essential — the baseline cannot fit a full row in UB while
the optimized kernel tiles freely.  KernelBench reference perf: 15,866 µs.
"""

import triton
import triton.language as tl
import torch.nn as nn


@triton.jit
def _softmax_row_fwd_db_kernel(
    x_ptr,
    y_ptr,
    n_rows,
    n_cols,
    stride_row,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < n_rows

    # Initialize accumulators: masked rows get neutral values.
    row_max = tl.where(row_mask, -float("inf"), 0.0)
    row_sum = tl.where(row_mask, 0.0, 1.0)

    # ── Pass 1: online numerically-stable max + exp-sum ───────────────────
    for start in tl.range(0, n_cols, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        offs = tl.multiple_of(offs, BLOCK_N)  # Pattern 11: alignment hint
        offs = tl.max_contiguous(offs, BLOCK_N)  # Pattern 11: contiguity hint
        col_mask = offs < n_cols
        mask = row_mask[:, None] & col_mask[None, :]
        x_ptrs = x_ptr + rows[:, None] * stride_row + offs[None, :]
        x = tl.load(x_ptrs,
                    mask=mask,
                    other=-float("inf"),
                    eviction_policy="evict_first").to(tl.float32)  # Pattern 19
        block_max = tl.max(x, axis=1)
        new_row_max = tl.where(row_mask, tl.maximum(row_max, block_max), 0.0)
        exp_scale = tl.where(row_mask, tl.exp(row_max - new_row_max), 0.0)
        row_sum = tl.where(
            row_mask,
            row_sum * exp_scale +
            tl.sum(tl.exp(x - new_row_max[:, None]), axis=1),
            1.0,
        )
        row_max = new_row_max

    # ── Pass 2: normalize and store ───────────────────────────────────────
    for start in tl.range(0, n_cols, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        offs = tl.multiple_of(offs, BLOCK_N)
        offs = tl.max_contiguous(offs, BLOCK_N)
        col_mask = offs < n_cols
        mask = row_mask[:, None] & col_mask[None, :]
        x_ptrs = x_ptr + rows[:, None] * stride_row + offs[None, :]
        y_ptrs = y_ptr + rows[:, None] * stride_row + offs[None, :]
        x = tl.load(x_ptrs,
                    mask=mask,
                    other=-float("inf"),
                    eviction_policy="evict_first").to(tl.float32)
        y = tl.exp(x - row_max[:, None]) / row_sum[:, None]
        tl.store(y_ptrs, y, mask=mask)


class ModelNew(nn.Module):

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        """Row-wise softmax over a 2-D tensor on Ascend NPU."""
        assert x.ndim == 2, "softmax expects a 2D tensor [rows, cols]"
        n_rows, n_cols = x.shape
        y = x.new_empty(x.shape)
        BLOCK_M = 2
        BLOCK_N = 2048
        grid = (triton.cdiv(n_rows, BLOCK_M), )
        _softmax_row_fwd_db_kernel[grid](
            x,
            y,
            n_rows,
            n_cols,
            x.stride(0),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=8,
            num_stages=1,
        )
        return y
