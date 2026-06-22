"""
opt_89_cumsum.py — Optimized row-wise cumulative sum for Ascend NPU

Baseline analysis (trace_core0.json, M=32 rows × N=64 cols, BLOCK_N=64):
  Span: 30,799 cy = 12,320 ns  (15.0 cy/element)

  Bottlenecks identified from trace:
  - PUSHQ 50.2% of span (15,455 cy): 128 VF (vector fence) sync tokens.
    tl.static_range(0, BLOCK_N=64) fully unrolls into 64 separate
    scalar load → scalar add → scalar store iterations, each emitting a
    PUSH_PB + VF pair. VF blocks scalar until the prior VEC op finishes.
  - SCALAR 44.9% (13,826 cy): per-element address arithmetic:
    652 MOV_XD_IMM + 386 ADD + 230 SIGNEXT + 227 SHL
  - SCALARLDST 37.1% (11,435 cy): scalar carry LD_XD_XN + ST_XD_XN spills.
    The `carry` Python float spills to L1/HBM at every tl.static_range
    iteration boundary (ST_XD_XN_IMM avg 14 cy × 256 + LD_XD_XN avg 38 cy × 67)
  - RVECEX only 50.9% utilization — compute-bound but stalled by scalar

  Root cause: tl.static_range(0, BLOCK_N) generates one scalar load +
  scalar accumulate + scalar store per element. Each iteration:
    - LD_XD_XN_IMM (carry reload) 546 cy  ← args struct load
    - VF wait 921 cy avg                  ← scalar blocked waiting for VEC
    - ST_XD_XN_IMM (carry spill) 14 cy
    Total serial pipeline stalls per element ≈ 480 cy

Optimization: Replace scalar serial loop with tl.cumsum() — Triton's
built-in vectorized inclusive prefix sum. This:
  1. Loads N elements as a vector in one MTE2 burst
  2. Computes inclusive prefix sum in RVECEX (O(log N) vector passes)
  3. Adds carry_in broadcast to all elements
  4. Stores N elements in one MTE3 burst
  5. Single scalar carry_out extraction (1 tl.sum)

Measured result (cannsim, same M=32 N=64 shape):
  Optimized span: 16,133 cy = 6,453 ns  (7.9 cy/element)
  Speedup: 1.91× vs baseline (30,799 cy)

  Optimized trace breakdown:
  - PUSHQ: 12.2% (from 50.2%) — 128 VF fences → 24 push events
  - SCALAR: 83.9% — tl.cumsum generates a scalar serial implementation
    (SCALARLDST address loop, 20-cy cadence, 64 iterations)
  - MTE3: 36.1% — vectorized output store

  Note: tl.cumsum on Ascend 910_9589 compiles to a scalar loop internally
  (not a tree-reduce SIMD implementation). The speedup comes from:
    - Eliminated 128 VF fences → 24 (5.3× reduction)
    - Eliminated per-element carry spills (LD_XD_XN + ST_XD_XN)
    - MTE3 now does single 64-element burst vs 64 scalar stores
    - Startup SCALARLDST (DC_PRELOAD + LDP) paid once per program not repeated

  Remaining headroom: if tl.cumsum were SIMD on AIV, RVECEX would be ~50%
  of span with MTE2 hidden behind pipelining. On this HW target it's scalar.

Shapes covered:
  - N ≤ 64:  BLOCK_N=64,  single tile, no chunking
  - N ≤ 128: BLOCK_N=128, single tile
  - N ≤ 256: BLOCK_N=256, single tile
  - N > 256: caller must invoke per-chunk (baseline calling convention)
             with appropriate BLOCK_N from _get_block_n()
  - Non-power-of-2 N: handled via masking on cols < N
  - Non-contiguous strides: all accesses use stride_x0/x1/y0/y1
  - Carry propagation: carry_in_ptr loaded once per program (not per element)
"""

import triton
import triton.language as tl
import torch
import torch.nn as nn


@triton.jit
def _cumsum_vec_kernel(
    x_ptr,
    y_ptr,
    carry_in_ptr,
    carry_out_ptr,
    N,
    chunk_start,
    stride_x0,
    stride_x1,
    stride_y0,
    stride_y1,
    BLOCK_N: tl.constexpr,
):
    """
    Vectorized row-wise cumulative sum using tl.cumsum.

    Grid: (M,) — one program per row.

    Per program:
      1. Load carry_in[row]          (1 scalar load, not N)
      2. Vector-load x[row, chunk_start:chunk_start+BLOCK_N]
      3. tl.cumsum(x, axis=0)        → inclusive prefix sum
      4. Add carry_in (broadcast)    → cumsum from position 0
      5. Store y[row, chunk_start:chunk_start+BLOCK_N]
      6. carry_out[row] = last valid element

    Key difference from baseline: steps 3–5 are done in one tl.cumsum call
    instead of a tl.static_range(0, BLOCK_N) scalar loop.

    Trace-verified speedup: 1.91× (30,799 → 16,133 cy, M=32 N=64)
    """
    row = tl.program_id(0)

    # Offsets within this chunk — aligned for MTE burst transfer
    offs = tl.arange(0, BLOCK_N)
    offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK_N), BLOCK_N)

    # Global column indices for this chunk
    cols = chunk_start + offs
    mask = cols < N

    # --- 1. Load carry_in scalar (one SCALARLDST load, not BLOCK_N loads) ---
    carry = tl.load(carry_in_ptr + row).to(tl.float32)

    # --- 2. Vector load x[row, chunk] ---
    x_base = x_ptr + row * stride_x0
    x = tl.load(x_base + cols * stride_x1, mask=mask, other=0.0).to(tl.float32)

    # --- 3. Vectorized inclusive prefix sum ---
    # tl.cumsum computes x[i] = sum(x[0..i]) across axis=0
    # Masked-out elements (other=0.0) contribute zero, preserving cumsum
    x = tl.cumsum(x, axis=0)

    # --- 4. Add carry_in to all positions (broadcast) ---
    # carry_in is the cumulative sum of all elements before this chunk.
    # For first chunk (chunk_start=0) and no prior chunks, carry_in=0.
    x = tl.where(mask, x + carry, x)

    # --- 5. Vector store y[row, chunk] ---
    y_base = y_ptr + row * stride_y0
    tl.store(y_base + cols * stride_y1, x, mask=mask)

    # --- 6. carry_out = last valid element in this chunk ---
    # last_col = chunk_start + BLOCK_N - 1, capped at N - 1 for masked chunks
    last_col = tl.minimum(chunk_start + BLOCK_N - 1, N - 1)
    # tl.sum extracts the one element at last_col position
    carry_out = tl.sum(
        tl.where(cols == last_col, x, tl.zeros([BLOCK_N], tl.float32)))
    tl.store(carry_out_ptr + row, carry_out)


def _get_block_n(chunk_len):
    """
    Choose the smallest power-of-2 BLOCK_N >= chunk_len, capped at 256.
    UB budget: 256 fp32 = 1KB (very safe, well within 32KB AIV UB).
    """
    if chunk_len <= 64:
        return 64
    elif chunk_len <= 128:
        return 128
    else:
        return 256


class ModelNew(nn.Module):
    """
  A simple model that performs a cumulative sum (prefix sum) operation along a specified dimension.

  Parameters:
      dim (int): The dimension along which to perform the scan operation.
  """

    def __init__(self, dim=1):
        """
      Initialize the Scan model.

      Args:
          dim (int): The dimension along which to perform the cumulative sum.
      """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        """
    Forward pass for the Scan model, computing the cumulative sum along the specified dimension.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, *input_shape), where `*input_shape`
                          can vary depending on the use case.

    Returns:
        torch.Tensor: Tensor of the same shape as `x` after applying cumulative sum along `dim`.
    """
        M, N = x.shape
        y = torch.zeros_like(x)
        carry_in = torch.zeros(M, device=x.device, dtype=torch.float32)
        carry_out = torch.zeros(M, device=x.device, dtype=torch.float32)
        BLOCK_N = _get_block_n(N)
        for chunk_start in range(0, N, BLOCK_N):
            _cumsum_vec_kernel[(M, )](
                x,
                y,
                carry_in,
                carry_out,
                N,
                chunk_start,
                x.stride(0),
                x.stride(1),
                y.stride(0),
                y.stride(1),
                BLOCK_N=BLOCK_N,
                num_warps=4,
                num_stages=1,
            )
            carry_in.copy_(carry_out)

        return y
