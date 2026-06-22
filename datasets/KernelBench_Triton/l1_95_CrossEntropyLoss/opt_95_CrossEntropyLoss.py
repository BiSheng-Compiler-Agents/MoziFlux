"""
Optimized Cross-Entropy Loss — v4 (trace-driven, Ascend NPU).

Trace analysis of v3 (BLOCK_M=4, N=128, C=1000, span=4579 cy):

  Timeline:
    t=  20–580  (560cy):  SCALARLDST LD — loading x_ptr/t_ptr from args struct
    t= 624–1337 (713cy):  MTE2 first 2-row sub-load
    t= 624–1338 (714cy):  VEC WAIT_FLAG_MTE2  (exp stalls for data)
    t= 649–1899 (1250cy): SCALARLDST ST — writing 2nd MTE2 DMA descriptor
    t=1974–3259 (1285cy): MTE2 second 2-row sub-load starts 75cy after ST finishes
       → 637 cy dead gap between end of batch-1 MTE2 and start of batch-2 MTE2
    t=3896–4570 (674cy):  MTE3 output store

  Root causes:
    A) 637 cy MTE2 serialisation: the 2nd DMA descriptor (ST_XD_XN_IMM 1250cy)
       is not issued until after the 1st exp completes → num_stages=2 pipelines it.
    B) RVECEX 101% of span (all 4660 cy overlap with MTE2/MTE3 + SCALAR)
       — good utilisation, no obvious RVECEX waste.
    C) SCALAR LDP_XI_XJ_XN (2234cy), LD_DEV_XD_XN_IMM12 (1127cy), DC_PRELOAD (1067cy)
       — args-struct loads. Already amortised 4× vs v2 with BLOCK_M=4.
       Further amortisation: BLOCK_M=8 halves these again.

  V4 changes vs V3:
    1. num_stages=2 in small-C kernel → overlaps 2nd MTE2 with RVECEX of 1st.
       Expected saving: ~600 cy → span ≈ 3900 cy.
    2. Adaptive BLOCK_M dispatch (BLOCK_M=2/4/8) chosen to keep grid ≥ num_cores
       while maximising SCALAR amortisation.
    3. large-C kernel: same [BLOCK_M] vector accumulator approach as v3 but
       compiled with num_stages=2 to pipeline consecutive column-chunk loads.
"""

import triton
import triton.language as tl
import torch.nn as nn


@triton.jit
def _ce_v4_small(
    x_ptr,
    t_ptr,
    out_ptr,
    stride_x_row,
    stride_x_col,
    N,
    C,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    rmsk = rows < N

    # Issue target-index load early — overlaps with SCALAR preload startup
    tgt = tl.load(t_ptr + rows, mask=rmsk, other=0)
    tgt = tgt.to(tl.int64)

    # 2D load [BLOCK_M, BLOCK_C] with Pattern 11 hints for MTE2 burst coalescing
    cols = tl.arange(0, BLOCK_C)
    cols = tl.multiple_of(cols, BLOCK_C)
    cols = tl.max_contiguous(cols, BLOCK_C)
    cmsk = cols < C
    x = tl.load(
        x_ptr + rows[:, None].to(tl.int64) * stride_x_row +
        cols[None, :] * stride_x_col,
        mask=rmsk[:, None] & cmsk[None, :],
        other=float("-inf"),
        eviction_policy="evict_first",
    )

    m = tl.max(x, axis=1)
    expx = tl.math.exp(x - m[:, None])
    sumexp = tl.sum(expx, axis=1)
    logsumexp = tl.math.log(sumexp) + m

    x_t = tl.load(
        x_ptr + rows.to(tl.int64) * stride_x_row + tgt * stride_x_col,
        mask=rmsk,
        other=0.0,
    )
    tl.store(out_ptr + rows, logsumexp - x_t, mask=rmsk)


@triton.jit
def _ce_v4_large(
    x_ptr,
    t_ptr,
    out_ptr,
    stride_x_row,
    stride_x_col,
    N,
    C,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    rmsk = rows < N

    tgt = tl.load(t_ptr + rows, mask=rmsk, other=0)
    tgt = tgt.to(tl.int64)

    row_max = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    row_sum = tl.zeros([BLOCK_M], dtype=tl.float32)

    cols_base = tl.arange(0, BLOCK_C)
    cols_base = tl.multiple_of(cols_base, BLOCK_C)
    cols_base = tl.max_contiguous(cols_base, BLOCK_C)

    for col_start in tl.range(0, C, BLOCK_C):
        c = col_start + cols_base
        cmsk = c < C
        x = tl.load(
            x_ptr + rows[:, None].to(tl.int64) * stride_x_row +
            c[None, :] * stride_x_col,
            mask=rmsk[:, None] & cmsk[None, :],
            other=float("-inf"),
            eviction_policy="evict_first",
        )
        blk_max = tl.max(x, axis=1)
        new_max = tl.maximum(row_max, blk_max)
        row_sum = (row_sum * tl.math.exp(row_max - new_max) +
                   tl.sum(tl.math.exp(x - new_max[:, None]), axis=1))
        row_max = new_max

    logsumexp = tl.math.log(row_sum) + row_max
    x_t = tl.load(
        x_ptr + rows.to(tl.int64) * stride_x_row + tgt * stride_x_col,
        mask=rmsk,
        other=0.0,
    )
    tl.store(out_ptr + rows, logsumexp - x_t, mask=rmsk)


class ModelNew(nn.Module):
    """
    A model that computes Cross Entropy Loss for multi-class classification tasks.

    Parameters:
        None
    """

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x, t):
        """
        x: float32 [N, C] contiguous logits
        t: int64   [N]    target class indices
        returns float32 [N] per-sample NLL
        """
        assert x.is_contiguous()
        N, C = x.shape
        out = x.new_empty((N, ))

        # Adaptive BLOCK_M: use largest BLOCK_M s.t. grid >= 32 (all AIV cores active)
        # and BLOCK_M * BLOCK_C * 4 bytes fits in ~32 KB UB per core.
        BLOCK_C = min(triton.next_power_of_2(C), 2048)
        for BLOCK_M in [8, 4, 2, 1]:
            if triton.cdiv(N, BLOCK_M) >= 32:
                break
        grid = (triton.cdiv(N, BLOCK_M), )

        kw = dict(
            x_ptr=x,
            t_ptr=t,
            out_ptr=out,
            stride_x_row=x.stride(0),
            stride_x_col=x.stride(1),
            N=N,
            C=C,
            BLOCK_M=BLOCK_M,
            BLOCK_C=BLOCK_C,
            num_warps=4,
            num_stages=2,
        )
        if C <= 2048:
            _ce_v4_small[grid](**kw)
        else:
            _ce_v4_large[grid](**kw)
        return out
