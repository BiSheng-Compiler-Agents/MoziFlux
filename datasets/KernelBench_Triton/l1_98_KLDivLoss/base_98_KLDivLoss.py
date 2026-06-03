import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _kl_div_batch_sum_kernel(
    pred_ptr, targ_ptr, out_ptr,
    B, D,
    stride_pb, stride_pd,
    stride_tb, stride_td,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < B

    pred_row_ptrs = pred_ptr + rows[:, None] * stride_pb
    targ_row_ptrs = targ_ptr + rows[:, None] * stride_tb
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for k in range(0, D, BLOCK_N):
        k_cols = k + cols
        mask = row_mask[:, None] & (k_cols[None, :] < D)
        p = tl.load(pred_row_ptrs + k_cols[None, :] * stride_pd, mask=mask, other=1.0).to(tl.float32)
        t = tl.load(targ_row_ptrs + k_cols[None, :] * stride_td, mask=mask, other=0.0).to(tl.float32)

        t_pos = t > 0.0
        t_safe = tl.where(t_pos, t, 1.0)
        contrib = tl.where(t_pos, t * tl.log(t_safe / p), 0.0)
        acc += tl.sum(contrib, axis=1)

    tl.store(out_ptr + rows, acc, mask=row_mask)


@triton.jit
def _kl_div_batch_sum_kernel_full_tiles(
    pred_ptr, targ_ptr, out_ptr,
    D,
    stride_pb,
    stride_tb,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)

    pred_ptrs = pred_ptr + rows[:, None] * stride_pb + cols[None, :]
    targ_ptrs = targ_ptr + rows[:, None] * stride_tb + cols[None, :]
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for _ in range(0, D, BLOCK_N):
        p = tl.load(pred_ptrs).to(tl.float32)
        t = tl.load(targ_ptrs).to(tl.float32)

        t_pos = t > 0.0
        t_safe = tl.where(t_pos, t, 1.0)
        contrib = tl.where(t_pos, t * tl.log(t_safe / p), 0.0)
        acc += tl.sum(contrib, axis=1)
        pred_ptrs += BLOCK_N
        targ_ptrs += BLOCK_N

    tl.store(out_ptr + rows, acc)


@triton.jit
def _kl_div_batch_sum_kernel_shape_16384(
    pred_ptr, targ_ptr, out_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)

    pred_ptrs = pred_ptr + rows[:, None] * 16384 + cols[None, :]
    targ_ptrs = targ_ptr + rows[:, None] * 16384 + cols[None, :]
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for _ in range(0, 32):
        p = tl.load(pred_ptrs).to(tl.float32)
        t = tl.load(targ_ptrs).to(tl.float32)

        t_pos = t > 0.0
        t_safe = tl.where(t_pos, t, 1.0)
        contrib = tl.where(t_pos, t * tl.log(t_safe / p), 0.0)
        acc += tl.sum(contrib, axis=1)
        pred_ptrs += BLOCK_N
        targ_ptrs += BLOCK_N

    tl.store(out_ptr + rows, acc)


class ModelNew(nn.Module):
    """
    A model that computes Kullback-Leibler Divergence for comparing two distributions.

    Parameters:
        None
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        if predictions.device.type != "npu" or targets.device.type != "npu":
            raise ValueError("ModelNew expects predictions and targets on Ascend NPU")
        if predictions.ndim != 2 or targets.ndim != 2:
            raise ValueError("ModelNew expects 2D predictions and targets")
        if predictions.shape != targets.shape:
            raise ValueError("ModelNew expects predictions and targets with matching shapes")

        # Ensure contiguous memory for predictable strides
        p = predictions.contiguous()
        t = targets.contiguous()
        B, D = p.shape

        # Per-row accumulators (float32 for numeric stability)
        row_sums = torch.empty(B, device=p.device, dtype=torch.float32)

        # Strides in elements
        stride_pb, stride_pd = p.stride()
        stride_tb, stride_td = t.stride()

        # Batch multiple rows per program to amortize launch and pointer overheads.
        block_m = 12
        block_n = 512
        use_full_tiles = (
            stride_pd == 1
            and stride_td == 1
            and B % block_m == 0
            and D % block_n == 0
        )

        use_shape_16384 = (
            stride_pd == 1
            and stride_td == 1
            and B == 16384
            and D == 16384
        )

        if use_shape_16384:
            grid = (B // block_m,)
            _kl_div_batch_sum_kernel_shape_16384[grid](
                p, t, row_sums,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                num_warps=4,
                num_stages=4,
            )
        elif use_full_tiles:
            grid = (B // block_m,)
            _kl_div_batch_sum_kernel_full_tiles[grid](
                p, t, row_sums,
                D,
                stride_pb,
                stride_tb,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                num_warps=4,
                num_stages=4,
            )
        else:
            grid = (triton.cdiv(B, block_m),)
            _kl_div_batch_sum_kernel[grid](
                p, t, row_sums,
                B, D,
                stride_pb, stride_pd,
                stride_tb, stride_td,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                num_warps=4,
                num_stages=4,
            )
        # 'batchmean' reduction: sum over all elements divided by batch size
        return row_sums.sum() / B
batch_size = 8192 * 2
input_shape = (8192 * 2,)
dim = 1

def get_inputs():
    scale = torch.rand(())
    return [(torch.rand(batch_size, *input_shape)*scale).softmax(dim=-1), torch.rand(batch_size, *input_shape).softmax(dim=-1)]
def get_init_inputs():
    return []
