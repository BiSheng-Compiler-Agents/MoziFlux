import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def _triplet_margin_row_kernel(
    anchor_ptr,
    pos_ptr,
    neg_ptr,
    out_ptr,
    B,
    D,
    stride_a0,
    stride_a1,
    stride_p0,
    stride_p1,
    stride_n0,
    stride_n1,
    eps,
    margin,
    BLOCK_M: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    N_ITERS: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < B

    col_offsets = tl.arange(0, BLOCK_SIZE)
    a_row_ptr = anchor_ptr + rows[:, None] * stride_a0
    p_row_ptr = pos_ptr + rows[:, None] * stride_p0
    n_row_ptr = neg_ptr + rows[:, None] * stride_n0

    acc_ap = tl.zeros((BLOCK_M, ), dtype=tl.float32)
    acc_an = tl.zeros((BLOCK_M, ), dtype=tl.float32)

    for i in tl.static_range(N_ITERS):
        offs = i * BLOCK_SIZE + col_offsets
        col_mask = offs < D
        mask = row_mask[:, None] & col_mask[None, :]

        a = tl.load(a_row_ptr + offs[None, :] * stride_a1,
                    mask=mask,
                    other=0.0)
        p = tl.load(p_row_ptr + offs[None, :] * stride_p1,
                    mask=mask,
                    other=0.0)
        n = tl.load(n_row_ptr + offs[None, :] * stride_n1,
                    mask=mask,
                    other=0.0)

        da = a - p
        dn = a - n

        acc_ap += tl.sum(da * da, axis=1)
        acc_an += tl.sum(dn * dn, axis=1)

    d_ap = tl.sqrt(acc_ap + eps)
    d_an = tl.sqrt(acc_an + eps)
    loss = tl.maximum(d_ap - d_an + margin, 0.0)

    tl.store(out_ptr + rows, loss, mask=row_mask)


@triton.jit
def _triplet_margin_row_kernel_aligned(
    anchor_ptr,
    pos_ptr,
    neg_ptr,
    out_ptr,
    B,
    stride_a0,
    stride_p0,
    stride_n0,
    eps,
    margin,
    BLOCK_M: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    N_ITERS: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < B

    col_offsets = tl.arange(0, BLOCK_SIZE)
    a_ptrs = anchor_ptr + rows[:, None] * stride_a0 + col_offsets[None, :]
    p_ptrs = pos_ptr + rows[:, None] * stride_p0 + col_offsets[None, :]
    n_ptrs = neg_ptr + rows[:, None] * stride_n0 + col_offsets[None, :]

    acc_ap = tl.zeros((BLOCK_M, ), dtype=tl.float32)
    acc_an = tl.zeros((BLOCK_M, ), dtype=tl.float32)

    for _ in tl.static_range(N_ITERS):
        a = tl.load(a_ptrs, mask=row_mask[:, None], other=0.0)
        p = tl.load(p_ptrs, mask=row_mask[:, None], other=0.0)
        n = tl.load(n_ptrs, mask=row_mask[:, None], other=0.0)

        da = a - p
        dn = a - n

        acc_ap += tl.sum(da * da, axis=1)
        acc_an += tl.sum(dn * dn, axis=1)

        a_ptrs += BLOCK_SIZE
        p_ptrs += BLOCK_SIZE
        n_ptrs += BLOCK_SIZE

    d_ap = tl.sqrt(acc_ap + eps)
    d_an = tl.sqrt(acc_an + eps)
    loss = tl.maximum(d_ap - d_an + margin, 0.0)

    tl.store(out_ptr + rows, loss, mask=row_mask)


def _triplet_margin_loss_triton(anchor: torch.Tensor,
                                positive: torch.Tensor,
                                negative: torch.Tensor,
                                margin: float = 1.0,
                                eps: float = 1e-6) -> torch.Tensor:
    if anchor.ndim != 2 or positive.ndim != 2 or negative.ndim != 2:
        raise ValueError(
            "triplet margin loss expects 2D tensors shaped [batch, features]")
    if anchor.shape != positive.shape or anchor.shape != negative.shape:
        raise ValueError(
            "anchor, positive, and negative must have the same shape")
    if anchor.device != positive.device or anchor.device != negative.device:
        raise ValueError(
            "anchor, positive, and negative must be on the same device")
    if anchor.device.type != "npu":
        raise RuntimeError(
            "triplet margin loss Triton kernel requires Ascend NPU tensors")

    a = anchor.contiguous()
    p = positive.contiguous()
    n = negative.contiguous()

    if a.dtype != torch.float32:
        a = a.float()
    if p.dtype != torch.float32:
        p = p.float()
    if n.dtype != torch.float32:
        n = n.float()

    B, D = a.shape
    out = torch.empty(B, device=a.device, dtype=torch.float32)

    if D == 8192:
        BLOCK_SIZE = 2048
        BLOCK_M = 5
    else:
        if D >= 2048:
            BLOCK_SIZE = 1024
        elif D >= 1024:
            BLOCK_SIZE = 512
        elif D >= 512:
            BLOCK_SIZE = 256
        else:
            BLOCK_SIZE = 128
        BLOCK_M = 4

    N_ITERS = triton.cdiv(D, BLOCK_SIZE)
    contiguous_fast_path = (a.stride(1) == 1 and p.stride(1) == 1
                            and n.stride(1) == 1 and D % BLOCK_SIZE == 0)

    grid = (triton.cdiv(B, BLOCK_M), )
    if contiguous_fast_path:
        _triplet_margin_row_kernel_aligned[grid](
            a,
            p,
            n,
            out,
            B,
            a.stride(0),
            p.stride(0),
            n.stride(0),
            eps,
            float(margin),
            BLOCK_M=BLOCK_M,
            BLOCK_SIZE=BLOCK_SIZE,
            N_ITERS=N_ITERS,
        )
    else:
        _triplet_margin_row_kernel[grid](
            a,
            p,
            n,
            out,
            B,
            D,
            a.stride(0),
            a.stride(1),
            p.stride(0),
            p.stride(1),
            n.stride(0),
            n.stride(1),
            eps,
            float(margin),
            BLOCK_M=BLOCK_M,
            BLOCK_SIZE=BLOCK_SIZE,
            N_ITERS=N_ITERS,
        )
    return out.mean()


class ModelNew(nn.Module):
    """
    A model that computes Triplet Margin Loss for metric learning tasks.

    Parameters:
        margin (float): The margin between the positive and negative samples.
    """

    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.margin = float(margin)
        self.eps = 1e-6

    def forward(self, anchor, positive, negative):
        return _triplet_margin_loss_triton(anchor, positive, negative,
                                           self.margin, self.eps)


batch_size = 32768
input_shape = (8192, )
dim = 1


def get_inputs():
    scale = torch.rand(())
    return [
        torch.rand(batch_size, *input_shape) * scale,
        torch.rand(batch_size, *input_shape),
        torch.rand(batch_size, *input_shape)
    ]


def get_init_inputs():
    return [1.0]  # Default margin
