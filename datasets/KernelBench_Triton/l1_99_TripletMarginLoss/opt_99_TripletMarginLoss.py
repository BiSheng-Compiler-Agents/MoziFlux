import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl

_MAX_PROGRAMS = 65535
_ATOMIC_ROWS_THRESHOLD = 4096


@triton.jit
def _triplet_margin_row_direct_kernel(
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
    BLOCK_SIZE: tl.constexpr,
    N_ITERS: tl.constexpr,
):
    row = tl.program_id(0)
    valid_row = row < B
    a_row_ptr = anchor_ptr + row * stride_a0
    p_row_ptr = pos_ptr + row * stride_p0
    n_row_ptr = neg_ptr + row * stride_n0
    acc_ap = tl.zeros((), dtype=tl.float32)
    acc_an = tl.zeros((), dtype=tl.float32)
    for i in tl.static_range(N_ITERS):
        offs = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = valid_row & (offs < D)
        a = tl.load(a_row_ptr + offs * stride_a1, mask=mask, other=0.0)
        p = tl.load(p_row_ptr + offs * stride_p1, mask=mask, other=0.0)
        n = tl.load(n_row_ptr + offs * stride_n1, mask=mask, other=0.0)
        da = a - p
        dn = a - n
        acc_ap += tl.sum(da * da, axis=0)
        acc_an += tl.sum(dn * dn, axis=0)
    d_ap = tl.sqrt(acc_ap + eps)
    d_an = tl.sqrt(acc_an + eps)
    loss = tl.maximum(d_ap - d_an + margin, 0.0)
    tl.store(out_ptr + row, loss, mask=valid_row)


@triton.jit
def _triplet_margin_row_persistent_kernel(
    anchor_ptr,
    pos_ptr,
    neg_ptr,
    out_ptr,
    B,
    D,
    n_programs,
    eps,
    margin,
    BLOCK_SIZE: tl.constexpr,
    N_ITERS: tl.constexpr,
):
    row = tl.program_id(0)
    offs_base = tl.arange(0, BLOCK_SIZE)
    while row < B:
        a_row_ptr = anchor_ptr + row * D
        p_row_ptr = pos_ptr + row * D
        n_row_ptr = neg_ptr + row * D
        acc_ap = tl.zeros((), dtype=tl.float32)
        acc_an = tl.zeros((), dtype=tl.float32)
        for i in tl.static_range(N_ITERS):
            offs = i * BLOCK_SIZE + offs_base
            mask = offs < D
            a = tl.load(a_row_ptr + offs, mask=mask, other=0.0)
            p = tl.load(p_row_ptr + offs, mask=mask, other=0.0)
            n = tl.load(n_row_ptr + offs, mask=mask, other=0.0)
            da = a - p
            dn = a - n
            acc_ap += tl.sum(da * da, axis=0)
            acc_an += tl.sum(dn * dn, axis=0)
        d_ap = tl.sqrt(acc_ap + eps)
        d_an = tl.sqrt(acc_an + eps)
        loss = tl.maximum(d_ap - d_an + margin, 0.0)
        tl.store(out_ptr + row, loss, mask=True)
        row += n_programs


@triton.jit
def _triplet_margin_row_atomic_kernel(
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
    BLOCK_SIZE: tl.constexpr,
    N_ITERS: tl.constexpr,
):
    row = tl.program_id(0)
    a_row_ptr = anchor_ptr + row * stride_a0
    p_row_ptr = pos_ptr + row * stride_p0
    n_row_ptr = neg_ptr + row * stride_n0
    acc_ap = tl.zeros((), dtype=tl.float32)
    acc_an = tl.zeros((), dtype=tl.float32)
    for i in tl.static_range(N_ITERS):
        offs = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < D
        a = tl.load(a_row_ptr + offs * stride_a1, mask=mask, other=0.0)
        p = tl.load(p_row_ptr + offs * stride_p1, mask=mask, other=0.0)
        n = tl.load(n_row_ptr + offs * stride_n1, mask=mask, other=0.0)
        da = a - p
        dn = a - n
        acc_ap += tl.sum(da * da, axis=0)
        acc_an += tl.sum(dn * dn, axis=0)
    d_ap = tl.sqrt(acc_ap + eps)
    d_an = tl.sqrt(acc_an + eps)
    loss = tl.maximum(d_ap - d_an + margin, 0.0) / B
    tl.atomic_add(out_ptr, loss, sem="relaxed")


def _select_block_size(D: int) -> int:
    if D >= 2048:
        return 1024
    if D >= 1024:
        return 512
    if D >= 512:
        return 256
    return 128


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
    if B == 0 or D == 0:
        return torch.empty((), device=a.device, dtype=torch.float32).fill_(0.0)
    BLOCK_SIZE = _select_block_size(D)
    N_ITERS = triton.cdiv(D, BLOCK_SIZE)
    if B <= _ATOMIC_ROWS_THRESHOLD:
        out = torch.empty((1, ), device=a.device, dtype=torch.float32)
        out.zero_()
        _triplet_margin_row_atomic_kernel[(B, )](
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
            BLOCK_SIZE=BLOCK_SIZE,
            N_ITERS=N_ITERS,
        )
        return out[0]
    out = torch.empty(B, device=a.device, dtype=torch.float32)
    if B <= _MAX_PROGRAMS:
        _triplet_margin_row_direct_kernel[(B, )](
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
            BLOCK_SIZE=BLOCK_SIZE,
            N_ITERS=N_ITERS,
        )
    else:
        n_programs = _MAX_PROGRAMS
        _triplet_margin_row_persistent_kernel[(n_programs, )](
            a,
            p,
            n,
            out,
            B,
            D,
            n_programs,
            eps,
            float(margin),
            BLOCK_SIZE=BLOCK_SIZE,
            N_ITERS=N_ITERS,
        )
    return out.mean()


class ModelNew(nn.Module):
    """Optimized Triplet Margin Loss for 2D [batch, features] tensors."""

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
    return [1.0]
