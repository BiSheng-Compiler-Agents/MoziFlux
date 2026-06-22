import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def _cumsum_lastdim_kernel(
    x_ptr,  # *dtype
    out_ptr,  # *dtype
    M,  # int32
    N,  # int32
    stride_xm,  # int32
    stride_xn,  # int32
    stride_om,  # int32
    stride_on,  # int32
    BLOCK_N: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    row_x = x_ptr + pid_m * stride_xm
    row_o = out_ptr + pid_m * stride_om

    offs = tl.arange(0, BLOCK_N)
    carry = tl.zeros((), dtype=tl.float32)

    for block_idx in range(NUM_BLOCKS):
        cols = block_idx * BLOCK_N + offs
        mask = cols < N
        vals = tl.load(row_x + cols * stride_xn, mask=mask,
                       other=0.0).to(tl.float32)
        scan = tl.cumsum(vals, axis=0) + carry
        tl.store(row_o + cols * stride_on, scan, mask=mask)
        carry += tl.sum(vals, axis=0)


class ModelNew(nn.Module):
    """
    A model that performs a masked cumulative sum, only summing elements that satisfy a condition.

    Parameters:
        dim (int): The dimension along which to perform the masked cumulative sum.
    """

    def __init__(self, dim=1):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x, mask):
        return masked_cumsum(x, mask, dim=self.dim)


def masked_cumsum(x, mask, dim=-1):
    if x.device.type != "npu" or mask.device.type != "npu":
        raise ValueError("masked_cumsum requires NPU tensors")
    if x.shape != mask.shape:
        raise ValueError("x and mask must have the same shape")
    if x.ndim == 0:
        raise ValueError("masked_cumsum requires at least one dimension")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(
            "masked_cumsum supports float16, bfloat16, and float32 only")

    dim = dim % x.ndim
    mask = mask.to(torch.bool)
    y = x * mask.to(dtype=x.dtype)

    if dim != y.ndim - 1:
        y = y.movedim(dim, -1)
    y = y.contiguous()

    m_size = y.numel() // y.shape[-1]
    n_size = y.shape[-1]
    y2d = y.view(m_size, n_size)
    out2d = torch.empty_like(y2d)

    stride_xm, stride_xn = y2d.stride()
    stride_om, stride_on = out2d.stride()

    if n_size <= 128:
        block_n = 128
    elif n_size <= 512:
        block_n = 256
    elif n_size <= 2048:
        block_n = 1024
    elif n_size <= 8192:
        block_n = 2048
    else:
        block_n = 8192
    num_blocks = triton.cdiv(n_size, block_n)

    _cumsum_lastdim_kernel[(m_size, )](
        y2d,
        out2d,
        m_size,
        n_size,
        stride_xm,
        stride_xn,
        stride_om,
        stride_on,
        BLOCK_N=block_n,
        NUM_BLOCKS=num_blocks,
        num_warps=8,
        num_stages=4,
    )

    out = out2d.view_as(y)
    if dim != x.ndim - 1:
        out = out.movedim(-1, dim)
    return out


batch_size = 32768
input_shape = (32768, )
dim = 1


def get_inputs():
    x = torch.rand(batch_size, *input_shape)
    mask = torch.randint(0, 2, x.shape).bool()  # Random boolean mask
    return [x, mask]


def get_init_inputs():
    return [dim]
