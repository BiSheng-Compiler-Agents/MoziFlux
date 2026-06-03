import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _l2norm_rowwise_kernel(
    x_ptr, y_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    cols = tl.arange(0, BLOCK_N)
    x_row_ptrs = x_ptr + rows[:, None] * stride_xm
    y_row_ptrs = y_ptr + rows[:, None] * stride_ym
    col_ptrs_x = cols[None, :] * stride_xn
    col_ptrs_y = cols[None, :] * stride_yn

    sumsq = tl.zeros([BLOCK_M], dtype=tl.float32)
    n = 0
    while n < N:
        offs = n + cols
        mask = row_mask[:, None] & (offs[None, :] < N)
        x = tl.load(x_row_ptrs + (n * stride_xn) + col_ptrs_x, mask=mask, other=0.0)
        xf = x.to(tl.float32)
        sumsq += tl.sum(xf * xf, axis=1)
        n += BLOCK_N

    inv_norm = tl.rsqrt(sumsq)

    n = 0
    while n < N:
        offs = n + cols
        mask = row_mask[:, None] & (offs[None, :] < N)
        x = tl.load(x_row_ptrs + (n * stride_xn) + col_ptrs_x, mask=mask, other=0.0)
        y = x * inv_norm[:, None]
        tl.store(y_row_ptrs + (n * stride_yn) + col_ptrs_y, y, mask=mask)
        n += BLOCK_N


def _select_config(N: int):
    if N >= 32768:
        block_m, block_n, warps, stages = 8, 2048, 8, 2
    elif N >= 8192:
        block_m, block_n, warps, stages = 4, 1024, 4, 2
    elif N >= 2048:
        block_m, block_n, warps, stages = 2, 1024, 4, 2
    else:
        block_m, block_n, warps, stages = 1, 256, 2, 2
    return block_m, block_n, warps, stages


class ModelNew(nn.Module):
    """
    Simple model that performs L2 normalization.
    """
    def __init__(self):
        """
        Initializes the L2Norm layer.

        Args:
            dim (int): Dimension along which to normalize.
        """
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L2 normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, D).

        Returns:
            torch.Tensor: Output tensor with L2 normalization applied, same shape as input.
        """
        if not getattr(x, "is_npu", False):
            raise RuntimeError("ModelNew expects an input tensor on Ascend NPU")
        if x.dim() != 2:
            raise ValueError("ModelNew expects a 2D input tensor")
        if x.requires_grad:
            raise RuntimeError("ModelNew does not support autograd-enabled inputs")

        x_c = x.contiguous()
        B, D = x_c.shape
        y = torch.empty_like(x_c)

        stride_xm, stride_xn = x_c.stride()
        stride_ym, stride_yn = y.stride()

        BLOCK_M, BLOCK_N, num_warps, num_stages = _select_config(D)
        grid = (triton.cdiv(B, BLOCK_M),)

        _l2norm_rowwise_kernel[grid](
            x_c, y,
            B, D,
            stride_xm, stride_xn,
            stride_ym, stride_yn,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return y
batch_size = 32768
dim = 65535

def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]
def get_init_inputs():
    return []
