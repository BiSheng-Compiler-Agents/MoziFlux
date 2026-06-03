import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(dict(BLOCK=128), num_warps=2, num_stages=2),
        triton.Config(dict(BLOCK=256), num_warps=4, num_stages=2),
        triton.Config(dict(BLOCK=512), num_warps=4, num_stages=2),
    ],
    key=["N"],
)
@triton.jit
def _cumprod_rowwise_kernel_vectorized(
    x_ptr,
    y_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    BLOCK: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    if pid_m >= M:
        return

    # Base pointers for this row
    x_row_ptr = x_ptr + pid_m * stride_xm
    y_row_ptr = y_ptr + pid_m * stride_ym

    carry = 1.0

    # Strictly sequential scan along the row to preserve cumprod semantics
    i = 0
    while i < N:
        v = tl.load(x_row_ptr + i * stride_xn)
        carry = carry * v
        tl.store(y_row_ptr + i * stride_yn, carry)
        i += 1


@triton.jit
def _touch_first_elem(y_ptr, M, stride_ym, stride_yn):
    pid = tl.program_id(axis=0)
    if pid >= M:
        return
    row_ptr = y_ptr + pid * stride_ym
    v = tl.load(row_ptr + 0 * stride_yn)
    tl.store(row_ptr + 0 * stride_yn, v)


class ModelNew(nn.Module):
    """
    A model that performs a cumulative product operation along a specified dimension.

    Parameters:
        dim (int): The dimension along which to perform the cumulative product operation.
    """

    def __init__(self, dim=1):
        """
        Initialize the CumulativeProductModel.

        Args:
            dim (int): The dimension along which to perform the cumulative product.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        """
        Forward pass, computing the cumulative product along the specified dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape).

        Returns:
            torch.Tensor: Tensor of the same shape as `x` after applying cumulative product along `dim`.
        """
        if not x.is_npu:
            raise RuntimeError("ModelNew expects inputs on Ascend NPU")
        if x.ndim == 0:
            raise RuntimeError("cumprod requires at least one dimension")

        dim = self.dim
        if dim < 0:
            dim += x.ndim
        if dim < 0 or dim >= x.ndim:
            raise IndexError(f"dim={self.dim} is out of range for ndim={x.ndim}")

        if x.numel() == 0:
            return torch.empty_like(x)

        perm = [idx for idx in range(x.ndim) if idx != dim] + [dim]
        moved = x.permute(perm).contiguous()
        scan_len = moved.shape[-1]
        flat = moved.reshape(-1, scan_len)
        out_flat = torch.empty_like(flat)

        grid = (flat.shape[0],)
        _cumprod_rowwise_kernel_vectorized[grid](
            flat,
            out_flat,
            flat.shape[0],
            flat.shape[1],
            flat.stride(0),
            flat.stride(1),
            out_flat.stride(0),
            out_flat.stride(1),
        )

        out = out_flat.reshape(moved.shape)
        inv_perm = [0] * x.ndim
        for idx, src in enumerate(perm):
            inv_perm[src] = idx
        return out.permute(inv_perm)
batch_size = 32768
input_shape = (32768,)
dim = 1

def get_inputs():
    return [torch.rand(batch_size, *input_shape)]
def get_init_inputs():
    return [dim]
