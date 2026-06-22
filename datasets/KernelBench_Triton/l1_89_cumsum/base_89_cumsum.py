import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _rowwise_cumsum_kernel(
    x_ptr,
    y_ptr,
    carry_in_ptr,
    carry_out_ptr,
    M,
    N,
    chunk_start,
    stride_x0,
    stride_x1,
    stride_y0,
    stride_y1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    cols = chunk_start + tl.arange(0, BLOCK_N)
    col_mask = cols < N
    mask = row_mask[:, None] & col_mask[None, :]

    x_ptrs = x_ptr + rows[:, None] * stride_x0 + cols[None, :] * stride_x1
    y_ptrs = y_ptr + rows[:, None] * stride_y0 + cols[None, :] * stride_y1
    carry = tl.load(carry_in_ptr + rows, mask=row_mask, other=0.0)
    values = tl.load(x_ptrs, mask=mask, other=0.0)
    values_f32 = values.to(tl.float32)
    partial = tl.cumsum(values_f32, axis=1)
    outputs = partial + carry[:, None]
    tl.store(y_ptrs, outputs, mask=mask)
    tl.store(carry_out_ptr + rows,
             carry + tl.sum(values_f32, axis=1),
             mask=row_mask)


def cumsum_npu(x, dim=1):
    if not hasattr(torch, "npu") or not x.is_npu:
        raise ValueError("cumsum_npu requires an Ascend NPU tensor input")
    if x.dtype not in (torch.float16, torch.float32):
        raise TypeError("cumsum_npu supports float16 and float32 inputs only")
    if x.ndim == 0:
        raise ValueError("cumsum_npu requires at least one dimension")

    dim = dim if dim >= 0 else dim + x.ndim
    if dim < 0 or dim >= x.ndim:
        raise IndexError(f"dim={dim} is out of range for ndim={x.ndim}")

    x_work = x.movedim(dim, -1).contiguous()
    outer, width = x_work.numel() // x_work.shape[-1], x_work.shape[-1]
    if outer == 0:
        return torch.empty_like(x)

    x_2d = x_work.reshape(outer, width)
    y_2d = torch.empty_like(x_2d)
    carry = torch.zeros((outer, ), device=x.device, dtype=torch.float32)
    next_carry = torch.empty_like(carry)
    stride_x0, stride_x1 = x_2d.stride()
    stride_y0, stride_y1 = y_2d.stride()
    block_m = 32

    for chunk_start in range(0, width, 16):
        _rowwise_cumsum_kernel[(triton.cdiv(outer, block_m), )](
            x_2d,
            y_2d,
            carry,
            next_carry,
            outer,
            width,
            chunk_start,
            stride_x0,
            stride_x1,
            stride_y0,
            stride_y1,
            BLOCK_M=block_m,
            BLOCK_N=16,
            num_warps=4,
            num_stages=1,
        )
        carry, next_carry = next_carry, carry
    return y_2d.reshape(x_work.shape).movedim(-1, dim)


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
        return cumsum_npu(x, dim=self.dim)


batch_size = 32768
input_shape = (32768, )
dim = 1


def get_inputs():
    """
    Generates random inputs for testing the Scan model.

    Returns:
        list: A list containing a single randomly generated tensor with shape
              (batch_size, *input_shape).
    """
    return [torch.rand(batch_size, *input_shape)]


def get_init_inputs():
    """
    Returns the initialization parameters for the Scan model.

    Returns:
        list: A list containing the `dim` parameter for model initialization.
    """
    return [dim]
