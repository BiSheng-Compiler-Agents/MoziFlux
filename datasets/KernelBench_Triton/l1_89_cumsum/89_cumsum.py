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
    N,
    chunk_start,
    stride_x0,
    stride_x1,
    stride_y0,
    stride_y1,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    x_row_base = x_ptr + row * stride_x0
    y_row_base = y_ptr + row * stride_y0
    carry = tl.load(carry_in_ptr + row)

    for offset in tl.static_range(0, BLOCK_N):
        col = chunk_start + offset
        mask = col < N
        value = tl.load(x_row_base + col * stride_x1, mask=mask, other=0.0)
        carry += value.to(tl.float32)
        tl.store(y_row_base + col * stride_y1, carry, mask=mask)

    tl.store(carry_out_ptr + row, carry)


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

    for chunk_start in range(0, width, 256):
        _rowwise_cumsum_kernel[(outer, )](
            x_2d,
            y_2d,
            carry,
            next_carry,
            width,
            chunk_start,
            stride_x0,
            stride_x1,
            stride_y0,
            stride_y1,
            BLOCK_N=256,
            num_warps=8,
            num_stages=2,
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
