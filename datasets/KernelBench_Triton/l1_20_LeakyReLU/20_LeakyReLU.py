import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def _leaky_relu_kernel(x_ptr, y_ptr, n_elements, neg,
                       BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.multiple_of(offsets, 16)
    tl.max_contiguous(offsets, 16)

    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    zero = tl.zeros([BLOCK_SIZE], dtype=x.dtype)
    y = tl.where(x >= zero, x, x * neg)
    tl.store(y_ptr + offsets, y, mask=mask)


class ModelNew(nn.Module):
    """
    Simple model that performs a LeakyReLU activation.
    """

    def __init__(self, negative_slope: float = 0.01):
        """
        Initializes the LeakyReLU module.

        Args:
            negative_slope (float, optional): The negative slope of the activation function. Defaults to 0.01.
        """
        super(ModelNew, self).__init__()
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies LeakyReLU activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with LeakyReLU applied, same shape as input.
        """
        if x.device.type != "npu":
            raise ValueError("ModelNew expects an Ascend NPU tensor input")
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError(
                "ModelNew supports only float16, bfloat16, and float32 tensors"
            )
        if x.numel() == 0:
            return torch.empty_like(x)

        x_contig = x.contiguous()
        n_elements = x.numel()
        BLOCK_SIZE = 4096
        grid = (triton.cdiv(n_elements, BLOCK_SIZE), )
        y = torch.empty_like(x_contig)
        _leaky_relu_kernel[grid](
            x_contig.view(-1),
            y.view(-1),
            n_elements,
            float(self.negative_slope),
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
            num_stages=1,
        )
        return y.view_as(x)


batch_size = 4096
dim = 393216


def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]


def get_init_inputs():
    return []  # No special initialization inputs needed
