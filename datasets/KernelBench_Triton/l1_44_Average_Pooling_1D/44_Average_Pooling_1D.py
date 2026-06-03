import math
import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl


DEFAULT_KERNEL_SIZE = 8
DEFAULT_STRIDE = 1
DEFAULT_PADDING = 4


@triton.jit
def avgpool1d_forward_kernel(
    x_ptr,  # *[N_ROWS, L_IN]
    y_ptr,  # *[N_ROWS, L_OUT]
    N_ROWS: tl.constexpr,
    L_IN: tl.constexpr,
    L_OUT: tl.constexpr,
    N_COL_BLOCKS: tl.constexpr,
    stride_x_row: tl.constexpr,
    stride_y_row: tl.constexpr,
    STRIDE: tl.int32,
    PADDING: tl.int32,
    KERNEL_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(axis=0)
    if row_id >= N_ROWS:
        return

    # Row base pointers
    x_row_ptr = x_ptr + row_id * stride_x_row
    y_row_ptr = y_ptr + row_id * stride_y_row

    invK = 1.0 / float(KERNEL_SIZE)
    block_offsets = tl.arange(0, BLOCK)
    for col_block in tl.range(0, N_COL_BLOCKS):
        offs = col_block * BLOCK + block_offsets
        mask_o = offs < L_OUT
        j = offs * STRIDE - PADDING
        acc = tl.zeros([BLOCK], dtype=tl.float32)

        # Keep clamped addresses for safety while using L2-friendly cache modifier.
        for k in tl.static_range(0, KERNEL_SIZE):
            pos = j + k
            in_bounds = (pos >= 0) & (pos < L_IN)
            mask_k = in_bounds & mask_o
            pos_safe = tl.maximum(tl.minimum(pos, L_IN - 1), 0)
            vals_k = tl.load(
                x_row_ptr + pos_safe,
                mask=mask_k,
                other=0.0,
                cache_modifier=".cg",
            )
            acc += vals_k.to(tl.float32)

        out = acc * invK
        tl.store(y_row_ptr + offs, out, mask=mask_o)


class ModelNew(nn.Module):
    """
    Simple model that performs 1D Average Pooling using a Triton kernel.
    Semantics match nn.AvgPool1d with count_include_pad=True and ceil_mode=False.
    """
    def __init__(
        self,
        kernel_size: int = DEFAULT_KERNEL_SIZE,
        stride: int = DEFAULT_STRIDE,
        padding: int = DEFAULT_PADDING,
    ):
        super(ModelNew, self).__init__()
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type != "npu":
            raise RuntimeError("ModelNew expects inputs on Ascend NPU")
        if x.dtype not in (torch.float16, torch.float32, torch.bfloat16):
            raise RuntimeError(
                f"Unsupported dtype for ModelNew: {x.dtype}. "
                "Supported dtypes are float16, float32, and bfloat16."
            )
        if x.ndim != 3:
            raise RuntimeError(
                f"ModelNew expects a 3D tensor shaped [batch, channels, length], got {tuple(x.shape)}"
            )
        if x.requires_grad:
            raise RuntimeError("ModelNew does not support autograd-tracked inputs")
        if self.kernel_size <= 0 or self.stride <= 0 or self.padding < 0:
            raise RuntimeError("kernel_size and stride must be positive, and padding must be non-negative")

        # Ensure contiguous memory for predictable strides
        x = x.contiguous()
        B, C, L_in = x.shape

        # Output length following PyTorch formula (ceil_mode=False)
        L_out = (L_in + 2 * self.padding - self.kernel_size) // self.stride + 1
        if L_out <= 0:
            return x.new_empty((B, C, 0))

        # Allocate output
        y = torch.empty((B, C, L_out), device=x.device, dtype=x.dtype)

        # Flatten batch and channels into rows for kernel
        x2 = x.view(B * C, L_in)
        y2 = y.view(B * C, L_out)

        N_ROWS = B * C
        # Choose a reasonable block size; favor fewer CTAs to reduce overhead while keeping occupancy
        if L_out >= 2048:
            BLOCK = 256
            num_warps = 8
        else:
            BLOCK = 128
            num_warps = 4

        n_col_blocks = triton.cdiv(L_out, BLOCK)
        grid = (N_ROWS,)
        avgpool1d_forward_kernel[grid](
            x2,
            y2,
            N_ROWS,
            L_in,
            L_out,
            n_col_blocks,
            x2.stride(0),  # stride over rows in elements
            y2.stride(0),
            self.stride,
            self.padding,
            KERNEL_SIZE=self.kernel_size,
            BLOCK=BLOCK,
            num_warps=num_warps,
            num_stages=4,
        )

        return y


def avg_pool1d(
    x: torch.Tensor,
    kernel_size: int = DEFAULT_KERNEL_SIZE,
    stride: int = DEFAULT_STRIDE,
    padding: int = DEFAULT_PADDING,
) -> torch.Tensor:
    return ModelNew(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )(x)
batch_size = 64
in_channels = 128
input_length = 65536
kernel_size = 8
stride = 1
padding = 4

def get_inputs():
    x = torch.rand(batch_size, in_channels, input_length)
    return [x]
def get_init_inputs():
    return [kernel_size, stride, padding]
