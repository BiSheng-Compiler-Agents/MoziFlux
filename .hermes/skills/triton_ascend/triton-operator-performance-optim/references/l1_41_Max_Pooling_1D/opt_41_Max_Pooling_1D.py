import torch
import torch.nn as nn
import triton
import triton.language as tl

DEFAULT_KERNEL_SIZE = 8
DEFAULT_STRIDE = 1
DEFAULT_PADDING = 4
DEFAULT_DILATION = 3
DEFAULT_RETURN_INDICES = False


@triton.jit
def _maxpool1d_forward_kernel(
        x_ptr,  # *T,  input [NC][L_in]
        y_ptr,  # *T,  output [NC][L_out]
        idx_ptr,  # *int64, indices [NC][L_out] (optional)
        L_in,  # int32
        L_out,  # int32
        STRIDE,  # int32
        PADDING,  # int32
        DILATION,  # int32
        line_stride_x,  # int32 = L_in
        line_stride_y,  # int32 = L_out
        HAS_INDEX: tl.constexpr,  # bool, whether to write indices
        K: tl.constexpr,  # kernel size (compile-time)
        BLOCK: tl.constexpr,  # tile size along output length
):
    pid_nc = tl.program_id(axis=0)
    pid_o_blk = tl.program_id(axis=1)

    o_offsets = pid_o_blk * BLOCK + tl.arange(0, BLOCK)
    mask_o = o_offsets < L_out
    starts = o_offsets * STRIDE - PADDING

    base_x = x_ptr + pid_nc * line_stride_x
    base_y = y_ptr + pid_nc * line_stride_y

    if not HAS_INDEX:
        k_offsets = tl.arange(0, K)
        pos = starts[None, :] + k_offsets[:, None] * DILATION
        valid = (pos >= 0) & (pos < L_in) & mask_o[None, :]
        addr = tl.minimum(tl.maximum(pos, 0), L_in - 1)
        values = tl.load(base_x + addr, mask=valid, other=-float("inf"))
        y_max = tl.max(values, axis=0)
    else:
        pos = starts
        valid0 = (pos >= 0) & (pos < L_in) & mask_o
        addr0 = tl.minimum(tl.maximum(pos, 0), L_in - 1)
        x0 = tl.load(base_x + addr0, mask=mask_o, other=0)
        y_max = tl.where(valid0, x0, -float("inf"))
        chosen_pos = pos

        for _ in tl.static_range(1, K):
            pos = pos + DILATION
            validk = (pos >= 0) & (pos < L_in) & mask_o
            addrk = tl.minimum(tl.maximum(pos, 0), L_in - 1)
            xk = tl.load(base_x + addrk, mask=mask_o, other=0)
            vk = tl.where(validk, xk, -float("inf"))
            better = vk > y_max
            y_max = tl.where(better, vk, y_max)
            chosen_pos = tl.where(better, pos, chosen_pos)

    tl.store(base_y + o_offsets, y_max, mask=mask_o)

    if HAS_INDEX:
        chosen_pos = tl.maximum(0, tl.minimum(chosen_pos, L_in - 1))
        base_i = idx_ptr + pid_nc * line_stride_y
        tl.store(base_i + o_offsets, chosen_pos.to(tl.int64), mask=mask_o)


class ModelNew(nn.Module):
    """
    Simple model that performs Max Pooling 1D.
    """

    def __init__(
        self,
        kernel_size: int = DEFAULT_KERNEL_SIZE,
        stride: int = DEFAULT_STRIDE,
        padding: int = DEFAULT_PADDING,
        dilation: int = DEFAULT_DILATION,
        return_indices: bool = DEFAULT_RETURN_INDICES,
    ):
        """
        Initializes the Max Pooling 1D layer.

        Args:
            kernel_size (int): Size of the window to take a max over.
            stride (int, optional): Stride of the window. Defaults to None (same as kernel_size).
            padding (int, optional): Implicit zero padding to be added on both sides. Defaults to 0.
            dilation (int, optional): Spacing between kernel elements. Defaults to 1.
            return_indices (bool, optional): Whether to return the indices of the maximum values. Defaults to False.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = int(kernel_size)
        self.stride = int(kernel_size if stride is None else stride)
        self.padding = int(padding)
        self.dilation = int(dilation)
        self.return_indices = bool(return_indices)

    def _out_length(self, L_in: int) -> int:
        # PyTorch formula with ceil_mode=False
        return (L_in + 2 * self.padding - self.dilation *
                (self.kernel_size - 1) - 1) // self.stride + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 1D to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, sequence_length).

        Returns:
            torch.Tensor: Output tensor with Max Pooling 1D applied, shape (batch_size, num_features, output_sequence_length).
        """
        if x.device.type != "npu":
            raise ValueError("ModelNew expects Ascend NPU inputs")
        if x.dtype not in (torch.float16, torch.float32, torch.bfloat16):
            raise TypeError(
                "ModelNew supports float16, float32, and bfloat16 inputs only")

        x = x.contiguous()
        N, C, L_in = x.shape
        L_out = self._out_length(L_in)

        if L_out <= 0:
            raise ValueError(
                "Invalid pooling configuration produces a non-positive output length"
            )

        y = torch.empty((N, C, L_out), device=x.device, dtype=x.dtype)
        indices = None
        if self.return_indices:
            indices = torch.empty((N, C, L_out),
                                  device=x.device,
                                  dtype=torch.int64)

        # Keep the original launch geometry and specialize the no-index hot path.
        NC = N * C
        BLOCK = 128
        grid = (NC, triton.cdiv(L_out, BLOCK))

        _maxpool1d_forward_kernel[grid](
            x,
            y,
            indices if self.return_indices else torch.empty(
                0, device=x.device, dtype=torch.int64),
            L_in,
            L_out,
            self.stride,
            self.padding,
            self.dilation,
            L_in,
            L_out,
            HAS_INDEX=self.return_indices,
            K=self.kernel_size,
            BLOCK=BLOCK,
            num_warps=4,
            num_stages=4,
        )

        if self.return_indices:
            return y, indices
        return y


def max_pool1d(
    x: torch.Tensor,
    kernel_size: int = DEFAULT_KERNEL_SIZE,
    stride: int = DEFAULT_STRIDE,
    padding: int = DEFAULT_PADDING,
    dilation: int = DEFAULT_DILATION,
    return_indices: bool = DEFAULT_RETURN_INDICES,
):
    return ModelNew(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        return_indices=return_indices,
    )(x)


batch_size = 64
features = 192
sequence_length = 65536
kernel_size = 8
stride = 1
padding = 4
dilation = 3
return_indices = False


def get_inputs():
    x = torch.rand(batch_size, features, sequence_length)
    return [x]


def get_init_inputs():
    return [kernel_size, stride, padding, dilation, return_indices]
