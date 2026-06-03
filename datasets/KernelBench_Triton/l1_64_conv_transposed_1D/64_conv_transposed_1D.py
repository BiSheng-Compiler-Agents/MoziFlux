import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _deconv1d_stride1_kernel(
    x_ptr,         # *f32/f16/bf16 [B, C_IN, L_IN]
    w_ptr,         # *f32/f16/bf16 [C_IN, C_OUT, K]
    b_ptr,         # *f32[C_OUT] or dummy if no bias
    y_ptr,         # *dtype(x)[B, C_OUT, L_OUT]
    B,
    C_IN: tl.constexpr,
    C_OUT,
    L_IN,
    K: tl.constexpr,
    L_OUT,
    BLOCK_T: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)

    batch_idx = pid0 // C_OUT
    oc_idx = pid0 % C_OUT
    t_offsets = pid1 * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = t_offsets < L_OUT

    acc = tl.zeros((BLOCK_T,), dtype=tl.float32)

    x_batch_base = batch_idx * (C_IN * L_IN)
    y_base = batch_idx * (C_OUT * L_OUT) + oc_idx * L_OUT

    for cin_idx in tl.static_range(0, C_IN):
        x_base = x_batch_base + cin_idx * L_IN
        w_base = (cin_idx * C_OUT + oc_idx) * K
        for k_idx in tl.static_range(0, K):
            t_in = t_offsets - k_idx
            valid_t = (t_in >= 0) & (t_in < L_IN) & t_mask
            safe_t_in = tl.where(valid_t, t_in, 0)
            x_vals = tl.load(x_ptr + x_base + safe_t_in, mask=valid_t, other=0.0).to(tl.float32)
            w_val = tl.load(w_ptr + w_base + k_idx).to(tl.float32)
            acc += w_val * x_vals

    if HAS_BIAS:
        acc += tl.load(b_ptr + oc_idx).to(tl.float32)

    tl.store(y_ptr + y_base + t_offsets, acc, mask=t_mask)


class ModelNew(nn.Module):
    """
    Performs a transposed 1D convolution operation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        output_padding (int, optional): Additional size added to one side of the output shape. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 3,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.conv1d_transpose = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 1D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        mod = self.conv1d_transpose
        stride_ok = mod.stride == (1,) or mod.stride == 1
        padding_ok = mod.padding == (0,) or mod.padding == 0
        outpad_ok = mod.output_padding == (0,) or mod.output_padding == 0
        groups_ok = mod.groups == 1

        if x.device.type != "npu":
            raise RuntimeError("ModelNew expects NPU inputs and does not provide a non-NPU fallback.")
        if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise RuntimeError(f"Unsupported dtype for ModelNew: {x.dtype}.")
        if not (stride_ok and padding_ok and outpad_ok and groups_ok):
            raise RuntimeError(
                "ModelNew only supports stride=1, padding=0, output_padding=0, groups=1."
            )

        x_contig = x.contiguous()
        w = mod.weight.contiguous()  # [C_IN, C_OUT, K]
        b = mod.bias
        B, C_IN, L_IN = x_contig.shape
        _, C_OUT, K = w.shape

        # For stride=1, padding=0, output_padding=0: L_OUT = L_IN + K - 1
        L_OUT = L_IN + K - 1
        # Allocate output directly in input dtype to avoid an extra cast later
        y = torch.empty((B, C_OUT, L_OUT), device=x.device, dtype=x.dtype)

        BLOCK_T = 64

        grid = (
            B * C_OUT,
            triton.cdiv(L_OUT, BLOCK_T),
        )

        b_ptr = b.contiguous() if b is not None else y  # dummy if no bias
        _deconv1d_stride1_kernel[grid](
            x_contig,
            w,
            b_ptr,
            y,
            B,
            C_IN,
            C_OUT,
            L_IN,
            K,
            L_OUT,
            BLOCK_T=BLOCK_T,
            HAS_BIAS=(b is not None),
        )
        return y
batch_size = 64
in_channels = 128
out_channels = 128
kernel_size = 3
length = 65536

def get_inputs():
    x = torch.rand(batch_size, in_channels, length)
    return [x]
def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization