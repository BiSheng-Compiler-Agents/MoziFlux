import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu  # noqa: F401
import triton
import triton.language as tl

DEFAULT_IN_CHANNELS = 48
DEFAULT_OUT_CHANNELS = 24
DEFAULT_KERNEL_SIZE = 3


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False))


@triton.jit
def _flip_transpose_5d(
    inp_ptr,  # [Cin, Cout, Kd, Kh, Kw]
    out_ptr,  # [Cout, Cin, Kd, Kh, Kw]
    Cin: tl.constexpr,
    Cout: tl.constexpr,
    Kd: tl.constexpr,
    Kh: tl.constexpr,
    Kw: tl.constexpr,
    n_elements: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * BLOCK
    offs = base + tl.arange(0, BLOCK)
    mask = offs < n_elements

    # Out tensor strides for [Cout, Cin, Kd, Kh, Kw]
    stride_out_kh = Kw
    stride_out_kd = Kh * Kw
    stride_out_ci = Kd * stride_out_kd
    stride_out_co = Cin * stride_out_ci

    co = offs // stride_out_co
    rem = offs - co * stride_out_co
    ci = rem // stride_out_ci
    rem = rem - ci * stride_out_ci
    kz = rem // stride_out_kd
    rem = rem - kz * stride_out_kd
    ky = rem // stride_out_kh
    kx = rem - ky * stride_out_kh

    in_kz = Kd - 1 - kz
    in_ky = Kh - 1 - ky
    in_kx = Kw - 1 - kx

    # In tensor strides for [Cin, Cout, Kd, Kh, Kw]
    stride_in_kw = 1
    stride_in_kh = Kw
    stride_in_kd = Kh * Kw
    stride_in_co = Kd * stride_in_kd
    stride_in_ci = Cout * stride_in_co

    in_idx = (ci * stride_in_ci + co * stride_in_co + in_kz * stride_in_kd +
              in_ky * stride_in_kh + in_kx * stride_in_kw)

    vals = tl.load(inp_ptr + in_idx, mask=mask, other=0)
    tl.store(out_ptr + offs, vals, mask=mask)


class ModelNew(nn.Module):
    """
    Performs a transposed 3D convolution operation with asymmetric input and a square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int or tuple, optional): Stride of the convolution. Defaults to 1.
        padding (int or tuple, optional): Padding applied to the input. Defaults to 0.
        output_padding (int or tuple, optional): Additional size added to one side of each dimension in the output shape.
                                                  Defaults to 0.
        dilation (int or tuple, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """

    def __init__(
        self,
        in_channels: int = DEFAULT_IN_CHANNELS,
        out_channels: int = DEFAULT_OUT_CHANNELS,
        kernel_size: int = DEFAULT_KERNEL_SIZE,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(
            in_channels,
            out_channels, (kernel_size, kernel_size, kernel_size),
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

        # Cache for transformed weights to amortize cost across forwards
        self._cached_conv_weight = None  # [Cout, Cin, Kd, Kh, Kw]
        self._cached_version = None
        self._cached_meta = None  # (device, dtype, shape)
        # Precompute fast-path padding (k-1, k-1, k-1) for square kernel
        ks = self.conv_transpose3d.kernel_size
        self._fastpad = (ks[0] - 1, ks[1] - 1, ks[2] - 1)

    def _maybe_get_transformed_weight(self, target_dtype: torch.dtype):
        # Transform weight from [Cin, Cout, Kd, Kh, Kw] to flipped [Cout, Cin, Kd, Kh, Kw]
        source_w = self.conv_transpose3d.weight
        Cin, Cout, Kd, Kh, Kw = source_w.shape
        device = source_w.device
        version = getattr(source_w, "_version", None)
        need_rebuild = (self._cached_conv_weight is None
                        or self._cached_meta != (device, target_dtype,
                                                 (Cout, Cin, Kd, Kh, Kw))
                        or self._cached_version != version)
        if need_rebuild:
            w = source_w.to(dtype=target_dtype).contiguous()
            if device.type != "npu":
                raise RuntimeError(
                    "ModelNew requires ConvTranspose3d weights to reside on Ascend NPU"
                )
            out_w = torch.empty((Cout, Cin, Kd, Kh, Kw),
                                device=device,
                                dtype=target_dtype)
            n_elements = out_w.numel()
            if n_elements > 0:
                BLOCK = 2048

                def grid(META):
                    return (triton.cdiv(n_elements, BLOCK), )

                _flip_transpose_5d[grid](w,
                                         out_w,
                                         Cin,
                                         Cout,
                                         Kd,
                                         Kh,
                                         Kw,
                                         n_elements,
                                         BLOCK=BLOCK)
            self._cached_conv_weight = out_w
            self._cached_version = version
            self._cached_meta = (device, target_dtype, (Cout, Cin, Kd, Kh, Kw))
        return self._cached_conv_weight

    def _fastpath_supported(self):
        ct = self.conv_transpose3d
        return (isinstance(ct.stride, tuple) and ct.stride == (1, 1, 1)
                and isinstance(ct.padding, tuple) and ct.padding == (0, 0, 0)
                and isinstance(ct.dilation, tuple) and ct.dilation == (1, 1, 1)
                and isinstance(ct.output_padding, tuple)
                and ct.output_padding == (0, 0, 0) and ct.groups == 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        if not _is_npu_tensor(x):
            raise RuntimeError("ModelNew expects input tensors on Ascend NPU")
        if not self._fastpath_supported():
            raise RuntimeError(
                "ModelNew only supports stride=1, padding=0, dilation=1, output_padding=0, and groups=1"
            )
        if not x.is_contiguous():
            x = x.contiguous()

        # Fast path using equivalence:
        # ConvTranspose3d(x, W, stride=1, padding=0, dilation=1, groups=1)
        # == Conv3d(x, W_T_flipped, padding=K-1)
        w_conv = self._maybe_get_transformed_weight(x.dtype)
        bias = self.conv_transpose3d.bias
        if bias is not None:
            bias = bias.to(dtype=x.dtype)
        return F.conv3d(
            x,
            w_conv,
            bias=bias,
            stride=1,
            padding=self._fastpad,
            dilation=1,
            groups=1,
        )


batch_size = 8
in_channels = 48
out_channels = 24
kernel_size = 3
depth = 96
height = 96
width = 96


def get_inputs():
    x = torch.rand(batch_size, in_channels, depth, height, width, device='npu')
    return [x]


def get_init_inputs():
    return [
        in_channels, out_channels, kernel_size
    ]  # Provide in_channels, out_channels, kernel_size for initialization
