import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu  # noqa: F401
import triton
import triton.language as tl


DEFAULT_IN_CHANNELS = 32
DEFAULT_OUT_CHANNELS = 64
DEFAULT_KERNEL_SIZE = 3
DEFAULT_STRIDE = 5
DEFAULT_PADDING = 1
DEFAULT_DILATION = 2


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False))


def _normalize_square_param(value, name: str) -> int:
    if isinstance(value, tuple):
        if len(value) != 2 or value[0] != value[1]:
            raise RuntimeError(f"ModelNew only supports symmetric {name} values, got {value}")
        value = value[0]
    value = int(value)
    if value <= 0 and name in {"kernel_size", "stride", "dilation"}:
        raise RuntimeError(f"ModelNew requires positive {name}, got {value}")
    if value < 0 and name == "padding":
        raise RuntimeError(f"ModelNew requires non-negative padding, got {value}")
    return value


@triton.jit
def _flip_transpose_4d_kernel(
    inp_ptr,
    out_ptr,
    Cin: tl.constexpr,
    Cout: tl.constexpr,
    K: tl.constexpr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * BLOCK
    offs = base + tl.arange(0, BLOCK)
    mask = offs < n_elements

    stride_out_kw = 1
    stride_out_kh = K
    stride_out_ci = K * K
    stride_out_co = Cin * stride_out_ci

    co = offs // stride_out_co
    rem = offs - co * stride_out_co
    ci = rem // stride_out_ci
    rem = rem - ci * stride_out_ci
    ky = rem // stride_out_kh
    kx = rem - ky * stride_out_kh

    in_ky = K - 1 - ky
    in_kx = K - 1 - kx

    stride_in_kw = 1
    stride_in_kh = K
    stride_in_co = K * K
    stride_in_ci = Cout * stride_in_co

    in_idx = (
        ci * stride_in_ci
        + co * stride_in_co
        + in_ky * stride_in_kh
        + in_kx * stride_in_kw
    )
    vals = tl.load(inp_ptr + in_idx, mask=mask, other=0.0)
    tl.store(out_ptr + offs, vals, mask=mask)


@triton.jit
def _stride_insert_zeros_2d_kernel(
    inp_ptr,
    out_ptr,
    N,
    C,
    H,
    W,
    stride_in_n,
    stride_in_c,
    stride_in_h,
    stride_in_w,
    stride_out_n,
    stride_out_c,
    stride_out_h,
    stride_out_w,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    h = tl.program_id(1)

    if pid_nc >= N * C or h >= H:
        return

    n = pid_nc // C
    c = pid_nc % C

    in_base = n * stride_in_n + c * stride_in_c
    out_base = n * stride_out_n + c * stride_out_c

    w_start = 0
    while w_start < W:
        w = w_start + tl.arange(0, BLOCK_W)
        mask = w < W

        vals = tl.load(
            inp_ptr + in_base + h * stride_in_h + w * stride_in_w,
            mask=mask,
            other=0.0,
        )
        tl.store(
            out_ptr + out_base + (h * STRIDE_H) * stride_out_h + (w * STRIDE_W) * stride_out_w,
            vals,
            mask=mask,
        )
        w_start += BLOCK_W


class ModelNew(nn.Module):
    """
    Performs a 2D transposed convolution with asymmetric input, square kernel,
    dilation, padding, and stride via an Ascend Triton-assisted path.
    """

    def __init__(
        self,
        in_channels: int = DEFAULT_IN_CHANNELS,
        out_channels: int = DEFAULT_OUT_CHANNELS,
        kernel_size: int = DEFAULT_KERNEL_SIZE,
        stride: int = DEFAULT_STRIDE,
        padding: int = DEFAULT_PADDING,
        dilation: int = DEFAULT_DILATION,
        bias: bool = False,
    ):
        super().__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self._cached_conv_weight = None
        self._cached_version = None
        self._cached_meta = None

    def _resolve_runtime_params(self):
        ct = self.conv_transpose2d
        kernel_size = _normalize_square_param(ct.kernel_size, "kernel_size")
        stride = _normalize_square_param(ct.stride, "stride")
        padding = _normalize_square_param(ct.padding, "padding")
        dilation = _normalize_square_param(ct.dilation, "dilation")
        output_padding = _normalize_square_param(ct.output_padding, "padding")
        if output_padding != 0:
            raise RuntimeError("ModelNew only supports output_padding=0")
        if ct.groups != 1:
            raise RuntimeError("ModelNew only supports groups=1")
        effective_padding = dilation * (kernel_size - 1) - padding
        if effective_padding < 0:
            raise RuntimeError(
                "ModelNew only supports configurations with dilation * (kernel_size - 1) >= padding"
            )
        return kernel_size, stride, padding, dilation, effective_padding

    def _maybe_get_transformed_weight(self, target_dtype: torch.dtype, kernel_size: int) -> torch.Tensor:
        source_w = self.conv_transpose2d.weight
        cin, cout, _, _ = source_w.shape
        device = source_w.device
        version = getattr(source_w, "_version", None)
        meta = (device, target_dtype, cin, cout, kernel_size)
        rebuild = (
            self._cached_conv_weight is None
            or self._cached_version != version
            or self._cached_meta != meta
        )
        if rebuild:
            if device.type != "npu":
                raise RuntimeError("ModelNew requires ConvTranspose2d weights on Ascend NPU")
            weight = source_w.to(dtype=target_dtype).contiguous()
            flipped = torch.empty((cout, cin, kernel_size, kernel_size), device=device, dtype=target_dtype)
            n_elements = flipped.numel()
            if n_elements > 0:
                block = 1024
                grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK"]),)
                _flip_transpose_4d_kernel[grid](
                    weight,
                    flipped,
                    cin,
                    cout,
                    kernel_size,
                    n_elements,
                    BLOCK=block,
                )
            self._cached_conv_weight = flipped
            self._cached_version = version
            self._cached_meta = meta
        return self._cached_conv_weight

    def _upsample_input(self, x: torch.Tensor, stride: int) -> torch.Tensor:
        n, c, h, w = x.shape
        out_h = (h - 1) * stride + 1
        out_w = (w - 1) * stride + 1
        expanded = torch.zeros((n, c, out_h, out_w), device=x.device, dtype=x.dtype)
        if n > 0 and c > 0 and h > 0 and w > 0:
            BLOCK_W = 256
            grid = (n * c, h)
            _stride_insert_zeros_2d_kernel[grid](
                x,
                expanded,
                n,
                c,
                h,
                w,
                x.stride(0),
                x.stride(1),
                x.stride(2),
                x.stride(3),
                expanded.stride(0),
                expanded.stride(1),
                expanded.stride(2),
                expanded.stride(3),
                STRIDE_H=stride,
                STRIDE_W=stride,
                BLOCK_W=BLOCK_W,
            )
        return expanded

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise RuntimeError(f"ModelNew expects a 4D input tensor, got shape {tuple(x.shape)}")
        if not _is_npu_tensor(x):
            raise RuntimeError("ModelNew expects input tensors on Ascend NPU")
        if not _is_npu_tensor(self.conv_transpose2d.weight):
            raise RuntimeError("ModelNew expects ConvTranspose2d weights on Ascend NPU")
        if x.dtype not in (torch.float16, torch.float32):
            raise TypeError(f"ModelNew only supports float16/float32 inputs, got {x.dtype}")

        kernel_size, stride, _padding, dilation, effective_padding = self._resolve_runtime_params()
        x = x.contiguous()
        flipped_weight = self._maybe_get_transformed_weight(x.dtype, kernel_size)
        expanded = self._upsample_input(x, stride)
        bias = self.conv_transpose2d.bias
        bias_term = None if bias is None else bias.to(device=x.device, dtype=x.dtype).contiguous()
        return F.conv2d(
            expanded,
            flipped_weight,
            bias_term,
            stride=1,
            padding=effective_padding,
            dilation=dilation,
            groups=1,
        )
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = 3
height_in = 64
width_in = 128
stride = 5
padding = 1
dilation = 2

def get_inputs():
    x = torch.rand(batch_size, in_channels, height_in, width_in)
    return [x]
def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]
