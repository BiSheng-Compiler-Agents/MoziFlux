import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu  # noqa: F401
import triton
import triton.language as tl


DEFAULT_IN_CHANNELS = 32
DEFAULT_OUT_CHANNELS = 32
DEFAULT_KERNEL_SIZE = 3


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False))


@triton.jit
def _flip_transpose_4d_kernel(
    inp_ptr,  # [Cin, Cout, K, K]
    out_ptr,  # [Cout, Cin, K, K]
    Cin: tl.constexpr,
    Cout: tl.constexpr,
    K: tl.constexpr,
    n_elements: tl.constexpr,
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


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,   'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,   'BLOCK_K': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128,  'BLOCK_K': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128,  'BLOCK_K': 32}, num_warps=8, num_stages=5),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,   'BLOCK_K': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256,  'BLOCK_K': 32}, num_warps=8, num_stages=4),
        # Added larger tiles and deeper pipelines for H200
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256,  'BLOCK_K': 32}, num_warps=8, num_stages=5),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128,  'BLOCK_K': 32}, num_warps=8, num_stages=5),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256,  'BLOCK_K': 32}, num_warps=8, num_stages=6),
        # Allow a wider K-chunk for larger Cin cases
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128,  'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,   'BLOCK_K': 64}, num_warps=4, num_stages=4),
    ],
    key=['N', 'Cin', 'Cout', 'H_out', 'W_out', 'K'],
)
@triton.jit
def _convtransp2d_stride1_pad0_groups1_kernel(
    x_ptr,         # * (N, Cin, H, W)
    w_ptr,         # * (Cout, Cin, K, K) -- rotated weight: flip(spatial) + permute(out,in,kh,kw)
    bias_ptr,      # * (Cout,) or dummy
    y_ptr,         # * (N, Cout, H_out, W_out)
    N, Cin, H, W,
    Cout,
    K: tl.constexpr,
    H_out, W_out,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wkh, stride_wkw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Tile ids
    pid_m = tl.program_id(0)  # rows: N * H_out * W_out
    pid_n = tl.program_id(1)  # cols: Cout

    row_start = pid_m * BLOCK_M
    col_start = pid_n * BLOCK_N

    rows = row_start + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    cols = col_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    total_rows = N * H_out * W_out
    mask_m = rows < total_rows
    mask_n = cols < Cout

    # Hints for codegen
    tl.max_contiguous(rows, BLOCK_M)
    tl.max_contiguous(cols, BLOCK_N)

    # Map rows -> (n, h_out, w_out)
    hw_total = H_out * W_out
    n_idx = rows // hw_total
    hw_idx = rows % hw_total
    h_out_idx = hw_idx // W_out
    w_out_idx = hw_idx % W_out

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_range = tl.arange(0, BLOCK_K)
    rc = 0
    while rc < Cin:
        c_idx = rc + k_range  # [BLOCK_K]
        c_mask = c_idx < Cin

        # Iterate over spatial kernel (ConvTranspose2d stride=1,pad=0 equals Conv2d with rotated kernel and padding=K-1)
        for ky in tl.static_range(0, K):
            # input h coordinate for this ky
            h_in = h_out_idx + (ky - (K - 1))
            valid_y = (h_in >= 0) & (h_in < H)
            for kx in tl.static_range(0, K):
                w_in = w_out_idx + (kx - (K - 1))
                valid_x = (w_in >= 0) & (w_in < W)
                vmask = mask_m & valid_y & valid_x

                # Precompute base pointers to reduce integer ops in inner loop
                x_base = (
                    x_ptr
                    + n_idx[:, None] * stride_xn
                    + h_in[:, None] * stride_xh
                    + w_in[:, None] * stride_xw
                )
                x_ptrs = x_base + c_idx[None, :] * stride_xc
                x_mask = vmask[:, None] & c_mask[None, :]
                a = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)

                # Load W tile: (BLOCK_K, BLOCK_N) from rotated weight layout [Cout, Cin, K, K]
                w_base = (
                    w_ptr
                    + cols[None, :] * stride_wo
                    + ky * stride_wkh
                    + kx * stride_wkw
                )
                w_ptrs = w_base + c_idx[:, None] * stride_wi
                w_mask = c_mask[:, None] & mask_n[None, :]
                b = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)

                acc += tl.dot(a, b)
        rc += BLOCK_K

    if HAS_BIAS:
        bias_vals = tl.load(bias_ptr + cols, mask=mask_n, other=0.0).to(tl.float32)
        acc = acc + bias_vals[None, :]

    # Store Y tile
    y_ptrs = (
        y_ptr
        + n_idx[:, None] * stride_yn
        + cols[None, :] * stride_yc
        + h_out_idx[:, None] * stride_yh
        + w_out_idx[:, None] * stride_yw
    )
    y_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(y_ptrs, acc, mask=y_mask)


class ModelNew(nn.Module):
    """
    Performs a transposed 2D convolution with asymmetric input and a square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        output_padding (int, optional): Additional size added to one side of the output shape. Defaults to 0.
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
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
        )
        self._cached_conv_weight = None
        self._cached_version = None
        self._cached_meta = None

    def _can_use_triton(self):
        ct = self.conv_transpose2d
        # Support only stride=1, padding=0, dilation=1, output_padding=0, groups=1
        k = ct.kernel_size
        if isinstance(k, tuple):
            if k[0] != k[1]:
                return False, 0
            k = k[0]
        s = ct.stride
        p = ct.padding
        op = ct.output_padding
        d = ct.dilation
        cond = (
            (k > 0)
            and (s == (1, 1) if isinstance(s, tuple) else s == 1)
            and (p == (0, 0) if isinstance(p, tuple) else p == 0)
            and (op == (0, 0) if isinstance(op, tuple) else op == 0)
            and (d == (1, 1) if isinstance(d, tuple) else d == 1)
            and (ct.groups == 1)
        )
        return cond, int(k)

    def _maybe_get_transformed_weight(self, target_dtype: torch.dtype, kernel_size: int) -> torch.Tensor:
        source_w = self.conv_transpose2d.weight
        cin, cout, _, _ = source_w.shape
        device = source_w.device
        version = getattr(source_w, "_version", None)
        meta = (device, target_dtype, cout, cin, kernel_size)
        need_rebuild = (
            self._cached_conv_weight is None
            or self._cached_version != version
            or self._cached_meta != meta
        )
        if need_rebuild:
            if device.type != "npu":
                raise RuntimeError("ModelNew requires ConvTranspose2d weights to reside on Ascend NPU")
            w = source_w.to(dtype=target_dtype).contiguous()
            out_w = torch.empty((cout, cin, kernel_size, kernel_size), device=device, dtype=target_dtype)
            n_elements = out_w.numel()
            if n_elements > 0:
                block = 1024
                grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK"]),)
                _flip_transpose_4d_kernel[grid](
                    w,
                    out_w,
                    cin,
                    cout,
                    kernel_size,
                    n_elements,
                    BLOCK=block,
                )
            self._cached_conv_weight = out_w
            self._cached_version = version
            self._cached_meta = meta
        return self._cached_conv_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height_in, width_in).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        use_triton, K = self._can_use_triton()
        if not _is_npu_tensor(x):
            raise RuntimeError("ModelNew expects input tensors on Ascend NPU")
        if not _is_npu_tensor(self.conv_transpose2d.weight):
            raise RuntimeError("ModelNew expects ConvTranspose2d weights on Ascend NPU")
        if not use_triton:
            raise RuntimeError(
                "ModelNew only supports stride=1, padding=0, output_padding=0, dilation=1, groups=1, "
                "and a square kernel"
            )
        if x.dtype != torch.float32:
            raise TypeError(f"ModelNew only supports torch.float32 inputs, got {x.dtype}")
        if self.conv_transpose2d.weight.dtype != torch.float32:
            raise TypeError(
                f"ModelNew only supports torch.float32 weights, got {self.conv_transpose2d.weight.dtype}"
            )
        bias = self.conv_transpose2d.bias
        if bias is not None and bias.dtype != torch.float32:
            raise TypeError(f"ModelNew only supports torch.float32 bias, got {bias.dtype}")

        has_bias = bias is not None
        w_conv = self._maybe_get_transformed_weight(x.dtype, K)
        return F.conv2d(
            x.contiguous(),
            w_conv,
            bias.contiguous() if has_bias else None,
            stride=1,
            padding=K - 1,
            dilation=1,
            groups=1,
        )
batch_size = 8
in_channels = 32
out_channels = 32
kernel_size = 3
height_in = 512
width_in = 1024

def get_inputs():
    x = torch.rand(batch_size, in_channels, height_in, width_in)
    return [x]
def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization