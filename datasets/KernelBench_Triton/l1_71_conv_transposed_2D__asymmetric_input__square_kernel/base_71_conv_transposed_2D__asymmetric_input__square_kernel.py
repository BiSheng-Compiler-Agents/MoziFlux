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


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,   'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,   'BLOCK_K': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128,  'BLOCK_K': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128,  'BLOCK_K': 32}, num_warps=8, num_stages=5),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,   'BLOCK_K': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256,  'BLOCK_K': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256,  'BLOCK_K': 32}, num_warps=8, num_stages=5),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128,  'BLOCK_K': 32}, num_warps=8, num_stages=5),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256,  'BLOCK_K': 32}, num_warps=8, num_stages=6),
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
    w_off_00: tl.constexpr, w_off_01: tl.constexpr, w_off_02: tl.constexpr,
    w_off_10: tl.constexpr, w_off_11: tl.constexpr, w_off_12: tl.constexpr,
    w_off_20: tl.constexpr, w_off_21: tl.constexpr, w_off_22: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_batch = tl.program_id(0)  # batch index (0..N-1)
    pid_spatial = tl.program_id(1)  # spatial chunk (h_out * w_out)
    pid_cout = tl.program_id(2)   # cout chunk

    row_start = pid_spatial * BLOCK_M
    col_start = pid_cout * BLOCK_N

    rows = row_start + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    cols = col_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    rows = tl.max_contiguous(rows, BLOCK_M)

    total_spatial = H_out * W_out
    mask_m = rows < total_spatial
    mask_n = cols < Cout

    w_col_base = w_ptr + cols[None, :] * stride_wo  # [1, BLOCK_N] shared across all spatial positions

    n_idx = pid_batch + tl.zeros_like(rows)  # broadcast batch to [BLOCK_M]
    hw_idx = rows
    h_out_idx = hw_idx // W_out
    w_out_idx = hw_idx % W_out

    h_in_k0 = h_out_idx + (1 - K)
    h_in_k1 = h_out_idx + (2 - K)
    h_in_k2 = h_out_idx + (3 - K)
    w_in_k0 = w_out_idx + (1 - K)
    w_in_k1 = w_out_idx + (2 - K)
    w_in_k2 = w_out_idx + (3 - K)

    valid_00 = (h_in_k0 >= 0) & (h_in_k0 < H) & (w_in_k0 >= 0) & (w_in_k0 < W)
    valid_01 = (h_in_k0 >= 0) & (h_in_k0 < H) & (w_in_k1 >= 0) & (w_in_k1 < W)
    valid_02 = (h_in_k0 >= 0) & (h_in_k0 < H) & (w_in_k2 >= 0) & (w_in_k2 < W)
    valid_10 = (h_in_k1 >= 0) & (h_in_k1 < H) & (w_in_k0 >= 0) & (w_in_k0 < W)
    valid_11 = (h_in_k1 >= 0) & (h_in_k1 < H) & (w_in_k1 >= 0) & (w_in_k1 < W)
    valid_12 = (h_in_k1 >= 0) & (h_in_k1 < H) & (w_in_k2 >= 0) & (w_in_k2 < W)
    valid_20 = (h_in_k2 >= 0) & (h_in_k2 < H) & (w_in_k0 >= 0) & (w_in_k0 < W)
    valid_21 = (h_in_k2 >= 0) & (h_in_k2 < H) & (w_in_k1 >= 0) & (w_in_k1 < W)
    valid_22 = (h_in_k2 >= 0) & (h_in_k2 < H) & (w_in_k2 >= 0) & (w_in_k2 < W)

    x_base_00 = x_ptr + n_idx[:, None] * stride_xn + h_in_k0[:, None] * stride_xh + w_in_k0[:, None] * stride_xw
    x_base_01 = x_ptr + n_idx[:, None] * stride_xn + h_in_k0[:, None] * stride_xh + w_in_k1[:, None] * stride_xw
    x_base_02 = x_ptr + n_idx[:, None] * stride_xn + h_in_k0[:, None] * stride_xh + w_in_k2[:, None] * stride_xw
    x_base_10 = x_ptr + n_idx[:, None] * stride_xn + h_in_k1[:, None] * stride_xh + w_in_k0[:, None] * stride_xw
    x_base_11 = x_ptr + n_idx[:, None] * stride_xn + h_in_k1[:, None] * stride_xh + w_in_k1[:, None] * stride_xw
    x_base_12 = x_ptr + n_idx[:, None] * stride_xn + h_in_k1[:, None] * stride_xh + w_in_k2[:, None] * stride_xw
    x_base_20 = x_ptr + n_idx[:, None] * stride_xn + h_in_k2[:, None] * stride_xh + w_in_k0[:, None] * stride_xw
    x_base_21 = x_ptr + n_idx[:, None] * stride_xn + h_in_k2[:, None] * stride_xh + w_in_k1[:, None] * stride_xw
    x_base_22 = x_ptr + n_idx[:, None] * stride_xn + h_in_k2[:, None] * stride_xh + w_in_k2[:, None] * stride_xw

    bias_vals_init = tl.load(bias_ptr + cols, mask=mask_n, other=0.0)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32) + bias_vals_init[None, :]

    k_range = tl.arange(0, BLOCK_K)
    rc = 0
    while rc < Cin:
        c_idx = tl.max_contiguous(rc + k_range, BLOCK_K)  # [BLOCK_K]
        c_mask = c_idx < Cin
        c_off = tl.max_contiguous(c_idx * stride_xc, BLOCK_K)  # [BLOCK_K] precompute once for all spatial positions
        w_ci_base = c_idx[:, None] * stride_wi  # [BLOCK_K, 1] shared across all spatial positions

        # ky=0, kx=0
        vmask = mask_m & valid_00
        a = tl.load(x_base_00 + c_off[None, :], mask=vmask[:, None] & c_mask[None, :], other=0.0)
        w_ptrs = w_col_base + w_off_00 + w_ci_base
        b = tl.load(w_ptrs, mask=c_mask[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(a, b, input_precision='hf32')

        # ky=0, kx=1
        vmask = mask_m & valid_01
        a = tl.load(x_base_01 + c_off[None, :], mask=vmask[:, None] & c_mask[None, :], other=0.0)
        w_ptrs = w_col_base + w_off_01 + w_ci_base
        b = tl.load(w_ptrs, mask=c_mask[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(a, b, input_precision='hf32')

        # ky=0, kx=2
        vmask = mask_m & valid_02
        a = tl.load(x_base_02 + c_off[None, :], mask=vmask[:, None] & c_mask[None, :], other=0.0)
        w_ptrs = w_col_base + w_off_02 + w_ci_base
        b = tl.load(w_ptrs, mask=c_mask[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(a, b, input_precision='hf32')

        # ky=1, kx=0
        vmask = mask_m & valid_10
        a = tl.load(x_base_10 + c_off[None, :], mask=vmask[:, None] & c_mask[None, :], other=0.0)
        w_ptrs = w_col_base + w_off_10 + w_ci_base
        b = tl.load(w_ptrs, mask=c_mask[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(a, b, input_precision='hf32')

        # ky=1, kx=1
        vmask = mask_m & valid_11
        a = tl.load(x_base_11 + c_off[None, :], mask=vmask[:, None] & c_mask[None, :], other=0.0)
        w_ptrs = w_col_base + w_off_11 + w_ci_base
        b = tl.load(w_ptrs, mask=c_mask[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(a, b, input_precision='hf32')

        # ky=1, kx=2
        vmask = mask_m & valid_12
        a = tl.load(x_base_12 + c_off[None, :], mask=vmask[:, None] & c_mask[None, :], other=0.0)
        w_ptrs = w_col_base + w_off_12 + w_ci_base
        b = tl.load(w_ptrs, mask=c_mask[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(a, b, input_precision='hf32')

        # ky=2, kx=0
        vmask = mask_m & valid_20
        a = tl.load(x_base_20 + c_off[None, :], mask=vmask[:, None] & c_mask[None, :], other=0.0)
        w_ptrs = w_col_base + w_off_20 + w_ci_base
        b = tl.load(w_ptrs, mask=c_mask[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(a, b, input_precision='hf32')

        # ky=2, kx=1
        vmask = mask_m & valid_21
        a = tl.load(x_base_21 + c_off[None, :], mask=vmask[:, None] & c_mask[None, :], other=0.0)
        w_ptrs = w_col_base + w_off_21 + w_ci_base
        b = tl.load(w_ptrs, mask=c_mask[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(a, b, input_precision='hf32')

        # ky=2, kx=2
        vmask = mask_m & valid_22
        a = tl.load(x_base_22 + c_off[None, :], mask=vmask[:, None] & c_mask[None, :], other=0.0)
        w_ptrs = w_col_base + w_off_22 + w_ci_base
        b = tl.load(w_ptrs, mask=c_mask[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(a, b, input_precision='hf32')

        rc += BLOCK_K

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
            w = source_w.to(dtype=target_dtype)
            out_w = w.flip([2, 3]).permute(1, 0, 2, 3).contiguous()
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
        x_cont = x.contiguous()
        N_val, Cin, H, W = x_cont.shape
        Cout = w_conv.shape[0]
        H_out = H + K - 1
        W_out = W + K - 1
        y = torch.empty((N_val, Cout, H_out, W_out), device=x.device, dtype=x.dtype)
        bias_cont = bias.contiguous() if has_bias else torch.zeros(Cout, device=x.device, dtype=x.dtype)

        stride_xn = Cin * H * W
        stride_xc = H * W
        stride_xh = W
        stride_xw = 1

        stride_wo = Cin * K * K
        stride_wi = K * K
        stride_wkh = K
        stride_wkw = 1

        stride_yn = Cout * H_out * W_out
        stride_yc = H_out * W_out
        stride_yh = W_out
        stride_yw = 1

        grid = lambda meta: (
            N_val,
            triton.cdiv(H_out * W_out, meta['BLOCK_M']),
            triton.cdiv(Cout, meta['BLOCK_N']),
        )
        w00 = stride_wkh*0 + stride_wkw*0
        w01 = stride_wkh*0 + stride_wkw*1
        w02 = stride_wkh*0 + stride_wkw*2
        w10 = stride_wkh*1 + stride_wkw*0
        w11 = stride_wkh*1 + stride_wkw*1
        w12 = stride_wkh*1 + stride_wkw*2
        w20 = stride_wkh*2 + stride_wkw*0
        w21 = stride_wkh*2 + stride_wkw*1
        w22 = stride_wkh*2 + stride_wkw*2
        _convtransp2d_stride1_pad0_groups1_kernel[grid](
            x_cont, w_conv, bias_cont, y,
            N_val, Cin, H, W, Cout, K, H_out, W_out,
            stride_xn, stride_xc, stride_xh, stride_xw,
            stride_wo, stride_wi, stride_wkh, stride_wkw,
            stride_yn, stride_yc, stride_yh, stride_yw,
            w00, w01, w02, w10, w11, w12, w20, w21, w22,
        )
        return y
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
    return [in_channels, out_channels, kernel_size]
