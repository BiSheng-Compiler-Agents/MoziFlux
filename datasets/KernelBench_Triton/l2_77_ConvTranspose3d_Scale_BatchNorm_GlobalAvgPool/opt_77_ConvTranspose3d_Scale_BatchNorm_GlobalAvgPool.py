import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

_MAX_GRID = 65535
_SUM_BLOCK = 2048
_USE_SUM_SHORTCUT = True


@triton.jit
def _spatial_sum3d_direct_kernel(x_ptr, y_ptr, M, L: tl.constexpr,
                                 BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    valid = pid < M
    offs = tl.arange(0, BLOCK)
    total = 0.0
    for start in tl.range(0, L, BLOCK):
        idx = start + offs
        mask = valid & (idx < L)
        vals = tl.load(x_ptr + pid * L + idx, mask=mask,
                       other=0.0).to(tl.float32)
        total += tl.sum(vals, axis=0)
    tl.store(y_ptr + pid, total, mask=valid)


@triton.jit
def _spatial_sum3d_persistent_kernel(x_ptr, y_ptr, M, n_programs,
                                     L: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    for row in range(pid, M, n_programs):
        total = 0.0
        for start in tl.range(0, L, BLOCK):
            idx = start + offs
            mask = idx < L
            vals = tl.load(x_ptr + row * L + idx, mask=mask,
                           other=0.0).to(tl.float32)
            total += tl.sum(vals, axis=0)
        tl.store(y_ptr + row, total)


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False) or x.device.type == "npu")


def _triple(v):
    if isinstance(v, tuple):
        return v
    return (v, v, v)


def _convtranspose3d_out_elems(conv: nn.ConvTranspose3d, d: int, h: int,
                               w: int) -> int:
    sd, sh, sw = _triple(conv.stride)
    pd, ph, pw = _triple(conv.padding)
    opd, oph, opw = _triple(conv.output_padding)
    dd, dh, dw = _triple(conv.dilation)
    kd, kh, kw = _triple(conv.kernel_size)
    od = (d - 1) * sd - 2 * pd + dd * (kd - 1) + opd + 1
    oh = (h - 1) * sh - 2 * ph + dh * (kh - 1) + oph + 1
    ow = (w - 1) * sw - 2 * pw + dw * (kw - 1) + opw + 1
    return od * oh * ow


def _spatial_sum3d_triton(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 5, "Input must be NCDHW"
    n, c, d, h, w = x.shape
    rows = n * c
    L = d * h * w
    x_flat = x.contiguous().view(-1)
    sums = torch.empty((n, c), device=x.device, dtype=torch.float32)
    grid = (rows, ) if rows <= _MAX_GRID else (_MAX_GRID, )
    if rows <= _MAX_GRID:
        _spatial_sum3d_direct_kernel[grid](x_flat,
                                           sums,
                                           rows,
                                           L=L,
                                           BLOCK=_SUM_BLOCK,
                                           num_warps=8,
                                           num_stages=2)
    else:
        _spatial_sum3d_persistent_kernel[grid](x_flat,
                                               sums,
                                               rows,
                                               grid[0],
                                               L=L,
                                               BLOCK=_SUM_BLOCK,
                                               num_warps=8,
                                               num_stages=2)
    return sums


def _can_use_sum_shortcut(model: "ModelNew", x: torch.Tensor) -> bool:
    conv = model.conv_transpose
    return (_USE_SUM_SHORTCUT and (not model.batch_norm.training)
            and getattr(model.batch_norm, "track_running_stats", True)
            and conv.groups == 1 and _triple(conv.stride) == (1, 1, 1)
            and _triple(conv.padding) == (0, 0, 0)
            and _triple(conv.output_padding) == (0, 0, 0)
            and _triple(conv.dilation) == (1, 1, 1) and _is_npu_tensor(x))


class ModelNew(nn.Module):
    """
    Optimized ConvTranspose3d -> scalar scale -> BatchNorm3d -> GlobalAvgPool.

    In eval mode with the default ConvTranspose3d geometry, the global sum of the
    transposed convolution is computed from input spatial sums and per-output-channel
    kernel sums, avoiding materialization of the large 5D convolution output.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 scale_factor,
                 eps=1e-5,
                 momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels,
                                                 kernel_size)
        self.scale_factor = scale_factor
        self.batch_norm = nn.BatchNorm3d(out_channels,
                                         eps=eps,
                                         momentum=momentum)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def _fallback_forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.conv_transpose.weight * self.scale_factor
        b = None if self.conv_transpose.bias is None else self.conv_transpose.bias * self.scale_factor
        x = F.conv_transpose3d(
            x,
            w,
            bias=b,
            stride=self.conv_transpose.stride,
            padding=self.conv_transpose.padding,
            output_padding=self.conv_transpose.output_padding,
            groups=self.conv_transpose.groups,
            dilation=self.conv_transpose.dilation)
        x = self.batch_norm(x)
        return F.adaptive_avg_pool3d(x, (1, 1, 1))

    def forward(self, x):
        if not _can_use_sum_shortcut(self, x):
            return self._fallback_forward(x)

        dtype = x.dtype
        n, _, d, h, w = x.shape
        out_elems = _convtranspose3d_out_elems(self.conv_transpose, d, h, w)

        # Sum input spatial planes with Triton, then use a tiny GEMM for channel mixing.
        input_sums = _spatial_sum3d_triton(x)
        weight_sums = self.conv_transpose.weight.to(torch.float32).sum(dim=(2,
                                                                            3,
                                                                            4))
        y_sum = torch.matmul(input_sums, weight_sums) * float(
            self.scale_factor)
        if self.conv_transpose.bias is not None:
            y_sum = y_sum + self.conv_transpose.bias.to(
                torch.float32) * (float(self.scale_factor) * out_elems)
        mean = y_sum * (1.0 / float(out_elems))

        bn = self.batch_norm
        rm = bn.running_mean.to(torch.float32).view(1, -1)
        rv = bn.running_var.to(torch.float32).view(1, -1)
        affine = torch.rsqrt(rv + bn.eps)
        if bn.affine:
            affine = affine * bn.weight.to(torch.float32).view(1, -1)
            bias = bn.bias.to(torch.float32).view(1, -1)
        else:
            bias = 0.0
        out = (mean - rm) * affine + bias
        return out.view(n, self.conv_transpose.out_channels, 1, 1, 1).to(dtype)


batch_size = 16
in_channels = 64
out_channels = 128
depth, height, width = 16, 32, 32
kernel_size = 5
scale_factor = 2.0
_MODEL_CACHE = {}


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scale_factor]


def _get_default_model(device: torch.device, dtype: torch.dtype) -> ModelNew:
    cache_key = (device.type, getattr(device, "index", None), dtype)
    if cache_key not in _MODEL_CACHE:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            model = ModelNew(*get_init_inputs())
        model = model.to(device=device, dtype=dtype)
        model.eval()
        _MODEL_CACHE[cache_key] = model
    return _MODEL_CACHE[cache_key]


def conv_transpose3d_scale_batch_norm_global_avg_pool(
        x: torch.Tensor) -> torch.Tensor:
    model = _get_default_model(x.device, x.dtype)
    with torch.no_grad():
        return model(x)
