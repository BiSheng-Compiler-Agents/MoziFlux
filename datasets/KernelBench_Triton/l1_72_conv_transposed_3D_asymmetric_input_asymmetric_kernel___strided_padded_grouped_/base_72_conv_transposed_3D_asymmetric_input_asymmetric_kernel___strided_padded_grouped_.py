import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SCATTER_BLOCK_D = 8
SCATTER_BLOCK_H = 2
SCATTER_BLOCK_W = 128
SCATTER_NUM_WARPS = 4
SCATTER_NUM_STAGES = 2
CACHE_CONV_WEIGHT = True


@triton.jit
def _upsample3d_scatter_kernel(
    in_ptr,
    out_ptr,
    N,
    C,
    Di,
    Hi,
    Wi,
    Do,
    Ho,
    Wo,
    SD,
    SH,
    SW,
    in_stride_n,
    in_stride_c,
    in_stride_d,
    in_stride_h,
    in_stride_w,
    out_stride_n,
    out_stride_c,
    out_stride_d,
    out_stride_h,
    out_stride_w,
    NUM_WBLK,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    tl.static_assert(BLOCK_D > 0)
    tl.static_assert(BLOCK_H > 0)
    tl.static_assert(BLOCK_W > 0)
    pid_hblk = tl.program_id(axis=0)
    pid_dblk = tl.program_id(axis=1)
    pid_combined = tl.program_id(axis=2)
    pid_nc = pid_combined // NUM_WBLK
    pid_wblk = pid_combined % NUM_WBLK
    n = pid_nc // C
    c = pid_nc % C
    d_off = pid_dblk * BLOCK_D + tl.arange(0, BLOCK_D)[:, None, None]
    h_off = pid_hblk * BLOCK_H + tl.arange(0, BLOCK_H)[None, :, None]
    w_off = pid_wblk * BLOCK_W + tl.arange(0, BLOCK_W)[None, None, :]
    valid = (n < N) & (c < C) & (d_off < Di) & (h_off < Hi) & (w_off < Wi)
    in_ptrs = in_ptr + n * in_stride_n + c * in_stride_c + d_off * in_stride_d + h_off * in_stride_h + w_off * in_stride_w
    out_ptrs = out_ptr + n * out_stride_n + c * out_stride_c + (d_off * SD) * out_stride_d + (h_off * SH) * out_stride_h + (w_off * SW) * out_stride_w
    x = tl.load(in_ptrs, mask=valid, other=0.0)
    tl.store(out_ptrs, x, mask=valid)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int = 32, out_channels: int = 64, kernel_size: tuple = (3, 5, 7), stride: tuple = (2, 2, 2), padding: tuple = (1, 2, 3), output_padding: tuple = (1, 1, 1), groups: int = 4, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)
        self._cached_w_conv = None
        self._cached_w_conv_key = None

    @staticmethod
    def _weight_to_conv3d(weight: torch.Tensor, groups: int) -> torch.Tensor:
        g = groups
        cin, co_g, k_d, k_h, k_w = weight.shape
        ci_g = cin // g
        co = co_g * g
        w_flip = weight.flip(dims=(2, 3, 4))
        w_g = w_flip.view(g, ci_g, co_g, k_d, k_h, k_w)
        return w_g.permute(0, 2, 1, 3, 4, 5).contiguous().view(co, ci_g, k_d, k_h, k_w)

    def _get_conv_weight(self) -> torch.Tensor:
        weight = self.conv_transpose3d.weight
        if not CACHE_CONV_WEIGHT:
            return self._weight_to_conv3d(weight, self.conv_transpose3d.groups).contiguous()
        key = (weight.data_ptr(), str(weight.device), weight.dtype)
        if self._cached_w_conv_key != key or self._cached_w_conv is None:
            self._cached_w_conv = self._weight_to_conv3d(weight, self.conv_transpose3d.groups).contiguous()
            self._cached_w_conv_key = key
        return self._cached_w_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not getattr(x, 'is_npu', False):
            raise RuntimeError('ModelNew requires an Ascend NPU tensor input')
        if any(d != 1 for d in self.conv_transpose3d.dilation):
            raise RuntimeError('ModelNew only supports dilation=(1, 1, 1)')
        sd, sh, sw = self.conv_transpose3d.stride
        pd, ph, pw = self.conv_transpose3d.padding
        od, oh, ow = self.conv_transpose3d.output_padding
        groups = self.conv_transpose3d.groups
        bias = self.conv_transpose3d.bias
        weight = self.conv_transpose3d.weight
        k_d, k_h, k_w = weight.shape[2], weight.shape[3], weight.shape[4]
        n, cin, di, hi, wi = x.shape
        du = (di - 1) * sd + 1 + od
        hu = (hi - 1) * sh + 1 + oh
        wu = (wi - 1) * sw + 1 + ow
        pad_d = k_d - 1 - pd
        pad_h = k_h - 1 - ph
        pad_w = k_w - 1 - pw
        if (pad_d < 0) or (pad_h < 0) or (pad_w < 0):
            raise RuntimeError('ModelNew requires kernel_size - 1 >= padding in every spatial dimension')
        w_conv = self._get_conv_weight()
        x_up = torch.zeros((n, cin, du, hu, wu), dtype=x.dtype, device=x.device)
        num_wblk = triton.cdiv(wi, SCATTER_BLOCK_W)
        grid = (triton.cdiv(hi, SCATTER_BLOCK_H), triton.cdiv(di, SCATTER_BLOCK_D), num_wblk * n * cin)
        in_strides = x.stride()
        out_strides = x_up.stride()
        _upsample3d_scatter_kernel[grid](x, x_up, n, cin, di, hi, wi, du, hu, wu, sd, sh, sw, in_strides[0], in_strides[1], in_strides[2], in_strides[3], in_strides[4], out_strides[0], out_strides[1], out_strides[2], out_strides[3], out_strides[4], num_wblk, BLOCK_D=SCATTER_BLOCK_D, BLOCK_H=SCATTER_BLOCK_H, BLOCK_W=SCATTER_BLOCK_W, num_warps=SCATTER_NUM_WARPS, num_stages=SCATTER_NUM_STAGES)
        return F.conv3d(x_up, w_conv, bias=bias, stride=1, padding=(pad_d, pad_h, pad_w), dilation=1, groups=groups)


_MODEL_CACHE = {}


def run_operator(x: torch.Tensor) -> torch.Tensor:
    key = (str(x.device), x.dtype)
    model = _MODEL_CACHE.get(key)
    if model is None:
        model = ModelNew().to(device=x.device, dtype=x.dtype)
        model.eval()
        _MODEL_CACHE[key] = model
    with torch.no_grad():
        return model(x)


batch_size = 8
in_channels = 32
out_channels = 32
kernel_size = (3, 5, 7)
depth = 12
height = 24
width = 48
stride = (2, 2, 2)
padding = (1, 2, 3)
output_padding = (1, 1, 1)
groups = 4


def get_inputs():
    x = torch.rand(batch_size, in_channels, depth, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, groups]
