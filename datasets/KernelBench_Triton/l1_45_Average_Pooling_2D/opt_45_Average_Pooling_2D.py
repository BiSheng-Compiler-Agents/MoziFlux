import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

_MAX_PROGRAMS = 65535


@triton.jit
def _avg_pool2d_tile_kernel(
    x_ptr,
    y_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    H,
    W,
    OH,
    OW,
    in_stride_n,
    in_stride_c,
    in_stride_h,
    in_stride_w,
    out_stride_n,
    out_stride_c,
    out_stride_h,
    out_stride_w,
    total_tiles,
    KH: tl.constexpr,
    KW: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
):
    tile_id = tl.program_id(0)
    ow = tile_id % OW
    tmp = tile_id // OW
    oh = tmp % OH
    tmp = tmp // OH
    c = tmp % C
    n = tmp // C

    ih0 = oh * SH
    iw0 = ow * SW
    x_base = x_ptr + n * in_stride_n + c * in_stride_c
    y_off = n * out_stride_n + c * out_stride_c + oh * out_stride_h + ow * out_stride_w

    acc = 0.0
    for kh in tl.static_range(0, KH):
        row_ptr = x_base + (ih0 + kh) * in_stride_h
        for kw in tl.static_range(0, KW):
            acc += tl.load(row_ptr + (iw0 + kw) * in_stride_w).to(tl.float32)
    tl.store(y_ptr + y_off, acc / ((KH * KW) + 0.0))


@triton.jit
def _avg_pool2d_persistent_kernel(
    x_ptr,
    y_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    H,
    W,
    OH,
    OW,
    in_stride_n,
    in_stride_c,
    in_stride_h,
    in_stride_w,
    out_stride_n,
    out_stride_c,
    out_stride_h,
    out_stride_w,
    total_tiles,
    n_programs,
    KH: tl.constexpr,
    KW: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
):
    pid = tl.program_id(0)
    for tile_id in range(pid, total_tiles, n_programs):
        ow = tile_id % OW
        tmp = tile_id // OW
        oh = tmp % OH
        tmp = tmp // OH
        c = tmp % C
        n = tmp // C

        ih0 = oh * SH
        iw0 = ow * SW
        x_base = x_ptr + n * in_stride_n + c * in_stride_c
        y_off = n * out_stride_n + c * out_stride_c + oh * out_stride_h + ow * out_stride_w

        acc = 0.0
        for kh in tl.static_range(0, KH):
            row_ptr = x_base + (ih0 + kh) * in_stride_h
            for kw in tl.static_range(0, KW):
                acc += tl.load(row_ptr + (iw0 + kw) * in_stride_w).to(
                    tl.float32)
        tl.store(y_ptr + y_off, acc / ((KH * KW) + 0.0))


def _avg_pool2d_triton(x: torch.Tensor,
                       kernel_size: int,
                       stride: int | None = None,
                       padding: int = 0):
    if not x.is_npu:
        raise RuntimeError(
            "Average Pooling 2D Triton operator requires an Ascend NPU tensor input."
        )
    if x.numel() == 0:
        raise ValueError(
            "Average Pooling 2D Triton operator does not support empty tensors."
        )
    if stride is None:
        stride = kernel_size

    N, C, H, W = x.shape
    KH = KW = int(kernel_size)
    SH = SW = int(stride)
    PH = PW = int(padding)
    if PH != 0 or PW != 0:
        return F.avg_pool2d(x,
                            kernel_size=KH,
                            stride=SH,
                            padding=PH,
                            count_include_pad=False)

    OH = (H - KH) // SH + 1
    OW = (W - KW) // SW + 1
    assert OH > 0 and OW > 0, "Invalid output size; check kernel/stride/padding."

    y = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)
    in_stride_n, in_stride_c, in_stride_h, in_stride_w = x.stride()
    out_stride_n, out_stride_c, out_stride_h, out_stride_w = y.stride()
    total_tiles = N * C * OH * OW

    if total_tiles <= _MAX_PROGRAMS:
        _avg_pool2d_tile_kernel[(total_tiles, )](
            x,
            y,
            N,
            C,
            H,
            W,
            OH,
            OW,
            in_stride_n,
            in_stride_c,
            in_stride_h,
            in_stride_w,
            out_stride_n,
            out_stride_c,
            out_stride_h,
            out_stride_w,
            total_tiles,
            KH=KH,
            KW=KW,
            SH=SH,
            SW=SW,
            num_warps=1,
            num_stages=2,
        )
    else:
        n_programs = _MAX_PROGRAMS
        _avg_pool2d_persistent_kernel[(n_programs, )](
            x,
            y,
            N,
            C,
            H,
            W,
            OH,
            OW,
            in_stride_n,
            in_stride_c,
            in_stride_h,
            in_stride_w,
            out_stride_n,
            out_stride_c,
            out_stride_h,
            out_stride_w,
            total_tiles,
            n_programs,
            KH=KH,
            KW=KW,
            SH=SH,
            SW=SW,
            num_warps=1,
            num_stages=2,
        )
    return y


class ModelNew(nn.Module):
    """2D Average Pooling with grid-cap dispatch and no-padding fast path."""

    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = int(kernel_size)
        self.stride = int(stride) if stride is not None else None
        self.padding = int(padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _avg_pool2d_triton(x, self.kernel_size, self.stride,
                                  self.padding)


batch_size = 16
channels = 64
height = 2048
width = 2048
kernel_size = 11


def get_inputs():
    x = torch.rand(batch_size, channels, height, width, device='npu')
    return [x]


def get_init_inputs():
    return [kernel_size]
