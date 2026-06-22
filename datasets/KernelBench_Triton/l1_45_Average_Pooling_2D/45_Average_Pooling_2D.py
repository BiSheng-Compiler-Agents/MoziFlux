import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def avg_pool2d_fwd_kernel(
    x_ptr,
    y_ptr,
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
    KH: tl.constexpr,
    KW: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    PH: tl.constexpr,
    PW: tl.constexpr,
):
    pid = tl.program_id(0)

    ow = pid % OW
    tmp = pid // OW
    oh = tmp % OH
    tmp = tmp // OH
    c = tmp % C
    n = tmp // C

    ih0 = oh * SH - PH
    iw0 = ow * SW - PW

    x_base = x_ptr + n * in_stride_n + c * in_stride_c
    y_offset = n * out_stride_n + c * out_stride_c + oh * out_stride_h + ow * out_stride_w

    acc = 0.0
    count = 0
    for kh in tl.static_range(0, KH):
        ih = ih0 + kh
        if (ih >= 0) and (ih < H):
            row_ptr = x_base + ih * in_stride_h
            for kw in tl.static_range(0, KW):
                iw = iw0 + kw
                if (iw >= 0) and (iw < W):
                    x_offset = row_ptr + iw * in_stride_w
                    acc += tl.load(x_offset).to(tl.float32)
                    count += 1

    out = acc / count
    tl.store(y_ptr + y_offset, out)


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

    # shapes
    N, C, H, W = x.shape
    KH = KW = int(kernel_size)
    SH = SW = int(stride)
    PH = PW = int(padding)

    # output sizes (floor as in PyTorch AvgPool2d)
    OH = (H + 2 * PH - KH) // SH + 1
    OW = (W + 2 * PW - KW) // SW + 1
    assert OH > 0 and OW > 0, "Invalid output size; check kernel/stride/padding."

    # allocate output
    y = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)

    # get strides in elements
    in_stride_n, in_stride_c, in_stride_h, in_stride_w = x.stride()
    out_stride_n, out_stride_c, out_stride_h, out_stride_w = y.stride()

    grid = (N * C * OH * OW, )

    avg_pool2d_fwd_kernel[grid](x,
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
                                KH=KH,
                                KW=KW,
                                SH=SH,
                                SW=SW,
                                PH=PH,
                                PW=PW,
                                num_warps=1,
                                num_stages=1)
    return y


class ModelNew(nn.Module):
    """
    Simple model that performs 2D Average Pooling using a custom Triton kernel on Ascend NPU.
    """

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
