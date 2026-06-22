import torch
import torch.nn as nn
import triton
import triton.language as tl


def _triple(v):
    if isinstance(v, tuple):
        assert len(v) == 3
        return v
    return (v, v, v)


@triton.jit
def avgpool3d_kernel(
    x_ptr,
    y_ptr,
    N,
    C,
    D,
    H,
    W,
    OD,
    OH,
    OW,
    SD,
    SH,
    SW,
    PD,
    PH,
    PW,
    KSIZE_D: tl.constexpr,
    KSIZE_H: tl.constexpr,
    KSIZE_W: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_w = tl.program_id(axis=0)
    pid_ohod = tl.program_id(axis=1)
    pid_nc = tl.program_id(axis=2)

    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_w = offs_w < OW

    oh = pid_ohod % OH
    od = pid_ohod // OH
    id_base = od * SD - PD
    ih_base = oh * SH - PH
    iw_base = offs_w * SW - PW

    acc = tl.zeros([BLOCK_W], dtype=tl.float32)
    base_nc = pid_nc.to(tl.int64)
    base_ncD = base_nc * D

    for kd in tl.static_range(KSIZE_D):
        idv = id_base + kd
        md = (idv >= 0) & (idv < D)
        id_safe = tl.maximum(0, tl.minimum(idv, D - 1))
        for kh in tl.static_range(KSIZE_H):
            ihv = ih_base + kh
            mh = (ihv >= 0) & (ihv < H)
            ih_safe = tl.maximum(0, tl.minimum(ihv, H - 1))
            row_base = ((base_ncD + id_safe) * H + ih_safe) * W
            for kw in tl.static_range(KSIZE_W):
                iwv = iw_base + kw
                m = mask_w & md & mh & (iwv >= 0) & (iwv < W)
                iw_safe = tl.maximum(0, tl.minimum(iwv, W - 1))
                v = tl.load(x_ptr + row_base + iw_safe, mask=m, other=0.0)
                acc += v.to(tl.float32)

    scale = 1.0 / float(KSIZE_D * KSIZE_H * KSIZE_W)
    out = acc * scale
    out_base = ((base_nc * OD + od) * OH + oh) * OW
    tl.store(y_ptr + out_base + offs_w, out, mask=mask_w)


class ModelNew(nn.Module):
    """
    3D average pooling backed by a Triton kernel on Ascend NPU.
    """

    def __init__(self,
                 kernel_size: int = 3,
                 stride: int = 2,
                 padding: int = 1):
        super(ModelNew, self).__init__()
        self.avg_pool = nn.AvgPool3d(kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not hasattr(x, "is_npu") or not x.is_npu:
            raise ValueError("ModelNew expects an Ascend NPU tensor input.")

        x = x.contiguous()
        N, C, D, H, W = x.shape

        kD, kH, kW = _triple(self.avg_pool.kernel_size)
        stride = self.avg_pool.stride if self.avg_pool.stride is not None else self.avg_pool.kernel_size
        sD, sH, sW = _triple(stride)
        pD, pH, pW = _triple(self.avg_pool.padding)

        OD = (D + 2 * pD - kD) // sD + 1
        OH = (H + 2 * pH - kH) // sH + 1
        OW = (W + 2 * pW - kW) // sW + 1

        y = torch.empty((N, C, OD, OH, OW), device=x.device, dtype=x.dtype)

        block_w = 128

        def grid(META):
            return (triton.cdiv(OW, META["BLOCK_W"]), OD * OH, N * C)

        avgpool3d_kernel[grid](
            x,
            y,
            N,
            C,
            D,
            H,
            W,
            OD,
            OH,
            OW,
            sD,
            sH,
            sW,
            pD,
            pH,
            pW,
            KSIZE_D=kD,
            KSIZE_H=kH,
            KSIZE_W=kW,
            BLOCK_W=block_w,
            num_warps=8,
            num_stages=1,
        )
        return y


batch_size = 16
channels = 32
depth = 128
height = 128
width = 256
kernel_size = 3
stride = 2
padding = 1


def get_inputs():
    x = torch.rand(batch_size, channels, depth, height, width)
    return [x]


def get_init_inputs():
    return [kernel_size, stride, padding]
