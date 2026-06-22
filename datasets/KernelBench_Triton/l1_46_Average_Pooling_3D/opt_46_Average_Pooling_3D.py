import torch
import torch.nn as nn
import triton
import triton.language as tl

_MAX_PROGRAMS = 65535
_BLOCK_W = 64


def _triple(v):
    if isinstance(v, (tuple, list)):
        assert len(v) == 3
        return int(v[0]), int(v[1]), int(v[2])
    v = int(v)
    return (v, v, v)


@triton.jit
def _avgpool3d_tile_kernel(
    x_ptr,
    y_ptr,
    total_tiles,
    N: tl.constexpr,
    C: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    OD: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    SD: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    PD: tl.constexpr,
    PH: tl.constexpr,
    PW: tl.constexpr,
    KSIZE_D: tl.constexpr,
    KSIZE_H: tl.constexpr,
    KSIZE_W: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    tile_id = tl.program_id(0)
    n_w_tiles = tl.cdiv(OW, BLOCK_W)
    w_tile = tile_id % n_w_tiles
    row = tile_id // n_w_tiles

    ow = w_tile * BLOCK_W + tl.arange(0, BLOCK_W)
    tl.multiple_of(ow, 16)
    tl.max_contiguous(ow, BLOCK_W)
    mask_ow = ow < OW

    oh = row % OH
    t = row // OH
    od = t % OD
    t = t // OD
    c = t % C
    n = t // C

    id_base = od * SD - PD
    ih_base = oh * SH - PH
    iw_base = ow * SW - PW

    base_nc = (n * C + c) * D * H * W
    out_idx = (((n * C + c) * OD + od) * OH + oh) * OW + ow
    hw = H * W
    acc = tl.zeros((BLOCK_W, ), dtype=tl.float32)

    for kd in tl.static_range(0, KSIZE_D):
        idv = id_base + kd
        md = (idv >= 0) & (idv < D)
        z_base = idv * hw
        for kh in tl.static_range(0, KSIZE_H):
            ihv = ih_base + kh
            mh = (ihv >= 0) & (ihv < H)
            row_base = base_nc + z_base + ihv * W
            for kw in tl.static_range(0, KSIZE_W):
                iwv = iw_base + kw
                mw = (iwv >= 0) & (iwv < W)
                m = mask_ow & md & mh & mw
                vals = tl.load(x_ptr + row_base + iwv, mask=m, other=0.0)
                acc += vals.to(tl.float32)

    scale = 1.0 / float(KSIZE_D * KSIZE_H * KSIZE_W)
    tl.store(y_ptr + out_idx, acc * scale, mask=mask_ow)


@triton.jit
def _avgpool3d_persistent_kernel(
    x_ptr,
    y_ptr,
    total_tiles,
    n_programs,
    N: tl.constexpr,
    C: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    OD: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    SD: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    PD: tl.constexpr,
    PH: tl.constexpr,
    PW: tl.constexpr,
    KSIZE_D: tl.constexpr,
    KSIZE_H: tl.constexpr,
    KSIZE_W: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    n_w_tiles = tl.cdiv(OW, BLOCK_W)
    for tile_id in range(pid, total_tiles, n_programs):
        w_tile = tile_id % n_w_tiles
        row = tile_id // n_w_tiles

        ow = w_tile * BLOCK_W + tl.arange(0, BLOCK_W)
        tl.multiple_of(ow, 16)
        tl.max_contiguous(ow, BLOCK_W)
        mask_ow = ow < OW

        oh = row % OH
        t = row // OH
        od = t % OD
        t = t // OD
        c = t % C
        n = t // C

        id_base = od * SD - PD
        ih_base = oh * SH - PH
        iw_base = ow * SW - PW

        base_nc = (n * C + c) * D * H * W
        out_idx = (((n * C + c) * OD + od) * OH + oh) * OW + ow
        hw = H * W
        acc = tl.zeros((BLOCK_W, ), dtype=tl.float32)

        for kd in tl.static_range(0, KSIZE_D):
            idv = id_base + kd
            md = (idv >= 0) & (idv < D)
            z_base = idv * hw
            for kh in tl.static_range(0, KSIZE_H):
                ihv = ih_base + kh
                mh = (ihv >= 0) & (ihv < H)
                row_base = base_nc + z_base + ihv * W
                for kw in tl.static_range(0, KSIZE_W):
                    iwv = iw_base + kw
                    mw = (iwv >= 0) & (iwv < W)
                    m = mask_ow & md & mh & mw
                    vals = tl.load(x_ptr + row_base + iwv, mask=m, other=0.0)
                    acc += vals.to(tl.float32)

        scale = 1.0 / float(KSIZE_D * KSIZE_H * KSIZE_W)
        tl.store(y_ptr + out_idx, acc * scale, mask=mask_ow)


class ModelNew(nn.Module):
    """3D average pooling optimized for NCDHW contiguous tensors on Ascend NPU."""

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
        sD, sH, sW = _triple(self.avg_pool.stride if self.avg_pool.
                             stride is not None else self.avg_pool.kernel_size)
        pD, pH, pW = _triple(self.avg_pool.padding)

        OD = (D + 2 * pD - kD) // sD + 1
        OH = (H + 2 * pH - kH) // sH + 1
        OW = (W + 2 * pW - kW) // sW + 1
        if OD <= 0 or OH <= 0 or OW <= 0:
            return torch.empty((N, C, max(OD, 0), max(OH, 0), max(OW, 0)),
                               device=x.device,
                               dtype=x.dtype)

        y = torch.empty((N, C, OD, OH, OW), device=x.device, dtype=x.dtype)
        total_tiles = N * C * OD * OH * triton.cdiv(OW, _BLOCK_W)
        common = dict(
            N=N,
            C=C,
            D=D,
            H=H,
            W=W,
            OD=OD,
            OH=OH,
            OW=OW,
            SD=sD,
            SH=sH,
            SW=sW,
            PD=pD,
            PH=pH,
            PW=pW,
            KSIZE_D=kD,
            KSIZE_H=kH,
            KSIZE_W=kW,
            BLOCK_W=_BLOCK_W,
        )
        if total_tiles <= _MAX_PROGRAMS:
            _avgpool3d_tile_kernel[(total_tiles, )](x,
                                                    y,
                                                    total_tiles,
                                                    **common,
                                                    num_warps=8,
                                                    num_stages=2)
        else:
            n_programs = _MAX_PROGRAMS
            _avgpool3d_persistent_kernel[(n_programs, )](x,
                                                         y,
                                                         total_tiles,
                                                         n_programs,
                                                         **common,
                                                         num_warps=8,
                                                         num_stages=2)
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
