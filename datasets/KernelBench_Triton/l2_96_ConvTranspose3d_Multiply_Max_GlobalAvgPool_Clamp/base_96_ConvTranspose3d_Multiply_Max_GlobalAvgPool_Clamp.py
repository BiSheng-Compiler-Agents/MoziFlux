import torch
import torch.nn as nn
import triton
import triton.language as tl


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False) or x.device.type == "npu")


@triton.jit
def _fused_scale_maxpool3d_gap_clamp(
    x_ptr,
    out_ptr,
    N,
    C,
    D,
    H,
    W,
    scale,
    clamp_min,
    clamp_max,
    KSIZE: tl.constexpr,
    DP: tl.constexpr,
    HP: tl.constexpr,
    WP: tl.constexpr,
    NWINS: tl.constexpr,
    BLOCK_WINS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    n = pid // C
    c = pid % C

    sN = C * D * H * W
    sC = D * H * W
    sD = H * W
    sH = W

    base = x_ptr + n * sN + c * sC

    offs = tl.arange(0, BLOCK_WINS)
    sumv = tl.zeros((), dtype=tl.float32)
    hpwp = HP * WP
    full_limit = (NWINS // BLOCK_WINS) * BLOCK_WINS
    row_stride = sD + sH
    row_stride_plus_one = row_stride + 1

    if KSIZE == 2 and scale >= 0:
        for start in range(0, full_limit, BLOCK_WINS):
            idx = start + offs

            dp = idx // hpwp
            rem = idx - dp * hpwp
            hp = rem // WP
            wp = rem - hp * WP

            di0 = dp * KSIZE
            hi0 = hp * KSIZE
            wi0 = wp * KSIZE
            base_offsets = di0 * sD + hi0 * sH + wi0

            base_ptrs = base + base_offsets
            base_ptrs_h = base_ptrs + sH
            base_ptrs_d = base_ptrs + sD
            v000 = tl.load(base_ptrs).to(tl.float32)
            v001 = tl.load(base_ptrs + 1).to(tl.float32)
            v010 = tl.load(base_ptrs_h).to(tl.float32)
            v011 = tl.load(base_ptrs_h + 1).to(tl.float32)
            v100 = tl.load(base_ptrs_d).to(tl.float32)
            v101 = tl.load(base_ptrs_d + 1).to(tl.float32)
            v110 = tl.load(base_ptrs + row_stride).to(tl.float32)
            v111 = tl.load(base_ptrs + row_stride_plus_one).to(tl.float32)
            max01 = tl.maximum(v000, v001)
            max23 = tl.maximum(v010, v011)
            max45 = tl.maximum(v100, v101)
            max67 = tl.maximum(v110, v111)
            max0123 = tl.maximum(max01, max23)
            max4567 = tl.maximum(max45, max67)
            sumv += tl.sum(tl.maximum(max0123, max4567), axis=0) * scale

        if full_limit < NWINS:
            idx = full_limit + offs
            mask = idx < NWINS

            dp = idx // hpwp
            rem = idx - dp * hpwp
            hp = rem // WP
            wp = rem - hp * WP

            di0 = dp * KSIZE
            hi0 = hp * KSIZE
            wi0 = wp * KSIZE
            base_offsets = di0 * sD + hi0 * sH + wi0

            base_ptrs = base + base_offsets
            base_ptrs_h = base_ptrs + sH
            base_ptrs_d = base_ptrs + sD
            v000 = tl.load(base_ptrs, mask=mask,
                           other=-float("inf")).to(tl.float32)
            v001 = tl.load(base_ptrs + 1, mask=mask,
                           other=-float("inf")).to(tl.float32)
            v010 = tl.load(base_ptrs_h, mask=mask,
                           other=-float("inf")).to(tl.float32)
            v011 = tl.load(base_ptrs_h + 1, mask=mask,
                           other=-float("inf")).to(tl.float32)
            v100 = tl.load(base_ptrs_d, mask=mask,
                           other=-float("inf")).to(tl.float32)
            v101 = tl.load(base_ptrs_d + 1, mask=mask,
                           other=-float("inf")).to(tl.float32)
            v110 = tl.load(base_ptrs + row_stride,
                           mask=mask,
                           other=-float("inf")).to(tl.float32)
            v111 = tl.load(base_ptrs + row_stride_plus_one,
                           mask=mask,
                           other=-float("inf")).to(tl.float32)
            max01 = tl.maximum(v000, v001)
            max23 = tl.maximum(v010, v011)
            max45 = tl.maximum(v100, v101)
            max67 = tl.maximum(v110, v111)
            max0123 = tl.maximum(max01, max23)
            max4567 = tl.maximum(max45, max67)
            tail = tl.where(mask, tl.maximum(max0123, max4567) * scale, 0.0)
            sumv += tl.sum(tail, axis=0)
    else:
        for start in range(0, NWINS, BLOCK_WINS):
            idx = start + offs
            mask = idx < NWINS

            dp = idx // hpwp
            rem = idx - dp * hpwp
            hp = rem // WP
            wp = rem - hp * WP

            di0 = dp * KSIZE
            hi0 = hp * KSIZE
            wi0 = wp * KSIZE
            base_offsets = di0 * sD + hi0 * sH + wi0

            maxi = tl.full(offs.shape, -float("inf"), tl.float32)
            for kd in tl.static_range(0, KSIZE):
                for kh in tl.static_range(0, KSIZE):
                    for kw in tl.static_range(0, KSIZE):
                        ptrs = base + base_offsets + kd * sD + kh * sH + kw
                        v = tl.load(ptrs, mask=mask, other=-float("inf"))
                        vf32 = v.to(tl.float32) * scale
                        vf32 = tl.where(mask, vf32, -float("inf"))
                        maxi = tl.maximum(maxi, vf32)

            maxi = tl.where(mask, maxi, 0.0)
            sumv += tl.sum(maxi, axis=0)

    meanv = sumv / tl.full((), NWINS, tl.float32)
    meanv = tl.minimum(tl.maximum(meanv, clamp_min), clamp_max)
    tl.store(out_ptr + n * C + c, meanv)


class ModelNew(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 scale, maxpool_kernel_size):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels,
                                                 out_channels,
                                                 kernel_size,
                                                 stride=stride,
                                                 padding=padding)
        self.scale = scale
        self.maxpool = nn.MaxPool3d(kernel_size=maxpool_kernel_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.clamp_min = 0
        self.clamp_max = 1

    def forward(self, x):
        if not _is_npu_tensor(x):
            raise RuntimeError(
                "ModelNew expects Ascend NPU tensors so the Triton kernel path is exercised."
            )

        y = self.conv_transpose(x)
        if not _is_npu_tensor(y):
            raise RuntimeError(
                "ConvTranspose3d output must stay on Ascend NPU for the Triton kernel path."
            )

        if not y.is_contiguous():
            y = y.contiguous()

        k = self.maxpool.kernel_size
        s = getattr(self.maxpool, "stride", None)
        p = getattr(self.maxpool, "padding", 0)
        dila = getattr(self.maxpool, "dilation", 1)
        ceil_m = getattr(self.maxpool, "ceil_mode", False)

        if isinstance(k, (tuple, list)):
            if not (k[0] == k[1] == k[2]):
                raise RuntimeError(
                    "ModelNew requires a cubic MaxPool3d kernel for the fused Triton path."
                )
            k = k[0]
        if s is None:
            s = k
        elif isinstance(s, (tuple, list)):
            if not (s[0] == s[1] == s[2] == k):
                raise RuntimeError(
                    "ModelNew requires MaxPool3d stride to match the cubic kernel size."
                )
            s = s[0]
        if isinstance(p, (tuple, list)):
            if p != (0, 0, 0):
                raise RuntimeError(
                    "ModelNew requires zero MaxPool3d padding for the fused Triton path."
                )
        elif p != 0:
            raise RuntimeError(
                "ModelNew requires zero MaxPool3d padding for the fused Triton path."
            )
        if isinstance(dila, (tuple, list)):
            if dila != (1, 1, 1):
                raise RuntimeError(
                    "ModelNew requires unit MaxPool3d dilation for the fused Triton path."
                )
        elif dila != 1:
            raise RuntimeError(
                "ModelNew requires unit MaxPool3d dilation for the fused Triton path."
            )
        if s != k or ceil_m:
            raise RuntimeError(
                "ModelNew requires non-ceil MaxPool3d with stride equal to kernel size."
            )

        N, C, D, H, W = y.shape
        if D < k or H < k or W < k:
            raise RuntimeError(
                "ConvTranspose3d output must be at least one pooling window in every spatial dimension."
            )

        DP = (D - k) // k + 1
        HP = (H - k) // k + 1
        WP = (W - k) // k + 1
        NWINS = DP * HP * WP

        out = torch.empty((N, C, 1, 1, 1), device=y.device, dtype=y.dtype)
        _fused_scale_maxpool3d_gap_clamp[(N * C, )](
            y,
            out.view(-1),
            N,
            C,
            D,
            H,
            W,
            float(self.scale),
            float(self.clamp_min),
            float(self.clamp_max),
            KSIZE=k,
            DP=DP,
            HP=HP,
            WP=WP,
            NWINS=NWINS,
            BLOCK_WINS=256,
            num_warps=8,
            num_stages=2,
        )
        return out


batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
scale = 0.5
maxpool_kernel_size = 2


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [
        in_channels, out_channels, kernel_size, stride, padding, scale,
        maxpool_kernel_size
    ]


_MODEL_CACHE = None


def build_model():
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        with torch.random.fork_rng():
            torch.manual_seed(2026)
            _MODEL_CACHE = ModelNew(*get_init_inputs()).to("npu")
        _MODEL_CACHE.eval()
    return _MODEL_CACHE


def run_operator(x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return build_model()(x)
