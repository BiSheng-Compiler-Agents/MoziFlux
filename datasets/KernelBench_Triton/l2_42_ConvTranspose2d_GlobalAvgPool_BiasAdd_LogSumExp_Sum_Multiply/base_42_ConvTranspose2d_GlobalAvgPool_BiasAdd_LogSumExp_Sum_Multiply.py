import torch
import torch.nn as nn
import triton
import triton.language as tl


batch_size = 16
in_channels = 64
out_channels = 128
height = width = 512
kernel_size = 3
bias_shape = (out_channels, 1, 1)

FP32_MEAN_BLOCK_C = 16
FP32_MEAN_BLOCK_HW_CAP = 256
FP32_MEAN_NUM_WARPS = 8
FP32_MEAN_NUM_STAGES = 4
FP32_LSE_BLOCK_C = 64
FP32_LSE_NUM_WARPS = 4
FP32_LSE_NUM_STAGES = 3
NON_FP32_BLOCK_C = 64
NON_FP32_BLOCK_HW_CAP = 128


def _next_pow2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


@triton.jit
def _fused_mean_bias_lse(
    x_ptr,
    bias_ptr,
    out_ptr,
    N, C, H, W,
    stride_n, stride_c,
    bias_stride_c,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid
    if n >= N:
        return

    HW = H * W
    inv_hw = 1.0 / HW
    n_base = n * stride_n

    NEG_INF = -float("inf")
    m = tl.full((), NEG_INF, dtype=tl.float32)
    s = tl.zeros((), dtype=tl.float32)
    c_arange = tl.arange(0, BLOCK_C)
    hw_arange = tl.arange(0, BLOCK_HW)

    for c_start in range(0, C, BLOCK_C):
        c_idx = c_start + c_arange
        c_mask = c_idx < C
        sum_c = tl.zeros((BLOCK_C,), dtype=tl.float32)
        base_c = n_base + c_idx * stride_c
        ptrs_base = base_c[:, None]

        for hw_start in range(0, HW, BLOCK_HW):
            offs_hw = hw_start + hw_arange
            load_mask = c_mask[:, None] & (offs_hw < HW)[None, :]
            tile = tl.load(x_ptr + ptrs_base + offs_hw[None, :], mask=load_mask, other=0.0, cache_modifier=".cg")
            sum_c += tl.sum(tile, axis=1)

        mean_c = sum_c * inv_hw
        bias = tl.load(bias_ptr + c_idx * bias_stride_c, mask=c_mask, other=0.0, cache_modifier=".ca")
        v = tl.where(c_mask, mean_c + bias, NEG_INF)
        tile_max = tl.max(v, axis=0)
        m2 = tl.maximum(m, tile_max)
        s = s * tl.exp(m - m2) + tl.sum(tl.exp(v - m2), axis=0)
        m = m2

    tl.store(out_ptr + n, 10.0 * (tl.log(s) + m))


@triton.jit
def _channel_mean_kernel(
    x_ptr,
    mean_ptr,
    N, C, HW, C_TILES,
    stride_n, stride_c,
    mean_stride_n, mean_stride_c,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_n = pid // C_TILES
    pid_c = pid - pid_n * C_TILES
    c_start = pid_c * BLOCK_C
    if pid_n >= N or c_start >= C:
        return

    c_idx = c_start + tl.arange(0, BLOCK_C)
    c_mask = c_idx < C
    base_c = pid_n * stride_n + c_idx * stride_c
    offs_hw = tl.arange(0, BLOCK_HW)
    sums = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for hw_start in range(0, HW, BLOCK_HW):
        idx = hw_start + offs_hw
        mask = c_mask[:, None] & (idx < HW)[None, :]
        vals = tl.load(
            x_ptr + base_c[:, None] + idx[None, :],
            mask=mask,
            other=0.0,
            cache_modifier=".cg",
        )
        sums += tl.sum(vals, axis=1)

    means = sums * (1.0 / HW)
    tl.store(mean_ptr + pid_n * mean_stride_n + c_idx * mean_stride_c, means, mask=c_mask)


@triton.jit
def _bias_lse_kernel(
    mean_ptr,
    bias_ptr,
    out_ptr,
    N, C,
    mean_stride_n, mean_stride_c,
    bias_stride_c,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= N:
        return

    row_base = pid * mean_stride_n
    NEG_INF = -float("inf")
    m = tl.full((), NEG_INF, dtype=tl.float32)
    s = tl.zeros((), dtype=tl.float32)
    offs_c = tl.arange(0, BLOCK_C)

    for c_start in range(0, C, BLOCK_C):
        c_idx = c_start + offs_c
        mask = c_idx < C
        mean = tl.load(
            mean_ptr + row_base + c_idx * mean_stride_c,
            mask=mask,
            other=0.0,
            cache_modifier=".ca",
        )
        bias = tl.load(
            bias_ptr + c_idx * bias_stride_c,
            mask=mask,
            other=0.0,
            cache_modifier=".ca",
        )
        v = tl.where(mask, mean + bias, NEG_INF)
        tile_max = tl.max(v, axis=0)
        m2 = tl.maximum(m, tile_max)
        s = s * tl.exp(m - m2) + tl.sum(tl.exp(v - m2), axis=0)
        m = m2

    tl.store(out_ptr + pid, 10.0 * (tl.log(s) + m))


class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, global average pooling, adds a bias, applies log-sum-exp, sum, and multiplication.
    """
    def __init__(
        self,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        bias_shape=bias_shape,
    ):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        if x.device.type != "npu":
            raise RuntimeError(
                f"ModelNew expects Ascend NPU inputs, but received device {x.device!s}."
            )

        x = self.conv_transpose(x)

        y = x.contiguous()
        N, C, H, W = y.shape
        HW = H * W
        means = torch.empty((N, C), device=y.device, dtype=torch.float32)
        out = torch.empty((N,), device=y.device, dtype=torch.float32)

        if y.dtype == torch.float32:
            BLOCK_HW = min(FP32_MEAN_BLOCK_HW_CAP, _next_pow2(HW))
            BLOCK_C = min(FP32_LSE_BLOCK_C, _next_pow2(C))
            BLOCK_MEAN_C = min(FP32_MEAN_BLOCK_C, _next_pow2(C))
            c_tiles = triton.cdiv(C, BLOCK_MEAN_C)
            _channel_mean_kernel[(N * c_tiles,)](
                y,
                means,
                N,
                C,
                HW,
                c_tiles,
                y.stride(0),
                y.stride(1),
                means.stride(0),
                means.stride(1),
                BLOCK_C=BLOCK_MEAN_C,
                BLOCK_HW=BLOCK_HW,
                num_warps=FP32_MEAN_NUM_WARPS,
                num_stages=FP32_MEAN_NUM_STAGES,
            )

            _bias_lse_kernel[(N,)](
                means,
                self.bias,
                out,
                N,
                C,
                means.stride(0),
                means.stride(1),
                self.bias.stride(0),
                BLOCK_C=BLOCK_C,
                num_warps=FP32_LSE_NUM_WARPS,
                num_stages=FP32_LSE_NUM_STAGES,
            )
        else:
            BLOCK_C = min(NON_FP32_BLOCK_C, _next_pow2(C))
            BLOCK_HW = min(NON_FP32_BLOCK_HW_CAP, _next_pow2(HW))
            tile_work = BLOCK_C * BLOCK_HW
            _fused_mean_bias_lse[(N,)](
                y,
                self.bias,
                out,
                N,
                C,
                H,
                W,
                y.stride(0),
                y.stride(1),
                self.bias.stride(0),
                BLOCK_C=BLOCK_C,
                BLOCK_HW=BLOCK_HW,
                num_warps=8 if tile_work >= 8192 else 4,
                num_stages=4,
            )

        return out.view(N, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]
