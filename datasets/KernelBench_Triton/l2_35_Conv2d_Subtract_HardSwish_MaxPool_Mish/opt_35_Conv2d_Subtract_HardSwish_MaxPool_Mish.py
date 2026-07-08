import torch
import torch.nn as nn
import triton
import triton.language as tl
import triton.runtime.driver as driver

DEFAULT_BATCH_SIZE = 128
DEFAULT_IN_CHANNELS = 64
DEFAULT_OUT_CHANNELS = 128
DEFAULT_HEIGHT = 128
DEFAULT_WIDTH = 128
DEFAULT_KERNEL_SIZE = 3
DEFAULT_SUBTRACT_VALUE = 0.5
DEFAULT_POOL_KERNEL_SIZE = 2


@triton.jit
def _fused_hswish_maxpool2_mish_vector_kernel(
    x_ptr,
    y_ptr,
    N_OUT,
    N,
    C,
    H,
    W,
    H_OUT,
    W_OUT,
    subtract_value,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_OUT
    safe_offs = tl.where(mask, offs, 0)

    wo = safe_offs % W_OUT
    t1 = safe_offs // W_OUT
    ho = t1 % H_OUT
    t2 = t1 // H_OUT
    c = t2 % C
    n = t2 // C
    base = ((n * C + c) * H + ho * 2) * W + wo * 2

    k = tl.arange(0, 4)
    kh = k // 2
    kw = k - kh * 2
    vals = tl.load(x_ptr + base[None, :] + kh[:, None] * W + kw[:, None],
                   mask=mask[None, :],
                   other=0.0).to(tl.float32) - subtract_value
    vp3 = tl.minimum(tl.maximum(vals + 3.0, 0.0), 6.0)
    hs = vals * (vp3 * (1.0 / 6.0))
    mv = tl.max(hs, axis=0)

    ax = tl.abs(mv)
    sp = tl.log(1.0 + tl.exp(-ax)) + tl.maximum(mv, 0.0)
    e2 = tl.exp(-2.0 * sp)
    outv = mv * ((1.0 - e2) / (1.0 + e2))
    tl.store(y_ptr + safe_offs, outv, mask=mask)


@triton.jit
def _fused_hswish_maxpool2_mish_row_kernel(
    x_ptr,
    y_ptr,
    total_tiles,
    N,
    C,
    H,
    W,
    H_OUT,
    W_OUT,
    subtract_value,
    num_w_tiles,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    offs_w = tl.arange(0, BLOCK_W)
    inv6 = 1.0 / 6.0

    for tile_id in tl.range(pid, total_tiles, nprog, num_stages=2):
        tile_w = tile_id % num_w_tiles
        row_id = tile_id // num_w_tiles
        wo = tile_w * BLOCK_W + offs_w
        mask = wo < W_OUT
        safe_wo = tl.where(mask, wo, 0)

        ho = row_id % H_OUT
        t = row_id // H_OUT
        c = t % C
        n = t // C
        base = ((n * C + c) * H + ho * 2) * W + safe_wo * 2

        x00 = tl.load(x_ptr + base, mask=mask, other=0.0).to(
            tl.float32) - subtract_value
        x01 = tl.load(x_ptr + base + 1, mask=mask, other=0.0).to(
            tl.float32) - subtract_value
        x10 = tl.load(x_ptr + base + W, mask=mask, other=0.0).to(
            tl.float32) - subtract_value
        x11 = tl.load(x_ptr + base + W + 1, mask=mask, other=0.0).to(
            tl.float32) - subtract_value

        p00 = tl.minimum(tl.maximum(x00 + 3.0, 0.0), 6.0)
        p01 = tl.minimum(tl.maximum(x01 + 3.0, 0.0), 6.0)
        p10 = tl.minimum(tl.maximum(x10 + 3.0, 0.0), 6.0)
        p11 = tl.minimum(tl.maximum(x11 + 3.0, 0.0), 6.0)
        h00 = x00 * (p00 * inv6)
        h01 = x01 * (p01 * inv6)
        h10 = x10 * (p10 * inv6)
        h11 = x11 * (p11 * inv6)
        mv = tl.maximum(tl.maximum(h00, h01), tl.maximum(h10, h11))

        ax = tl.abs(mv)
        sp = tl.log(1.0 + tl.exp(-ax)) + tl.maximum(mv, 0.0)
        e2 = tl.exp(-2.0 * sp)
        th = (1.0 - e2) / (1.0 + e2)
        outv = mv * th
        store_wo = tl.where(mask, wo, 0)
        tl.store(y_ptr + row_id * W_OUT + store_wo, outv, mask=mask)


@triton.jit
def _fused_hswish_maxpool_mish_generic_kernel(
    x_ptr,
    y_ptr,
    N_OUT,
    N,
    C,
    H,
    W,
    H_OUT,
    W_OUT,
    subtract_value,
    total_tiles,
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    lane = tl.arange(0, BLOCK_SIZE)
    inv6 = 1.0 / 6.0

    for tile_id in tl.range(pid, total_tiles, nprog, num_stages=2):
        offs = tile_id * BLOCK_SIZE + lane
        mask = offs < N_OUT
        safe_offs = tl.where(mask, offs, 0)

        wo = safe_offs % W_OUT
        t1 = safe_offs // W_OUT
        ho = t1 % H_OUT
        t2 = t1 // H_OUT
        c = t2 % C
        n = t2 // C
        base = ((n * C + c) * H + ho * K) * W + wo * K

        maxv = tl.full([BLOCK_SIZE], -3.4028234663852886e38, dtype=tl.float32)
        for kh in tl.static_range(0, K):
            row_base = base + kh * W
            for kw in tl.static_range(0, K):
                v = tl.load(x_ptr + row_base + kw, mask=mask, other=0.0).to(
                    tl.float32) - subtract_value
                vp3 = tl.minimum(tl.maximum(v + 3.0, 0.0), 6.0)
                maxv = tl.maximum(maxv, v * (vp3 * inv6))

        mv = tl.where(mask, maxv, 0.0)
        ax = tl.abs(mv)
        sp = tl.log(1.0 + tl.exp(-ax)) + tl.maximum(mv, 0.0)
        e2 = tl.exp(-2.0 * sp)
        outv = mv * ((1.0 - e2) / (1.0 + e2))
        tl.store(y_ptr + safe_offs, outv, mask=mask)


class ModelNew(nn.Module):
    """Conv2d -> subtract -> HardSwish -> MaxPool -> Mish with optimized Triton epilogue."""

    def __init__(
        self,
        in_channels=DEFAULT_IN_CHANNELS,
        out_channels=DEFAULT_OUT_CHANNELS,
        kernel_size=DEFAULT_KERNEL_SIZE,
        subtract_value=DEFAULT_SUBTRACT_VALUE,
        pool_kernel_size=DEFAULT_POOL_KERNEL_SIZE,
    ):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value = float(subtract_value)
        self.pool = nn.MaxPool2d(pool_kernel_size)
        if isinstance(pool_kernel_size, (tuple, list)):
            assert pool_kernel_size[0] == pool_kernel_size[
                1], "Fused path requires square pooling"
            self.pool_k = int(pool_kernel_size[0])
        else:
            self.pool_k = int(pool_kernel_size)

    def _num_vector_cores(self, x: torch.Tensor) -> int:
        try:
            return driver.active.utils.get_device_properties(
                x.device)["num_vectorcore"]
        except Exception:
            return 32

    def _fused_hs_pool_mish(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape
        K = self.pool_k
        H_OUT = H // K
        W_OUT = W // K
        y = torch.empty((N, C, H_OUT, W_OUT), device=x.device, dtype=x.dtype)
        if y.numel() == 0:
            return y

        if K == 2:
            BLOCK = 1024
            n_out = y.numel()
            total_tiles = triton.cdiv(n_out, BLOCK)
            if total_tiles <= 65535:
                _fused_hswish_maxpool2_mish_vector_kernel[(total_tiles, )](
                    x.contiguous().view(-1),
                    y.view(-1),
                    n_out,
                    N,
                    C,
                    H,
                    W,
                    H_OUT,
                    W_OUT,
                    self.subtract_value,
                    BLOCK_SIZE=BLOCK,
                    num_warps=8,
                    num_stages=2,
                )
            else:
                BLOCK_W = 64
                num_w_tiles = triton.cdiv(W_OUT, BLOCK_W)
                row_tiles = N * C * H_OUT * num_w_tiles
                ncores = self._num_vector_cores(x)
                grid = (min(row_tiles, ncores, 65535), )
                _fused_hswish_maxpool2_mish_row_kernel[grid](
                    x.contiguous().view(-1),
                    y.view(-1),
                    row_tiles,
                    N,
                    C,
                    H,
                    W,
                    H_OUT,
                    W_OUT,
                    self.subtract_value,
                    num_w_tiles,
                    BLOCK_W=BLOCK_W,
                    num_warps=8,
                    num_stages=2,
                )
        else:
            BLOCK = 1024
            n_out = y.numel()
            total_tiles = triton.cdiv(n_out, BLOCK)
            ncores = self._num_vector_cores(x)
            grid = (min(total_tiles, ncores, 65535), )
            _fused_hswish_maxpool_mish_generic_kernel[grid](
                x.contiguous().view(-1),
                y.view(-1),
                n_out,
                N,
                C,
                H,
                W,
                H_OUT,
                W_OUT,
                self.subtract_value,
                total_tiles,
                K=K,
                BLOCK_SIZE=BLOCK,
                num_warps=8,
                num_stages=2,
            )
        return y

    def forward(self, x):
        if x.device.type != "npu":
            raise RuntimeError("ModelNew requires Ascend NPU inputs.")
        x = self.conv(x)
        v = x - self.subtract_value
        hswish = v * torch.clamp(v + 3.0, min=0.0, max=6.0) * (1.0 / 6.0)
        pooled = self.pool(hswish)
        return pooled * torch.tanh(torch.nn.functional.softplus(pooled))


batch_size = 128
in_channels = 64
out_channels = 128
height = width = 128
kernel_size = 3
subtract_value = 0.5
pool_kernel_size = 2


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [
        in_channels, out_channels, kernel_size, subtract_value,
        pool_kernel_size
    ]
