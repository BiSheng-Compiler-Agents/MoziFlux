import torch
import torch.nn as nn
import triton
import triton.language as tl

DEFAULT_BATCH_SIZE = 128
DEFAULT_IN_CHANNELS = 64
DEFAULT_OUT_CHANNELS = 128
DEFAULT_HEIGHT = 128
DEFAULT_WIDTH = 128
DEFAULT_KERNEL_SIZE = 3
DEFAULT_SUBTRACT_VALUE = 0.5
DEFAULT_POOL_KERNEL_SIZE = 2


@triton.jit
def _sub_hswish_kernel(x_ptr, out_ptr, N, subtract_value, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(x_ptr + offs, mask=mask, other=0).to(tl.float32)
    v = x - subtract_value
    # HardSwish: v * clamp(v + 3, 0, 6) / 6
    vp3 = v + 3.0
    vp3 = tl.minimum(tl.maximum(vp3, 0.0), 6.0)
    y = v * (vp3 * (1.0 / 6.0))
    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def _mish_kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(x_ptr + offs, mask=mask, other=0).to(tl.float32)
    # Stable softplus: log1p(exp(-abs(x))) + max(x, 0)
    ax = tl.abs(x)
    sp = tl.log(1.0 + tl.exp(-ax)) + tl.maximum(x, 0.0)
    # tanh(sp) = (1 - exp(-2*sp)) / (1 + exp(-2*sp))
    e2 = tl.exp(-2.0 * sp)
    th = (1.0 - e2) / (1.0 + e2)
    y = x * th
    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def _fused_hswish_maxpool_mish_kernel(
    x_ptr,                  # *flat* input pointer (N*C*H*W)
    y_ptr,                  # *flat* output pointer (N*C*H_out*W_out)
    N_OUT,                  # total number of output elements
    N, C, H, W,             # input dims
    H_OUT, W_OUT,           # output spatial dims
    subtract_value,         # scalar
    K: tl.constexpr,        # pooling kernel size (square), stride = K
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_OUT

    # Decompose flat output index -> (n, c, ho, wo)
    wo = offs % W_OUT
    t1 = offs // W_OUT
    ho = t1 % H_OUT
    t2 = t1 // H_OUT
    c = t2 % C
    n = t2 // C

    # Base input offset for the top-left of the pooling window
    in_h0 = ho * K
    in_w0 = wo * K
    base = ((n * C + c) * H + in_h0) * W + in_w0

    # Initialize running max with a very negative value for valid lanes
    maxv = tl.full([BLOCK_SIZE], -1e30, dtype=tl.float32)

    # Iterate over KxK window, apply HardSwish on-the-fly, then reduce max
    inv6 = 1.0 / 6.0
    for kh in tl.static_range(K):
        row_base = base + kh * W
        for kw in tl.static_range(K):
            v = tl.load(x_ptr + row_base + kw, mask=mask, other=0).to(tl.float32) - subtract_value
            vp3 = v + 3.0
            vp3 = tl.minimum(tl.maximum(vp3, 0.0), 6.0)
            hs = v * (vp3 * inv6)
            maxv = tl.maximum(maxv, hs)

    # Apply Mish on the pooled result for valid outputs only, avoid NaNs for invalid lanes
    mv = tl.where(mask, maxv, 0.0)
    ax = tl.abs(mv)
    sp = tl.log(1.0 + tl.exp(-ax)) + tl.maximum(mv, 0.0)
    e2 = tl.exp(-2.0 * sp)
    th = (1.0 - e2) / (1.0 + e2)
    outv = mv * th

    tl.store(y_ptr + offs, outv, mask=mask)


class ModelNew(nn.Module):
    """
    Model that performs a convolution, subtracts a value, applies HardSwish, MaxPool, and Mish activation functions.
    """
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
        # cache pooling kernel size (assume square for fused path)
        if isinstance(pool_kernel_size, (tuple, list)):
            assert pool_kernel_size[0] == pool_kernel_size[1], "Fused path requires square pooling"
            self.pool_k = int(pool_kernel_size[0])
        else:
            self.pool_k = int(pool_kernel_size)

    def _sub_hswish_triton(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.empty_like(x)
        N = y.numel()
        if N == 0:
            return y
        BLOCK = 4096
        grid = (triton.cdiv(N, BLOCK),)
        _sub_hswish_kernel[grid](
            x.view(-1), y.view(-1),
            N,
            self.subtract_value,
            BLOCK_SIZE=BLOCK,
            num_warps=8,
            num_stages=2,
        )
        return y

    def _mish_triton(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.empty_like(x)
        N = y.numel()
        if N == 0:
            return y
        BLOCK = 4096
        grid = (triton.cdiv(N, BLOCK),)
        _mish_kernel[grid](
            x.view(-1), y.view(-1),
            N,
            BLOCK_SIZE=BLOCK,
            num_warps=8,
            num_stages=2,
        )
        return y

    def _fused_hs_pool_mish(self, x: torch.Tensor) -> torch.Tensor:
        # x is output of conv: (N, C, H, W), contiguous float32 CUDA
        N, C, H, W = x.shape
        K = self.pool_k
        H_OUT = H // K
        W_OUT = W // K
        y = torch.empty((N, C, H_OUT, W_OUT), device=x.device, dtype=x.dtype)
        N_OUT = y.numel()
        if N_OUT == 0:
            return y
        BLOCK = 4096
        grid = (triton.cdiv(N_OUT, BLOCK),)
        _fused_hswish_maxpool_mish_kernel[grid](
            x.view(-1),
            y.view(-1),
            N_OUT, N, C, H, W, H_OUT, W_OUT,
            self.subtract_value,
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
        return self._fused_hs_pool_mish(x.contiguous())
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
    return [in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size]