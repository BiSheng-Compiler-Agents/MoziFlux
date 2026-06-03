import math
import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def _softmax_pool2_fused_kernel(
    x_ptr, y_ptr,
    x_stride_n, x_stride_c, x_stride_d, x_stride_h,
    y_stride_n, y_stride_c, y_stride_d, y_stride_h,
    C, OD, OH, OW,
    K: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_OH: tl.constexpr,
    BLOCK_OW: tl.constexpr,
):
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)

    blocks_oh = tl.cdiv(OH, BLOCK_OH)
    oh_block = pid0 % blocks_oh
    t = pid0 // blocks_oh
    od = t % OD
    n = t // OD

    offs_c = tl.arange(0, BLOCK_C)
    offs_s = tl.arange(0, BLOCK_OH * BLOCK_OW)
    offs_oh = oh_block * BLOCK_OH + offs_s // BLOCK_OW
    offs_ow = pid1 * BLOCK_OW + offs_s % BLOCK_OW

    mask_c = offs_c < C
    mask_s = (offs_oh < OH) & (offs_ow < OW)
    tile_mask = mask_c[:, None] & mask_s[None, :]

    in_d0 = od * K
    base_ptrs = (
        n * x_stride_n
        + in_d0 * x_stride_d
        + offs_oh[None, :] * (K * x_stride_h)
        + offs_ow[None, :] * K
        + offs_c[:, None] * x_stride_c
    )
    acc = tl.full([BLOCK_C, BLOCK_OH * BLOCK_OW], -float("inf"), dtype=tl.float32)

    for kd in range(0, K):
        kd_ptrs = base_ptrs + kd * x_stride_d
        for kh in range(0, K):
            kh_ptrs = kd_ptrs + kh * x_stride_h
            for kw in range(0, K):
                x = tl.load(x_ptr + kh_ptrs + kw, mask=tile_mask, other=-float("inf")).to(tl.float32)
                x_max = tl.max(x, axis=0)
                x = tl.exp(x - x_max[None, :])
                x_sum = tl.sum(x, axis=0)
                y_tile = x / x_sum[None, :]
                acc = tl.maximum(acc, y_tile)

    out_ptrs = (
        y_ptr
        + n * y_stride_n
        + od * y_stride_d
        + offs_oh[None, :] * y_stride_h
        + offs_ow[None, :]
        + offs_c[:, None] * y_stride_c
    )
    tl.store(out_ptrs, acc, mask=tile_mask)


def _softmax_then_two_pools_fused_triton(x: torch.Tensor, pool_kernel_size: int) -> torch.Tensor:
    # x: [N, C, D, H, W], compute softmax along C then two MaxPool3d(K) (stride=K)
    # fused into one with Kf = K*K (stride=K*K).
    if x.ndim != 5:
        raise RuntimeError("Expected a 5D NCDHW tensor.")
    if x.device.type not in {"cuda", "npu"}:
        raise RuntimeError("Triton fused path requires CUDA or Ascend NPU tensors.")
    x = x.contiguous()
    N, C, D, H, W = x.shape
    K1 = int(pool_kernel_size)
    Kf = K1 * K1

    # Output dims for single pool with kernel=Kf, stride=Kf, padding=0 (ceil_mode=False)
    def odim(L, K):
        if L < K:
            return 0
        return (L - K) // K + 1

    OD = odim(D, Kf)
    OH = odim(H, Kf)
    OW = odim(W, Kf)

    y = torch.empty((N, C, OD, OH, OW), device=x.device, dtype=x.dtype)
    if OD == 0 or OH == 0 or OW == 0:
        return y

    xs = x.stride()
    ys = y.stride()

    block_c = 16 if C <= 16 else min(64, 1 << (C - 1).bit_length())
    block_oh = 4 if OH >= 4 else max(1, 1 << (OH - 1).bit_length())
    block_ow = 8 if OW <= 8 else min(16, 1 << (OW - 1).bit_length())

    grid = (N * OD * triton.cdiv(OH, block_oh), triton.cdiv(OW, block_ow))
    _softmax_pool2_fused_kernel[grid](
        x, y,
        xs[0], xs[1], xs[2], xs[3],
        ys[0], ys[1], ys[2], ys[3],
        C, OD, OH, OW,
        K=Kf,
        BLOCK_C=block_c,
        BLOCK_OH=block_oh,
        BLOCK_OW=block_ow,
        num_warps=4,
        num_stages=4,
    )
    return y


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, applies Softmax, and performs two max pooling operations.
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 16,
        kernel_size: int = 3,
        pool_kernel_size: int = 2,
    ):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, depth, height, width)
        Returns:
            Output tensor of shape (batch_size, out_channels, depth', height', width') where depth', height', width' are the dimensions after pooling.
        """
        if x.device.type != "npu":
            raise RuntimeError("ModelNew requires Ascend NPU input tensors.")

        x = self.conv(x)
        return _softmax_then_two_pools_fused_triton(x, self.pool_kernel_size)
batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
pool_kernel_size = 2

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]
def get_init_inputs():
    return [in_channels, out_channels, kernel_size, pool_kernel_size]
