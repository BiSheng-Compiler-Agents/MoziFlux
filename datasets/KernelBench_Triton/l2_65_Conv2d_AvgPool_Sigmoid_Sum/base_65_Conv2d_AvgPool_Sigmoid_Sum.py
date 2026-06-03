import torch
import torch_npu  # noqa: F401
import torch.nn as nn
import triton
import triton.language as tl


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False) or x.device.type == "npu")


@triton.jit
def _pool_sigmoid_channel_tile_kernel(
    x_ptr,
    out_ptr,
    B,
    C,
    H,
    W,
    STRIDE_B,
    STRIDE_C,
    STRIDE_H,
    STRIDE_W,
    OUT_STRIDE_B,
    OUT_STRIDE_C,
    NUM_C_TILES,
    K: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // NUM_C_TILES
    c_tile = pid % NUM_C_TILES

    if b >= B:
        return

    c_idx = c_tile * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_idx < C
    H_OUT = H // K
    W_OUT = W // K
    inv_area = 1.0 / (K * K)

    base = x_ptr + b * STRIDE_B + c_idx[:, None] * STRIDE_C
    total = tl.zeros((BLOCK_C,), dtype=tl.float32)
    lane_w = tl.arange(0, BLOCK_W)

    for h_out in range(0, H_OUT):
        h_base = h_out * K
        for w_start in range(0, W_OUT, BLOCK_W):
            w_out = w_start + lane_w
            w_mask = w_out < W_OUT
            acc = tl.zeros((BLOCK_C, BLOCK_W), dtype=tl.float32)
            w_in_left = w_out * K
            for ky in tl.static_range(0, K):
                row = base + (h_base + ky) * STRIDE_H
                for kx in tl.static_range(0, K):
                    ptrs = row + (w_in_left[None, :] + kx) * STRIDE_W
                    mask = c_mask[:, None] & w_mask[None, :]
                    vals = tl.load(ptrs, mask=mask, other=0.0)
                    acc += vals
            avg = acc * inv_area
            sig = 1.0 / (1.0 + tl.exp(-avg))
            total += tl.sum(tl.where(c_mask[:, None] & w_mask[None, :], sig, 0.0), axis=1)

    out_ptrs = out_ptr + b * OUT_STRIDE_B + c_idx * OUT_STRIDE_C
    tl.store(out_ptrs, total, mask=c_mask)


@triton.jit
def _sum_channels_kernel(
    x_ptr,
    out_ptr,
    C,
    STRIDE_B,
    STRIDE_C,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_C)
    mask = offs < C
    vals = tl.load(x_ptr + pid * STRIDE_B + offs * STRIDE_C, mask=mask, other=0.0)
    total = tl.sum(vals, axis=0)
    tl.store(out_ptr + pid, total)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.avg_pool = nn.AvgPool2d(pool_kernel_size)
        if isinstance(pool_kernel_size, tuple):
            assert len(pool_kernel_size) == 2 and pool_kernel_size[0] == pool_kernel_size[1], (
                "Fused Triton path supports only square pooling kernels."
            )
            self.pool_kernel_size = int(pool_kernel_size[0])
        else:
            self.pool_kernel_size = int(pool_kernel_size)

    def forward(self, x):
        y = self.conv(x)
        if not _is_npu_tensor(y):
            raise RuntimeError("ModelNew expects Ascend NPU tensors.")

        if y.dtype != torch.float32:
            y = y.to(torch.float32)

        y = y.contiguous()
        B, C, H, W = y.shape
        K = self.pool_kernel_size
        if H < K or W < K:
            raise ValueError("Pooling kernel size must not exceed the convolution output size.")

        partial = torch.empty((B, C), device=y.device, dtype=torch.float32)
        out = torch.empty((B,), device=y.device, dtype=torch.float32)
        num_c_tiles = triton.cdiv(C, 16)

        _pool_sigmoid_channel_tile_kernel[(B * num_c_tiles,)](
            y,
            partial,
            B,
            C,
            H,
            W,
            y.stride(0),
            y.stride(1),
            y.stride(2),
            y.stride(3),
            partial.stride(0),
            partial.stride(1),
            num_c_tiles,
            K=K,
            BLOCK_C=16,
            BLOCK_W=16,
            num_warps=2,
            num_stages=1,
        )

        _sum_channels_kernel[(B,)](
            partial,
            out,
            C,
            partial.stride(0),
            partial.stride(1),
            BLOCK_C=64,
            num_warps=4,
            num_stages=1,
        )
        return out


batch_size = 128
in_channels = 8
out_channels = 64
height, width = 384, 384
kernel_size = 3
pool_kernel_size = 4


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, pool_kernel_size]
