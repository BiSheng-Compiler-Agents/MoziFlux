import torch
import torch_npu  # noqa: F401
import torch.nn as nn
import triton
import triton.language as tl


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False) or x.device.type == "npu")


@triton.jit
def _pool_sigmoid_channel_kernel(
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
    K: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // C
    c = pid % C

    if b >= B:
        return

    H_OUT = H // K
    W_OUT = W // K
    inv_area = 1.0 / (K * K)

    base = x_ptr + b * STRIDE_B + c * STRIDE_C
    total = tl.zeros((BLOCK_W,), dtype=tl.float32)
    lane = tl.arange(0, BLOCK_W)

    for h_out in range(0, H_OUT):
        h_base = h_out * K
        for w_start in range(0, W_OUT, BLOCK_W):
            w_out = w_start + lane
            mask = w_out < W_OUT
            acc = tl.zeros((BLOCK_W,), dtype=tl.float32)
            for ky in tl.static_range(0, K):
                row = base + (h_base + ky) * STRIDE_H
                for kx in tl.static_range(0, K):
                    ptrs = row + (w_out * K + kx) * STRIDE_W
                    vals = tl.load(ptrs, mask=mask, other=0.0)
                    acc += vals
            avg = acc * inv_area
            sig = 1.0 / (1.0 + tl.exp(-avg))
            total += tl.where(mask, sig, 0.0)

    partial = tl.sum(total, axis=0)
    tl.store(out_ptr + b * OUT_STRIDE_B + c * OUT_STRIDE_C, partial)


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
    """
    This model performs a convolution, average pooling, applies sigmoid, and sums the result.
    The Triton path fuses pool+sigmoid per channel, then reduces channels in a second Triton kernel.
    """

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

        block_w = 128
        block_c = 128
        if C <= 64:
            block_c = 64

        grid_bc = (B * C,)
        _pool_sigmoid_channel_kernel[grid_bc](
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
            K=K,
            BLOCK_W=block_w,
            num_warps=4,
            num_stages=2,
        )

        _sum_channels_kernel[(B,)](
            partial,
            out,
            C,
            partial.stride(0),
            partial.stride(1),
            BLOCK_C=block_c,
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
