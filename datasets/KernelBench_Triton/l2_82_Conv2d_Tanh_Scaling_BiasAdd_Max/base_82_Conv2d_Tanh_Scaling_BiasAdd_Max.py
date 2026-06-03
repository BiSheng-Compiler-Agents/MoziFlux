import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def _fused_tanh_scale_bias_maxpool2d(
    x_ptr,                # *f32 [B, C, H, W]
    bias_ptr,             # *f32 [C]
    y_ptr,                # *f32 [B, C, Hpo, Wpo]
    B, C, H, W,           # input dims
    HPO, WPO,             # pooled output dims
    STRIDE_B, STRIDE_C, STRIDE_H, STRIDE_W,       # input strides (in elements)
    O_STRIDE_B, O_STRIDE_C, O_STRIDE_H, O_STRIDE_W,  # output strides (in elements)
    scale,                # float scaling factor
    POOL_K: tl.constexpr, # pooling kernel size (assume stride=POOL_K, padding=0, ceil_mode=False)
    SCALE_NONNEG: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_bc = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    b = pid_bc // C
    c = pid_bc % C

    # Tiled coordinates in pooled space
    oh = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)[:, None]
    ow = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)[None, :]

    mask_hw = (oh < HPO) & (ow < WPO)

    # Base pointers for current (b, c) plane
    x_base = x_ptr + b * STRIDE_B + c * STRIDE_C
    y_base = y_ptr + b * O_STRIDE_B + c * O_STRIDE_C

    # Reduce tanh values first, then apply the uniform scalar once after pooling.
    if SCALE_NONNEG:
        acc = tl.full((BLOCK_H, BLOCK_W), -float("inf"), tl.float32)
    else:
        acc = tl.full((BLOCK_H, BLOCK_W), float("inf"), tl.float32)

    # Precompute start indices in input feature map for each pooled output position
    ih0 = oh * POOL_K
    iw0 = ow * POOL_K
    ih0s = ih0 * STRIDE_H
    iw0s = iw0 * STRIDE_W
    pool_base = x_base + ih0s + iw0s

    # Iterate over POOL_K x POOL_K window
    for kh in range(POOL_K):
        row_base = pool_base + kh * STRIDE_H
        for kw in range(POOL_K):
            ptrs = row_base + kw * STRIDE_W
            vals = tl.load(ptrs, mask=mask_hw, other=0.0).to(tl.float32)

            tanh_x = tl.tanh(vals)

            if SCALE_NONNEG:
                acc = tl.maximum(acc, tanh_x)
            else:
                acc = tl.minimum(acc, tanh_x)

    # Add per-channel bias after max-pooling (equivalent since bias is constant per channel)
    bias_val = tl.load(bias_ptr + c).to(tl.float32)
    acc = acc * scale + bias_val

    # Store results
    out_ptrs = y_base + oh * O_STRIDE_H + ow * O_STRIDE_W
    tl.store(out_ptrs, acc, mask=mask_hw)


class ModelNew(nn.Module):
    """
    A model that performs a convolution, applies tanh, scaling, adds a bias term, and then max-pools.
    Fused Triton kernel computes tanh + scale + bias + max-pooling on CUDA for speed.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scaling_factor = float(scaling_factor)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.max_pool = nn.MaxPool2d(pool_kernel_size)
        self._pool_k = pool_kernel_size if isinstance(pool_kernel_size, int) else pool_kernel_size[0]

    def forward(self, x):
        # Convolution
        x = self.conv(x)

        # Triton fast path on accelerators that support the compiled kernel.
        if x.is_cuda or x.device.type == "npu":
            B, C, H, W = x.shape
            K = self._pool_k
            # Output dims with stride=K, padding=0, ceil_mode=False
            HPO = H // K
            WPO = W // K

            y = torch.empty((B, C, HPO, WPO), device=x.device, dtype=x.dtype)

            # Flatten bias to [C]
            bias_flat = self.bias.view(C).contiguous()

            # Strides in elements
            sb, sc, sh, sw = x.stride()
            ob, oc, oh, ow = y.stride()

            # Keep the launch grid under the Ascend runtime limit for this shape.
            BLOCK_H = 63
            BLOCK_W = 40
            grid = (B * C, triton.cdiv(HPO, BLOCK_H), triton.cdiv(WPO, BLOCK_W))

            scale_nonneg = self.scaling_factor >= 0.0

            _fused_tanh_scale_bias_maxpool2d[grid](
                x, bias_flat, y,
                B, C, H, W,
                HPO, WPO,
                sb, sc, sh, sw,
                ob, oc, oh, ow,
                self.scaling_factor,
                POOL_K=K,
                SCALE_NONNEG=scale_nonneg,
                BLOCK_H=BLOCK_H,
                BLOCK_W=BLOCK_W,
                num_warps=4,
                num_stages=2,
            )
            return y
        else:
            # CPU fallback: identical semantics
            x = torch.tanh(x)
            x = x * self.scaling_factor
            x = x + self.bias
            x = self.max_pool(x)
            return x
batch_size = 128
in_channels = 8
out_channels = 64
height, width = 256, 256
kernel_size = 3
scaling_factor = 2.0
bias_shape = (out_channels, 1, 1)
pool_kernel_size = 4

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size]
