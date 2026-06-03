import torch
import torch.nn as nn
import triton
import triton.language as tl

STATS_BLOCK_HW = 256
APPLY_BLOCK_HW = 1024
POOL_BLOCK_W = 32


@triton.jit
def _groupnorm_stats_kernel(
    x_ptr,
    mean_ptr,
    rstd_ptr,
    B,
    C,
    H,
    W,
    G,
    eps,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G

    hw = tl.arange(0, BLOCK_HW)
    HW = H * W
    base = n * C * HW + g * 4 * HW

    s = tl.zeros([1], dtype=tl.float32)
    ss = tl.zeros([1], dtype=tl.float32)

    start = 0
    while start < HW:
        idx = start + hw
        mask = idx < HW

        x0 = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        x1 = tl.load(x_ptr + base + HW + idx, mask=mask, other=0.0)
        x2 = tl.load(x_ptr + base + 2 * HW + idx, mask=mask, other=0.0)
        x3 = tl.load(x_ptr + base + 3 * HW + idx, mask=mask, other=0.0)

        block_sum = x0 + x1 + x2 + x3
        block_sq_sum = x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3
        s += tl.sum(block_sum, axis=0)
        ss += tl.sum(block_sq_sum, axis=0)
        start += BLOCK_HW

    denom = tl.full([1], 4 * HW, dtype=tl.float32)
    mean = s / denom
    var = ss / denom - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    scalar_offs = tl.arange(0, 1)
    tl.store(mean_ptr + n * G + g + scalar_offs, mean, mask=scalar_offs == 0)
    tl.store(rstd_ptr + n * G + g + scalar_offs, rstd, mask=scalar_offs == 0)


def _launch_groupnorm_stats(x, mean, rstd, B, C, H, W, num_groups, eps):
    grid_stats = (B * num_groups,)
    _groupnorm_stats_kernel[grid_stats](
        x,
        mean,
        rstd,
        B,
        C,
        H,
        W,
        num_groups,
        float(eps),
        BLOCK_HW=STATS_BLOCK_HW,
        num_warps=8,
        num_stages=2,
    )


@triton.jit
def _groupnorm_apply_pool_clamp_kernel(
    x_ptr, y_ptr,            # *f32
    mean_ptr, rstd_ptr,      # *f32
    gamma_ptr, beta_ptr,     # *f32 (already fused with scale)
    B, C, H, W, G,           # i32
    Ho, Wo,                  # i32
    clamp_min, clamp_max,    # f32
    K: tl.constexpr,         # kernel size (square)
    STRIDE: tl.constexpr,    # stride == kernel size
    BLOCK_W: tl.constexpr,   # number of output columns processed per program
):
    # Grid: (B*C, Ho, ceil(Wo/BLOCK_W))
    pid_nc = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_w = tl.program_id(axis=2)

    c = pid_nc % C
    n = pid_nc // C
    h_out = pid_h

    # Group info
    Cpg = C // G
    g = c // Cpg

    # Strides
    in_plane_stride = H * W
    out_plane_stride = Ho * Wo
    in_base_nc = (n * C + c) * in_plane_stride
    out_base_nc = (n * C + c) * out_plane_stride

    # Scalars per (n, c)
    mean = tl.load(mean_ptr + n * G + g)
    rstd = tl.load(rstd_ptr + n * G + g)
    gamma = tl.load(gamma_ptr + c)
    beta = tl.load(beta_ptr + c)
    gamma_pos = tl.full([BLOCK_W], gamma, dtype=tl.float32) >= 0

    # Output column offsets
    w_offsets = tl.arange(0, BLOCK_W)
    w_out = pid_w * BLOCK_W + w_offsets
    w_out_mask = w_out < Wo

    tl.max_contiguous(w_out, BLOCK_W)
    tl.multiple_of(w_out, BLOCK_W)

    # Corresponding input row start
    h_in_base = h_out * STRIDE

    # We'll accumulate extreme of raw x (pre-normalization) depending on sign(gamma)
    # This allows postponing the affine to only the selected extreme element.
    if K == 2:
        h0 = h_in_base
        h1 = h_in_base + 1

        h0_mask = tl.full([BLOCK_W], h0, dtype=tl.int32) < H
        h1_mask = tl.full([BLOCK_W], h1, dtype=tl.int32) < H

        w_in0 = w_out * STRIDE
        w_in1 = w_in0 + 1

        acc_max = tl.full([BLOCK_W], -float("inf"), dtype=tl.float32)
        acc_min = tl.full([BLOCK_W], float("inf"), dtype=tl.float32)

        # Row 0
        row0_base = in_base_nc + h0 * W
        mask0a = w_out_mask & h0_mask & (w_in0 < W)
        mask0b = w_out_mask & h0_mask & (w_in1 < W)
        v0a = tl.load(x_ptr + row0_base + w_in0, mask=mask0a, other=0.0)
        v0b = tl.load(x_ptr + row0_base + w_in1, mask=mask0b, other=0.0)
        acc_max = tl.where(mask0a, tl.maximum(acc_max, v0a), acc_max)
        acc_min = tl.where(mask0a, tl.minimum(acc_min, v0a), acc_min)
        acc_max = tl.where(mask0b, tl.maximum(acc_max, v0b), acc_max)
        acc_min = tl.where(mask0b, tl.minimum(acc_min, v0b), acc_min)

        # Row 1
        row1_base = in_base_nc + h1 * W
        mask1a = w_out_mask & h1_mask & (w_in0 < W)
        mask1b = w_out_mask & h1_mask & (w_in1 < W)
        v1a = tl.load(x_ptr + row1_base + w_in0, mask=mask1a, other=0.0)
        v1b = tl.load(x_ptr + row1_base + w_in1, mask=mask1b, other=0.0)
        acc_max = tl.where(mask1a, tl.maximum(acc_max, v1a), acc_max)
        acc_min = tl.where(mask1a, tl.minimum(acc_min, v1a), acc_min)
        acc_max = tl.where(mask1b, tl.maximum(acc_max, v1b), acc_max)
        acc_min = tl.where(mask1b, tl.minimum(acc_min, v1b), acc_min)

        # Select based on sign of gamma
        acc_v = tl.where(gamma_pos, acc_max, acc_min)
        acc = ((acc_v - mean) * rstd) * gamma + beta
    else:
        # Generic KxK pooling path, choose extreme in raw x then apply affine once.
        acc_max = tl.full([BLOCK_W], -float("inf"), dtype=tl.float32)
        acc_min = tl.full([BLOCK_W], float("inf"), dtype=tl.float32)
        for kh in tl.static_range(0, K):
            h_in = h_in_base + kh
            h_mask = tl.full([BLOCK_W], h_in, dtype=tl.int32) < H
            row_h_base = in_base_nc + h_in * W
            for kw in tl.static_range(0, K):
                w_in = w_out * STRIDE + kw
                in_mask = w_out_mask & h_mask & (w_in < W)
                v = tl.load(x_ptr + row_h_base + w_in, mask=in_mask, other=0.0)
                acc_max = tl.where(in_mask, tl.maximum(acc_max, v), acc_max)
                acc_min = tl.where(in_mask, tl.minimum(acc_min, v), acc_min)
        acc_v = tl.where(gamma_pos, acc_max, acc_min)
        acc = ((acc_v - mean) * rstd) * gamma + beta

    # Clamp
    acc = tl.minimum(acc, clamp_max)
    acc = tl.maximum(acc, clamp_min)

    # Store
    out_row_base = out_base_nc + h_out * Wo
    tl.store(y_ptr + out_row_base + w_out, acc, mask=w_out_mask)


@triton.jit
def _groupnorm_apply_kernel(
    x_ptr, y_ptr,            # *f32
    mean_ptr, rstd_ptr,      # *f32
    gamma_ptr, beta_ptr,     # *f32
    B, C, H, W, G,           # i32
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)  # linear over (B, C)
    n = pid // C
    c = pid % C

    HW = H * W
    Cpg = C // G
    g = c // Cpg

    base = (n * C + c) * HW
    mean = tl.load(mean_ptr + n * G + g)
    rstd = tl.load(rstd_ptr + n * G + g)
    gamma = tl.load(gamma_ptr + c)
    beta = tl.load(beta_ptr + c)

    offs = tl.arange(0, BLOCK_HW)
    start = 0
    while start < HW:
        idx = start + offs
        mask = idx < HW
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        y = ((x - mean) * rstd) * gamma + beta
        tl.store(y_ptr + base + idx, y, mask=mask)
        start += BLOCK_HW


def _group_norm_scale_triton(x: torch.Tensor, gamma_scaled: torch.Tensor, beta_scaled: torch.Tensor,
                             num_groups: int, eps: float) -> torch.Tensor:
    # x: (B, C, H, W) contiguous float32 on Ascend NPU
    B, C, H, W = x.shape
    device = x.device
    y = torch.empty_like(x, dtype=torch.float32)

    mean = torch.empty((B, num_groups), device=device, dtype=torch.float32)
    rstd = torch.empty((B, num_groups), device=device, dtype=torch.float32)

    _launch_groupnorm_stats(x, mean, rstd, B, C, H, W, num_groups, eps)

    # Apply normalization and affine (with scale fused)
    grid_apply = (B * C,)
    _groupnorm_apply_kernel[grid_apply](
        x, y, mean, rstd, gamma_scaled, beta_scaled,
        B, C, H, W, num_groups,
        BLOCK_HW=APPLY_BLOCK_HW,
        num_warps=4, num_stages=2
    )
    return y


def _group_norm_pool_clamp_triton(x: torch.Tensor, gamma_scaled: torch.Tensor, beta_scaled: torch.Tensor,
                                  num_groups: int, eps: float,
                                  K: int, STRIDE: int, clamp_min: float, clamp_max: float,
                                  Ho: int, Wo: int) -> torch.Tensor:
    # x: (B, C, H, W)
    B, C, H, W = x.shape
    device = x.device
    dtype = torch.float32

    # Allocate outputs and intermediate stats
    y = torch.empty((B, C, Ho, Wo), device=device, dtype=dtype)
    mean = torch.empty((B, num_groups), device=device, dtype=torch.float32)
    rstd = torch.empty((B, num_groups), device=device, dtype=torch.float32)

    _launch_groupnorm_stats(x, mean, rstd, B, C, H, W, num_groups, eps)

    # Apply + pool + clamp
    BLOCK_W = POOL_BLOCK_W
    grid_apply = (B * C, Ho, triton.cdiv(Wo, BLOCK_W))
    _groupnorm_apply_pool_clamp_kernel[grid_apply](
        x, y, mean, rstd, gamma_scaled, beta_scaled,
        B, C, H, W, num_groups,
        Ho, Wo,
        float(clamp_min), float(clamp_max),
        K=K, STRIDE=STRIDE, BLOCK_W=BLOCK_W,
        num_warps=8, num_stages=3
    )
    return y


class ModelNew(nn.Module):
    """
    Model that performs convolution, group normalization, scaling, max pooling, and clamping.
    Fuses GroupNorm + Scale + MaxPool + Clamp via Triton on Ascend NPU.
    """
    def __init__(self, in_channels=None, out_channels=None, kernel_size=None, num_groups=None,
                 scale_shape=None, maxpool_kernel_size=None, clamp_min=None, clamp_max=None):
        super(ModelNew, self).__init__()
        in_channels = in_channels if in_channels is not None else globals()["in_channels"]
        out_channels = out_channels if out_channels is not None else globals()["out_channels"]
        kernel_size = kernel_size if kernel_size is not None else globals()["kernel_size"]
        num_groups = num_groups if num_groups is not None else globals()["num_groups"]
        scale_shape = scale_shape if scale_shape is not None else globals()["scale_shape"]
        maxpool_kernel_size = (
            maxpool_kernel_size if maxpool_kernel_size is not None else globals()["maxpool_kernel_size"]
        )
        clamp_min = clamp_min if clamp_min is not None else globals()["clamp_min"]
        clamp_max = clamp_max if clamp_max is not None else globals()["clamp_max"]
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel_size)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width).
        Returns:
            Output tensor of shape (batch_size, out_channels, height', width').
        """
        if x.device.type != "npu":
            raise RuntimeError("ModelNew requires Ascend NPU inputs for the Triton execution path.")
        if self.training or x.requires_grad:
            raise RuntimeError("ModelNew Triton path supports inference-only execution without autograd.")

        x = self.conv(x)

        _, C, _, _ = x.shape
        x = x.contiguous()

        # Prepare affine parameters with scale fused: (z*gamma + beta) * scale
        scale_flat = self.scale.view(-1)
        if self.group_norm.affine:
            gamma = self.group_norm.weight
            beta = self.group_norm.bias
        else:
            gamma = torch.ones(C, device=x.device, dtype=x.dtype)
            beta = torch.zeros(C, device=x.device, dtype=x.dtype)
        gamma_scaled = (gamma * scale_flat).to(dtype=torch.float32, device=x.device).contiguous()
        beta_scaled = (beta * scale_flat).to(dtype=torch.float32, device=x.device).contiguous()

        x = _group_norm_scale_triton(
            x.to(torch.float32),
            gamma_scaled,
            beta_scaled,
            self.group_norm.num_groups,
            self.group_norm.eps,
        )
        x = self.maxpool(x)
        x = torch.clamp(x, self.clamp_min, self.clamp_max)
        return x
batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128 
kernel_size = 3
num_groups = 16
scale_shape = (out_channels, 1, 1)
maxpool_kernel_size = 4
clamp_min = 0.0
clamp_max = 1.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max]
