import torch
import torch.nn as nn
import triton
import triton.language as tl

batch_size = 1024
input_size = 8192
hidden_size = 8192
num_groups = 512


@triton.jit
def fused_linear_groupnorm_lrelu_double(
    x_ptr,  # float32 [N, K]
    w_ptr,  # float32 [C, K]
    b_ptr,  # float32 [C]
    gamma_ptr,  # float32 [C]
    beta_ptr,  # float32 [C]
    y_ptr,  # float32 [N, C]
    N: tl.constexpr,  # batch size
    C: tl.constexpr,  # hidden size (channels)
    K: tl.constexpr,  # input size
    groups: tl.constexpr,  # number of groups
    eps,  # float32
    neg_slope,  # float32
    stride_xm,
    stride_xk,
    stride_wc,
    stride_wk,
    stride_ym,
    stride_yc,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,  # set at launch to group_size = C // groups
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)  # tile along batch dimension
    pid_g = tl.program_id(axis=1)  # group id

    # Offsets along M and within the group's channels
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < N

    group_size = BLOCK_N
    c0 = pid_g * group_size
    offs_n = tl.arange(0, BLOCK_N)
    c_idx = c0 + offs_n
    mask_n = c_idx < C

    # Accumulator for matmul: [BM, BN]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K-loop: stream from global with L2-only cache (.cg)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # X tile [BM, BK]
        x_ptrs = x_ptr + (offs_m[:, None] * stride_xm +
                          offs_k[None, :] * stride_xk)
        x_tile = tl.load(x_ptrs,
                         mask=(mask_m[:, None] & mask_k[None, :]),
                         other=0.0,
                         cache_modifier=".cg")

        # W tile [BK, BN] from layout [C, K]
        w_ptrs = w_ptr + (offs_k[:, None] * stride_wk +
                          c_idx[None, :] * stride_wc)
        w_tile = tl.load(w_ptrs,
                         mask=(mask_k[:, None] & mask_n[None, :]),
                         other=0.0,
                         cache_modifier=".cg")

        acc += tl.dot(x_tile, w_tile)

    # Add bias per-channel
    b = tl.load(b_ptr + c_idx, mask=mask_n, other=0.0)
    acc = acc + b[None, :]

    # GroupNorm across channels in the group per sample (row in M)
    # Use E[x^2] - E[x]^2 formulation for fewer FLOPs
    acc_masked = tl.where(mask_n[None, :], acc, 0.0)
    sum_vals = tl.sum(acc_masked, axis=1)
    mean = sum_vals / group_size

    sq = acc_masked * acc_masked
    sum_sq = tl.sum(sq, axis=1)
    ex2 = sum_sq / group_size
    var = tl.maximum(ex2 - mean * mean, 0.0)
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Normalize
    norm = (acc_masked - mean[:, None]) * inv_std[:, None]

    # Affine
    gamma = tl.load(gamma_ptr + c_idx, mask=mask_n, other=1.0)
    beta = tl.load(beta_ptr + c_idx, mask=mask_n, other=0.0)
    out = norm * gamma[None, :] + beta[None, :]

    # Fused LeakyReLU + doubling (x + x)
    pos_scale = 2.0
    neg_scale = 2.0 * neg_slope
    out = tl.where(out >= 0, out * pos_scale, out * neg_scale)

    # Store
    y_ptrs = y_ptr + (offs_m[:, None] * stride_ym + c_idx[None, :] * stride_yc)
    tl.store(y_ptrs, out, mask=(mask_m[:, None] & mask_n[None, :]))


class ModelNew(nn.Module):
    """
    A model that performs a matrix multiplication, group normalization, leaky ReLU activation, and element-wise sum.
    """

    def __init__(
        self,
        input_size=input_size,
        hidden_size=hidden_size,
        num_groups=num_groups,
        eps=1e-5,
        negative_slope=0.01,
    ):
        super(ModelNew, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.gn = nn.GroupNorm(num_groups=num_groups,
                               num_channels=hidden_size,
                               eps=eps)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, x):
        """
        Performs the forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Output tensor of shape (batch_size, hidden_size).
        """
        if x.device.type != "npu" or x.dtype not in (torch.float16,
                                                     torch.float32):
            raise RuntimeError(
                "ModelNew requires float16 or float32 inputs on Ascend NPU.")

        N, K = x.shape
        C = self.fc.out_features
        G = self.gn.num_groups
        assert C % G == 0, "hidden_size must be divisible by num_groups"
        group_size = C // G

        w = self.fc.weight
        b = self.fc.bias
        gamma = self.gn.weight if getattr(self.gn, "affine", True) else None
        beta = self.gn.bias if getattr(self.gn, "affine", True) else None

        target_dtype = x.dtype
        target_device = x.device
        x_c = x.contiguous()
        w_c = w.to(device=target_device, dtype=target_dtype).contiguous()
        b_c = (b if b is not None else torch.zeros(
            C, device=target_device, dtype=target_dtype)).to(
                device=target_device, dtype=target_dtype).contiguous()
        gamma_c = (gamma if gamma is not None else torch.ones(
            C, device=target_device, dtype=target_dtype)).to(
                device=target_device, dtype=target_dtype).contiguous()
        beta_c = (beta if beta is not None else torch.zeros(
            C, device=target_device, dtype=target_dtype)).to(
                device=target_device, dtype=target_dtype).contiguous()

        y = torch.empty((N, C), device=x.device, dtype=x.dtype)

        BLOCK_M = 64
        BLOCK_K = 128
        grid = (triton.cdiv(N, BLOCK_M), G)
        fused_linear_groupnorm_lrelu_double[grid](
            x_c,
            w_c,
            b_c,
            gamma_c,
            beta_c,
            y,
            N,
            C,
            K,
            G,
            self.gn.eps,
            self.leaky_relu.negative_slope,
            x_c.stride(0),
            x_c.stride(1),
            w_c.stride(0),
            w_c.stride(1),
            y.stride(0),
            y.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_N=group_size,
            BLOCK_K=BLOCK_K,
        )
        return y


batch_size = 1024
input_size = 8192
hidden_size = 8192
num_groups = 512


def get_inputs():
    return [torch.rand(batch_size, input_size)]


def get_init_inputs():
    return [input_size, hidden_size, num_groups]
