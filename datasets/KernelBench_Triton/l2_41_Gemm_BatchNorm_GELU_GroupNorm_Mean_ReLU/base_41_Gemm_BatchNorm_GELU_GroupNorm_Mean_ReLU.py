import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def _fused_gelu_groupnorm_mean_relu(
    x_ptr,               # [N, C]
    weight_ptr,          # [C]
    bias_ptr,            # [C]
    out_ptr,             # [N, 1]
    N,                   # int: batch size
    C,                   # int: number of features (channels)
    GROUP_SIZE,          # int: channels per group = C // NUM_GROUPS
    EPS: tl.constexpr,   # groupnorm eps (compile-time)
    NUM_GROUPS: tl.constexpr,  # number of groups (compile-time)
    BLOCK_SIZE: tl.constexpr,  # equals GROUP_SIZE (compile-time)
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < N
    row_ptrs = x_ptr + rows[:, None] * C
    total = tl.zeros((BLOCK_M,), dtype=tl.float32)
    offs = tl.arange(0, BLOCK_SIZE)
    inv_sqrt2 = 0.7071067811865476  # 1/sqrt(2)
    gs = tl.full((), GROUP_SIZE, tl.float32)

    tl.multiple_of(offs, 16)
    tl.max_contiguous(offs, BLOCK_SIZE)

    for g in tl.static_range(NUM_GROUPS):
        c_start = g * GROUP_SIZE
        cols = c_start + offs
        mask = row_mask[:, None]

        x = tl.load(row_ptrs + cols[None, :], mask=mask, other=0.0, cache_modifier=".cg").to(tl.float32)
        xg = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))

        sum_x = tl.sum(xg, axis=1)
        sum_x2 = tl.sum(xg * xg, axis=1)
        mu = sum_x / gs
        var = sum_x2 / gs - mu * mu
        rstd = tl.rsqrt(var + EPS)

        gamma = tl.load(weight_ptr + cols, cache_modifier=".ca").to(tl.float32)
        beta = tl.load(bias_ptr + cols, cache_modifier=".ca").to(tl.float32)
        sum_gamma_x = tl.sum(xg * gamma[None, :], axis=1)
        sum_gamma = tl.sum(gamma, axis=0)
        sum_beta = tl.sum(beta, axis=0)
        contrib = (sum_gamma_x - mu * sum_gamma) * rstd + sum_beta
        total += contrib

    invC = 1.0 / C
    mean_row = total * invC
    out_val = tl.maximum(mean_row, 0.0)
    tl.store(out_ptr + rows, out_val, mask=row_mask)


class ModelNew(nn.Module):
    """
    Model that performs a GEMM, BatchNorm, GELU, GroupNorm, Mean, and ReLU operations in sequence.
    Fuses GELU+GroupNorm+Mean+ReLU with a Triton kernel for improved performance.
    """
    def __init__(self, in_features=None, out_features=None, num_groups=None):
        super(ModelNew, self).__init__()
        in_features = in_features_default if in_features is None else in_features
        out_features = out_features_default if out_features is None else out_features
        num_groups = num_groups_default if num_groups is None else num_groups
        if out_features % num_groups != 0:
            raise ValueError("out_features must be divisible by num_groups")
        self.gemm = nn.Linear(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        x = self.gemm(x)
        x = self.batch_norm(x)

        if x.device.type != "npu":
            raise RuntimeError("ModelNew expects inputs on Ascend NPU")
        if x.requires_grad:
            raise RuntimeError("ModelNew does not support autograd-tracked inputs")

        N, C = x.shape
        G = self.group_norm.num_groups
        if C % G != 0:
            raise ValueError("out_features must be divisible by num_groups")
        group_size = C // G

        x_contig = x.contiguous()
        weight = self.group_norm.weight.contiguous()
        bias = self.group_norm.bias.contiguous()

        out = torch.empty((N, 1), device=x.device, dtype=x.dtype)

        grid = (triton.cdiv(N, 8),)
        _fused_gelu_groupnorm_mean_relu[grid](
            x_contig,
            weight,
            bias,
            out,
            N,
            C,
            group_size,
            EPS=self.group_norm.eps,
            NUM_GROUPS=G,
            BLOCK_SIZE=group_size,
            BLOCK_M=8,
            num_warps=2,
            num_stages=3,
        )
        return out


batch_size = 128
in_features_default = 512
out_features_default = 1024
num_groups_default = 8

def get_inputs():
    return [torch.randn(batch_size, in_features_default, device="npu")]

def get_init_inputs():
    return [in_features_default, out_features_default, num_groups_default]
