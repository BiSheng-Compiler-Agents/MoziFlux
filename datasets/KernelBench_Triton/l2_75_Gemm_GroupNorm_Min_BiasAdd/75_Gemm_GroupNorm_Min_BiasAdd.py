import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=4),
        triton.Config({}, num_warps=8, num_stages=2),
    ],
    key=["N"],
)
@triton.jit
def _fused_groupnorm_min_bias_kernel(
    x_ptr,            # [N, C]
    gamma_ptr,        # [C]
    beta_ptr,         # [C]
    bias_ptr,         # [C]
    out_ptr,          # [1, C, N, 1] - we index via strides over C and N
    N, C,             # sizes
    STRIDE_XN,        # stride between rows in x
    STRIDE_OC,        # stride over C in output
    STRIDE_ON,        # stride over N in output
    EPS,              # epsilon
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if pid >= N:
        return

    row_base = pid * STRIDE_XN

    # 2D channel indexing: [NUM_GROUPS, GROUP_SIZE] covering all channels
    offs_g = tl.arange(0, GROUP_SIZE)[None, :]     # [1, GS]
    offs_grp = tl.arange(0, NUM_GROUPS)[:, None]   # [NG, 1]
    ch_offs_2d = offs_grp * GROUP_SIZE + offs_g    # [NG, GS]

    # Load x and affine parameters (promote to f32 for numerics)
    x = tl.load(x_ptr + row_base + ch_offs_2d).to(tl.float32)     # [NG, GS]
    gamma = tl.load(gamma_ptr + ch_offs_2d).to(tl.float32)        # [NG, GS]
    beta = tl.load(beta_ptr + ch_offs_2d).to(tl.float32)          # [NG, GS]

    # Group-wise mean/var (unbiased=False)
    inv_gs = 1.0 / GROUP_SIZE
    sum1 = tl.sum(x, axis=1)                                      # [NG]
    sum2 = tl.sum(x * x, axis=1)                                  # [NG]
    mean = sum1 * inv_gs                                          # [NG]
    var = sum2 * inv_gs - mean * mean                             # [NG]
    inv_std = tl.rsqrt(var + EPS)                                 # [NG]

    # Normalize + affine
    y = (x - mean[:, None]) * inv_std[:, None]                    # [NG, GS]
    y = y * gamma + beta                                          # [NG, GS]

    # Row-wise min over all channels
    gmin = tl.min(y, axis=1)                                      # [NG]
    row_min = tl.min(gmin, axis=0)                                # scalar

    # Add bias and write to output: O[0, c, n, 0] = bias[c] + row_min
    bias = tl.load(bias_ptr + ch_offs_2d).to(tl.float32)          # [NG, GS]
    out_tile = bias + row_min                                     # [NG, GS]
    out_ptrs = out_ptr + ch_offs_2d * STRIDE_OC + pid * STRIDE_ON
    tl.store(out_ptrs, out_tile)


def _groupnorm_min_bias_triton(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, bias: torch.Tensor, num_groups: int, eps: float):
    """
    Fused Triton path:
    - computes GroupNorm(x, num_groups, gamma, beta)
    - computes min over channels (dim=1, keepdim=True)
    - adds channel bias with broadcast to shape [1, C, N, 1]
    Returns: tensor of shape [1, C, N, 1]
    """
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D input tensor [N, C], got shape {tuple(x.shape)}")

    if x.device.type != "npu":
        raise RuntimeError("The Triton fused GroupNorm-Min-Bias operator requires Ascend NPU tensors.")

    N, C = x.shape
    if C % num_groups != 0:
        raise ValueError(f"Channel count {C} must be divisible by num_groups={num_groups}")

    if gamma.numel() != C or beta.numel() != C:
        raise ValueError("GroupNorm affine parameters must have shape [C].")

    bias = bias.reshape(-1)
    if bias.numel() != C:
        raise ValueError("Bias must contain exactly C elements.")

    tensors = {
        "gamma": gamma,
        "beta": beta,
        "bias": bias,
    }
    for name, tensor in tensors.items():
        if tensor.device != x.device:
            raise RuntimeError(f"{name} must be on the same device as x.")

    group_size = C // num_groups
    # Ensure contiguity for pointer arithmetic
    x_ctg = x.contiguous()
    gamma_ctg = gamma.contiguous()
    beta_ctg = beta.contiguous()
    bvec = bias.reshape(-1).contiguous()

    # Output [1, C, N, 1]
    out = torch.empty((1, C, N, 1), device=x.device, dtype=x.dtype)

    grid = (N,)
    _fused_groupnorm_min_bias_kernel[grid](
        x_ctg, gamma_ctg, beta_ctg, bvec, out,
        N, C,
        x_ctg.stride(0),
        out.stride(1), out.stride(2),
        eps,
        GROUP_SIZE=group_size, NUM_GROUPS=num_groups,
    )
    return out


def gemm_groupnorm_min_bias_add(
    x: torch.Tensor,
    linear_weight: torch.Tensor,
    linear_bias: torch.Tensor,
    group_norm_weight: torch.Tensor,
    group_norm_bias: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
    eps: float = 1e-5,
):
    """
    Public entrypoint:
    1. GEMM via a dense linear projection
    2. Fused Triton GroupNorm + channelwise min + bias add
    """
    gemm_out = F.linear(x, linear_weight, linear_bias)
    return _groupnorm_min_bias_triton(
        gemm_out,
        group_norm_weight,
        group_norm_bias,
        bias,
        num_groups,
        eps,
    )


class ModelNew(nn.Module):
    """
    Model that performs a GEMM, Group Normalization, Minimum operation, and Bias addition.
    """
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        return gemm_groupnorm_min_bias_add(
            x,
            self.gemm.weight,
            self.gemm.bias,
            self.group_norm.weight,
            self.group_norm.bias,
            self.bias,
            self.group_norm.num_groups,
            self.group_norm.eps,
        )
batch_size = 1024
in_features = 8192
out_features = 8192
num_groups = 512
bias_shape = (1, out_features, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_features)]
def get_init_inputs():
    return [in_features, out_features, num_groups, bias_shape]