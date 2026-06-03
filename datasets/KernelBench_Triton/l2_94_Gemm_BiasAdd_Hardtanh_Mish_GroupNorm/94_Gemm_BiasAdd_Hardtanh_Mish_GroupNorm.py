import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def fused_bias_act_gn_kernel(
    x_ptr,  # [N, C]
    extra_bias_ptr,  # [C]
    gamma_ptr,  # [C] groupnorm weight
    beta_ptr,  # [C] groupnorm bias
    out_ptr,  # [N, C]
    N,
    C,
    G,  # ints
    GROUP_SIZE,  # int = C // G
    eps,  # float
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    n = pid // G
    g = pid % G

    offs = tl.arange(0, BLOCK_SIZE)
    base = n * C
    g_off = g * GROUP_SIZE
    c = g_off + offs
    mask_c = offs < GROUP_SIZE
    mask = mask_c & (n < N)
    idx = base + c

    x = tl.load(x_ptr + idx, mask=mask, other=0.0)
    b = tl.load(extra_bias_ptr + c, mask=mask_c, other=0.0)
    v = x + b
    v = tl.minimum(tl.maximum(v, -1.0), 1.0)

    vf = v.to(tl.float32)
    abs_v = tl.abs(vf)
    zero = tl.zeros_like(vf)
    one = zero + 1.0
    twenty = zero + 20.0
    neg_twenty = zero - 20.0
    softplus_mid = tl.where(vf > zero, vf, zero) + tl.log(one + tl.exp(-abs_v))
    softplus = tl.where(vf > twenty, vf,
                        tl.where(vf < neg_twenty, tl.exp(vf), softplus_mid))
    mish = vf * tl.tanh(softplus)

    mish_masked = tl.where(mask, mish, 0.0)
    sum1 = tl.sum(mish_masked, axis=0)
    sum2 = tl.sum(mish_masked * mish_masked, axis=0)
    gs = tl.full((), GROUP_SIZE, dtype=tl.float32)
    mean = sum1 / gs
    var = sum2 / gs - mean * mean
    inv_std = tl.rsqrt(var + eps)

    y = (mish - mean) * inv_std

    gamma = tl.load(gamma_ptr + c, mask=mask_c, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + c, mask=mask_c, other=0.0).to(tl.float32)
    y = y * gamma + beta

    tl.store(out_ptr + idx, y.to(x.dtype), mask=mask)


def _next_power_of_2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


batch_size = 128
in_features = 512
out_features = 1024
bias_shape = (out_features,)
num_groups = 32


class ModelNew(nn.Module):
    """
    A model that performs a GEMM, BiasAdd, Hardtanh, Mish, and GroupNorm operations in sequence.
    """
    def __init__(
        self,
        in_features=in_features,
        out_features=out_features,
        bias_shape=bias_shape,
        num_groups=num_groups,
    ):
        super(ModelNew, self).__init__()
        if out_features % num_groups != 0:
            raise ValueError("out_features must be divisible by num_groups")
        self.gemm = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.hardtanh = nn.Hardtanh()
        self.mish = nn.Mish()
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=out_features)

    def _fused_post_gemm(self, y: torch.Tensor) -> torch.Tensor | None:
        # Fused BiasAdd -> Hardtanh -> Mish -> GroupNorm using Triton
        N, C = y.shape
        G = self.groupnorm.num_groups
        if (C % G) != 0:
            raise RuntimeError("Channels must be divisible by num_groups for GroupNorm.")
        GROUP_SIZE = C // G
        if GROUP_SIZE > 256:
            raise RuntimeError("GROUP_SIZE > 256 is not supported by the fused Triton path.")

        # Ensure contiguity
        y_in = y.contiguous()
        extra_bias = self.bias.contiguous()
        gamma = self.groupnorm.weight.contiguous()
        beta = self.groupnorm.bias.contiguous()
        eps = float(self.groupnorm.eps)

        out = torch.empty_like(y_in)

        # Choose an efficient block size (power-of-two, capped)
        BLOCK_SIZE = _next_power_of_2(GROUP_SIZE)
        BLOCK_SIZE = min(max(BLOCK_SIZE, 32), 256)
        # Prefer fewer warps for small groups to reduce overhead
        if BLOCK_SIZE <= 32:
            num_warps = 1
        elif BLOCK_SIZE <= 64:
            num_warps = 2
        else:
            num_warps = 4

        grid = (N * G,)

        fused_bias_act_gn_kernel[grid](
            y_in, extra_bias, gamma, beta, out,
            N, C, G, GROUP_SIZE, eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            num_stages=2,
        )
        return out

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        if x.device.type != "npu":
            raise RuntimeError("ModelNew expects Ascend NPU inputs.")
        if not self.groupnorm.affine:
            raise RuntimeError("ModelNew requires affine GroupNorm parameters.")

        y = self.gemm(x)
        return self._fused_post_gemm(y)
batch_size = 1024
in_features = 8192
out_features = 8192
bias_shape = (out_features,)
num_groups = 256

def get_inputs():
    return [torch.rand(batch_size, in_features)]
def get_init_inputs():
    return [in_features, out_features, bias_shape, num_groups]