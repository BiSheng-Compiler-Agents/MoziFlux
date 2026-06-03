import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def fused_bias_act_gn_kernel(
    x_ptr,
    extra_bias_ptr,
    gamma_ptr,
    beta_ptr,
    out_ptr,
    N, C, G,
    GROUP_SIZE,
    eps,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_g = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_SIZE)
    c = pid_g * GROUP_SIZE + offs_n

    mask_m = offs_m < N
    mask_c = offs_n < BLOCK_SIZE

    x_base = x_ptr + offs_m[:, None] * C + c[None, :]
    x = tl.load(x_base, mask=(mask_m[:, None] & mask_c[None, :]), other=0.0)

    b = tl.load(extra_bias_ptr + c, mask=mask_c, other=0.0)
    v = x + b[None, :]
    v = tl.minimum(tl.maximum(v, -1.0), 1.0)

    vf = v.to(tl.float32)
    pos = vf > 0.0
    softplus = tl.where(pos, vf, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(vf)))
    mish = vf * tl.tanh(softplus)

    sum1 = tl.sum(mish, axis=1)
    sum2 = tl.sum(mish * mish, axis=1)
    gs = tl.full((), GROUP_SIZE, dtype=tl.float32)
    mean = sum1 / gs
    var = sum2 / gs - mean * mean
    inv_std = tl.rsqrt(var + eps)

    norm = (mish - mean[:, None]) * inv_std[:, None]

    gamma = tl.load(gamma_ptr + c, mask=mask_c, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + c, mask=mask_c, other=0.0).to(tl.float32)
    y = norm * gamma[None, :] + beta[None, :]

    out_base = out_ptr + offs_m[:, None] * C + c[None, :]
    tl.store(out_base, y.to(v.dtype), mask=(mask_m[:, None] & mask_c[None, :]))


class ModelNew(nn.Module):
    def __init__(
        self,
        in_features=512,
        out_features=1024,
        bias_shape=None,
        num_groups=32,
    ):
        super(ModelNew, self).__init__()
        if bias_shape is None:
            bias_shape = (out_features,)
        if out_features % num_groups != 0:
            raise ValueError("out_features must be divisible by num_groups")
        self.gemm = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.hardtanh = nn.Hardtanh()
        self.mish = nn.Mish()
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=out_features)

    def _fused_post_gemm(self, y: torch.Tensor) -> torch.Tensor | None:
        N, C = y.shape
        G = self.groupnorm.num_groups
        if (C % G) != 0:
            raise RuntimeError("Channels must be divisible by num_groups for GroupNorm.")
        GROUP_SIZE = C // G

        y_in = y.contiguous()
        extra_bias = self.bias.contiguous()
        gamma = self.groupnorm.weight.contiguous()
        beta = self.groupnorm.bias.contiguous()
        eps = float(self.groupnorm.eps)

        out = torch.empty_like(y_in)

        BLOCK_SIZE = GROUP_SIZE
        BLOCK_M = 8

        grid = (triton.cdiv(N, BLOCK_M), G)

        fused_bias_act_gn_kernel[grid](
            y_in, extra_bias, gamma, beta, out,
            N, C, G, GROUP_SIZE, eps,
            BLOCK_SIZE=BLOCK_SIZE,
            BLOCK_M=BLOCK_M,
            num_stages=2,
        )
        return out

    def forward(self, x):
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
