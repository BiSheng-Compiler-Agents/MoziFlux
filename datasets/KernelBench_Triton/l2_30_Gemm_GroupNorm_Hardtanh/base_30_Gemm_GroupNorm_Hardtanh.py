import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _groupnorm_hardtanh_kernel(
    x_ptr,
    gamma_ptr,
    beta_ptr,
    out_ptr,
    N,
    C,
    G,
    Cg,
    eps,
    minv,
    maxv,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs = tl.arange(0, BLOCK_SIZE)
    tl.max_contiguous(offs, BLOCK_SIZE)
    tl.multiple_of(offs, 16)

    offs_c = pid_g * Cg + offs
    row_mask = rows < N
    ch_mask = offs < Cg
    mask = row_mask[:, None] & ch_mask[None, :]

    x_ptrs = x_ptr + rows[:, None] * C + offs_c[None, :]
    out_ptrs = out_ptr + rows[:, None] * C + offs_c[None, :]

    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + offs_c, mask=ch_mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + offs_c, mask=ch_mask, other=0.0).to(tl.float32)

    inv_cg = 1.0 / tl.full((), Cg, tl.float32)
    sum_x = tl.sum(x, axis=1)
    sum_x2 = tl.sum(x * x, axis=1)
    mean = sum_x * inv_cg
    var = sum_x2 * inv_cg - mean * mean

    inv_std = tl.rsqrt(var + eps)
    scale = inv_std[:, None] * gamma[None, :]
    shift = beta[None, :] - mean[:, None] * scale
    y = x * scale + shift
    y = tl.maximum(tl.minimum(y, maxv), minv)

    tl.store(out_ptrs, y, mask=mask)


class ModelNew(nn.Module):

    def __init__(
        self,
        in_features=8192,
        out_features=8192,
        num_groups=16,
        hardtanh_min=-2.0,
        hardtanh_max=2.0,
    ):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.hardtanh = nn.Hardtanh(min_val=hardtanh_min, max_val=hardtanh_max)

    def forward(self, x):
        y = self.gemm(x)
        if y.device.type != "npu":
            raise RuntimeError(
                "ModelNew expects NPU tensors and does not provide a non-NPU fallback."
            )

        y = y.contiguous()
        N, C = y.shape
        G = self.group_norm.num_groups
        if C % G != 0:
            raise RuntimeError("out_features must be divisible by num_groups")
        Cg = C // G

        gamma = self.group_norm.weight.contiguous()
        beta = self.group_norm.bias.contiguous()
        eps = float(self.group_norm.eps)
        minv = float(self.hardtanh.min_val)
        maxv = float(self.hardtanh.max_val)

        out = torch.empty_like(y)

        def next_power_of_two(v: int) -> int:
            return 1 if v <= 1 else 1 << ((v - 1).bit_length())

        block_size = next_power_of_two(Cg)
        grid = (triton.cdiv(N, 12), G)
        _groupnorm_hardtanh_kernel[grid](
            y,
            gamma,
            beta,
            out,
            N,
            C,
            G,
            Cg,
            eps,
            minv,
            maxv,
            BLOCK_SIZE=block_size,
            BLOCK_M=12,
            num_warps=8,
            num_stages=1,
        )
        return out


batch_size = 1024
in_features = 8192
out_features = 8192
num_groups = 16
hardtanh_min = -2.0
hardtanh_max = 2.0


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, num_groups, hardtanh_min, hardtanh_max]
