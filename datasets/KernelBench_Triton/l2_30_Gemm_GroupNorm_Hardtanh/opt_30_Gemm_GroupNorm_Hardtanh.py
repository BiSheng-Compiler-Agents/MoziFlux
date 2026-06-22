import torch
import torch.nn as nn
import triton
import triton.language as tl

_MAX_GRID = 65535
_MAX_GROUP_ELEMS = 2048
_MAX_GROUP_BLOCK = 4


@triton.jit
def _groupnorm_hardtanh_groupblock_kernel(
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
    GROUPS_PER_ROW,
    BLOCK_SIZE: tl.constexpr,
    GROUP_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // GROUPS_PER_ROW
    group_tile = pid - n * GROUPS_PER_ROW

    gidx = group_tile * GROUP_BLOCK + tl.arange(0, GROUP_BLOCK)
    offs = tl.arange(0, BLOCK_SIZE)
    tl.max_contiguous(offs, BLOCK_SIZE)
    tl.multiple_of(offs, 16)

    offs_c = gidx[:, None] * Cg + offs[None, :]
    mask = (n < N) & (gidx[:, None] < G) & (offs[None, :] < Cg)
    ch_mask = (gidx[:, None] < G) & (offs[None, :] < Cg)
    base = n * C + offs_c

    x = tl.load(x_ptr + base, mask=mask, other=0.0,
                care_padding=False).to(tl.float32)
    gamma = tl.load(gamma_ptr + offs_c,
                    mask=ch_mask,
                    other=0.0,
                    care_padding=False).to(tl.float32)
    beta = tl.load(beta_ptr + offs_c,
                   mask=ch_mask,
                   other=0.0,
                   care_padding=False).to(tl.float32)

    inv_cg = 1.0 / tl.full((), Cg, tl.float32)
    sum_x = tl.sum(x, axis=1)
    mean = sum_x * inv_cg
    x_centered = x - mean[:, None]
    var = tl.sum(x_centered * x_centered, axis=1) * inv_cg
    inv_std = tl.rsqrt(var + eps)
    y = x_centered * inv_std[:, None]
    y = y * gamma + beta
    y = tl.maximum(tl.minimum(y, maxv), minv)

    tl.store(out_ptr + base, y, mask=mask)


@triton.jit
def _groupnorm_hardtanh_groupblock_persistent_kernel(
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
    GROUPS_PER_ROW,
    TOTAL_TILES,
    N_PROGRAMS,
    BLOCK_SIZE: tl.constexpr,
    GROUP_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    for tile_id in range(pid, TOTAL_TILES, N_PROGRAMS):
        n = tile_id // GROUPS_PER_ROW
        group_tile = tile_id - n * GROUPS_PER_ROW

        gidx = group_tile * GROUP_BLOCK + tl.arange(0, GROUP_BLOCK)
        offs = tl.arange(0, BLOCK_SIZE)
        tl.max_contiguous(offs, BLOCK_SIZE)
        tl.multiple_of(offs, 16)

        offs_c = gidx[:, None] * Cg + offs[None, :]
        mask = (n < N) & (gidx[:, None] < G) & (offs[None, :] < Cg)
        ch_mask = (gidx[:, None] < G) & (offs[None, :] < Cg)
        base = n * C + offs_c

        x = tl.load(x_ptr + base, mask=mask, other=0.0,
                    care_padding=False).to(tl.float32)
        gamma = tl.load(gamma_ptr + offs_c,
                        mask=ch_mask,
                        other=0.0,
                        care_padding=False).to(tl.float32)
        beta = tl.load(beta_ptr + offs_c,
                       mask=ch_mask,
                       other=0.0,
                       care_padding=False).to(tl.float32)

        inv_cg = 1.0 / tl.full((), Cg, tl.float32)
        sum_x = tl.sum(x, axis=1)
        mean = sum_x * inv_cg
        x_centered = x - mean[:, None]
        var = tl.sum(x_centered * x_centered, axis=1) * inv_cg
        inv_std = tl.rsqrt(var + eps)
        y = x_centered * inv_std[:, None]
        y = y * gamma + beta
        y = tl.maximum(tl.minimum(y, maxv), minv)

        tl.store(out_ptr + base, y, mask=mask)


class ModelNew(nn.Module):
    """GEMM followed by fused GroupNorm + HardTanh for NPU execution."""

    def __init__(
        self,
        in_features=1024,
        out_features=512,
        num_groups=8,
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
        assert C % G == 0, "out_features must be divisible by num_groups"
        Cg = C // G

        gamma = self.group_norm.weight.contiguous()
        beta = self.group_norm.bias.contiguous()
        eps = float(self.group_norm.eps)
        minv = float(self.hardtanh.min_val)
        maxv = float(self.hardtanh.max_val)
        out = torch.empty_like(y)

        def next_power_of_two(v: int) -> int:
            return 1 if v <= 1 else 1 << ((v - 1).bit_length())

        BLOCK_SIZE = next_power_of_two(Cg)
        GROUP_BLOCK = max(
            1, min(_MAX_GROUP_BLOCK, G, _MAX_GROUP_ELEMS // BLOCK_SIZE))
        groups_per_row = triton.cdiv(G, GROUP_BLOCK)
        total_tiles = N * groups_per_row

        if total_tiles > _MAX_GRID:
            n_programs = _MAX_GRID
            _groupnorm_hardtanh_groupblock_persistent_kernel[(n_programs, )](
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
                groups_per_row,
                total_tiles,
                n_programs,
                BLOCK_SIZE=BLOCK_SIZE,
                GROUP_BLOCK=GROUP_BLOCK,
                num_warps=4,
                num_stages=2,
            )
        else:
            _groupnorm_hardtanh_groupblock_kernel[(total_tiles, )](
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
                groups_per_row,
                BLOCK_SIZE=BLOCK_SIZE,
                GROUP_BLOCK=GROUP_BLOCK,
                num_warps=4,
                num_stages=2,
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
