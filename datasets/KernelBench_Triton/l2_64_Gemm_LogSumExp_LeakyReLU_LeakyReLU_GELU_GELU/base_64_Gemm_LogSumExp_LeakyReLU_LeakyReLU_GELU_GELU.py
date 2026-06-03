import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _rowwise_lse_leaky_gelu2(
    x_ptr,
    y_ptr,
    bsz,
    stride_xm,
    stride_xn,
    stride_ym,
    NEG_SLOPE: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < bsz
    r = tl.arange(0, BLOCK_N)
    slope_sq = NEG_SLOPE * NEG_SLOPE
    inv_sqrt2 = 0.7071067811865476
    for rm in range(0, BLOCK_M):
        cur_row = pid * BLOCK_M + rm
        cur_row_mask = cur_row < bsz
        row_ptr = x_ptr + cur_row * stride_xm

        m = tl.full((1,), -float("inf"), dtype=tl.float32)
        s = tl.zeros((1,), dtype=tl.float32)
        for n0 in range(0, N, BLOCK_N):
            offs = n0 + r
            mask = cur_row_mask & (offs < N)
            vals = tl.load(row_ptr + offs * stride_xn, mask=mask, other=-float("inf")).to(tl.float32)
            tile_max = tl.max(vals, axis=0)
            m_new = tl.maximum(m, tile_max)
            s = s * tl.exp(m - m_new) + tl.sum(tl.exp(vals - m_new), axis=0)
            m = m_new

        lse = m + tl.log(s)
        x = tl.where(lse >= 0.0, lse, lse * slope_sq)
        x = 0.5 * x * (1.0 + tl.math.erf(x * inv_sqrt2))
        x = 0.5 * x * (1.0 + tl.math.erf(x * inv_sqrt2))
        tl.store(y_ptr + cur_row * stride_ym + tl.arange(0, 1), x, mask=cur_row_mask)


batch_size = 1024
in_features = 8192
out_features = 8192


class ModelNew(nn.Module):
    def __init__(self, in_features=None, out_features=None, bias=True):
        super(ModelNew, self).__init__()
        in_features = in_features if in_features is not None else globals()["in_features"]
        out_features = out_features if out_features is not None else globals()["out_features"]
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.neg_slope = 0.01

    def forward(self, x):
        if x.device.type != "npu":
            raise RuntimeError("ModelNew expects input tensors on Ascend NPU")

        x = self.linear(x)
        bsz, n_cols = x.shape
        x_c = x.contiguous()
        y = torch.empty((bsz, 1), device=x.device, dtype=x.dtype)

        block_m = 4
        grid = (triton.cdiv(bsz, block_m),)
        block_n = min(1024, triton.next_power_of_2(n_cols))
        _rowwise_lse_leaky_gelu2[grid](
            x_c,
            y,
            bsz,
            x_c.stride(0),
            x_c.stride(1),
            y.stride(0),
            NEG_SLOPE=self.neg_slope,
            N=n_cols,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=8,
            num_stages=2,
        )
        return y


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features]
