import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _rowwise_lse_leaky_gelu2(
    x_ptr,           # pointer to [B, N] input
    y_ptr,           # pointer to [B, 1] output
    stride_xm,       # stride between rows for x
    stride_xn,       # stride between cols for x
    stride_ym,       # stride between rows for y
    NEG_SLOPE: tl.constexpr,  # leaky ReLU negative slope
    N: tl.constexpr,          # number of columns (out_features)
    BLOCK_N: tl.constexpr,    # tile size along N
):
    pid = tl.program_id(0)  # row id

    # Precompute arange once and base row pointer for better ILP
    r = tl.arange(0, BLOCK_N)
    row_ptr = x_ptr + pid * stride_xm

    # Streaming LogSumExp in a single pass for numeric stability and fewer global loads
    m = tl.full((1,), -float("inf"), dtype=tl.float32)  # running max
    s = tl.zeros((1,), dtype=tl.float32)                # running sum of exp shifted by m
    for n0 in range(0, N, BLOCK_N):
        offs = n0 + r
        mask = offs < N
        # Load masked lanes as -inf so we can drop further masking/where ops
        vals = tl.load(row_ptr + offs * stride_xn, mask=mask, other=-float("inf")).to(tl.float32)
        # tile max
        tile_max = tl.max(vals, axis=0)
        m_new = tl.maximum(m, tile_max)
        # accumulate sum in the new max domain
        s = s * tl.exp(m - m_new) + tl.sum(tl.exp(vals - m_new), axis=0)
        m = m_new

    # Final LogSumExp
    lse = m + tl.log(s)

    # Two LeakyReLU applications fused into one: for x<0 multiply by slope^2
    slope_sq = NEG_SLOPE * NEG_SLOPE
    x = tl.where(lse >= 0.0, lse, lse * slope_sq)

    # Two GELU (exact, erf-based) applications
    inv_sqrt2 = 0.7071067811865476  # 1/sqrt(2)
    x = 0.5 * x * (1.0 + tl.math.erf(x * inv_sqrt2))
    x = 0.5 * x * (1.0 + tl.math.erf(x * inv_sqrt2))

    # Store result
    y_offs = pid * stride_ym + tl.arange(0, 1)
    tl.store(y_ptr + y_offs, x)


class ModelNew(nn.Module):
    """
    Model that performs a matrix multiplication (Gemm), followed by LogSumExp, LeakyReLU, 
    LeakyReLU, GELU, and GELU activations.
    """
    def __init__(self, in_features=1024, out_features=512, bias=True):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.neg_slope = 0.01

    def forward(self, x):
        if x.device.type != "npu":
            raise RuntimeError("ModelNew expects input tensors on Ascend NPU")

        # Gemm using PyTorch linear on NPU
        x = self.linear(x)

        # Fused row-wise LogSumExp + 2x LeakyReLU + 2x GELU using Triton
        B, N = x.shape
        x_c = x.contiguous()
        y = torch.empty((B, 1), device=x.device, dtype=x.dtype)

        grid = (B,)
        block_n = min(1024, triton.next_power_of_2(N))
        _rowwise_lse_leaky_gelu2[grid](
            x_c,
            y,
            x_c.stride(0),
            x_c.stride(1),
            y.stride(0),
            NEG_SLOPE=self.neg_slope,
            N=N,
            BLOCK_N=block_n,
            num_warps=4,
            num_stages=2,
        )
        return y
batch_size = 1024
in_features = 8192
out_features = 8192

def get_inputs():
    return [torch.rand(batch_size, in_features)]
def get_init_inputs():
    return [in_features, out_features]