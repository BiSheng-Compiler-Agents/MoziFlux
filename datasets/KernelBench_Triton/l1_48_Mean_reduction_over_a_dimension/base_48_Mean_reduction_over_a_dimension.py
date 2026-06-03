import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


# Reduce over last dimension (dim=2): x[b, m, :] -> y[b, m]
@triton.jit
def _mean_reduce_last_kernel(
    x_ptr, y_ptr,
    B, M, N,
    stride_b, stride_m, stride_n,
    y_stride_b, y_stride_m,
    invN,  # float32
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    b = pid // M
    m = pid % M
    if (b >= B) | (m >= M):
        return

    base_ptr = x_ptr + b * stride_b + m * stride_m
    offs = tl.arange(0, BLOCK_N)

    # Accumulate across N in a vector register; reduce once at the end
    acc_vec = tl.zeros([BLOCK_N], dtype=tl.float32)
    k = 0
    UNROLL: tl.constexpr = 4
    while k < N:
        # Software unrolling for better ILP
        for u in tl.static_range(UNROLL):
            idx = k + u * BLOCK_N + offs
            mask = idx < N
            vals = tl.load(base_ptr + idx * stride_n, mask=mask, other=0.0)
            acc_vec += vals.to(tl.float32)
        k += UNROLL * BLOCK_N

    total = tl.sum(acc_vec, axis=0)
    mean = total * invN
    out_ptr = y_ptr + b * y_stride_b + m * y_stride_m
    tl.store(out_ptr, mean)


# Reduce over middle dimension (dim=1): x[b, :, n] -> y[b, n]
# Tile along contiguous N to keep loads coalesced.
@triton.jit
def _mean_reduce_mid_multirow_kernel(
    x_ptr, y_ptr,
    B, M, N,
    stride_b, stride_m, stride_n,
    y_stride_b, y_stride_n,
    invM,  # float32
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    UNROLL_M: tl.constexpr,
):
    b_block = tl.program_id(axis=0)
    n_block = tl.program_id(axis=1)

    offs_b = b_block * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)
    b_mask = offs_b < B
    n_mask = offs_n < N

    acc = tl.zeros([BLOCK_B, BLOCK_N], dtype=tl.float32)
    base_ptr = x_ptr + offs_b[:, None] * stride_b + offs_n[None, :] * stride_n
    m = 0
    # Unroll over M to reduce loop overhead; guard with mask for tail
    while m < M:
        for u in tl.static_range(UNROLL_M):
            mi = m + u
            mi_valid = mi < M
            ptr = base_ptr + mi * stride_m
            vals = tl.load(
                ptr,
                mask=b_mask[:, None] & n_mask[None, :] & mi_valid,
                other=0.0,
            ).to(tl.float32)
            acc += vals
        m += UNROLL_M

    mean = acc * invM
    out_ptr = y_ptr + offs_b[:, None] * y_stride_b + offs_n[None, :] * y_stride_n
    tl.store(out_ptr, mean, mask=b_mask[:, None] & n_mask[None, :])


# Reduce over first dimension (dim=0): x[:, m, n] -> y[m, n]
# Tile along contiguous N to keep loads coalesced.
@triton.jit
def _mean_reduce_first_tiled_kernel(
    x_ptr, y_ptr,
    B, M, N,
    stride_b, stride_m, stride_n,
    y_stride_m, y_stride_n,
    invB,  # float32
    BLOCK_N: tl.constexpr,
):
    m = tl.program_id(axis=0)
    n_block = tl.program_id(axis=1)

    if m >= M:
        return

    n_start = n_block * BLOCK_N
    offs_n = n_start + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    b = 0
    UNROLL: tl.constexpr = 4
    # Unroll over B to improve ILP; guard tail with mask
    while b < B:
        for u in tl.static_range(UNROLL):
            bi = b + u
            bi_valid = bi < B
            ptr = x_ptr + bi * stride_b + m * stride_m + offs_n * stride_n
            vals = tl.load(ptr, mask=n_mask & bi_valid, other=0.0).to(tl.float32)
            acc += vals
        b += UNROLL

    mean = acc * invB
    out_ptr = y_ptr + m * y_stride_m + offs_n * y_stride_n
    tl.store(out_ptr, mean, mask=n_mask)


class ModelNew(nn.Module):
    """
    Simple model that performs mean reduction over a specific dimension.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reduces the input tensor along the specified dimension by taking the mean.

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape.

        Returns:
            torch.Tensor: Output tensor with reduced dimension. The shape of the output is the same as the input except for the reduced dimension which is removed.
        """
        return mean_reduction_over_a_dimension(x, self.dim)

def mean_reduction_over_a_dimension(x: torch.Tensor, dim: int) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")

    if not x.is_npu:
        raise ValueError("mean_reduction_over_a_dimension expects an Ascend NPU tensor")

    if x.dim() != 3:
        raise ValueError("mean_reduction_over_a_dimension expects a 3D tensor")

    if x.dtype != torch.float32:
        raise TypeError("mean_reduction_over_a_dimension only supports torch.float32 inputs")

    dim = int(dim)
    if dim < 0:
        dim += x.dim()
    if dim not in (0, 1, 2):
        raise ValueError(f"invalid reduction dim {dim} for input rank {x.dim()}")

    B, M, N = x.shape
    if (dim == 0 and B == 0) or (dim == 1 and M == 0) or (dim == 2 and N == 0):
        raise ValueError("mean_reduction_over_a_dimension does not support empty reduction axes")

    x = x.contiguous()

    device = x.device
    dtype = x.dtype

    # Choose tile along the contiguous N dimension for coalesced loads.
    if dim == 2:
        y = torch.empty((B, M), device=device, dtype=dtype)
        BLOCK_N = 512 if N >= 512 else (256 if N >= 256 else (128 if N >= 128 else 64))
        grid = (B * M,)
        _mean_reduce_last_kernel[grid](
            x, y,
            B, M, N,
            x.stride(0), x.stride(1), x.stride(2),
            y.stride(0), y.stride(1),
            1.0 / float(N),
            BLOCK_N=BLOCK_N,
            num_warps=8,
            num_stages=4,
        )
        return y

    if dim == 1:
        y = torch.empty((B, N), device=device, dtype=dtype)
        BLOCK_B = 8
        BLOCK_N = 256
        UNROLL_M = 4
        grid = (triton.cdiv(B, BLOCK_B), triton.cdiv(N, BLOCK_N))
        _mean_reduce_mid_multirow_kernel[grid](
            x, y,
            B, M, N,
            x.stride(0), x.stride(1), x.stride(2),
            y.stride(0), y.stride(1),
            1.0 / float(M),
            BLOCK_B=BLOCK_B,
            BLOCK_N=BLOCK_N,
            UNROLL_M=UNROLL_M,
            num_warps=8 if BLOCK_B * BLOCK_N >= 512 else 4,
            num_stages=4,
        )
        return y

    y = torch.empty((M, N), device=device, dtype=dtype)
    BLOCK_N = 256 if N >= 256 else 128 if N >= 128 else 64
    grid = (M, triton.cdiv(N, BLOCK_N))
    _mean_reduce_first_tiled_kernel[grid](
        x, y,
        B, M, N,
        x.stride(0), x.stride(1), x.stride(2),
        y.stride(0), y.stride(1),
        1.0 / float(B),
        BLOCK_N=BLOCK_N,
        num_warps=8 if BLOCK_N >= 256 else 4,
        num_stages=4,
    )
    return y
batch_size = 128
dim1 = 4096
dim2 = 4095

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2)
    return [x]
def get_init_inputs():
    return [1]
