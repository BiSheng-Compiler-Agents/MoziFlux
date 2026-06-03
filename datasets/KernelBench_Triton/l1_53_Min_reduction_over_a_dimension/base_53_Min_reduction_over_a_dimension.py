import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def _min_reduce_last_kernel(
    x_ptr, out_ptr,
    B, M, N,
    stride_b, stride_m, stride_n,
    out_stride_b, out_stride_m,
    BLOCK_K: tl.constexpr,
):
    m = tl.program_id(axis=0)
    b = tl.program_id(axis=1)

    in_bounds = (b < B) & (m < M)
    if ~in_bounds:
        return

    base = b * stride_b + m * stride_m

    offs_k = tl.arange(0, BLOCK_K)
    tl.max_contiguous(offs_k, BLOCK_K)

    k = 0
    idx0 = k + offs_k
    mask0 = idx0 < N
    ptrs0 = x_ptr + base + idx0 * stride_n
    x0 = tl.load(ptrs0, mask=mask0, other=float("inf"), cache_modifier=".ca")
    acc = tl.min(x0, axis=0)
    k += BLOCK_K

    while k < N:
        for u in tl.static_range(0, 2):
            idx = k + offs_k + u * BLOCK_K
            mask = idx < N
            ptrs = x_ptr + base + idx * stride_n
            x = tl.load(ptrs, mask=mask, other=float("inf"), cache_modifier=".ca")
            acc = tl.minimum(acc, tl.min(x, axis=0))
        k += 2 * BLOCK_K

    out_off = b * out_stride_b + m * out_stride_m
    tl.store(out_ptr + out_off, acc)


@triton.jit
def _min_reduce_mid_kernel(
    x_ptr, out_ptr,
    B, M, N,
    stride_b, stride_m, stride_n,
    out_stride_b, out_stride_n,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    tiles_n = tl.cdiv(N, BLOCK_N)
    b = pid // tiles_n
    tile_n = pid % tiles_n
    if b >= B:
        return

    offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    tl.max_contiguous(offs_n, BLOCK_N)

    base = b * stride_b + offs_n[None, :] * stride_n

    offs_k = tl.arange(0, BLOCK_K)
    tl.max_contiguous(offs_k, BLOCK_K)

    k = 0
    idx0 = k + offs_k[:, None]
    mask0 = idx0 < M
    ptrs0 = x_ptr + base + idx0 * stride_m
    x0 = tl.load(ptrs0, mask=mask0, other=float("inf"), cache_modifier=".cg")
    acc = tl.min(x0, axis=0)
    k += BLOCK_K

    while k < M:
        for u in tl.static_range(0, 2):
            idx = k + offs_k[:, None] + u * BLOCK_K
            mask = idx < M
            ptrs = x_ptr + base + idx * stride_m
            x = tl.load(ptrs, mask=mask, other=float("inf"), cache_modifier=".cg")
            acc = tl.minimum(acc, tl.min(x, axis=0), propagate_nan=tl.PropagateNan.ALL)
        k += 2 * BLOCK_K

    out_off = b * out_stride_b + offs_n * out_stride_n
    tl.store(out_ptr + out_off, acc, mask=n_mask)


@triton.jit
def _min_reduce_first_kernel(
    x_ptr, out_ptr,
    B, M, N,
    stride_b, stride_m, stride_n,
    out_stride_m, out_stride_n,
    BLOCK_K: tl.constexpr,
):
    n = tl.program_id(axis=0)
    m = tl.program_id(axis=1)

    in_bounds = (m < M) & (n < N)
    if ~in_bounds:
        return

    base = m * stride_m + n * stride_n

    offs_k = tl.arange(0, BLOCK_K)
    tl.max_contiguous(offs_k, BLOCK_K)

    k = 0
    idx0 = k + offs_k
    mask0 = idx0 < B
    ptrs0 = x_ptr + base + idx0 * stride_b
    x0 = tl.load(ptrs0, mask=mask0, other=float("inf"), cache_modifier=".ca")
    acc = tl.min(x0, axis=0)
    k += BLOCK_K

    while k < B:
        for u in tl.static_range(0, 2):
            idx = k + offs_k + u * BLOCK_K
            mask = idx < B
            ptrs = x_ptr + base + idx * stride_b
            x = tl.load(ptrs, mask=mask, other=float("inf"), cache_modifier=".ca")
            acc = tl.minimum(acc, tl.min(x, axis=0))
        k += 2 * BLOCK_K

    out_off = m * out_stride_m + n * out_stride_n
    tl.store(out_ptr + out_off, acc)


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def _choose_block_and_warps(self, K: int):
        if K >= 256:
            return 256, 8
        elif K >= 128:
            return 128, 4
        elif K >= 64:
            return 64, 2
        else:
            return 32, 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"ModelNew expects a 3D tensor, got shape {tuple(x.shape)}")
        if not hasattr(torch, "npu") or x.device.type != "npu":
            raise ValueError("ModelNew requires an Ascend NPU tensor input")

        dim = self.dim
        if dim < 0:
            dim += x.ndim
        if dim not in (0, 1, 2):
            raise ValueError(f"Unsupported reduction dim {self.dim} for 3D input")

        B, M, N = x.shape
        sb, sm, sn = x.stride()

        if B == 0 or M == 0 or N == 0:
            raise ValueError("Zero-sized reductions are not supported")

        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError(f"Unsupported dtype for Triton reduction: {x.dtype}")

        if dim == 2 and sn == 1:
            out = torch.empty((B, M), device=x.device, dtype=x.dtype)
            ob, om = out.stride()
            grid = (M, B)
            BK, NW = self._choose_block_and_warps(N)
            _min_reduce_last_kernel[grid](
                x, out,
                B, M, N,
                sb, sm, sn,
                ob, om,
                BLOCK_K=BK,
                num_warps=NW, num_stages=4,
            )
            return out
        elif dim == 1:
            out = torch.empty((B, N), device=x.device, dtype=x.dtype)
            ob, on = out.stride()
            BN = 128
            grid = (B * triton.cdiv(N, BN),)
            BK = 64
            NW = 8
            _min_reduce_mid_kernel[grid](
                x, out,
                B, M, N,
                sb, sm, sn,
                ob, on,
                BLOCK_K=BK, BLOCK_N=BN,
                num_warps=NW, num_stages=4,
            )
            return out
        elif dim == 0:
            out = torch.empty((M, N), device=x.device, dtype=x.dtype)
            om, on = out.stride()
            grid = (N, M)
            BK, NW = self._choose_block_and_warps(B)
            _min_reduce_first_kernel[grid](
                x, out,
                B, M, N,
                sb, sm, sn,
                om, on,
                BLOCK_K=BK,
                num_warps=NW, num_stages=4,
            )
            return out
        else:
            raise ValueError(
                f"Reduction dim {dim} requires a contiguous reduction axis, got strides {tuple(x.stride())}"
            )


def min_reduce_triton(x: torch.Tensor, dim: int) -> torch.Tensor:
    return ModelNew(dim)(x)
batch_size = 128
dim1 = 4096
dim2 = 4095

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2)
    return [x]
def get_init_inputs():
    return [1]
