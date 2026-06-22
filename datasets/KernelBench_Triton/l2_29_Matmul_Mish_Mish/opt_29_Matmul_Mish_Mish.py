import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
from triton.runtime import driver

DEFAULT_BATCH_SIZE = 1024
DEFAULT_IN_FEATURES = 8192
DEFAULT_OUT_FEATURES = 8192
_MAX_CORE_DIM = 65535


@triton.jit
def _mish_twice(x):
    threshold = 20.0
    abs_x = tl.abs(x)
    sp_stable = tl.log(1.0 + tl.exp(-abs_x)) + tl.maximum(x, 0.0)
    sp = tl.where(x > threshold, x, sp_stable)
    t = tl.exp(-2.0 * sp)
    tanh_sp = 1.0 - 2.0 * t / (1.0 + t)
    m1 = x * tanh_sp

    abs_m1 = tl.abs(m1)
    sp2_stable = tl.log(1.0 + tl.exp(-abs_m1)) + tl.maximum(m1, 0.0)
    sp2 = tl.where(m1 > threshold, m1, sp2_stable)
    t2 = tl.exp(-2.0 * sp2)
    tanh_sp2 = 1.0 - 2.0 * t2 / (1.0 + t2)
    return m1 * tanh_sp2


@triton.jit
def linear_mish2_dot_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_wk: tl.constexpr,
    stride_ym: tl.constexpr,
    stride_yn: tl.constexpr,
    TOTAL_TILES: tl.constexpr,
    NUM_BLOCKS_M: tl.constexpr,
    NUM_BLOCKS_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_THRESHOLD: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    n_programs = tl.num_programs(axis=0)

    offs_m_base = tl.arange(0, BLOCK_M)
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    tl.max_contiguous(offs_k, BLOCK_K)

    for tile_id in tl.range(pid, TOTAL_TILES, n_programs):
        if NUM_BLOCKS_M >= BLOCK_THRESHOLD and NUM_BLOCKS_N >= BLOCK_THRESHOLD:
            tile_m = tile_id % NUM_BLOCKS_M
            tile_n = (tile_id // NUM_BLOCKS_M) % NUM_BLOCKS_N
        else:
            tile_m = tile_id // NUM_BLOCKS_N
            tile_n = tile_id - tile_m * NUM_BLOCKS_N

        offs_m = tile_m * BLOCK_M + offs_m_base
        offs_n = tile_n * BLOCK_N + offs_n_base
        m_mask = offs_m < M
        n_mask = offs_n < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in tl.range(0, K, BLOCK_K):
            k_idxs = k0 + offs_k
            k_mask = k_idxs < K
            a = tl.load(
                x_ptr + offs_m[:, None] * stride_xm +
                k_idxs[None, :] * stride_xk,
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
                care_padding=False,
            )
            b = tl.load(
                w_ptr + offs_n[None, :] * stride_wn +
                k_idxs[:, None] * stride_wk,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0,
                care_padding=False,
            )
            acc = tl.dot(a, b, acc)
            al.compile_hint(acc, "dot_pad_only_k")

        bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
        out = _mish_twice(acc + bias[None, :])
        tl.store(
            y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
            out.to(y_ptr.dtype.element_ty),
            mask=m_mask[:, None] & n_mask[None, :],
        )


def _num_aicore() -> int:
    try:
        device = torch.npu.current_device()
        return int(
            driver.active.utils.get_device_properties(device)["num_aicore"])
    except Exception:
        return 24


def _mish2_torch(z: torch.Tensor) -> torch.Tensor:
    z = z * torch.tanh(F.softplus(z, beta=1, threshold=20))
    return z * torch.tanh(F.softplus(z, beta=1, threshold=20))


def _matmul_mish_mish_triton(x: torch.Tensor, weight: torch.Tensor,
                             bias: torch.Tensor) -> torch.Tensor:
    m, k = x.shape
    n = weight.shape[0]
    y = torch.empty((m, n), device=x.device, dtype=x.dtype)
    if m == 0 or n == 0:
        return y

    block_m = 64
    block_n = 64
    block_k = 64
    tiles_m = triton.cdiv(m, block_m)
    tiles_n = triton.cdiv(n, block_n)
    total_tiles = tiles_m * tiles_n
    grid_x = min(total_tiles, _num_aicore(), _MAX_CORE_DIM)
    linear_mish2_dot_kernel[(grid_x, )](
        x,
        weight,
        bias,
        y,
        m,
        n,
        k,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        y.stride(0),
        y.stride(1),
        total_tiles,
        tiles_m,
        tiles_n,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        BLOCK_THRESHOLD=4,
    )
    return y


def matmul_mish_mish(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    force_triton: bool = False,
) -> torch.Tensor:
    if x.device.type != "npu":
        raise RuntimeError("matmul_mish_mish requires Ascend NPU tensors.")
    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError(
            "matmul_mish_mish expects x and weight to be 2D tensors.")
    if x.shape[1] != weight.shape[1]:
        raise ValueError("x.shape[1] must match weight.shape[1].")
    if x.device != weight.device:
        raise ValueError("x and weight must be on the same device.")
    if x.dtype != weight.dtype:
        raise ValueError("x and weight must use the same dtype.")

    if bias is None:
        bias = torch.zeros(weight.shape[0], device=x.device, dtype=x.dtype)
    elif bias.ndim != 1 or bias.shape[0] != weight.shape[0]:
        raise ValueError(
            "bias must be a 1D tensor with shape [weight.shape[0]].")
    elif bias.device != x.device:
        raise ValueError("bias must be on the same device as x.")
    elif bias.dtype != x.dtype:
        raise ValueError("bias must use the same dtype as x.")

    if x.dtype not in (torch.float16, torch.float32, torch.bfloat16):
        raise TypeError(
            "matmul_mish_mish supports float16, float32, and bfloat16 inputs.")

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    if force_triton or os.environ.get("KB29_FORCE_TRITON", "0") == "1":
        return _matmul_mish_mish_triton(x, weight, bias)

    return _mish2_torch(F.linear(x, weight, bias))


class ModelNew(nn.Module):
    """Matrix multiplication followed by Mish and Mish again."""

    def __init__(
        self,
        in_features: int = DEFAULT_IN_FEATURES,
        out_features: int = DEFAULT_OUT_FEATURES,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.linear.weight.to(device=x.device, dtype=x.dtype)
        bias = self.linear.bias.to(
            device=x.device,
            dtype=x.dtype) if self.linear.bias is not None else None
        return matmul_mish_mish(x, weight, bias)


batch_size = 1024
in_features = 8192
out_features = 8192


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features]
