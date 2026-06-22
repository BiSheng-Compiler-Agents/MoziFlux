import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

_MAX_GRID = 65535
GROUP_M = 4
PERSISTENT_BLOCK_M = 128
PERSISTENT_BLOCK_N = 128
PERSISTENT_BLOCK_K = 64


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False))


@triton.autotune(
    configs=[
        triton.Config({
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 32
        },
                      num_stages=3,
                      num_warps=8),
        triton.Config({
            "BLOCK_M": 128,
            "BLOCK_N": 64,
            "BLOCK_K": 32
        },
                      num_stages=4,
                      num_warps=4),
        triton.Config({
            "BLOCK_M": 64,
            "BLOCK_N": 128,
            "BLOCK_K": 32
        },
                      num_stages=4,
                      num_warps=4),
        triton.Config({
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "BLOCK_K": 32
        },
                      num_stages=4,
                      num_warps=4),
        triton.Config({
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 64
        },
                      num_stages=4,
                      num_warps=8),
        triton.Config({
            "BLOCK_M": 64,
            "BLOCK_N": 128,
            "BLOCK_K": 64
        },
                      num_stages=4,
                      num_warps=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _linear_mul_leaky_kernel_auto(
    A_ptr,
    B_ptr,
    Bias_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    multiplier,
    negative_slope,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    offs_k = tl.arange(0, BLOCK_K)
    a_base = A_ptr + offs_m[:, None] * stride_am
    b_base = B_ptr + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        k_offset = k_start * BLOCK_K
        k = k_offset + offs_k
        k_mask = k < K
        a = tl.load(
            a_base + k[None, :] * stride_ak,
            mask=mask_m[:, None] & k_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            b_base + k[:, None] * stride_bk,
            mask=k_mask[:, None] & mask_n[None, :],
            other=0.0,
        )
        acc += tl.dot(a, b)

    bias = tl.load(Bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc = (acc + bias[None, :]) * multiplier
    out = tl.where(acc >= 0.0, acc, acc * negative_slope)
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm +
                      offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, out, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _linear_mul_leaky_kernel_persistent(
    A_ptr,
    B_ptr,
    Bias_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    multiplier,
    negative_slope,
    NUM_BLOCKS_M: tl.constexpr,
    NUM_BLOCKS_N: tl.constexpr,
    TOTAL_TILES: tl.constexpr,
    GROUP_M: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    for tile_id in tl.range(pid, TOTAL_TILES, nprog):
        group_width: tl.constexpr = GROUP_M * NUM_BLOCKS_N
        group_id = tile_id // group_width
        first_m = group_id * GROUP_M
        group_size = tl.minimum(NUM_BLOCKS_M - first_m, GROUP_M)
        pid_in_group = tile_id % group_width
        pid_m = first_m + (pid_in_group % group_size)
        pid_n = pid_in_group // group_size

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < M
        mask_n = offs_n < N
        a_base = A_ptr + offs_m[:, None] * stride_am
        b_base = B_ptr + offs_n[None, :] * stride_bn
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in tl.range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K
            a = tl.load(a_base + offs_k[None, :] * stride_ak,
                        mask=mask_m[:, None] & k_mask[None, :],
                        other=0.0,
                        care_padding=False)
            b = tl.load(b_base + offs_k[:, None] * stride_bk,
                        mask=k_mask[:, None] & mask_n[None, :],
                        other=0.0,
                        care_padding=False)
            al.compile_hint(a, "dot_pad_only_k")
            al.compile_hint(b, "dot_pad_only_k")
            acc += tl.dot(a, b)

        bias = tl.load(Bias_ptr + offs_n,
                       mask=mask_n,
                       other=0.0,
                       care_padding=False)
        acc = (acc + bias[None, :]) * multiplier
        out = tl.where(acc >= 0.0, acc, acc * negative_slope)
        c_ptrs = C_ptr + (offs_m[:, None] * stride_cm +
                          offs_n[None, :] * stride_cn)
        tl.store(c_ptrs, out, mask=mask_m[:, None] & mask_n[None, :])


def _fused_linear_mul_leaky(x: torch.Tensor, weight: torch.Tensor,
                            bias: torch.Tensor, multiplier: float,
                            negative_slope: float) -> torch.Tensor:
    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError(
            "_fused_linear_mul_leaky expects 2D input and weight tensors")
    if x.shape[1] != weight.shape[1]:
        raise ValueError("Input and weight must agree on the K dimension")
    if x.dtype != weight.dtype:
        raise TypeError("Input and weight must use the same dtype")
    if bias is not None:
        if bias.ndim != 1 or bias.shape[0] != weight.shape[0]:
            raise ValueError(
                "Bias must be a 1D tensor with length equal to weight.shape[0]"
            )
        if bias.dtype != x.dtype:
            raise TypeError("Bias dtype must match the input dtype")
    if not _is_npu_tensor(x) or not _is_npu_tensor(weight):
        raise ValueError("_fused_linear_mul_leaky requires Ascend NPU tensors")
    if x.device != weight.device:
        raise ValueError("Input and weight must be on the same device")
    if bias is not None and bias.device != x.device:
        raise ValueError("Bias must be on the same device as the input")

    M, K = x.shape
    N = weight.shape[0]

    if M < 512 or K < 4096 or N < 4096:
        return F.leaky_relu(F.linear(x, weight, bias) * float(multiplier),
                            negative_slope=float(negative_slope))

    x_c = x if x.is_contiguous() else x.contiguous()
    weight_c = weight if weight.is_contiguous() else weight.contiguous()
    bias_buf = bias if bias is not None else torch.zeros(
        N, device=x.device, dtype=x.dtype)
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    num_blocks_m = triton.cdiv(M, PERSISTENT_BLOCK_M)
    num_blocks_n = triton.cdiv(N, PERSISTENT_BLOCK_N)
    total_tiles = num_blocks_m * num_blocks_n

    if total_tiles <= _MAX_GRID:

        def grid(meta):
            return (triton.cdiv(M, meta["BLOCK_M"]),
                    triton.cdiv(N, meta["BLOCK_N"]))

        _linear_mul_leaky_kernel_auto[grid](
            x_c,
            weight_c,
            bias_buf,
            y,
            M,
            N,
            K,
            x_c.stride(0),
            x_c.stride(1),
            weight_c.stride(0),
            weight_c.stride(1),
            y.stride(0),
            y.stride(1),
            float(multiplier),
            float(negative_slope),
        )
    else:
        _linear_mul_leaky_kernel_persistent[(min(total_tiles, _MAX_GRID), )](
            x_c,
            weight_c,
            bias_buf,
            y,
            M,
            N,
            K,
            x_c.stride(0),
            x_c.stride(1),
            weight_c.stride(0),
            weight_c.stride(1),
            y.stride(0),
            y.stride(1),
            float(multiplier),
            float(negative_slope),
            NUM_BLOCKS_M=num_blocks_m,
            NUM_BLOCKS_N=num_blocks_n,
            TOTAL_TILES=total_tiles,
            GROUP_M=GROUP_M,
            BLOCK_M=PERSISTENT_BLOCK_M,
            BLOCK_N=PERSISTENT_BLOCK_N,
            BLOCK_K=PERSISTENT_BLOCK_K,
        )
    return y


class ModelNew(nn.Module):
    """Simple model that performs a Gemm, multiplies the result, and applies LeakyReLU."""

    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.multiplier = multiplier
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, x):
        if not _is_npu_tensor(x):
            raise ValueError("ModelNew expects input tensors on Ascend NPU")
        return _fused_linear_mul_leaky(
            x,
            self.gemm.weight,
            self.gemm.bias,
            float(self.multiplier),
            float(self.leaky_relu.negative_slope),
        )


batch_size = 1024
in_features = 8192
out_features = 8192
multiplier = 2.0
negative_slope = 0.1


def get_inputs():
    return [torch.rand(batch_size, in_features, device="npu")]


def get_init_inputs():
    return [in_features, out_features, multiplier, negative_slope]
