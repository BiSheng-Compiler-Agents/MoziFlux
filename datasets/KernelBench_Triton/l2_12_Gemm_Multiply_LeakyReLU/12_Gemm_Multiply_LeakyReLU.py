import torch
import torch.nn as nn
import triton
import triton.language as tl


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
def _linear_mul_leaky_kernel(
    A_ptr,  # [M, K]
    B_ptr,  # [N, K]  (weight)
    Bias_ptr,  # [N]
    C_ptr,  # [M, N]
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
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        k_offset = k_start * BLOCK_K
        k_mask = (k_offset + offs_k) < K
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am +
                          (k_offset + offs_k)[None, :] * stride_ak)
        b_ptrs = B_ptr + (offs_n[None, :] * stride_bn +
                          (k_offset + offs_k)[:, None] * stride_bk)
        a_mask = (offs_m[:, None] < M) & k_mask[None, :]
        b_mask = (offs_n[None, :] < N) & k_mask[:, None]

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)

    # Add bias [N] broadcast across M
    bias = tl.load(Bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    # Multiply by scalar
    acc = acc * multiplier

    # LeakyReLU
    out = tl.where(acc >= 0, acc, acc * negative_slope)

    # Store
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm +
                      offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, out, mask=c_mask)


def _fused_linear_mul_leaky(x: torch.Tensor, weight: torch.Tensor,
                            bias: torch.Tensor, multiplier: float,
                            negative_slope: float) -> torch.Tensor:
    """Run the fused GEMM, multiply, and LeakyReLU path on Ascend NPU."""
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
    x_c = x if x.is_contiguous() else x.contiguous()
    weight_c = weight if weight.is_contiguous() else weight.contiguous()
    bias_buf = bias if bias is not None else torch.zeros(
        N, device=x.device, dtype=x.dtype)
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    def grid(meta):
        return (triton.cdiv(M,
                            meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))

    _linear_mul_leaky_kernel[grid](
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
    return y


class ModelNew(nn.Module):
    """
    Simple model that performs a Gemm, multiplies the result, and applies LeakyReLU.
    """

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
    return [torch.rand(batch_size, in_features, device='npu')]


def get_init_inputs():
    return [in_features, out_features, multiplier, negative_slope]
