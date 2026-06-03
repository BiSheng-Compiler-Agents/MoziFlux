import torch
import torch.nn as nn
import triton
import triton.language as tl
import torch_npu  # noqa: F401


FP16_FAST_PATH = True
FP16_FAST_USE_MAX_CONTIGUOUS = True
FP16_FAST_BLOCK_M = 64
FP16_FAST_BLOCK_N = 256
FP16_FAST_BLOCK_K = 32
ACCURATE_BLOCK_M = 64
ACCURATE_BLOCK_N = 64
ACCURATE_BLOCK_K = 64


@triton.jit
def _matmul_bias_relu_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    USE_MAX_CONTIGUOUS: tl.constexpr,
    ADD_BIAS: tl.constexpr,
    APPLY_RELU: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    if USE_MAX_CONTIGUOUS:
        offs_m = tl.max_contiguous(offs_m, 16)
        offs_n = tl.max_contiguous(offs_n, 16)
        offs_k = tl.max_contiguous(offs_k, 16)

    tl.multiple_of(offs_m, 16)
    tl.multiple_of(offs_n, 16)
    tl.multiple_of(offs_k, 16)

    a_mask_m = offs_m < M
    b_mask_n = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_offs = k + offs_k
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak)
        b_ptrs = b_ptr + (k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        a_mask = a_mask_m[:, None] & (k_offs[None, :] < K)
        b_mask = (k_offs[:, None] < K) & b_mask_n[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        dot_a = a if a.dtype == tl.float32 else a.to(tl.float32)
        dot_b = b if b.dtype == tl.float32 else b.to(tl.float32)

        acc += tl.dot(dot_a, dot_b)

    if ADD_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=b_mask_n, other=0.0).to(tl.float32)
        acc = acc + bias[None, :]

    if APPLY_RELU:
        acc = tl.maximum(acc, 0.0)

    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=a_mask_m[:, None] & b_mask_n[None, :])


def _validate_inputs(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if x.ndim != 2 or weight.ndim != 2 or bias.ndim != 1:
        raise ValueError("expected x to be 2D, weight to be 2D, and bias to be 1D")
    if x.device.type != "npu" or weight.device.type != "npu" or bias.device.type != "npu":
        raise ValueError("fused_gemm_add_relu requires NPU tensors")
    if x.device != weight.device or x.device != bias.device:
        raise ValueError("x, weight, and bias must be on the same NPU device")
    if x.dtype != weight.dtype or x.dtype != bias.dtype:
        raise ValueError("x, weight, and bias must share the same dtype")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"unsupported dtype for fused_gemm_add_relu: {x.dtype}")
    if x.requires_grad or weight.requires_grad or bias.requires_grad:
        raise ValueError("fused_gemm_add_relu does not support autograd-tracked tensors")

    m, k = x.shape
    n = weight.shape[0]
    if weight.shape[1] != k:
        raise ValueError("weight shape is incompatible with x")
    if bias.shape[0] != n:
        raise ValueError("bias shape is incompatible with weight")
    return x.contiguous(), weight.contiguous(), bias.contiguous()


def fused_gemm_add_relu(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    a, b, bias_c = _validate_inputs(x, weight, bias)
    m, k = a.shape
    n = b.shape[0]
    use_triton_kernel = a.dtype == torch.float16 and m >= 256 and n >= 1024 and k >= 1024
    if not use_triton_kernel:
        out = torch.matmul(a, b.transpose(0, 1))
        out = out + bias_c
        return torch.relu(out)

    b_kn = b.transpose(0, 1).contiguous()
    use_fp16_fast_path = FP16_FAST_PATH and a.dtype == torch.float16 and m >= 256 and n >= 1024 and k >= 1024

    if use_fp16_fast_path:
        block_m = FP16_FAST_BLOCK_M
        block_n = FP16_FAST_BLOCK_N
        block_k = FP16_FAST_BLOCK_K
        use_max_contiguous = FP16_FAST_USE_MAX_CONTIGUOUS
        out_dtype = torch.float32
    else:
        block_m = ACCURATE_BLOCK_M
        block_n = ACCURATE_BLOCK_N
        block_k = ACCURATE_BLOCK_K
        use_max_contiguous = False
        out_dtype = torch.float32

    c = torch.empty((m, n), device=a.device, dtype=out_dtype)
    stride_am, stride_ak = a.stride()
    stride_bk, stride_bn = b_kn.stride()
    stride_cm, stride_cn = c.stride()
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))

    _matmul_bias_relu_kernel[grid](
        a,
        b_kn,
        bias_c,
        c,
        m,
        n,
        k,
        stride_am,
        stride_ak,
        stride_bn,
        stride_bk,
        stride_cm,
        stride_cn,
        use_max_contiguous,
        True,
        True,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return c if c.dtype == a.dtype else c.to(dtype=a.dtype)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias_shape):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        weight = self.gemm.weight.detach().to(device=x.device, dtype=x.dtype)
        bias = self.bias.detach().to(device=x.device, dtype=x.dtype)
        return fused_gemm_add_relu(x, weight, bias)


batch_size = 1024
in_features = 8192
out_features = 8192
bias_shape = (out_features,)


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, bias_shape]
