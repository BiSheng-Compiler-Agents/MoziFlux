import torch
import torch.nn as nn
import triton
import triton.language as tl
import torch_npu  # noqa: F401


@triton.jit
def _matmul_bias_relu_kernel(
    a_ptr,  # [M, K]
    b_ptr,  # [N, K] (weight), accessed as [K, N] via strides
    bias_ptr,  # [N]
    c_ptr,  # [M, N]
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    ADD_BIAS: tl.constexpr,
    APPLY_RELU: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Provide alignment/contiguity hints to the compiler for better codegen
    tl.multiple_of(offs_m, 16)
    tl.multiple_of(offs_n, 16)
    tl.multiple_of(offs_k, 16)

    # Masks for M and N bounds (K handled per-iteration)
    a_mask_m = offs_m < M
    b_mask_n = offs_n < N

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Regular pipelined K loop; let Triton pipeline via num_stages
    k = 0
    while k < K:
        k_offs = k + offs_k

        # Compute tile pointers
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak)  # [BM, BK]
        b_ptrs = b_ptr + (k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn)  # [BK, BN]

        # Masks for this tile
        a_mask = (a_mask_m[:, None]) & (k_offs[None, :] < K)
        b_mask = (k_offs[:, None] < K) & (b_mask_n[None, :])

        # Load tiles
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Use Tensor Cores: cast inputs to fp16 and accumulate in fp32
        a = a.to(tl.float16)
        b = b.to(tl.float16)

        acc += tl.dot(a, b)
        k += BLOCK_K

    # Epilogue: bias and ReLU
    if ADD_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=b_mask_n, other=0.0).to(tl.float32)
        acc = acc + bias[None, :]

    if APPLY_RELU:
        acc = tl.maximum(acc, 0.0)

    # Store results
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
    c = torch.empty((m, n), device=a.device, dtype=torch.float32)

    stride_am, stride_ak = a.stride()
    stride_bn, stride_bk = b.stride()
    stride_cm, stride_cn = c.stride()

    if (m >= 128) and (n >= 512) and (k >= 512):
        block_m = 64
        block_n = 64
        block_k = 64
        num_warps = 4
        num_stages = 4
    else:
        block_m = 64
        block_n = 128
        block_k = 64
        num_warps = 8
        num_stages = 4

    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _matmul_bias_relu_kernel[grid](
        a,
        b,
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
        True,
        True,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return c if a.dtype == torch.float32 else c.to(dtype=a.dtype)


class ModelNew(nn.Module):
    """
    Simple model that performs a matrix multiplication, adds a bias term, and applies ReLU.
    Uses a fused Triton kernel on Ascend NPU.
    """
    def __init__(self, in_features, out_features, bias_shape):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor with shape (batch_size, out_features).
        """
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