import torch
import torch.nn as nn
import triton
import triton.language as tl
import triton.runtime.driver as driver
import torch_npu  # noqa: F401


BLOCK_M = 64
BLOCK_N = 256
BLOCK_K = 64
_MIN_TRITON_M = 256
_MIN_TRITON_N = 1024
_MIN_TRITON_K = 1024


@triton.jit
def _matmul_bias_relu_kernel_opt(
    a_ptr,  # [M, K]
    b_ptr,  # [K, N] contiguous transposed weight
    bias_ptr,  # [N]
    c_ptr,  # [M, N], fp32 accumulator output
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    NUM_BLOCKS_M: tl.constexpr,
    NUM_BLOCKS_N: tl.constexpr,
    USE_MAX_CONTIGUOUS: tl.constexpr,
    BLOCK_M_: tl.constexpr,
    BLOCK_N_: tl.constexpr,
    BLOCK_K_: tl.constexpr,
):
    pid = tl.program_id(0)
    total_tiles: tl.constexpr = NUM_BLOCKS_M * NUM_BLOCKS_N

    # 1D persistent-style scheduling keeps the launch bounded to physical AI Cores.
    for tile_id in range(pid, total_tiles, tl.num_programs(0)):
        pid_m = tile_id // NUM_BLOCKS_N
        pid_n = tile_id - pid_m * NUM_BLOCKS_N

        offs_m = pid_m * BLOCK_M_ + tl.arange(0, BLOCK_M_)
        offs_n = pid_n * BLOCK_N_ + tl.arange(0, BLOCK_N_)
        offs_k = tl.arange(0, BLOCK_K_)

        if USE_MAX_CONTIGUOUS:
            offs_m = tl.max_contiguous(offs_m, BLOCK_M_)
            offs_n = tl.max_contiguous(offs_n, BLOCK_N_)
            offs_k = tl.max_contiguous(offs_k, BLOCK_K_)
        tl.multiple_of(offs_m, 16)
        tl.multiple_of(offs_n, 16)
        tl.multiple_of(offs_k, 16)

        mask_m = offs_m < M
        mask_n = offs_n < N
        a_base = a_ptr + offs_m[:, None] * stride_am
        b_base = b_ptr + offs_n[None, :] * stride_bn
        acc = tl.zeros((BLOCK_M_, BLOCK_N_), dtype=tl.float32)

        for k0 in tl.range(0, K, BLOCK_K_):
            k_idxs = k0 + offs_k
            k_mask = k_idxs < K
            a = tl.load(
                a_base + k_idxs[None, :] * stride_ak,
                mask=mask_m[:, None] & k_mask[None, :],
                other=0.0,
                care_padding=False,
            )
            b = tl.load(
                b_base + k_idxs[:, None] * stride_bk,
                mask=k_mask[:, None] & mask_n[None, :],
                other=0.0,
                care_padding=False,
            )
            # In-place Cube accumulation avoids the fp32 temporary produced by acc += tl.dot(...).
            acc = tl.dot(a, b, acc)

        bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0, care_padding=False).to(tl.float32)
        acc = tl.maximum(acc + bias[None, :], 0.0)
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


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


def _torch_fallback(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return torch.relu(torch.matmul(a, b.transpose(0, 1)) + bias)


def fused_gemm_add_relu(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    a, b, bias_c = _validate_inputs(x, weight, bias)
    m, k = a.shape
    n = b.shape[0]

    # Keep fp32/bf16 and small shapes on ACL/PyTorch; the custom path is tuned for the stated fp16 large GEMM.
    if a.dtype != torch.float16 or m < _MIN_TRITON_M or n < _MIN_TRITON_N or k < _MIN_TRITON_K:
        return _torch_fallback(a, b, bias_c)

    b_kn = b.transpose(0, 1).contiguous()
    c = torch.empty((m, n), device=a.device, dtype=torch.float32)
    num_blocks_m = triton.cdiv(m, BLOCK_M)
    num_blocks_n = triton.cdiv(n, BLOCK_N)
    total_tiles = num_blocks_m * num_blocks_n
    device = torch.npu.current_device()
    num_aicore = driver.active.utils.get_device_properties(device)["num_aicore"]
    grid = (min(total_tiles, num_aicore),)

    _matmul_bias_relu_kernel_opt[grid](
        a,
        b_kn,
        bias_c,
        c,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b_kn.stride(1),
        b_kn.stride(0),
        c.stride(0),
        c.stride(1),
        NUM_BLOCKS_M=num_blocks_m,
        NUM_BLOCKS_N=num_blocks_n,
        USE_MAX_CONTIGUOUS=True,
        BLOCK_M_=BLOCK_M,
        BLOCK_N_=BLOCK_N,
        BLOCK_K_=BLOCK_K,
        num_warps=4,
        num_stages=2,
    )
    return c.to(dtype=a.dtype)


class ModelNew(nn.Module):
    """GEMM + bias add + ReLU using a large-fp16 Triton path and ACL fallback."""

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
