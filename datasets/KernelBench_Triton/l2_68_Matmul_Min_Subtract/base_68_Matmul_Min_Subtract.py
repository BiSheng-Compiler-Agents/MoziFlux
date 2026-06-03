import torch
import torch.nn as nn
import triton
import triton.language as tl
import torch_npu  # noqa: F401


@triton.jit
def _fused_linear_min_sub_kernel(
    X_ptr, W_ptr, B_ptr, C_ptr, Y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_b,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    x_base_ptrs = X_ptr + offs_m[:, None] * stride_xm
    w_base_ptrs = W_ptr + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + offs_k
        k_mask = k_offsets < K
        x = tl.load(
            x_base_ptrs + k_offsets[None, :] * stride_xk,
            mask=mask_m[:, None] & k_mask[None, :],
            other=0.0,
        )
        w = tl.load(
            w_base_ptrs + k_offsets[:, None] * stride_wk,
            mask=k_mask[:, None] & mask_n[None, :],
            other=0.0,
        )
        acc += tl.dot(x, w, out_dtype=tl.float32)

    b = tl.load(B_ptr + offs_n * stride_b, mask=mask_n, other=0.0).to(tl.float32)
    c = tl.load(C_ptr).to(tl.float32)
    bc = b - c
    out = tl.minimum(acc + bc[None, :], 0.0)

    y_ptrs = Y_ptr + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn)
    tl.store(y_ptrs, out, mask=mask_m[:, None] & mask_n[None, :])


def fused_linear_min_sub(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, constant: torch.Tensor):
    if x.device.type != "npu":
        raise ValueError("fused_linear_min_sub requires NPU tensors")
    if weight.device != x.device or bias.device != x.device or constant.device != x.device:
        raise ValueError("x, weight, bias, and constant must be on the same NPU device")
    if x.ndim != 2 or weight.ndim != 2 or bias.ndim != 1:
        raise ValueError("expected x to be 2D, weight to be 2D, and bias to be 1D")
    if x.dtype != weight.dtype or x.dtype != bias.dtype or x.dtype != constant.dtype:
        raise ValueError("x, weight, bias, and constant must share the same dtype")
    if x.requires_grad or weight.requires_grad or bias.requires_grad or constant.requires_grad:
        raise ValueError("fused_linear_min_sub does not support autograd-tracked tensors")

    # Shapes
    M, K = x.shape
    N = weight.shape[0]
    if weight.shape[1] != K:
        raise ValueError("weight shape is incompatible with x")
    if bias.shape[0] != N:
        raise ValueError("bias shape is incompatible with weight")
    if constant.numel() != 1:
        raise ValueError("constant must contain exactly one element")

    x_c = x.contiguous()
    w_c = weight.transpose(0, 1).contiguous()
    b_c = bias.contiguous()
    c_c = constant.contiguous()

    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
    _fused_linear_min_sub_kernel[grid](
        x_c, w_c, b_c, c_c, y,
        M, N, K,
        x_c.stride(0), x_c.stride(1),
        w_c.stride(1), w_c.stride(0),
        b_c.stride(0),
        y.stride(0), y.stride(1),
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
        num_stages=4, num_warps=4,
    )
    return y


class ModelNew(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies minimum, and subtracts a constant.
    """
    def __init__(self, in_features, out_features, constant):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.constant = nn.Parameter(torch.tensor(constant))

    def forward(self, x):
        if x.device.type != "npu":
            raise ValueError("ModelNew.forward requires NPU inputs")
        if x.requires_grad:
            raise ValueError("ModelNew.forward does not support autograd inputs")

        weight = self.linear.weight.detach().to(device=x.device, dtype=x.dtype)
        if self.linear.bias is None:
            bias = torch.zeros(weight.shape[0], device=x.device, dtype=x.dtype)
        else:
            bias = self.linear.bias.detach().to(device=x.device, dtype=x.dtype)
        constant = self.constant.detach().to(device=x.device, dtype=x.dtype)
        return fused_linear_min_sub(x, weight, bias, constant)
batch_size = 128
in_features = 16384
out_features = 16384
constant = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_features)]
def get_init_inputs():
    return [in_features, out_features, constant]
