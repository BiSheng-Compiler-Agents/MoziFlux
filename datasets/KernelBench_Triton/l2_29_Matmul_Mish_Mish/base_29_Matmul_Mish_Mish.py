import torch
import torch.nn as nn
import triton
import triton.language as tl

DEFAULT_BATCH_SIZE = 1024
DEFAULT_IN_FEATURES = 8192
DEFAULT_OUT_FEATURES = 8192


@triton.jit
def linear_mish2_rowwise(x_ptr, w_ptr, b_ptr, y_ptr, M: tl.constexpr,
                         N: tl.constexpr, K: tl.constexpr, stride_xm,
                         stride_xk, stride_wn, stride_wk, stride_ym, stride_yn,
                         BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                         USE_HINTS: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if USE_HINTS:
        offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)
    acc = tl.zeros((BLOCK_N, ), dtype=tl.float32)
    x_row_ptr = x_ptr + pid_m * stride_xm
    k0 = 0
    while k0 < K:
        offs_k = k0 + tl.arange(0, BLOCK_K)
        if USE_HINTS:
            offs_k = tl.max_contiguous(tl.multiple_of(offs_k, BLOCK_K),
                                       BLOCK_K)
        k_mask = offs_k < K
        x_vals = tl.load(x_row_ptr + offs_k * stride_xk,
                         mask=k_mask,
                         other=0.0).to(tl.float32)
        w_ptrs = w_ptr + (offs_n[:, None] * stride_wn +
                          offs_k[None, :] * stride_wk)
        w_mask = (offs_n[:, None] < N) & k_mask[None, :]
        w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)
        acc += tl.sum(w_vals * x_vals[None, :], axis=1)
        k0 += BLOCK_K
    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc += bias
    threshold = 20.0
    abs_acc = tl.abs(acc)
    sp1_stable = tl.log(1.0 + tl.exp(-abs_acc)) + tl.maximum(acc, 0.0)
    sp1 = tl.where(acc > threshold, acc, sp1_stable)
    t1 = tl.exp(-2.0 * sp1)
    tanh_sp1 = 1.0 - 2.0 * t1 / (1.0 + t1)
    m1 = acc * tanh_sp1
    abs_m1 = tl.abs(m1)
    sp2_stable = tl.log(1.0 + tl.exp(-abs_m1)) + tl.maximum(m1, 0.0)
    sp2 = tl.where(m1 > threshold, m1, sp2_stable)
    t2 = tl.exp(-2.0 * sp2)
    tanh_sp2 = 1.0 - 2.0 * t2 / (1.0 + t2)
    out = m1 * tanh_sp2
    y_ptrs = y_ptr + pid_m * stride_ym + offs_n * stride_yn
    tl.store(y_ptrs, out, mask=offs_n < N)


def matmul_mish_mish(x: torch.Tensor,
                     weight: torch.Tensor,
                     bias: torch.Tensor | None = None) -> torch.Tensor:
    if x.device.type != 'npu':
        raise RuntimeError('matmul_mish_mish requires Ascend NPU tensors.')
    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError(
            'matmul_mish_mish expects x and weight to be 2D tensors.')
    if x.shape[1] != weight.shape[1]:
        raise ValueError('x.shape[1] must match weight.shape[1].')
    if x.device != weight.device:
        raise ValueError('x and weight must be on the same device.')
    if x.dtype != weight.dtype:
        raise ValueError('x and weight must use the same dtype.')
    if bias is None:
        bias = torch.zeros(weight.shape[0], device=x.device, dtype=x.dtype)
    elif bias.ndim != 1 or bias.shape[0] != weight.shape[0]:
        raise ValueError(
            'bias must be a 1D tensor with shape [weight.shape[0]].')
    elif bias.device != x.device:
        raise ValueError('bias must be on the same device as x.')
    elif bias.dtype != x.dtype:
        raise ValueError('bias must use the same dtype as x.')
    if x.dtype not in (torch.float16, torch.float32, torch.bfloat16):
        raise TypeError(
            'matmul_mish_mish supports float16, float32, and bfloat16 inputs.')
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    m, k = x.shape
    n = weight.shape[0]
    y = torch.empty((m, n), device=x.device, dtype=x.dtype)
    if m == 0 or n == 0:
        return y
    if x.dtype == torch.float16 and m == 1024 and n == 8192 and k == 8192:
        block_n = 32
        block_k = 12
        use_hints = False
        num_warps = 4
        num_stages = 1
    else:
        block_n = 32
        block_k = 16
        use_hints = False
        num_warps = 4
        num_stages = 1
    grid = (m, triton.cdiv(n, block_n))
    linear_mish2_rowwise[grid](x,
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
                               BLOCK_N=block_n,
                               BLOCK_K=block_k,
                               USE_HINTS=use_hints,
                               num_warps=num_warps,
                               num_stages=num_stages)
    return y


class ModelNew(nn.Module):

    def __init__(self,
                 in_features: int = DEFAULT_IN_FEATURES,
                 out_features: int = DEFAULT_OUT_FEATURES):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.linear.weight.to(device=x.device, dtype=x.dtype)
        bias = None if self.linear.bias is None else self.linear.bias.to(
            device=x.device, dtype=x.dtype)
        return matmul_mish_mish(x, weight, bias)


batch_size = 1024
in_features = 8192
out_features = 8192


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features]
