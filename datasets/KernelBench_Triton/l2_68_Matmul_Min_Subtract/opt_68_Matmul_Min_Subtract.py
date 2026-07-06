import torch
import torch.nn as nn
import triton
import triton.language as tl
import torch_npu  # noqa: F401


@triton.jit
def _matmul_min_sub_kernel(
    X_ptr,
    WT_ptr,
    B_ptr,
    C_ptr,
    Y_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wk,
    stride_wn,
    stride_b,
    stride_ym,
    stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in tl.range(0, K, BLOCK_K):
        k_idxs = k_start + offs_k
        k_mask = k_idxs < K
        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + k_idxs[None, :] * stride_xk
        w_ptrs = WT_ptr + k_idxs[:, None] * stride_wk + offs_n[None, :] * stride_wn
        x = tl.load(x_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0, care_padding=False)
        w = tl.load(w_ptrs, mask=k_mask[:, None] & mask_n[None, :], other=0.0, care_padding=False)
        acc = tl.dot(x, w, acc, out_dtype=tl.float32)

    b = tl.load(B_ptr + offs_n * stride_b, mask=mask_n, other=0.0).to(tl.float32)
    c = tl.load(C_ptr).to(tl.float32)
    out = tl.minimum(acc + b[None, :] - c, 0.0)
    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, out, mask=mask_m[:, None] & mask_n[None, :])


def _cache_key(t: torch.Tensor):
    return (t.device.index, t.dtype, t.data_ptr(), tuple(t.shape), tuple(t.stride()))


def fused_linear_min_sub(x: torch.Tensor, weight_t: torch.Tensor, bias: torch.Tensor, constant: torch.Tensor):
    if x.device.type != "npu":
        raise ValueError("fused_linear_min_sub requires NPU tensors")
    if weight_t.device != x.device or bias.device != x.device or constant.device != x.device:
        raise ValueError("x, weight, bias, and constant must be on the same NPU device")
    if x.ndim != 2 or weight_t.ndim != 2 or bias.ndim != 1:
        raise ValueError("expected x to be 2D, transposed weight to be 2D, and bias to be 1D")
    if x.dtype != weight_t.dtype or x.dtype != bias.dtype or x.dtype != constant.dtype:
        raise ValueError("x, weight, bias, and constant must share the same dtype")
    if x.requires_grad or weight_t.requires_grad or bias.requires_grad or constant.requires_grad:
        raise ValueError("fused_linear_min_sub does not support autograd-tracked tensors")

    M, K = x.shape
    if weight_t.shape[0] != K:
        raise ValueError("weight shape is incompatible with x")
    N = weight_t.shape[1]
    if bias.shape[0] != N:
        raise ValueError("bias shape is incompatible with weight")
    if constant.numel() != 1:
        raise ValueError("constant must contain exactly one element")

    x_c = x.contiguous()
    wt_c = weight_t.contiguous()
    b_c = bias.contiguous()
    c_c = constant.contiguous()
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    # 64x128x64 fits comfortably in UB/L1 and halves the K-loop trip count vs
    # the baseline's 32-wide K tiles while preserving 16x16 Cube granularity.
    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 64
    num_blocks_m = triton.cdiv(M, BLOCK_M)
    num_blocks_n = triton.cdiv(N, BLOCK_N)
    grid = (num_blocks_m, num_blocks_n)

    _matmul_min_sub_kernel[grid](
        x_c,
        wt_c,
        b_c,
        c_c,
        y,
        M,
        N,
        K,
        x_c.stride(0),
        x_c.stride(1),
        wt_c.stride(0),
        wt_c.stride(1),
        b_c.stride(0),
        y.stride(0),
        y.stride(1),
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
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
        self._cached_weight_t = None
        self._cached_bias = None
        self._cached_constant = None
        self._cached_key = None

    def _materialize_params(self, x: torch.Tensor):
        key = (_cache_key(self.linear.weight),
               None if self.linear.bias is None else _cache_key(self.linear.bias),
               _cache_key(self.constant),
               x.device.index,
               x.dtype)
        if key != self._cached_key:
            weight = self.linear.weight.detach().to(device=x.device, dtype=x.dtype)
            self._cached_weight_t = weight.transpose(0, 1).contiguous()
            if self.linear.bias is None:
                self._cached_bias = torch.zeros(weight.shape[0], device=x.device, dtype=x.dtype)
            else:
                self._cached_bias = self.linear.bias.detach().to(device=x.device, dtype=x.dtype).contiguous()
            self._cached_constant = self.constant.detach().to(device=x.device, dtype=x.dtype).contiguous()
            self._cached_key = key
        return self._cached_weight_t, self._cached_bias, self._cached_constant

    def forward(self, x):
        if x.device.type != "npu":
            raise ValueError("ModelNew.forward requires NPU inputs")
        if x.requires_grad:
            raise ValueError("ModelNew.forward does not support autograd inputs")
        weight_t, bias, constant = self._materialize_params(x)
        return fused_linear_min_sub(x, weight_t, bias, constant)


batch_size = 128
in_features = 16384
out_features = 16384
constant = 2.0


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, constant]
