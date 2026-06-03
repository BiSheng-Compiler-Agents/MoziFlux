import torch
import torch.nn as nn
import triton
import triton.language as tl


def _require_npu_tensor(name: str, tensor: torch.Tensor) -> None:
    if tensor.device.type != "npu":
        raise AssertionError(f"{name} must be an NPU tensor")


@triton.jit
def _fused_linear_sigmoid_sum_kernel(
    x_ptr,         # float* [B, I]
    w_ptr,         # float* [H, I]
    b_ptr,         # float* [H]
    out_ptr,       # float* [B]
    B: tl.constexpr,
    I: tl.constexpr,
    H: tl.constexpr,
    stride_xb,
    stride_xi,
    stride_wh,
    stride_wi,
    stride_bo,
    stride_ob,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    acc_total = tl.zeros((), dtype=tl.float32)
    k_offsets = tl.arange(0, BLOCK_K)

    h_start = 0
    while h_start < H:
        h0_offsets = h_start + tl.arange(0, BLOCK_H)
        h1_offsets = h_start + BLOCK_H + tl.arange(0, BLOCK_H)
        h0_mask = h0_offsets < H
        h1_mask = h1_offsets < H

        z0 = tl.zeros((BLOCK_H,), dtype=tl.float32)
        z1 = tl.zeros((BLOCK_H,), dtype=tl.float32)

        k_start = 0
        while k_start < I:
            k_idx = k_start + k_offsets
            k_mask = k_idx < I

            x_ptrs = x_ptr + pid_b * stride_xb + k_idx * stride_xi
            x_vals = tl.load(x_ptrs, mask=k_mask, other=0.0).to(tl.float32)

            w0_ptrs = w_ptr + (h0_offsets[:, None] * stride_wh + k_idx[None, :] * stride_wi)
            w1_ptrs = w_ptr + (h1_offsets[:, None] * stride_wh + k_idx[None, :] * stride_wi)
            w0_tile = tl.load(w0_ptrs, mask=h0_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
            w1_tile = tl.load(w1_ptrs, mask=h1_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)

            z0 += tl.sum(w0_tile * x_vals[None, :], axis=1)
            z1 += tl.sum(w1_tile * x_vals[None, :], axis=1)

            k_start += BLOCK_K

        b0 = tl.load(b_ptr + h0_offsets * stride_bo, mask=h0_mask, other=0.0).to(tl.float32)
        b1 = tl.load(b_ptr + h1_offsets * stride_bo, mask=h1_mask, other=0.0).to(tl.float32)
        z0 = z0 + b0
        z1 = z1 + b1

        s0 = 1.0 / (1.0 + tl.exp(-z0))
        s1 = 1.0 / (1.0 + tl.exp(-z1))
        acc_total += tl.sum(tl.where(h0_mask, s0, 0.0), axis=0)
        acc_total += tl.sum(tl.where(h1_mask, s1, 0.0), axis=0)

        h_start += 2 * BLOCK_H

    tl.store(out_ptr + pid_b * stride_ob, acc_total)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        return matmul_sigmoid_sum(x, self.linear.weight, self.linear.bias)


def matmul_sigmoid_sum(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    _require_npu_tensor("x", x)
    _require_npu_tensor("weight", weight)
    _require_npu_tensor("bias", bias)
    if x.dim() != 2:
        raise ValueError(f"x must be 2D, got shape {tuple(x.shape)}")
    if weight.dim() != 2:
        raise ValueError(f"weight must be 2D, got shape {tuple(weight.shape)}")
    if bias.dim() != 1:
        raise ValueError(f"bias must be 1D, got shape {tuple(bias.shape)}")

    B, I = x.shape
    H, weight_k = weight.shape
    if weight_k != I:
        raise ValueError(f"weight second dimension must match x second dimension, got {weight_k} and {I}")
    if bias.shape[0] != H:
        raise ValueError(f"bias length must match weight first dimension, got {bias.shape[0]} and {H}")

    x_in = x.contiguous()
    weight_in = weight.contiguous()
    bias_in = bias.contiguous()
    out = torch.zeros((B,), device=x_in.device, dtype=torch.float32)

    stride_xb, stride_xi = x_in.stride()
    stride_wh, stride_wi = weight_in.stride()
    stride_bo = bias_in.stride(0)
    stride_ob = out.stride(0)

    block_h = 64
    block_k = 128
    grid = (B,)

    _fused_linear_sigmoid_sum_kernel[grid](
        x_in,
        weight_in,
        bias_in,
        out,
        B,
        I,
        H,
        stride_xb,
        stride_xi,
        stride_wh,
        stride_wi,
        stride_bo,
        stride_ob,
        BLOCK_H=block_h,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=2,
    )
    return out[:, None]


batch_size = 128
input_size = 32768
hidden_size = 32768


def get_inputs():
    return [torch.rand(batch_size, input_size)]


def get_init_inputs():
    return [input_size, hidden_size]
