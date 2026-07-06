import torch
import torch.nn as nn
import triton
import triton.language as tl


def _require_npu_tensor(name: str, tensor: torch.Tensor) -> None:
    if tensor.device.type != "npu":
        raise AssertionError(f"{name} must be an NPU tensor")


@triton.jit
def _gemm_sigmoid_rowsum_dot_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    row_ptr,
    B,
    K,
    H,
    stride_xm,
    stride_xk,
    stride_wj,
    stride_wk,
    stride_b,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_h_base = tl.arange(0, BLOCK_H)
    offs_k_base = tl.arange(0, BLOCK_K)
    m_mask = offs_m < B

    row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)

    h_start = 0
    while h_start < H:
        offs_h = h_start + offs_h_base
        h_mask = offs_h < H
        acc = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)

        k_start = 0
        while k_start < K:
            offs_k = k_start + offs_k_base
            k_mask = offs_k < K
            x_tile = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            w_tile = tl.load(
                w_ptr + offs_h[None, :] * stride_wj + offs_k[:, None] * stride_wk,
                mask=k_mask[:, None] & h_mask[None, :],
                other=0.0,
            )
            acc = tl.dot(x_tile, w_tile, acc)
            k_start += BLOCK_K

        bias = tl.load(b_ptr + offs_h * stride_b, mask=h_mask, other=0.0).to(tl.float32)
        z = acc + bias[None, :]
        sig = tl.sigmoid(z)
        sig = tl.where(m_mask[:, None] & h_mask[None, :], sig, 0.0)
        row_sum += tl.sum(sig, axis=1)
        h_start += BLOCK_H

    tl.store(row_ptr + offs_m, row_sum, mask=m_mask)


@triton.jit
def _logsumexp_kernel(inp_ptr, out_ptr, B, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    acc_max = -1.0e30
    acc_sum = 0.0

    start = 0
    while start < B:
        idx = start + offs
        mask = idx < B
        vals = tl.load(inp_ptr + idx, mask=mask, other=-1.0e30).to(tl.float32)
        tile_max = tl.max(vals, axis=0)
        tile_sum = tl.sum(tl.exp(vals - tile_max), axis=0)
        new_max = tl.maximum(acc_max, tile_max)
        acc_sum = acc_sum * tl.exp(acc_max - new_max) + tile_sum * tl.exp(tile_max - new_max)
        acc_max = new_max
        start += BLOCK

    tl.store(out_ptr, acc_max + tl.log(acc_sum))


def gemm_sigmoid_sum_logsumexp(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    _require_npu_tensor("x", x)
    _require_npu_tensor("weight", weight)
    _require_npu_tensor("bias", bias)
    B, K = x.shape
    H = weight.shape[0]

    x_c = x.contiguous()
    w_c = weight.contiguous()
    b_c = bias.contiguous()
    rows = torch.empty((B,), device=x.device, dtype=torch.float32)
    out = torch.empty((1,), device=x.device, dtype=torch.float32)

    stride_xm, stride_xk = x_c.stride()
    stride_wj, stride_wk = w_c.stride()
    stride_b = b_c.stride(0)

    block_m = 16
    grid_rows = (triton.cdiv(B, block_m),)
    _gemm_sigmoid_rowsum_dot_kernel[grid_rows](
        x_c,
        w_c,
        b_c,
        rows,
        B,
        K,
        H,
        stride_xm,
        stride_xk,
        stride_wj,
        stride_wk,
        stride_b,
        BLOCK_M=16,
        BLOCK_H=32,
        BLOCK_K=16,
        num_warps=1,
        num_stages=2,
    )
    _logsumexp_kernel[(1,)](rows, out, B, BLOCK=256, num_warps=1, num_stages=2)
    return out[0]


class ModelNew(nn.Module):
    """Optimized model for logsumexp(sum(sigmoid(linear1(x))), dim=0)."""

    def __init__(self, input_size, hidden_size, output_size):
        super(ModelNew, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        if self.linear1.bias is None:
            raise RuntimeError("ModelNew requires linear1.bias for the fused Triton path")
        return gemm_sigmoid_sum_logsumexp(x, self.linear1.weight, self.linear1.bias)


batch_size = 128
input_size = 10
hidden_size = 20
output_size = 5


def get_inputs():
    return [torch.randn(batch_size, input_size)]


def get_init_inputs():
    return [input_size, hidden_size, output_size]
