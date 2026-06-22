import torch
import torch.nn as nn
import triton
import triton.language as tl


def _require_npu_tensor(name: str, tensor: torch.Tensor) -> None:
    if tensor.device.type != "npu":
        raise AssertionError(f"{name} must be an NPU tensor")


@triton.jit
def _fused_linear_sigmoid_row_sum_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    B,
    K,
    H,
    stride_xm,
    stride_xk,
    stride_wj,
    stride_wk,
    stride_b,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= B:
        return

    x_row_ptr = x_ptr + pid * stride_xm
    row_sum = tl.zeros((), dtype=tl.float32)

    j_arange = tl.arange(0, BLOCK_H)
    k_arange = tl.arange(0, BLOCK_K)

    if K <= BLOCK_K:
        k_offsets = k_arange
        k_mask = k_offsets < K
        x_vals = tl.load(
            x_row_ptr + k_offsets * stride_xk,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)

        j_start = 0
        while j_start < H:
            j_offsets = j_start + j_arange
            j_mask = j_offsets < H

            w_tile = tl.load(
                w_ptr + j_offsets[:, None] * stride_wj +
                k_offsets[None, :] * stride_wk,
                mask=j_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            acc = tl.sum(w_tile * x_vals[None, :], axis=1)

            b_vals = tl.load(b_ptr + j_offsets * stride_b,
                             mask=j_mask,
                             other=0.0).to(tl.float32)
            acc = acc + b_vals

            s = tl.sigmoid(acc)
            s = tl.where(j_mask, s, 0.0)
            row_sum += tl.sum(s, axis=0)

            j_start += BLOCK_H
    else:
        j_start = 0
        while j_start < H:
            j_offsets = j_start + j_arange
            j_mask = j_offsets < H

            acc = tl.zeros((BLOCK_H, ), dtype=tl.float32)

            k_start = 0
            while k_start < K:
                k_offsets = k_start + k_arange
                k_mask = k_offsets < K

                x_vals = tl.load(
                    x_row_ptr + k_offsets * stride_xk,
                    mask=k_mask,
                    other=0.0,
                ).to(tl.float32)

                w_tile = tl.load(
                    w_ptr + j_offsets[:, None] * stride_wj +
                    k_offsets[None, :] * stride_wk,
                    mask=j_mask[:, None] & k_mask[None, :],
                    other=0.0,
                ).to(tl.float32)

                acc += tl.sum(w_tile * x_vals[None, :], axis=1)
                k_start += BLOCK_K

            b_vals = tl.load(b_ptr + j_offsets * stride_b,
                             mask=j_mask,
                             other=0.0).to(tl.float32)
            acc = acc + b_vals

            s = tl.sigmoid(acc)
            s = tl.where(j_mask, s, 0.0)
            row_sum += tl.sum(s, axis=0)

            j_start += BLOCK_H

    tl.store(out_ptr + pid, row_sum)


def _fused_linear_sigmoid_row_sum(x: torch.Tensor, weight: torch.Tensor,
                                  bias: torch.Tensor) -> torch.Tensor:
    _require_npu_tensor("x", x)
    _require_npu_tensor("weight", weight)
    _require_npu_tensor("bias", bias)
    B, K = x.shape
    H = weight.shape[0]

    x_c = x.contiguous()
    w_c = weight.contiguous()
    b_c = bias.contiguous()

    out = torch.empty(B, device=x.device, dtype=torch.float32)

    stride_xm, stride_xk = x_c.stride()
    stride_wj, stride_wk = w_c.stride()
    stride_b = b_c.stride(0)

    grid = (B, )

    _fused_linear_sigmoid_row_sum_kernel[grid](
        x_c,
        w_c,
        b_c,
        out,
        B,
        K,
        H,
        stride_xm,
        stride_xk,
        stride_wj,
        stride_wk,
        stride_b,
        BLOCK_H=32,
        BLOCK_K=8,
        num_warps=1,
        num_stages=2,
    )
    return out


@triton.jit
def _logsumexp_kernel(inp_ptr, out_ptr, B, BLOCK: tl.constexpr):
    if B <= BLOCK:
        idx = tl.arange(0, BLOCK)
        mask = idx < B
        vals = tl.load(inp_ptr + idx, mask=mask, other=-1.0e30)
        m = tl.max(vals, axis=0)
        sum_exp = tl.sum(tl.exp(vals - m), axis=0)
        result = m + tl.log(sum_exp)
        tl.store(out_ptr, result)
        return

    acc_max = -1.0e30
    acc_sum = 0.0

    offset = 0
    while offset < B:
        idx = offset + tl.arange(0, BLOCK)
        mask = idx < B
        vals = tl.load(inp_ptr + idx, mask=mask, other=-1.0e30)

        tile_max = tl.max(vals, axis=0)
        tile_sum = tl.sum(tl.exp(vals - tile_max), axis=0)

        new_max = tl.maximum(acc_max, tile_max)
        acc_sum = acc_sum * tl.exp(acc_max - new_max) + tile_sum * tl.exp(
            tile_max - new_max)
        acc_max = new_max

        offset += BLOCK

    result = acc_max + tl.log(acc_sum)
    tl.store(out_ptr, result)


def _logsumexp_triton(x: torch.Tensor) -> torch.Tensor:
    _require_npu_tensor("x", x)
    B = x.numel()
    out = torch.empty(1, device=x.device, dtype=torch.float32)
    _logsumexp_kernel[(1, )](x, out, B, BLOCK=128, num_warps=1, num_stages=1)
    return out[0]


@triton.jit
def _fused_rowsum_logsumexp_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    B,
    K,
    H,
    stride_xm,
    stride_xk,
    stride_wj,
    stride_wk,
    stride_b,
    BLOCK_B: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid != 0:
        return

    acc_max = -1.0e30
    acc_sum = 0.0

    rows_arange = tl.arange(0, BLOCK_B)
    j_arange = tl.arange(0, BLOCK_H)
    k_arange = tl.arange(0, BLOCK_K)

    b_start = 0
    while b_start < B:
        rows = b_start + rows_arange
        row_mask = rows < B

        row_sums = tl.zeros((BLOCK_B, ), dtype=tl.float32)

        j_start = 0
        while j_start < H:
            j_offsets = j_start + j_arange
            j_mask = j_offsets < H

            acc = tl.zeros((BLOCK_B, BLOCK_H), dtype=tl.float32)

            k_start = 0
            while k_start < K:
                k_offsets = k_start + k_arange
                k_mask = k_offsets < K

                x_tile = tl.load(
                    x_ptr + rows[:, None] * stride_xm +
                    k_offsets[None, :] * stride_xk,
                    mask=row_mask[:, None] & k_mask[None, :],
                    other=0.0,
                ).to(tl.float32)

                w_tile = tl.load(
                    w_ptr + j_offsets[:, None] * stride_wj +
                    k_offsets[None, :] * stride_wk,
                    mask=j_mask[:, None] & k_mask[None, :],
                    other=0.0,
                ).to(tl.float32)

                acc += tl.sum(x_tile[:, None, :] * w_tile[None, :, :], axis=2)

                k_start += BLOCK_K

            b_vals = tl.load(b_ptr + j_offsets * stride_b,
                             mask=j_mask,
                             other=0.0).to(tl.float32)
            acc = acc + b_vals[None, :]

            # Pre-mask invalid H entries; row_mask handled later in logsumexp step
            acc = tl.where(j_mask[None, :], acc, -50.0)
            s = tl.sigmoid(acc)

            row_sums += tl.sum(s, axis=1)

            j_start += BLOCK_H

        masked_vals = tl.where(row_mask, row_sums, -1.0e30)
        tile_max = tl.max(masked_vals, axis=0)
        tile_sum = tl.sum(tl.exp(masked_vals - tile_max), axis=0)

        new_max = tl.maximum(acc_max, tile_max)
        acc_sum = acc_sum * tl.exp(acc_max - new_max) + tile_sum * tl.exp(
            tile_max - new_max)
        acc_max = new_max

        b_start += BLOCK_B

    result = acc_max + tl.log(acc_sum)
    tl.store(out_ptr, result)


def _fused_rowsum_logsumexp(x: torch.Tensor, weight: torch.Tensor,
                            bias: torch.Tensor) -> torch.Tensor:
    _require_npu_tensor("x", x)
    _require_npu_tensor("weight", weight)
    _require_npu_tensor("bias", bias)
    B, K = x.shape
    H = weight.shape[0]

    x_c = x.contiguous()
    w_c = weight.contiguous()
    b_c = bias.contiguous()

    stride_xm, stride_xk = x_c.stride()
    stride_wj, stride_wk = w_c.stride()
    stride_b = b_c.stride(0)

    out = torch.empty(1, device=x.device, dtype=torch.float32)

    _fused_rowsum_logsumexp_kernel[(1, )](
        x_c,
        w_c,
        b_c,
        out,
        B,
        K,
        H,
        stride_xm,
        stride_xk,
        stride_wj,
        stride_wk,
        stride_b,
        BLOCK_B=64,
        BLOCK_H=32,  # power-of-2 required for vector alignment on Ascend NPU
        BLOCK_K=8,
        num_warps=4,
        num_stages=2,
    )
    return out[0]


def gemm_sigmoid_sum_logsumexp(x: torch.Tensor, weight: torch.Tensor,
                               bias: torch.Tensor) -> torch.Tensor:
    return _fused_rowsum_logsumexp(x, weight, bias)


class ModelNew(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(ModelNew, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        if self.linear1.bias is None:
            raise RuntimeError(
                "ModelNew requires linear1.bias for the fused Triton path")
        return gemm_sigmoid_sum_logsumexp(x, self.linear1.weight,
                                          self.linear1.bias)


batch_size = 128
input_size = 10
hidden_size = 20
output_size = 5


def get_inputs():
    return [torch.randn(batch_size, input_size)]


def get_init_inputs():
    return [input_size, hidden_size, output_size]
