import torch
import torch.nn as nn
import triton
import triton.language as tl


def _require_npu_tensor(name: str, tensor: torch.Tensor) -> None:
    if tensor.device.type != "npu":
        raise AssertionError(f"{name} must be an NPU tensor")


@triton.jit
def _fused_linear_sigmoid_sum_kernel(
        x_ptr,  # float* [B, I]
        w_ptr,  # float* [H, I]
        b_ptr,  # float* [H]
        out_ptr,  # float* [B, 1]
        B: tl.constexpr,
        I: tl.constexpr,  # noqa: E741
        H: tl.constexpr,
        stride_xb,
        stride_xi,
        stride_wh,
        stride_wi,
        stride_bo,
        stride_ob,
        BLOCK_H: tl.constexpr,  # tile in hidden dimension (power-of-two)
        BLOCK_K: tl.constexpr,  # tile in input dimension (power-of-two)
):
    # One program per batch row
    pid_b = tl.program_id(0)

    # Accumulator for final sum over hidden dim
    acc_total = tl.zeros((), dtype=tl.float32)

    # Tile over hidden dimension
    h_start = 0
    while h_start < H:
        h_offsets = h_start + tl.arange(0, BLOCK_H)
        h_mask = h_offsets < H

        # Accumulator for current hidden tile (size BLOCK_H)
        z = tl.zeros((BLOCK_H, ), dtype=tl.float32)

        # Tile over input dimension
        k_start = 0
        while k_start < I:
            k_offsets = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_offsets < I

            # Load a tile of x[b, k]
            x_ptrs = x_ptr + pid_b * stride_xb + k_offsets * stride_xi
            x_vals = tl.load(x_ptrs, mask=k_mask, other=0.0).to(tl.float32)

            # Load corresponding tile of W[h, k]
            w_ptrs = w_ptr + (h_offsets[:, None] * stride_wh +
                              k_offsets[None, :] * stride_wi)
            w_tile = tl.load(w_ptrs,
                             mask=h_mask[:, None] & k_mask[None, :],
                             other=0.0).to(tl.float32)

            # Accumulate z[h] += dot(W[h, k_tile], x[k_tile])
            z += tl.sum(w_tile * x_vals[None, :], axis=1)

            k_start += BLOCK_K

        # Add bias
        b_vals = tl.load(b_ptr + h_offsets * stride_bo, mask=h_mask,
                         other=0.0).to(tl.float32)
        z = z + b_vals

        # Sigmoid and reduce over current hidden tile
        s = 1.0 / (1.0 + tl.exp(-z))
        s = tl.where(h_mask, s, 0.0)
        acc_total += tl.sum(s, axis=0)

        h_start += BLOCK_H

    # Store result to out[b, 0]
    out_ptrs = out_ptr + pid_b * stride_ob
    tl.store(out_ptrs, acc_total)


class ModelNew(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies sigmoid, and sums the result.
    Implemented with a fused Triton kernel on Ascend NPU.
    """

    def __init__(self, input_size, hidden_size):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Output tensor of shape (batch_size, 1).
        """
        return matmul_sigmoid_sum(x, self.linear.weight, self.linear.bias)


def matmul_sigmoid_sum(x: torch.Tensor, weight: torch.Tensor,
                       bias: torch.Tensor) -> torch.Tensor:
    _require_npu_tensor("x", x)
    _require_npu_tensor("weight", weight)
    _require_npu_tensor("bias", bias)
    if x.dim() != 2:
        raise ValueError(f"x must be 2D, got shape {tuple(x.shape)}")
    if weight.dim() != 2:
        raise ValueError(f"weight must be 2D, got shape {tuple(weight.shape)}")
    if bias.dim() != 1:
        raise ValueError(f"bias must be 1D, got shape {tuple(bias.shape)}")

    B, In = x.shape
    H, weight_k = weight.shape
    if weight_k != In:
        raise ValueError(
            f"weight second dimension must match x second dimension, got {weight_k} and {In}"
        )
    if bias.shape[0] != H:
        raise ValueError(
            f"bias length must match weight first dimension, got {bias.shape[0]} and {H}"
        )

    x_in = x.contiguous()
    weight_in = weight.contiguous()
    bias_in = bias.contiguous()

    out = torch.empty((B, 1), device=x_in.device, dtype=torch.float32)

    stride_xb, stride_xi = x_in.stride()
    stride_wh, stride_wi = weight_in.stride()
    stride_bo = bias_in.stride(0)
    stride_ob, _ = out.stride()

    grid = (B, )
    block_h = 128
    block_k = 128

    _fused_linear_sigmoid_sum_kernel[grid](
        x_in,
        weight_in,
        bias_in,
        out,
        B,
        In,
        H,
        stride_xb,
        stride_xi,
        stride_wh,
        stride_wi,
        stride_bo,
        stride_ob,
        BLOCK_H=block_h,
        BLOCK_K=block_k,
        num_warps=1,
        num_stages=1,
    )
    return out


batch_size = 128
input_size = 32768
hidden_size = 32768


def get_inputs():
    return [torch.rand(batch_size, input_size)]


def get_init_inputs():
    return [input_size, hidden_size]
