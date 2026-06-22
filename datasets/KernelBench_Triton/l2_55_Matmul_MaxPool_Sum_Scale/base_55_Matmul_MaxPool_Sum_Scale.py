import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _linear_maxpool_sum_scale_kernel(
    x_ptr,
    wt_ptr,
    b_ptr,
    out_ptr,
    B,
    IN_F,
    OUT_F,
    X_STRIDE,
    WT_ROW_STRIDE,
    WT_COL_STRIDE,
    SCALE,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    pair_idx = tl.arange(0, BLOCK_N)
    sum_acc = tl.zeros((BLOCK_M, ), dtype=tl.float32)

    for n_start in range(0, OUT_F, BLOCK_N * 2):
        offs_n0 = n_start + pair_idx * 2
        offs_n1 = offs_n0 + 1
        acc0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, IN_F, BLOCK_K):
            k_idx = k_start + offs_k
            x_ptrs = x_ptr + offs_m[:, None] * X_STRIDE + k_idx[None, :]
            wt0_ptrs = wt_ptr + k_idx[:, None] * WT_ROW_STRIDE + offs_n0[
                None, :] * WT_COL_STRIDE
            wt1_ptrs = wt_ptr + k_idx[:, None] * WT_ROW_STRIDE + offs_n1[
                None, :] * WT_COL_STRIDE

            x_mask = (offs_m[:, None] < B) & (k_idx[None, :] < IN_F)
            wt0_mask = (k_idx[:, None] < IN_F) & (offs_n0[None, :] < OUT_F)
            wt1_mask = (k_idx[:, None] < IN_F) & (offs_n1[None, :] < OUT_F)

            x_block = tl.load(x_ptrs, mask=x_mask, other=0.0)
            wt0_block = tl.load(wt0_ptrs, mask=wt0_mask, other=0.0)
            wt1_block = tl.load(wt1_ptrs, mask=wt1_mask, other=0.0)

            acc0 += tl.dot(x_block, wt0_block)
            acc1 += tl.dot(x_block, wt1_block)

        b0 = tl.load(b_ptr + offs_n0, mask=offs_n0 < OUT_F, other=0.0)
        b1 = tl.load(b_ptr + offs_n1, mask=offs_n1 < OUT_F, other=0.0)
        val0 = acc0 + b0[None, :]
        val1 = acc1 + b1[None, :]
        pooled = tl.maximum(val0, val1)
        sum_acc += tl.sum(pooled, axis=1)

    sum_acc *= SCALE
    tl.store(out_ptr + offs_m, sum_acc, mask=offs_m < B)


class ModelNew(nn.Module):
    """
    Model that performs matrix multiplication, max pooling, sum, and scaling.
    """

    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.max_pool = nn.MaxPool1d(
            kernel_size)  # kept for state/compat; computation is fused
        self.scale_factor = scale_factor

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size,).
        """
        if x.device.type != "npu":
            raise RuntimeError(
                f"ModelNew expects an NPU tensor, got device={x.device!s}")

        # Ensure contiguous tensors
        x = x.contiguous()
        Wt = self.matmul.weight.t().contiguous()
        b = self.matmul.bias
        if b is None:
            b = torch.zeros(Wt.shape[1], device=Wt.device, dtype=Wt.dtype)
        else:
            b = b.contiguous()

        B = x.shape[0]
        IN_F = x.shape[1]
        OUT_F = Wt.shape[1]

        # Output tensor
        out = torch.empty(B, device=x.device, dtype=torch.float32)

        BLOCK_M = 16
        BLOCK_N = 128
        BLOCK_K = 32
        grid = (triton.cdiv(B, BLOCK_M), )

        _linear_maxpool_sum_scale_kernel[grid](
            x,
            Wt,
            b,
            out,
            B,
            IN_F,
            OUT_F,
            x.stride(0),
            Wt.stride(0),
            Wt.stride(1),
            float(self.scale_factor),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_warps=8,
            num_stages=2,
        )
        return out


def run_model(x, weight, bias, kernel_size, scale_factor):
    if x.device.type != "npu":
        raise RuntimeError(
            f"run_model expects x on NPU, got device={x.device!s}")
    if weight.device.type != "npu":
        raise RuntimeError(
            f"run_model expects weight on NPU, got device={weight.device!s}")
    if bias.device.type != "npu":
        raise RuntimeError(
            f"run_model expects bias on NPU, got device={bias.device!s}")
    if x.ndim != 2:
        raise ValueError(
            f"run_model expects x to be 2D, got shape={tuple(x.shape)}")
    if weight.ndim != 2:
        raise ValueError(
            f"run_model expects weight to be 2D, got shape={tuple(weight.shape)}"
        )
    if bias.ndim != 1:
        raise ValueError(
            f"run_model expects bias to be 1D, got shape={tuple(bias.shape)}")
    if x.shape[1] != weight.shape[1]:
        raise ValueError(
            f"Input feature mismatch: x.shape[1]={x.shape[1]} vs weight.shape[1]={weight.shape[1]}"
        )
    if weight.shape[0] != bias.shape[0]:
        raise ValueError(
            f"Output feature mismatch: weight.shape[0]={weight.shape[0]} vs bias.shape[0]={bias.shape[0]}"
        )

    model = ModelNew(
        in_features=x.shape[1],
        out_features=weight.shape[0],
        kernel_size=kernel_size,
        scale_factor=scale_factor,
    ).to(device="npu", dtype=weight.dtype)
    with torch.no_grad():
        model.matmul.weight.copy_(weight.contiguous())
        model.matmul.bias.copy_(bias.contiguous())
    return model(x)


batch_size = 128
in_features = 32768
out_features = 32768
kernel_size = 2
scale_factor = 0.5


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, kernel_size, scale_factor]
