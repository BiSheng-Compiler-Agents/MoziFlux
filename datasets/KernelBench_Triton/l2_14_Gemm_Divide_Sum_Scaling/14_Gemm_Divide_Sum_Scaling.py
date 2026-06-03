import torch
import torch.nn as nn
import triton
import triton.language as tl


DEFAULT_INPUT_SIZE = 8192
DEFAULT_HIDDEN_SIZE = 8192
DEFAULT_SCALING_FACTOR = 1.5


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False) or x.device.type == "npu")


@triton.jit
def _rowwise_dot_kernel(
    x_ptr,           # *f32/*f16, shape [M, K]
    s_ptr,           # *f32/*f16, shape [K]
    out_ptr,         # *f32/*f16, shape [M, 1] (we write column 0)
    M: tl.constexpr, # int
    K: tl.constexpr, # int
    stride_xm,       # int: stride for dim-0 of x in elements
    stride_xk,       # int: stride for dim-1 of x in elements
    stride_outm,     # int: stride for dim-0 of out in elements
    scale,           # f32 scalar (apply once at the end)
    BLOCK_M: tl.constexpr,  # tile size along M
    BLOCK_K: tl.constexpr,  # tile size along K
):
    pid_m = tl.program_id(axis=0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    # Accumulator for each row in the block
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    k0 = 0
    while k0 < K:
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # Load X tile [BLOCK_M, BLOCK_K]
        x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
        x = tl.load(
            x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0, cache_modifier=".cg"
        ).to(tl.float32)

        # Load S tile [BLOCK_K]
        s = tl.load(s_ptr + offs_k, mask=mask_k, other=0.0, cache_modifier=".cg").to(tl.float32)

        # Accumulate row-wise dot products
        acc += tl.sum(x * s[None, :], axis=1)
        k0 += BLOCK_K

    # Apply the final scale once
    acc = acc * scale

    # Store result directly
    tl.store(out_ptr + offs_m * stride_outm, acc, mask=mask_m)


class ModelNew(nn.Module):
    """
    Model that performs a matrix multiplication, division, summation, and scaling.
    Equivalent fused form:
      y = scaling_factor * sum((x @ W^T) / 2, dim=1, keepdim=True)
        = (scaling_factor / 2) * (x @ sum(W, dim=0))
      Output shape: (batch_size, 1)
    """
    def __init__(
        self,
        input_size: int = DEFAULT_INPUT_SIZE,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        scaling_factor: float = DEFAULT_SCALING_FACTOR,
    ):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        if not _is_npu_tensor(x):
            raise RuntimeError("ModelNew expects Ascend NPU inputs.")
        if not _is_npu_tensor(self.weight):
            raise RuntimeError("ModelNew weights must be moved to Ascend NPU before execution.")
        if torch.is_grad_enabled() or x.requires_grad:
            raise RuntimeError("ModelNew only supports inference execution on the Triton kernel path.")

        x = x.contiguous()
        M, K = x.shape

        # Compute s = sum(weight, dim=0) and fuse host-side scaling to reduce per-tile work
        s_eff = (self.weight.sum(dim=0) * (float(self.scaling_factor) * 0.5)).contiguous()

        out = torch.empty((M, 1), device=x.device, dtype=x.dtype)

        # Fixed tiling reduces Python overhead and performs well for small K on H200
        BLOCK_M = 128
        BLOCK_K = 128
        grid = ((M + BLOCK_M - 1) // BLOCK_M,)

        # scale is already fused into s_eff; pass 1.0 here
        _rowwise_dot_kernel[grid](
            x, s_eff, out,
            M, K,
            x.stride(0), x.stride(1),
            out.stride(0),
            1.0,
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )
        return out
batch_size   = 1024  
input_size   = 8192  
hidden_size  = 8192 
scaling_factor = 1.5

def get_inputs():
    return [torch.rand(batch_size, input_size)]
def get_init_inputs():
    return [input_size, hidden_size, scaling_factor]