import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl

# PyTorch SELU constants
_SELU_SCALE = 1.0507009873554805
_SELU_ALPHA = 1.6732632423543772


@triton.jit
def _selu_kernel(
    x_ptr,
    y_ptr,
    n_elements,  # keep runtime arg to avoid recompiles on same shape
    ALPHA: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Coalescing & scheduling hints
    tl.max_contiguous(offsets, BLOCK_SIZE)
    tl.multiple_of(offsets, 16)

    mask = offsets < n_elements
    # Streaming load hint: we don't reuse x, prefer evict-first
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0, eviction_policy='evict_first')
    x32 = x.to(tl.float32)

    # Precompute constants
    alpha_scale = SCALE * ALPHA

    # Branchless SELU:
    # pos = max(x, 0), neg = min(x, 0)
    # out = SCALE * pos + alpha_scale * (exp(neg) - 1)
    neg = tl.minimum(x32, 0.0)
    pos = tl.maximum(x32, 0.0)
    expm1_neg = tl.exp(neg) - 1.0
    out32 = pos * SCALE + expm1_neg * alpha_scale

    tl.store(y_ptr + offsets, out32.to(x.dtype), mask=mask)


class ModelNew(nn.Module):
    """
    Simple model that performs a SELU activation using a Triton kernel on Ascend NPU tensors.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies SELU activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with SELU applied, same shape as input.
        """
        supported_dtypes = {torch.float16, torch.bfloat16, torch.float32}
        if x.device.type != "npu":
            raise RuntimeError("ModelNew expects inputs on Ascend NPU")
        if x.dtype not in supported_dtypes:
            raise RuntimeError(f"Unsupported dtype for ModelNew: {x.dtype}")
        if x.requires_grad:
            raise RuntimeError("ModelNew does not support autograd-tracked inputs")

        # Handle empty tensors
        if x.numel() == 0:
            return x.contiguous()

        x_contig = x.contiguous()
        y = torch.empty_like(x_contig)

        n_elements = x_contig.numel()

        # Use a single tuned configuration to minimize launch overhead
        BLOCK_SIZE = 4096
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        _selu_kernel[grid](
            x_contig, y, n_elements,
            ALPHA=_SELU_ALPHA, SCALE=_SELU_SCALE, BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4, num_stages=2,
        )
        return y
batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]
def get_init_inputs():
    return []  # No special initialization inputs needed