import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _sigmoid_kernel_generic(x_ptr, y_ptr, n_elements, n_programs,
                            BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    stride = n_programs * BLOCK_SIZE
    block_start = pid * BLOCK_SIZE
    while block_start < n_elements:
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        x32 = x.to(tl.float32)
        y = tl.sigmoid(x32)
        out = y.to(x.dtype)
        tl.store(y_ptr + offsets, out, mask=mask)
        block_start += stride


class ModelNew(nn.Module):

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        supported_dtypes = {torch.float16, torch.bfloat16, torch.float32}
        if x.device.type != "npu":
            raise RuntimeError("ModelNew expects inputs on Ascend NPU")
        if x.dtype not in supported_dtypes:
            raise RuntimeError(f"Unsupported dtype for ModelNew: {x.dtype}")
        if x.requires_grad:
            raise RuntimeError(
                "ModelNew does not support autograd-tracked inputs")

        x_contig = x.contiguous()
        n_elements = x_contig.numel()
        if n_elements == 0:
            return x_contig

        y = torch.empty_like(x_contig)
        block_size = 8192
        max_programs = 32768
        n_programs = min(triton.cdiv(n_elements, block_size), max_programs)
        grid = (n_programs, )
        _sigmoid_kernel_generic[grid](
            x_contig,
            y,
            n_elements,
            n_programs,
            BLOCK_SIZE=block_size,
            num_warps=4,
            num_stages=2,
        )
        return y


batch_size = 4096
dim = 393216


def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]


def get_init_inputs():
    return []
