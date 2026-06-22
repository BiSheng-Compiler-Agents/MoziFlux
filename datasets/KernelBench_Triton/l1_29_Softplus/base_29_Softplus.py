import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def _softplus_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    THRESHOLD: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    x32 = x.to(tl.float32)

    cond = x32 > THRESHOLD
    abs_x = tl.abs(x32)
    stable_term = tl.maximum(x32, 0.0) + tl.log(1.0 + tl.exp(-abs_x))
    y32 = tl.where(cond, x32, stable_term)

    y = y32.to(x.dtype)
    tl.store(y_ptr + offs, y, mask=mask)


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
        block_size = 4096

        def grid(meta):
            return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )

        _softplus_kernel[grid](
            x_contig.view(-1),
            y.view(-1),
            n_elements,
            THRESHOLD=20.0,
            BLOCK_SIZE=block_size,
            num_warps=8,
            num_stages=3,
        )
        return y


batch_size = 4096
dim = 393216


def get_inputs():
    x = torch.rand(batch_size, dim, device="npu")
    return [x]


def get_init_inputs():
    return []
