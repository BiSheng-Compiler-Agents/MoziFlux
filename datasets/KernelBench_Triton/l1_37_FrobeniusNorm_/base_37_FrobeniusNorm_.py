import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def _sumsq_kernel(
    x_ptr,
    n_elements,
    n_blocks,
    program_stride,
    out_ptr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_idx = pid
    acc = 0.0
    while block_idx < n_blocks:
        block_start = block_idx * BLOCK
        offsets = block_start + tl.arange(0, BLOCK)
        offsets = tl.max_contiguous(offsets, BLOCK)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(x * x, axis=0)
        block_idx += program_stride
    tl.atomic_add(out_ptr, acc)


@triton.jit
def _reduce_partials_kernel(partials_ptr, n_partials, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    acc = 0.0
    idx = 0
    while idx < n_partials:
        offsets = idx + offs
        mask = offsets < n_partials
        vals = tl.load(partials_ptr + offsets, mask=mask, other=0.0)
        acc += tl.sum(vals, axis=0)
        idx += BLOCK
    tl.store(out_ptr, acc)


@triton.jit
def _scale_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    n_blocks,
    program_stride,
    sumsq_ptr,
    BLOCK: tl.constexpr,
):
    ss = tl.load(sumsq_ptr)
    inv_norm = tl.rsqrt(ss)
    pid = tl.program_id(axis=0)
    block_idx = pid
    while block_idx < n_blocks:
        block_start = block_idx * BLOCK
        offsets = block_start + tl.arange(0, BLOCK)
        offsets = tl.max_contiguous(offsets, BLOCK)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = x.to(tl.float32) * inv_norm
        tl.store(y_ptr + offsets, y, mask=mask)
        block_idx += program_stride


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type != "npu":
            raise ValueError("ModelNew expects an Ascend NPU tensor")
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(
                f"ModelNew supports float16, bfloat16, and float32 inputs, got {x.dtype}"
            )
        if x.numel() == 0:
            raise ValueError("ModelNew does not support empty tensors")

        original_shape = x.shape
        x_contig = x if x.is_contiguous() else x.contiguous()
        n_elements = x_contig.numel()

        BLOCK = 16384
        n_blocks = triton.cdiv(n_elements, BLOCK)
        sum_programs = min(n_blocks, 4096)
        scale_programs = min(n_blocks, 32768)

        sumsq = torch.zeros(1, device=x_contig.device, dtype=torch.float32)
        _sumsq_kernel[(sum_programs,)](
            x_contig,
            n_elements,
            n_blocks,
            sum_programs,
            sumsq,
            BLOCK=BLOCK,
            num_warps=8,
            num_stages=4,
        )

        out_dtype = torch.promote_types(x_contig.dtype, torch.float32)
        y = torch.empty_like(x_contig, dtype=out_dtype)
        _scale_kernel[(scale_programs,)](
            x_contig,
            y,
            n_elements,
            n_blocks,
            scale_programs,
            sumsq,
            BLOCK=BLOCK,
            num_warps=8,
            num_stages=4,
        )

        return y.view(original_shape)


batch_size = 112
features = 64
dim1 = 512
dim2 = 512


def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2)
    return [x]


def get_init_inputs():
    return []
