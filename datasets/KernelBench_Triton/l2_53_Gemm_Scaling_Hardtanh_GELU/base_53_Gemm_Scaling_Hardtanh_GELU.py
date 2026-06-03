import torch
import torch.nn as nn
import torch_npu  # noqa: F401

import triton
import triton.language as tl


@triton.jit
def _scale_hardtanh_gelu_kernel(
    x_ptr,
    y_ptr,
    numel,
    scale,
    minv,
    maxv,
    BLOCK_SIZE: tl.constexpr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * BLOCK_SIZE * BLOCKS_PER_PROGRAM
    lane = tl.arange(0, BLOCK_SIZE)
    inv_sqrt2 = 0.7071067811865476

    for block_idx in tl.static_range(BLOCKS_PER_PROGRAM):
        offs = base + block_idx * BLOCK_SIZE + lane
        mask = offs < numel

        tl.max_contiguous(offs, BLOCK_SIZE)
        tl.multiple_of(offs, 16)

        full_tile = (base + (block_idx + 1) * BLOCK_SIZE) <= numel
        if full_tile:
            x = tl.load(x_ptr + offs, cache_modifier=".cg")
        else:
            x = tl.load(x_ptr + offs, mask=mask, other=0.0, cache_modifier=".cg")
        xf = x.to(tl.float32) * scale
        xf = tl.minimum(tl.maximum(xf, minv), maxv)
        y32 = 0.5 * xf * (1.0 + tl.math.erf(xf * inv_sqrt2))
        y = y32.to(x.dtype)
        if full_tile:
            tl.store(y_ptr + offs, y)
        else:
            tl.store(y_ptr + offs, y, mask=mask)


class ModelNew(nn.Module):
    """
    Model that performs a GEMM, scaling, hardtanh, and GELU activation.
    Fuses scale + hardtanh + GELU into a single Triton kernel for speed.
    """
    def __init__(
        self,
        in_features=1024,
        out_features=512,
        scaling_factor=0.5,
        hardtanh_min=-2.0,
        hardtanh_max=2.0,
    ):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scaling_factor = float(scaling_factor)
        self.hardtanh = nn.Hardtanh(min_val=hardtanh_min, max_val=hardtanh_max)
        self.gelu = nn.GELU()  # kept for parity with original interface

    def _post_ops_triton(self, y: torch.Tensor) -> torch.Tensor:
        if y.device.type != "npu":
            raise RuntimeError("ModelNew expects GEMM outputs on Ascend NPU")
        # Ensure contiguous last-dim for coalesced access (Linear output is typically contiguous)
        if not y.is_contiguous():
            y = y.contiguous()

        flat = y.reshape(-1)
        numel = flat.numel()
        block_size = 1024
        blocks_per_program = 5
        grid = (triton.cdiv(numel, block_size * blocks_per_program),)
        _scale_hardtanh_gelu_kernel[grid](
            flat,
            flat,
            numel,
            self.scaling_factor,
            float(self.hardtanh.min_val),
            float(self.hardtanh.max_val),
            BLOCK_SIZE=block_size,
            BLOCKS_PER_PROGRAM=blocks_per_program,
            num_warps=8 if block_size >= 512 else 4,
            num_stages=1,
        )
        return y

    def forward(self, x):
        supported_dtypes = {torch.float16, torch.bfloat16, torch.float32}
        if x.device.type != "npu":
            raise RuntimeError("ModelNew expects inputs on Ascend NPU")
        if x.dtype not in supported_dtypes:
            raise RuntimeError(f"Unsupported dtype for ModelNew: {x.dtype}")
        if x.requires_grad:
            raise RuntimeError("ModelNew does not support autograd-tracked inputs")

        y = self.gemm(x)
        return self._post_ops_triton(y)
batch_size = 2048
in_features = 8192
out_features = 8192
scaling_factor = 0.5
hardtanh_min = -2
hardtanh_max = 2

def get_inputs():
    return [torch.rand(batch_size, in_features)]
def get_init_inputs():
    return [in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max]
