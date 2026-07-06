import torch
import torch.nn as nn
import torch_npu  # noqa: F401

import triton
import triton.language as tl


_MAX_PROGRAMS = 65535
_BLOCK_SIZE = 4096


@triton.jit
def _scale_hardtanh_gelu_flat_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    scale,
    minv,
    maxv,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.max_contiguous(offsets, BLOCK_SIZE)
    tl.multiple_of(offsets, 16)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0, care_padding=False)
    xf = x.to(tl.float32) * scale
    xf = tl.minimum(tl.maximum(xf, minv), maxv)
    y32 = 0.5 * xf * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    tl.store(y_ptr + offsets, y32.to(x.dtype), mask=mask)


@triton.jit
def _scale_hardtanh_gelu_persistent_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    n_programs,
    scale,
    minv,
    maxv,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_tiles = tl.cdiv(n_elements, BLOCK_SIZE)
    for tile_id in range(pid, n_tiles, n_programs):
        offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        tl.max_contiguous(offsets, BLOCK_SIZE)
        tl.multiple_of(offsets, 16)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0, care_padding=False)
        xf = x.to(tl.float32) * scale
        xf = tl.minimum(tl.maximum(xf, minv), maxv)
        y32 = 0.5 * xf * (1.0 + tl.math.erf(xf * 0.7071067811865476))
        tl.store(y_ptr + offsets, y32.to(x.dtype), mask=mask)


class ModelNew(nn.Module):
    """GEMM followed by fused scale + Hardtanh + exact GELU epilogue."""

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
        self.gelu = nn.GELU()

    def _post_ops_triton(self, y: torch.Tensor) -> torch.Tensor:
        if y.device.type != "npu":
            raise RuntimeError("ModelNew expects GEMM outputs on Ascend NPU")
        if not y.is_contiguous():
            y = y.contiguous()
        n_elements = y.numel()
        n_tiles = triton.cdiv(n_elements, _BLOCK_SIZE)
        if n_tiles > _MAX_PROGRAMS:
            n_programs = _MAX_PROGRAMS
            _scale_hardtanh_gelu_persistent_kernel[(n_programs,)](
                y,
                y,
                n_elements,
                n_programs,
                self.scaling_factor,
                float(self.hardtanh.min_val),
                float(self.hardtanh.max_val),
                BLOCK_SIZE=_BLOCK_SIZE,
                num_stages=2,
            )
        else:
            _scale_hardtanh_gelu_flat_kernel[(n_tiles,)](
                y,
                y,
                n_elements,
                self.scaling_factor,
                float(self.hardtanh.min_val),
                float(self.hardtanh.max_val),
                BLOCK_SIZE=_BLOCK_SIZE,
                num_stages=2,
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
