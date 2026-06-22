import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None

_MAX_PROGRAMS = 65535


@triton.jit
def _rownorm_addmul_direct_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    B,
    F,
    stride_x,
    stride_y,
    stride_out,
    eps,
    inv_F,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    tl.multiple_of(offs, 16)
    tl.max_contiguous(offs, BLOCK)
    mask = (pid < B) & (offs < F)

    x_row = tl.load(x_ptr + pid * stride_x + offs, mask=mask, other=0.0)
    y_row = tl.load(y_ptr + pid * stride_y + offs, mask=mask, other=0.0)

    sum_x = tl.sum(x_row, axis=0)
    sum_x2 = tl.sum(x_row * x_row, axis=0)
    mean = sum_x * inv_F
    var = tl.maximum(sum_x2 * inv_F - mean * mean, 0.0)
    rstd = tl.rsqrt(var + eps)
    y_sq = y_row * y_row
    out_row = y_sq + y_row * (x_row - mean) * rstd
    tl.store(out_ptr + pid * stride_out + offs, out_row, mask=mask)


@triton.jit
def _rownorm_addmul_persistent_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    B,
    F,
    stride_x,
    stride_y,
    stride_out,
    eps,
    inv_F,
    n_programs,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    tl.multiple_of(offs, 16)
    tl.max_contiguous(offs, BLOCK)
    for row in range(pid, B, n_programs):
        mask = offs < F
        x_row = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=0.0)
        y_row = tl.load(y_ptr + row * stride_y + offs, mask=mask, other=0.0)
        sum_x = tl.sum(x_row, axis=0)
        sum_x2 = tl.sum(x_row * x_row, axis=0)
        mean = sum_x * inv_F
        var = tl.maximum(sum_x2 * inv_F - mean * mean, 0.0)
        rstd = tl.rsqrt(var + eps)
        y_sq = y_row * y_row
        out_row = y_sq + y_row * (x_row - mean) * rstd
        tl.store(out_ptr + row * stride_out + offs, out_row, mask=mask)


def _next_power_of_2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False) or x.device.type == "npu")


def _fused_linear_instance_norm_sum_residual_add_multiply(
    x: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must be 2D tensors")
    if weight.ndim != 2:
        raise ValueError("weight must be a 2D tensor")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have matching batch dimensions")
    if x.shape[1] != weight.shape[1]:
        raise ValueError(
            "x and weight must agree on the input feature dimension")
    if y.shape[1] != weight.shape[0]:
        raise ValueError(
            "y width must match the weight output feature dimension")
    if x.dtype != y.dtype or x.dtype != weight.dtype:
        raise TypeError("x, y, and weight must use the same dtype")
    if x.device != y.device or x.device != weight.device:
        raise ValueError("x, y, and weight must be on the same device")
    if bias is not None:
        if bias.ndim != 1 or bias.shape[0] != weight.shape[0]:
            raise ValueError(
                "bias must be a 1D tensor with length equal to weight.shape[0]"
            )
        if bias.dtype != x.dtype:
            raise TypeError("bias dtype must match the input dtype")
        if bias.device != x.device:
            raise ValueError("bias must be on the same device as x")
    if not _is_npu_tensor(x) or not _is_npu_tensor(y) or not _is_npu_tensor(
            weight):
        raise RuntimeError("The fused operator requires Ascend NPU tensors")

    x_c = x.contiguous()
    y_c = y.contiguous()
    weight_c = weight.contiguous()
    bias_c = None if bias is None else bias.contiguous()

    linear_out = F.linear(x_c, weight_c, bias_c)
    batch_size, features = linear_out.shape
    out = torch.empty_like(y_c)
    block = _next_power_of_2(features)
    inv_f = 1.0 / float(features)
    if batch_size > _MAX_PROGRAMS:
        n_programs = _MAX_PROGRAMS
        _rownorm_addmul_persistent_kernel[(n_programs, )](
            linear_out,
            y_c,
            out,
            batch_size,
            features,
            linear_out.stride(0),
            y_c.stride(0),
            out.stride(0),
            float(eps),
            inv_f,
            n_programs,
            BLOCK=block,
        )
    else:
        _rownorm_addmul_direct_kernel[(batch_size, )](
            linear_out,
            y_c,
            out,
            batch_size,
            features,
            linear_out.stride(0),
            y_c.stride(0),
            out.stride(0),
            float(eps),
            inv_f,
            BLOCK=block,
        )
    return out


class ModelNew(nn.Module):
    """Linear + per-row InstanceNorm + residual sum with y + multiply by y."""

    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.bmm = nn.Linear(in_features, out_features)
        self.instance_norm = nn.InstanceNorm2d(out_features,
                                               eps=eps,
                                               momentum=momentum)
        self.eps = float(eps)

    def forward(self, x, y):
        return _fused_linear_instance_norm_sum_residual_add_multiply(
            x, y, self.bmm.weight, self.bmm.bias, eps=self.eps)


_MODEL_CACHE = {}


def _seed_for_shape(in_features: int, out_features: int) -> int:
    return 1009 + in_features * 131 + out_features * 17


def _build_cached_model(in_features: int, out_features: int,
                        device: torch.device, dtype: torch.dtype, eps: float):
    key = (str(device), dtype, in_features, out_features, float(eps))
    model = _MODEL_CACHE.get(key)
    if model is None:
        model = ModelNew(in_features, out_features, eps=eps).to(device=device,
                                                                dtype=dtype)
        seed = _seed_for_shape(in_features, out_features)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            weight = torch.randn((out_features, in_features),
                                 generator=generator,
                                 dtype=torch.float32).to(device=device,
                                                         dtype=dtype)
            bias = torch.randn((out_features, ),
                               generator=generator,
                               dtype=torch.float32).to(device=device,
                                                       dtype=dtype)
            model.bmm.weight.copy_(weight)
            model.bmm.bias.copy_(bias)
        model.eval()
        _MODEL_CACHE[key] = model
    return model


def run_operator(x: torch.Tensor,
                 y: torch.Tensor,
                 eps: float = 1e-5) -> torch.Tensor:
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("run_operator expects 2D x and y tensors")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have matching batch dimensions")
    if x.dtype != y.dtype:
        raise TypeError("x and y must use the same dtype")
    if x.device != y.device:
        raise ValueError("x and y must be on the same device")
    if not _is_npu_tensor(x) or not _is_npu_tensor(y):
        raise RuntimeError("run_operator requires Ascend NPU tensors")
    model = _build_cached_model(x.shape[1], y.shape[1], x.device, x.dtype,
                                float(eps))
    return model(x, y)


batch_size = 1024
in_features = 8192
out_features = 8192


def get_inputs():
    return [
        torch.rand(batch_size, in_features),
        torch.rand(batch_size, out_features)
    ]


def get_init_inputs():
    return [in_features, out_features]
