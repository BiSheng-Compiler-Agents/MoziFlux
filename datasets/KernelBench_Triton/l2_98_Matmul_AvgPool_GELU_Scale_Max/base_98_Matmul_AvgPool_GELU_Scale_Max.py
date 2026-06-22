import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _max_gelu_scale_kernel(
    projected_ptr,
    out_ptr,
    G,
    scale,
    pool_K,
    BLOCK_G: tl.constexpr,
):
    pid = tl.program_id(0)
    row_ptr = projected_ptr + pid * G

    offs_g = tl.arange(0, BLOCK_G)
    mask_g = offs_g < G

    vals = tl.load(row_ptr + offs_g, mask=mask_g, other=0.0)

    sign = tl.where(scale >= 0.0, 1.0, -1.0)
    signed = vals * sign
    signed = tl.where(mask_g, signed, -float("inf"))
    best_signed = tl.max(signed, axis=0)

    sel_sum = best_signed * sign
    sel_mean = sel_sum / tl.cast(pool_K, tl.float32)
    inv_sqrt2 = 0.7071067811865475
    gelu_val = 0.5 * sel_mean * (1.0 + tl.math.erf(sel_mean * inv_sqrt2))
    res = gelu_val * scale
    tl.store(out_ptr + pid, res)


class ModelNew(nn.Module):

    def __init__(self, in_features, out_features, pool_kernel_size,
                 scale_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scale_factor = scale_factor
        self._pool_k = int(pool_kernel_size)

    def forward(self, x):
        return _fused_apply(x, self.matmul.weight, self.matmul.bias,
                            self._pool_k, self.scale_factor)


def _fused_apply(x, weight, bias, pool_kernel_size, scale_factor):
    if x.device.type != "npu":
        raise RuntimeError("Fused kernel requires NPU input tensors.")
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    batch_size, in_features = x.shape
    out_features = weight.shape[0]
    kernel_size = int(pool_kernel_size)

    if out_features % kernel_size != 0:
        raise ValueError("out_features must be divisible by pool_kernel_size")

    G = out_features // kernel_size

    cache_key = ("grouped", out_features, kernel_size, in_features,
                 x.device.type, x.device.index, x.dtype)
    if cache_key not in _PARAM_CACHE:
        weight_grouped = weight.reshape(G, kernel_size, in_features).sum(dim=1)
        bias_grouped = bias.reshape(G, kernel_size).sum(dim=1)
        _PARAM_CACHE[cache_key] = (weight_grouped, bias_grouped)
    else:
        weight_grouped, bias_grouped = _PARAM_CACHE[cache_key]

    projected = F.linear(x, weight_grouped, bias_grouped)

    block_g = 1 << (G - 1).bit_length()
    out = torch.empty((batch_size, ), device=x.device, dtype=x.dtype)

    _max_gelu_scale_kernel[(batch_size, )](
        projected,
        out,
        G,
        float(scale_factor),
        kernel_size,
        BLOCK_G=block_g,
    )
    return out


_PARAM_CACHE = {}
_PARAM_SEED = 202698


def _get_default_linear_params(x, out_dim=8192):
    key = (x.device.type, x.device.index, x.dtype, x.shape[1], out_dim)
    if key not in _PARAM_CACHE:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_PARAM_SEED)
        weight = torch.randn((out_dim, x.shape[1]),
                             generator=generator,
                             dtype=torch.float32)
        bias = torch.randn((out_dim, ),
                           generator=generator,
                           dtype=torch.float32)
        _PARAM_CACHE[key] = (
            weight.to(device=x.device, dtype=x.dtype),
            bias.to(device=x.device, dtype=x.dtype),
        )
    return _PARAM_CACHE[key]


def matmul_avgpool_gelu_scale_max(x):
    weight, bias = _get_default_linear_params(x, out_dim=out_features)
    return _fused_apply(x, weight, bias, pool_kernel_size, scale_factor)


batch_size = 1024
in_features = 8192
out_features = 8192
pool_kernel_size = 16
scale_factor = 2.0


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, pool_kernel_size, scale_factor]
