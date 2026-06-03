import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _gelu_groupnorm_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    z_ptr,
    y_ptr,
    N, C, H, W,
    G,
    eps,
    STATS_BLOCK: tl.constexpr,
    NORM_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G

    HW = H * W
    cpg = C // G
    group_elems = cpg * HW

    c_start = g * cpg
    n_base = n * C * H * W
    group_base = n_base + c_start * H * W

    stats_offs = tl.arange(0, STATS_BLOCK)
    norm_offs = tl.arange(0, NORM_BLOCK)
    inv_sqrt2 = 0.7071067811865476

    x_group_ptr = x_ptr + group_base
    z_group_ptr = z_ptr + group_base
    y_group_ptr = y_ptr + group_base
    acc1 = tl.zeros([STATS_BLOCK], dtype=tl.float32)
    acc2 = tl.zeros([STATS_BLOCK], dtype=tl.float32)

    idx = 0
    while idx < group_elems:
        i = idx + stats_offs
        i = tl.max_contiguous(tl.multiple_of(i, 16), 16)
        i = i
        i = i
        i = i
        i = i
        mask = i < group_elems
        x = tl.load(x_group_ptr + i, mask=mask, other=0.0).to(tl.float32)
        z = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))
        z = tl.where(mask, z, 0.0)
        acc1 += z
        acc2 += z * z
        tl.store(z_group_ptr + i, z, mask=mask)
        idx += STATS_BLOCK

    s1 = tl.sum(acc1, axis=0)
    s2 = tl.sum(acc2, axis=0)

    ge_f = tl.full((), group_elems, dtype=tl.float32)
    mean = s1 / ge_f
    var = s2 / ge_f - mean * mean
    rstd = tl.rsqrt(var + eps)

    ch = 0
    while ch < cpg:
        gamma0 = tl.load(w_ptr + c_start + ch).to(tl.float32)
        beta0 = tl.load(b_ptr + c_start + ch).to(tl.float32)
        gamma1 = tl.load(w_ptr + c_start + ch + 1).to(tl.float32)
        beta1 = tl.load(b_ptr + c_start + ch + 1).to(tl.float32)
        gamma2 = tl.load(w_ptr + c_start + ch + 2).to(tl.float32)
        beta2 = tl.load(b_ptr + c_start + ch + 2).to(tl.float32)
        gamma3 = tl.load(w_ptr + c_start + ch + 3).to(tl.float32)
        beta3 = tl.load(b_ptr + c_start + ch + 3).to(tl.float32)

        ch0_base = ch * HW
        ch1_base = (ch + 1) * HW
        ch2_base = (ch + 2) * HW
        ch3_base = (ch + 3) * HW

        off = 0
        while off < HW:
            idx_hw = off + norm_offs
            idx_hw = tl.max_contiguous(tl.multiple_of(idx_hw, 16), 16)
            idx_hw = idx_hw
            idx_hw = idx_hw
            idx_hw = idx_hw
            idx_hw = idx_hw
            mask_hw = idx_hw < HW

            z0 = tl.load(z_group_ptr + ch0_base + idx_hw, mask=mask_hw, other=0.0).to(tl.float32)
            y0 = (z0 - mean) * rstd
            y0 = y0 * gamma0 + beta0
            tl.store(y_group_ptr + ch0_base + idx_hw, y0, mask=mask_hw)

            z1 = tl.load(z_group_ptr + ch1_base + idx_hw, mask=mask_hw, other=0.0).to(tl.float32)
            y1 = (z1 - mean) * rstd
            y1 = y1 * gamma1 + beta1
            tl.store(y_group_ptr + ch1_base + idx_hw, y1, mask=mask_hw)

            z2 = tl.load(z_group_ptr + ch2_base + idx_hw, mask=mask_hw, other=0.0).to(tl.float32)
            y2 = (z2 - mean) * rstd
            y2 = y2 * gamma2 + beta2
            tl.store(y_group_ptr + ch2_base + idx_hw, y2, mask=mask_hw)

            z3 = tl.load(z_group_ptr + ch3_base + idx_hw, mask=mask_hw, other=0.0).to(tl.float32)
            y3 = (z3 - mean) * rstd
            y3 = y3 * gamma3 + beta3
            tl.store(y_group_ptr + ch3_base + idx_hw, y3, mask=mask_hw)

            off += NORM_BLOCK
        ch += 4


def gelu_groupnorm_fused(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, num_groups: int, eps: float):
    assert x.device.type == "npu", "Triton kernel requires an NPU tensor"
    assert x.dtype in (torch.float16, torch.bfloat16, torch.float32), "Unsupported input dtype"
    N, C, H, W = x.shape
    assert C % num_groups == 0, "num_groups must divide C"
    z = torch.empty((N, C, H, W), device=x.device, dtype=torch.float32)
    y = torch.empty_like(x)

    assert weight.device == x.device and bias.device == x.device, "Affine parameters must be on the same NPU device"
    w = weight.contiguous()
    b = bias.contiguous()

    grid = (N * num_groups,)
    STATS_BLOCK = 4224
    NORM_BLOCK = 4608
    _gelu_groupnorm_kernel[grid](
        x, w, b, z, y,
        N, C, H, W,
        num_groups,
        eps,
        STATS_BLOCK=STATS_BLOCK,
        NORM_BLOCK=NORM_BLOCK,
    )
    return y


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

    def forward(self, x):
        if x.device.type != "npu":
            raise RuntimeError("ModelNew expects NPU inputs")
        x = self.conv_transpose(x)
        return gelu_groupnorm_fused(
            x,
            self.group_norm.weight,
            self.group_norm.bias,
            self.group_norm.num_groups,
            self.group_norm.eps,
        )


batch_size = 128
in_channels = 64
out_channels = 64
height = width = 256
kernel_size = 3
stride = 1
groups = 8
num_groups = 8


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device="npu")]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, groups, num_groups]


_MODEL_CACHE = {}


def _resolve_dtype_model(dtype: torch.dtype):
    key = str(dtype)
    model = _MODEL_CACHE.get(key)
    if model is None:
        model = ModelNew(*get_init_inputs()).to("npu").eval()
        if dtype in (torch.float16, torch.bfloat16):
            model = model.to(dtype=dtype)
        _MODEL_CACHE[key] = model
    return model


@torch.no_grad()
def run_operator(x: torch.Tensor):
    x_npu = x.contiguous()
    if x_npu.device.type != "npu":
        x_npu = x_npu.to("npu")
    model = _resolve_dtype_model(x_npu.dtype)
    return model(x_npu)
