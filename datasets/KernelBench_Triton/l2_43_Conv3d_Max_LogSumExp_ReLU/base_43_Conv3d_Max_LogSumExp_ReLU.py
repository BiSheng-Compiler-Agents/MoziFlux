# Round 7: BLOCK=64 on fast-math baseline
import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl


DEFAULT_IN_CHANNELS = 32
DEFAULT_OUT_CHANNELS = 64
DEFAULT_KERNEL_SIZE = 3
DEFAULT_STRIDE = 1
DEFAULT_PADDING = 1


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return bool(getattr(x, "is_npu", False))


@triton.jit
def _pool_lse_relu_stream_kernel(
    x_ptr, y_ptr, M, DO, HO, WO,
    sxn, sxc, sxd, sxh, sxw,
    syn, syc, syd, syh, syw,
    C: tl.constexpr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M
    ZYX = DO * HO * WO
    YX = HO * WO
    n = offs // ZYX
    rem = offs % ZYX
    do = rem // YX
    rem = rem % YX
    ho = rem // WO
    wo = rem % WO
    n64 = n.to(tl.int64)
    do64 = do.to(tl.int64)
    ho64 = ho.to(tl.int64)
    wo64 = wo.to(tl.int64)
    sxn = tl.full([], sxn, tl.int64)
    sxc = tl.full([], sxc, tl.int64)
    sxd = tl.full([], sxd, tl.int64)
    sxh = tl.full([], sxh, tl.int64)
    sxw = tl.full([], sxw, tl.int64)
    syn = tl.full([], syn, tl.int64)
    syc = tl.full([], syc, tl.int64)
    syd = tl.full([], syd, tl.int64)
    syh = tl.full([], syh, tl.int64)
    syw = tl.full([], syw, tl.int64)
    two = tl.full([], 2, tl.int64)
    di0 = do64 * two
    hi0 = ho64 * two
    wi0 = wo64 * two
    base_x = n64 * sxn + di0 * sxd + hi0 * sxh + wi0 * sxw
    base_y = n64 * syn + do64 * syd + ho64 * syh + wo64 * syw
    o0 = tl.full([], 0, tl.int64)
    o1 = sxw
    o2 = sxh
    o3 = sxh + sxw
    o4 = sxd
    o5 = sxd + sxw
    o6 = sxd + sxh
    o7 = sxd + sxh + sxw
    neg_inf = tl.full((BLOCK,), float("-inf"), tl.float32)
    m = neg_inf
    s = tl.zeros((BLOCK,), dtype=tl.float32)
    p0 = x_ptr + base_x
    for c in tl.static_range(0, C):
        pc = p0 + c * sxc
        v0 = tl.load(pc + o0, mask=mask, other=-float("inf"))
        v1 = tl.load(pc + o1, mask=mask, other=-float("inf"))
        v2 = tl.load(pc + o2, mask=mask, other=-float("inf"))
        v3 = tl.load(pc + o3, mask=mask, other=-float("inf"))
        v4 = tl.load(pc + o4, mask=mask, other=-float("inf"))
        v5 = tl.load(pc + o5, mask=mask, other=-float("inf"))
        v6 = tl.load(pc + o6, mask=mask, other=-float("inf"))
        v7 = tl.load(pc + o7, mask=mask, other=-float("inf"))
        vmax = tl.maximum(v0, v1)
        vmax = tl.maximum(vmax, v2)
        vmax = tl.maximum(vmax, v3)
        vmax = tl.maximum(vmax, v4)
        vmax = tl.maximum(vmax, v5)
        vmax = tl.maximum(vmax, v6)
        vmax = tl.maximum(vmax, v7)
        val = vmax.to(tl.float32)
        new_m = tl.maximum(m, val)
        s = tl.exp(m - new_m) * s + tl.exp(val - new_m)
        m = new_m
    out = tl.log(s) + m
    out = tl.maximum(out, 0.0)
    tl.store(y_ptr + base_y, out, mask=mask)


def _pool_lse_relu_triton(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 5
    if not _is_npu_tensor(x):
        raise ValueError("_pool_lse_relu_triton requires an Ascend NPU tensor input")
    pooled = torch.nn.functional.max_pool3d(x, kernel_size=2, stride=2)
    return _lse_relu_triton(pooled)


@triton.jit
def _lse_relu_reduce_c_kernel(
    x_ptr, out_ptr, M, Z, Y, X,
    stride_n, stride_c, stride_z, stride_y, stride_x,
    C: tl.constexpr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M
    ZYX = Z * Y * X
    YX = Y * X
    n = offs // ZYX
    rem = offs % ZYX
    z = rem // YX
    rem = rem % YX
    y = rem // X
    x = rem % X
    n64 = n.to(tl.int64)
    z64 = z.to(tl.int64)
    y64 = y.to(tl.int64)
    x64 = x.to(tl.int64)
    sN = tl.full([], stride_n, tl.int64)
    sC = tl.full([], stride_c, tl.int64)
    sZ = tl.full([], stride_z, tl.int64)
    sY = tl.full([], stride_y, tl.int64)
    sX = tl.full([], stride_x, tl.int64)
    base_in = n64 * sN + z64 * sZ + y64 * sY + x64 * sX
    ZYX64 = tl.full([], ZYX, tl.int64)
    YX64 = tl.full([], YX, tl.int64)
    X64 = tl.full([], X, tl.int64)
    base_out = n64 * ZYX64 + z64 * YX64 + y64 * X64 + x64
    neg_inf = tl.full((BLOCK,), float("-inf"), tl.float32)
    m = neg_inf
    s = tl.zeros((BLOCK,), dtype=tl.float32)

    p = x_ptr + base_in
    for c in tl.static_range(0, C):
        val = tl.load(p + c * sC, mask=mask, other=-float("inf"))
        new_m = tl.maximum(m, val, propagate_nan=tl.PropagateNan.ALL)
        s = tl.math.exp(m - new_m) * s + tl.math.exp(val - new_m)
        m = new_m

    out_f32 = tl.math.log(s) + m
    out_f32 = tl.maximum(out_f32, 0.0, propagate_nan=tl.PropagateNan.ALL)
    tl.store(out_ptr + base_out, out_f32, mask=mask)


def _lse_relu_triton(x: torch.Tensor) -> torch.Tensor:
    if not _is_npu_tensor(x):
        raise ValueError("_lse_relu_triton requires an Ascend NPU tensor")
    orig_dtype = x.dtype
    if x.dtype != torch.float32:
        x = x.float()
    if not x.is_contiguous():
        x = x.contiguous()
    N, C, Z, Y, X = x.shape
    out = torch.empty((N, 1, Z, Y, X), device=x.device, dtype=torch.float32)
    sN, sC, sZ, sY, sX = x.stride()
    M = N * Z * Y * X
    grid = lambda META: (triton.cdiv(M, META["BLOCK"]),)
    _lse_relu_reduce_c_kernel[grid](
        x, out, M, Z, Y, X,
        sN, sC, sZ, sY, sX,
        C=C, BLOCK=64,
    )
    if orig_dtype != torch.float32:
        out = out.to(orig_dtype)
    return out


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels=DEFAULT_IN_CHANNELS,
        out_channels=DEFAULT_OUT_CHANNELS,
        kernel_size=DEFAULT_KERNEL_SIZE,
        stride=DEFAULT_STRIDE,
        padding=DEFAULT_PADDING,
    ):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = self.conv(x)
        if not _is_npu_tensor(x):
            raise ValueError("ModelNew.forward requires Ascend NPU inputs and weights")
        return _pool_lse_relu_triton(x)


batch_size = 4
in_channels = 32
out_channels = 64
depth, height, width = 32, 128, 128
kernel_size = 3
stride = 1
padding = 1


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]
