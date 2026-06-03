import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=16, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=16, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=3),
    ],
    key=["N", "HAS_VECTOR_SCALE"],
)
@triton.jit
def _scale_softmax_row_kernel(
    x_ptr,  # *[B, N]
    s_ptr,  # *[1] or *[N]
    out_ptr,  # *[B, N]
    stride_xm,
    stride_xn,
    stride_om,
    stride_on,
    N,  # number of columns
    HAS_VECTOR_SCALE: tl.
    constexpr,  # 0 for scalar scale, 1 for per-column scale
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Row pointers
    x_row_ptr = x_ptr + pid * stride_xm + offs * stride_xn
    o_row_ptr = out_ptr + pid * stride_om + offs * stride_on

    # Hints for better codegen/coalescing
    tl.max_contiguous(offs, BLOCK_SIZE)
    tl.multiple_of(offs, 16)

    # Load logits to fp32
    x = tl.load(x_row_ptr, mask=mask, other=0.0,
                cache_modifier=".cg").to(tl.float32)

    LOG2E = 1.4426950408889634  # for exp2

    if HAS_VECTOR_SCALE:
        # Per-column scale; masked lanes neutral
        s = tl.load(s_ptr + offs, mask=mask, other=1.0,
                    cache_modifier=".cg").to(tl.float32)
        z = x * s
        # Ensure masked lanes don't affect reductions
        z = tl.where(mask, z, -float("inf"))
        m = tl.max(z, axis=0)
        e = tl.exp2((z - m) * LOG2E)
        den = tl.sum(e, axis=0)
        out = e * (1.0 / den)
    else:
        # Scalar scale; sign-aware centering to avoid extra reduction work
        s = tl.load(s_ptr).to(tl.float32)
        # Compute reductions ignoring masked lanes robustly
        pos_inf = float("inf")
        neg_inf = -float("inf")
        if s >= 0:
            x_for_max = tl.where(mask, x, neg_inf)
            mx = tl.max(x_for_max, axis=0)
            z = (x - mx) * s
        else:
            x_for_min = tl.where(mask, x, pos_inf)
            mn = tl.min(x_for_min, axis=0)
            z = (x - mn) * s
        # Mask OOB elements to -inf so they don't affect softmax
        z = tl.where(mask, z, neg_inf)
        e = tl.exp2(z * LOG2E)
        den = tl.sum(e, axis=0)
        out = e * (1.0 / den)

    tl.store(o_row_ptr, out, mask=mask)


def _next_power_of_2(x: int) -> int:
    return 1 if x <= 1 else 1 << ((x - 1).bit_length())


def _fused_scale_softmax(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if x.device.type != "npu":
        raise RuntimeError("_fused_scale_softmax expects inputs on Ascend NPU")
    if x.ndim != 2:
        raise RuntimeError(f"_fused_scale_softmax expects a 2D tensor, got shape {tuple(x.shape)}")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise RuntimeError(f"Unsupported dtype for _fused_scale_softmax: {x.dtype}")

    B, N = x.shape
    if N > 1024:
        raise RuntimeError(f"_fused_scale_softmax supports up to 1024 columns, got {N}")

    x_contig = x.contiguous()
    out = torch.empty_like(x_contig)

    if scale.numel() == 1:
        has_vector_scale = 0
        s_buf = scale.to(device=x.device, dtype=x.dtype).contiguous()
    elif scale.numel() == N:
        has_vector_scale = 1
        s_buf = scale.view(N).to(device=x.device, dtype=x.dtype).contiguous()
    else:
        raise RuntimeError(
            f"_fused_scale_softmax only supports scalar or length-{N} scale tensors, got {tuple(scale.shape)}"
        )

    # Tile one full row per program
    BLOCK_SIZE = min(1024, _next_power_of_2(N))
    grid = (B,)

    _scale_softmax_row_kernel[grid](
        x_contig, s_buf, out,
        x_contig.stride(0), x_contig.stride(1),
        out.stride(0), out.stride(1),
        N,
        HAS_VECTOR_SCALE=has_vector_scale,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    """
    Model that performs a matrix multiplication (Gemm), Batch Normalization, scaling, and Softmax.
    """
    def __init__(
        self,
        in_features=1024,
        out_features=512,
        bn_eps=1e-5,
        bn_momentum=0.1,
        scale_shape=(1,),
        device="npu",
        dtype=torch.float32,
    ):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gemm = nn.Linear(in_features, out_features, device=device, dtype=dtype)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum, device=device, dtype=dtype)
        self.scale = nn.Parameter(torch.ones(scale_shape, device=device, dtype=dtype))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        if x.device.type != "npu":
            raise RuntimeError("ModelNew expects inputs on Ascend NPU")
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise RuntimeError(
                f"ModelNew expects shape [batch, {self.in_features}], got {tuple(x.shape)}"
            )
        if x.requires_grad:
            raise RuntimeError("ModelNew does not support autograd-enabled inputs")

        param = next(self.parameters())
        if param.device != x.device or param.dtype != x.dtype:
            self.to(device=x.device, dtype=x.dtype)

        x = self.gemm(x.contiguous())
        x = self.bn(x)
        return _fused_scale_softmax(x, self.scale)
batch_size = 1024
in_features = 8192
out_features = 8192
bn_eps = 1e-5
bn_momentum = 0.1
scale_shape = (1,)

def get_inputs():
    return [torch.rand(batch_size, in_features)]
def get_init_inputs():
    return [in_features, out_features, bn_eps, bn_momentum, scale_shape]