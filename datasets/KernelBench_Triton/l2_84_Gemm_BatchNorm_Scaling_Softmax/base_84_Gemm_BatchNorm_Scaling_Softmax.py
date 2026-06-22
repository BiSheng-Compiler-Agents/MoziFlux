import torch
import torch.nn as nn
import triton
import triton.language as tl

BLOCK_N = 2048
MAX_TILES = 4
BLOCK_M = 4


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
    ],
    key=["N", "HAS_VECTOR_SCALE"],
)
@triton.jit
def _row_max_kernel(
    x_ptr,
    s_ptr,
    max_ptr,
    stride_xm,
    stride_xn,
    N,
    B,
    HAS_VECTOR_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_TILES: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < B
    offs = tl.arange(0, BLOCK_SIZE)
    row_ptrs = x_ptr + rows[:, None] * stride_xm + offs[None, :] * stride_xn
    row_max = tl.full((BLOCK_M, ), -float("inf"), tl.float32)

    if not HAS_VECTOR_SCALE:
        scalar_scale = tl.load(s_ptr).to(tl.float32)

    for tile_idx in tl.static_range(NUM_TILES):
        col_offsets = tile_idx * BLOCK_SIZE + offs
        mask = row_mask[:, None] & (col_offsets[None, :] < N)
        x = tl.load(
            row_ptrs + tile_idx * BLOCK_SIZE * stride_xn,
            mask=mask,
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)
        if HAS_VECTOR_SCALE:
            scale = tl.load(
                s_ptr + col_offsets,
                mask=col_offsets < N,
                other=1.0,
                cache_modifier=".cg",
            ).to(tl.float32)[None, :]
            z = x * scale
        else:
            z = x * scalar_scale
        z = tl.where(mask, z, -float("inf"))
        row_max = tl.maximum(row_max, tl.max(z, axis=1))

    tl.store(max_ptr + rows, row_max, mask=row_mask)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
    ],
    key=["N", "HAS_VECTOR_SCALE"],
)
@triton.jit
def _row_exp_sum_store_kernel(
    x_ptr,
    s_ptr,
    max_ptr,
    out_ptr,
    stride_xm,
    stride_xn,
    stride_om,
    stride_on,
    N,
    B,
    HAS_VECTOR_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_TILES: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < B
    offs = tl.arange(0, BLOCK_SIZE)
    row_x_ptrs = x_ptr + rows[:, None] * stride_xm + offs[None, :] * stride_xn
    row_o_ptrs = out_ptr + rows[:,
                                None] * stride_om + offs[None, :] * stride_on
    row_max = tl.load(max_ptr + rows, mask=row_mask, other=-float("inf"))
    row_sum = tl.zeros((BLOCK_M, ), dtype=tl.float32)

    if not HAS_VECTOR_SCALE:
        scalar_scale = tl.load(s_ptr).to(tl.float32)

    for tile_idx in tl.static_range(NUM_TILES):
        col_offsets = tile_idx * BLOCK_SIZE + offs
        mask = row_mask[:, None] & (col_offsets[None, :] < N)
        x_raw = tl.load(
            row_x_ptrs + tile_idx * BLOCK_SIZE * stride_xn,
            mask=mask,
            other=0.0,
            cache_modifier=".cg",
        )
        x = x_raw.to(tl.float32)
        if HAS_VECTOR_SCALE:
            scale = tl.load(
                s_ptr + col_offsets,
                mask=col_offsets < N,
                other=1.0,
                cache_modifier=".cg",
            ).to(tl.float32)[None, :]
            z = x * scale
        else:
            z = x * scalar_scale
        z = tl.where(mask, z, -float("inf"))
        exp_z = tl.exp(z - row_max[:, None])
        row_sum += tl.sum(exp_z, axis=1)
    inv_sum = 1.0 / row_sum

    for tile_idx in tl.static_range(NUM_TILES):
        col_offsets = tile_idx * BLOCK_SIZE + offs
        mask = row_mask[:, None] & (col_offsets[None, :] < N)
        x_raw = tl.load(
            row_x_ptrs + tile_idx * BLOCK_SIZE * stride_xn,
            mask=mask,
            other=0.0,
            cache_modifier=".cg",
        )
        x = x_raw.to(tl.float32)
        if HAS_VECTOR_SCALE:
            scale = tl.load(
                s_ptr + col_offsets,
                mask=col_offsets < N,
                other=1.0,
                cache_modifier=".cg",
            ).to(tl.float32)[None, :]
            z = x * scale
        else:
            z = x * scalar_scale
        z = tl.where(mask, z, -float("inf"))
        exp_z = tl.exp(z - row_max[:, None])
        rounded_exp = exp_z.to(x_raw.dtype)
        out = rounded_exp.to(tl.float32) * inv_sum[:, None]
        tl.store(row_o_ptrs + tile_idx * BLOCK_SIZE * stride_on,
                 out,
                 mask=mask)


def _fused_scale_softmax(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if x.device.type != "npu":
        raise RuntimeError("_fused_scale_softmax expects inputs on Ascend NPU")
    if x.ndim != 2:
        raise RuntimeError(
            f"_fused_scale_softmax expects a 2D tensor, got shape {tuple(x.shape)}"
        )
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise RuntimeError(
            f"Unsupported dtype for _fused_scale_softmax: {x.dtype}")

    batch, cols = x.shape
    if cols > BLOCK_N * MAX_TILES:
        raise RuntimeError(
            f"_fused_scale_softmax supports up to {BLOCK_N * MAX_TILES} columns, got {cols}"
        )

    x_contig = x.contiguous()
    out = torch.empty_like(x_contig)
    row_max = torch.empty((batch, ), device=x.device, dtype=torch.float32)
    if scale.numel() == 1:
        has_vector_scale = 0
        scale_buf = scale.to(device=x.device, dtype=x.dtype).contiguous()
    elif scale.numel() == cols:
        has_vector_scale = 1
        scale_buf = scale.reshape(cols).to(device=x.device,
                                           dtype=x.dtype).contiguous()
    else:
        raise RuntimeError(
            f"_fused_scale_softmax only supports scalar or length-{cols} scale tensors, got {tuple(scale.shape)}"
        )

    grid = (triton.cdiv(batch, BLOCK_M), )
    _row_max_kernel[grid](
        x_contig,
        scale_buf,
        row_max,
        x_contig.stride(0),
        x_contig.stride(1),
        cols,
        batch,
        HAS_VECTOR_SCALE=has_vector_scale,
        BLOCK_M=BLOCK_M,
        BLOCK_SIZE=BLOCK_N,
        NUM_TILES=MAX_TILES,
    )
    _row_exp_sum_store_kernel[grid](
        x_contig,
        scale_buf,
        row_max,
        out,
        x_contig.stride(0),
        x_contig.stride(1),
        out.stride(0),
        out.stride(1),
        cols,
        batch,
        HAS_VECTOR_SCALE=has_vector_scale,
        BLOCK_M=BLOCK_M,
        BLOCK_SIZE=BLOCK_N,
        NUM_TILES=MAX_TILES,
    )
    return out


class ModelNew(nn.Module):

    def __init__(
            self,
            in_features=1024,
            out_features=512,
            bn_eps=1e-5,
            bn_momentum=0.1,
            scale_shape=(1, ),
            device="npu",
            dtype=torch.float32,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gemm = nn.Linear(in_features,
                              out_features,
                              device=device,
                              dtype=dtype)
        self.bn = nn.BatchNorm1d(out_features,
                                 eps=bn_eps,
                                 momentum=bn_momentum,
                                 device=device,
                                 dtype=dtype)
        self.scale = nn.Parameter(
            torch.ones(scale_shape, device=device, dtype=dtype))

    def forward(self, x):
        if x.device.type != "npu":
            raise RuntimeError("ModelNew expects inputs on Ascend NPU")
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise RuntimeError(
                f"ModelNew expects shape [batch, {self.in_features}], got {tuple(x.shape)}"
            )
        if x.requires_grad:
            raise RuntimeError(
                "ModelNew does not support autograd-enabled inputs")

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
scale_shape = (1, )


def get_inputs():
    return [torch.rand(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, bn_eps, bn_momentum, scale_shape]
