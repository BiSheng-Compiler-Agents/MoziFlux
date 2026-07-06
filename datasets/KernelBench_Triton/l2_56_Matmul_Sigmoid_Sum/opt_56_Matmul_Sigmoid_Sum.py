import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.runtime import driver


_MAX_GRID = 65535
_BLOCK_M = 16
_BLOCK_H = 64
_BLOCK_K = 64
_REDUCE_BLOCK_H = 1024


def _require_npu_tensor(name: str, tensor: torch.Tensor) -> None:
    if tensor.device.type != "npu":
        raise AssertionError(f"{name} must be an NPU tensor")


def _device_prop(name: str, default: int) -> int:
    try:
        dev = torch.npu.current_device()
        return int(driver.active.utils.get_device_properties(dev)[name])
    except Exception:
        try:
            return int(driver.active.utils.get_device_properties("npu")[name])
        except Exception:
            return default


@triton.jit
def _matmul_logits_kernel(
    x_ptr,
    w_ptr,
    logits_ptr,
    B: tl.constexpr,
    I: tl.constexpr,
    H: tl.constexpr,
    stride_xb,
    stride_xi,
    stride_wh,
    stride_wi,
    NUM_M_TILES: tl.constexpr,
    NUM_H_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    total_tiles: tl.constexpr = NUM_M_TILES * NUM_H_TILES

    offs_m = tl.arange(0, BLOCK_M)
    offs_h = tl.arange(0, BLOCK_H)
    offs_k = tl.arange(0, BLOCK_K)

    for tile_id in tl.range(pid, total_tiles, nprog):
        m_tile = tile_id // NUM_H_TILES
        h_tile = tile_id - m_tile * NUM_H_TILES
        rows = m_tile * BLOCK_M + offs_m
        hs = h_tile * BLOCK_H + offs_h
        row_mask = rows < B
        h_mask = hs < H

        acc = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)
        for k0 in tl.range(0, I, BLOCK_K):
            ks = k0 + offs_k
            k_mask = ks < I
            x = tl.load(
                x_ptr + rows[:, None] * stride_xb + ks[None, :] * stride_xi,
                mask=row_mask[:, None] & k_mask[None, :],
                other=0.0,
                care_padding=False,
            )
            w = tl.load(
                w_ptr + hs[None, :] * stride_wh + ks[:, None] * stride_wi,
                mask=k_mask[:, None] & h_mask[None, :],
                other=0.0,
                care_padding=False,
            )
            acc = tl.dot(x, w, acc)

        tl.store(
            logits_ptr + rows[:, None] * H + hs[None, :],
            acc,
            mask=row_mask[:, None] & h_mask[None, :],
        )


@triton.jit
def _sigmoid_sum_kernel(
    logits_ptr,
    bias_ptr,
    out_ptr,
    B: tl.constexpr,
    H: tl.constexpr,
    stride_bo,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    total = tl.zeros((BLOCK_H,), dtype=tl.float32)
    start = 0
    while start < H:
        hs = start + offs
        mask = hs < H
        z = tl.load(logits_ptr + row * H + hs, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(bias_ptr + hs * stride_bo, mask=mask, other=0.0).to(tl.float32)
        s = 1.0 / (1.0 + tl.exp(-(z + b)))
        total += tl.where(mask, s, 0.0)
        start += BLOCK_H
    tl.store(out_ptr + row, tl.sum(total, axis=0), mask=row < B)


def matmul_sigmoid_sum(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    _require_npu_tensor("x", x)
    _require_npu_tensor("weight", weight)
    _require_npu_tensor("bias", bias)
    if x.dim() != 2:
        raise ValueError(f"x must be 2D, got shape {tuple(x.shape)}")
    if weight.dim() != 2:
        raise ValueError(f"weight must be 2D, got shape {tuple(weight.shape)}")
    if bias.dim() != 1:
        raise ValueError(f"bias must be 1D, got shape {tuple(bias.shape)}")

    B, In = x.shape
    H, weight_k = weight.shape
    if weight_k != In:
        raise ValueError(f"weight second dimension must match x second dimension, got {weight_k} and {In}")
    if bias.shape[0] != H:
        raise ValueError(f"bias length must match weight first dimension, got {bias.shape[0]} and {H}")
    if B == 0:
        return torch.empty((0, 1), device=x.device, dtype=torch.float32)

    x_in = x.contiguous()
    weight_in = weight.contiguous()
    bias_in = bias.contiguous()
    logits = torch.empty((B, H), device=x.device, dtype=torch.float32)
    out = torch.empty((B, 1), device=x.device, dtype=torch.float32)

    num_m_tiles = triton.cdiv(B, _BLOCK_M)
    num_h_tiles = triton.cdiv(H, _BLOCK_H)
    total_tiles = num_m_tiles * num_h_tiles

    stride_xb, stride_xi = x_in.stride()
    stride_wh, stride_wi = weight_in.stride()
    stride_bo = bias_in.stride(0)

    num_aicore = _device_prop("num_aicore", 20)
    grid_dot = (max(1, min(num_aicore, _MAX_GRID, total_tiles)),)
    _matmul_logits_kernel[grid_dot](
        x_in, weight_in, logits,
        B, In, H,
        stride_xb, stride_xi, stride_wh, stride_wi,
        num_m_tiles, num_h_tiles,
        BLOCK_M=_BLOCK_M, BLOCK_H=_BLOCK_H, BLOCK_K=_BLOCK_K,
        num_stages=2,
    )

    _sigmoid_sum_kernel[(B,)](
        logits, bias_in, out, B, H, stride_bo,
        BLOCK_H=_REDUCE_BLOCK_H,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    """Linear(x), sigmoid, then hidden sum with Cube matmul plus fused row reduction."""

    def __init__(self, input_size, hidden_size):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        return matmul_sigmoid_sum(x, self.linear.weight, self.linear.bias)


batch_size = 128
input_size = 32768
hidden_size = 32768


def get_inputs():
    return [torch.rand(batch_size, input_size)]


def get_init_inputs():
    return [input_size, hidden_size]
