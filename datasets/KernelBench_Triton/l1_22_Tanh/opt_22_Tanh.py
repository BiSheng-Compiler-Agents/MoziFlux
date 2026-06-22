"""
opt_22_Tanh.py — Optimized Tanh kernel for Ascend NPU

Optimizations over baseline (22_Tanh.py):
  1. tl.math.tanh → single AIV vector instruction vs 5+ op manual exp approximation
  2. @triton.autotune over BLOCK_SIZE {256, 512, 1024, 2048, 4096}
  3. care_padding=False on all load/store (safe with explicit mask)
  4. Two-path dispatch: direct (no-loop) for n_elements <= 16,776,960,
     persistent while-loop for larger tensors
  5. Bucketed autotune key (n_elements_pow2) to avoid per-value cache explosion
"""

import torch
import triton
import triton.language as tl
from triton.language.math import tanh as tl_tanh

# ── Constants ────────────────────────────────────────────────────────────────

MIN_BLOCK = 256  # smallest autotune BLOCK_SIZE — determines safe threshold
MAX_PROGRAMS = 65535  # Ascend FFTS grid dimension limit
MAX_ELEMS = MAX_PROGRAMS * MIN_BLOCK  # 16,776,960 — safe for ALL autotune configs

# ── Direct kernel (no loop, single program per tile) ─────────────────────────


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}),
        triton.Config({"BLOCK_SIZE": 512}),
        triton.Config({"BLOCK_SIZE": 1024}),
        triton.Config({"BLOCK_SIZE": 2048}),
        triton.Config({"BLOCK_SIZE": 4096}),
    ],
    key=["n_elements_pow2"],
)
@triton.jit
def _tanh_direct_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    n_elements_pow2,  # bucketed key for autotune cache
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load with care_padding=False (safe: mask prevents OOB access)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0, care_padding=False)

    # FP32 upcast for precision
    x32 = x.to(tl.float32)

    # Single AIV vector instruction — tl.math.tanh maps to hardware tanh
    y32 = tl_tanh(x32)

    y = y32.to(x.dtype)
    tl.store(y_ptr + offsets, y, mask=mask)


# ── Persistent kernel (while-loop, amortizes FFTS dispatch for huge tensors) ─


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}),
        triton.Config({"BLOCK_SIZE": 512}),
        triton.Config({"BLOCK_SIZE": 1024}),
        triton.Config({"BLOCK_SIZE": 2048}),
        triton.Config({"BLOCK_SIZE": 4096}),
    ],
    key=["n_elements_pow2"],
)
@triton.jit
def _tanh_persistent_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    n_elements_pow2,  # bucketed key for autotune cache
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    tile_id = pid

    while tile_id * BLOCK_SIZE < n_elements:
        offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0, care_padding=False)
        x32 = x.to(tl.float32)
        y32 = tl_tanh(x32)
        y = y32.to(x.dtype)
        tl.store(y_ptr + offsets, y, mask=mask)

        tile_id += n_programs


# ── Dispatch logic ───────────────────────────────────────────────────────────


def _tanh_triton(x: torch.Tensor) -> torch.Tensor:
    """Apply tanh to a flat 1D tensor using the optimal dispatch path."""
    assert x.is_contiguous(), "Input must be contiguous"
    n = x.numel()
    y = torch.empty_like(x)

    # Compute bucketed key for autotune cache
    n_elements_pow2 = 1 << (n - 1).bit_length()

    # Route: persistent if cdiv(n, MIN_BLOCK) exceeds MAX_PROGRAMS
    n_tiles_at_min = triton.cdiv(n, MIN_BLOCK)
    if n_tiles_at_min > MAX_PROGRAMS:
        # Persistent path: grid capped at MAX_PROGRAMS
        grid = (min(n_tiles_at_min, MAX_PROGRAMS), )
        _tanh_persistent_kernel[grid](
            x,
            y,
            n,
            n_elements_pow2,
        )
    else:
        # Direct path: safe for all autotune configs
        grid = (n_tiles_at_min, )
        _tanh_direct_kernel[grid](
            x,
            y,
            n,
            n_elements_pow2,
        )

    return y


# ── Host interface (ModelNew) ────────────────────────────────────────────────


class ModelNew(torch.nn.Module):
    """Torch module wrapper for the optimized Tanh kernel.

    Provides a consistent interface matching the KernelBench convention.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _tanh_triton(x)


# ── Shape helpers for profiling ──────────────────────────────────────────────


def get_inputs() -> list:
    """Return a list of (x,) tuples for common test shapes."""
    return [
        (torch.randn(1_000_000, device="npu", dtype=torch.float16), ),
        (torch.randn(10_000_000, device="npu", dtype=torch.float16), ),
        (torch.randn(100_000_000, device="npu", dtype=torch.float16), ),
    ]


def get_init_inputs() -> list:
    """Return a list of (args,) tuples for ModelNew constructor."""
    return [()]
