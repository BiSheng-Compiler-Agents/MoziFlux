"""
Optimized Sigmoid kernel for Ascend NPU.
- @triton.autotune with bucketed autotune key
- care_padding=False for load/store
- Two-path dispatch: direct for small N, persistent for large N
- Simplified computation: 1 - inv for negative branch (sub instead of mul)
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl

# ---------- autotune ----------

MAX_PROGRAMS = 65535  # Ascend FFTS grid cap

# The MIN_BLOCK is the smallest BLOCK_SIZE in the autotune configs
MIN_BLOCK = 256


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
        triton.Config({'BLOCK_SIZE': 4096}),
    ],
    key=['n_elements_pow2'],
)
@triton.jit
def _sigmoid_direct(x_ptr, y_ptr, n_elements, n_elements_pow2: tl.constexpr,
                    BLOCK_SIZE: tl.constexpr):
    """Direct dispatch kernel — one program per tile, no while loop."""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0, care_padding=False)
    x32 = x.to(tl.float32)

    # Numerically-stable sigmoid:
    # z = exp(-|x|); inv = 1 / (1 + z)
    # if x >= 0: y = inv
    # else:      y = 1 - inv
    z = tl.exp(-tl.abs(x32))
    inv = 1.0 / (1.0 + z)
    y32 = tl.where(x32 >= 0.0, inv, 1.0 - inv)

    y = y32.to(x.dtype)
    tl.store(y_ptr + offsets, y, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
        triton.Config({'BLOCK_SIZE': 4096}),
    ],
    key=['n_elements_pow2'],
)
@triton.jit
def _sigmoid_persistent(x_ptr, y_ptr, n_elements, n_programs: tl.constexpr,
                        n_elements_pow2: tl.constexpr,
                        BLOCK_SIZE: tl.constexpr):
    """Persistent dispatch kernel — caps grid at MAX_PROGRAMS, loops."""
    pid = tl.program_id(axis=0)

    n_tiles = tl.cdiv(n_elements, BLOCK_SIZE)
    for tile_id in range(pid, n_tiles, n_programs):
        offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0, care_padding=False)
        x32 = x.to(tl.float32)

        z = tl.exp(-tl.abs(x32))
        inv = 1.0 / (1.0 + z)
        y32 = tl.where(x32 >= 0.0, inv, 1.0 - inv)

        y = y32.to(x.dtype)
        tl.store(y_ptr + offsets, y, mask=mask)


# ---------- Host interface ----------


class ModelNew(nn.Module):
    """Optimized Sigmoid activation using Triton on Ascend NPU."""

    def __init__(self, dtype=torch.float16):
        super().__init__()
        self.dtype = dtype

    def forward(self, x):
        assert x.is_cuda or x.device.type == 'npu', 'Input must be on NPU/CUDA device'
        x_flat = x.view(-1)
        n = x_flat.numel()
        y = torch.empty_like(x_flat)
        n_elements_pow2 = 1 << (n - 1).bit_length()

        # Two-path dispatch: direct for small N, persistent for large N
        if triton.cdiv(n, MIN_BLOCK) > MAX_PROGRAMS:
            n_programs = MAX_PROGRAMS
            _sigmoid_persistent[(n_programs, )](
                x_flat, y, n, n_programs, n_elements_pow2=n_elements_pow2)
        else:

            def grid(meta):
                return (triton.cdiv(n, meta['BLOCK_SIZE']), )

            _sigmoid_direct[grid](x_flat,
                                  y,
                                  n,
                                  n_elements_pow2=n_elements_pow2)

        return y.view_as(x)


# ---------- Unit test (runs at import if executed directly) ----------


def test_sigmoid():
    """Verify precision against PyTorch reference for various shapes and dtypes."""
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else \
             ('npu' if hasattr(torch, 'npu') and torch.npu.is_available() else 'cpu')

    ref_fn = torch.sigmoid
    model = ModelNew(dtype=torch.float16)

    test_shapes = [
        (127, ),
        (128, ),
        (255, ),
        (256, ),
        (1023, ),
        (1024, ),
        (4096, ),
    ]

    for shape in test_shapes:
        for dtype in [torch.float16, torch.float32, torch.bfloat16]:
            x = torch.randn(*shape, device=device, dtype=dtype) * 5.0
            y_ref = ref_fn(x)
            y_opt = model(x)
            try:
                torch.testing.assert_close(y_opt, y_ref, rtol=1e-3, atol=1e-3)
                print(f'PASS: shape={shape}, dtype={dtype}')
            except AssertionError as e:
                print(f'FAIL: shape={shape}, dtype={dtype}')
                print(f'  {e}')
                raise

    print('All tests passed!')


if __name__ == '__main__':
    test_sigmoid()
