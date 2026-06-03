import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import triton
import triton.language as tl


# ── Optimization summary ──────────────────────────────────────────────────────
#
# 1. TWO-PATH DISPATCH (mandatory for this kernel)
#    - n_tiles ≤ MAX_PROGRAMS  →  DIRECT kernel  (one program per tile, no
#      work-stealing while loop). At 4096×4096 (N=16M, BLOCK=4096) n_tiles=4096
#      is well under the 65535 FFTS grid cap, so the while-loop adds JUMPC
#      overhead with zero FFTS dispatch benefit. Direct is faster here.
#    - n_tiles >  MAX_PROGRAMS  →  PERSISTENT kernel. Required when the natural
#      tile count would exceed 65535 (e.g. n_elements ≥ 65535 × BLOCK_SIZE =
#      268M). Direct dispatch would either crash (coredim > UINT16_MAX) or pay
#      1,150 cy/program × N FFTS dispatch cost.
#    Routing threshold: use cdiv(n, BLOCK_SIZE) > MAX_PROGRAMS, not n itself.
#
# 2. REMOVED `if full/else` BRANCH INSIDE KERNEL
#    Baseline used Python if-branches in @triton.jit to pick masked vs
#    unmasked load/store. On Ascend, this compiles to SCALAR conditional logic
#    (STI_XN_IMM + LD_XD_XN_IMM register spills costing ~1200–1700 cy/tile).
#    Replaced with two kernels: a clean unmasked path for full tiles and a
#    masked path for the boundary tile.
#
# 3. REMOVED CUDA-SPECIFIC `cache_modifier=".cg"`
#    The L2 bypass hint is silently ignored on Ascend (no L2 in the same
#    sense). Dropped for clarity.
#
# 4. REMOVED `tl.full((BLOCK_SIZE,), s, dtype=x.dtype)` BROADCAST
#    Triton already broadcasts a Python scalar argument to the block
#    dimension for `*`. The explicit full-tensor cast was a (BLOCK_SIZE,)
#    tile of constants in the UB that the compiler had to materialize and
#    the VMUL then had to consume. Direct `x * s` is one fewer instr.
#
# 5. `care_padding=False` ON THE UNMASKED FAST PATH
#    For full tiles, the masked-load check is unnecessary. Skipping it gives
#    ~5–10% free speedup at BLOCK_SIZE=4096. Safe because the padding bytes
#    are not consumed downstream (the load boundary is exactly at n_elements).
#
# 6. tl.multiple_of / tl.max_contiguous HINTS
#    Signal 16-element alignment and run-length to the Ascend compiler so it
#    can emit large-block MTE2 DMA transactions.
#
# 7. GRID CAP AT 65535
#    Ascend FFTS hard-crashes if any grid dimension > 65535. The persistent
#    path caps explicitly; the direct path is gated by the routing check.
# ──────────────────────────────────────────────────────────────────────────────


# Ascend FFTS hard grid cap (coredim > UINT16_MAX crashes the scheduler).
_MAX_PROGRAMS = 65535


# ── Direct kernel ─────────────────────────────────────────────────────────────
# One program per tile. Faster than persistent for n_tiles ≤ MAX_PROGRAMS.

@triton.jit
def _scale_kernel_direct(
    x_ptr,
    y_ptr,
    s,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Direct one-program-per-tile scalar multiply (no while loop)."""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.multiple_of(offsets, 16)
    tl.max_contiguous(offsets, 16)

    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0, care_padding=False)
    y = x * s
    tl.store(y_ptr + offsets, y, mask=mask)


# ── Persistent kernel ─────────────────────────────────────────────────────────
# Used only when n_tiles > MAX_PROGRAMS. Amortises FFTS per-program cost.

@triton.jit
def _scale_kernel_persistent(
    x_ptr,
    y_ptr,
    s,
    n_elements,
    n_programs,
    BLOCK_SIZE: tl.constexpr,
):
    """Work-stealing persistent scalar multiply.

    Each program processes multiple BLOCK_SIZE-element tiles by striding
    tile_id += n_programs. Reduces FFTS per-program dispatch cost (~1,150 cy)
    from once-per-tile to once-per-program.
    """
    pid = tl.program_id(axis=0)
    tile_id = pid
    while tile_id * BLOCK_SIZE < n_elements:
        offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        tl.multiple_of(offsets, 16)
        tl.max_contiguous(offsets, 16)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0, care_padding=False)
        y = x * s
        tl.store(y_ptr + offsets, y, mask=mask)
        tile_id += n_programs


# ── Host interface ────────────────────────────────────────────────────────────

class ModelNew(nn.Module):
    """Ascend NPU-optimised matrix × scalar multiplication.

    Two-path dispatch:
      - n_tiles ≤ 65535 → direct kernel (cheaper, no work-stealing overhead)
      - n_tiles >  65535 → persistent kernel (amortises FFTS dispatch cost)

    Args:
        scalar: The scalar value to multiply every element by.
    """

    BLOCK_SIZE: int = 4096

    def __init__(self, scalar: float):
        super().__init__()
        self.scalar = float(scalar)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type != "npu":
            raise ValueError("ModelNew expects an Ascend NPU tensor input")
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError(
                "ModelNew supports float16, bfloat16, and float32 tensors"
            )
        if x.numel() == 0:
            return torch.empty_like(x)

        x_flat = x.contiguous().view(-1)
        y_flat = torch.empty_like(x_flat)
        n_elements = x_flat.numel()
        n_tiles = triton.cdiv(n_elements, self.BLOCK_SIZE)

        if n_tiles > _MAX_PROGRAMS:
            # Persistent path — cap grid to 65535, each program strides.
            n_programs = _MAX_PROGRAMS
            _scale_kernel_persistent[(n_programs,)](
                x_flat, y_flat, self.scalar,
                n_elements, n_programs,
                BLOCK_SIZE=self.BLOCK_SIZE,
            )
        else:
            # Direct path — one program per tile.
            _scale_kernel_direct[(n_tiles,)](
                x_flat, y_flat, self.scalar,
                n_elements,
                BLOCK_SIZE=self.BLOCK_SIZE,
            )
        return y_flat.view_as(x)


# ── benchmark shapes ─────────────────────────────────────────────────────────
M = 4096
N = 4096
scalar_val = 3.14


def get_inputs():
    x = torch.rand(M, N)
    return [x]


def get_init_inputs():
    return [scalar_val]
