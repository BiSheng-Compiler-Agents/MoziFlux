import triton
import triton.language as tl

@triton.jit
def _fused_add_hswish_mul_kernel(x_ptr, add_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    add = tl.load(add_ptr + offs, mask=mask, other=0.0)
    z = x + add

    # Compute HardSwish(z) = z * clip(z + 3, 0, 6) / 6
    three = 3.0
    six = 6.0
    z_p3 = z + three
    clipped = tl.minimum(tl.maximum(z_p3, 0.0), six)
    hswish = z * (clipped / six)

    out = hswish
    tl.store(out_ptr + offs, out, mask=mask)
