import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=8, num_stages=3),
    ],
    key=["n_elements"],
)
@triton.jit
def _fused_min_gelu_mul_kernel(
    x_ptr, y_ptr,
    add_value, multiply_value,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    # Alignment/contiguity hints for better codegen
    tl.multiple_of(offs, 256)
    tl.max_contiguous(offs, BLOCK_SIZE)

    # Load and upcast to fp32 for numerics
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    in_dtype = x.dtype
    x32 = x.to(tl.float32)

    # x = x + add_value; x = min(x, 0.0)
    x32 = x32 + add_value
    x32 = tl.minimum(x32, 0.0)

    # GELU exact: 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.7071067811865476
    t = x32 * inv_sqrt2
    e = tl.math.erf(t)
    scale = 0.5 * multiply_value
    y32 = x32 * (1.0 + e) * scale

    # Store back in original dtype
    tl.store(y_ptr + offs, y32.to(in_dtype), mask=mask)
