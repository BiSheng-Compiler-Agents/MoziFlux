import triton
import triton.language as tl

@triton.jit
def _fused_add_swish_tanh_gelu_hardtanh(
    x_ptr,          # *f32 [M, N]
    add_ptr,        # *f32 [N]
    out_ptr,        # *f32 [M, N]
    M: tl.constexpr,
    N: tl.constexpr,
    stride_xm,      # int
    stride_xn,      # int
    stride_om,      # int
    stride_on,      # int
    min_val,        # f32
    max_val,        # f32
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Provide aliasing/contiguity hints to aid vectorization
    tl.multiple_of(offs_m, BLOCK_M)
    tl.multiple_of(offs_n, BLOCK_N)
    tl.max_contiguous(offs_n, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on

    # Load tile and broadcast add vector across rows
    # Use cache modifiers: x is read-once -> prefer L2 (.cg), add vector reused across rows -> keep in cache (.ca)
    x = tl.load(x_ptrs, mask=mask, other=0.0, cache_modifier=".cg")
    add = tl.load(add_ptr + offs_n, mask=mask_n, other=0.0, cache_modifier=".ca")[None, :]

    # (x + add) -> Swish -> Tanh -> GELU(exact) -> Hardtanh
    y = x + add

    # Swish: y * sigmoid(y)
    y = y * tl.sigmoid(y)

    # Tanh via Triton math intrinsics for Ascend compatibility.
    y = tl.math.tanh(y)

    # GELU exact: 0.5 * y * (1 + erf(y / sqrt(2)))
    inv_sqrt2 = 0.7071067811865476
    y = 0.5 * y * (1.0 + tl.math.erf(y * inv_sqrt2))

    # Hardtanh clamp [-1, 1]
    y = tl.minimum(tl.maximum(y, min_val), max_val)

    tl.store(out_ptrs, y, mask=mask)
