import triton
import triton.language as tl

@triton.jit
def _scale_kernel(x_ptr, y_ptr, scale, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.multiple_of(offs, 16)
    tl.max_contiguous(offs, 16)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    s = tl.full([1], scale, x.dtype)
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)

@triton.jit
def _bn_fuse_params_kernel(
    mean_ptr, var_ptr, gamma_ptr, beta_ptr, convb_ptr,
    g_out_ptr, b_out_ptr,
    eps, scale, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.multiple_of(offs, 8)
    tl.max_contiguous(offs, 8)
    mask = offs < n_elements

    m = tl.load(mean_ptr + offs, mask=mask, other=0.0)
    v = tl.load(var_ptr + offs, mask=mask, other=0.0)
    g = tl.load(gamma_ptr + offs, mask=mask, other=1.0)
    b = tl.load(beta_ptr + offs, mask=mask, other=0.0)
    cb = tl.load(convb_ptr + offs, mask=mask, other=0.0)

    eps_t = tl.full([1], eps, dtype=v.dtype)
    s_t = tl.full([1], scale, dtype=g.dtype)

    std = tl.sqrt(v + eps_t)
    g_ch = (g * s_t) / std
    b_ch = b * s_t + (cb - m) * g_ch

    tl.store(g_out_ptr + offs, g_ch, mask=mask)
    tl.store(b_out_ptr + offs, b_ch, mask=mask)
