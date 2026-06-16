import triton
import triton.language as tl


@triton.jit
def _mish_tanh_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.multiple_of(offs, 16)
    tl.max_contiguous(offs, 16)
    mask = offs < n_elements

    # Load and upcast to fp32 for numerical stability
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)

    # Compute tanh(softplus(x)) in a log-free, stable way:
    # exp(2*softplus(x)) = exp(2*max(x,0)) * (1 + exp(-|x|))^2
    ax = tl.abs(x_f32)
    e_neg_ax = tl.exp(-ax)
    one_plus = 1.0 + e_neg_ax
    e2max = tl.exp(2.0 * tl.maximum(x_f32, 0.0))
    e2s = e2max * one_plus * one_plus
    tanh_sp = 1.0 - 2.0 / (1.0 + e2s)

    # mish(x) = x * tanh(softplus(x))
    mish = x_f32 * tanh_sp

    # tanh(mish) using stable formulation: tanh(a) = sign(a) * (1 - 2 / (1 + exp(2|a|)))
    am = tl.abs(mish)
    e2a = tl.exp(2.0 * am)
    sign = tl.where(mish >= 0.0, 1.0, -1.0)
    out_f32 = sign * (1.0 - 2.0 / (1.0 + e2a))

    # Downcast and store
    out = out_f32.to(x.dtype)
    tl.store(y_ptr + offs, out, mask=mask)
