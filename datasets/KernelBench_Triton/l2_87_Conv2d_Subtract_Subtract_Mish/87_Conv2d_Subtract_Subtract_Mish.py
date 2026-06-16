import triton
import triton.language as tl


@triton.jit
def _fused_sub_mish_kernel(
    x_ptr,  # in-place pointer to tensor
    n_elements,  # total number of elements
    sub1,  # subtract_value_1 (scalar)
    sub2,  # subtract_value_2 (scalar)
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load and upcast for numerics
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x32 = x.to(tl.float32)

    # Apply sequential subtractions: (x - sub1) - sub2
    x32 = x32 - sub1
    x32 = x32 - sub2

    zero = tl.zeros_like(x32)
    one = zero + 1.0
    twenty = zero + 20.0
    neg_twenty = zero - 20.0

    # Match PyTorch softplus threshold behavior for better numerical parity.
    abs_x = tl.abs(x32)
    sp_mid = tl.where(x32 > zero, x32, zero) + tl.log(one + tl.exp(-abs_x))
    sp = tl.where(x32 > twenty, x32,
                  tl.where(x32 < neg_twenty, tl.exp(x32), sp_mid))
    y32 = x32 * tl.tanh(sp)
    y = y32.to(x.dtype)

    tl.store(x_ptr + offsets, y, mask=mask)
