import triton
import triton.language as tl


@triton.jit
def _clamp_divide_inplace_kernel(
    x_ptr,  # input/output pointer (contiguous tensor)
    n_elements,  # total number of elements
    min_value,  # clamp minimum (scalar)
    divisor,  # division scalar
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.multiple_of(offsets, 8)
    x = tl.load(x_ptr + offsets,
                mask=mask,
                other=0.0,
                eviction_policy="evict_last")
    # Clamp to min then divide. Use maximum to match torch.clamp(min=...) semantics incl. NaN propagation.
    x = tl.maximum(x, min_value)
    x = x / divisor
    tl.store(x_ptr + offsets, x, mask=mask, eviction_policy="evict_last")
