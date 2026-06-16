import triton
import triton.language as tl


@triton.jit
def _hinge_loss_sum_kernel(pred_ptr, targ_ptr, out_ptr, n_elements,
                           BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    tl.multiple_of(block_start, BLOCK_SIZE)
    tl.max_contiguous(offsets, BLOCK_SIZE)

    p = tl.load(pred_ptr + offsets, mask=mask, other=1.0)
    t = tl.load(targ_ptr + offsets, mask=mask, other=1.0)
    z = 1.0 - p * t
    z = tl.maximum(z, 0.0)
    part = tl.sum(z, axis=0)

    if n_elements <= BLOCK_SIZE:
        if pid == 0:
            tl.store(out_ptr, part)
    else:
        tl.atomic_add(out_ptr, part)
