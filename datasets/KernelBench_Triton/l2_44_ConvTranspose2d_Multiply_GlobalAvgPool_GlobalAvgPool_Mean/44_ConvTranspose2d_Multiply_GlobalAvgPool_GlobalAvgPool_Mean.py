import triton
import triton.language as tl


@triton.jit
def _global_avg_mul_kernel(
    x_ptr,
    out_ptr,
    NC,
    HW,
    multiplier,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    in_bounds = pid < NC
    base = x_ptr + pid * HW
    offs = tl.arange(0, BLOCK_SIZE)

    acc = tl.zeros((), dtype=tl.float32)
    start = 0
    while start < HW:
        idx0 = start + offs
        mask0 = (idx0 < HW) & in_bounds
        vals0 = tl.load(base + idx0, mask=mask0, other=0.0)
        acc += tl.sum(vals0.to(tl.float32), axis=0)

        idx1 = idx0 + BLOCK_SIZE
        mask1 = (idx1 < HW) & in_bounds
        vals1 = tl.load(base + idx1, mask=mask1, other=0.0)
        acc += tl.sum(vals1.to(tl.float32), axis=0)
        start += 2 * BLOCK_SIZE

    tl.store(out_ptr + pid, acc * multiplier / HW, mask=in_bounds)
