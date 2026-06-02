import triton
import triton.language as tl

    @triton.jit
    def _mse_partial_sum_kernel(
        x_ptr,  # *T
        y_ptr,  # *T
        out_ptr,  # *fp32, single element (accumulator)
        n_elements,  # total number of elements
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        num_pid = tl.num_programs(axis=0)

        acc = 0.0  # scalar fp32 accumulator
        arange = tl.arange(0, BLOCK_SIZE)
        tl.multiple_of(arange, 128)

        # Each program processes strided chunks; we unroll by 2 to reduce loop overhead
        offset = pid * BLOCK_SIZE
        stride = BLOCK_SIZE * num_pid

        while offset < n_elements:
            offsets0 = offset + arange
            tl.max_contiguous(offsets0, BLOCK_SIZE)
            mask0 = offsets0 < n_elements
            x0 = tl.load(x_ptr + offsets0, mask=mask0, other=0.0)
            y0 = tl.load(y_ptr + offsets0, mask=mask0, other=0.0)
            diff0 = (x0 - y0).to(tl.float32)
            acc += tl.sum(diff0 * diff0, axis=0)

            offsets1 = offset + stride + arange
            tl.max_contiguous(offsets1, BLOCK_SIZE)
            mask1 = offsets1 < n_elements
            x1 = tl.load(x_ptr + offsets1, mask=mask1, other=0.0)
            y1 = tl.load(y_ptr + offsets1, mask=mask1, other=0.0)
            diff1 = (x1 - y1).to(tl.float32)
            acc += tl.sum(diff1 * diff1, axis=0)

            offset += 2 * stride

        tl.atomic_add(out_ptr + 0, acc)
