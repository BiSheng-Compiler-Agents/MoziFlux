import triton
import triton.language as tl

@triton.jit
def _smooth_l1_mean_atomic_kernel(
    pred_ptr, tgt_ptr, out_mean_ptr,
    n_elements,
    inv_n,
    beta: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # Process the block in smaller chunks to reduce register pressure and increase occupancy
    CHUNK: tl.constexpr = 1024
    inv_beta = 1.0 / beta
    acc = tl.zeros((), dtype=tl.float32)

    for off in range(0, BLOCK_SIZE, CHUNK):
        offsets = block_start + off + tl.arange(0, CHUNK)
        mask = offsets < n_elements

        tl.multiple_of(offsets, CHUNK)
        tl.max_contiguous(offsets, CHUNK)

        # Load and compute in fp32 for stability
        p = tl.load(pred_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        t = tl.load(tgt_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        d = p - t
        ad = tl.abs(d)

        # Huber / Smooth L1 with beta
        small = 0.5 * (d * d) * inv_beta
        large = ad - 0.5 * beta
        loss = tl.where(ad < beta, small, large)
        loss = tl.where(mask, loss, 0.0)

        acc += tl.sum(loss, axis=0)

    # Accumulate mean contribution atomically (only once per program)
    tl.atomic_add(out_mean_ptr, acc * inv_n)
