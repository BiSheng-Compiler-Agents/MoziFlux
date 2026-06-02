import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 256}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK": 512}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK": 2048}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK": 4096}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK": 8192}, num_warps=8, num_stages=5),
    ],
    key=["L"],
)
@triton.jit
def _global_avg_pool3d_ncdhw_kernel(x_ptr, y_ptr, M, L: tl.constexpr, BLOCK: tl.constexpr):
    # One program per (n, c) pair; reduce over contiguous DHW block of length L.
    pid = tl.program_id(axis=0)
    valid_pid = pid < M

    base = pid * L
    offs = tl.arange(0, BLOCK)

    # Software pipelined reduction with fp32 accumulation
    total = 0.0
    # Preload first chunk
    idx0 = offs
    mask0 = valid_pid & (idx0 < L)
    vals0 = tl.load(x_ptr + base + idx0, mask=mask0, other=0.0).to(tl.float32)

    for start in range(BLOCK, L, BLOCK):
        idx1 = start + offs
        mask1 = valid_pid & (idx1 < L)
        vals1 = tl.load(x_ptr + base + idx1, mask=mask1, other=0.0).to(tl.float32)
        # Reduce previously loaded chunk while the next is in flight
        total += tl.sum(vals0, axis=0)
        vals0 = vals1

    # Final chunk reduction
    total += tl.sum(vals0, axis=0)

    mean = total * (1.0 / L)
    tl.store(y_ptr + pid, mean, mask=valid_pid)
