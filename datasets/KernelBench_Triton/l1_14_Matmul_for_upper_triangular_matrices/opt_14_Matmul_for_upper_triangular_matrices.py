import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al


@triton.autotune(
    configs=[
        triton.Config({
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 32
        },
                      num_warps=8),
        triton.Config({
            "BLOCK_M": 64,
            "BLOCK_N": 128,
            "BLOCK_K": 32
        },
                      num_warps=4),
        triton.Config({
            "BLOCK_M": 128,
            "BLOCK_N": 64,
            "BLOCK_K": 32
        },
                      num_warps=4),
        triton.Config({
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "BLOCK_K": 64
        },
                      num_warps=4),
        triton.Config({
            "BLOCK_M": 128,
            "BLOCK_N": 256,
            "BLOCK_K": 64
        },
                      num_warps=8),
        triton.Config({
            "BLOCK_M": 256,
            "BLOCK_N": 128,
            "BLOCK_K": 64
        },
                      num_warps=8),
    ],
    key=["N"],
)
@triton.jit
def _upper_tri_matmul_kernel_opt(
    A_ptr,
    B_ptr,
    C_ptr,
    N,
    stride_Am,
    stride_Ak,
    stride_Bk,
    stride_Bn,
    stride_Cm,
    stride_Cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Optimized upper-triangular matrix multiplication kernel for Ascend NPU.

    1D grid with GROUP_M swizzle for L2 cache reuse.
    Each program processes multiple tiles via intra-core looping.
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    # Number of tile blocks in each dimension
    NUM_PID_M = tl.cdiv(N, BLOCK_M)
    NUM_PID_N = tl.cdiv(N, BLOCK_N)
    total_tiles = NUM_PID_M * NUM_PID_N

    # GROUP_M swizzle: group consecutive M-blocks for better L2 reuse
    GROUP_M: tl.constexpr = 4
    group_width = GROUP_M * NUM_PID_N

    # Iterate over tiles assigned to this program
    for start_idx in range(pid, total_tiles, num_programs):
        # GROUP_M swizzle to map flat index to (pid_m, pid_n)
        group_id = start_idx // group_width
        first_pid_m = group_id * GROUP_M
        group_size_m = tl.minimum(NUM_PID_M - first_pid_m, GROUP_M)
        pid_m = first_pid_m + (start_idx % group_size_m)
        pid_n = (start_idx // group_size_m) % NUM_PID_N

        m0 = pid_m * BLOCK_M
        n0 = pid_n * BLOCK_N

        # Gate: only process tiles that are in-bounds AND on/above diagonal
        valid_tile = m0 < N
        valid_tile = valid_tile & (n0 < N)
        valid_tile = valid_tile & (m0 <= n0 + BLOCK_N - 1)
        if valid_tile:
            rm = m0 + tl.arange(0, BLOCK_M)
            rn = n0 + tl.arange(0, BLOCK_N)
            m_in = rm < N
            n_in = rn < N

            # Multiple-of annotations for compiler
            tl.multiple_of(rm, BLOCK_M)
            tl.multiple_of(rn, BLOCK_N)

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            # K sweep using tiled dot-product
            # A and B are loaded inside the loop; Ascend multibuffer overlaps DMA with compute
            for k0 in range(0, N, BLOCK_K):
                k = k0 + tl.arange(0, BLOCK_K)
                k_in = k < N
                tl.multiple_of(k, BLOCK_K)

                a = tl.load(
                    A_ptr + (rm[:, None] * stride_Am + k[None, :] * stride_Ak),
                    mask=m_in[:, None] & k_in[None, :],
                    other=0.0,
                    care_padding=False,
                )
                b = tl.load(
                    B_ptr + (k[:, None] * stride_Bk + rn[None, :] * stride_Bn),
                    mask=k_in[:, None] & n_in[None, :],
                    other=0.0,
                    care_padding=False,
                )

                # Ascend-specific optimizations for Cube unit: pad only K, double-buffer
                al.compile_hint(a, "dot_pad_only_k")
                al.compile_hint(b, "dot_pad_only_k")

                # Double buffering: compiler overlaps next K-tile DMA with current compute
                al.multibuffer(a, size=2)
                al.multibuffer(b, size=2)

                acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

            # Store only upper-triangular region; tl.store casts to C dtype.
            tile_all_upper = (m0 + BLOCK_M - 1) <= n0
            full_in_bounds = (m0 + BLOCK_M) <= N
            full_in_bounds = full_in_bounds & ((n0 + BLOCK_N) <= N)
            use_fast_store = tile_all_upper & full_in_bounds

            c_ptrs = C_ptr + (rm[:, None] * stride_Cm +
                              rn[None, :] * stride_Cn)

            if use_fast_store:
                # Fast path: entirely on or above diagonal + fully in bounds
                tl.store(c_ptrs, acc)
            else:
                store_mask = (rm[:, None]
                              <= rn[None, :]) & m_in[:, None] & n_in[None, :]
                tl.store(c_ptrs, acc, mask=store_mask)


class ModelNew(torch.nn.Module):
    """Host interface for upper-triangular matrix multiplication."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def get_inputs():
        """Return a representative input tensor pair for benchmarking."""
        N = 1024
        A = torch.randn(N, N, device="npu", dtype=torch.float16)
        B = torch.randn(N, N, device="npu", dtype=torch.float16)
        return [A, B]

    @staticmethod
    def get_init_inputs():
        """Return init args (none needed for this model)."""
        return []

    def forward(self, A, B):
        """
        Compute upper-triangular part of A @ B.

        Args:
            A: (N, N) tensor
            B: (N, N) tensor
        Returns:
            C: (N, N) tensor — upper triangular part of A @ B
        """
        assert A.shape == B.shape and A.ndim == 2
        N = A.shape[0]
        C = torch.empty((N, N), device=A.device, dtype=A.dtype)

        # Grid: 1D over valid tiles, capped at physical cores
        num_cores = min(32, triton.cdiv(N, 64) * triton.cdiv(N, 64))
        grid = (num_cores, )

        _upper_tri_matmul_kernel_opt[grid](
            A,
            B,
            C,
            N,
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(1),
            C.stride(0),
            C.stride(1),
        )
        return C
