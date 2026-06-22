"""
opt_16_Matmul_with_transposed_A.py — Optimized kernel for C = A^T @ B

A is stored as (K, M), B as (K, N), C as (M, N).
The kernel computes C[m, n] = sum_k A[k, m] * B[k, n].

Optimizations applied (see Optimizations.md for details):
  1. Removed cache_modifier=".cg" (silently kills Ascend compilation)
  2. al.compile_hint "dot_pad_only_k" on A and B tiles
  3. care_padding=False on all tl.load
  4. Hoisted M/N loop-invariant masks outside K loop
  5. GROUP_M=4 pid swizzle (1D grid) for L2 cache reuse
  6. Hoisted base pointer computation
  7. tl.max_contiguous on arange offsets
"""

import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

# ── Autotune ──────────────────────────────────────────────────────────────────


@triton.autotune(
    configs=[
        # Balanced tiles
        triton.Config({
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 64
        },
                      num_warps=8,
                      num_stages=5),
        triton.Config({
            "BLOCK_M": 64,
            "BLOCK_N": 128,
            "BLOCK_K": 64
        },
                      num_warps=4,
                      num_stages=4),
        triton.Config({
            "BLOCK_M": 128,
            "BLOCK_N": 64,
            "BLOCK_K": 64
        },
                      num_warps=4,
                      num_stages=4),
        triton.Config({
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "BLOCK_K": 32
        },
                      num_warps=4,
                      num_stages=2),
        # Wider N or M
        triton.Config({
            "BLOCK_M": 128,
            "BLOCK_N": 256,
            "BLOCK_K": 32
        },
                      num_warps=8,
                      num_stages=4),
        triton.Config({
            "BLOCK_M": 256,
            "BLOCK_N": 128,
            "BLOCK_K": 32
        },
                      num_warps=8,
                      num_stages=4),
        triton.Config({
            "BLOCK_M": 64,
            "BLOCK_N": 256,
            "BLOCK_K": 64
        },
                      num_warps=8,
                      num_stages=4),
        # Deeper K
        triton.Config({
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 128
        },
                      num_warps=8,
                      num_stages=4),
        triton.Config({
            "BLOCK_M": 64,
            "BLOCK_N": 128,
            "BLOCK_K": 128
        },
                      num_warps=4,
                      num_stages=4),
        triton.Config({
            "BLOCK_M": 128,
            "BLOCK_N": 64,
            "BLOCK_K": 128
        },
                      num_warps=4,
                      num_stages=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_AT_B_kernel_opt(
    A_ptr,  # A: (K, M)
    B_ptr,  # B: (K, N)
    C_ptr,  # C: (M, N)
    M,
    N,
    K,
    stride_a_k,
    stride_a_m,
    stride_b_k,
    stride_b_n,
    stride_c_m,
    stride_c_n,
    GROUP_M: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # ── 1D grid with GROUP_M swizzle ──────────────────────────────────────
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n

    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = num_pid_m - first_pid_m
    if group_size_m > GROUP_M:
        group_size_m = GROUP_M
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ── Tile offsets ───────────────────────────────────────────────────────
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    # Alignment hints
    tl.multiple_of(rm, BLOCK_M)
    tl.multiple_of(rn, BLOCK_N)
    tl.multiple_of(rk, BLOCK_K)
    tl.max_contiguous(tl.multiple_of(rk, BLOCK_K), BLOCK_K)

    # ── Loop-invariant masks (M and N bounds don't change per K iteration) ─
    m_mask = rm < M
    n_mask = rn < N
    # A: [BK, BM] — mask cols by M (rows are K-bound, checked per-iter)
    a_mask_cols = m_mask[None, :]
    # B: [BK, BN] — mask cols by N (rows are K-bound, checked per-iter)
    b_mask_cols = n_mask[None, :]

    # ── Hoisted base pointers ──────────────────────────────────────────────
    a_base = A_ptr + rm[None, :] * stride_a_m  # [1, BM]
    b_base = B_ptr + rn[None, :] * stride_b_n  # [1, BN]

    # ── Accumulator ────────────────────────────────────────────────────────
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ── K loop ─────────────────────────────────────────────────────────────
    k0 = 0
    while k0 < K:
        off_k = k0 + rk  # [BK]

        # Load A tile at (K, M) as [BK, BM], then transpose for tl.dot
        a_ptrs = a_base + off_k[:, None] * stride_a_k  # [BK, BM]
        k_mask = off_k[:, None] < K
        a = tl.load(a_ptrs,
                    mask=k_mask & a_mask_cols,
                    other=0.0,
                    care_padding=False)
        al.compile_hint(a, "dot_pad_only_k")
        a_t = tl.trans(a).to(tl.float32)  # [BM, BK]

        # Load B tile at (K, N) as [BK, BN]
        b_ptrs = b_base + off_k[:, None] * stride_b_k  # [BK, BN]
        b = tl.load(b_ptrs,
                    mask=k_mask & b_mask_cols,
                    other=0.0,
                    care_padding=False)
        al.compile_hint(b, "dot_pad_only_k")
        b_t = b.to(tl.float32)  # [BK, BN]

        # Accumulate: C += A^T[tile] @ B[tile]
        acc += tl.dot(a_t, b_t)

        k0 += BLOCK_K

    # ── Write output ───────────────────────────────────────────────────────
    c_ptrs = C_ptr + rm[:, None] * stride_c_m + rn[None, :] * stride_c_n
    c_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(c_ptrs, acc, mask=c_mask)


# ── Dispatch function ────────────────────────────────────────────────────────


def matmul_at_b(a, b):
    """
    Compute C = A^T @ B where:
      a: (K, M) — stored in transposed layout
      b: (K, N)
    Returns: C: (M, N)
    """
    K, M = a.shape
    K2, N = b.shape
    assert K == K2, f"A dim 0 ({K}) must match B dim 0 ({K2})"

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    def grid(META):
        return (triton.cdiv(M, META["BLOCK_M"]) *
                triton.cdiv(N, META["BLOCK_N"]), )

    _matmul_AT_B_kernel_opt[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),  # stride_a_k, stride_a_m
        b.stride(0),
        b.stride(1),  # stride_b_k, stride_b_n
        c.stride(0),
        c.stride(1),  # stride_c_m, stride_c_n
        GROUP_M=4,
    )
    return c


# ── Host model interface ─────────────────────────────────────────────────────


class ModelNew(torch.nn.Module):
    """Wrapper for the optimized A^T @ B kernel."""

    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return matmul_at_b(a, b)

    # ── Input helpers for benchmarking / correctness ────────────────────────
    @staticmethod
    def get_inputs():
        """Return example input tensors matching typical shapes."""
        K, M, N = 2048, 4096, 4096
        a = torch.randn(K, M, device="npu", dtype=torch.float16)
        b = torch.randn(K, N, device="npu", dtype=torch.float16)
        return (a, b)

    @staticmethod
    def get_init_inputs():
        """Return init args (none needed)."""
        return ()
