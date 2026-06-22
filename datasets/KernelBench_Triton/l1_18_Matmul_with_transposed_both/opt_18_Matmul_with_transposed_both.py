"""
opt_18_Matmul_with_transposed_both.py — Optimized kernel for C = A^T @ B^T

A is stored as (K, M), B as (N, K), C as (M, N).
The kernel computes C[m, n] = sum_k A[k, m] * B[n, k].

Optimizations applied (see Optimizations.md for details):
  1. Removed cache_modifier=".cg" (P0 — silently kills Ascend compilation)
  2. al.compile_hint "dot_pad_only_k" on A and B tiles
  3. care_padding=False on all tl.load
  4. Hoisted M/N loop-invariant masks outside K loop
  5. GROUP_M=4 pid swizzle (1D grid) for L2 cache reuse
  6. Hoisted base pointer computation outside K loop
  7. tl.max_contiguous on arange offsets for better DMA codegen
  8. Expanded autotune configs with BLOCK_K=128 variants
"""

import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

# ── Autotune ──────────────────────────────────────────────────────────────────


@triton.autotune(
    configs=[
        # Balanced tiles — original configs adapted for 1D grid
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "BLOCK_K": 64,
                "GROUP_M": 8
            },
            num_warps=8,
            num_stages=4),
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 128,
                "BLOCK_K": 64,
                "GROUP_M": 8
            },
            num_warps=4,
            num_stages=4),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 64,
                "BLOCK_K": 64,
                "GROUP_M": 8
            },
            num_warps=4,
            num_stages=4),
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
                "GROUP_M": 8
            },
            num_warps=4,
            num_stages=2),
        # Wider N or M
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 256,
                "BLOCK_K": 32,
                "GROUP_M": 8
            },
            num_warps=8,
            num_stages=4),
        triton.Config(
            {
                "BLOCK_M": 256,
                "BLOCK_N": 128,
                "BLOCK_K": 32,
                "GROUP_M": 8
            },
            num_warps=8,
            num_stages=4),
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 256,
                "BLOCK_K": 64,
                "GROUP_M": 8
            },
            num_warps=8,
            num_stages=4),
        triton.Config(
            {
                "BLOCK_M": 256,
                "BLOCK_N": 64,
                "BLOCK_K": 64,
                "GROUP_M": 8
            },
            num_warps=8,
            num_stages=4),
        # Deep K configs
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "BLOCK_K": 128,
                "GROUP_M": 8
            },
            num_warps=8,
            num_stages=4),
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 128,
                "BLOCK_K": 128,
                "GROUP_M": 8
            },
            num_warps=4,
            num_stages=4),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 64,
                "BLOCK_K": 128,
                "GROUP_M": 8
            },
            num_warps=4,
            num_stages=4),
        # Large tile configs
        triton.Config(
            {
                "BLOCK_M": 256,
                "BLOCK_N": 256,
                "BLOCK_K": 64,
                "GROUP_M": 8
            },
            num_warps=8,
            num_stages=4),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 256,
                "BLOCK_K": 64,
                "GROUP_M": 8
            },
            num_warps=8,
            num_stages=4),
        triton.Config(
            {
                "BLOCK_M": 256,
                "BLOCK_N": 128,
                "BLOCK_K": 64,
                "GROUP_M": 8
            },
            num_warps=8,
            num_stages=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_AT_BT_kernel_opt(
    A_ptr,  # A: (K, M)
    B_ptr,  # B: (N, K)
    C_ptr,  # C: (M, N)
    M,
    N,
    K,
    stride_a_k,
    stride_a_m,
    stride_b_n,
    stride_b_k,
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
    rk = tl.max_contiguous(tl.arange(0, BLOCK_K), BLOCK_K)

    # Alignment hints
    tl.multiple_of(rm, BLOCK_M)
    tl.multiple_of(rn, BLOCK_N)
    tl.multiple_of(rk, BLOCK_K)
    tl.static_assert(BLOCK_K % 16 == 0)

    # ── Loop-invariant masks (M and N bounds don't change per K iteration) ─
    m_mask = rm < M
    n_mask = rn < N
    # A tile: (BLOCK_K, BLOCK_M) — mask columns by M-bound
    a_mask_cols = m_mask[None, :]  # [1, BM]
    # B tile: (BLOCK_K, BLOCK_N) — mask columns by N-bound
    b_mask_cols = n_mask[None, :]  # [1, BN]
    # Output mask
    out_mask = m_mask[:, None] & n_mask[None, :]

    # ── Hoisted base pointers ──────────────────────────────────────────────
    # A is (K, M): element [k, m] = A_ptr + k * stride_a_k + m * stride_a_m
    # Loaded tile shape: (BLOCK_K, BLOCK_M) → rows=K, cols=M
    a_base = A_ptr + rm[None, :] * stride_a_m  # [1, BM]

    # B is (N, K): element [n, k] = B_ptr + n * stride_b_n + k * stride_b_k
    # Loaded tile shape: (BLOCK_K, BLOCK_N) → rows=K, cols=N
    b_base = B_ptr + rn[None, :] * stride_b_n  # [1, BN]

    # ── Accumulator ────────────────────────────────────────────────────────
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ── K loop ─────────────────────────────────────────────────────────────
    a_ptrs = a_base + rk[:, None] * stride_a_k  # [BK, BM]
    b_ptrs = b_base + rk[:, None] * stride_b_k  # [BK, BN]

    for k0 in range(0, K, BLOCK_K):
        k_mask = (k0 + rk) < K
        k_mask_cols = k_mask[:, None]  # [BK, 1]

        # Load A tile at (K, M) as [BK, BM], then transpose for tl.dot
        a = tl.load(a_ptrs,
                    mask=k_mask_cols & a_mask_cols,
                    other=0.0,
                    care_padding=False)
        al.compile_hint(a, "dot_pad_only_k")
        a_t = tl.trans(a)  # [BM, BK]

        # Load B tile at (N, K) as [BK, BN]
        b = tl.load(b_ptrs,
                    mask=k_mask_cols & b_mask_cols,
                    other=0.0,
                    care_padding=False)
        al.compile_hint(b, "dot_pad_only_k")

        # Accumulate: C += A^T @ B^T — tl.dot(a, b) where
        #   a_t: [BM, BK], b: [BK, BN] → output: [BM, BN]
        acc += tl.dot(a_t, b, out_dtype=tl.float32)

        a_ptrs += BLOCK_K * stride_a_k
        b_ptrs += BLOCK_K * stride_b_k

    # ── Write output ───────────────────────────────────────────────────────
    c_ptrs = C_ptr + rm[:, None] * stride_c_m + rn[None, :] * stride_c_n
    tl.store(c_ptrs, acc, mask=out_mask)


# ── Dispatch function ────────────────────────────────────────────────────────


def matmul_at_bt(a, b):
    """
    Compute C = A^T @ B^T where:
      a: (K, M) — transposed layout
      b: (N, K) — transposed layout
    Returns: C: (M, N)
    """
    K, M = a.shape
    N, K2 = b.shape
    assert K == K2, f"A dim 0 ({K}) must match B dim 0 ({K2})"

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    def grid(META):
        return (triton.cdiv(M, META["BLOCK_M"]) *
                triton.cdiv(N, META["BLOCK_N"]), )

    _matmul_AT_BT_kernel_opt[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),  # stride_a_k, stride_a_m
        b.stride(0),
        b.stride(1),  # stride_b_n, stride_b_k
        c.stride(0),
        c.stride(1),  # stride_c_m, stride_c_n
    )
    return c


# ── Host model interface ─────────────────────────────────────────────────────


class ModelNew(torch.nn.Module):
    """Wrapper for the optimized A^T @ B^T kernel."""

    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return matmul_at_bt(a, b)

    # ── Input helpers for benchmarking / correctness ────────────────────────
    @staticmethod
    def get_inputs():
        """Return example input tensors matching typical shapes."""
        K, M, N = 2048, 4096, 4096
        a = torch.randn(K, M, device="npu", dtype=torch.float16)
        b = torch.randn(N, K, device="npu", dtype=torch.float16)
        return (a, b)

    @staticmethod
    def get_init_inputs():
        """Return init args (none needed)."""
        return ()


# ── Correctness test ─────────────────────────────────────────────────────────


def unit_test(device="npu"):
    """Correctness test against PyTorch reference."""
    torch.manual_seed(42)
    print("=" * 60)
    print("Correctness Test: Matmul with Transposed Both (C = A^T @ B^T)")
    print("=" * 60)

    shapes = [
        (128, 256, 192),
        (256, 128, 64),
        (512, 512, 256),
        (1024, 1024, 512),
        (127, 255, 191),  # non-power-of-2
        (33, 67, 128),
        (1, 1, 1),  # edge case
        (64, 1024, 768),
    ]

    model = ModelNew().to(device=device, dtype=torch.float16).eval()
    all_pass = True

    for M, N, K in shapes:
        a = torch.randn(K, M, device=device, dtype=torch.float16) * 0.1
        b = torch.randn(N, K, device=device, dtype=torch.float16) * 0.1

        with torch.no_grad():
            ref = torch.matmul(a.T, b.T)
            out = model(a, b)

        rtol = 1e-3
        atol = 1e-3
        close = torch.allclose(out, ref, rtol=rtol, atol=atol)
        if close:
            print(f"  [PASS] M={M:4d} N={N:4d} K={K:4d}")
        else:
            max_err = (out - ref).abs().max().item()
            print(
                f"  [FAIL] M={M:4d} N={N:4d} K={K:4d}  max_err={max_err:.6f}")
            all_pass = False

    if all_pass:
        print("\nAll tests PASSED")
    else:
        print("\nSome tests FAILED")
    return all_pass


if __name__ == "__main__":
    import sys
    device = sys.argv[1] if len(sys.argv) > 1 else "npu"
    unit_test(device=device)
