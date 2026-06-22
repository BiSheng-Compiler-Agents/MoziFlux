import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al


@triton.autotune(
    configs=[
        # Original baseline configs — adapted with GROUP_M for 1D grid
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "BLOCK_K": 32,
                "GROUP_M": 8
            },
            num_warps=8,
            num_stages=4),
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 128,
                "BLOCK_K": 32,
                "GROUP_M": 8
            },
            num_warps=4,
            num_stages=4),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
                "GROUP_M": 8
            },
            num_warps=4,
            num_stages=4),
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_K": 64,
                "GROUP_M": 8
            },
            num_warps=4,
            num_stages=3),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "BLOCK_K": 64,
                "GROUP_M": 8
            },
            num_warps=8,
            num_stages=3),
        triton.Config(
            {
                "BLOCK_M": 32,
                "BLOCK_N": 256,
                "BLOCK_K": 32,
                "GROUP_M": 8
            },
            num_warps=8,
            num_stages=4),
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
        # Tensor-core friendly with larger BLOCK_K
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 256,
                "BLOCK_K": 128,
                "GROUP_M": 4
            },
            num_warps=8,
            num_stages=4),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 64,
                "BLOCK_K": 128,
                "GROUP_M": 4
            },
            num_warps=4,
            num_stages=4),
        # New larger-tile configs
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
                "BLOCK_M": 256,
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
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 256,
                "BLOCK_K": 64,
                "GROUP_M": 8
            },
            num_warps=8,
            num_stages=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _a_bt_matmul_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    Optimized A @ B^T matmul:
    - 1D grid with GROUP_M swizzle (L2 cache reuse)
    - Hoisted masks outside K-loop
    - dot_pad_only_k compile hint for UB efficiency
    - al.multibuffer double buffering (DMA/compute overlap)
    - care_padding=False on loads (free speedup)
    - Expanded autotune configs with larger BLOCK_K
    """
    # ---- 1D grid with GROUP_M pid swizzle ----
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(GROUP_M, num_pid_m - first_pid_m)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid // group_size_m) % num_pid_n

    # ---- Tile offsets ----
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.max_contiguous(tl.arange(0, BLOCK_K), BLOCK_K)

    # ---- Alignment hints ----
    tl.multiple_of(offs_k, 16)
    tl.multiple_of(offs_m, 16)
    tl.multiple_of(offs_n, 16)
    tl.static_assert(BLOCK_K % 16 == 0)

    # ---- Hoist masks outside K-loop ----
    m_mask = offs_m < M
    n_mask = offs_n < N
    out_mask = m_mask[:, None] & n_mask[None, :]

    # ---- FP32 accumulator ----
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ---- Base pointers for first K tile ----
    a_ptrs = A_ptr + (offs_m[:, None] * stride_am +
                      offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk +
                      offs_n[None, :] * stride_bn)

    # ---- K loop ----
    for k0 in range(0, K, BLOCK_K):
        k_mask = (k0 + offs_k) < K
        a_mask = m_mask[:, None] & k_mask[None, :]
        b_mask = k_mask[:, None] & n_mask[None, :]

        a = tl.load(a_ptrs,
                    mask=a_mask,
                    other=0.0,
                    cache_modifier=".cg",
                    care_padding=False)
        b = tl.load(b_ptrs,
                    mask=b_mask,
                    other=0.0,
                    cache_modifier=".cg",
                    care_padding=False)

        # Ascend-specific compiler hints
        al.compile_hint(a, "dot_pad_only_k")
        al.compile_hint(b, "dot_pad_only_k")
        al.multibuffer(a, size=2)
        al.multibuffer(b, size=2)

        acc += tl.dot(a, b, out_dtype=tl.float32)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # ---- Store output ----
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm +
                      offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=out_mask)


class ModelNew(torch.nn.Module):
    """
    Host wrapper for optimized matmul with transposed B: C = A @ B^T
    A: (M, K)
    B: (N, K)  — stored as (N, K), accessed as transposed (K, N)
    C: (M, N)
    """

    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        assert A.is_contiguous()
        assert B.is_contiguous()
        M, K = A.shape
        N, K2 = B.shape
        assert K == K2, f"Inner dim mismatch: A({K}) vs B({K2})"

        C = torch.empty((M, N), device=A.device, dtype=A.dtype)

        def grid(meta):
            grid_m = triton.cdiv(M, meta["BLOCK_M"])
            grid_n = triton.cdiv(N, meta["BLOCK_N"])
            return (grid_m * grid_n, )

        _a_bt_matmul_kernel[grid](
            A,
            B,
            C,
            M,
            N,
            K,
            A.stride(0),
            A.stride(1),
            B.stride(1),
            B.stride(0),  # transposed access
            C.stride(0),
            C.stride(1),
        )
        return C


def get_init_inputs():
    """Return init args for ModelNew."""
    return []


def unit_test(device="npu"):
    """Correctness test against torch reference."""
    torch.manual_seed(42)
    print("=" * 60)
    print("Unit Test: Matmul with Transposed B")
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
        A = torch.randn((M, K), device=device, dtype=torch.float16) * 0.1
        B = torch.randn((N, K), device=device, dtype=torch.float16) * 0.1

        with torch.no_grad():
            ref = torch.matmul(A, B.T)
            out = model(A, B)

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
