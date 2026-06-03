"""
Optimized square matrix multiplication for Ascend NPU (Triton-Ascend).

Final state: v1 (5.5x speedup vs baseline). The episode-42 v2 patterns
(multibuffer + care_padding) were tested at sub-kernel scale and did NOT
help this specific kernel — see Optimizations.md and performance_report.md
for the data. v1 is the local optimum here.

v1 optimizations applied (from episode 41, l1_1 reference):
  1. Shape-specialized exact kernel for N=4096 (mask-free loads,
     BLOCK=128x128x32).
  2. 1D grid of 1024 programs (32×32 tiles) with GROUP_M=4 pid swizzle
     for L2 cache locality.
  3. tl.static_range K loop unrolling (EXACT_K constexpr → 128 iters
     fully unrolled at compile time, no loop-counter overhead).
  4. al.compile_hint("dot_pad_only_k") on both A and B tiles (Cube only
     pads the K dim; M=128, N=128 are already 16-aligned).
  5. Generic fallback for other shapes (pre-hoisted row/col masks, same
     block sizes, dot_pad_only_k).

v2 patterns tested (from episode 42, l1_2 reference) — NOT adopted:
  - al.multibuffer(a, size=2) + al.multibuffer(b, size=2)
  - care_padding=False on exact-kernel loads
  - tl.multiple_of / tl.max_contiguous alignment hints

Why v2 didn't help: l1_2's v1 had *dynamic* K range; v2 added static_range
+ multibuffer together for a 9.2x win. l1_1's v1 already has static_range
(episode 41), so the multibuffer has nothing to hide. Sub-kernel evidence:
  - K=2 iters: v2 14629 cyc vs v1 14710 cyc → -0.6% (within noise)
  - K=4 iters: v2 24595 cyc vs v1 23510 cyc → +4.6% REGRESSION
    (MTE2 busy_cyc 3548→15602 from multibuffer's extra sync overhead)
"""

import os

import torch
import torch.nn as nn

import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

# ── Tuned constants for 4096×4096 benchmark shape ────────────────────────────
EXACT_N     = 4096
EXACT_BM    = 128
EXACT_BN    = 128
EXACT_BK    = 32
EXACT_GROUP = 4


# ── Exact (mask-free) kernel ─────────────────────────────────────────────────
@triton.jit
def _matmul_kernel_exact(
    a_ptr, b_ptr, c_ptr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    NUM_PID_M: tl.constexpr,
    NUM_PID_N: tl.constexpr,
    EXACT_K:   tl.constexpr,   # = EXACT_N (4096), constexpr → tl.static_range unrolls
    BLOCK_M:   tl.constexpr,
    BLOCK_N:   tl.constexpr,
    BLOCK_K:   tl.constexpr,
    GROUP_M:   tl.constexpr,
):
    # 1D grid with GROUP_M pid swizzle for L2 cache locality
    pid = tl.program_id(0)
    group_width  = GROUP_M * NUM_PID_N
    group_id     = pid // group_width
    first_pid_m  = group_id * GROUP_M
    group_size_m = tl.minimum(NUM_PID_M - first_pid_m, GROUP_M)
    pid_in_group = pid % group_width
    pid_m        = first_pid_m + (pid_in_group % group_size_m)
    pid_n        = pid_in_group // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # tl.static_range: K loop fully unrolled at compile time (no loop-counter
    # overhead, bishengir can pipeline across K iterations).
    NUM_K_ITERS: tl.constexpr = EXACT_K // BLOCK_K
    for _ in tl.static_range(0, NUM_K_ITERS):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        # dot_pad_only_k: M/N are 128-aligned → only K dimension needs padding
        # to cube granularity. Saves ~30-50% UB vs padding all three dims.
        al.compile_hint(a, "dot_pad_only_k")
        al.compile_hint(b, "dot_pad_only_k")
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc)   # output is FP32 (acc already FP32)


# ── Generic (masked) kernel for other shapes ─────────────────────────────────
@triton.jit
def _matmul_kernel_generic(
    a_ptr, b_ptr, c_ptr,
    m, n, k,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Hoist row/col masks out of the K loop — they don't change per iteration.
    m_mask = offs_m < m
    n_mask = offs_n < n

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k_start in range(0, k, BLOCK_K):
        k_mask = (k_start + offs_k) < k
        a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        al.compile_hint(a, "dot_pad_only_k")
        al.compile_hint(b, "dot_pad_only_k")
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


# ── Host helpers ──────────────────────────────────────────────────────────────

def _require_supported_runtime(tensor: torch.Tensor) -> None:
    if tensor.is_cuda:
        return
    if tensor.device.type == "npu":
        return
    if os.environ.get("TRITON_INTERPRET") == "1":
        return
    raise RuntimeError(
        "This operator requires CUDA or NPU tensors, or TRITON_INTERPRET=1."
    )


def _validate_inputs(a: torch.Tensor, b: torch.Tensor):
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("ModelNew expects two 2D tensors.")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Incompatible shapes: {tuple(a.shape)} @ {tuple(b.shape)}")
    if a.shape[0] != a.shape[1] or b.shape[0] != b.shape[1]:
        raise ValueError("This operator is defined for square matrix inputs.")
    if a.device != b.device:
        raise ValueError("Inputs must be on the same device.")
    if a.dtype != b.dtype:
        raise ValueError("Inputs must have the same dtype.")
    if a.dtype not in {torch.float16, torch.float32}:
        raise TypeError(f"Unsupported dtype: {a.dtype}")
    _require_supported_runtime(a)
    return a.contiguous(), b.contiguous()


def _triton_square_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, b = _validate_inputs(a, b)
    m, k = a.shape
    _, n = b.shape

    # Output in FP32 (accumulator precision); cast back to input dtype at the end
    c = torch.empty((m, n), device=a.device, dtype=torch.float32)

    if m == EXACT_N and n == EXACT_N and k == EXACT_N:
        # ── Fast path: benchmark shape 4096×4096 ──────────────────────────────
        num_pid_m = EXACT_N // EXACT_BM    # 32
        num_pid_n = EXACT_N // EXACT_BN    # 32
        total_programs = num_pid_m * num_pid_n  # 1024
        grid = (total_programs,)
        _matmul_kernel_exact[grid](
            a, b, c,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            NUM_PID_M = num_pid_m,
            NUM_PID_N = num_pid_n,
            EXACT_K   = EXACT_N,
            BLOCK_M   = EXACT_BM,
            BLOCK_N   = EXACT_BN,
            BLOCK_K   = EXACT_BK,
            GROUP_M   = EXACT_GROUP,
        )
    else:
        # ── Generic path: any square matrix ──────────────────────────────────
        BM = min(128, m)
        BN = min(128, n)
        BK = 32
        grid = (triton.cdiv(m, BM), triton.cdiv(n, BN))
        _matmul_kernel_generic[grid](
            a, b, c,
            m, n, k,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        )

    return c.to(dtype=a.dtype)


# ── Module interface ──────────────────────────────────────────────────────────

class ModelNew(nn.Module):
    """
    Optimized square matrix multiplication (C = A * B) for Ascend NPU.

    Key improvements over baseline (5.5× measured on 4096×4096 fp32):
      - Exact kernel for 4096×4096: mask-free, larger blocks (128×128×32),
        GROUP_M=4 pid swizzle (L2 reuse), tl.static_range K unroll,
        al.compile_hint(dot_pad_only_k).
      - Generic kernel: pre-hoisted masks, al.compile_hint, 128×128 blocks.
    """

    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Args:
            A: Input matrix of shape (N, N), float16 or float32, on NPU.
            B: Input matrix of shape (N, N), same dtype and device as A.
        Returns:
            C: Output matrix of shape (N, N), same dtype as A.
        """
        return _triton_square_matmul(A, B)


# ── Benchmark helpers ─────────────────────────────────────────────────────────
N = 4096


def get_inputs():
    device = "npu" if hasattr(torch, "npu") and torch.npu.is_available() else "cpu"
    A = torch.rand(N, N, device=device)
    B = torch.rand(N, N, device=device)
    return [A, B]


def get_init_inputs():
    return []
