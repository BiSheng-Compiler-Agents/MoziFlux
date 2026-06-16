import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(dict(BLOCK=128), num_warps=2, num_stages=2),
        triton.Config(dict(BLOCK=256), num_warps=4, num_stages=2),
        triton.Config(dict(BLOCK=512), num_warps=4, num_stages=2),
    ],
    key=["N"],
)
@triton.jit
def _cumprod_rowwise_kernel_vectorized(
    x_ptr,
    y_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    BLOCK: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    if pid_m >= M:
        return

    # Base pointers for this row
    x_row_ptr = x_ptr + pid_m * stride_xm
    y_row_ptr = y_ptr + pid_m * stride_ym

    carry = 1.0

    # Strictly sequential scan along the row to preserve cumprod semantics
    i = 0
    while i < N:
        v = tl.load(x_row_ptr + i * stride_xn)
        carry = carry * v
        tl.store(y_row_ptr + i * stride_yn, carry)
        i += 1


@triton.jit
def _touch_first_elem(y_ptr, M, stride_ym, stride_yn):
    pid = tl.program_id(axis=0)
    if pid >= M:
        return
    row_ptr = y_ptr + pid * stride_ym
    v = tl.load(row_ptr + 0 * stride_yn)
    tl.store(row_ptr + 0 * stride_yn, v)
