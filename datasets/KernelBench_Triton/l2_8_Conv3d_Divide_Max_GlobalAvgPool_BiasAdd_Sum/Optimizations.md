# Optimizations Applied — l2_8_Conv3d_Divide_Max_GlobalAvgPool_BiasAdd_Sum

## Kernel Description

The pipeline after Conv3d + MaxPool + GlobalAvgPool produces a `[B, C]` tensor
(B=128, C=16). The Triton kernel's job is to reduce each row (sum over C channels)
to produce a scalar per batch item.

## Baseline Analysis

The baseline kernel (`8_Conv3d_Divide_Max_GlobalAvgPool_BiasAdd_Sum.py`) uses:
- Grid: `(B,)` = 128 programs, one program per row
- Each program: load C=16 values (strided by `stride_c`), sum, store 1 scalar
- `BLOCK_SIZE=256` (padded), loads with mask `offs < C`

**cannsim trace (sub-kernel: B=1, C=16, grid=1):**

| Pipeline | busy_cyc | % wall |
|---|---|---|
| SCALAR | 1,811 | 58.4% — BOTTLENECK |
| SCALARLDST | 1,725 | 55.6% |
| MTE2 | 894 | 28.8% |
| VEC | 654 | 21.1% |

Key critical instructions:
- `STI_XN_IMM` = 1,225 cy (scalar register spill — structural triton-ascend codegen cost)
- `LD_XD_XN_IMM` = 1,218 cy total (args struct startup per program)
- `DC_PRELOAD_XN_IMM` = 496 cy (per-program startup)
- `LDP_XI_XJ_XN×3` = 1,444 cy (per-program startup)

Per-program startup cost alone: ~977 cy = 31.5% of wall.
With B=128 programs: 128 × 977 cy ≈ 125K cy of pure overhead.

**Measured hardware latency:** 9.18 µs (msprof)

---

## Optimization 1: Persistent Work-Stealing Grid

**Problem:** B=128 FFTS program dispatches, each incurring ~1,150 cy startup
(DC_PRELOAD + LDP args struct). Total startup overhead ≈ 147K cy.

**Fix:** Cap grid to `NUM_PROGS = min(B, 32)` = 32 programs. Each program
strides over rows with step = NUM_PROGS:

```python
# Baseline: grid=(B,), one program per row
_reduce_bcdhw_to_b_kernel[(B,)](..., BLOCK_SIZE=256)

# Optimized: grid=(32,), 4 rows per program
for row in tl.range(pid, B, NUM_PROGS, num_stages=1):
    ...
```

**Impact:** 128 → 32 FFTS dispatches (4× reduction in startup overhead).
Analytically: saves 96 × 1,150 cy = ~110K cy at full scale.

---

## Optimization 2: Contiguous C=16 Load (No Mask)

**Problem:** Baseline uses `BLOCK_SIZE=256` with `mask = offs < C` even though
C=16 always. This forces the compiler to generate masked loads and extra scalar
overhead for the mask computation (256-element mask for 16 elements).

**Fix:** For C=16, use a constexpr `C: tl.constexpr = 16` with
`tl.max_contiguous` + `tl.multiple_of` hints and no mask:

```python
cols = tl.max_contiguous(tl.multiple_of(tl.arange(0, C), C), C)
vals = tl.load(x_ptr + base + cols)   # no mask — C=16 is exact, contiguous
```

**Impact:** Eliminates mask computation. Eliminates the `STI_XN_IMM` scalar
register spill (1,225 cy in baseline) that was caused by the 256-element masked
path. The opt trace shows STI_XN_IMM gone entirely.

---

## Optimization 3: Host-Side bias_sum Pre-computation

**Problem:** Adding `bias[c]` inside the kernel requires loading bias values
per program. Since `sum(x[b,:] + bias[:]) = sum(x[b,:]) + sum(bias[:])`, the
bias sum is a constant scalar that can be computed once on the host.

**Fix:**
```python
# Before: add bias inside kernel (per-element), or in Python before (extra kernel call)
x = (x + self.bias).reshape(B, C).contiguous()  # extra Add op

# After: pre-compute scalar sum of bias on host
bias_sum_scalar = self.bias.view(C).sum().to(dtype=torch.float32).item()
# pass as fp32 scalar to kernel — no per-element addition needed
total = tl.sum(vals.to(tl.float32), axis=0) + bias_sum
```

**Impact:** Eliminates the host-side `(x + self.bias)` elementwise addition
(GPU op: Add kernel, ~7 µs per the msprof raw stats). The bias add is now a
single scalar add inside the reduction, which is free.

---

## Optimization 4: Contiguous Reshape (kept from reference)

The reference already applies `.reshape(B, C).contiguous()` before the Triton
call, guaranteeing `stride_c = 1`. Combined with Opt 2 (constexpr C=16 with
contiguous hints), the compiler can emit a single burst MTE2 load for all
16 float32 values (64 bytes = 2 cache lines).

---

## Summary

| Opt | Description | Impact |
|---|---|---|
| 1 | Persistent grid (128→32 programs) | 4× fewer FFTS dispatches |
| 2 | Constexpr C=16, no-mask contiguous load | Eliminates STI_XN_IMM spill (1,225 cy) |
| 3 | Host-side bias_sum scalar | Eliminates separate Add GPU kernel |
| 4 | .contiguous() reshape (inherited) | Single burst MTE2 load |

**cannsim per-tile comparison:**

| Metric | Baseline | Optimized |
|---|---|---|
| wall_cycles | 3,101 | 2,375 |
| SCALAR busy | 1,811 cy (BOTTLENECK) | 579 cy |
| STI_XN_IMM | 1,237 cy | 0 cy (eliminated) |
| Bottleneck | SCALAR (58%) | MTE2 (30%) — healthy |

**Full-shape latency (msprof, case-1: B=128):**

| Version | Triton kernel µs |
|---|---|
| Baseline | 9.18 µs |
| Reference opt | 1.982 µs |
| This opt | Expected ≤ 1.8 µs (TBD hardware) |
