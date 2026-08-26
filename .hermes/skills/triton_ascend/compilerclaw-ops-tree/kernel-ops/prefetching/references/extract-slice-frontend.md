# extract_slice / insert_slice frontend landmines (triton-ascend 3.2.0)

Locally probed (no NPU, ASTSource + GPUTarget compile) with `extract_slice_probe.py`
variants kA/kB/kC; error locations in the installed package
`triton/language/extra/cann/extension/vec_ops.py`.

## constexpr rules: SIZES vs OFFSETS

| Construct | Result |
|---|---|
| bare local `HALF = BLOCK_N // 2` used in SIZES | FAIL: `ValueError("_builder argument must be provided")` (vec_ops.py:115 sizes assert — the local does not stay constexpr) |
| `HALF: tl.constexpr = BLOCK_N // 2` in SIZES | OK — the annotation forces constexpr evaluation |
| constexpr value in the OFFSETS tuple | FAIL: positional-builder rejection in `semantic.to_tensor(o, _builder)` (vec_ops.py:133) |
| runtime tensor offset, e.g. `off_hi = (i * 0) + HALF` | OK |
| loop var `i` (tensor) and int literal `0` in OFFSETS | OK |

Rule of thumb: annotate `: tl.constexpr` for anything used in SIZES; keep OFFSETS
runtime-typed (tensor or int literal), never a constexpr-annotated local.

## 256B single-shot rule extends to mask-compare vectors

Inside a manual row loop, an i64 comparison vector of 64 lanes = 512B > 256B single-shot
vector capacity. Full-tile i64 masks are auto-split by the compiler; row-level ones are not.

Symptom: `Exceeds vector capacity` (region: hfusion-legalize-bool) reported at the mask
line, followed by a cascaded bogus `'hivm.hir.fixpipe' op if dual_dst_mode is enabled...`
error — the fixpipe message is NOT the root cause, do not chase it.

Trigger: index scalars derived from `al.sub_vec_id()` (returns i64) flowing into row-level
compares, e.g. `m_base = sub_vec_id() * SUB_M + ...` then `col_idx <= m_base + i`.

Fix: cast the index scalar once — `m_base = (...).to(tl.int32)` — before the compare.

## Where the 256B rule is enforced: outlined VFs only

The auto-split behavior depends on where the row op lives:

- **Inside an outlined VF** (`al.scope(outline=True)`): NO auto-tiling — every op
  must be ≤256B or compilation fails. A plain `[1,128]` f32 (512B) row op fails with
  `error: Exceeds vector capacity` at the row-op line (plus the bogus fixpipe cascade,
  variant text: "the data movement must be performed from L0C to UB!"). Verified on the
  real 950 toolchain with the full FA kernel (bn128 row loop without the two-half split).
  → the manual two-half split for BLOCK_N=128 is mandatory ONLY because the row loop is
  outlined.
- **At function level** (plain, non-outlined row loop inside the vector scope): the
  compiler DOES split oversized row ops itself — a 512B-row extract_slice loop passed
  the full MLIR pipeline (local a5 compile) with no capacity error.

So "the compiler would tile it anyway" is true at function level and false inside
`outline=True` VFs. Choosing outline=True means taking over ALL tiling ≥256B manually.

## Outlined row-wise softmax pattern

`_softmax` dispatches on `BLOCK_N == 128` to `_softmax_rows_bn128`, else
`_softmax_rows_bn64`. Each helper body wraps its work in
`with al.scope(vector_mode="simd", outline=True):` (outline=True is the one scope kwarg
with real effect — dedicated outlined AIV `hivm.vector_function`):

- loop 1 over SUB_M rows: extract_slice one row → `tl.max` → insert_slice into m_ij;
- loop 2 over SUB_M rows: l_ij + unnormalized P rows;
- bn128: every row op done twice, on two 64-wide (256B f32) halves via extract/insert_slice.

Numerics bit-identical to the full-tile vectorized online-softmax form on every harness
shape. Perf (profiler_avg, fa_forward_v2): causal 8.24→5.89 ms (N1024 D128) and
7.83→5.55 ms (N1024 D64); noncausal ~3-5% slower (8.39→8.65, 7.77→8.05, 553→579).

## Verifying a BLOCK_N=128 dispatch path when the harness only launches BLOCK_N=64

Copy the kernel to the scratch dir with BLOCK_M=BLOCK_N=128 and check a realistic shape
(B8 H8 N1024 D64) directly against `torch_npu.npu_fusion_attention`
(causal: `sparse_mode=1` + explicit `~tril` mask). Reference numbers (fa_forward_v2,
2026-08-05): noncausal max_abs 2.44e-04 / mean 1.14e-05; causal 1.95e-03 / 2.02e-05 —
same tolerance band as the bn64 harness PASS thresholds.
