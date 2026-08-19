# Outlined `al.scope` restrictions (device-verified 2026-08-06)

Context: fa_forward_v2 online-softmax restructure — moving the running-stat update
into `al.scope(vector_mode="simd", outline=True)` scopes. Three traps, all reproduced
on the Ascend950PR_9579 toolchain (triton-ascend 3.2.0).

## 1. Outlined scope inside a runtime `if/else` fails bufferization

Structure that fails:

```python
if (sid % 2) == 0:
    with al.scope(vector_mode="simd", outline=True):
        ... assigns p, m_new, l_new ...
else:
    with al.scope(vector_mode="simd", outline=True):
        ... same assigns ...
```

Backend error (buildFinalHIVMPipelines / BiShengHIR pipeline):

```text
error: 'vector.mask' op body must bufferize in-place
```

Root cause: the outlined body's tensor live-outs become scf.if branch results, and
vector-function bufferization requires in-place bodies — the if-yielded tensors are
rejected. Fix: hoist the outlined scope OUT of the runtime if (one unconditional
scope); keep only small per-slot scalar / `[SUB_M]`-vector selects inside the if.

## 2. Whole-tile ops inside an outlined VF: `Exceeds vector capacity`

`p = p * (beta / l_new)[:, None]` on `[SUB_M, BLOCK_N]` f32 (8 KB) inside an outlined
scope → `error: Exceeds vector capacity` (loc attributed to a line inside the scope
body). The SAME whole-tile op at plain function level inside the big vector scope
compiles fine — the auto-tiler splits it there. Inside an outlined vector function
there is NO auto-tiling rescue: every op must fit the 256B single-shot rule on its
own (per-row `[1, BLOCK_N<=64]` f32, or `[SUB_M]` vectors). Keep whole-tile
elementwise at function level; compute only per-row/per-vector stats in the outlined
scope.

## 3. Runtime-if single-branch assignment (frontend, not Ascend-specific)

A name assigned in only ONE branch of a runtime `if` and read after it raises
`NameError('<name> is not defined')` at the use site: the frontend builds the scf.if
live-out merge only for names defined before the `if` or in BOTH branches. Fix for
ping-pong slot state: pass both slots' values in as loop-carried args and return both
(pass-through), so each branch may update just one.

## Working structure (15/15 harness PASS, numerics bit-identical)

```python
with al.scope(vector_mode="simd", outline=True, no_inline=True):
    m_new = tl.maximum(m_i, m_ij)
    alpha = tl.exp(m_i - m_new)
    beta = tl.exp(m_ij - m_new)
    l_new = alpha * l_i + beta * l_ij
    acc_scale = l_i / l_new * alpha
    p_scale = beta / l_new
p = p * p_scale[:, None]                 # whole-tile: stays at function level
if (sid % 2) == 0:
    acc_scale_a = acc_scale              # slot select: args pre-exist, merge legal
else:
    acc_scale_b = acc_scale
```

Perf effect of outlining the update + per-slot scales (profiler_avg, vs the
function-level update): -4.6% to -7.7% on all five FA shapes (ID1 579->552 ms,
ID3 8.65->8.15, ID4 8.05->7.43, causal ID7 5.89->5.60, ID8 5.55->5.22).

## Local compile note (950 dual_dst kernels)

Local full-pipeline repro of a kernel using fixpipe `dual_dst_mode` needs
`GPUTarget("npu", "Ascend950PR_9579", ...)`: with a 910_95 target the in-triton
pipeline stops early at `'hivm.hir.fixpipe' op The current hardware does not support
dual_dst_mode is enabled`. Keep `TRITON_ASCEND_ARCH` on a whitelisted 910 value
(`libdevice.py` rejects 950 strings at import). The downstream
ConvertLinalgRToBinary step shells out to `bishengir-compile-a5`; if that binary is
not on PATH the local run stops there ("Cannot find bishengir-compile-a5 under
$PATH") — everything before it (frontend checks, ttir/ttadapter dumps) still works.
