# Stage 2 performance model — WHY each ping-pong optimization pays

Evidence base: isolated A/B runs of the SAME flash-attention forward kernel
(`datasets/flash_attention/fa_forward_v2.py`, Ascend950PR_9579, triton-ascend
3.2.0). Only the named ingredient changed between runs and numerics stayed
bit-identical (15-case harness vs `torch_npu.npu_fusion_attention`), so the
attribution is solid. Numbers are profiler_avg ms.

Shapes: ID1 = B128 H8 N8192 D128 noncausal; ID3 = N1024 D128 noncausal;
ID4 = N1024 D64 noncausal; ID7 = N1024 D128 causal; ID8 = N1024 D64 causal.
Stage-1 (v1) baseline: 630 / 9.99 / 8.60 / 8.99 / 8.08 (ID1/3/4/7/8).

## Measured ingredient chain (ID1 / ID3 / ID4 / ID7 / ID8)

Benching practice that made this readable: the device degrades mid-run on
occasion (one light-chain run inflated even the v1 control 7x by the last
shape). Always read a variant against its IN-RUN control ratio (v2/v1), not
the absolute number, and re-run before trusting a single outlier — ID4/ID8
wobble ±18% run-to-run even on quiet days.

| Step | Result (ms) | Delta |
|---|---|---|
| Stage 1 (v1) | 630 / 9.99 / 8.60 / 8.99 / 8.08 | — |
| + ping-pong (textual loop-of-2, light 3-event chain; causal still Stage-1) | 473 / 7.01 / 6.42 / 9.01 / 8.09 | noncausal 1.33-1.43x; causal unchanged |
| + full slot-free ack handshake (6 events + prefree/postwait) | 550 / 8.39 / 7.78 / 9.75 / 8.92 | +16-21% on noncausal; causal slightly worse than v1 |
| + unified causal/noncausal loop (causal ping-pongs too) | 553 / 8.39 / 7.77 / 8.24 / 7.83 | causal −13%; noncausal unchanged |
| + row-wise outlined SIMD softmax | 579 / 8.65 / 8.05 / 5.89 / 5.55 | causal −29%; noncausal +3-5% |
| + outlined online update + per-slot carried scales | 552-564 / 8.15 / 7.43-8.78 / 5.60 / 5.22 | −4-8% on every shape |

## Mechanism per ingredient

1. **Ping-pong itself** — removes the serial Cube↔Vector alternation: Cube
   computes chunk i+1 while Vector processes chunk i. Pays when there are ≥2
   chunks per tile and BOTH sides have substantial work per chunk; deeper chunk
   streams win more (ID1: 128 chunks/tile → biggest win).
2. **Sync-edge count** — the full handshake costs +16-21% on noncausal streams
   vs the light chain. Every set/wait pair per chunk is a potential stall, and
   the producer's slot-free wait bounds the overlap depth by the consumer's ack
   latency. The light 3-event chain leans on in-order pipe transitivity (a later
   wait implies earlier reads already finished) and stalls less. Use the
   lightest protocol whose safety argument holds; explicit acks only when slot
   reuse is genuinely out-of-order.

   Writing the transitivity proof for a new kernel (required before dropping any
   ack): for EACH ping-ponged buffer, name the existing wait edge on the
   producer's path to overwriting a slot and show it implies the consumer's
   previous read of that slot already finished. FA example: qk_ub slot (i%2) is
   overwritten by QK(i+2), which follows PV(i+1)'s p-ready wait ⟹ vector passed
   SM(i+1) ⟹ read of qk(i) done; p_l1 slot overwritten after e0(i), and QK(i)
   follows PV(i-1) in cube order ⟹ PV(i-2) drained the slot; pv_ub slot
   overwritten after e4(i), and the vector's SM batch follows the previous
   pair's ACC batch ⟹ ACC(i-2) done. If NO existing edge on the overwrite path
   implies the read, that buffer keeps its explicit ack.
3. **One loop structure for all modes** — a constexpr mode split that routes one
   mode (causal) to a simpler loop forfeits the ENTIRE pipeline for that mode
   (9.75→8.24 the moment causal joined the ping-pong loop). Mode = loop bound +
   in-body mask, never a separate loop structure.
4. **Row-wise outlined SIMD vector stages** — per-row ops are exact 256B
   single-shots, exp is computed once per element, and the outlined VF is a
   clean AIV scheduling boundary. But 2×SUB_M extract/insert_slice ops per chunk
   add instruction overhead. Pays when the VECTOR side is the exposed bottleneck
   (shallow pipelines: causal ≈8 chunks/tile average, small N_CTX, vector-heavy
   stages); costs when Cube is the bottleneck and deep streams already hide the
   vector latency (long noncausal streams).
   **ATTRIBUTION A/B: `outline=True` is THE
   enabler, not a nicety.** Same row-loop code with `outline=True` removed from
   all three scopes: 15/15 bit-identical but ~6.6-7x SLOWER on EVERY shape
   (3666 vs 552, 57.4 vs 8.15, 56.8 vs ~8, 37.0 vs 5.60, 36.6 vs 5.22) — far
   slower than even the pre-row-loop kernel. Mechanism (traced in the
   bufferized IR, a5 passlog A/B): with outline=True the whole row loop lives
   INSIDE one AIV VF (`@_fwd_kernel_scope_N`) — ONE call per chunk, per-row
   subview+transfer_read/compute/transfer_write in a plain scf.for the compiler
   can software-pipeline across rows. Without it, the auto-outliner
   (PullSliceIntoVectorFunction) lifts each per-row BODY into its own
   `no_inline` VF: the main-kernel row loop then makes ~3 no_inline VF calls
   per row (~96 per chunk instead of 1), each a hard call boundary with
   call/return overhead and no cross-row pipelining. Loop-inside-function vs
   function-call-inside-loop. Whole-tile (v1-style) vector code never hits
   this: there are no per-row bodies to fragment, it auto-vectorizes in place.
   Never evaluate the row-wise form without outline=True.
   WHY the outlined row loop then beats the whole-tile form (v1) on
   vector-bound shapes (bufferized-IR counts, same chunk): v1's whole-tile
   softmax lowers to 13 SEPARATE auto-tiled row passes per chunk (43 static
   transfer ops × 32 rows ≈ 1376 UB transfer executions — every op its own UB
   round-trip pass). This is WITH AutoVectorizeV2 ON (v1's launcher never sets
   the flag; passlog shows hfusion-auto-vectorize-v2 ran before the counted
   stage): V2 fuses elementwise chains, but the softmax chain is punctuated by
   REDUCTIONS (tl.max/tl.sum over the tile), each reduction boundary breaks the
   chain, so each segment stays its own UB pass. The outlined row loop fuses
   across those boundaries BY CONSTRUCTION — a per-row reduction ([1,64]→[1])
   is a single-shot register op, so loop 1 = load+scale+mask+max in one row
   pass, loop 2 = load+sub+exp+sum+store in another: ~5 transfer ops/row ≈ 165
   executions per chunk, ~8x fewer UB round-trips. That saving is exposed on
   Vector-bound (causal, shallow) pipelines (−29-35%) and hidden behind Cube
   work on deep noncausal streams (only the slice-op overhead shows, +3-5%).
5. **Outlined per-row online update + loop-carried per-slot state** — the
   [SUB_M] stat math gets its own VF scheduling boundary, and keeping both slot
   scales live as pass-through args removes recompute/reshuffle at the consumer.
   −4-8% everywhere.

## Placement rules (correctness constraints that also decide where code lives)

- Only ≤256B single-shot vector ops inside an outlined VF; whole-tile f32 ops
  (>256B) stay at function level where the auto-tiler handles them
  (`Exceeds vector capacity` otherwise).
- Never nest an outlined VF call inside a runtime `if/else` (`'vector.mask' op
  body must bufferize in-place`); hoist the VF out and do the slot select on
  its small outputs afterwards.
- A name assigned in only one branch of a runtime `if` must exist before the
  `if` (pass-through arg) or the frontend cannot build the merge (NameError).

## Regime decision rule

Measure which side is exposed. Deep noncausal-style streams are Cube-bound →
keep whole-tile auto-tiled vector forms and minimize sync edges. Shallow or
vector-heavy pipelines (causal, small N_CTX, heavy epilogues) are Vector-bound →
row-wise outlined VFs pay up to ~1.4x on that stage.

(The Cube-bound/Vector-bound interpretation is inference from the measured arc,
consistent across all five shapes; cannsim pipe traces would confirm it
directly.)

## Open question — RESOLVED (measured 2026-08-07)

The light-chain + row-loop combination was tested (scratch
`fa_scratch/light_chain/`: current v2 structure minus events 2/6/10 and all
prefree/postwait; correctness 15/15 bit-identical — the transitivity proof
holds on device). Result is regime-dependent, normalized by the in-run v1
control to factor out device noise (v2/v1 ratio, lower = better):

| shape | handshake | light chain | verdict |
|---|---|---|---|
| ID1 N8192 noncausal | 0.85-0.86 | 0.82-0.83 | light chain slightly better |
| ID3 N1024 D128 noncausal | 0.82 | 0.76 (both runs) | light chain ~7% better |
| ID4 N1024 D64 noncausal | 0.86-1.02 | 0.80-0.96 | noisy, ~parity |
| ID7 N1024 D128 causal | 0.62 | 0.60-0.61 | parity |
| ID8 N1024 D64 causal | 0.65 | 1.21 / 1.96 | REAL ~2x regression, both runs |

So the ack handshake is pure overhead on the noncausal streams and D128 causal,
but one of the dropped edges is load-bearing for PERFORMANCE on D64 causal.
Prime suspect: e6 (p_l1-free) — without it the vector's UB→L1 P copy (MTE3) can
race the cube's L1 operand feed (MTE1) on the same slot, and the short D64
causal dots make the collision window dominate (hypothesis, not yet bisected;
next experiment: restore ONLY e6, then ONLY e10). ID1's 473 from the original
textual loop-of-2 remains unrecovered even by the light chain (522 best), so
part of the original gap is the batch-loop/helper packaging, not the protocol.

Practical rule refined: the lightest protocol whose safety argument holds is
the right DEFAULT, but a dropped ack edge can be load-bearing for performance
(not correctness) on some regimes — A/B every protocol change per shape before
adopting it.

## Later V2 optimization chain: persistent scopes, named state, and beta-free recurrence

Evidence below uses the preserved source sequence and same-run physical-NPU A/B
measurements at `BLOCK_M=BLOCK_N=128`. Local compiler probes used one identical
full-shape specialization (`Z=128,H=8,N_CTX=8192,D=64,noncausal`) for every
variant and dumped TTIR/TTAdapter with `TRITON_KERNEL_DUMP=1`. The toolkit
`bishengir-compile` wrapper was also run with `--mlir-print-ir-after-all`; its
A5 backend rejected the 910_95 dual-destination fixpipe late, but emitted all
IR through `InferHIVMMemScope`, which is sufficient for structural counts.

### Source chronology and measured effects

1. **Outer tile loop moved inside one persistent Cube scope and one persistent
   Vector scope.** The old source had one tile loop around a Cube/Vector scope
   pair. The new source duplicates the deterministic tile traversal inside the
   two scopes. TTIR changes from 9 to 10 `scf.for` ops: instead of re-entering
   both scopes per tile, Cube and Vector own independent long-lived streams and
   synchronize only through the existing buffers/events. This is the structural
   prerequisite for cross-tile overlap. The source move and initial hint edit
   were not preserved as separate files, so attribute only their combined
   measured gain unless a fresh no-hint A/B is run.

2. **Named alpha ping-pong state.** Exact syntax:
   `al.compile_hint(alpha_scale_a, "alpha_scale_a")` and likewise for B.
   Dedicated ID4 A/B: scope-loop/no-hint 4.765372 ms versus alpha hints
   4.733328 ms (0.67% lower). Normalized TTIR/TTAdapter are identical after
   removing the two `annotation.mark` ops; allocation/copy counts do not
   change. The final text changes, so this is a late backend allocation or
   scheduling perturbation, not arithmetic elimination. Beta hints produced
   no stable benefit and regressed the causal geometric mean; arbitrary names
   are not generally semantic compiler directives. Apply named hints only to
   measured loop-carried/ping-pong state and require a same-run A/B.

3. **Beta-free online recurrence.** P must be formed relative to the updated
   running maximum:

   ```text
   m_new = max(m_i, m_tile)
   alpha = exp(m_i - m_new)
   P = exp(QK - m_new)
   l_new = alpha*l_i + sum(P)
   numerator = alpha*numerator + P@V
   ```

   This removes `beta=exp(m_tile-m_new)`, `beta*l_tile`, both beta ping-pong
   vectors, and `beta*PV`. Full-shape compiler counts confirm the mechanism:

   | metric | alpha+beta | beta-free | delta |
   |---|---:|---:|---:|
   | TTIR `math.exp` | 4 | 3 | −1 |
   | TTIR `arith.mulf` | 8 | 5 | −3 |
   | TTAdapter reductions | 4 | 3 | −1 |
   | late BiShengIR `memref.alloc` | 51 | 45 | −6 |
   | late BiShengIR `hivm.hir.copy` | 23 | 19 | −4 |
   | late BiShengIR `hivm.hir.vbrc` | 10 | 8 | −2 |

   The reduced reduction count also comes from computing
   `max(maximum(row_lo,row_hi))` instead of
   `maximum(max(row_lo),max(row_hi))`: one elementwise max plus one horizontal
   reduction replaces two horizontal reductions plus a scalar max.

4. **Do not duplicate the running-max merge across outlined scopes.** A failed
   intermediate recomputed `maximum(m_i,m_ij)` after `m_ij` already contained
   `maximum(m_i,tmp_max)` and also retained an unused half-max reduction. It was
   correct but about 2.18x slower than the qualified V2. Removing only those
   redundancies changed the compiler's cross-scope materialization and made the
   beta-free kernel faster than alpha+beta on every canonical shape. Treat a
   full-vector op repeated across a no-inline boundary as a severe performance
   bug even when algebraically idempotent; inspect all producer/consumer scopes,
   not only source-level FLOP count.

5. **Keep the short recurrence in one no-inline scope.** Moving `alpha`,
   `l_new=alpha*l_i+l_ij`, and `m_i=m_ij` together into the online-update scope,
   while making row helpers return only `(m_ij,l_ij,P)`, improved every shape:
   1.0180x geometric mean (1.77% latency reduction). TTIR operation counts and
   late allocation/copy counts are unchanged; final npubin changes. Root cause:
   alpha no longer crosses the row-helper boundary, shortening the helper
   interface and its live range and giving the backend one explicit scheduling
   boundary for the recurrence. This is a lifetime/allocation win, not fewer
   operations.

6. **Produce consumer layout directly.** For P→NZ, allocate P in UB as
   `[BLOCK_N//16, SUB_M//16*16, 16]`, insert each row in that order, then use
   only `reshape(BLOCK_N//16,SUB_M//16,16,16)`. Do not build ND P and repair it
   with a later `permute`. The BN128 specialization is byte-identical before and
   after the BN64-only source cleanup, proving no all-128 performance claim can
   be attributed to this change. BN64 and BN128 correctness pass; benchmark
   BN64 separately before claiming a speedup.

### Why final V2 beats beta-free V1

Same-run all-shape geometric mean: V2 is 1.2512x faster than V1 (noncausal
1.0735x; causal 1.4583x). V2 does MORE static IR work: TTAdapter has 5 scopes,
10 loops, 12 sets + 12 waits, and 13 allocs versus V1's 2 scopes, 3 loops,
3 sets + 3 waits, and 6 allocs. Therefore the gain is not instruction-count
reduction. V1 uses one QK/P/PV slot and serial per-chunk Cube↔Vector handoff;
V2 ping-pongs all three handoffs and batches two chunks, allowing QK(i+1),
softmax(i), and PV/acc(i-1) to overlap. The larger causal gain is consistent
with Vector latency being exposed in shallow causal streams; V2's outlined
row VFs and overlap hide that latency, while deep noncausal streams are already
more Cube-bound.

### General decision rules

- First remove algebraic scale vectors/exponentials; then verify the new max
  reference preserves the online-softmax invariant.
- Count reductions, broadcasts, allocs, copies, and loop-carried values in
  TTIR/TTAdapter. Source FLOPs alone miss materialization costs.
- Put persistent traversal inside each independently scheduled core scope only
  when both scopes visit the same task order and every communication edge is
  explicit.
- Put a short loop-carried recurrence in one outlined/no-inline scope; minimize
  its inputs/outputs and never recompute idempotent full-vector state across it.
- Compile hints are backend perturbations unless a documented key is consumed;
  require same-run hardware A/B and inspect normalized IR before generalizing.
- Produce the consumer's physical layout at the producer. A reshape is free
  only when the producer allocation/insert order makes the collapsed dimensions
  contiguous.
- Compare full-shape specializations. Tiny one-tile probes can canonicalize
  persistent-loop variants into byte-identical binaries and hide the real
  scheduling delta.
