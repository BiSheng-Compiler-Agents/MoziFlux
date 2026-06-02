# AIV Hardware Findings from Cannsim Traces

Extracted from actual trace_core0.json analysis on Ascend 910 / AIV core.
All claims are empirically verified by cannsim, not inferred from docs.

---

## fp16 vs fp32 for `tl.maximum` (ReLU)

**Problem:** `tl.maximum(x.to(tl.float16), 0.0)` → routes to **VEC** fixed-function unit.
The VEC unit is slow, generates WAIT_FLAG stalls, and causes MTE3 (store DMA)
to be blocked while VEC drains. At BLOCK_HW=1024, VEC constituted 35% of all cycles.

**Fix:** cast to fp32 before the maximum, cast back after:
```python
x_f32 = x.to(tl.float32)
y_f32 = tl.maximum(x_f32 + b, 0.0)
y = y_f32.to(tl.float16)
```
This routes to **RVECEX** which pipelines with MTE2/MTE3 and doesn't stall.

**Speedup observed:** 7,132 → 4,838 cycles (1.47×) for the Conv2D ReLU BiasAdd kernel.

---

## `tl.zeros` inside `tl.range` loop

**Problem:** `acc = tl.zeros(...)` inside a `for k in range(...)` loop generates
a **VEC zeroing op on every iteration**. This produces one WAIT_FLAG_VEC per
outer loop step, serializing memory and compute.

**Fix:** Hoist accumulator initialization before the loop:
```python
acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)  # hoist outside
for k in tl.range(K, ...):
    a = tl.load(...)
    b = tl.load(...)
    acc += tl.dot(a, b)
```

**Observable in trace:** Look for `WAIT_FLAG_VEC` events that correlate with
loop iteration boundaries (not just at the end).

---

## UB (Unified Buffer) size limits for fp32

The AIV Unified Buffer is approximately **32KB** per core.

For a kernel with 3 live fp32 tensors of tile size BLOCK_HW:
```
3 * BLOCK_HW * 4 bytes ≤ ~32,768 bytes
BLOCK_HW ≤ 2730 → safe max power-of-2 is 2048
```

**BLOCK_HW=4096 causes silent UB overflow:**
- Kernel compiles and runs without error
- `instr.bin` is ~800KB (nearly empty) vs 391MB for 2048
- `trace_core0.json` will have no meaningful vector instructions recorded
- Cycle counts look suspiciously low
- The kernel is silently reading/writing garbage memory

**Verification:** After simulation, check `instr.bin` size. <1MB = overflow or crash.

---

## `num_stages=1` crash on AIV

`num_stages=1` causes a scalar div-by-zero crash inside the AIV core at runtime.
The binary is truncated to ~640KB. The mechanism is unknown but consistent.

**Rule:** Always use `num_stages=2` (or higher if you want software pipelining).
Never use `num_stages=1`. This applies regardless of the kernel type.

---

## Startup cost vs compute cost (small-HW regime)

For small spatial tiles (HW ≤ ~1024 elements), the per-program AIV startup
cost (~1,150 cycles) can dominate total runtime, dwarfing actual compute.

Example: Conv2D ReLU BiasAdd with HW=196 (14×14 feature map):
- 252 programs for the 2D grid approach → startup cost = 252 * 1150 ≈ 290,000 cy
- Actual vector work = tiny

**Fix: persistent kernel** — launch 32 programs (constant) that each loop
over many (N*C / 32) spatial tiles. Startup cost = 32 * 1150 ≈ 37K cy.

```python
@triton.jit
def _relu_bias_persistent(x_ptr, y_ptr, b_ptr, N_TILES, HW, BLOCK_HW: tl.constexpr):
    pid = tl.program_id(0)
    total = tl.num_programs(0)  # = 32
    for i in tl.range(pid, N_TILES, total):  # stride over all tiles
        ...
```

**Threshold empirically determined:** HW ≤ 1024 → persistent; HW > 1024 → loop kernel.

---

## SCALAR overhead from grid index decomposition

The baseline 2D grid `(N*C*H, cdiv(W, BLOCK_W))` requires computing `row = pid % (C*H)`
and `col = pid // (C*H)` at runtime. This is 2 DIV + 2 REM operations per program.

In the trace this shows as:
- `SCALAR` ≈ 24% of cycles (integer arithmetic on the scalar ALU)
- `SCALARLDST` ≈ 23% of cycles (scalar register loads from the address computation)

**Fix:** flatten to 1D grid `(N*C,)` and precompute the batch/channel index:
```python
pid = tl.program_id(0)   # indexes into N*C
c = pid % C
n = pid // C
hw_offset = (n * C + c) * HW
```
This eliminates the per-program DIV/REM and drops SCALAR+SCALARLDST overhead.

---

## Cannsim report command (CANN 9.0.0)

The correct form uses `-e <exp_dir>` (experiment directory from `cannsim record`) — NOT the
old `-i instr.bin -d log_ca/` flags, which were removed in CANN 9.0.0.

```bash
# 1. Record (do NOT use -o flag — cannsim auto-creates cannsim_<ts>_<binary>/ in CWD)
cd <job_dir>
cannsim record <binary> -s Ascend950

# 2. Report (use -e with the timestamped experiment dir)
cannsim report -e ./cannsim_<timestamp>_<binary> \
               -o ./cannsim_<timestamp>_<binary>/report \
               -n 0
# Produces: ./cannsim_<timestamp>_<binary>/report/trace_core0.json
```

- `-e <exp_dir>` — experiment directory auto-created by `cannsim record` (contains instr.bin + log_ca/)
- `-o <out_dir>` — output directory for trace_core0.json; must be inside or alongside exp_dir
- `-n 0` — core 0 (the AIV core); try 0–5 if core 0 shows only SCALAR work
- **No `--timeline` flag** — it does not exist in CANN 9.0.0; argparse rejects it

The `cannsim-remote` plugin handles all of this automatically via `cannsim_remote_run(gen_report=True)`.

Key fields in the Chrome-trace JSON (analysed via aggregate_trace.py):
- Wall cycles: `t_end - t_start` across all X-events
- BOTTLENECK: pipeline with highest `busy_cyc`
- CRITICAL: instruction with highest `total_cyc` OR `avg_cyc >= 0.25 * wall_cycles`
- SCALAR high → too much index math / two-pass / Python accumulators
- MTE2 high → memory-bound; increase contiguity or BLOCK
- VEC high → fixed-function vector (slow); fp16 tl.maximum → upcast to fp32 for RVECEX
- WAIT_FLAG_VEC >10% → serialisation between VEC and MTE units (e.g. tl.zeros inside loop)
