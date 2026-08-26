# Ascend NPU Hardware Architecture

## Core Architecture

### AI Core Composition

For the A5 / Ascend950PR programming model used by Triton-Ascend, an AI Core has:

| Component | Count | Function | Directly read/write memories |
|-----------|-------|----------|------------------------------|
| **AIC / Cube core** | 1 | Matrix/Cube computation (`tl.dot`, matmul/conv-like work) | L0A, L0B, L0C, BT, FP |
| **AIV / Vector core** | 2 | Vector computation (elementwise, reductions, softmax, activation, stores) | UB |
| **Scalar control unit** | 1 per AIC and 1 per AIV | Control flow, address/control scalar work | Controls its owning core |

Each AIC and AIV has its own scalar unit for control. For CV-hybrid operators, the Cube/AIC and
Vector/AIV sides run as separate hardware components and must be synchronized explicitly when data
crosses from one side to the other.

### Core Type Selection

| Operator Type | Core to Use | Method to Get Core Count |
|---------------|-------------|--------------------------|
| Pure vector computation (elementwise, reduction, activation) | AIV / Vector Core | `get_npu_vectorcore_num()` |
| Matrix multiplication (`tl.dot`) | AIC / Cube Core | `get_npu_aicore_num()` |
| CV hybrid operators (attention, matmul + softmax/activation) | AIC + AIV | `get_npu_aicore_num()` for block scheduling; use sub-vector tiling inside each AI Core |

**Cube Core matrix throughput is far higher than Vector Core element-wise computation.** Matrix
multiplication operators (including GEMV, i.e. GEMM with N=1) **must** use `tl.dot` + AIC/Cube.
Degrading matrix multiplication to Vector Core element-wise multiply-add is a severe performance
anti-pattern. Even when N=1, pad to the Cube granularity and use `tl.dot`.

```python
import torch
import triton.runtime.driver as driver


def get_npu_aicore_num():
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)["num_aicore"]


def get_npu_vectorcore_num():
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)["num_vectorcore"]
```

---

## Memory Hierarchy

### Memories

The relevant memory/storage levels for A5 / Ascend950PR Triton-Ascend kernels are:

| Memory | Role | Main direct user |
|--------|------|------------------|
| **GM** | Global/device memory, large capacity, high latency | Host-visible tensors; source/sink for kernel inputs/outputs |
| **L2** | Shared on-chip cache/intermediate level between GM and AI Core local memories | MTE2/MTE3/fixpipe transfer path |
| **L1** | AIC-side local memory / staging buffer | Cube-side staging; source/destination for MTE/fixpipe paths |
| **UB** | Unified Buffer, main local memory for vector computation | AIV directly reads/writes UB |
| **L0A** | Cube A-operand local buffer | AIC directly reads/writes |
| **L0B** | Cube B-operand local buffer | AIC directly reads/writes |
| **L0C** | Cube accumulator / matmul output storage | AIC writes matmul outputs; fixpipe drains to L2/UB |
| **BT** | Bias Table | AIC directly reads/writes |
| **FP** | Fixpipe destination/intermediate storage | AIC/fixpipe data path |

---

## Data Movement Engines and Flow

### MTE / Fixpipe Transfer Semantics

| Engine | Direction / writes | Typical use |
|--------|--------------------|-------------|
| **MTE1** | Writes **from L1 to L0A, L0B, BT**; also **from L1 to UB** (A5 only) | Stage A/B operands and bias-table data for Cube use; on A5 also feed Vector |
| **MTE2** | Writes **from L2 to L1 and UB** | Bring global/shared data into AI Core local memories |
| **MTE3** | Writes **from L1 to L2** and **from UB to L2**; also **from UB to L1** (A5 only) | Store Vector results, move local data back toward global/shared memory |
| **Fixpipe** | Writes **from L1 to FP**, **from L0C to L2 and L1**, and **from L0C to UB** (A5 only) | Drain Cube accumulator/output, format/convert output, hand Cube result to Vector or memory |

**Platform note — 910B/A2 vs A5 differences:** on Ascend 910B/A2, everything above is the same
as A5 **except**:

1. **Fixpipe cannot write directly from L0C to UB.** The Cube accumulator must be drained through
   L2/L1 (e.g. L0C → L2, then MTE2 into UB).
2. **L1 cannot read or write from UB.** MTE1 L1→UB and MTE3 UB→L1 do not exist on A2. The only
   L1↔UB path on A2 goes through L2: L1 →(MTE3) L2 →(MTE2) UB, or UB →(MTE3) L2 →(MTE2) L1.

On A5 / Ascend950PR, both restrictions are lifted: L0C → UB via Fixpipe and direct L1 ↔ UB
transfers via MTE1/MTE3 are supported, making Cube→Vector and Vector→Cube handoffs cheaper.
These are the key architectural enablers for the prefetching/preloading CV pipeline pattern.

### Vector Path

```text
   input                    compute            output

                        ┌──────────┐
  L2 ──MTE2───────────► │          │ ──MTE3──► L2 (all platforms) ──► GM
                        │          │
  L1 ──MTE1───────────► │    UB    │ ──MTE3──► L1 (A5 only)
   (A5 only)            │          │
                        └──────────┘
                        AIV0 / AIV1
                        (+ scalar each)
```

Key point: **UB is the main memory for Vector cores**. AIV cores directly read/write UB.
Inputs arrive via `L2 → UB` (MTE2, all platforms) or `L1 → UB` (MTE1, A5 only). Outputs leave
via `UB → L2` (MTE3, all platforms) or `UB → L1` (MTE3, A5 only; on A2 route `UB → L2 → L1`).

### Cube Path

```text
            input                     compute                 output

  L2 ──MTE2──► L1 ──MTE1─────► L0A ──┐──► AIC / Cube ──► L0C (matmul out)
               L1 ──MTE1─────► L0B ──│                    ├─Fixpipe─► L2 / L1 (all platforms)
               L1 ──MTE1─────► BT  ──│                    ├─Fixpipe─► UB (A5 only)
               L1 ──Fixpipe──► FP  ──┘                    └─Fixpipe─► L2 ─MTE2─► UB (910B/A2 alt)

```

Key point: **L0A, L0B, BT, FP, and L0C are directly read/write for Cube-side computation**.
L0C is the matmul output/accumulator storage. On A5, L0C drains directly to UB via Fixpipe;
on 910B/A2 it cannot — the accumulator must go `L0C →(Fixpipe) L2 →(MTE2) UB`, since L1 has
no read/write path to/from UB.

---

## Memory Alignment Requirements
### Alignment Rules

| Scenario               | Alignment   | Requirement Description               |
|------------------------|-------------|---------------------------------------|
| VV (Vector-Vector)	   | 32 bytes	   | Pure vector computation               |
| CV (Cube-Vector)	     | 512 bytes   | Matrix + vector hybrid computation    |
| UB buffer	             | 32 bytes    | All UB allocations                    |
| Single value buffer	   | 32 bytes	   | Reduction results like mean, variance |
| Matrix operation BLOCK | 512-bytes    | For FP16, BLOCK_M/N/K are multiples of 16 (512B ÷ 2B = 256 elements ÷ 16 granularity = 16)                                                     |

## Grid Distribution Strategy

### Merged Grid Distribution (auto-blockify)

When the logical grid exceeds the physical AI Core count — or exceeds the FFTS `coreDim` hard
limit of **65535** — the compiler can automatically merge blocks so each physical program
internally processes multiple logical blocks. This is the compiler-generated equivalent of a
hand-written persistent loop.

Two knobs control it:

1. **`TRITON_ALL_BLOCKS_PARALLEL=1`** — opt-in switch. Without it, `auto_blockify_size` is
   clamped to 1 and the feature is a no-op:
   ```bash
   export TRITON_ALL_BLOCKS_PARALLEL=1
   ```

2. **`auto_blockify_size=N`** — `NPUOptions` launch argument (default 1) passed at kernel
   launch. `N` is the number of logical blocks each physical program processes:
   ```python
   kernel[grid](..., auto_blockify_size=2)
   ```

How it works (from the triton-ascend source, `third_party/ascend/lib/AutoBlockify/`):

1. **TTIR `auto-blockify` pass** (`ascend.passes.ttir.add_auto_blockify`) — runs when
   `auto_blockify_size > 1`. For each function it checks that every `tl.program_id` use is
   blockifiable, then expands the kernel's highest tensor dimension by `N` and wraps
   non-expandable ops in a generated block loop. Marks the function with the
   `auto_blockify_size` attribute.
2. **`bishengir-compile --enable-auto-blockify-loop`** — also gated on the env var. Generates
   the actual per-program loop over the blockified dimension during binary lowering.

Result:

```text
logical grid (e.g. 131072 programs)
  → each physical program processes N logical blocks internally
  → hardware launches a small, legal grid (≤ physical cores, ≤ 65535)
```

Legality and caveats:

- Only functions whose `tl.program_id` uses are **blockifiable** (loads, stores, splats,
  reductions, loops, slices, elementwise, etc.) are transformed. If any use is not
  blockifiable, that function is skipped with a `Cannot apply auto blockify` warning and runs
  with its original (possibly oversized) grid.
- `auto_blockify_size=1` (the default without the env var) is a **no-op** — the pass returns
  immediately. You must pass the option explicitly for it to do anything.
- The compiler path trades launch efficiency for legality and may be slower than a
  hand-written persistent loop that reuses on-chip state between blocks; benchmark both.

Manual alternative — hand-written persistent loop (no env var needed):

```python
grid = (min(num_aicore, total_tiles),)

for tile_id in tl.range(pid, total_tiles, num_pid):
    start_m = tile_id % num_blocks_m
    off_hz   = tile_id // num_blocks_m
    ...
```

Use the manual form when you need per-tile state reuse (e.g. FlashAttention accumulators) or
when auto-blockify legality fails; use `TRITON_ALL_BLOCKS_PARALLEL` when the kernel is
blockifiable and you want the compiler to handle it without code changes.

##  Data Type Optimization
### Avoid Using INT64

Some Vector Core operations do not support INT64 and will degrade to scalar operations:
| Operation	 | Unsupported Data Types |
|------------|------------------------|
| Vector ADD | int64                  |
| Vector CMP | int64/int32            |

**Solution:** Use FP32 for comparison operations
```python
# Before: int64 comparison, degrades to scalar
cols = tl.arange(0, BLOCK_N)
xbar = tl.where(cols < N, x - mean, 0.0)

# After: Convert to FP32 for comparison
cols_cmp = cols.to(tl.float32)
xbar = tl.where(cols_cmp < N, x - mean, 0.0)
```

---

## Common Issues
### UB Overflow

**Symptom:** ub overflow, requires xxxx bits while 1572684 bits available!

**Causes:**
- Excessive data volume in a single loop
- Unreasonable buffer allocation
- Extra overhead from unaligned memory access

**Solutions:**
- Reduce BLOCK_SIZE
- Use intra-core loop partitioning
- Ensure memory access alignment

### coreDim Exceeds Limit

**Symptom:** coreDim=xxxx can't be greater than UINT16_MAX

**Cause:** grid size exceeds 65535

**Solutions:**
- Increase BLOCK_SIZE
- Set TRITON_ALL_BLOCKS_PARALLEL=1
- Use intra-core loops to reduce grid count

### Precision Loss

**Symptom:** Inaccurate results with FP16 input

**Cause:** Insufficient precision in reduction operations

**Solutions:**
- Upcast to FP32 before reduction
- Complete all computations in FP32
- Downcast to output type at the end

---

## Platform Differences (A2/A3 vs 910_95)
| Feature | A2/A3 | 910_95 |
|---------|-------|--------|
| tl.multibuffer default | ✅ | ❌ |
| auto_bind_sub_block default | ✅ | ❌ |
| FP8 support | ❌ | ✅ |
| sync_solver | ✅ | ❌ |
| inject_block_all | ✅ | ❌ |
| overflow_mode="saturate" | Via FP32 intermediate (slower) | Native support |

**Implications:**
- 910_95 requires manual enablement of tl.multibuffer() and tl.parallel(bind_sub_block=True)
- 910_95 has better saturate cast performance (native instructions)
- FP8 quantization scenarios are only available on 910_95
