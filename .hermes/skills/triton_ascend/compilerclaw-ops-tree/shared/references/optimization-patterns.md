# Ascend Triton Performance Optimization Patterns

For API signatures and compiler options see: `triton-operator-shared/references/triton-api-reference.md`

---

## 1. Hardware Constraints (Quick Reference)

| Resource | Value | Notes |
|---|---|---|
| Unified Buffer (UB) | 192 KB | Per AI Core. All active tensors must fit simultaneously. |
| L1 Buffer | 1 MB | Cube core only. Staging for matrix A/B tiles. |
| 32-byte alignment | Required | All load/store addresses and buffer starts. |
| Cube granularity | 16×16 minimum | `tl.dot` operands — BLOCK_M, BLOCK_N, BLOCK_K must all be multiples of 16. |
| UB safety factor | ×0.65–0.8 | Compiler adds ~15–35% overhead beyond your estimates. |
| Max grid (1D) | `num_aicore` or `num_vectorcore` | Exceeding physical core count adds host scheduling overhead. |

**Core count query:**
```python
from triton.runtime import driver
props = driver.active.utils.get_device_properties("npu")
num_cube_cores   = props["num_aicore"]     # for kernels with tl.dot
num_vector_cores = props["num_vectorcore"] # for pure vector kernels
```

**UB estimation:**
```python
def estimate_ub_bytes(BLOCK_M, BLOCK_N, BLOCK_K, dtype_size=2):
    a_tile    = BLOCK_M * BLOCK_K * dtype_size   # input A
    b_tile    = BLOCK_K * BLOCK_N * dtype_size   # input B
    acc       = BLOCK_M * BLOCK_N * 4            # FP32 accumulator
    masks     = BLOCK_M * BLOCK_N                # mask array
    return a_tile + b_tile + acc + masks

AVAILABLE_UB = int(192 * 1024 * 0.65)  # ~128 KB usable after safety factor

# For 2D tiling: also add offset arrays
# offset_arrays = 2 * ROWS * COLS * 4   (int32 row/col offsets)
```

**`next_power_of_2` trap:** `triton.next_power_of_2(48) = 64`, which may exceed UB. Use floor rounding when needed: `1 << (n.bit_length() - 1)`.

---

## 2. Core Optimization Rules

### Rule 1: Block Sizing
- Vector ops: BLOCK_SIZE 1024–2048 for FP16 to fit UB with headroom.
- Matrix ops: BLOCK_M/N/K multiples of 16. Safe starting point: M=128, N=256, K=64–256.
- Use `@triton.autotune` to sweep block sizes rather than hardcoding.
- Grid: 1D preferred, equal to physical core count — not more, not multidimensional.

### Rule 2: Memory Access Contiguity (highest-impact rule)
**Contiguous access allows the compiler to merge many small transactions into large-block DMA instructions. Non-contiguous patterns (2D broadcasting offsets, stride access) force many tiny MTE instructions and can cause 86× more DMA ops than expected.**

```python
# Good: contiguous, compiler can merge into large DMA
offsets = block_start + tl.arange(0, BLOCK_SIZE)

# Bad: non-contiguous, blocks large DMA merging
offsets = block_start + tl.arange(0, BLOCK_SIZE) * stride

# Bad: 2D broadcast creates non-contiguous pattern
off = row_off[:, None] + col_off[None, :]   # prevents DMA merging

# Good: host-side expand+contiguous eliminates broadcast stride
cos_flat = cos.expand(x_shape).contiguous().reshape(total_rows, D)
# Now row * D + col is contiguous — DMA engine runs at full speed
```
Diagnosis: compare theoretical MTE2 count (`total_data / ideal_block_size`) vs actual. Gap >10× means tiling prevents DMA merging.

### Rule 3: Single Pass Over Multi-Pass
Loading the same data multiple times multiplies scalar overhead. When the whole row fits in UB, load once.
```python
# Bad: 3 loads of x, huge scalar overhead
for i in range(passes):
    x = tl.load(x_ptr + off)  # re-read each pass

# Good: load once, all computation in UB
x = tl.load(x_ptr + tot_off, mask=mask, other=0.0).to(tl.float32)
sum_x  = tl.sum(x, 1)
mean   = sum_x / D
var    = tl.sum(x * x, 1) / D - (mean ** 2)
x_norm = (x - mean[:, None]) * tl.rsqrt(var + eps)[:, None]
tl.store(y_ptr + tot_off, x_norm * w + b, mask=mask)
```
NBLOCK can be as large as 8192 when D ≤ UB capacity / num_buffers.

### Rule 3b: `care_padding=False` — skip padding check for ~5–10% free speedup
```python
x = tl.load(ptr + offsets, mask=mask, other=0.0, care_padding=False)
```
Safe when: padding values do not affect downstream computation (masked elements are ignored, alignment is already guaranteed).

### Rule 4: Operator Fusion
Eliminating intermediate GM round-trips transforms memory-bound → compute-bound.
```python
# Before: 2 GM round-trips (load x → store y → load y → store w)
# After: 1 GM round-trip (load x → relu → softmax → store w)
x = tl.load(x_ptr + offsets, mask=mask)
w = tl.softmax(tl.where(x > 0, x, 0.0).to(tl.float32))
tl.store(w_ptr + offsets, w.to(tl.float16), mask=mask)
```

### Rule 5: Precision Rules
```python
# All reductions: upcast to FP32 first
x_fp32 = x.to(tl.float32)
mean = tl.sum(x_fp32, axis=-1) / D

# Matrix multiply: FP16 load, FP32 accumulate, FP16 store
acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
acc += tl.dot(a_fp16, b_fp16)               # FP32 accumulate
tl.store(c_ptr + ..., acc.to(tl.float16))   # FP16 write-back

# Scalar buffers: 32-byte aligned minimum
# mean_buffer = 32 bytes, not 4
```

**Precision verification — always test boundary sizes:**
```python
torch.testing.assert_close(triton_output, torch_output, rtol=1e-3, atol=1e-3)
for size in [127, 128, 255, 256, 1023, 1024, 4096]:
    verify_correctness(size)
```

### Rule 6: Compile-Time Constants and Loop Unrolling
```python
# Use tl.constexpr for block sizes — enables compiler to statically schedule
def kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
    for i in tl.static_range(0, BLOCK_SIZE):  # static unroll
        ...

# Mask for safe boundary access (Ascend has zero tolerance for OOB)
mask = offsets < n_elements
x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
```

### Rule 7: Intra-Core Tiling When UB Is Tight
When BLOCK_SIZE × dtype × num_buffers > UB, tile further inside the core:
```python
for sub_start in range(0, BLOCK_SIZE, SUB_BLOCK_SIZE):
    offsets = start + sub_start + tl.arange(0, SUB_BLOCK_SIZE)
    mask = offsets < n_elements
    x_chunk = tl.load(x_ptr + offsets, mask=mask)
    tl.store(y_ptr + offsets, process(x_chunk), mask=mask)
```

---

## 2.5 Ascend-Specific Compiler Hints & Inline APIs

These are usage patterns for AL extension calls that have direct performance impact, complementing the API signatures in `triton-api-reference.md`.

### `al.compile_hint` — key hints

| Hint name | Value | Effect | When to use |
|---|---|---|---|
| `"dot_pad_only_k"` | (none) | Pad only the K dimension in matmul, reducing UB usage 30–50% | When BLOCK_M/N are already 16-aligned; only K needs padding |
| `"overflow_mode"` | `"saturate"` / `"trunc"` | Control overflow on type conversion | Use `"saturate"` when clamping is safer than wrap-around |
| `"multi_buffer"` | `2` | Enable double-buffering on this tensor (equivalent to `al.multibuffer(tensor, 2)`) | Any tensor loaded repeatedly in a loop |

```python
import triton.language.extra.cann.extension as al

al.compile_hint(acc,    "dot_pad_only_k")                  # MatMul: pad K only
al.compile_hint(tensor, "overflow_mode", "saturate")        # saturate on cast
al.compile_hint(a,      "multi_buffer",  2)                 # double-buffer a
```

### `al.multibuffer` — double buffering
```python
a = tl.load(a_ptr + ...)
a = al.multibuffer(a, size=2)  # only size=2 supported; enabled by default on A2/A3
# Overlaps DMA transfers with computation (ping-pong)
```

### `al.cast` — type conversion with overflow control
```python
y = al.cast(x, tl.float16, fp_downcast_rounding="rtne")          # round-to-nearest-even
y = al.cast(x, tl.int8,    overflow_mode="saturate")              # clamp instead of wrap
y = al.cast(x, tl.float16, overflow_mode="trunc")                 # default: truncate
```

### `al.sync_block_set` / `al.sync_block_wait` — Cube-Vector pipeline sync
Pipeline synchronization between Cube and Vector engines. 16 event IDs (0–15) available; reuse different IDs for multi-stage pipelines.
```python
# Cube side: signal Vector after tl.dot completes
al.sync_block_set(sender="cube", receiver="vector", event_id=0)

# Vector side: wait before processing accumulator output
al.sync_block_wait(sender="cube", receiver="vector", event_id=0)
```

### `num_warps` / `num_stages` on Ascend
`num_warps` and `num_stages` kernel arguments have **no effect** on Ascend NPU — the hardware does not use a warp model. These parameters are accepted without error but are silently ignored by the backend. Use `NPUOptions` (`multibuffer`, `num_stages` for pipeline depth hint) instead.

---

## 3. Reference Kernel Implementations

### GEMM (Matrix Multiplication)
```python
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M
    rn = pid_n * BLOCK_N
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptr + (rm + tl.arange(0, BLOCK_M))[:, None] * stride_am
                           + (k  + tl.arange(0, BLOCK_K))[None, :] * stride_ak)
        b = tl.load(b_ptr + (k  + tl.arange(0, BLOCK_K))[:, None] * stride_bk
                           + (rn + tl.arange(0, BLOCK_N))[None, :] * stride_bn)
        acc += tl.dot(a, b)   # FP32 accumulate; triggers Cube unit
    tl.store(c_ptr + (rm + tl.arange(0, BLOCK_M))[:, None] * stride_cm
                   + (rn + tl.arange(0, BLOCK_N))[None, :] * stride_cn,
             acc.to(tl.float16))
```
Key points: BLOCK_M/N/K multiples of 16; FP32 acc; FP16 write-back.

**Diagonal grid scheduling** (large GEMMs — reduces L2 cache thrashing):
```python
BLOCK_THRESHOLD: tl.constexpr = 4  # autotune 4–8
pid = tl.program_id(0)
NUM_BLOCKS_M = tl.cdiv(M, BLOCK_M)
NUM_BLOCKS_N = tl.cdiv(N, BLOCK_N)
for block_idx in range(pid, NUM_BLOCKS_M * NUM_BLOCKS_N, tl.num_programs(0)):
    if NUM_BLOCKS_M >= BLOCK_THRESHOLD and NUM_BLOCKS_N >= BLOCK_THRESHOLD:
        task_m = block_idx % NUM_BLOCKS_M
        task_n = (block_idx // NUM_BLOCKS_M) % NUM_BLOCKS_N
    else:
        task_m = block_idx // NUM_BLOCKS_N
        task_n = block_idx % NUM_BLOCKS_N
```

**Multi-Vector Core post-dot parallelism** (`al.parallel`):
```python
import triton.language.extra.cann.extension as al
SUB_BLK_M: tl.constexpr = BLOCK_M // 2
for s in al.parallel(0, 2, bind_sub_block=True):
    sub = al.extract_slice(acc, (s * SUB_BLK_M, 0), (SUB_BLK_M, BLOCK_N), (1, 1))
    sub = tl.where(sub > 0.0, sub, 0.0).to(tl.float16)  # ReLU + cast on separate vector cores
    tl.store(c_ptr + ..., sub)
```

### LayerNorm
```python
@triton.jit
def layernorm_kernel(x_ptr, gamma_ptr, beta_ptr, y_ptr, M, N, eps: tl.constexpr,
                     BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + pid * N + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    x_c  = x - mean
    var  = tl.sum(x_c * x_c, axis=0) / N
    x_n  = x_c * tl.rsqrt(var + eps)
    gamma = tl.load(gamma_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    beta  = tl.load(beta_ptr  + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + pid * N + offsets, (x_n * gamma + beta).to(tl.float16), mask=mask)
```
Key points: load once; FP32 throughout; single pass computes mean + var.

### Online Softmax
```python
@triton.jit
def softmax_kernel(x_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + pid * N + offsets, mask=mask, other=-float('inf')).to(tl.float32)
    x_shifted = x - tl.max(x, axis=0)          # numerical stability
    exp_x = tl.exp(x_shifted)
    y = exp_x / tl.sum(exp_x, axis=0)
    tl.store(y_ptr + pid * N + offsets, y.to(tl.float16), mask=mask)
```

### Flash Attention (simplified)
```python
@triton.jit
def flash_attention_kernel(q_ptr, k_ptr, v_ptr, o_ptr, B, H, S, D,
                            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                            BLOCK_D: tl.constexpr):
    pid_b, pid_h, pid_m = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    q_start = pid_b * H * S * D + pid_h * S * D + pid_m * BLOCK_M * D
    q = tl.load(q_ptr + q_start
                + tl.arange(0, BLOCK_M)[:, None] * D
                + tl.arange(0, BLOCK_D)[None, :])
    acc      = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    max_s    = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    sum_e    = tl.zeros([BLOCK_M], dtype=tl.float32)
    for n in range(0, S, BLOCK_N):
        kv_start = pid_b * H * S * D + pid_h * S * D + n * D
        k = tl.load(k_ptr + kv_start + tl.arange(0, BLOCK_N)[:, None] * D
                                      + tl.arange(0, BLOCK_D)[None, :])
        v = tl.load(v_ptr + kv_start + tl.arange(0, BLOCK_N)[:, None] * D
                                      + tl.arange(0, BLOCK_D)[None, :])
        qk      = tl.dot(q, tl.trans(k))            # [BLOCK_M, BLOCK_N]
        new_max = tl.maximum(max_s, tl.max(qk, 1))
        exp_qk  = tl.exp(qk - new_max[:, None])
        new_sum = sum_e * tl.exp(max_s - new_max) + tl.sum(exp_qk, 1)
        acc     = acc * (sum_e / new_sum)[:, None] + tl.dot(exp_qk / new_sum[:, None], v)
        max_s, sum_e = new_max, new_sum
    tl.store(o_ptr + ..., acc.to(tl.float16))
```

---

## 4. Pitfalls & Case Studies

### 4.1 General Pitfalls

**G1: Python-style slice and scalar index operations are unsupported**
```python
# Bad: compilation error or AssertionError
subx = x[1:3]
y[2:4] = suby
local_vec[i] = 123.0
val = tensor[i]

# Good: use Triton/AL APIs
import triton.language.extra.cann.extension as al
subx = al.extract_slice(x, offsets=(1,), sizes=(2,), strides=(1,))
y    = al.insert_slice(y, suby, offsets=(2,), sizes=(2,), strides=(1,))
local_vec = tl.where(tl.arange(0, N) == i, 123.0, local_vec)
val       = tl.gather(tensor, index, axis=0)
```

**G2: `return` / `break` inside loops**
Triton uses MLIR structured control flow (`scf.for`) — early exit is unsupported.
```python
# Bad: compilation error
for i in range(N):
    if cond:
        return val   # "Cannot have return statements inside while or for"
        break        # "unsupported AST node type: Break"

# Good: use tl.where to mask out unnecessary computation
result = tl.where(cond, compute_val(), identity_val())
```
Note: early returns in helper functions called from inside loops also trigger this error.

**G3: Integer comparison in `tl.where` mask**
```python
# Bad: int comparison produces unexpected results
x    = tl.load(x_ptr)   # int32
mask = x > 0            # broken on Ascend

# Good: cast to float first
mask = x.to(tl.float32) > 0
res  = tl.where(mask, a, b)
```

**G4: Multi-dimensional grid / grid > physical core count**
```python
# Bad: multi-dim grid gives no benefit; grid > cores adds scheduling overhead
grid = (2, 2, 2)    # same as (8,) but worse
grid = (1024,)      # if 1024 > num_cores, host overhead dominates

# Good
grid = (min(num_cores, tl.cdiv(N, BLOCK_SIZE)),)
```

**G5: Cube unit never activated**
`tl.dot` is the only op that dispatches to the Cube engine. Elementwise loops stay on Vector. If a matmul kernel shows `aic_cube_ratio ≈ 0` in msprof, check that `tl.dot` is actually present (not replaced by a loop).

**G6: FP16 overflow during intermediate computation**
```python
# Bad: intermediate may overflow FP16 range
x_fp16 = tl.load(x_ptr)
y = x_fp16 * 1000          # overflow

# Good
y = tl.load(x_ptr).to(tl.float32) * 1000
tl.store(out_ptr, y.to(tl.float16))
```

**G7: CPU-NPU synchronization in hot path**
```python
# Bad: .item() triggers a device sync on each call
for t in tensors:
    v = t.item()

# Good: operate on device, avoid sync
max_val = torch.max(tensor)   # stays on device
```

---

### 4.2 Case Study: RoPE (`npu_rotary_mul`)

| Version | Task Time | Bottleneck | Key Change |
|---|---|---|---|
| per-row loop (div/mod) | 3605 µs | MTE3 95.6% (183K MTE2 ops) | baseline |
| 2D Tiling RPT=64 | 1362 µs | scalar 85% | batch load/store |
| Incremental pointer tracking | ~17800 µs | branch divergence | **anti-pattern** |
| Contiguous access (expand) | **752 µs** | memory BW | expand+contiguous |
| torch_npu (CANN C++) | 550 µs | MTE2 99.3% | hand-tuned |

**Pitfall 1 — MTE granularity trap in per-row loop**
Each row = 128 B → 183K MTE2 instructions. torch_npu uses 1,271 (7.7 KB/instruction). Fixed overhead per MTE dispatch dominates when transactions are small. Solution: 2D tiling to batch rows.

**Pitfall 2 — Scalar overhead of 2D tiling**
2D broadcast `row_off[:, None] + col_off[None, :]` creates a `[64,64]` int32 intermediate array plus integer division. Scalar ratio jumped from 40% → 85%. Scalar overhead is O(total_elements), independent of tile size — reducing ROWS_PER_TILE doesn't help, it only makes MTE worse again.

**Pitfall 3 — Conditional branches inside loops are catastrophic**
Replacing `global_row // S` with `cur_s += 1; if cur_s == S: ...` caused 3450 µs → 17783 µs (5× slower). Triton compiles `if` inside loops into masked operations; branching on multiple variables generates extremely inefficient code. Integer division, though slow, is far better than branches.

**Pitfall 4 — Precomputed offset tables trigger compiler assertion**
Reading offsets from `tl.load` and using them for two broadcasting operations triggers:
```
Assertion `addptrRes.hasOneUse() && "Invalid: tt.addptr has multiple users"` failed.
```
Triton-Ascend's `BlockPtrAnalysis` requires each `tt.addptr` to have exactly one user. The same loaded pointer value cannot feed two separate load operations.

**Pitfall 5 — UB overflow from offset arrays (2D tiling)**
Planning accounted for data buffers (8 × 16 KB = 128 KB) but not:
- `off_first`, `off_second`: 32 KB each (`128 × 64 × 4B`)
- `half_mask`: 8 KB
- Total ~200 KB > 192 KB limit

Always include in UB budget: `data_buffers + offset_arrays + mask_arrays + index_arrays`.

**Pitfall 6 — Tiling strategy determines MTE2 instruction granularity**
Both torch_npu and the Triton kernel use only Vector Core, but:
- torch_npu: 24.5 GB/s, 1,271 MTE2 ops (7.7 KB each)
- Triton 2D tiling: 6.95 GB/s, 106,191 MTE2 ops (~90 B each)

2D broadcast `row_off[:, None] + col_off[None, :]` appears non-contiguous to the compiler even though each tile is 8 KB. The compiler cannot merge these into large-block DMA. Linear offsets (`row * D + col`) allow merging.

Note: `MIX_AIC` label in op_statistic does NOT mean Cube engine is active — check `aic_cube_ratio` in `PipeUtilization.csv`.

**Pitfall 7 — Broadcast stride vs expand+contiguous**
Although `expand()` (without `.contiguous()`) avoids physical allocation, its stride layout forces the kernel to use integer division + stride multiplication for row positioning — non-contiguous access. After `expand().contiguous()`, all four tensors (`x`, `cos`, `sin`, `out`) share the same `row * D + col` offset pattern, enabling full-speed DMA.

```python
# Host side: one-time cost, but enables contiguous kernel access
cos_flat = cos.expand(x_shape).contiguous().reshape(total_rows, D)
sin_flat = sin.expand(x_shape).contiguous().reshape(total_rows, D)
# Inside kernel: uniform offset for all tensors
row_off = global_rows * D
off = row_off[:, None] + col_off[None, :]
```
Memory access contiguity > data reuse count. Even re-loading `cos`/`sin` with contiguous access beats broadcasting with strided access.

---

### 4.3 Case Study: GroupNorm+Swish

| Version | Avg (µs) | vs torch_npu |
|---|---|---|
| Two-pass | 36.3 | 8.4× slower |
| Single pass | 3.3 | 0.77× (faster) |
| torch_npu | 4.3 | 1.0× |

**Pitfall 8 — Two-pass pattern for reductions**
Two-pass: pass 1 computes mean/var, pass 2 reloads `x` for normalization. `x` is read 3×, each with loop+mask+`tl.where` → `aiv_scalar_ratio = 96.2%`.

Single-pass fix — load the full row once when `NBLOCK = next_power_of_2(D)` fits in UB:
```python
# ❌ Two-pass: 3 reads of x
for i in range(0, N, BLOCK_N): m_sum += tl.sum(tl.where(mask, tl.load(...), 0.0))
for i in range(0, N, BLOCK_N): v_sum += ...
for i in range(0, N, BLOCK_N): tl.store(...)

# ✅ Single pass: 1 read of x, all computation stays in UB
x     = tl.load(x_ptr + tot_off, mask=tot_mask, other=0.0).to(tl.float32)
sum_x = tl.sum(x, 1)
mean  = sum_x / D
var   = tl.sum(x * x, 1) / D - mean ** 2
xn    = (x - mean[:, None]) * tl.rsqrt(var + eps)[:, None]
tl.store(y_ptr + tot_off, xn * w + b, mask=tot_mask)
```
`tl.sum(x, 1)` performs axis=1 reduction on the loaded 2D block; the compiler keeps `x` in UB without re-reading. NBLOCK up to 8192 is safe — don't be overly conservative.


## Further references
- [Triton-Ascend Official Repository](https://github.com/triton-lang/triton-ascend)
- [Triton-Ascend Tutorials](https://github.com/triton-lang/triton-ascend/tree/main/python/tutorials)
- [Triton-Ascend Performance Guidelines](https://github.com/triton-lang/triton-ascend/blob/main/docs/en/migration_guide/performance_guidelines.md)
