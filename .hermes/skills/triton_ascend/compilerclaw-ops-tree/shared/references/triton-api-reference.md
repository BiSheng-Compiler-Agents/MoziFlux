# Triton-Ascend API Reference (A2/A3)

Official docs: https://ascend.github.io/triton-ascend/

Three API layers:
1. **Standard `triton` / `triton.language`** — core JIT, tl ops, benchmarking (section 1-3 below)
2. **AL Extension** (`triton.language.extra.cann.extension` as `al`) — Ascend NPU-specific ops
3. **BL Extension** (`triton.extension.buffer.language` as `bl`) — explicit on-chip buffer management

---

## 1. `triton` — Top-Level API

### Kernels

| API | Description |
|---|---|
| `@triton.jit` | JIT-compile a function using the Triton compiler |

### Programming Model

| API | Description |
|---|---|
| `triton.cdiv(a, b)` | Ceiling division of two numbers |
| `triton.num_warps()` | Returns the number of warps |

### Compiler

| API | Description |
|---|---|
| `triton.compile(fn, ...)` | Compiles a Triton function ahead-of-time |
| `triton.compiler.CompiledKernel` | A compiled Triton kernel object |

---

## 2. `triton.language` — Kernel Language (`tl`)

Import: `import triton.language as tl`

Docs: https://ascend.github.io/triton-ascend/sources/python-api/triton.language.html

### Programming Model

| API | Description |
|---|---|
| `tl.tensor` | N-dimensional array of values or pointers |
| `tl.program_id(axis)` | ID of the current program instance along `axis` |
| `tl.num_programs(axis)` | Number of program instances along `axis` |

### Creation Ops

| API | Description |
|---|---|
| `tl.arange(start, end)` | Contiguous integer range `[start, end)` |
| `tl.full(shape, value, dtype)` | Tensor filled with a scalar value |
| `tl.zeros(shape, dtype)` | Tensor filled with 0 |
| `tl.zeros_like(x)` | Zero tensor matching shape/dtype of `x` |
| `tl.cat(a, b)` | Concatenate two blocks |
| `tl.cast(x, dtype)` | Cast tensor to given dtype |

### Shape Manipulation Ops

| API | Description |
|---|---|
| `tl.broadcast(a, b)` | Broadcast two tensors to common shape |
| `tl.broadcast_to(x, shape)` | Broadcast `x` to `shape` |
| `tl.expand_dims(x, axis)` | Insert new length-1 dimension |
| `tl.reshape(x, shape)` | Reshape without changing elements |
| `tl.view(x, shape)` | Reinterpret shape (no copy) |
| `tl.ravel(x)` | Flatten to 1D |
| `tl.trans(x, ...)` | Permute dimensions |
| `tl.permute(x, dims)` | Permute dimensions by explicit order |
| `tl.join(a, b)` | Join tensors in a new minor dimension |
| `tl.split(x)` | Split along last dim (size must be 2) |
| `tl.interleave(a, b)` | Interleave values along last dimension |

### Linear Algebra Ops

| API | Description |
|---|---|
| `tl.dot(a, b, acc?, ...)` | Blocked matrix multiply (maps to Cube/Tensor Core) |

### Memory / Pointer Ops

| API | Description |
|---|---|
| `tl.load(ptr, mask?, other?, ...)` | Load from memory at pointer location |
| `tl.store(ptr, value, mask?, ...)` | Store tensor to memory at pointer location |
| `tl.make_block_ptr(base, shape, strides, offsets, block_shape, order)` | Structured 2D block pointer for coalesced access |
| `tl.advance(ptr, offsets)` | Advance a block pointer by offsets |

### Indexing Ops

| API | Description |
|---|---|
| `tl.where(cond, x, y)` | Select elements from `x` or `y` based on `cond` |
| `tl.flip(x, dim?)` | Reverse elements along a dimension |
| `tl.multiple_of(x, value)` | Hint: values in `x` are multiples of `value` |

### Math Ops

| API | Description |
|---|---|
| `tl.abs(x)` | Element-wise absolute value |
| `tl.cdiv(x, y)` | Element-wise ceiling division |
| `tl.ceil(x)` | Element-wise ceiling |
| `tl.floor(x)` | Element-wise floor |
| `tl.clamp(x, min, max)` | Clamp values to `[min, max]` |
| `tl.cos(x)` | Element-wise cosine |
| `tl.sin(x)` | Element-wise sine |
| `tl.exp(x)` | Element-wise natural exponential |
| `tl.exp2(x)` | Element-wise base-2 exponential |
| `tl.log(x)` | Element-wise natural logarithm |
| `tl.log2(x)` | Element-wise base-2 logarithm |
| `tl.sqrt(x)` | Element-wise fast square root |
| `tl.sqrt_rn(x)` | Element-wise precise square root (IEEE round-to-nearest) |
| `tl.rsqrt(x)` | Element-wise inverse square root |
| `tl.erf(x)` | Element-wise error function |
| `tl.sigmoid(x)` | Element-wise sigmoid |
| `tl.softmax(x)` | Element-wise softmax |
| `tl.maximum(x, y)` | Element-wise maximum |
| `tl.minimum(x, y)` | Element-wise minimum |
| `tl.fma(x, y, z)` | Element-wise fused multiply-add |
| `tl.div_rn(x, y)` | Element-wise precise division (IEEE round-to-nearest) |
| `tl.fdiv(x, y)` | Element-wise fast division |
| `tl.umulhi(x, y)` | Upper N bits of 2N-bit product of `x` and `y` |

### Reduction Ops

| API | Description |
|---|---|
| `tl.sum(x, axis)` | Sum along `axis` |
| `tl.max(x, axis)` | Maximum along `axis` |
| `tl.min(x, axis)` | Minimum along `axis` |
| `tl.argmax(x, axis)` | Index of maximum along `axis` |
| `tl.argmin(x, axis)` | Index of minimum along `axis` |
| `tl.xor_sum(x, axis)` | XOR reduction along `axis` |
| `tl.reduce(x, axis, combine_fn)` | Custom reduction with `combine_fn` |

### Scan / Sort Ops

| API | Description |
|---|---|
| `tl.associative_scan(x, axis, combine_fn)` | Prefix scan with `combine_fn` |
| `tl.cumsum(x, axis)` | Cumulative sum along `axis` |
| `tl.cumprod(x, axis)` | Cumulative product along `axis` |
| `tl.sort(x, axis?, descending?)` | Sort tensor |
| `tl.histogram(x, num_bins)` | Histogram with `num_bins` unit-width bins starting at 0 |

### Atomic Ops

| API | Description |
|---|---|
| `tl.atomic_add(ptr, val, mask?, ...)` | Atomic add at memory location |
| `tl.atomic_and(ptr, val, mask?, ...)` | Atomic bitwise AND |
| `tl.atomic_or(ptr, val, mask?, ...)` | Atomic bitwise OR |
| `tl.atomic_xor(ptr, val, mask?, ...)` | Atomic bitwise XOR |
| `tl.atomic_max(ptr, val, mask?, ...)` | Atomic max |
| `tl.atomic_min(ptr, val, mask?, ...)` | Atomic min |
| `tl.atomic_xchg(ptr, val, mask?, ...)` | Atomic exchange |
| `tl.atomic_cas(ptr, cmp, val, ...)` | Atomic compare-and-swap |

### Random Number Generation

| API | Description |
|---|---|
| `tl.rand(seed, offset)` | Uniform floats in U(0,1) |
| `tl.randn(seed, offset)` | Normal floats from N(0,1) |
| `tl.randint(seed, offset)` | Single block of random integers |
| `tl.randint4x(seed, offset)` | Four blocks of random integers |

### Iterators

| API | Description |
|---|---|
| `tl.range(start, end, step?, num_stages?)` | Loop with optional software pipelining |
| `tl.static_range(start, end, step?)` | Compile-time unrolled loop |

### Inline Assembly

| API | Description |
|---|---|
| `tl.inline_asm_elementwise(asm, constraints, args, dtype, ...)` | Execute inline assembly over a tensor |

### Compiler Hint Ops

| API | Description |
|---|---|
| `tl.debug_barrier()` | Barrier to synchronize all threads in a block |
| `tl.max_constancy(x, values)` | Hint: first `values` elements are constant |
| `tl.max_contiguous(x, values)` | Hint: first `values` elements are contiguous |
| `tl.multiple_of(x, values)` | Hint: all values are multiples of `values` |

### Debug Ops

| API | Description |
|---|---|
| `tl.static_print(*args)` | Print at compile time |
| `tl.static_assert(cond, msg?)` | Assert at compile time |
| `tl.device_print(*args)` | Print at runtime from device |
| `tl.device_assert(cond, msg?)` | Assert at runtime from device |

---

## 3. `triton.testing` — Benchmarking

Docs: https://ascend.github.io/triton-ascend/sources/python-api/triton.testing.html

### Functions

| API | Signature | Description |
|---|---|---|
| `do_bench` | `do_bench(fn, warmup=25, rep=100, grad_to_none=None, quantiles=None, fast_flush=True, return_mode='mean', device=None)` | Benchmark fn runtime; returns float ms |
| `do_bench_cudagraph` | `do_bench_cudagraph(fn, rep=20, grad_to_none=None, quantiles=None, return_mode='mean')` | Benchmark via CUDA graphs; returns float ms |
| `perf_report` | `perf_report(benchmarks)` | Decorator: marks fn as perf benchmark for plotting |
| `assert_close` | `assert_close(input, target, atol=None, rtol=None, err_msg='', device=None)` | Assert element-wise mean abs diff is within tolerance |
| `get_dram_gbps` | `get_dram_gbps(device=None)` | DRAM bandwidth of device in GB/s |
| `get_max_simd_tflops` | `get_max_simd_tflops(dtype, device=None)` | Max SIMD TFLOPS for dtype/device |
| `get_max_tensorcore_tflops` | `get_max_tensorcore_tflops(dtype, clock_rate=None, device=None)` | Max tensor core TFLOPS for dtype/device |

### Classes

| Class | Parameters | Description |
|---|---|---|
| `Benchmark` | `x_names, x_vals, line_arg, line_vals, line_names, plot_name, args, x_log=False, y_log=False, x_label=None, y_label=None` | Benchmark configuration object |
| `Mark` | `fn, benchmarks` | Marks a function for benchmarking with given configs |

---

## 4. AL Extension — `triton.language.extra.cann.extension`

Ascend NPU-specific language extensions for JIT kernels. Covers data movement (GM/L1/UB/L0), inter-core sync, custom ops, and tensor manipulation.

---

### Enumerations

#### `CORE` — which core type executes an op

| Member | Description |
|---|---|
| `CORE.VECTOR` | Vector core only |
| `CORE.CUBE` | Cube core only |
| `CORE.CUBE_OR_VECTOR` | Either Cube or Vector |
| `CORE.CUBE_AND_VECTOR` | Both Cube and Vector |

#### `PIPE` — hardware pipeline

| Member | Description |
|---|---|
| `PIPE.PIPE_S` | Scalar pipe |
| `PIPE.PIPE_V` | Vector pipe |
| `PIPE.PIPE_M` | Matrix (Cube) pipe |
| `PIPE.PIPE_MTE1` | Memory Transfer Engine 1 |
| `PIPE.PIPE_MTE2` | Memory Transfer Engine 2 |
| `PIPE.PIPE_MTE3` | Memory Transfer Engine 3 |
| `PIPE.PIPE_ALL` | All pipes |
| `PIPE.PIPE_FIX` | Fixpipe pipeline |

#### `MODE` — execution mode

| Member | Description |
|---|---|
| `MODE.SIMD` | Single Instruction, Multiple Data |
| `MODE.SIMT` | Single Instruction, Multiple Threads |
| `MODE.MIX` | Mixed SIMD/SIMT |

#### `FixpipeDMAMode` — Fixpipe DMA layout

| Member | Description |
|---|---|
| `NZ2DN` | NZ → DN layout |
| `NZ2ND` | NZ → ND layout (default) |
| `NZ2NZ` | No layout change |

#### `FixpipeDualDstMode` — Fixpipe dual destination

| Member | Description |
|---|---|
| `NO_DUAL` | Single destination |
| `COLUMN_SPLIT` | Column-split dual dst |
| `ROW_SPLIT` | Row-split dual dst |

#### `FixpipePreQuantMode` — Fixpipe pre-quant

| Member | Description |
|---|---|
| `NO_QUANT` | No quantization |
| `F322BF16` | FP32 → BF16 |
| `F322F16` | FP32 → FP16 |
| `S322I8` | INT32 → INT8 |

#### `FixpipePreReluMode` — Fixpipe pre-ReLU

| Member | Description |
|---|---|
| `NO_RELU` | No activation |
| `NORMAL_RELU` | Standard ReLU |
| `LEAKY_RELU` | Leaky ReLU |
| `P_RELU` | Parametric ReLU |

#### `SYNC_IN_VF` — Vector/Fixpipe sync modes (used with `debug_barrier`)

| Member | Description |
|---|---|
| `VV_ALL` | Vector-Vector all sync |
| `VST_VLD` | Vector Store → Vector Load |
| `VLD_VST` | Vector Load → Vector Store |
| `VST_VST` | Vector Store → Vector Store |
| `VS_ALL` | Vector-Scalar all sync |
| `VST_LD` | Vector Store → Load |
| `VLD_ST` | Vector Load → Store |
| `VST_ST` | Vector Store → Store |
| `SV_ALL` | Scalar-Vector all sync |
| `ST_VLD` | Store → Vector Load |
| `LD_VST` | Load → Vector Store |
| `ST_VST` | Store → Vector Store |

---

### Types

| API | Description | Parameters | Returns |
|---|---|---|---|
| `int64(value)` | Wraps a Python int as `tl.int64` (default is int32 on device) | `value: int` | `int64` with `.type == tl.int64` |
| `ascend_address_space` | Singleton with named address-space objects: `.UB`, `.L1`, `.L0A`, `.L0B`, `.L0C`, `.GM` | — | `ascend_address_space_base` per attr |

---

### Core Operations

| API | Description | Parameters | Returns | Errors |
|---|---|---|---|---|
| `@builtin` | Marks a function as an Ascend builtin; `_builder` is injected by JIT | `fn: callable` | Wrapped fn | `ValueError` if called outside JIT |
| `is_builtin(fn)` | Checks if `fn` was decorated with `@builtin` | `fn: callable` | `bool` | — |
| `sub_vec_id()` | Index of the current Vector Core on this AI Core | — | `tl.tensor[int64]` scalar | — |
| `sub_vec_num()` | Number of Vector Cores per AI Core (compile-time constant) | — | `tl.constexpr` | — |
| `copy(src, dst)` | Copy UB→UB or UB→L1. **910_95 only** | `src: UB buffer`, `dst: UB or L1 buffer` | `None` | `RuntimeError` wrong HW; `TypeError` space/shape/dtype mismatch |
| `copy_from_ub_to_l1(src, dst)` | **Deprecated.** Use `copy` instead. UB→L1 copy | Same as `copy` | `None` | Same as `copy` |
| `fixpipe(src, dst, dma_mode, dual_dst_mode)` | L0C→UB transfer via Fixpipe. **910_95 only** | `src: tl.tensor in L0C`, `dst: bl.buffer in UB`, `dma_mode=NZ2ND`, `dual_dst_mode=NO_DUAL` | `None` | `RuntimeError` wrong HW; `TypeError` bad src/dst; `ValueError` alignment |
| `debug_barrier(sync_mode)` | Insert sync barrier in Vector/Fixpipe pipeline (debugging) | `sync_mode: SYNC_IN_VF` | `None` | — |

**`fixpipe` alignment rules:**
- 32-bit types: last dim aligned to 8; non-NZ2ND last dim aligned to 16; column-split aligned to 32; NZ2DN first dim aligned to 8
- 16-bit types: last dim aligned to 16; NZ2DN first dim aligned to 16

---

### Synchronization Operations

| API | Description | Parameters | Returns | Errors |
|---|---|---|---|---|
| `sync_block_all(mode, event_id)` | Full barrier across all cores of the given type | `mode: "all_cube"\|"all_vector"\|"all"\|"all_sub_vector"`, `event_id: int [0,15]` | `None` | `AssertionError` bad mode or id |
| `sync_block_set(sender, receiver, event_id, sender_pipe?, receiver_pipe?)` | Signal event from sender to receiver (producer side) | `sender/receiver: "cube"\|"vector"` (must differ), `event_id: int [0,15]`, pipes optional | `None` | `ValueError` sender==receiver; `TypeError` bad pipe; `AssertionError` bad args |
| `sync_block_wait(sender, receiver, event_id, sender_pipe?, receiver_pipe?)` | Wait for event from sender (consumer side) | Same as `sync_block_set` | `None` | Same as `sync_block_set` |

Default pipes: cube sender → `PIPE_FIX`; vector sender → `PIPE_MTE3`; receiver → `PIPE_MTE2`.

---

### Scope

| API | Description | Parameters | Returns | Errors |
|---|---|---|---|---|
| `with al.scope(core_mode=...)` | Context manager — annotates a code block as Cube or Vector core code | `core_mode: "cube"\|"vector"` | `scope` context manager | `ValueError` invalid mode; `RuntimeError` used outside JIT |

```python
with al.scope(core_mode="cube"):
    result = tl.dot(a, b)
```

---

### Custom Operations

| API | Description | Parameters | Returns | Errors |
|---|---|---|---|---|
| `custom(name, *args, **kwargs)` | Invoke a registered custom op by name | `name: str`, `*args`, `out=...` kwarg for output tensor | `tl.tensor \| tuple \| None` | `AssertionError` unregistered name (unless `__builtin_` prefix) |
| `@register_custom_op` | Class decorator to register a custom op | Class must define `core: CORE`, `pipe: PIPE`, `mode: MODE`. Optional: `name`, `symbol`, `bitcode`, `source`, `compile`, `extra_attr` | Same class (registered) | `AssertionError` missing fields or name already taken |
| `custom_semantic(name, *args, _builder, **kwargs)` | IR-level implementation behind `custom()`. Not called directly. | Same as `custom` + `_builder` | `tl.tensor \| tuple \| None` | — |

**`@register_custom_op` class fields:**

| Field | Required | Description |
|---|---|---|
| `core` | Yes | `CORE` enum value |
| `pipe` | Yes | `PIPE` enum value |
| `mode` | Yes | `MODE` enum value |
| `name` | No | Op name (default: class name) |
| `symbol` | For non-builtin | Symbol name in bitcode |
| `bitcode` | For non-builtin | Path to `.bc` file |
| `source` | No | Source file path |
| `compile` | No | Compilation flags |
| `extra_attr` | No | e.g. `"src_stride_len=3"` |

---

### Math Operations

| API | Signature | Description | Input Types | Returns | Errors |
|---|---|---|---|---|---|
| `atan2` | `atan2(y, x)` | Element-wise `arctan(y/x)`, all quadrants. Internally computed in fp32. | `fp16 \| fp32 \| bf16` | tensor same dtype as `x` | `static_assert` if non-float or int8/int1 |
| `isfinited` | `isfinited(x)` | Element-wise finite check (not NaN, not Inf) | `fp16 \| fp32 \| bf16` | `int1` boolean tensor | `static_assert` if int8/int1 or non-float |
| `finitef` | `finitef(x)` | Same as `isfinited` but **float32 only** | `float32` only | `int1` boolean tensor | `static_assert` if not float32 |

---

### Auxiliary Operations

| API | Description | Parameters | Returns | Errors |
|---|---|---|---|---|
| `parallel(arg1, arg2?, step?, num_stages?, loop_unroll_factor?, bind_sub_block?)` | Loop iterator with parallel execution semantics across Vector Cores | `arg1`: start (or end if arg2 absent); `arg2`: end; `step`: stride; `bind_sub_block=True`: distribute iterations across vector cores (max 2 on 910B) | Iterator | — |
| `compile_hint(ptr, hint_name, hint_val?)` | Attach a named compiler hint to a tensor **value** by emitting an `annotation.mark` op on its SSA handle. No-op in SIMT mode. | `ptr: tensor`, `hint_name: str`, `hint_val: bool\|int\|str\|list[int]\|None` (default None) | `None` | `AssertionError` non-string name; `ValueError` unsupported type |
| `multibuffer(src, size)` | Mark tensor for double-buffering (pipeline optimization). Pure sugar for `compile_hint(src, "hivm.multi_buffer", 2)` — byte-identical IR | `src: tensor`, `size: int` — **only `2` supported** | `None` | `AssertionError` size != 2 |

**`compile_hint` mechanics** (aux_ops.py; IR form verified by dump):

```mlir
annotation.mark %3 {hivm.multi_buffer = 2 : i32} : tensor<64xf32>   // compile_hint(x, "hivm.multi_buffer", 2)
annotation.mark %3 {trans_k} : tensor<64xf32>                       // compile_hint(k, "trans_k") — no value
```

- Value conversion by type (`bool` checked FIRST, so explicit `False` stays a bool attr): `bool` → bool attr; `None`/falsy → **unit attr** (presence-only flag — the key's existence on the mark is the signal); `int` → int32 attr; `constexpr` → str attr of its value; `list` → i64 array attr.
- The mark survives ttir → ttadapter and is consumed by the Annotation-dialect passes (`AnnotationMark`/`AnnotationLowering`) before the npuir stage.
- Hint names are **not validated** — arbitrary keys compile fine. A key only has an effect if a downstream pass pattern-matches it; `hivm.multi_buffer` (marks the tensor's buffer for multi-buffering) is the proven-consumed key. Misspelled/unrecognized keys are silently dead attributes.

---

### Vector / Tensor Operations

| API | Description | Parameters | Returns | Errors |
|---|---|---|---|---|
| `insert_slice(ful, sub, offsets, sizes, strides)` | Insert sub-tensor into full tensor at given position | `ful, sub: tensor` (same rank); `offsets, sizes, strides: tuple[int\|constexpr]` (sizes≥1, strides≥0) | New tensor, same shape/type as `ful` | `AssertionError` rank mismatch or bad sizes/strides |
| `extract_slice(ful, offsets, sizes, strides)` | Extract sub-tensor from full tensor | `ful: tensor`; `offsets, sizes, strides: tuple` (sizes≥1, strides≥0) | Tensor with shape=`sizes` | `AssertionError` rank mismatch or bad values |
| `get_element(src, indice)` | Read scalar at given multi-dim index | `src: tensor`, `indice: tuple[int\|constexpr]` — length must match rank | Scalar tensor same dtype as `src` | `ValueError` wrong number of indices |
| `sort(ptr, dim?, descending?)` | Sort tensor along **last dimension only** | `ptr: tensor`, `dim=-1` (must be last dim), `descending=False` | Sorted tensor, same shape | `TypeError` non-int dim or unsupported dtype; `ValueError` rank<1 or wrong dim |
| `flip(ptr, dim?)` | Reverse elements along a dimension. SIMD: hardware flip; SIMT: XOR-swap (power-of-2 sizes required) | `ptr: tensor`, `dim=-1` (supports negative indexing) | Flipped tensor, same shape | `TypeError` non-int dim; `ValueError` rank<1 or out-of-range |
| `cast(input, dtype, fp_downcast_rounding?, bitcast?, overflow_mode?)` | Type-cast tensor; supports FP8 on 910_95 | `input: tensor`, `dtype: tl.dtype`, `fp_downcast_rounding: "rtne"\|"rtz"`, `bitcast=False`, `overflow_mode: "trunc"\|"saturate"` | Cast tensor | `ValueError` invalid combo or unsupported HW; `AssertionError` impossible cast |

**`sort` supported dtypes:** `int8`, `int16`, `bf16`, `fp16`, `fp32`, `int32`, `int64`, `fp8e4nv`, `fp8e5`

---

### Memory Operations

All ops require float dtypes (`fp16`, `bf16`, `fp32`) for pointer/src/dst tensors unless noted. Index tensors must be integer.

| API | Direction | Description |
|---|---|---|
| `index_put` | UB→GM | Scatter values into GM using index along `dim`. 2D–5D. |
| `gather_out_to_ub` | GM→UB | Gather elements from GM to UB using index along `dim`. 1D–5D. Optional `other` for OOB. |
| `scatter_ub_to_out` | UB→GM | Scatter UB tile into GM using index along `dim`. `value` can be tensor or scalar (broadcast). 1D–5D. |
| `index_select_simd` | GM→UB | SIMD-accelerated index-select: loads whole slices from GM to UB along `dim` (not trailing dim). |

#### `index_put(ptr, index, value, dim, index_boundary, end_offset, start_offset, dst_stride)`

| Parameter | Type | Description |
|---|---|---|
| `ptr` | tensor (GM ptr) | Destination in Global Memory |
| `index` | tensor (UB, int) | Index tensor; auto-reshaped if not 1D |
| `value` | tensor (UB) | Values to write; rank 2–5 |
| `dim` | int | Scatter dimension; `0 <= dim < rank(value)-1` |
| `index_boundary` | int | Upper bound for index values |
| `end_offset` | tuple[int] | Per-dim end offsets; length == rank(value) |
| `start_offset` | tuple[int] | Per-dim start offsets; length == rank(value) |
| `dst_stride` | tuple[int] | Per-dim strides of destination; length == rank(value) |

2D semantics (dim=0): `out[index[i]][s1:e1] = value[i][0:e1-s1]`

#### `gather_out_to_ub(src, index, index_boundary, dim, src_stride, end_offset, start_offset, other?)`

| Parameter | Type | Description |
|---|---|---|
| `src` | tensor (GM ptr) | Source in Global Memory |
| `index` | tensor (UB, int) | Index tensor; rank 1–5 |
| `index_boundary` | int | Upper bound for index values |
| `dim` | int | Gather dimension; `0 <= dim < rank(index)` |
| `src_stride` | tuple[int] | Per-dim source strides; length == rank(index) |
| `end_offset` | tuple[int] | Per-dim end offsets |
| `start_offset` | tuple[int] | Per-dim start offsets |
| `other` | scalar (optional) | Default value for OOB indices |

Returns tensor with same shape as `index`.
2D semantics (dim=0): `out[i][j] = src[start[0]+index[i][j]][start[1]+j]`

#### `scatter_ub_to_out(ptr, value, index, index_boundary, dim, dst_stride, end_offset, start_offset)`

| Parameter | Type | Description |
|---|---|---|
| `ptr` | tensor (GM ptr) | Destination in Global Memory |
| `value` | tensor or scalar (UB) | Values to scatter; scalar is broadcast |
| `index` | tensor (UB, int) | Index tensor; rank 1–5 |
| `index_boundary` | int | Upper bound for index values |
| `dim` | int | Scatter dimension; `0 <= dim < rank(index)` |
| `dst_stride` | tuple[int] | Per-dim destination strides |
| `end_offset` | tuple[int] | Per-dim end offsets |
| `start_offset` | tuple[int] | Per-dim start offsets |

2D semantics (dim=0): `out[start[0]+index[i][j]][start[1]+j] = value[i][j]`

#### `index_select_simd(src, dim, index, src_shape, src_offset, read_shape)`

| Parameter | Type | Description |
|---|---|---|
| `src` | tensor (GM ptr) | Source tensor in Global Memory |
| `dim` | int | Selection dimension — **not** the trailing dim |
| `index` | tensor (UB, 1D int) | Indices to select |
| `src_shape` | list[int\|tensor] | Full shape of source |
| `src_offset` | list[int\|tensor] | Read start offsets; use `-1` at `dim` position |
| `read_shape` | list[int\|tensor] | Tile shape to read; use `-1` at `dim` position |

Returns tensor in UB with shape = `read_shape` but `dim` replaced by `len(index)`.

---

### Built-in Custom Ops (pre-registered, internal use)

| Name | Description |
|---|---|
| `__builtin_index_select` | SIMT index-select GM→UB; 2D–5D src, 1D–2D index |
| `__builtin_index_put` | SIMT index-put UB→GM; 2D–5D value |
| `__builtin_gather_load` | SIMT gather-load GM→UB; 1D–5D index |
| `__builtin_scatter_store` | SIMT scatter-store UB→GM; 1D–5D index |

Invoked internally via `al.custom()`. No user registration needed.

---

### Memory Hierarchy Summary

```
Global Memory (GM)
    |  MTE1/2/3
    v
L1 Buffer
    |
    v
L0A / L0B / L0C
    |  Fixpipe
    v
Unified Buffer (UB)
```

| Operation | Direction |
|---|---|
| `gather_out_to_ub`, `index_select_simd` | GM → UB |
| `scatter_ub_to_out`, `index_put` | UB → GM |
| `copy`, `copy_from_ub_to_l1` | UB → L1 |
| `fixpipe` | L0C → UB |

---

## 5. BL Extension — `triton.extension.buffer.language`

Provides explicit on-chip memory allocation and tensor↔buffer interop inside JIT kernels.

---

### Classes

#### `address_space` (abstract base)

Override `to_ir(builder) -> ir.type` to define a custom address space.

#### `buffer_type`

Type descriptor for buffers.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `element_ty` | `tl.dtype` | Yes | Scalar element type |
| `shape` | `list[int]` | Yes | Dimensions |
| `space` | `address_space` | No | Address space (default None) |
| `strides` | `list[int]` | No | Memory strides (default [] = contiguous) |

#### `buffer`

Core buffer value type. Extends `tl._value`.

| Attribute | Type | Description |
|---|---|---|
| `type` | `buffer_type` | Full type descriptor |
| `dtype` | `tl.dtype` | Element type |
| `shape` | `list[int]` | Dimensions |
| `space` | `address_space\|None` | Address space |
| `strides` | `list[int]` | Memory strides |

Methods (all `@builtin`, must be called inside `@triton.jit`):

| Method | Parameters | Returns | Notes |
|---|---|---|---|
| `buf.subview(offsets, sizes, strides)` | `offsets, sizes, strides: list[constexpr]` | `buffer` sub-region | 32-byte alignment required |
| `buf.to_tensor(writable?, target_shape?)` | `writable=True`, `target_shape=None` | `tl.tensor` | Layout conversion if `target_shape` given |

---

### Functions

| API | Description | Parameters | Returns | Errors |
|---|---|---|---|---|
| `@builtin` | Marks fn as buffer-language builtin; injects `_builder` in JIT | `fn: callable` | Wrapped fn | `ValueError` outside JIT |
| `is_builtin(fn)` | Check if fn is a buffer builtin | `fn: callable` | `bool` | — |
| `alloc(etype, shape, _address_space?, is_mem_unique?)` | Allocate local (on-chip) memory buffer | `etype: tl.dtype` (not int1), `shape: list[constexpr]`, `_address_space=None`, `is_mem_unique=False` | `buffer` | `TypeError` int1 or bad shape |
| `to_buffer(tensor, space?, bind_buffer?)` | Convert `tl.tensor` to `buffer`. If `bind_buffer` given, writes tensor into it. | `tensor: tl.tensor` (non-scalar), `space=None`, `bind_buffer=None` | `buffer` | `TypeError` scalar input; `ValueError` bad bind_buffer |
| `to_tensor(memref, writable?, target_shape?)` | Convert `buffer` to `tl.tensor` | `memref: buffer`, `writable=True`, `target_shape=None` | `tl.tensor` | `TypeError` non-buffer; `AssertionError` target_shape == current shape |
| `subview(src, offsets, sizes, strides)` | Slice a buffer. Shares underlying memory. 32-byte alignment enforced. | `src: buffer`, `offsets/sizes/strides: list[constexpr]` | `buffer` sub-region | `TypeError` alignment violation; `ValueError` negative offset |

**`subview` / `bl.subview` alignment rules:**
1. Flat byte offset from buffer start must be 32-byte aligned
2. All strides must be 1
3. Start of second row (last dim) must also be 32-byte aligned

Dynamic offsets (as `tl.tensor`) skip compile-time alignment checks.

---

### Builder Utilities (internal — for extension authors)

| API | Description | Parameters |
|---|---|---|
| `create_builder_method_wrapper_with_buffer_builder(main, delegate, method_name)` | Wraps a delegate builder method, syncing insertion points around the call | `main_builder`, `delegate_builder`, `method_name: str` |
| `attach_builder_methods_with_buffer_builder(main, delegate, method_names)` | Attaches multiple delegate methods onto main builder | `main_builder`, `delegate_builder`, `method_names: list[str]` |
| `setup_unified_builder_with_buffer_builder(main, buffer_builder)` | Attaches all buffer ops (`alloc`, `to_buffer`, `to_tensor`, `subview`, `get_null_attr`, `get_str_array_attr`) to main builder | `main_builder`, `buffer_builder` |

---

## 6. Module Structure

### AL Extension (`triton.language.extra.cann.extension`)

| Module | Contents |
|---|---|
| `core.py` | Enums, types, `builtin`, fundamental ops |
| `scope.py` | `scope` context manager |
| `custom_op.py` | Custom op registry and invocation |
| `builtin_custom_ops.py` | Pre-registered `__builtin_*` ops |
| `math_ops.py` | `atan2`, `isfinited`, `finitef` |
| `aux_ops.py` | `parallel`, `compile_hint`, `multibuffer` |
| `vec_ops.py` | `sort`, `flip`, `cast`, `insert_slice`, `extract_slice`, `get_element` |
| `mem_ops.py` | `index_put`, `gather_out_to_ub`, `scatter_ub_to_out`, `index_select_simd` |
| `semantic.py` | IR-level implementation layer |
| `builder.py` | Unified builder setup (Ascend ↔ Triton) |

### BL Extension (`triton.extension.buffer.language`)

| Module | Contents |
|---|---|
| `core.py` | `address_space`, `buffer_type`, `buffer`, `alloc`, `to_buffer`, `to_tensor`, `subview` |
| `semantic.py` | IR construction and type checking |
| `builder.py` | Builder delegation pattern utilities |

---

## 7. `NPUOptions` — Kernel Compiler Options

These options are passed as keyword arguments when calling a `@triton.jit` kernel on Ascend NPU (or via `triton.compile`). They are defined in `triton/backends/ascend/compiler.py` as a frozen dataclass and map directly to `bishengir-compile` flags.

Usage example:
```python
kernel[(grid,)](arg0, arg1, ..., multibuffer=True, sync_solver=True, num_stages=2)
```

### Execution Model

| Option | Type | Default | Description | bishengir-compile flag |
|---|---|---|---|---|
| `compile_mode` | `str` | `"simd"` | Top-level mode selector. `"simd"` (default), `"unstructured_in_simt"` (sets `force_simt_template=True`), `"simt_only"` (sets `force_simt_only=True`, `parallel_mode="simt"`). Overrides the individual flags below. | — |
| `parallel_mode` | `str` | `"simd"` | Low-level parallel mode: `"simd"` or `"simt"`. Set automatically by `compile_mode`. | — |
| `force_simt_only` | `bool` | `False` | Force pure SIMT path (bypasses SIMD/hivm pipeline). Uses `ttir_to_npubin` instead of `linalg_to_bin`. | `--pure-simt` |
| `force_simt_template` | `bool` | `False` | Force unstructured SIMT template (historical compat). Set by `compile_mode="unstructured_in_simt"`. | internal pass flag |
| `mix_mode` | `str` | `""` | Mix mode string (`"aic"`, `"aiv"`, `"mix"`). `"aic"` disables hfusion vectorize. | `--disable-hfusion-vectorize=true` (if `"aic"`) |

### Parallelism & Warps

| Option | Type | Default | Description | bishengir-compile flag |
|---|---|---|---|---|
| `num_warps` | `int` | `32` | Number of warps per block | `--num-warps=N` (simt_only only) |
| `warp_size` | `int` | `32` | Threads per warp | `--threads-per-warp=N` (simt_only only) |
| `num_ctas` | `int` | `1` | Number of cooperative thread arrays | — |
| `num_stages` | `int` | `1` | Software pipeline stages (affects loop pipelining) | — |
| `auto_blockify_size` | `int` | `1` | Auto blockify tile size. Overridden to 1 if `TRITON_AUTO_MAP_PARALLEL_BLOCKS` env is not set. | `--enable-auto-blockify-loop` (env-gated) |
| `cluster_dims` | `tuple` | `(1,1,1)` | Cluster dimensions | — |

### Multi-Buffering & Memory

| Option | Type | Default | Description | bishengir-compile flag |
|---|---|---|---|---|
| `multibuffer` | `bool` | `True` (non-910_95), `False` (910_95) | Enable auto multi-buffering (double-buffering of local buffers). Default is `not is_compile_on_910_95`, i.e. off on 910_95 unless requested. | `--enable-auto-multi-buffer=<bool>` |
| `enable_ubuf_saving` | `bool` | `None` | Enable unified buffer saving optimizations (A2/A3 path only; parsed but silently ignored on 910_95) | `--enable-ubuf-saving=<bool>` |
| `enable_preload` | `bool` | `None` | Enable preload optimization (A2/A3 path only; parsed but silently ignored on 910_95) | `--enable-preload=<bool>` |
| `limit_auto_multi_buffer_only_for_local_buffer` | `bool` | `None` | Binary help: "When enable-auto-multi-buffer = true, limit it only work for local buffer" | `--limit-auto-multi-buffer-only-for-local-buffer=<bool>` |
| `limit_auto_multi_buffer_of_local_buffer` | `str` | `None` | Binary help: "When enable-auto-multi-buffer = true, limit local buffer mode". String enum; `"no-limit"` disables the limit. | `--limit-auto-multi-buffer-of-local-buffer=<str>` |
| `set_workspace_multibuffer` | `int` | `None` | Binary help: "Override number of multibuffers for workspace, defaults to 1 (off)" | `--set-workspace-multibuffer=<int>` |
| `disable_tightly_coupled_buffer_reuse` | `bool` | `False` | Disable tightly coupled buffer reuse (910_95 only) | `--disable-tightly-coupled-buffer-reuse` |
| `shared_mem_dynamic_size` | `int` | `221184` (simd), `122880` (simt_only) | Dynamic shared memory size in bytes. Set automatically by `compile_mode`. | `--shared-mem-dynamic-size=<int>` (simt_only only) |

### Synchronization

| Option | Type | Default | Description | bishengir-compile flag |
|---|---|---|---|---|
| `sync_solver` | `bool` | `None` | Binary help: "Enable HIVM Graph-Sync-Solver Auto-Sync Pass" — auto-inserts block syncs between Cube/Vector stages. On A2/A3, also enables cross-core GSS. Device-verified: its auto injection deadlocks or silently corrupts manual `sync_block` edges inside loops; keep it off and use `disable_auto_inject_block_sync=True` with hand-written syncs. | `--enable-hivm-graph-sync-solver=<bool>` (+ `--enable-hivm-cross-core-gss=<bool>` on A2/A3) |
| `unit_flag` | `bool` | `None` | Enable hivm unit-flag sync mode | `--enable-hivm-unit-flag-sync=<bool>` |
| `inject_barrier_all` | `bool` | `None` | Inject barrier-all sync between all pipeline stages | `--enable-hivm-inject-barrier-all-sync=<bool>` |
| `inject_block_all` | `bool` | `None` | Inject block-all sync | `--enable-hivm-inject-block-all-sync=<bool>` |
| `enable_sync_block_lock` | `bool` | `False` | Enable sync block lock mechanism (used with `al.sync_block_set/wait`) | internal pass flag |
| `disable_auto_inject_block_sync` | `bool` | `None` | Disable the graph-sync-solver's automatic block-sync injection. Mandatory when manual `sync_block_set/wait` edges repeat inside loops (device-verified deadlock / silent corruption otherwise). | `--disable-auto-inject-block-sync=<bool>` |
| `enable_auto_bind_sub_block` | `bool` | `None` | Override auto bind-sub-block behavior. `None` = use value from IR module attribute. | `--enable-auto-bind-sub-block=<bool>` |
| `enable_cce_vf_auto_sync` | `bool` | `None` | CCE/LLVM-level VF auto sync; forwarded to bisheng as an `-mllvm` option | `--append-bisheng-options=-mllvm --cce-vf-auto-sync=<bool>` |
| `enable_cce_vf_remove_membar` | `bool` | `None` | CCE/LLVM-level VF membar removal; forwarded to bisheng as an `-mllvm` option | `--append-bisheng-options=-mllvm --cce-vf-remove-membar=<bool>` |

### Vectorization & Fusion

| Option | Type | Default | Description | bishengir-compile flag |
|---|---|---|---|---|
| `enable_hivm_auto_cv_balance` | `bool` | `None` | Binary help: "Enable balancing during cv-pipelining" — auto Cube/Vector balance scheduling | `--enable-hivm-auto-cv-balance=<bool>` |
| `enable_mixed_cv` | `bool` | `None` | Enable mixed Cube-Vector execution (910_95 only) | `--enable-mixed-cv=<bool>` |
| `enable_vf_fusion` | `bool` | `False` | Binary help: "Enable vf fusion" — turns on the VFFusion pipeline (`hfusion-merge-vf` + VFFusionAnalyzer/Outliner) that outlines and merges vector-function (VF) regions (`hivm.vector_function`) | `--enable-vf-fusion` |
| `vf_merge_level` | `int` | `1` | VF-merge aggressiveness, verbatim from binary: "Merging level. 0: no merge; 1: merge VFs only without dependency; 2: merge all VFs." VFs = outlined vector-function regions, NOT Vector-Fixpipe. Quirk: declared twice in the dataclass (`0` at line 734, then `1` at line 737) — the second wins, so the effective default is 1. | `--enable-vf-merge-level=<int>` |
| `enable_auto_vectorize_v2` | `bool` | `None` (pass default) | Gates the HFusion AutoVectorizeV2 pass (`hfusion-auto-vectorize-v2`): fuses elementwise op chains into fused vector ops and fuses independent sibling loops (`LoopFuseSiblingOp`) so they vectorize as one. 910_95 path only. Device-verified caveat: sibling-loop fusion has SIGSEGV'd/miscompiled on paired masked chunk blocks (FA ping-pong kernels) — set `False` to bypass. | `--enable-auto-vectorize-v2=<bool>` |
| `auto_vectorize_v2_max_fused_ops_num` | `int` | `None` | Binary help: "Maximum number of ops to fuse in AutoVectorizeV2 (Default: pass default)" | `--hfusion-max-fused-ops-in-auto-vectorize-v2=<int>` |
| `prevec_max_fused_ops_num` | `int` | `None` | Max fused elementwise ops in pre-vectorize hfusion | `--hfusion-max-fused-elementwise-ops=<int>` |
| `hfusion_enable_multiple_consumer_fusion` | `bool` | `False` | Allow hfusion to fuse ops with multiple consumers | `--hfusion-enable-multiple-consumer-fusion=<bool>` |
| `add_auto_scheduling` | `bool` | `False` | Insert DAG sync/scope/ssbuffer passes for auto-scheduling of Cube/Vector pipelines | DAG passes in ttir_to_linalg |

### Shape & Layout Transforms

| Option | Type | Default | Description | bishengir-compile flag |
|---|---|---|---|---|
| `enable_nd2nz_on_vector` | `bool` | `False` | Enable ND→NZ layout conversion on Vector core | internal linalg pass flag |
| `enable_drop_unit_dims` | `bool` | `None` | Drop unit (size-1) dimensions before codegen | `--enable-drop-unit-dims=<bool>` |
| `enable_flatten` | `bool` | `None` | Flatten multi-dim accesses to 1D | `--enable-flatten=<bool>` |
| `tile_mix_vector_loop` | `int` | `None` | Tile size for vector loop in mixed Cube-Vector kernels (A2/A3 path only; parsed but silently ignored on 910_95) | `--tile-mix-vector-loop=<int>` |
| `tile_mix_cube_loop` | `int` | `None` | Tile size for cube loop in mixed Cube-Vector kernels (A2/A3 path only; parsed but silently ignored on 910_95) | `--tile-mix-cube-loop=<int>` |
| `optimize_dynamic_offset` | `bool` | `False` | Optimize dynamic offset computation during lowering | internal pass flag |
| `enable_mask_fallback_conversion` | `bool` | `False` | Convert masked ops to fallback representation when hardware lacks native support | internal pass flag |
| `enable_select_analysis` | `bool` | `True` | Enable select-analysis pass in linalg lowering | internal linalg pass flag |

### SIMT-Specific Options

| Option | Type | Default | Description | bishengir-compile flag |
|---|---|---|---|---|
| `enable_bishengir_simt_optimization` | `int` | `0` | SIMT optimization level (non-zero to enable). Only used in `simt_only` mode. | `--enable-bishengir-simt-optimization=<int>` |
| `simt_stack_limit` | `int` | `None` | SIMT stack size limit | `--simt-stack-limit=<int>` |
| `enable_simt_reorder_instruction` | `bool` | `False` | Enable reorder-instruction pattern for SIMT | `--enable-simt-reorder-instruction=true` |
| `disable_fma` | `bool` | `False` | Disable FMA optimization (improves precision at cost of perf) | `--disable-fma` |

### Vectorized Load/Cast

| Option | Type | Default | Description | bishengir-compile flag |
|---|---|---|---|---|
| `enable_cce_vf_auto_sync` | `bool` | `None` | Enable CCE VF auto-sync pass (passed via `--append-bisheng-options`) | `--append-bisheng-options=-mllvm --cce-vf-auto-sync=<bool>` |
| `enable_cce_vf_remove_membar` | `bool` | `None` | Remove memory barriers in CCE VF pipeline | `--append-bisheng-options=-mllvm --cce-vf-remove-membar=<bool>` |
| `disable_size_align_for_cast` | `bool` | `None` | Disable size alignment enforcement during cast (A2/A3 only) | `--disable-size-align-for-cast=<bool>` |

### Misc / Passthrough

| Option | Type | Default | Description | bishengir-compile flag |
|---|---|---|---|---|
| `bisheng_options` | `str` | `"-cce-link-aicore-ll-module <libdevice.bc>"` | Raw options appended to `bishengir-compile` via `--append-bisheng-options=...`. Use to pass arbitrary compiler flags not exposed as structured options. | `--append-bisheng-options=<str>` |
| `debug` | `bool` | `False` | Dump intermediate IR files (`.ttir.mlir`, `.ttadapter.mlir`, `.npuir.mlir`) and print compile commands | `--bishengir-print-ir-after=hivm-graph-sync-solver` |
| `extern_libs` | `dict` | `None` | External libraries to link | — |
| `stream` | `int` | `None` | CANN stream handle for async dispatch | — |

### Hardware Target

| Option | Type | Default | Description |
|---|---|---|---|
| `arch` | `str` | `""` (auto-detected) | Target architecture string (e.g. `"ascend910b"`, `"ascend910_93"`). Auto-filled from `GPUTarget.arch`. |
| `compile_on_910_95` | `bool` | auto | Use 910_95 compilation path. Auto-detected from environment. |

### Precision & Numerics

| Option | Type | Default | Description |
|---|---|---|---|
| `sanitize_overflow` | `bool` | `True` | Enable overflow sanitization |
| `default_dot_input_precision` | `str` | `"ieee"` | Default precision for `tl.dot` inputs: `"ieee"` or `"hf32"` |
| `allowed_dot_input_precisions` | `tuple` | `("ieee","hf32")` | Allowed precision strings for dot ops |
| `max_num_imprecise_acc_default` | `int` | `0` | Max number of imprecise accumulations allowed |
| `allow_fp8e4nv` | `bool` | `False` | Allow FP8 e4m3nv dtype (910_95 only) |
| `supported_fp8_dtypes` | `tuple` | `("fp8e5","fp8e4b15","fp8e4nv","fp8e4b8","fp8e5b16")` | FP8 dtypes recognized by the backend |
| `enable_fp_fusion` | `bool` | `True` | Enable floating-point fusion (e.g. FMA formation) |

---

## Hardware Notes

| Fact | Detail |
|---|---|
| `copy`, `fixpipe`, FP8, FP64 | 910_95 series only — raise `RuntimeError` on other HW |
| `bind_sub_block=True` in `parallel` | Max 2 vector cores on 910B |
| `sub_vec_num()` | Queries hardware at compile time via `NPUUtils` |
| Default int width | Python `int` args → `int32` on device; use `al.int64(x)` for 64-bit |
