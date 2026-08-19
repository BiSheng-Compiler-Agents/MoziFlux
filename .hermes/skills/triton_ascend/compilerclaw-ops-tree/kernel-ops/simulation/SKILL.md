---
name: simulation
description: Run Triton-Ascend kernels without a physical NPU using cannsim. Covers compilation (ttir→ttadapter→npubin), host C++ launcher, cannsim invocation, and trace analysis. General knowledge only — per-kernel case studies and session-specific data go in references/ or episodes.
tags: [triton, ascend, cannsim, simulation, npu]
required_plugins:
  - cannsim-local
required_environment_variables:
  - CANNSIM_SETENV_PATH
  - CANNSIM_SOC_VERSION
  - CONDA_BIN
  - CONDA_ENV
---

# Cannsim Simulation [LEAF NODE]

> **Skill design principle**: This SKILL.md contains only general, reusable knowledge for any agent/user/platform. Per-kernel case studies, session-specific trace data, and dated findings go in `references/` or episodes — not here. Keep concise and task-focused.

Run Triton-Ascend kernels without a physical NPU using the CANN cannsim simulator.
Covers compilation (ttir → ttadapter → npubin), host C++ launcher authoring, and cannsim
invocation.

## CRITICAL WORKFLOW RULES

### Rule 1 — Always use a sub-kernel host for cannsim

cannsim simulates every instruction cycle-by-cycle. Simulation time scales linearly with
instruction count = grid_size × loop_iters × instr_per_tile. A full-shape run is
impractical (e.g. 4096×4096 GEMM takes ~1500s). Use a sub-kernel host instead: same
.npubin, grid=(1,1,1), minimal loop iters.

Sub-kernel pattern:
- **grid = (1, 1, 1)** — one block is enough to see the bottleneck
- **M = BLOCK_M, N = BLOCK_N** — one tile of data
- **K = 1×BLOCK_K** (single iteration) — minimizes cannsim instruction count; use 2× only if single iteration is insufficient to exercise the bottleneck
- **BLOCK_M/BLOCK_N/K constexpr values must NOT change** — compiled into .npubin
- Allocate buffers sized for exactly 1 tile

**Fair A/B when tuning BLOCK_K:** if the optimization changes `BLOCK_K`, compare both kernels at the same absolute probe `K` (for example `max(old_BLOCK_K, new_BLOCK_K)` or an LCM), even though this gives the smaller-BLOCK kernel multiple loop iterations. Do not compare `K=1×old_BLOCK_K` against `K=1×new_BLOCK_K`; that conflates optimization with different mathematical work.

What is preserved: bottleneck pipeline lane, WAIT_FLAG stall patterns, effect of any code fix.
What is lost: absolute cycle count (irrelevant), multi-block L2 cache effects.

FFTS dispatch savings are invisible at sub-kernel scale. If testing a persistent/work-stealing
grid optimization, the sub-kernel trace will show identical cycles for baseline and optimized —
the FFTS benefit is purely a dispatch-level effect requiring full-shape hardware to measure.

### Rule 2 — Always use trace files, not cycle counts

cannsim.log cycle counts are chip-level wall latency only. They tell you nothing about *why*
a kernel is slow. **All optimization decisions must be based on `trace_core0.json`.**

Trace-first workflow:
1. Write a sub-kernel host (grid=1, M=BLOCK_M, K=1×BLOCK_K)
2. Run cannsim record → generate trace_core0.json
3. Run `python scripts/aggregate_trace.py trace_core0.json` → trace_summary.txt
4. Analyze: pipeline breakdown (by_cat %), WAIT_FLAG stalls, dominant bottleneck
5. Apply fix → re-run → compare traces

---

## Prerequisites

CANN toolkit must be installed and `CANNSIM_SETENV_PATH` must point to the CANN setenv.bash
script. The cannsim binary will be on PATH after sourcing. Conda env name defaults to
`compilerclaw` but should be set via `CONDA_ENV`.

Required env vars: `CANNSIM_SETENV_PATH`, `CANNSIM_SOC_VERSION`, `CONDA_BIN`, `CONDA_ENV`.

---

## Required Triton Patches

Two patches to triton-ascend are needed for compilation without a physical NPU. Both are
idempotent and auto-applied by the cannsim-local/cannsim-remote plugins.

**Patch 1** — `tools/get_ascend_devices.py`: Add `TRITON_ASCEND_ARCH` env var check so
compilation works without hardware detection.

**Patch 2** — `backends/ascend/compiler.py`: Use `get_ascend_arch_from_env()` instead of
querying hardware for the `--target` flag.

If compiling manually without the plugins, verify these patches are applied before proceeding.

---

## Step 1 — Write the Kernel Compile Script

Write a Python compile script (`compile_kernel.py`) into the local job directory.
Use `result.asm["npubin"]` to extract the binary from the compile result — more reliable
than `TRITON_KERNEL_DUMP`.

Place a `CMakeLists.txt` and `test_kernel.cpp` (host launcher) alongside it.
Use `templates/run_kernel.sh` as the build wrapper.

**Mandatory setup rules:**
1. `TRITON_ASCEND_ARCH = "Ascend910_9589"` — set BEFORE importing triton
2. Clear `~/.triton/cache` BEFORE importing triton — otherwise cache hits bypass dump
3. Use PID-unique DUMP_DIR (`_triton_dump_<pid>`) — avoids stale hits across runs

```python
import os, glob, shutil, pathlib

SCRIPT_DIR  = str(pathlib.Path(__file__).parent.resolve())
DUMP_DIR    = os.path.join(SCRIPT_DIR, "_triton_dump_" + str(os.getpid()))
NPUBIN_DEST = os.path.join(SCRIPT_DIR, "my_kernel.npubin")

os.environ["TRITON_KERNEL_DUMP"]  = "1"
os.environ["TRITON_DUMP_DIR"]     = DUMP_DIR
os.environ["TRITON_ASCEND_ARCH"]  = "Ascend910_9589"
os.environ["TRITON_COMPILE_ONLY"] = "1"

import shutil as _shutil
_cache_dir = os.path.expanduser("~/.triton/cache")
if os.path.isdir(_cache_dir):
    _shutil.rmtree(_cache_dir, ignore_errors=True)

import triton
import triton.language as tl
from triton.compiler import compile, ASTSource
from triton.backends.compiler import GPUTarget

@triton.jit
def my_kernel(...):
    ...

result = compile(
    ASTSource(fn=my_kernel, signature={...}, constants={"BLOCK": 128}),
    target=GPUTarget("npu", "Ascend910_9589", 32),
)

# Prefer c.asm["npubin"] over TRITON_KERNEL_DUMP — more reliable
data = result.asm["npubin"]
with open(NPUBIN_DEST, "wb") as f:
    f.write(data)
```

See `references/cannsim-compile-npubin-asm-dict.md` for why `c.asm["npubin"]` is more
reliable than `TRITON_KERNEL_DUMP` on some triton-ascend versions.

If your compile script needs `torch` for helper functions, make sure the CANN environment
is sourced first (`source <cann>/bin/setenv.bash`). This prevents `torch_npu` autoload from
calling `aclInit` at import time.

### Using `@triton.autotune`

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}),
    ],
    key=["m", "n", "k"],
)
@triton.jit
def my_kernel(..., BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    ...
```

**CRITICAL:** `ASTSource` cannot wrap an `Autotuner` object. Extract the underlying JIT function:
```python
_jit = my_kernel.fn  # .fn holds the original @triton.jit function
src = ASTSource(fn=_jit, signature={...}, constants={...})
```

### ASTSource signature must exactly match kernel parameters

- Pointer parameters → `"*fp32"` (or `"*bf16"`, `"*fp16"`)
- Integer scalars (dims, strides) → `"i32"`
- `tl.constexpr` parameters → go in `constants` dict, NOT in `signature`
- Every non-constexpr parameter must appear in `signature`; every constexpr in `constants`

---

## Step 2 — Host C++ Launcher

### Use ONLY `rt*` APIs — never `acl*`

| ACL call | RT equivalent |
|---|---|
| `aclrtSetDevice(id)` | `rtSetDevice(id)` |
| `aclrtCreateStream(&s)` | `rtStreamCreate(&s, 0)` |
| `aclrtMalloc(&p, sz, flag)` | `rtMalloc(&p, sz, RT_MEMORY_HBM, 0)` |
| `aclrtMemcpy(dst,sz,src,sz,dir)` | `rtMemcpy(dst,sz,src,sz, RT_MEMCPY_HOST_TO_DEVICE)` |
| `aclrtSynchronizeStream(s)` | `rtStreamSynchronize(s)` |
| `aclrtFree(p)` | `rtFree(p)` |
| `aclrtDestroyStream(s)` | `rtStreamDestroy(s)` |
| `aclrtResetDevice(id)` | `rtDeviceReset(id)` |

### Binary magic

Triton kernels use `RT_DEV_BINARY_MAGIC_ELF_AIVEC`. `RT_DEV_BINARY_MAGIC_ELF` also works
but produces ~2× more trace events (same execution, more verbose instrumentation).

### KernelArgs struct layout

```cpp
struct __attribute__((packed)) KernelArgs {
    void*   syncBlockLock;   // null — NOTE: camelCase, not snake_case
    void*   workspace_addr;  // null
    // kernel args: pointers first, then scalars, in fn signature order
    void*   a_ptr;
    // ... other args ...
    int32_t n;
    // grid dims last
    int32_t gridX, gridY, gridZ;
};
```

### Sub-kernel host pattern

```cpp
#include "runtime/rt.h"
#include <libgen.h>   // dirname() — required on GCC 14+
// BLOCK_M/N/K are constexpr compiled into .npubin — do NOT change them
const int M = BLOCK_M;       // one tile
const int N = BLOCK_N;
const int K = 1 * BLOCK_K;   // single iteration — minimizes cannsim time
const int gridX = 1;

// npubin sits in same directory as the binary
std::string binPath = std::string(dirname(argv[0])) + "/my_kernel.npubin";

rtDevBinary_t devBin;
devBin.magic  = RT_DEV_BINARY_MAGIC_ELF_AIVEC;
devBin.data   = kernelData.data();
devBin.length = kernelData.size();
// ... register, allocate, launch ...

// Use _exit(0) to bypass CANN 9.0.0 atexit teardown segfault
printf("[HOST] PASS\n");
fflush(stdout);
_exit(0);
```

**Dtype matching**: kernel signature dtype must match C++ host buffer dtype exactly.
If the kernel uses `*fp16`, host buffers must be `uint16_t`, not `float`. See
`references/cannsim-cpp-host-dtype-matching.md` for conversion helpers.

**Build**: see `references/cpp-host-build-pitfalls.md` for GCC 14+ `dirname()` include,
`-Wl,--allow-shlib-undefined`, and fp16 conversion UB fixes.

### CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.10)
project(test_kernel CXX)
set(CMAKE_CXX_STANDARD 17)

if(NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
  set(ASCEND_PATH $ENV{ASCEND_HOME_PATH})
else()
  set(ASCEND_PATH "/usr/local/Ascend/cann")
endif()

include_directories(
  ${ASCEND_PATH}/include
  ${ASCEND_PATH}/x86_64-linux/pkg_inc
  ${ASCEND_PATH}/x86_64-linux/pkg_inc/runtime
  ${ASCEND_PATH}/x86_64-linux/pkg_inc/profiling
  ${ASCEND_PATH}/x86_64-linux/pkg_inc/toolchain
)

add_executable(test_kernel test_kernel.cpp)
target_link_libraries(test_kernel PRIVATE ${ASCEND_PATH}/lib64/libruntime.so)
target_link_options(test_kernel PRIVATE -Wl,--allow-shlib-undefined)
```

Build: `mkdir -p build && cd build && cmake .. -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE "-DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined" && make -j$(nproc)`

---

## Step 3 — Run with cannsim

### Local (`cannsim_local_run`)

```python
result = cannsim_local_run(
    local_dir="path/to/local_dir",
    run_script="run_kernel.sh",
    binary_name="test_kernel",
    gen_report=True,
    timeout=1800,
)
# Returns: trace_local_path, trace_json_size_bytes, cannsim_log_tail, ...
# Full trace_json content is opt-in: pass return_trace_json=True if needed.
```

### Remote (`cannsim_remote_run`)

Requires `CANNSIM_REMOTE_HOST`, `CANNSIM_REMOTE_USER`, `CANNSIM_REMOTE_PASS` in `~/.hermes/.env`.

```python
result = cannsim_remote_run(
    local_dir="path/to/local_dir",
    run_script="run_kernel.sh",
    binary_name="test_kernel",
    gen_report=True,
    timeout=600,
)
```

### Manual run

```bash
source /path/to/cannsim/setenv.bash
cannsim record -s Ascend950 -- ./test_kernel
cannsim report -e . -o ./report -n 0   # -e points at dir containing log_ca/
```

If `cannsim report` fails with "instr log file is not found", check BOTH the job root
and the experiment subdir for `instr.bin`/`log_ca/`. cannsim writes them relative to CWD.

---

## Step 4 — Analyze Trace

```bash
python scripts/aggregate_trace.py /path/to/trace_core0.json
# Writes trace_summary.txt next to the input file
```

### Pipeline vocabulary

| CANNSIM pipe | Meaning |
|---|---|
| VECTOR / RVECEX | Vector execution unit |
| SCALAR | Scalar execution unit |
| CUBE / AIC | Matrix multiply execution unit |
| MTE1 | Data movement: L1 → {L0A/L0B, UBUF} |
| MTE2 | Data movement: {DDR/GM, L2} → {L1, L0A/B, UBUF} |
| MTE3 | Data movement: UBUF → {DDR/GM, L2, L1} |
| FIXP | Data movement: FIXPIPE L0C → OUT/L1 |
| FLOWCTRL | Control-flow instructions |
| ICACHELOAD | ICache miss activity |
| PUSHQ | Queue push / instruction dispatch pressure |
| RVECLD | Vector-side local-buffer load activity |
| RVECST | Vector-side local-buffer store activity |
| RVECSU | Vector support activity |

**BOTTLENECK**: pipeline with highest busy_cyc.
**CRITICAL**: instruction with highest total_cyc OR avg_cyc ≥ 25% of wall-clock.

### Cycle → time conversion

`hardware_time_ns = cycles × 0.4` (ref_period = 0.40 ns/cycle)

---

## Key Optimization Patterns

### `tl.dot(a, b, acc)` in-place accumulation

Standard `acc += tl.dot(a, b)` creates a 64 KB intermediate tile (for 128×128 fp32).
`tl.dot(a, b, acc)` accumulates inside Cube hardware, eliminating the temporary.
Typical gain: −36% wall_cycles, −98% RVECEX, −33% RVECST.

### `tl.range` vs `while` loop

`while` loops with advancing pointers create a SCALAR storm (SHL, ADD_IMM, CMP_IMM per
iteration). `tl.range(0, tl.cdiv(K, BLOCK_K))` with index-based offset recomputation
eliminates all advancing-pointer SCALAR ops. Typical: −89% SCALAR, −99% PUSHQ.

### `al.multibuffer` — use with caution

`al.multibuffer(tensor, size=2)` double-buffers in UB. Combined with fp32 accumulator
(64 KB for 128×128), total UB can exceed the ~128 KB hardware limit. cannsim may succeed
while real hardware silently produces wrong results. Prefer `tl.dot(a, b, acc)` in-place
accumulation instead — larger gain, zero UB overflow risk.

### UB size budget

AIV Unified Buffer ≈ 32 KB per core. For 3 live fp32 tensors:
`3 × BLOCK_HW × 4 bytes ≤ 32,768 → BLOCK_HW ≤ 2048`.
BLOCK_HW=4096 with fp32 causes silent UB overflow (empty trace, ~28 cycles, 100% SCALAR).

### `care_padding=False` on `tl.load`

Safe for matmul (zero-input tiles produce zero contribution to `tl.dot`). Unsafe for
reductions (zeros affect mean/count) and softmax (`exp(0)=1` before normalize → wrong).

---

## Pitfalls

0. **Trace file location**: `trace_core0.json` is written to the `report/` subdirectory of the cannsim job dir (e.g., `/tmp/cannsim_local/<job_name>/report/trace_core0.json`), NOT the job root. Always check `report/` first, or use `find /tmp/cannsim_local/ -name "trace_core0.json"`.

0. **Cannsim output too large for tool context**: `cannsim_local_run` output can exceed 200K characters (236 KB+), which exceeds tool result size limits. When this happens, read trace files directly from disk via `read_file` or terminal instead of relying on the tool return value.

0. **Unsafe early-exit wrapper recovery**: `cannsim_local_run` may return failure with `UNSAFE EARLY EXIT` after `instr.bin` is already produced, especially when the host binary exits before the wrapper's quiet-period heuristic is satisfied. Do not discard the run immediately: inspect `cannsim.log` for `[HOST] Kernel completed` / `[HOST] PASS` or simulator finish markers, then manually run `cannsim report -e . -o ./report -n 0` in the job directory. If `trace_core0.json` is generated and `aggregate_trace.py` succeeds, document that the wrapper failed but the report trace was recovered.

0. **Skill name disambiguation**: When loading skills by short name (e.g., `skill_view(name='optimization')`, plugin-registered skills may cause ambiguity (4+ matches). Always use the fully qualified name: `triton_ascend/compilerclaw-ops-tree/kernel-ops/optimization`.

0. **`#include <unistd.h>` required for `_exit()`**: GCC 14+ requires explicit `#include <unistd.h>` for `_exit()`. Add it to every cannsim C++ host — it's portable and needed to bypass CANN 9.0.0 atexit segfault. See `references/cpp-host-build-pitfalls.md`.

1. **`TRITON_ASCEND_ARCH`**: compile with `Ascend910_9589`; cannsim `-s` with `Ascend950`
2. **Triton patches**: cannsim-local plugin auto-applies both patches (get_ascend_devices.py + compiler.py)
3. **npubin path**: use `dirname(argv[0]) + "/my_kernel.npubin"` (same directory as the binary). Do NOT use `"../my_kernel.npubin"` — it breaks when cannsim runs the binary directly from the job root.
4. **`rtFunctionRegister` name**: must match the Python `def` name of the `@triton.jit` kernel, NOT the .npubin filename
5. **`syncBlockLock`**: camelCase in C++ struct, not `sync_block_lock`
6. **CANN 9.0.0 `rtFunctionRegister` host pattern**: some headers do not define `rtFunction_t`; use `static size_t func_stub = 0; rtFunctionRegister(handle, &func_stub, "kernel_name", (void*)"kernel_name", 0); rtKernelLaunch(&func_stub, ...)`. See `references/cpp-host-build-pitfalls.md`.
7. **CANN 9.0.0 atexit segfault**: use `_exit(0)` in host binary after successful simulation
7. **`[HOST]` diagnostic logging**: print `[HOST] Launching kernel...` before `rtKernelLaunch`, `[HOST] Kernel completed` after successful sync, and `[HOST] PASS` after correctness check. These markers are the diagnostic pattern that `cannsim_subkernel_timeout.md` says to check for in cannsim.log when debugging timeouts. Always use the `[HOST]` prefix for these specific markers; other informational prints can use `[INFO]`.
8. **`cache_modifier=".cg"`**: silently kills compilation on Ascend — never use
8. **`num_stages=1`**: causes scalar div-by-zero crash — always use `num_stages=2`
9. **`tl.compile_hint`**: does not exist in `triton.language` — use `al.compile_hint` from `triton.language.extra.cann.extension`
10. **`tl.multiple_of` / `tl.max_contiguous`**: pass values matching tensor dimensionality (1 value for scalar pointer, 2 for 2D tensor)
11. **instr.bin location**: cannsim writes to CWD (job root), not experiment subdir; check both
12. **C++ comments**: never use `#` at start of line in `.cpp` files — use `//`
13. **CMake `ASCEND_PATH`**: no stray closing quote in `set(ASCEND_PATH $ENV{ASCEND_HOME_PATH})`
14. **Stale CMakeCache**: always `rm -rf build` before rebuilding in cannsim_local_run
15. **False PASS**: use non-zero test data; zero-initialized output masks bugs. Use correctness checks in the host launcher: compute expected values from input pattern, convert bf16 output to fp32 via bit-shift, and fail with per-element diagnostics. See `references/bf16-fp32-conversion.md` for the conversion helper.
16. **Self-contained local_dir**: `cannsim_local_run` stages only `local_dir` into `/tmp/cannsim_local/<job>/`. Compile scripts must not rely on parent-relative paths such as `Path(SCRIPT_DIR).parent / "opt_*.py"` unless that parent file is also copied into `local_dir`; otherwise the build fails after staging. Either copy the needed Python source into the cannsim work dir or make `compile_kernel.py` fully self-contained.
17. **CANN env not sourced**: `run_kernel.sh` must source the CANN environment (`CANNSIM_SETENV_PATH` or `$ASCEND_HOME_PATH/bin/setenv.bash`) BEFORE the compile step (torch_npu/ACL headers) and the cmake step (runtime/rt.h include paths).
17. **CANN fp16 type**: CANN defines `fp16_t` (a struct wrapping `uint16_t`) in `fp16.h`. For raw host buffer storage, `uint16_t` is correct. If you need a typed fp16 variable, use `fp16_t`. See `references/cannsim-cpp-host-dtype-matching.md`.
18. **Cannsim sub-kernel timeout**: On machines with <32 GB RAM or <16 CPU cores, cannsim may time out even with a sub-kernel host. Mitigations: (a) use K=1×BLOCK_K (single iteration) instead of 2×BLOCK_K to halve instruction count, (b) use smaller matrix blocks (BLOCK_M/N=64 instead of 128) to reduce per-iteration work 4×, (c) for scalar-heavy vector epilogues such as activation+pool, shrink the diagnostic host much more aggressively (e.g. one NC tile, `BLOCK_HW=16`, tiny `H/W`) because the goal is bottleneck classification, not full tile throughput. Diagnostic: after cannsim record, search cannsim.log for `[HOST] Kernel completed` and `[HOST] PASS`. If missing, the kernel did not finish. See `references/cannsim_subkernel_timeout.md`.

---

## Constraints

- **Skill content policy**: Only general, reusable knowledge goes in this SKILL.md. Per-kernel case studies, session-specific trace data, and dated findings ("confirmed June 2026") go in `references/` or episodes — not here. Keep concise.
- **Always use a sub-kernel host** — grid=(1,1,1), M=BLOCK_M, K=1×BLOCK_K (single iteration). Full-shape runs are impractical.
- Always use `gen_report=True` — cycle counts alone are not actionable
- Only `parallel_mode = "simd"` kernels work in cannsim
- Never upload pre-built binaries — always build on the target machine
- `TRITON_COMPILE_ONLY=1` must NOT be set when running under cannsim

---

## Reference Examples

- **`./templates/compile_kernel_template.py`** — Canonical compile script with all mandatory rules
- **`./templates/cmake_template.txt`** — Canonical CMakeLists.txt
- **`./templates/test_kernel_template.cpp`** — Canonical sub-kernel host launcher
- **`./references/vector_add/`** — Simplest cannsim workflow (start here)
- **`./references/fused_softmax/`** — Reduction + normalization kernel example
- **`./references/rt-dev-binary-magic-elf.md`** — All 4 magic constants and hardware units
- **`./references/cannsim-trace-file-location.md`** — Trace file location diagnosis and large output handling
- **`./references/cannsim-instr-bin-location.md`** — instr.bin location diagnosis
- **`./references/subkernel-vs-multiblock-comparison.md`** — Sub-kernel vs multi-block validation
- **`./references/cannsim-compile-npubin-asm-dict.md`** — `c.asm["npubin"]` alternative to `TRITON_KERNEL_DUMP`
- **`./references/cannsim-cpp-host-dtype-matching.md`** — Kernel/host dtype matching and conversion helpers
- **`./references/cpp-host-build-pitfalls.md`** — GCC 14+ build issues and fp16 conversion UB
- **`./references/cannsim_subkernel_timeout.md`** — Sub-kernel cannsim timeout mitigation (K=1×BLOCK_K, smaller blocks)
- **`./templates/run_kernel.sh`** — Canonical build wrapper (compile → build → copy npubin)

### Shared references (in `../../shared/references/`)
- **`ascend-terminology.md`** — Hardware terminology: AI Core (Cube + Vector), memory hierarchy (GM/UB/L1), HIVM IR mapping table, pipeline stages. Useful for understanding cannsim trace output and pipeline vocabulary.
