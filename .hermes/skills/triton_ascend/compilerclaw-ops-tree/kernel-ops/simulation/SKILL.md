---
name: simulation
description: Run Triton-Ascend kernels without a physical NPU using cannsim. Covers compilation, host C++ launcher, cannsim invocation, and trace analysis.
tags: [triton, ascend, cannsim, simulation, npu]
required_plugins:
  - cannsim-local
required_environment_variables:
  # cannsim-local plugin
  - CANNSIM_BIN
  - CANNSIM_SOC_VERSION
  - CANNSIM_SETENV_PATH
  - CONDA_BIN
  - CONDA_ENV
  # kernel-episodes plugin (used by optimization/episode-memory)
  - KERNEL_EPISODES_DB

# Cannsim Simulation [LEAF NODE]

Run Triton-Ascend kernels without a physical NPU using the CANN cannsim simulator.
Covers compilation (ttir → ttadapter → npubin), host C++ launcher authoring, and cannsim
invocation. Tested with triton-ascend==3.2.1 + CANN 9.0.0 + Ascend910_9589 target.

## CRITICAL WORKFLOW RULES

> ⛔ **READ THIS FIRST.** Rule 1 below is non-negotiable. A full-shape cannsim run can
> take 25+ minutes for a 4096×4096 matmul because simulation time scales linearly with
> grid_size × loop_iters × instr_per_tile. The sub-kernel pattern (grid=1, M=BLOCK_M,
> N=BLOCK_N, K=2×BLOCK_K) brings the same run to ~85 seconds and gives identical
> bottleneck diagnosis. The user has explicitly called out violations of this rule.
> **If you ever write a cannsim host that runs grid>1 or allocates buffers >1 tile, you
> are doing it wrong.**

### Rule 1 — Always use a sub-kernel host for cannsim

> **cannsim simulates every instruction cycle-by-cycle. Simulation time scales
> linearly with instruction count = grid_size × loop_iters × instr_per_tile.
> A full-shape run is impractical (e.g. 4096×4096 GEMM takes ~1500s).
> Use a sub-kernel host instead: same .npubin, grid=(1,1,1), minimal loop iters.**
>
> Sub-kernel rule:
> - **grid = (1, 1, 1)** — one block is enough to see the bottleneck
> - **M = BLOCK_M, N = BLOCK_N** (or equivalent tile size) — one tile of data
> - **K = 2×BLOCK_K** (or 4× if you want to see steady-state pipelining) — minimum loop iters
> - **BLOCK_M/BLOCK_N/BLOCK_K constexpr values must NOT change** — they are compiled
>   into the .npubin and determine the instruction mix. Only runtime shape args change.
> - Allocate buffers sized for exactly 1 tile (tiny — bytes, not MB)
>
> What is preserved: bottleneck pipeline lane (MTE2/RVECEX/SCALAR/CUBE),
> WAIT_FLAG stall patterns, effect of any code fix, UB overflow symptoms.
> What is lost: absolute cycle count (irrelevant), multi-block L2 cache effects.
>
> **FFTS dispatch savings are invisible at sub-kernel scale.**
> If the optimization being tested is a persistent/work-stealing grid (reducing
> program count from N/B → min(N/B, 65535) to amortise per-program FFTS cost),
> the sub-kernel trace will show IDENTICAL cycles for baseline and optimized —
> because both run exactly 1 program with 1 loop iteration. The FFTS benefit
> (~1,150 cy × saved_programs) is purely a dispatch-level effect that requires
> a full-shape hardware run to measure. Do not conclude the optimization failed
> from an identical sub-kernel trace. The trace is still useful for verifying
> the per-tile instruction mix has not regressed.
>
> Speedup example: grid 1024→1, K_iters 32→2 = **16384× faster**.
> A 1500s run becomes ~0.1s.

### Rule 2 — Always use trace files, not cycle counts

> **The user has explicitly stated: "the cycle counts you may get from cannsim.log
> is worthless — you should base your decision only based on trace files."**
>
> cannsim.log cycle counts are chip-level wall latency only.
> They tell you nothing about *why* a kernel is slow.
> **All optimization decisions must be based on `trace_core0.json`.**
>
> Trace-first workflow (using `cannsim_remote_run`):
> 1. Write a **sub-kernel host** (grid=1, M=BLOCK_M, K=2×BLOCK_K — see Rule 1)
> 2. Call `cannsim_remote_run(..., gen_report=True)` — this runs `cannsim record`,
>    then automatically runs `cannsim report -e <exp_dir> -o <exp_dir>/report -n 0`,
>    downloads `trace_core0.json` to a local temp path, and returns it in
>    `result["trace_local_path"]` and inline in `result["trace_json"]`.
> 2. Analyze the trace: compute span, pipeline breakdown (by_cat %), WAIT_FLAG stalls
> 3. Identify the dominant bottleneck from the trace
> 4. Apply the fix
> 5. Re-run `cannsim_remote_run`, compare traces

---

## Prerequisites

| Requirement | Version |
|---|---|
| triton-ascend | 3.2.1 |
| CANN | 9.0.0 |
| cannsim | ships with CANN 9.0.0 |
| Host RAM | ≥ 32 GB (Ascend950 camodel is heavy) |

CANN toolkit must be installed and `CANNSIM_SETENV_PATH` must point to the CANN set_env.sh
script. The cannsim binary will be on PATH after sourcing. Conda env name defaults to
`compilerclaw` but should be set via `CONDA_ENV`.

Remote machine (if using `cannsim_remote_run`): set `CANNSIM_REMOTE_HOST`,
`CANNSIM_REMOTE_USER`, `CANNSIM_REMOTE_PASS` in `~/.hermes/.env`. Remote should have
GCC 11.4+ (GLIBCXX up to 3.4.30) — always build on remote.

---

## Required Patches to triton-ascend 3.2.1

These two patches must be applied once to any fresh triton-ascend 3.2.1 installation.
They allow compilation and execution without a physical NPU.

> Both patches are idempotent. The `cannsim-remote` Hermes plugin applies them
> automatically on the remote machine before each run. The `cannsim-local` plugin
> also applies them automatically on the local machine.
File: `$(python -c "import triton; print(triton.__file__.replace('__init__.py',''))")tools/get_ascend_devices.py`

```python
# Replace the last line:
#   is_compile_on_910_95 = pci_condition or npu_smi_condition
# With:
env_condition = os.getenv("TRITON_ASCEND_ARCH", "").strip().lower() in (
    "ascend910_9589", "ascend910b", "ascend950", "ascend910_95"
)
is_compile_on_910_95 = pci_condition or npu_smi_condition or env_condition
```

### Patch 2 — `compiler.py`: use `TRITON_ASCEND_ARCH` instead of querying hardware

File: `$(python -c "import triton; print(triton.__file__.replace('__init__.py',''))")backends/ascend/compiler.py`

**Change the import block:**
```python
# before
from triton.backends.ascend.driver import (
    NPUUtils
)
# after
from triton.backends.ascend.driver import (
    NPUUtils,
    get_ascend_arch_from_env,
)
```

**Change the `--target` line:**
```python
# before
f"--target={NPUUtils().get_arch()}",
# after
f"--target={get_ascend_arch_from_env() or NPUUtils().get_arch()}",
```

> Both patches are idempotent. The `cannsim-remote` Hermes plugin applies them
> automatically on the remote machine before each run.

---

## Step 1 — Write the Kernel Compile Script

Write a Python compile script (`compile_kernel.py`) into the local job directory.
**Do not run it on the local machine** — it will be invoked on the remote via `run_script`.

```python
# compile_kernel.py — placed in local_dir, run on remote by run_kernel.sh
import os, glob, shutil, pathlib

SCRIPT_DIR = str(pathlib.Path(__file__).parent.resolve())
DUMP_DIR   = os.path.join(SCRIPT_DIR, "_triton_dump")
NPUBIN_DEST = os.path.join(SCRIPT_DIR, "my_kernel.npubin")

os.environ["TRITON_KERNEL_DUMP"]  = "1"
os.environ["TRITON_DUMP_DIR"]     = DUMP_DIR
os.environ["TRITON_ASCEND_ARCH"]  = "Ascend910_9589"
os.environ["TRITON_COMPILE_ONLY"] = "1"

import triton
import triton.language as tl
from triton.compiler import compile, ASTSource
from triton.backends.compiler import GPUTarget

@triton.jit
def my_kernel(..., BLOCK: tl.constexpr):
    ...

compile(
    ASTSource(fn=my_kernel, signature={...}, constants={"BLOCK": 1024}),
    target=GPUTarget("npu", "Ascend910_9589", 32),
)

npubin = sorted(glob.glob(os.path.join(DUMP_DIR, "**", "my_kernel.npubin"), recursive=True))[0]
shutil.copy2(npubin, NPUBIN_DEST)
print(f"[COMPILE] npubin written to {NPUBIN_DEST}")
```

The compilation pipeline: `ast → ttir → ttadapter → npubin`

---

## Step 2 — Host C++ Launcher

### CRITICAL: Use ONLY `rt*` APIs — do NOT use `acl*` APIs

cannsim prepends the camodel directory to `LD_LIBRARY_PATH`, swapping
`libruntime.so` → `libruntime_camodel.so`. `libascendcl.so` is NOT in the
camodel dir — `aclInit` always hits the real driver and fails.

| ACL call | RT equivalent |
|---|---|
| `aclInit(nullptr)` | *(not needed)* |
| `aclrtSetDevice(id)` | `rtSetDevice(id)` |
| `aclrtCreateStream(&s)` | `rtStreamCreate(&s, 0)` |
| `aclrtMalloc(&p, sz, flag)` | `rtMalloc(&p, sz, RT_MEMORY_HBM, 0)` |
| `aclrtMemcpy(dst,sz,src,sz,dir)` | `rtMemcpy(dst,sz,src,sz, RT_MEMCPY_HOST_TO_DEVICE / DEVICE_TO_HOST)` |
| `aclrtSynchronizeStream(s)` | `rtStreamSynchronize(s)` |
| `aclrtFree(p)` | `rtFree(p)` |
| `aclrtDestroyStream(s)` | `rtStreamDestroy(s)` |
| `aclrtResetDevice(id)` | `rtDeviceReset(id)` |

### Binary magic — must match `mix_mode`

| mix_mode | Magic constant |
|---|---|
| `"aiv"` (vector) | `RT_DEV_BINARY_MAGIC_ELF_AIVEC` ← **use this for Triton kernels** |
| `"aic"` (cube) | `RT_DEV_BINARY_MAGIC_ELF` |

### Args struct layout (packed, matches triton-ascend driver)

```cpp
struct __attribute__((packed)) KernelArgs {
    void*   syncBlockLock;   // null
    void*   workspace_addr;  // null
    // kernel args: pointers first, then scalars, in fn signature order
    void*   x_ptr;
    // ... other args ...
    int32_t n;
    // grid dims last
    int32_t gridX;
    int32_t gridY;
    int32_t gridZ;
};
```

### Sub-kernel host pattern (ALWAYS use this for cannsim)

The host launcher must use sub-kernel dimensions — one tile, one block, minimal loop
iterations. This makes simulation take seconds instead of hours.

**What to set vs what to keep:**
- Change: M=BLOCK_M, N=BLOCK_N, K=2*BLOCK_K (or 4* for steady-state pipelining), gridX=1
- Keep: BLOCK_M/BLOCK_N/BLOCK_K constants (they are compiled into .npubin — do not change)
- Keep: one kernel launch only (no benchmark loop)

**Persistent / work-stealing kernels**: if the kernel uses a `while tile_id * BLOCK_SIZE < n_elements`
loop, set `n_elements = BLOCK_SIZE` (exactly one tile) AND `n_programs = 1`. The loop then runs
exactly one iteration — same per-tile instruction mix, minimal simulation time. Do NOT set
`n_programs = n_elements` (that leaves `n_programs` at full-shape, which changes nothing for grid=1
but is semantically misleading).

```cpp
#include "runtime/rt.h"

// Sub-kernel dimensions — override full-shape args with one-tile sizes
// BLOCK_M/N/K are constexpr compiled into the .npubin — do NOT change them.
// Only the runtime shape arguments (M, N, K) and gridX change.
const int BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 64;  // must match kernel constexprs
const int M = BLOCK_M;           // one tile row
const int N = BLOCK_N;           // one tile col
const int K = 2 * BLOCK_K;       // 2 loop iters (use 4 to see steady-state pipelining)
const int gridX = 1;             // one block — enough to see bottleneck

size_t sizeA = M * K * sizeof(float);
size_t sizeB = K * N * sizeof(float);
size_t sizeC = M * N * sizeof(float);

rtSetDevice(0);
rtStream_t stream;
rtStreamCreate(&stream, 0);

// Load npubin — path relative to argv[0] (cannsim changes CWD)
std::string binPath = std::string(dirname(argv[0])) + "/my_kernel.npubin";
// ... read file into kernelData ...

rtDevBinary_t devBin;
devBin.magic  = RT_DEV_BINARY_MAGIC_ELF_AIVEC;
devBin.data   = kernelData.data();
devBin.length = kernelData.size();
void* binHandle = nullptr;
rtDevBinaryRegister(&devBin, &binHandle);

static size_t funcStub = 0;
rtFunctionRegister(binHandle, &funcStub, "my_kernel", (void*)"my_kernel", 0);

void *aDev, *bDev, *cDev;
rtMalloc(&aDev, sizeA, RT_MEMORY_HBM, 0);
rtMalloc(&bDev, sizeB, RT_MEMORY_HBM, 0);
rtMalloc(&cDev, sizeC, RT_MEMORY_HBM, 0);
rtMemcpy(aDev, sizeA, aHost.data(), sizeA, RT_MEMCPY_HOST_TO_DEVICE);
rtMemcpy(bDev, sizeB, bHost.data(), sizeB, RT_MEMCPY_HOST_TO_DEVICE);

// One launch only — no warmup/repeat loops (each launch multiplies simulation time)
KernelArgs args = { nullptr, nullptr, aDev, bDev, cDev, M, N, K, gridX, 1, 1 };
rtKernelLaunch(&funcStub, gridX, &args, sizeof(args), nullptr, stream);
rtStreamSynchronize(stream);

rtMemcpy(cHost.data(), sizeC, cDev, sizeC, RT_MEMCPY_DEVICE_TO_HOST);
rtFree(aDev); rtFree(bDev); rtFree(cDev);
rtStreamDestroy(stream); rtDeviceReset(0);
```

> Correctness check: even with M=BLOCK_M, K=2*BLOCK_K you can verify correctness
> by comparing cHost against a reference computed with the same tiny sub-problem.
> Use non-zero test data (e.g. random floats) to avoid false PASS from zero output.

### CMakeLists.txt — link ONLY libruntime.so

```cmake
if(NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
  set(ASCEND_PATH $ENV{ASCEND_HOME_PATH})    # NO closing quote after ASCEND_HOME_PATH
else()
  set(ASCEND_PATH "/usr/local/Ascend/cann")
endif()

include_directories(
  ${ASCEND_PATH}/include
  ${ASCEND_PATH}/x86_64-linux/pkg_inc
  ${ASCEND_PATH}/x86_64-linux/pkg_inc/runtime
  ${ASCEND_PATH}/x86_64-linux/pkg_inc/profiling    # required: prof_common.h
  ${ASCEND_PATH}/x86_64-linux/pkg_inc/toolchain    # required: prof_api.h
)

target_link_libraries(my_target PRIVATE
  ${ASCEND_PATH}/lib64/libruntime.so
  # DO NOT link libascendcl.so
)
```

> **`prof_common.h` build error** — `runtime/base.h` transitively includes
> `toolchain/prof_api.h` which includes `prof_common.h`. On CANN 9.0.0 (conda install)
> this header lives at `x86_64-linux/pkg_inc/profiling/prof_common.h`, not in `toolchain/`.
> Fix: add BOTH `profiling/` and `toolchain/` to `include_directories`.
> Symptom: `fatal error: prof_common.h: No such file or directory` at compile line with
> `#include "runtime/rt.h"`.
> Discovery: `find ~/miniconda3/Ascend/cann-9.0.0/ -name 'prof_common.h'` returns
> `.../x86_64-linux/pkg_inc/profiling/prof_common.h`.

Build:
```bash
source ~/miniconda3/Ascend/cann/bin/setenv.bash
mkdir build && cd build
cmake .. -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE \
         -DCMAKE_EXE_LINKER_FLAGS="-Wl,--allow-shlib-undefined"
make -j$(nproc)
```

> Add `-DCMAKE_EXE_LINKER_FLAGS="-Wl,--allow-shlib-undefined"` to cmake.
> `libruntime.so` has transitive dependencies (`liberror_manager.so`, `libmmpa.so`)
> that are not present at link time but ARE provided by the camodel at runtime.

---

## Step 3 — Run with cannsim (remote or local)

### Option A: Remote (`cannsim_remote_run`) — requires SSH access

Local directory layout:
```
local_dir/
├── compile_kernel.py      ← kernel compile script (run on remote by run_kernel.sh)
├── run_kernel.sh          ← build-on-demand wrapper
├── CMakeLists.txt         ← cmake build file for the C++ host
├── test_my_kernel.cpp     ← RT-only C++ host (rt* APIs only, no acl*)
```

Call `cannsim_remote_run` once:

```python
result = cannsim_remote_run(
    local_dir="path/to/local_dir",
    run_script="run_kernel.sh",
    binary_name="test_my_kernel",
    job_name="my_kernel",
    gen_report=True,   # runs cannsim report automatically, downloads trace_core0.json
    timeout=600,
)
# On success:
# result["trace_local_path"]  — local path to trace_core0.json
# result["trace_json"]        — first 200 KB of trace inline
# result["cannsim_log_tail"]  — last 4000 chars of cannsim.log
```

### Option B: Local (`cannsim_local_run`) — no SSH needed

Same workflow but runs entirely on the local machine. Requires CANN toolkit
installed and `CANNSIM_SETENV_PATH` set to the CANN set_env.sh script, plus the
conda env with triton-ascend installed (`CONDA_BIN` and `CONDA_ENV`).

```python
result = cannsim_local_run(
    local_dir="path/to/local_dir",
    run_script="run_kernel.sh",
    binary_name="test_my_kernel",
    job_name="my_kernel",
    gen_report=True,
    timeout=1800,
)
# Same return shape as cannsim_remote_run, plus "patch_log" and "job_dir"
```

The local plugin (`cannsim-local`) automatically:
- Sources `CANNSIM_SETENV_PATH` (the CANN set_env.sh) for CANN env vars
- Applies the same triton patches (get_ascend_devices.py + compiler.py) so
  compilation works without a physical NPU
- Detects the CANN 9.0.0 cleanup bug (exit code 1 after successful simulation)
- Resolves cannsim from the sourced PATH (no hardcoded paths)

### `cannsim_remote_run` / `cannsim_local_run` parameter reference

| Parameter | Required | Default | Description |
|---|---|---|---|
| `local_dir` | yes | — | Dir with sources/binary + npubin + run_kernel.sh |
| `run_script` | yes | — | Shell wrapper filename (e.g. `run_kernel.sh`) |
| `binary_name` | no | `test_kernel` | Name of compiled binary cannsim wraps |
| `job_name` | no | basename+timestamp | Subdirectory name |
| `soc_version` | no | Ascend950 | cannsim -s value |
| `gen_report` | no | True | When True, runs `cannsim report` and returns trace_core0.json |
| `timeout` | no | 1800 | Timeout for cannsim record step (seconds) |
| `report_timeout` | no | 300 | Timeout for cannsim report step (seconds) |

> `local_dir` is the upload root — `run_script` must resolve inside it.
> Set `local_dir` to the directory that *contains* `run_kernel.sh`,
> and set `run_script` to just `"run_kernel.sh"`.

Returns: `success`, `job_name`, `job_dir`/`remote_job_dir`, `experiment_dir`,
`patch_log`, `build_log`, `cannsim_log_tail`, `report_log`, `trace_local_path`,
`trace_json`, `trace_truncated`.

---

## Step 4 — Extract Kernel Runtime and Analyze Trace

### Analyzing trace_core0.json

The trace_core0.json file contains a full execution trace — DO NOT read it fully into context.
Instead, run the aggregation script:

```bash
python scripts/aggregate_trace.py /path/to/trace_core0.json
```

The script writes its output to a `trace_summary.txt` file **next to the input file** (same
directory as the input, with `.txt` extension replacing `.json`). Read that file after running.

> **Note**: Earlier versions of this skill incorrectly stated the output always goes to
> `/tmp/trace_summary.txt`. The actual behavior is: output goes to the same directory as
> the input file. Always check next to your input.

The `aggregate_trace.py` script is available at:
`./scripts/aggregate_trace.py` (tree-internal, relative to this SKILL.md)

### Ascend 910_95 NPU Architecture — pipeline vocabulary

| CANNSIM pipe | Meaning |
|---|---|
| VECTOR / RVECEX | Vector execution unit |
| SCALAR | Scalar execution unit |
| Cube / AIC | Matrix multiply execution unit |
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

**BOTTLENECK annotation**: pipeline with highest busy_cyc.
**CRITICAL annotation**: instruction with highest total_cyc OR per-event avg_cyc ≥ 25% of wall-clock.

### cycle → hardware time conversion

```
ref_period = 0.40 ns/cycle  (confirmed from camodel init log)
hardware_time_ns = cycles × 0.4
```

Example: 248 cycles × 0.4 ns = **99.2 ns**

### Perfetto trace timing note

| Source | What it measures |
|---|---|
| `cannsim.log` `cycle: 248` | Chip-level wall-clock latency (all blocks in parallel) |
| `trace_core0.json` span | Core 0 instruction timeline (its share of blocks) |
| Perfetto "84.90 ms" | Sum of all pipeline lane durations — NOT wall time |

**If trace_core0.json shows only SCALAR work (tiny span, 0 RVECEX):**
Core 0 may not have been assigned any kernel programs. Try other cores:
```bash
for N in 0 1 2 3 4 5; do
    cannsim report -e <exp_dir> -o <exp_dir>/report_c${N} -n $N
done
```

---

## AIV Hardware Findings (Empirically Verified)

### fp16 vs fp32 for `tl.maximum` (ReLU)

`tl.maximum(x.to(tl.float16), 0.0)` → routes to **VEC** fixed-function unit (slow, WAIT_FLAG stalls).

**Fix:**
```python
x_f32 = x.to(tl.float32)
y_f32 = tl.maximum(x_f32, 0.0, propagate_nan=tl.PropagateNan.ALL)
y = y_f32.to(tl.float16)
```
Routes to **RVECEX** which pipelines with MTE2/MTE3.

Also: `propagate_nan=tl.PropagateNan.ALL` collapses the 3-op manual NaN pattern
(`tl.maximum + x!=x + tl.where`) to a single hardware instruction.
Confirmed: VCMP_NE and VSEL are completely absent from the fp32 optimized trace (l1_19_ReLU, June 2026).

**l1_19_ReLU sub-kernel trace comparison (BLOCK_SIZE=4096, grid=1):**
| Metric | Baseline (fp16 max) | Optimized (fp32 max) |
|--------|---------------------|----------------------|
| wall_cycles | 3373 | 3319 (-1.6%) |
| WAIT_FLAG_VEC (MTE3) | 1226 cy | 1151 cy (-6.1%) |
| WAIT_FLAG_MTE2 (VEC) | 1002 cy | 986 cy |
| NaN ops (RVECEX) | VCMP_NE+VMAXS+VSEL (×64 each) | absent (propagate_nan=ALL) |
| New ops | — | RV_VCVT_F2F×128 (fp16↔fp32 casts, 7 cy/op) |

Note: sub-kernel delta understates production gain — FFTS dispatch savings (persistent grid)
are invisible at grid=1 scale.

### `tl.zeros` inside `tl.range` loop

Problem: one WAIT_FLAG_VEC per outer loop step, serializing memory and compute.

**Fix:** Hoist accumulator initialization before the loop:
```python
acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)  # hoist outside
for k in tl.range(K, ...):
    acc += tl.dot(tl.load(...), tl.load(...))
```

### UB size limits for fp32

AIV Unified Buffer ≈ **32KB** per core.
For 3 live fp32 tensors: `3 × BLOCK_HW × 4 bytes ≤ 32,768 → BLOCK_HW ≤ 2048`.

BLOCK_HW=4096 causes **silent UB overflow**:
- `instr.bin` ~800KB (nearly empty) vs 391MB for correct
- trace shows only SCALAR work
- kernel silently reads/writes garbage

### `num_stages=1` crash

Always use `num_stages=2`. `num_stages=1` causes scalar div-by-zero inside the AIV core at runtime.

### ST_XD_XN_IMM scalar spill — structural cost in triton-ascend

Across baseline (BLOCK=32) and optimized (BLOCK=128) matmul traces, ST_XD_XN_IMM appears
as the top CRITICAL instruction at ~592 avg_cyc regardless of tile size. This is a structural
cost of triton-ascend code generation: pointer arithmetic intermediates (especially from
GROUP_M swizzle: modulo + division on program IDs) get spilled to the scalar register file.
Total cost: 75 stores × 592 cyc = 44,418 cycles — **exceeds CUBE compute at 4,540 cycles**.

This cost is not reducible from Triton Python. It's a codegen artifact. When you see
ST_XD_XN_IMM CRITICAL in a trace, don't spend optimization effort there — focus on
WAIT_FLAG stalls and CUBE utilization instead.

### Startup cost vs compute cost (small-HW regime)

Per-program AIV startup cost ≈ 1,150 cycles. For HW ≤ ~1024, startup dominates.
**Fix:** persistent kernel — launch 32 programs that each loop over N*C/32 tiles.
Threshold empirically: HW ≤ 1024 → persistent; HW > 1024 → loop kernel.

---

## Pitfalls

**cannsim compile/build setup pitfalls (must get right):**
- `TRITON_ASCEND_ARCH` for compile: MUST be `Ascend910_9589` (not `Ascend950` — the triton libdevice.py validates this and only accepts Ascend910_xxx values)
- `soc_version` for cannsim_remote_run / cannsim_local_run: MUST be `Ascend950` (cannsim only supports Ascend950, not Ascend910_9589)
- Must clear triton cache before compile: `shutil.rmtree(os.path.expanduser("~/.triton/cache"), ignore_errors=True)` before `import triton` — otherwise cache hits bypass `TRITON_KERNEL_DUMP` and no .npubin is written
- Use PID-unique DUMP_DIR: `_triton_dump_<os.getpid()>` pattern to avoid stale hits across runs
- Binary must be at job root for cannsim: add `cp "$BUILD_DIR/binary" "$SCRIPT_DIR/binary"` in run_kernel.sh after make

- **`run_kernel.sh` must NOT call `conda activate` — the plugin already runs the script inside the right env.**
  The plugins invoke `run_kernel.sh` via `conda run -n compilerclaw bash run_kernel.sh build`.
  If the script also calls `conda activate compilerclaw` (or `source conda.sh && conda activate`), the inner
  activation silently shadows the outer env and `triton` is not found:
  ```
  ModuleNotFoundError: No module named 'triton'
  ```
  **Fix: the only conda-related line in `run_kernel.sh` should be sourcing CANN:**
  ```bash
  # CORRECT — only source CANN, do NOT conda activate:
  source "$HOME/miniconda3/Ascend/cann/bin/setenv.bash"

  # WRONG — do not add these lines:
  # source "$HOME/miniconda3/etc/profile.d/conda.sh"
  # conda activate compilerclaw
  ```

Handled automatically by both `cannsim-remote` and `cannsim-local` plugins (no action needed):
- `-o <dir>` on `cannsim record` — plugins never pass `-o` (causes CWD issues and broken instr.bin)
- `-g` flag on `cannsim record` — plugins never pass `-g` (redundant; we always call report ourselves)
- `conda: command not found` — plugins find conda via full path
- `~` not expanded by paramiko SFTP (remote) — plugin resolves `$HOME` over SSH
- Stale job dir — plugins do `rm -rf` before each run
- GLIBCXX version mismatch (remote) — plugins never upload pre-built binaries
- triton patches — plugins auto-apply both patches (idempotent)
- **CANN 9.0.0 false failure** — cannsim record exits code 1 after a successful simulation because `_cleanup_user_env` calls `os.getcwd()` on a directory it already deleted (`FileNotFoundError`). Plugins detect this via the `current_dir = os.getcwd()` + `FileNotFoundError` + `_cleanup_user_env` + `all tasks are finished!` pattern and auto-recovers by continuing to the report step. Fixed in plugin v2 (June 2026).
- **Timeout** — with sub-kernel hosts (grid=1, M=BLOCK_M) simulation takes seconds; default 1800s is always sufficient. If you ever run a full-shape host (not recommended), 4096×4096 GEMM takes ~1500s — pass `timeout=3600`.

Kernel/host code pitfalls (still require attention):
1. **`cache_modifier=".cg"` silently kills compilation** — This CUDA L2-bypass hint causes `triton.compiler.compile()` to produce zero output (no .npubin, no error) on Ascend. Symptom: `_triton_dump/` directory is empty after `compile()` returns without raising. Never use `cache_modifier=".cg"` in Triton kernels targeting Ascend. Drop it unconditionally.
2. `do_issue_vector_instr not support mix task type` — wrong binary magic (`ELF` instead of `ELF_AIVEC`) or Patch 1 not applied
3. `Cannot open kernel binary` — .npubin must sit next to host binary; use `dirname(argv[0])`
4. cannsim OOM-killed — camodel needs ≥32 GB RAM; always use `cannsim_remote_run`
5. `simt` kernels crash — only `parallel_mode = "simd"` is supported
6. `aclInit failed 500000` — host uses `acl*` APIs; rewrite to `rt*` only
7. `torch_npu` crashes — `import torch` calls `aclInit` at import time; never import torch in cannsim host scripts
8. `TRITON_COMPILE_ONLY=1` must NOT be set when running under cannsim
9. BLOCK_HW=4096 fp32 causes UB overflow → empty kernel (symptom: ~28 cycles, 100% SCALAR)
10. `num_stages=1` + hoisted `tl.zeros` → scalar div-by-zero crash; use `num_stages=2`
11. False PASS from zero-initialized output — always use non-zero test data AND add correctness check
12. **Sub-kernel buffer sizing** — with M=BLOCK_M, K=2*BLOCK_K, buffers are tiny (e.g. 128×128×4=64KB). If you accidentally allocate full-shape buffers but pass sub-kernel dims, the kernel still works but you waste memory. Size buffers to match the sub-kernel dims exactly.
13. **Sub-kernel K too small** — K=1*BLOCK_K (only 1 loop iteration) may not show steady-state pipeline overlap between MTE2 and RVECEX. Use K=2*BLOCK_K minimum; use 4*BLOCK_K if overlap is not visible.
14. **al.multibuffer compilation errors** — `al.multibuffer(tensor, size=2)` is a side-effect hint only. Two common crashes:
    - `a = al.multibuffer(a, size=2)` then `tl.dot(a, b)` → `AttributeError: NoneType has no attribute 'type'` — multibuffer returns None; reassigning breaks tl.dot
    - `al.compile_hint(a, "dot_pad_only_k")` AFTER `a = al.multibuffer(a, ...)` → `AttributeError: NoneType has no attribute 'handle'`
    Fix: call `al.compile_hint` before `al.multibuffer`, and never reassign the multibuffer return.
15. **1D-grid kernels with GROUP_M swizzle** — if the kernel remaps a 1D program ID to
    (pid_m, pid_n) using NUM_PID_M/NUM_PID_N args at runtime, set NUM_PID_M=1 and NUM_PID_N=1
    alongside gridX=1. The swizzle degenerates to tile (0,0), which is correct.
    Do NOT leave NUM_PID_M/N at full-shape values with gridX=1 — the swizzle will compute
    out-of-bounds tile indices. The kernel signature must also carry these as i32 args.
16. The trace_core0.json file contains a full execution trace for all events which is too large of a data dump. So, DO NOT attempt to read that fully into your context. 
    Instead, run the accompanying aggregation/ summarizing script as below, which will output a condensed summary of the key metrics in a human/LLM readable format.
    ```bash
    python scripts/aggregate_trace.py /path/to/trace_core0.json
    ```
17. **Existing baseline hosts may have a full-size run PLUS a correctness-check run** — when
    creating a sub-kernel variant, strip BOTH (the 256x256 correctness check and the full 512x512
    run) and replace with a single sub-kernel launch that also does its own correctness check.
    Leaving the 256x256 correctness check in place causes cannsim to simulate that too (256x256
    grid = 64 programs → much slower than 1-program sub-kernel).
18. **Sub-kernel K must match between `compile_kernel.py` and the C++ host** — the .npubin is
    compiled with `EXACT_K: tl.constexpr` baked in (determining the `static_range` trip count).
    The C++ host's `SUB_K` (e.g. `2 * BLOCK_K`) must equal the `EXACT_K` used at compile time,
    otherwise the runtime K doesn't match the compiled pipeline depth. Symptom: a 256x256
    correctness check passes by coincidence but the 1-tile main run produces wrong results, OR
    the trace shows a completely different per-tile instruction mix than expected. Always
    define `EXACT_K` and `SUB_K` together in a single source of truth (e.g. a comment block
    that both files reference) and update both at once.
19. **`tl.compile_hint` does NOT exist on this triton-ascend** — `triton.language.compile_hint`
    raises `AttributeError: module 'triton.language' has no attribute 'compile_hint'` at
    runtime. The real API is `al.compile_hint` from
    `triton.language.extra.cann.extension`. A common mistake is to add a shim
    `tl.compile_hint = lambda tensor, hint: tensor` in `compile_kernel.py` (which makes
    compilation succeed but is a no-op), and the kernel file then crashes when imported
    outside the compile context. **Always use `al.compile_hint` in the kernel file, and
    import it: `import triton.language.extra.cann.extension as al`.**
20. **CMakeLists.txt: `set(ASCEND_PATH $ENV{ASCEND_HOME_PATH}")` — stray closing quote is a
    parse error** — the existing baseline/CMakeLists.txt had a bug
    `set(ASCEND_PATH $ENV{ASCEND_HOME_PATH}")` with a stray `"` after `ASCEND_HOME_PATH`.
    When the env var is set (cannsim-remote always sets it), the if-branch runs and the bug
    triggers: `CMake Error: Parse error. Function missing ending ')'.` Build fails with the
    error pointing at line 34 of CMakeLists.txt. Fix: remove the stray quote so the line
    reads `set(ASCEND_PATH $ENV{ASCEND_HOME_PATH})`. The else-branch (env unset) works
    because the bug is only on the if-branch line.
21. **cannsim-local plugin: no hardcoded paths** — The cannsim-local plugin must NOT
    hardcode any filesystem paths. After sourcing `CANNSIM_SETENV_PATH`, `cannsim` will be
    on PATH — just use `shutil.which("cannsim")`. The conda env bin path must be derived
    from `CONDA_BIN` (via `dirname(dirname(CONDA_BIN))/envs/{env}/bin`), not hardcoded as
    `/opt/miniconda3/envs/...`. The `check_fn` must also avoid hardcoded paths — only check
    `shutil.which("cannsim")` and the `CANNSIM_BIN` env var. See
    `references/cannsim_local_no_hardcoded_paths.md` for the full before/after diff.
22. **`rtFunctionRegister` KERNEL_NAME mismatch → error `0x7bc78` and segfault** —
    `rtFunctionRegister(handle, &stub, "name", (void*)"name", 0)` requires the name to
    match the **Python function name** of the `@triton.jit` kernel you passed to
    `triton.compiler.ASTSource(fn=...)`. The `.npubin` filename is irrelevant — what
    matters is the symbol the compiler emitted, which equals the Python `def` name.

    Easy mistake: your compile script does
    ```python
    @triton.jit
    def _scale_kernel_direct(x_ptr, y_ptr, s, n_elements, BLOCK_SIZE: tl.constexpr):
        ...
    NPUBIN_DEST = os.path.join(SCRIPT_DIR, "scale_kernel.npubin")  # arbitrary filename
    ```
    Then in the C++ host:
    ```cpp
    #define KERNEL_NAME "scale_kernel"   # WRONG — this is the .npubin filename
    rtFunctionRegister(binHandle, &funcStub, KERNEL_NAME, (void*)KERNEL_NAME, 0);
    ```
    This compiles and links fine, but the launch fails with `rtFunctionRegister ERROR:
    0x7bc78`, then a SIGSEGV when cannsim tries to find the symbol. Fix: set
    `KERNEL_NAME` to the actual `@triton.jit` Python function name
    (e.g. `"_scale_kernel_direct"`).

    Quick check before launching: search the `.npubin` for symbols:
    ```bash
    strings scale_kernel.npubin | grep -i 'kernel\|scale' | head
    ```
    The function name appears as a string inside the binary. Match that exactly.

    Rule of thumb: in the C++ host, `#define KERNEL_NAME` to a copy of the Python
    `def` line, kept in sync across the two files. The `.npubin` filename is just a
    filename — rename it anything, it doesn't matter.

## Constraints
- **Always use a sub-kernel host** — grid=(1,1,1), M=BLOCK_M, K=2×BLOCK_K. Full-shape runs are impractical (minutes to hours). Sub-kernel gives identical bottleneck diagnosis in seconds.
- Always use `gen_report=True` — cycle counts alone are not actionable
- Never upload pre-built binaries — always build on remote (with `cannsim_remote_run`) or locally (with `cannsim_local_run`)
- Only `parallel_mode = "simd"` kernels work in cannsim; simt is broken
- Local CANN install: set `CANNSIM_SETENV_PATH` to the CANN set_env.sh script, set
  `CONDA_BIN` to the conda binary path, set `CONDA_ENV` to the conda env name.
  The cannsim binary will be on PATH after sourcing set_env.sh — do not hardcode its path.
- Remote cannsim machine: set `CANNSIM_REMOTE_HOST`, `CANNSIM_REMOTE_USER`, `CANNSIM_REMOTE_PASS`
  in `~/.hermes/.env`.

---

## Reference Trace Baselines

- **`./references/matmul_baseline_trace_analysis.md`** — Baseline trace for the standard
  matmul kernel (l1_2) at BLOCK_M/N=32, BLOCK_K=32, sub-kernel M=32/N=32/K=64, grid=1x1x1.
  wall_cycles=8919. Key findings: CUBE at 7.2% utilization, ST_XD_XN_IMM CRITICAL at
  593 avg cyc, FLOWCTRL bottleneck via SET_INTRA_BLOCKI. Use as baseline comparison for
  optimization runs.

- **`./references/matmul_opt_trace_analysis.md`** — Optimized matmul trace (l1_2) at
  BLOCK_M/N=128, BLOCK_K=32, GROUP_M=4 swizzle, hoisted masks, dot_pad_only_k.
  Sub-kernel: M=128, N=128, K=64, grid=1x1x1, NUM_PID_M=1, NUM_PID_N=1.
  wall_cycles=16848 (1.9× more cycles, ~8× more compute → ~8× throughput per tile).
  CUBE improved to 26.9% of wall. Dominant bottlenecks: SET_INTRA_BLOCKI (1247 avg cyc),
  WAIT_FLAG_VEC@MTE3 (2488 avg cyc), ST_XD_XN_IMM scalar spill (592 avg cyc, 44418 total).
  Full comparison table vs baseline inside the file.

## Complete Working Reference Examples

- **`./templates/compile_kernel_template.py`** — Canonical compile_kernel.py template
  with all three mandatory setup rules pre-applied: cache clear before import, PID-unique
  DUMP_DIR, TRITON_ASCEND_ARCH=Ascend910_9589. Copy and replace the kernel definition.
  Includes a diagnostic error message that lists the checklist on npubin-not-found.

- **`./references/pending_optimization_antipatterns.md`** — Two optimization anti-patterns
  discovered in l2_8 (June 2026) that need merging into optimization/SKILL.md:
  (1) Oversized BLOCK_SIZE + mask for small C → STI_XN_IMM scalar spills (fix: constexpr C + no-mask)
  (2) Additive bias before per-row reduction → separate Add GPU kernel (fix: bias_sum host scalar)

These files are part of this tree and contain full working kernel + host + run-script examples:

- **`./references/vector_add/`** — Simplest possible cannsim workflow (start here):
  - `vector_add.py` — Triton vector-add kernel compile script
  - `cannsim_host/test_vector_add.cpp` — RT-only C++ host launcher (156 lines)
  - `cannsim_host/CMakeLists.txt` — cmake build file (link libruntime.so only)
  - `run_kernel.sh` — build-on-demand wrapper (compile Python → copy npubin → cmake → cannsim)

- **`./references/fused_softmax/`** — More realistic kernel (reduction + normalization):
  - `fused_softmax.py` — Triton fused-softmax kernel compile script (123 lines)
  - `cannsim_host/test_fused_softmax.cpp` — RT-only C++ host (205 lines, includes correctness check)
  - `cannsim_host/CMakeLists.txt` — cmake build file
  - `run_kernel.sh` — build-on-demand wrapper
