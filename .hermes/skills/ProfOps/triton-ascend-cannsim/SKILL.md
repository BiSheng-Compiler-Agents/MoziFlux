---
name: triton-ascend-cannsim
description: >
  Run Triton-Ascend kernels without a physical NPU using the CANN cannsim
  simulator. Covers compilation (ttir → ttadapter → npubin), host C++ launcher
  authoring, and cannsim invocation. Tested with triton-ascend==3.2.1 +
  CANN 9.0.0 + Ascend910_9589 target.
tags: [triton, ascend, npu, cannsim, cann, simulation]
required_plugins:
  - cannsim-remote
  - cannsim-local
required_environment_variables:
  # cannsim-local plugin
  - CANNSIM_BIN
  - CANNSIM_SOC_VERSION
  - CANNSIM_SETENV_PATH
  - CONDA_BIN
  - CONDA_ENV
  # cannsim-remote plugin
  - CANNSIM_REMOTE_HOST
  - CANNSIM_REMOTE_USER
  - CANNSIM_REMOTE_PASS
  - CANNSIM_REMOTE_PORT
  - CANNSIM_REMOTE_BASE_DIR
  - CANNSIM_REMOTE_CONDA_ENV
metadata:
  hermes:
    requires_tools:
      - cannsim_remote_run
      - cannsim_local_run
    related_skills:
      - triton-ascend-optimization-patterns
      - kernel-episode-memory
      - triton-ascend-kernel-profiling
---

# Triton-Ascend → cannsim (no physical NPU)

## ⚠️ CRITICAL WORKFLOW RULE: Always use trace files, not cycle counts

> **The user has explicitly stated: "the cycle counts you may get from cannsim.log
> is worthless — you should base your decision only based on trace files."**
>
> cannsim.log cycle counts are chip-level wall latency only.
> They tell you nothing about *why* a kernel is slow — which pipeline unit
> is bottlenecked, what the stall patterns are, or what to fix.
> **All optimization decisions must be based on `trace_core0.json`.**
>
> Always run `cannsim report` after `cannsim record`.
> Always analyze the trace before deciding what to change. Never iterate
> on cycle counts alone.
>
> Trace-first workflow (using `cannsim_remote_run`):
> 1. Call `cannsim_remote_run(..., gen_report=True)` — this runs `cannsim record`,
>    then automatically runs `cannsim report -e <exp_dir> -o <exp_dir>/report -n 0`,
>    downloads `trace_core0.json` to a local temp path, and returns it in
>    `result["trace_local_path"]` and inline in `result["trace_json"]`.
> 2. Analyze the trace: compute span, pipeline breakdown (by_cat %), WAIT_FLAG stalls
> 3. Identify the dominant bottleneck from the trace
> 4. Apply the fix
> 5. Re-run `cannsim_remote_run`, compare traces

## ⭐ RULE 1: Always use a sub-kernel host (grid=1, M=BLOCK_M, K=2×BLOCK_K)

> **cannsim simulates every instruction cycle-by-cycle. Simulation time
> scales linearly with instruction count = grid_size × loop_iters × instr_per_tile.
> A full-shape 4096×4096 GEMM takes ~1500s. A sub-kernel takes seconds.**

**This is not optional. Never run a full-shape host through cannsim.**

Sub-kernel rules:
- **grid = (1, 1, 1)** — one block is enough to see the bottleneck
- **M = BLOCK_M, N = BLOCK_N** (or equivalent tile size) — one tile of data
- **K = 2×BLOCK_K** (or 4× for steady-state pipelining) — minimum loop iters
- **BLOCK_M/BLOCK_N/BLOCK_K constexpr values must NOT change** — they're compiled
  into the .npubin and determine the instruction mix. Only runtime shape args change.
- Allocate buffers sized for exactly 1 tile (tiny — bytes, not MB)

What is preserved: bottleneck pipeline lane (MTE2/RVECEX/SCALAR/CUBE),
WAIT_FLAG stall patterns, effect of any code fix, UB overflow symptoms.
What is lost: absolute cycle count (irrelevant), multi-block L2 cache effects.

**Speedup example**: grid 1024→1, K_iters 128→2 = **65,536× faster**.
A 1500s run becomes ~0.02s.

**FFTS dispatch savings are invisible at sub-kernel scale.** Persistent/work-stealing
grid optimizations (reducing program count from N/B → min(N/B, 65535)) will show
IDENTICAL cycles in the sub-kernel trace for baseline and optimized — because
both run exactly 1 program. The FFTS benefit (~1,150 cy × saved_programs) is
purely a dispatch-level effect that requires full-shape hardware to measure.
Do NOT conclude the optimization failed from an identical sub-kernel trace — the
trace is still useful for verifying the per-tile instruction mix has not regressed.

### Don't propagate pre-existing full-shape setups

If you find an existing `cannsim_opt/` or similar directory in a previous
session that uses full-shape (e.g. 4096×4096) launches, **do not copy that
pattern** into your new v2/v3 setup. The previous session was wrong about that
too — fix it for your new work, and the new sub-kernel harness is the canonical
form. Keep the old full-shape dir as a reference if needed, but always author
new sub-kernel harnesses from scratch following this rule.

### Sub-kernel C++ host pattern

```cpp
#include "runtime/rt.h"

// Sub-kernel dimensions — override full-shape args with one-tile sizes
// BLOCK_M/N/K are constexpr compiled into the .npubin — do NOT change them.
// Only the runtime shape arguments (M, N, K) and gridX change.
const int BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 32;  // must match kernel constexprs
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

> **Sub-kernel buffer sizing** — with M=BLOCK_M, K=2*BLOCK_K, buffers are tiny
> (e.g. 128×128×4=64KB). If you accidentally allocate full-shape buffers but
> pass sub-kernel dims, the kernel still works but you waste memory. Size
> buffers to match the sub-kernel dims exactly.
>
> **Sub-kernel K too small** — K=1*BLOCK_K (only 1 loop iteration) may not show
> steady-state pipeline overlap between MTE2 and RVECEX. Use K=2*BLOCK_K
> minimum; use 4*BLOCK_K if overlap is not visible.

---

## Prerequisites

| Requirement | Version |
|---|---|
| triton-ascend | 3.2.1 |
| CANN | 9.0.0 |
| cannsim | ships with CANN 9.0.0 |
| Host RAM on remote | ≥ 32 GB (Ascend950 camodel is heavy) |
| System build tools | `cmake`, `make`, `g++` (needed to compile C++ host launcher) |

> **Docker image**: System build tools must be installed via `apt-get` in the
> Dockerfile. The `compilerclaw` conda env does not include them. See
> `references/local_cannsim_plugin.md` for the UID/GID mismatch fix needed in
> the Dockerfile and `run_container.sh` so the hermes user inside the container
> can write to the conda env.

---

## Required patches to triton-ascend 3.2.1

These two patches must be applied once to any fresh triton-ascend 3.2.1 installation.
They allow compilation and execution without a physical NPU.

### Patch 1 — `get_ascend_devices.py`: honour `TRITON_ASCEND_ARCH` env var

File: `$(python -c "import triton; print(triton.__file__.replace('__init__.py',''))")tools/get_ascend_devices.py`

Add `env_condition` so `is_compile_on_910_95 = True` when arch is set via env.
Without this, Triton picks the wrong `bishengir-compile` → wrong task-type metadata
→ `do_issue_vector_instr not support mix task type` in cannsim.

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

`NPUUtils().get_arch()` calls `rtGetSocVersion` which fails without a physical NPU.
`get_ascend_arch_from_env` is already defined in `driver.py` — just needs wiring up.

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

**Change the `--target` line (search for `NPUUtils().get_arch()`):**
```python
# before
f"--target={NPUUtils().get_arch()}",
# after
f"--target={get_ascend_arch_from_env() or NPUUtils().get_arch()}",
```

> Both patches are idempotent. The `cannsim-remote` Hermes plugin applies them
> automatically on the remote machine before each run.

---

## Step 1 — Write the kernel compile script (do NOT run it locally)

Write a Python compile script (`compile_kernel.py`) into the local job directory.
**Do not run it on the local machine** — it requires a CANN environment and a
910_95-compatible triton install. It will be invoked on the remote via `run_script`.

The script sets the required env vars, compiles the kernel using `triton.compiler.compile`,
and copies the resulting `.npubin` next to itself (so the C++ host binary can find it):

```python
import os
import glob, shutil, pathlib, subprocess

SCRIPT_DIR  = str(pathlib.Path(__file__).parent.resolve())
DUMP_DIR    = os.path.join(SCRIPT_DIR, "_triton_dump_" + str(os.getpid()))
NPUBIN_DEST = os.path.join(SCRIPT_DIR, "my_kernel.npubin")

# ⚠️ ORDERING CRITICAL: these four env vars MUST be set before `import triton`.
# Importing triton first causes ERR99999 ("UNKNOWN application exception") and
# an empty dump dir — no npubin is produced, with no clear error message.
os.environ["TRITON_KERNEL_DUMP"]  = "1"
os.environ["TRITON_DUMP_DIR"]     = DUMP_DIR
os.environ["TRITON_ASCEND_ARCH"]  = "Ascend910_9589"
os.environ["TRITON_COMPILE_ONLY"] = "1"

# Clear triton cache to force fresh dump (stale cache bypasses TRITON_KERNEL_DUMP)
import shutil as _shutil
_cache_dir = os.path.expanduser("~/.triton/cache")
if os.path.isdir(_cache_dir):
    _shutil.rmtree(_cache_dir, ignore_errors=True)

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

npubin = sorted(glob.glob(os.path.join(DUMP_DIR, "**", "my_kernel.npubin"), recursive=True))
if not npubin:
    r = subprocess.run(["find", DUMP_DIR, "-type", "f"], capture_output=True, text=True)
    print(f"[COMPILE] Dump dir contents:\n{r.stdout}")
    raise FileNotFoundError(f"No npubin found under {DUMP_DIR}")
shutil.copy2(npubin[0], NPUBIN_DEST)
print(f"[COMPILE] npubin written to {NPUBIN_DEST}")
```

The `run_kernel.sh` script must call this at build time before invoking the binary:

```bash
# Inside run_kernel.sh build step:
python "$SCRIPT_DIR/compile_kernel.py"
```

The compilation pipeline is:
```
ast → ttir → ttadapter → npubin
              (bishengir-compile via linalg_to_bin_enable_npu_compile_910_95)
```

---

## Step 2 — Host C++ launcher

### ⚠️ CRITICAL: Use ONLY `rt*` APIs — do NOT use `acl*` APIs

cannsim prepends the camodel directory to `LD_LIBRARY_PATH`, swapping
`libruntime.so` → `libruntime_camodel.so`. `libascendcl.so` is NOT in the
camodel dir — `aclInit` always hits the real driver and fails.

**The host binary must:**
- Include only `runtime/rt.h` (not `acl/acl.h`)
- Link only `libruntime.so` (not `libascendcl.so`)

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

### Minimal host pattern

```cpp
#include "runtime/rt.h"

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

void* xDev;
rtMalloc(&xDev, dataSize, RT_MEMORY_HBM, 0);
rtMemcpy(xDev, dataSize, xHost.data(), dataSize, RT_MEMCPY_HOST_TO_DEVICE);

KernelArgs args = { nullptr, nullptr, xDev, ..., N, gridX, 1, 1 };
rtKernelLaunch(&funcStub, blockNum, &args, sizeof(args), nullptr, stream);
rtStreamSynchronize(stream);

rtMemcpy(outHost.data(), dataSize, outDev, dataSize, RT_MEMCPY_DEVICE_TO_HOST);
rtFree(xDev); rtStreamDestroy(stream); rtDeviceReset(0);
```

### CMakeLists.txt — link ONLY libruntime.so

```cmake
if(NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
  set(ASCEND_PATH $ENV{ASCEND_HOME_PATH})
else()
  set(ASCEND_PATH "$ENV{HOME}/miniconda3/Ascend/cann-9.0.0")
endif()

# ⚠️ All four subdirs are required — omitting profiling/toolchain causes
# "prof_common.h: No such file or directory" and blocks compilation.
include_directories(
  ${ASCEND_PATH}/include
  ${ASCEND_PATH}/include/aclnn
  ${ASCEND_PATH}/include/experiment
  ${ASCEND_PATH}/include/experiment/msprof
  ${ASCEND_PATH}/x86_64-linux/pkg_inc
  ${ASCEND_PATH}/x86_64-linux/pkg_inc/profiling
  ${ASCEND_PATH}/x86_64-linux/pkg_inc/toolchain
  ${ASCEND_PATH}/x86_64-linux/pkg_inc/runtime
)

target_link_libraries(my_target PRIVATE
  ${ASCEND_PATH}/lib64/libruntime.so
  # DO NOT link libascendcl.so
)
```

Build:
```bash
source ~/miniconda3/Ascend/cann/bin/setenv.bash
mkdir build && cd build
cmake .. -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE \
         -DCMAKE_EXE_LINKER_FLAGS="-Wl,--allow-shlib-undefined"
make -j$(nproc)

# Verify — should print nothing:
ldd build/bin/my_binary | grep ascendcl
```

> ⚠️ Add `-DCMAKE_EXE_LINKER_FLAGS="-Wl,--allow-shlib-undefined"` to cmake.
> `libruntime.so` has transitive dependencies (`liberror_manager.so`, `libmmpa.so`)
> that are not present at link time but ARE provided by the camodel at runtime.
> Without this flag the linker emits `undefined reference to cce::runtime::...`
> errors that prevent the binary from building, even though it runs fine under cannsim.

> ⚠️ Build on the **same machine** that will run cannsim to avoid `GLIBCXX`
> version mismatches. The `run_kernel.sh` wrapper in the reference
> implementations builds on the remote automatically if no binary is found.

---

## Step 3 — Run with cannsim_remote_run

Once the local directory contains all required files:

```
local_dir/
├── compile_kernel.py      ← kernel compile script (written in Step 1, run on remote)
├── run_kernel.sh          ← build-on-demand wrapper: compiles kernel + C++ host, then runs binary
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
# result["cannsim_log_tail"]  — last 4000 chars of cannsim.log (look for [PASS]/cycle count)
```

The plugin handles everything: uploads the dir, patches triton on the remote,
runs the build step inside `run_kernel.sh`, runs `cannsim record`, runs
`cannsim report -e <exp_dir> -o <exp_dir>/report -n 0`, and downloads
`trace_core0.json` back to `result["trace_local_path"]`.


### Run on remote via `cannsim-remote` plugin

```python
cannsim_remote_run(
    local_dir="path/to/kernel_dir",   # contains sources + npubin + run_kernel.sh
    run_script="run_kernel.sh",
    job_name="my_kernel",
    gen_report=True,    # produces trace_core0.json for analysis
    timeout=600,        # cannsim simulations take 100-600s
)
```

> ⚠️ `gen_report` defaults to `False` in the plugin — always set it explicitly.

See **Remote cannsim execution** section below for full setup.

---

## Step 4 — Extract Kernel Runtime

> ⚠️ **User preference (enforced):** Base optimization decisions ONLY on
> `trace_core0.json` from `cannsim -g`. Cycle counts from `cannsim.log` are
> chip-level wall time only — they give no pipeline breakdown and cannot
> identify bottlenecks. Do not use cycle counts to justify or compare
> optimizations. Always get the trace first.

### ⭐ Fastest: read `cannsim.log` directly

After a successful run, `cannsim.log` contains:

```
SoC sub 0 1 all tasks are finished!
[Hardware] parallel simulation finish. sim time: 12.6s, cycle: 248, speed: 0.020KHz
SoC sub 0 3 all tasks are finished!
[Hardware] parallel simulation finish. sim time: 3.5s, cycle: 419, speed: 0.157KHz
```

- Pick the die with the **lowest cycle count** — that's the active die running the kernel
- Ignore idle dies (higher cycle counts from init/teardown)

**Cycles → hardware time** (`ref_period = 0.40 ns/cycle`, confirmed from camodel init log):
```
hardware_time_ns = cycles × 0.4
```
Example: 248 cycles × 0.4 ns = **99.2 ns**

> `sim time` is wall-clock simulation time (~3000× slower than hardware).
> Never use it as hardware time. Only use `cycle`.

### Programmatic: `log_ca/core*_summary_log`

After a successful run, `log_ca/core0_summary_log` contains:
```
kernal total ticks : <N>
system total ticks : <M>
```

Extract with:
```bash
grep "ticks" output/cannsim_*/log_ca/core0_summary_log
```

### Chrome Trace report

```bash
# Auto-generate during record (NO -o flag — see pitfall below):
cannsim record run_kernel.sh -s Ascend950
# cannsim writes output into: ./cannsim_<ts>_run_kernel.sh/{instr.bin, cannsim.log}

# Post-hoc report generation (use the timestamped experiment dir as -e):
cannsim report -e ./cannsim_<timestamp>_run_kernel.sh -o ./cannsim_<timestamp>_run_kernel.sh/report -n 0
```

Produces `report/trace_core0.json`

**If trace_core0.json shows only SCALAR work (tiny span, 0 RVECEX):**
Core 0 may not have been assigned any kernel programs. Try other cores:
```bash
for N in 0 1 2 3 4 5; do
    cannsim report -e <exp_dir> -o <exp_dir>/report_c${N} -n $N
done
# Check which core has RVECEX > 0 and a large span
```
The kernel runs on whichever AIV cores FFTS assigned. With 128 programs on 32 cores,
most cores run 4 programs each — find any core with substantial RVECEX to get the
representative trace. Core 0 is not guaranteed to be representative.

### Analyzing trace_core0.json for bottlenecks, instruction cycle breakdown etc

The trace_core0.json file contains a full execution trace for all events, which is too large of a data dump.
So, DO NOT attempt to read that fully into your context. Instead, run the accompanying aggregation/ summarizing script as below,
which will output a condensed summary of the key metrics in a human/LLM readable format.
```bash
python scripts/aggregate_trace.py /path/to/trace_core0.json
```

The script writes its output to a `trace_summary.txt` file **next to the input file** (same
directory as the input). Read that file after running.

> **Note**: Earlier versions of this skill incorrectly stated the output always goes to
> `/tmp/trace_summary.txt`. The actual behavior is: output goes to the same directory as
> the input file. Always check next to your input.

### Reading the trace summary and information about the Ascend 910_95 / A5-class NPU architecture

**1. Execution model**

Ascend AI Cores are not CUDA-style SIMT cores. They use heterogeneous execution resources with explicit data movement and visible pipeline queues.

Main resources:

* VECTOR / AIV: vector arithmetic, elementwise ops, reductions, masks, comparisons, selects, and special vector functions.
* Cube / AIC: matrix multiply / tensor compute, such as matmul or MMAD.
* SCALAR / FLOWCTRL: address calculation, control flow, waits, barriers, loop control, and synchronization.
* MTE: Memory Transfer Engine. Data-movement pipelines between global memory, cache, and local on-chip buffers.
* PUSHQ / dispatch: queue push and issue pressure into execution pipes.

**2. CANNSIM pipe vocabulary**

| CANNSIM pipe    | Meaning                                                  |
| --------------- | -------------------------------------------------------- |
| VECTOR / RVECEX | Vector execution unit                                    |
| SCALAR          | Scalar execution unit                                    |
| Cube / AIC      | Matrix multiply execution unit                           |
| MTE1            | Data movement: L1 → {L0A/L0B, UBUF}                      |
| MTE2            | Data movement: {DDR/GM, L2} → {L1, L0A/B, UBUF}          |
| MTE3            | Data movement: UBUF → {DDR/GM, L2, L1}, or L1 → {DDR/L2} |
| FIXP            | Data movement: FIXPIPE L0C → OUT/L1                      |
| FLOWCTRL        | Control-flow instructions                                |
| ICACHELOAD      | ICache miss activity                                     |
| PUSHQ           | Queue push / instruction dispatch pressure               |
| RVECLD          | Vector-side local-buffer load activity                   |
| RVECST          | Vector-side local-buffer store activity                  |
| RVECSU          | Vector support activity, such as masks or predicates     |

**3. Common instruction interpretation**

| Family / example              | Usual interpretation                              |
| ----------------------------- | ------------------------------------------------- |
| RV_VADD / RV_VCADD            | Vector add, accumulation, or reduction pressure   |
| RV_VMAX / RV_VCMAX            | Vector max or reduction pressure                  |
| RV_VCMP_* / RV_VSEL / RV_VAND | Masking, compare, select, predicate work          |
| RV_VLDI                       | Vector local-buffer load pressure                 |
| VF                            | Vector-function dispatch through PUSHQ            |
| MOV_* / DMA / COPY            | MTE data movement or alignment pressure           |
| LD_* / ST_* scalar forms      | Scalar memory, pointer, or index overhead         |
| WAIT_* / BAR / JUMP           | Synchronization or control-flow overhead          |
| MMAD / MATMUL-like ops        | Cube/AIC matrix-engine pressure                   |
| FIXP-related ops              | Fixed-pipe output movement from L0C toward OUT/L1 |

Note on annotations BOTTLENECK and CRITICAL:

* BOTTLENECK: the pipeline with the highest busy_cyc (most occupied pipeline in the trace window).
* CRITICAL: an instruction that either has the highest total_cyc across all top instructions, 
  or whose per-event average duration (avg_cyc) is ≥ 25% of total wall-clock cycles.

---

## Local cannsim execution via `cannsim-local` Hermes plugin

**Required plugin:** `cannsim-local`

Use this when cannsim runs on the same machine (no SSH). The plugin:
1. Auto-applies the triton patches locally (idempotent, skips if already applied)
2. Sources CANN environment from `CANNSIM_SETENV_PATH` (the CANN set_env.sh script)
3. Resolves `cannsim` from the sourced PATH — no hardcoded paths
4. Copies sources into a temp job dir (`/tmp/cannsim_local/<job_name>/`)
5. Runs the build step inside `run_kernel.sh` via `conda run -n <CONDA_ENV>`
6. Runs `cannsim record -s <soc>` locally
7. Runs `cannsim report -e <exp_dir> -o <exp_dir>/report -n 0` to produce `trace_core0.json`
8. Returns `trace_core0.json` content inline and its local path

> **⛑️ NO HARDCODED PATHS in the plugin.** After sourcing `CANNSIM_SETENV_PATH`, `cannsim`
> is on PATH — just use `shutil.which("cannsim")`. The conda env bin path is derived from
> `CONDA_BIN` (via `dirname(dirname(CONDA_BIN))/envs/{env}/bin`). The `check_fn` only checks
> `shutil.which("cannsim")` and the `CANNSIM_BIN` env var. Never hardcode `/opt/miniconda3/...`
> or any other absolute path in plugin code. See
> `references/cannsim_local_no_hardcoded_paths.md` for the full before/after diff.

### Tool: `cannsim_local_run`

| Parameter | Required | Default | Description |
|---|---|---|---|
| `local_dir` | yes | — | Dir with sources/binary + npubin + run_kernel.sh |
| `run_script` | yes | — | Shell wrapper filename (e.g. `run_kernel.sh`) |
| `binary_name` | no | `test_kernel` | Name of compiled binary cannsim wraps |
| `build_cmd` | no | `bash <run_script> build` | Custom build command |
| `job_name` | no | basename+timestamp | Temp subdirectory name under /tmp/cannsim_local/ |
| `soc_version` | no | Ascend950 | cannsim -s value |
| `gen_report` | no | true | When True, runs `cannsim report` and returns trace_core0.json |
| `timeout` | no | 1800 | Timeout for cannsim record step (seconds) |
| `report_timeout` | no | 300 | Timeout for cannsim report step (seconds) |

### Environment variables

| Variable | Required | Description |
|---|---|---|
| `CANNSIM_SETENV_PATH` | **yes** | Path to CANN set_env.sh — sources CANN env including cannsim on PATH |
| `CONDA_BIN` | **yes** | Path to conda binary — used to derive env bin path and run build |
| `CONDA_ENV` | no (default: `compilerclaw`) | Conda env name with triton-ascend installed |
| `CANNSIM_BIN` | no | Override path to cannsim binary (auto-detected from PATH if not set) |
| `CANNSIM_SOC_VERSION` | no (default: `Ascend950`) | cannsim -s value |

### Resolving cannsim binary

The plugin resolves cannsim in order:
1. `CANNSIM_BIN` env var (if set)
2. `shutil.which("cannsim")` using the PATH from the sourced CANN environment

That's it. No hardcoded fallback paths. If cannsim is not on PATH after sourcing
`CANNSIM_SETENV_PATH`, set `CANNSIM_BIN` explicitly.

### Local vs remote: when to use which

| Factor | `cannsim_local_run` | `cannsim_remote_run` |
|---|---|---|
| Machine | Local only | Remote SSH |
| RAM needed | ~8 GB (camodel in-process) | ≥ 32 GB recommended |
| Speed | No SSH overhead | SFTP upload + SSH exec |
| Use when | Quick iterations, small kernels | Large kernels, repeated runs |

### Pitfalls specific to local cannsim

- **cannsim binary not found** — ensure `CANNSIM_SETENV_PATH` points to a valid CANN
  set_env.sh. The plugin sources it and searches PATH. No hardcoded paths anywhere.
- **conda env not on PATH** — the plugin derives the env bin path from `CONDA_BIN`
  (not from a hardcoded `/opt/miniconda3/...` path). Ensure `CONDA_BIN` is set.
- **triton patches not applied** — the plugin applies them automatically. If you get
  `do_issue_vector_instr not support mix task type` or `rtGetSocVersion failed`
  when running manually, see "Required patches to triton-ascend 3.2.1" section above.

---

## Remote cannsim execution via `cannsim-remote` Hermes plugin

**Required plugin:** `cannsim-remote`

Use this when cannsim needs ≥32 GB RAM not available locally. The plugin:
1. Auto-applies the triton patches on the remote (idempotent, skips if already applied)
2. Cleans any stale job directory
3. Uploads the local kernel directory via SFTP
4. Runs the build step inside `run_kernel.sh` on the remote (compiles kernel + C++ host)
5. Runs `cannsim record -s Ascend950` on the remote
6. Runs `cannsim report -e <exp_dir> -o <exp_dir>/report -n 0` to produce `trace_core0.json`
7. Downloads `trace_core0.json` locally and returns it in `result["trace_local_path"]` and `result["trace_json"]`


### Tool: `cannsim_remote_run`

| Parameter | Required | Default | Description |
|---|---|---|---|
| `local_dir` | yes | — | Dir with sources/binary + npubin + run_kernel.sh |
| `run_script` | yes | — | Shell wrapper filename (e.g. `run_kernel.sh`) |
| `binary_name` | no | `test_kernel` | Name of compiled binary cannsim wraps |
| `job_name` | no | basename+timestamp | Remote subdirectory name |
| `soc_version` | no | Ascend950 | cannsim -s value |
| `gen_report` | no | false | When True, runs `cannsim report` after record and downloads trace_core0.json |
| `timeout` | no | 600 | SSH timeout for cannsim record step |
| `report_timeout` | no | 300 | SSH timeout for cannsim report step |

> ⚠️ **`local_dir` is the upload root — `run_script` must resolve inside it.**
> The plugin uploads ONLY the contents of `local_dir`, then executes
> `bash <remote_job_dir>/<run_script>`. Two common mistakes:
> - Pointing `local_dir` at a subdirectory (e.g. `references/vector_add/cannsim_host`)
>   and setting `run_script="../run_kernel.sh"` — fails with
>   `bash: .../run_kernel.sh: No such file or directory` because the parent
>   was never uploaded.
> - Using an absolute path for `run_script` — only filenames relative to
>   `local_dir` work.
>
> Rule: set `local_dir` to the directory that *contains* `run_kernel.sh`, and
> set `run_script` to just `"run_kernel.sh"`. The script itself can then use
> `SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"` to find its own sources/host
> subdirectories.

Returns: `success`, `job_name`, `remote_job_dir`, `remote_experiment_dir`, `patch_log`,
`build_log`, `cannsim_log_tail`, `report_log`, `trace_local_path`, `trace_json`,
`trace_truncated`. When `gen_report=True`, `trace_local_path` points to a local
`/tmp/cannsim_<job>_*_trace_core0.json` file with the full Chrome Trace.

> ⚠️ **Plugin invariants** (fixed in this skill's reference plugin — if you build
> your own SSH wrapper, replicate them exactly):
> 1. `cannsim record` must NOT use `-o <dir>` and must `cd` into the
>    job dir first so cannsim auto-creates `cannsim_<ts>_<binary>/` next to log_ca/.
> 2. `cannsim report` requires `-e <exp_dir> -o <exp_dir>/report -n 0`. There is
>    no `--timeline` flag in CANN 9.0.0 — passing it makes argparse reject the call.
> 3. The host binary must sit next to the .npubin (binary loads via `dirname(argv[0])`).
>    If your build emits the binary to `cannsim_host/build/bin/`, `run_kernel.sh`
>    must copy BOTH the binary AND the npubin into `$SCRIPT_DIR` before exec.

### Pitfalls discovered running cannsim-remote in practice

- **conda not on PATH over SSH** — non-interactive SSH sessions don't source
  `.bashrc`. Search common conda locations: `~/miniconda3/bin/conda`,
  `~/anaconda3/bin/conda`, `/opt/conda/bin/conda`.

- **tilde not expanded in SFTP chdir** — `sftp.chdir("~/cannsim_jobs/foo")`
  silently fails. Resolve `~` via `echo $HOME` over SSH first, then use
  the absolute path for all SFTP operations.

- **SFTP upload loses execute bit** — `sftp.put()` does not preserve file
  permissions. After upload, `chmod +x` everything in the remote job dir,
  not only the run script.

- **Stale binary from previous run causes GLIBCXX mismatch** — reusing the
  same `job_name` leaves the old binary in place. It crashes with
  `GLIBCXX_3.4.32 not found` if the remote GCC is older. Always `rm -rf`
  the remote job dir before re-uploading.

- **Never upload a pre-built binary** — build on the remote instead.
  A binary compiled locally with GCC 13 will not run on a remote with GCC 11.

- **`TRITON_COMPILE_ONLY=1` must be set BEFORE importing triton in compile_kernel.py** — all `os.environ["TRITON_*"]` assignments must appear at the very top of compile_kernel.py, before any `import triton` line. If set after import, the runtime already attempted NPU init and crashes with `ERR99999 UNKNOWN application exception`. Pattern: env vars first → clear `~/.triton/cache` → then `import triton`.

- **`prof_common.h: No such file or directory`** — CMakeLists must include `pkg_inc/profiling` and `pkg_inc/toolchain` in addition to `pkg_inc/runtime`. Required full set:
  ```cmake
  include_directories(
    ${ASCEND_PATH}/include
    ${ASCEND_PATH}/include/aclnn
    ${ASCEND_PATH}/include/experiment
    ${ASCEND_PATH}/include/experiment/msprof
    ${ASCEND_PATH}/x86_64-linux/pkg_inc
    ${ASCEND_PATH}/x86_64-linux/pkg_inc/profiling
    ${ASCEND_PATH}/x86_64-linux/pkg_inc/toolchain
    ${ASCEND_PATH}/x86_64-linux/pkg_inc/runtime
  )
  ```
  Use `target_link_options(target PRIVATE -Wl,--allow-shlib-undefined)` not the cmake-level flag form.
  A ready-to-use template is at [`templates/CMakeLists_full_includes.txt`](templates/CMakeLists_full_includes.txt).

- **cannsim report "log_ca not found" + no trace_core0.json** — `instr.bin` and `log_ca/` land at the job root (same directory as the test binary), not inside the `cannsim_<ts>_<binary>/` subdir that cannsim creates. `cannsim report -e <exp_dir>` looks for `log_ca` relative to `<exp_dir>`. If the trace is missing, check that `log_ca/` exists at the job root, and try passing the job root as `-e` instead of the cannsim subdir.

- **`ls | grep cannsim_` is fragile for exp_dir extraction** — `cannsim_host/` also matches. Use `find <job_root> -maxdepth 1 -name 'cannsim_2*' -type d` to isolate the timestamped experiment directory reliably.

- **Large-matrix cannsim is multi-hour** — a 4096×4096 matmul with BLOCK_M/N=32 generates 16384 kernel programs; the simulator runs them serially and requires 400+ min CPU time. For trace/bottleneck analysis, always use the smallest matrix size that exercises the same code path (256×256 or 512×512). Make the C++ host accept a size argument (`argv[1]`) so one binary covers both cannsim trace and full-size correctness checks. Reserve full 4096×4096 runs for real hardware benchmarking.
  Upload C++ source + CMakeLists.txt and build in `run_kernel.sh`:
  ```bash
  if [ ! -f "$SCRIPT_DIR/my_binary" ]; then
      cd "$SCRIPT_DIR" && mkdir -p build && cd build
      cmake .. -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE
      make -j$(nproc) && cp bin/my_binary "$SCRIPT_DIR/my_binary"
  fi
  ```

- **cannsim_log_tail does not contain kernel stdout** — `[PASS]`/`[FAIL]`/`[INFO]`
  lines are inside `output/cannsim_<timestamp>_<app>/cannsim.log`. SSH in
  and `cat` that file directly to debug failures.

---

## ⭐ Getting trace_core0.json — the only thing that matters for optimization

**The cycle count in `cannsim.log` is useless for optimization decisions.**
It only gives chip-level wall latency. You cannot see SCALAR%, WAIT_FLAG stalls,
MTE2/MTE3 breakdown, or RVECEX vs VEC usage from `cycle: 248`.

**The correct way to get the trace is to call `cannsim_remote_run` with `gen_report=True`.**
The plugin automatically runs `cannsim report -e <exp_dir> -o <exp_dir>/report -n 0`
after simulation and downloads `trace_core0.json` to a local path:

```python
result = cannsim_remote_run(
    local_dir="path/to/local_dir",
    run_script="run_kernel.sh",
    binary_name="test_my_kernel",
    job_name="my_kernel",
    gen_report=True,
    timeout=600,
)
trace_path = result["trace_local_path"]   # local path to trace_core0.json
trace_json = result["trace_json"]         # first 200 KB inline (use trace_path for full file)
```

> Note: `instr.bin` can be small (~160 KB for a tiny kernel like vector_add)
> and still produce a valid trace — size alone is not a correctness signal.
> Verify the trace by checking that pipeline categories (SCALAR, RVECEX, MTE2, ...)
> are present and event count is reasonable.

See `references/aiv_optimization_findings.md` for empirical findings from
trace_core0.json analysis: fp16 vs fp32 `tl.maximum` routing, `tl.zeros`
inside loops, UB size limits (BLOCK_HW ≤ 2048 for fp32), `num_stages=1`
crash, startup cost dominance for small HW, SCALAR overhead from 2D grid
index decomposition, and the **correct CANN 9.0.0 `cannsim report` command**
(`-e <exp_dir>` — the old `-i instr.bin -d log_ca/` form was removed).

**Key pitfall: do NOT pass `-o <output_dir>` to `cannsim record`.**
When `-o` is used, cannsim changes CWD to the timestamped subdir, so
`./log_ca` is not found → `instr.bin` may be truncated → no usable trace.
The `cannsim-remote` plugin never uses `-o` and handles this correctly.

---

## Pitfalls

### Handled automatically by the `cannsim-remote` plugin
These issues do NOT require action when using `cannsim_remote_run` — the plugin
takes care of them:
- **`-o <dir>` on `cannsim record`** — plugin never passes `-o`; always cd's into job dir
- **`cannsim` not on PATH in non-interactive SSH** — plugin sources the correct
  `~/miniconda3/Ascend/cann/bin/setenv.bash` symlink (not `cann-9.0.0`)
- **`conda: command not found` in SSH** — plugin finds conda via full path
- **`~` not expanded by paramiko SFTP** — plugin resolves `$HOME` over SSH before all SFTP ops
- **Stale remote job dir** — plugin does `rm -rf` before each upload
- **`GLIBCXX` version mismatch** — plugin never uploads pre-built binaries; always builds on remote
- **triton patches** — plugin auto-applies both patches (idempotent) before each run

---

### Kernel / host code pitfalls (still require attention)

1. **`do_issue_vector_instr not support mix task type`** — two causes (same error):
   - Patch 1 not applied → wrong `bishengir-compile` → wrong task type
   - Wrong binary magic (`ELF` instead of `ELF_AIVEC`) in the host C++ `rtDevBinary_t`
   Apply Patch 1 first, then check magic.

2. **`get_arch() returned NULL`** / **`rtGetSocVersion failed`** — Patch 2 not applied.
   Plugin applies it automatically, but if running manually ensure it is in place.

3. **`Cannot open kernel binary`** — the `.npubin` must sit next to the host binary.
   cannsim changes CWD on launch, so always use `dirname(argv[0])` to build the path.
   The `run_kernel.sh` build step must copy the npubin next to the compiled binary.

4. **cannsim OOM-killed** — camodel needs ≥32 GB RAM. Always use `cannsim_remote_run`
   rather than running cannsim locally.

4. `simt` kernels crash — only `parallel_mode = "simd"` is supported
10. **CANN 9.0.0 cleanup bug** (handled automatically by plugin): cannsim record exits code 1 after a successful simulation. `_cleanup_user_env` calls `os.getcwd()` on a directory it already deleted → `FileNotFoundError`. The plugin now detects this via `"current_dir = os.getcwd()" + "FileNotFoundError" + "_cleanup_user_env"` in stderr AND `"all tasks are finished!"` in stdout, and continues to the report step. No action needed — just be aware that `success: False` with this pattern is a false negative that is now auto-recovered.
11. **Timeout for large kernels**: default was 600s; now 1800s. GEMM 4096×4096 takes ~1500s to simulate. If you see a killed/terminated simulation with no `all tasks are finished!` in the log, increase `timeout` further (e.g. 3600s).

6. **`aclInit failed 500000`** — host uses `acl*` APIs. Rewrite to `rt*` only
   and drop `libascendcl.so` from the link step.

7. **`torch_npu` crashes under cannsim** — `import torch` calls `aclInit` at
   import time (error `507008`). Never import torch in cannsim host scripts.

- **`TRITON_COMPILE_ONLY=1` must NOT be set when running under cannsim** —
   `compile_kernel.py` sets it for compilation, but `run_kernel.sh` must NOT
   export it into the environment when launching the host binary. Keep it scoped
   to the `python compile_kernel.py` invocation only.

- **ERR99999 / empty `_triton_dump_*/` dir** — `TRITON_COMPILE_ONLY`, `TRITON_ASCEND_ARCH`,
   `TRITON_KERNEL_DUMP`, and `TRITON_DUMP_DIR` must all be set as `os.environ[...]`
   BEFORE any `import triton` line in `compile_kernel.py`. If triton is imported first
   it initialises the runtime, which then fails without a physical NPU (ERR99999). Symptom:
   `_triton_dump_<pid>/` exists but is completely empty; `No .npubin found` is raised.
   Fix: move all four `os.environ` assignments to the very top of the file, before even
   the cache-clearing `import shutil` block.

- **`prof_common.h: No such file or directory`** — CMakeLists missing include paths.
   The minimal set that works on CANN 9.0.0 requires all eight paths below; `profiling`
   and `toolchain` subdirs were absent from earlier templates:
   ```cmake
   include_directories(
     ${ASCEND_PATH}/include
     ${ASCEND_PATH}/include/aclnn
     ${ASCEND_PATH}/include/experiment
     ${ASCEND_PATH}/include/experiment/msprof
     ${ASCEND_PATH}/x86_64-linux/pkg_inc
     ${ASCEND_PATH}/x86_64-linux/pkg_inc/profiling
     ${ASCEND_PATH}/x86_64-linux/pkg_inc/toolchain
     ${ASCEND_PATH}/x86_64-linux/pkg_inc/runtime
   )
   ```
   Use `target_link_options(target PRIVATE -Wl,--allow-shlib-undefined)` not the cmake-level flag.

- **Large-program-count cannsim timeout** — 4096×4096 matmul with BLOCK_M/N=32 = 16,384
   programs; the simulator runs each serially and needs 400+ CPU-minutes — it will be
   killed or timeout. **For trace analysis, always use matrix sizes that keep total
   programs ≤ 1,024.** Rule of thumb: programs = `(M/BM) × (N/BN)` for 2D grid or
   `(M*N)/(BM*BN)` for 1D grid. A 4096×4096 kernel with BM=128, BN=128 = 1,024 programs
   and completes in ~15 min. Accept matrix size as `argv[1]` so one binary covers both
   cannsim trace (small) and full-size correctness/perf (4096).

- **Existing baseline hosts may have multiple kernel launches** — when creating a sub-kernel
   variant for cannsim, strip ALL launches (including correctness-check launches at e.g. 256×256)
   and replace with a single sub-kernel launch. A 256×256 correctness-check grid = 64 programs
   and is 64× slower than a grid=1 sub-kernel. The sub-kernel launch itself can perform
   its own correctness check using the small tensor dimensions.

- **Extracting fields from large cannsim_remote_run result files** — the tool writes a single
   giant JSON line (200KB+) to /tmp/hermes-results/. Python parsing may be blocked in review
   sessions. Use grep -oP to extract specific fields reliably:
   ```bash
   grep -oP '"trace_local_path"\s*:\s*"[^"]*"' /tmp/hermes-results/<result>.txt
   grep -oP '"cannsim_log_tail"\s*:\s*"[^"\\\\]*(?:\\\\.[^"\\\\]*)*"' /tmp/hermes-results/<result>.txt
   ```

9. **`rtGetAiCoreCount failed 0x32898`** — `NPUUtils.get_aicore_num()` calls
   `rtGetAiCoreCount` before the simulated device is ready. Fix: patch
   `getAiCoreNum` in `npu_utils.cpp` to fall back to an env var:
   ```cpp
   if (rtRet != RT_ERROR_NONE) {
       const char *env_val = getenv("TRITON_AICORE_NUM");
       aiCoreCnt = (env_val != nullptr) ? (uint32_t)atoi(env_val) : 20;
   }
   ```
   Then delete cached `.so`: `find ~/.triton/cache -name "npu_utils.so" -delete`

10. **BLOCK_HW=4096 fp32 causes UB overflow → empty kernel** — Ascend AIV UB ~32KB.
    A single fp32 tile with 3 live tensors: 3×4096×4=48KB overflows. Bishengir emits
    near-empty code silently. Symptom: cannsim completes in ~28 cycles, trace shows
    only SCALAR work. Max safe fp32 tile = 2048.

11. **num_stages=1 + hoisted tl.zeros → scalar div-by-zero crash** — Using
    `tl.zeros([N], fp32)` hoisted outside a `tl.range` loop with `num_stages=1` causes a
    scalar div-by-zero in the AIV core. Use `num_stages=2`. Symptom: "scalar_div: div by 0"
    in cannsim.log after "all tasks finished".

12. **False PASS from zero-initialized output buffer** — if test data produces
    near-zero outputs, they match the zero-initialized buffer even if the kernel wrote
    nothing. Always use test data that produces clearly nonzero values AND add a
    correctness check (e.g. sum-to-1 for softmax):
    ```cpp
    for (int r = 0; r < M; r++) {
        float s = 0.f;
        for (int c = 0; c < N; c++) s += outHost[r * N + c];
        assert(fabs(s - 1.0f) < 1e-3f);
    }
    ```

---

## Understanding `cannsim.log` cycle counts vs Perfetto trace timing

The two timing sources report **different things**:

| Source | Value (fused_softmax example) | What it measures |
|---|---|---|
| `cannsim.log` `cycle: 248` | 248 × 0.4 ns = **99 ns** | Chip-level wall-clock latency (all blocks in parallel) |
| `trace_core0.json` span | 9,940 units = **9.94 ms** (ts in µs) | Core 0 instruction timeline (its share of blocks) |
| Perfetto "84.90 ms" | sum of all pipeline lane durations | Total pipeline work across all units — NOT wall time |
| Perfetto "7,973 cycles" | from `core0_summary_log` | Per-core hardware cycles for core 0's blocks |

The `cannsim.log` cycle count is the chip-level **latency** — all 32 cores run
in parallel, each handling M/32 blocks. For a 128-block kernel on 32 cores,
each core does 4 blocks and the chip finishes in the time of ~4 blocks, not 128.

The `trace_core0.json` ts/dur units are **raw simulation cycles** but Perfetto
interprets them as microseconds (Chrome trace default). The "84.90 ms" shown by
Perfetto is the sum of all pipeline lane durations, not wall time.

To get hardware latency: use `cycle` from `cannsim.log` × 0.4 ns.
To understand bottlenecks: use `trace_core0.json` in Perfetto.

---

## Reference implementations

All reference files live in `references/` relative to this skill.
Pass the kernel dir as `local_dir` to `cannsim_remote_run` — the plugin
uploads everything and `run_kernel.sh` handles compile + build on the remote.

```
references/
├── vector_add/
│   ├── vector_add.py              ← kernel compile script (run on remote by run_kernel.sh)
│   ├── run_kernel.sh              ← build-on-demand: compiles kernel + C++ host, then runs binary
│   └── cannsim_host/
│       ├── CMakeLists.txt
│       └── test_vector_add.cpp    ← RT-only host, float32, N=1024
│
└── fused_softmax/
    ├── fused_softmax.py           ← kernel compile script (run on remote by run_kernel.sh)
    ├── run_kernel.sh              ← build-on-demand: compiles kernel + C++ host, then runs binary
    └── cannsim_host/
        ├── CMakeLists.txt
        ├── test_fused_softmax.cpp ← RT-only host, M=128 rows × N=1024 cols
        └── fused_softmax.npubin  ← cached npubin (regenerated by fused_softmax.py if absent)
```

### fused_softmax results (remote cannsim, Ascend950 camodel)

| Metric | Value |
|---|---|
| Input shape | 128 rows × 1024 cols (float32) |
| Grid | (128,) — one block per row |
| **Kernel cycles** | **248** |
| **Estimated hardware time** | **99.2 ns** (248 × 0.4 ns) |
| Verification | PASS — all 131072 elements match host reference (tol=1e-4) |
| Simulation wall time | 228 s (camodel overhead) |