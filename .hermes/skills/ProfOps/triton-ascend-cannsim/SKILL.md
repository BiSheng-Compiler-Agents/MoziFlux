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
metadata:
  hermes:
    requires_tools:
      - cannsim_remote_run
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

## Prerequisites

| Requirement | Version |
|---|---|
| triton-ascend | 3.2.1 |
| CANN | 9.0.0 |
| cannsim | ships with CANN 9.0.0 |
| Host RAM on remote | ≥ 32 GB (Ascend950 camodel is heavy) |

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
# compile_kernel.py  — placed in local_dir, run on remote by run_kernel.sh
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
  set(ASCEND_PATH "/usr/local/Ascend/cann")
endif()

include_directories(
  ${ASCEND_PATH}/include
  ${ASCEND_PATH}/x86_64-linux/pkg_inc
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
which will output a condensed summary of the key metrics in a human/LLM readable format into a file at the same location named trace_summary.txt
```bash
python scripts/aggregate_trace.py /path/to/report/trace_core0.json
```

This will output `/path/to/report/trace_summary.txt`

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

5. **`simt` kernels crash in cannsim** — only `parallel_mode = "simd"` is supported.

6. **`aclInit failed 500000`** — host uses `acl*` APIs. Rewrite to `rt*` only
   and drop `libascendcl.so` from the link step.

7. **`torch_npu` crashes under cannsim** — `import torch` calls `aclInit` at
   import time (error `507008`). Never import torch in cannsim host scripts.

8. **`TRITON_COMPILE_ONLY=1` must NOT be set when running under cannsim** —
   `compile_kernel.py` sets it for compilation, but `run_kernel.sh` must NOT
   export it into the environment when launching the host binary. Keep it scoped
   to the `python compile_kernel.py` invocation only.

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
