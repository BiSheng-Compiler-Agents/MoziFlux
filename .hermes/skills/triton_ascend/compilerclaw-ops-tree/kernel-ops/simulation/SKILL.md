# Cannsim Simulation [LEAF NODE]

Run Triton-Ascend kernels without a physical NPU using the CANN cannsim simulator.
Covers compilation (ttir → ttadapter → npubin), host C++ launcher authoring, and cannsim
invocation. Tested with triton-ascend==3.2.1 + CANN 9.0.0 + Ascend910_9589 target.

## CRITICAL WORKFLOW RULE: Always use trace files, not cycle counts

> **The user has explicitly stated: "the cycle counts you may get from cannsim.log
> is worthless — you should base your decision only based on trace files."**
>
> cannsim.log cycle counts are chip-level wall latency only.
> They tell you nothing about *why* a kernel is slow.
> **All optimization decisions must be based on `trace_core0.json`.**
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

---

## Prerequisites

| Requirement | Version |
|---|---|
| triton-ascend | 3.2.1 |
| CANN | 9.0.0 |
| cannsim | ships with CANN 9.0.0 |
| Host RAM on remote | ≥ 32 GB (Ascend950 camodel is heavy) |

Remote machine: 184.150.234.220, port 2021, user s00929845, conda env `compilerclaw`.
Remote has GCC 11.4 (GLIBCXX up to 3.4.30) — always build on remote.

---

## Required Patches to triton-ascend 3.2.1

These two patches must be applied once to any fresh triton-ascend 3.2.1 installation.
They allow compilation and execution without a physical NPU.

### Patch 1 — `get_ascend_devices.py`: honour `TRITON_ASCEND_ARCH` env var

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

> Both patches are idempotent. The `cannsim_remote` Hermes plugin applies them
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
```

> Add `-DCMAKE_EXE_LINKER_FLAGS="-Wl,--allow-shlib-undefined"` to cmake.
> `libruntime.so` has transitive dependencies (`liberror_manager.so`, `libmmpa.so`)
> that are not present at link time but ARE provided by the camodel at runtime.

---

## Step 3 — Run with cannsim_remote_run

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

### `cannsim_remote_run` parameter reference

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

> `local_dir` is the upload root — `run_script` must resolve inside it.
> Set `local_dir` to the directory that *contains* `run_kernel.sh`,
> and set `run_script` to just `"run_kernel.sh"`.

Returns: `success`, `job_name`, `remote_job_dir`, `remote_experiment_dir`, `patch_log`,
`build_log`, `cannsim_log_tail`, `report_log`, `trace_local_path`, `trace_json`,
`trace_truncated`.

---

## Step 4 — Extract Kernel Runtime and Analyze Trace

### Analyzing trace_core0.json

The trace_core0.json file contains a full execution trace — DO NOT read it fully into context.
Instead, run the aggregation script:

```bash
python scripts/aggregate_trace.py /path/to/report/trace_core0.json
```

This outputs `/path/to/report/trace_summary.txt`.

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
y_f32 = tl.maximum(x_f32 + b, 0.0)
y = y_f32.to(tl.float16)
```
Routes to **RVECEX** which pipelines with MTE2/MTE3. Speedup: 7,132 → 4,838 cycles (1.47×).

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

### Startup cost vs compute cost (small-HW regime)

Per-program AIV startup cost ≈ 1,150 cycles. For HW ≤ ~1024, startup dominates.
**Fix:** persistent kernel — launch 32 programs that each loop over N*C/32 tiles.
Threshold empirically: HW ≤ 1024 → persistent; HW > 1024 → loop kernel.

---

## Pitfalls

Handled automatically by the `cannsim_remote` plugin (no action needed when using `cannsim_remote_run`):
- `-o <dir>` on `cannsim record` — plugin never passes `-o`
- `conda: command not found` in SSH — plugin finds conda via full path
- `~` not expanded by paramiko SFTP — plugin resolves `$HOME` over SSH
- Stale remote job dir — plugin does `rm -rf` before each upload
- GLIBCXX version mismatch — plugin never uploads pre-built binaries
- triton patches — plugin auto-applies both patches (idempotent)

Kernel/host code pitfalls (still require attention):
1. `do_issue_vector_instr not support mix task type` — wrong binary magic (`ELF` instead of `ELF_AIVEC`) or Patch 1 not applied
2. `Cannot open kernel binary` — .npubin must sit next to host binary; use `dirname(argv[0])`
3. cannsim OOM-killed — camodel needs ≥32 GB RAM; always use `cannsim_remote_run`
4. `simt` kernels crash — only `parallel_mode = "simd"` is supported
5. `aclInit failed 500000` — host uses `acl*` APIs; rewrite to `rt*` only
6. `torch_npu` crashes — `import torch` calls `aclInit` at import time; never import torch in cannsim host scripts
7. `TRITON_COMPILE_ONLY=1` must NOT be set when running under cannsim
8. BLOCK_HW=4096 fp32 causes UB overflow → empty kernel (symptom: ~28 cycles, 100% SCALAR)
9. `num_stages=1` + hoisted `tl.zeros` → scalar div-by-zero crash; use `num_stages=2`
10. False PASS from zero-initialized output — always use non-zero test data AND add correctness check

## Constraints
- Always use `gen_report=True` — cycle counts alone are not actionable
- Never upload pre-built binaries — always build on remote
- Only `parallel_mode = "simd"` kernels work in cannsim; simt is broken
- Remote cannsim machine: 184.150.234.220, port 2021, user s00929845, env `compilerclaw`

---

## Complete Working Reference Examples

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
