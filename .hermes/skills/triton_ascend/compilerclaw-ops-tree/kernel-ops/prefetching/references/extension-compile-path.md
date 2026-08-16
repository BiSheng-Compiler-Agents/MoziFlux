# How al/bl extensions change the Triton-Ascend compile path

Evidence gathered from the installed triton-ascend package and the triton-lang/triton-ascend
source tree, plus local compile probes (compile to ttadapter stage, inspect
`~/.triton/dump/<hash>/kernel.ttadapter.mlir`). Use this when reasoning about what the compiler
will and will NOT do once a kernel uses `al.scope`, `al.sub_vec_id`, `al.sync_block_*`, or `bl.*`.

For concrete diagnostics, structural causes, and fix sequences, load
`compiler-errors-and-fixes.md`.

## 1. `al.sub_vec_id()` disables bishengir's auto sub-block binding

- `third_party/ascend/ascend_ir.cc` (`create_get_sub_vec_id`): sets module attribute
  `hivm.disable_auto_tile_and_bind_subblock` with the comment "NPU compiler will parse this
  attribute and disable auto tile and bind subblock pass".
- `backends/ascend/compiler.py` `_parse_linalg_metadata` regex-detects the attribute in the
  linalg text and sets `metadata["auto_tile_and_bind_subblock"] = False`; the bishengir-compile
  command line then gets `--enable-auto-bind-sub-block=False` (unless the user overrode
  `enable_auto_bind_sub_block`, which has higher priority — see `get_auto_bind_sub_block_option`).
- Empirical: identical kernel compiled with and without `al.sub_vec_id()`; the attribute appears
  in `kernel.ttadapter.mlir` only in the sub_vec_id version.
- Consequence: manual sub-vector partitioning (`extract_slice`/`insert_slice`) is not optional
  once `sub_vec_id()` is used — the compiler's automatic equivalent is switched off.

## 2. `al.scope` (any `scope.scope` op) skips the auto pipeliner (SSBUFFER)

- `third_party/ascend/lib/DynamicCVPipeline/PreCheckAvailable/PreCheckBlacklist.cpp`: walks the
  module; if a `scope.scope` (or `scf.while`) op is found, sets the fallback attr
  (`ERRCODE_IGNORED`) with the message "SSBUFFER will be skipped because scope.scope operation
  was found, which indicates that it has been optimized for the Ascend."
- `PreCheckMatmul.cpp` in the same directory: if no `linalg.matmul` exists (pure-vector kernel),
  SSBUFFER is also skipped.
- Empirical: a mixed dot+vector kernel written with `al.scope` + `al.sync_block_set/wait` reaches
  ttadapter with the `scope.scope` ops (carrying `hivm.tcore_type = CUBE/VECTOR`) and explicit
  `hivm.hir.sync_block_set/wait` ops intact — they flow to bishengir verbatim instead of
  compiler-injected scheduling.
- Consequence: once you hand-write scopes, pipelining/multi-buffering/sync is entirely yours.
  The compiler treats the module as already scheduled.

## 3. The compiler-inserted alternative (off by default)

- `compiler.py` (`ttir_to_linalg`): `metadata["add_auto_scheduling"]` (NPUOptions default
  `False`) gates `add_dag_sync` + `add_dag_scope` + `add_dag_ssbuffer` — the automatic
  scope/sync/buffer insertion passes. This is the auto counterpart of manual Stage 1; do not
  combine it blindly with hand-written `al.scope` (double scheduling).
- A larger `add_dynamic_cv_pipeline` pass set exists in the source tree
  (`third_party/ascend/lib/DynamicCVPipeline/`, 910_95-only, falls back to normal compilation on
  failure by restoring a module backup) but is NOT exposed in the installed wheel's
  `ascend.passes.ttir` Python bindings — verify availability before relying on it.

## 4. Other extension-triggered flag changes

- `mix_mode` is parsed from the linalg (`aiv` / `aic` / `mix`); `mix_mode == "aic"` adds
  `--disable-hfusion-vectorize=true`. `al.scope` itself does not change `mix_mode` (a mixed
  kernel is `"mix"` with or without scopes).
- `enable_mixed_cv`, `sync_solver`, `inject_barrier_all`, `inject_block_all`,
  `enable_hivm_auto_cv_balance`, `multibuffer`, `set_workspace_multibuffer` are all plain
  NPUOptions metadata → bishengir flags; none are auto-toggled by extension usage.
  The two automatic toggles are exactly: disable-auto-bind-sub-block (sub_vec_id) and
  skip-SSBUFFER (scope.scope).

## 5. Local verification recipe (no NPU needed)

The ttir → ttadapter (linalg) stages run locally; only the final linalg→bin stage needs a live
device (`rtGetSocVersion` / `get_arch` fail without one). To inspect what reaches bishengir:

```bash
source <cann>/set_env.sh
export LD_LIBRARY_PATH=<env>/lib/python3.11/site-packages/torch_npu/lib:$LD_LIBRARY_PATH
export SOC_VERSION=Ascend910_95
python - <<'EOF'
import triton
from triton.compiler.compiler import ASTSource
from triton.backends.compiler import GPUTarget
src = ASTSource(kernel, signature, constants)
triton.compile(src, target=GPUTarget("npu", "Ascend910_95", ""),
               options={"debug": True})   # raises at binary stage; dumps already written
EOF
# then inspect:
grep -c "scope.scope"                        ~/.triton/dump/<hash>/kernel.ttadapter.mlir
grep    "disable_auto_tile_and_bind_subblock" ~/.triton/dump/<hash>/kernel.ttadapter.mlir
grep -o 'mix_mode = "[a-z]*"'                ~/.triton/dump/<hash>/kernel.ttadapter.mlir
grep -c "sync_block"                         ~/.triton/dump/<hash>/kernel.ttadapter.mlir
```

This is the fast way to confirm a kernel's scopes/syncs/attributes survived lowering before
spending a remote compile+bench cycle.

### Running the FULL MLIR pipeline locally 

The linalg→bin stage can run locally far enough to surface real backend errors
(`Exceeds vector capacity`, `ub overflow`, pass crashes). The blocker is that
`is_compile_on_910_95` (tools/get_ascend_devices.py:52-58, computed at import
from PCI scan / npu-smi / env) gates which bishengir binary is used, and its env
whitelist — `{"ascend910_9589", "ascend910b", "ascend950", "ascend910_95"}` —
does NOT include `Ascend910_9579`. The arch-validation whitelist
(get_ascend_arch_from_env) is a DIFFERENT list (9579/9581/9589/9599 OK; bare
910_95 / 950 rejected). **`Ascend910_9589` is the only value passing both.**

With `TRITON_ASCEND_ARCH=Ascend910_9589` (no NPU needed):

- `is_compile_on_910_95=True` → `_get_npucompiler_path` returns the bundled
  `bishengir-a5/bin/bishengir-compile` DIRECTLY (no delegation to the
  non-shipped `bishengir-compile-a5` helper name) and prepends its dir to PATH
  itself (hivmc-a5 resolves).
- On this path `--target` = `metadata['target'].arch`, i.e. your GPUTarget
  string — so `GPUTarget("npu", "Ascend950PR_9579", "")` gives the REAL device
  target locally. The a5 binary is the same build as the device's (1.0.0
  ca88a6080dfc, assertions ON).
- The pipeline then runs through hivmc-a5 and dies only at the missing
  `bisheng` host-stub compiler: "Cannot find bisheng under $PATH" means the
  whole device-side MLIR pipeline PASSED.

### Attributing a perf difference to codegen 

When two kernel variants are bit-identical but differ in speed, diff what the
compiler DID, not the source: locally re-run the a5 binary on each variant's
`~/.triton/dump/<hash>/kernel.ttadapter.mlir` with the flags from the `[DEBUG]
cmd_list` line plus `--mlir-print-ir-after-all > passlog_<variant>.txt` (the
log up to any crash is still complete for the early/mid passes). Then compare,
per variant: the set of `func.func` definitions (`@<k>_scope_N` =
user-outlined, `@<k>_outlined_vf_N` = auto-outlined by
PullSliceIntoVectorFunction), the `func.call` count and whether calls sit
inside `scf.for` bodies, and per-function `vector.transfer_read/write` counts
(UB traffic) at the late (MarkMultiBuffer) stage. This settled the
`outline=True` question: without it the auto-outliner fragments a manual row
loop into per-row-body no_inline VFs (~3 calls/row vs 1 call/chunk) —
function-call-inside-loop instead of loop-inside-function — which no_inline
call-boundary stalls explain the ~7x gap.
