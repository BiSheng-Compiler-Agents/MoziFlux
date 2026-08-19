# Bisheng compiler errors and fixes for extension kernels

Use this reference when a Triton-Ascend kernel containing `al.*` or `bl.*` extensions fails during
frontend lowering, BishengIR compilation, memory planning, bufferization, or device execution.
Extension kernels expose explicit scopes, memories, events, and physical buffer ownership; diagnostics
may point at the first invalid IR consumer rather than the structural operation that caused the issue.

## Diagnosis policy

1. Preserve the last correct numerical implementation.
2. Classify the failure stage before editing code.
3. Capture the earliest diagnostic and the last pass that completed.
4. Compile with the real target and the exact launch options used by the source.
5. Inspect allocation roots, scope nesting, loop-carried values, and branch live-outs.
6. Change one structural property per attempt.
7. Verify the fix with normalized IR and the full correctness matrix.

Do not edit the source line named by a late verifier until the failing pass and invalid value are known.

## Failure-stage classification

| Stage | Typical evidence | First action |
|---|---|---|
| Python/Triton frontend | Python exception, AST error, missing `_builder`, undefined name | Inspect constexpr use and branch definitions |
| TTIR/TTAdapter lowering | MLIR verifier error before Bisheng invocation | Inspect tensor types, layouts, slices, and loop-carried values |
| BishengIR pipeline | `buildFinalHIVMPipelines`, pass crash, PlanMemory, bufferization error | Dump every pass and identify the first failing pass |
| Binary/tool invocation | missing `bisheng`, `bishengir-compile`, target rejection | Check target, PATH, CANN environment, and compiler package |
| Device execution | timeout, AICore error, silent corruption after successful compilation | Audit events, physical slots, memory visibility, and auto-sync options |

## Error signature matrix

### `'hivm.hir.pointer_cast' op addrs ... should not be empty`

Likely stage: PlanMemory or a later verifier consuming an unallocated memref.

Common structural causes:

- an implicit staging buffer from a block-pointer load/store has no address;
- a buffer root is allocated inside a runtime branch;
- repeated Cube/Vector scopes make liveness ownership ambiguous;
- a subview or converted buffer no longer traces to a concrete `bl.alloc` root;
- source specialization facts or launch options differ from the real JIT launch.

Fix sequence:

1. Read the last successful pass and the first pass that reports empty addresses.
2. Find the unallocated memref in the IR; do not trust only the source location attached to
   `pointer_cast`.
3. Hoist shared `bl.alloc` roots outside runtime branches.
4. Use explicit branches to select concrete buffers.
5. Put deterministic traversal loops inside long-lived engine scopes.
6. Replace the specific implicit block-pointer staging operation with explicit raw-pointer arithmetic
   when the staging memref is the unallocated object.
7. Reproduce the launch specialization and compiler options exactly.

False lead: reducing tile size or `num_stages` does not fix a structural root-ownership failure unless
the actual failure is capacity.

### `MemLivenessAnalysis` assertion or cast failure

Likely cause: repeated or nested scope regions produce liveness intervals the memory planner cannot
represent.

Fix:

- prefer one long-lived Cube scope and one long-lived Vector scope per traversal;
- put task/chunk loops inside those scopes;
- allocate cross-scope handoff buffers outside the scopes;
- keep stage-local temporaries inside their owning scope;
- if separate phases require additional scopes, ensure no tensor or implicit buffer crosses the phase
  boundary without an explicit memory object.

### Root allocation cannot be found, root is not dominant, or buffer has no root

Likely cause: a selected buffer or alias originates in only one runtime branch, or a helper returns a
branch-local tensor whose allocation does not dominate all consumers.

Fix:

```python
buffer0 = bl.alloc(...)
buffer1 = bl.alloc(...)

if slot == 0:
    x = bl.to_tensor(buffer0)
else:
    x = bl.to_tensor(buffer1)
```

Keep caller-owned output buffers and copy branch results into them explicitly. Do not return a local
buffer allocation from only one branch.

### SSA value does not dominate use or `bind_buffer` dominance failure

Likely cause: a tensor computed under a runtime branch or outlined region is bound to a buffer outside
the region without a valid dominating definition.

Fix:

- allocate the destination in the caller;
- compute a tensor in each branch or preserve pass-through values in both branches;
- materialize with `bl.to_buffer(..., space=...)`;
- use an explicit `al.copy` into the caller-owned destination;
- avoid nested `bind_buffer` writeback across branch or outline boundaries.

### `'vector.mask' op body must bufferize in-place`

Likely cause: an outlined Vector scope is nested in a runtime `if/else`, and tensor live-outs become
branch results that cannot be bufferized in place.

Fix:

- hoist one outlined scope outside the runtime branch;
- perform small slot/state selection before or after the outlined computation;
- pass concrete buffers or small state vectors explicitly;
- avoid returning full-tile tensors through `scf.if` from outlined functions.

See `outlined-scope-restrictions.md`.

### `Exceeds vector capacity`

Possible causes:

- one operation inside an outlined Vector function exceeds the single-operation capacity;
- an index remains `int64`, doubling a compare/mask vector's byte footprint;
- a whole-tile operation was moved from an auto-tiled region into a manually outlined function.

Fix sequence:

1. Determine the dtype and byte size of the exact failing operation.
2. Cast row/index scalars to `tl.int32` before broadcasting comparisons when 64-bit indexing is not
   required.
3. Split an oversized row/tile operation into capacity-safe pieces.
4. Move whole-tile elementwise work out of the outlined helper when function-level auto-tiling is
   available.
5. Keep reductions or state updates in the outlined helper only when each operation fits.

False lead: a later Fixpipe diagnostic may be cascading from the capacity failure. Fix the earliest
capacity error first.

### Fixpipe reports unsupported hardware, wrong memory path, or `dual_dst_mode` rejection

Possible causes:

- compilation uses a target that does not support the requested Fixpipe feature;
- the destination is not in the required memory space;
- the message is secondary to an earlier Vector legalization or capacity error.

Fix:

1. Locate the earliest diagnostic in the pass log.
2. Compile with the real device target.
3. Check the platform's L0C drain path in
   `../../../shared/references/hardware-architecture.md`.
4. Verify source and destination memory spaces and layout conversion mode.
5. If the Fixpipe message follows another failure, resolve the earlier failure before changing
   Fixpipe code.

### `memref.expand_shape` invalid output shape

Likely cause: reshape or permute is applied directly to a sliced/non-contiguous view whose physical
strides do not satisfy the requested expansion.

Fix:

- materialize the slice into a contiguous buffer before reshape/permute;
- keep physical layout dimensions explicit;
- verify element count and reassociation maps in TTAdapter IR.

### `_builder argument must be provided` in `extract_slice`/`insert_slice`

Likely cause: slice sizes lost constexpr status.

Fix:

```python
HALF: tl.constexpr = BLOCK_N // 2
```

Use constexpr values for `SIZES`. Keep `OFFSETS` runtime-typed or use integer literals as required by
the frontend. Do not pass a constexpr-annotated local where the offset conversion expects a runtime
tensor.

See `extract-slice-frontend.md`.

### Undefined name after runtime `if`, or branch result merge failure

Likely cause: a value is assigned in only one runtime branch and read afterward.

Fix:

- define the value before the branch and update it in both branches; or
- carry all slot values through the branch and update only the selected one while passing the others
  through unchanged.

### Loop-carried variable type mismatch

Typical message: initial type differs from reassigned type.

Likely cause: one variable name is reused for different shapes, layouts, address spaces, or dtypes
across a loop iteration.

Fix:

- use distinct names for Cube tiles, Vector sub-tiles, buffers, and tensor views;
- keep every loop-carried value's shape, dtype, and encoding invariant;
- materialize layout changes into a separate value.

### Block-pointer offsets or block shape must be 32-bit

Likely cause: a 64-bit scalar such as a sub-vector identifier reaches block-pointer offsets or shape
construction.

Fix:

- cast offsets to `tl.int32` where the address range permits;
- alternatively fold the sub-core displacement into the base pointer and keep block-pointer offsets
  32-bit.

### Return or break inside a JIT loop is rejected

Typical symptoms include an unsupported AST node for `break` or a diagnostic that return statements
cannot appear inside `for`/`while`, including returns in helpers called transitively from the loop.

Fix:

- express early exit as an active predicate carried through the loop;
- move helper returns outside loop control flow;
- use fixed bounds plus validity guards when the frontend cannot represent the desired exit.

### Data movement is assigned to the wrong engine or memory space

Possible symptoms include a copy/fixpipe operation rejected in a Cube scope, a destination-memory
diagnostic, a runtime hang when feeding `tl.dot`, or numerically wrong matrix operands.

Fix:

- place UB buffer staging and Vector-owned copies on the supported Vector/data-movement path;
- keep Cube operands in the required L1/L0 memory and physical fractal layout;
- use the platform-supported L0C drain path for Cube results consumed by Vector;
- materialize a full, correctly laid-out L1 operand when subview-fed matrix operands are unsupported;
- verify the actual address space and layout in TTAdapter IR, not only the logical tensor shape.

### UB/L1 overflow

This is a capacity failure, not a root-allocation failure.

Count all simultaneously live objects:

- ping/pong handoff slots;
- per-sub-core FP32 state;
- mask and index vectors;
- outlined-function temporaries;
- compiler-created staging buffers;
- alignment padding and multibuffer copies.

Fix by reducing live ranges first, then buffer multiplicity, then tile size. Never substitute a smaller
block silently; qualify or skip unsupported configurations explicitly.

### `fuseIndependentSiblingForLoops` crash or missing operations after vectorization

Likely cause: duplicated sibling Vector loops or duplicated masked bodies create an unsupported
fusion/vectorization structure.

Fix:

- express mode differences as bounds, masks, or constexpr dispatch around one computation body;
- avoid copy-pasted sibling loops with equivalent structure;
- if the compiler exposes a pass-specific disable option, use it only after confirming the option
  reaches the backend and the normalized IR retains all operations;
- compare the IR before and after the failing fusion pass.

### Generic `buildFinalHIVMPipelines` failure

This is a wrapper symptom, not a root cause.

Fix:

1. Capture the exact Bisheng command from debug output.
2. Rerun it with pass dumping enabled.
3. Find the first diagnostic and the last `IR Dump After <Pass>` entry.
4. Classify the failure using this matrix.

### Compiler process hangs

Fix:

- run compilation under a timeout in a fresh process;
- preserve stdout/stderr and the last emitted pass name;
- distinguish a compiler hang from a device hang by confirming whether a binary was produced;
- reduce the kernel structurally: stage body, scope, buffer, sync edge, loop, then outline;
- invalidate only the candidate's cache entry when stale artifacts are suspected.

### Missing `bisheng`, `bishengir-compile`, or host stub

A missing downstream executable can occur after earlier MLIR stages succeeded.

Fix:

- inspect the dump directory before assuming the kernel failed;
- restore the CANN/compiler environment and PATH;
- use the compiler binary selected for the real target;
- do not claim binary qualification when only TTIR/TTAdapter stages completed.

## Compiles successfully but hangs or corrupts output

### Manual sync conflicts with automatic sync injection

Symptom: compilation succeeds, then the device hangs or produces silent corruption in a loop with
manual Vector-to-Cube events.

Fix:

- ensure the launch disables competing automatic block-sync injection when required by the extension
  path;
- do not combine compiler DAG scheduling with a hand-written event graph without inspecting the IR;
- count every event set/wait, including initialization and drain tokens.

### Producer and consumer use different physical slots

Symptom: every output is numerically wrong although event counts balance.

Fix:

- write a table containing logical item, producer clock, producer slot, consumer clock, consumer slot,
  and release event;
- derive input and output phases independently;
- do not use one parity variable for both a consumed input and a produced output unless their phases
  are proven equal;
- test first item, first wrap, output transition, and drain clocks.

### Producer laps consumer

`sync_block_set` signals readiness but does not necessarily provide overwrite back-pressure.

Fix:

- add a slot-free acknowledgement when a physical slot can be reused before in-order dependencies
  guarantee consumption;
- initialize one free token per physical slot;
- drain surplus tokens at traversal end;
- verify sets equal waits for every event.

## IR inspection procedure

1. Compile with debug/dump output enabled and the real target.
2. Save the exact compiler command and all launch options.
3. Locate the TTIR, TTAdapter/Linalg, and Bisheng pass logs.
4. Rerun the exact Bisheng command with `--mlir-print-ir-after-all` appended when supported.
5. Inspect the first failing pass, not only the final verifier message.
6. Compare these structures between a passing and failing candidate:
   - `scope.scope` regions and engine attributes;
   - `scf.for` loop-carried values;
   - `scf.if` results and branch live-outs;
   - explicit `bl.alloc` roots and address spaces;
   - implicit staging memrefs from loads/stores;
   - outlined Vector functions and call placement;
   - `sync_block_set/wait` count and pipe arguments;
   - multibuffer transformations and live-buffer count.

Use `extension-compile-path.md` for environment setup and dump locations.

## Minimal structural bisect ladder

Create scratch candidates outside the project and add one property per rung:

1. one stage body with no extensions;
2. one engine scope;
3. explicit `bl.alloc` root;
4. one producer-to-consumer copy;
5. one event edge;
6. deterministic loop inside each scope;
7. second physical buffer slot;
8. slot-free acknowledgement;
9. runtime concrete-buffer selection;
10. outlined Vector helper;
11. full metadata/state ring;
12. production block and shape.

Run each candidate in a fresh process. Record compile success, first failing pass, device completion,
correctness, and buffer/event structure. Revert only the failing rung.

## Compiler issue report template

```text
Kernel and target:
Compiler and extension package:
Launch options:
Failure stage:
First diagnostic:
Last completed pass:
Reported source location:
Actual invalid IR value/root:
Minimal reproducer structure:
Passing neighboring structure:
Fix applied:
Correctness result:
IR evidence:
```

## Related references

- `extension-compile-path.md` — extension-triggered passes, target selection, and dump procedure.
- `outlined-scope-restrictions.md` — outlined Vector bufferization and capacity restrictions.
- `extract-slice-frontend.md` — slice constexpr, offset, and vector-capacity rules.
- `persistent-cv-state-ownership.md` — task, buffer, and persistent-state ownership.
- `../../../shared/references/triton-api-reference.md` — extension API and launch options.
- `../../../shared/references/hardware-architecture.md` — target memory paths and capacities.
