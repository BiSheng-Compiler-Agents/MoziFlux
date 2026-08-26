# CompilerClaw Ops Routing Protocol [MANDATORY]
Before handling any user task, you MUST execute the following routing flow:

## Phase 1: Select Skill
Based on the user's **complete conversation history + current prompt**, determine which skill to use.

| User Intent | Keyword Signals | Route Target |
|-------------|----------------|--------------|
| **Optimize a Triton kernel for Ascend NPU** (any phrasing: improve, speed up, bottleneck, faster) | optimize, optimization, improve performance, speedup, bottleneck, make faster | **Read BOTH `./kernel-ops/ROUTER.md` AND `./triton-operator/ROUTER.md` in parallel** — see Mandatory Dual-Tree Rule below |
| Hermes plugin development, lifecycle hook, tool registration, slash command, OTel/Phoenix tracing | plugin, hook, register_hook, register_tool, slash command, OTel, Phoenix, pre_tool_call, post_tool_call, on_session | Read `./hermes-infra/ROUTER.md` |
| Kernel cannsim simulation or trace analysis only (no optimization intent) | cannsim, cannsim_remote_run, trace_core0, npubin, rt*, aclInit, RVECEX, MTE2, SCALAR, instr.bin, gen_report, simulation | Read `./kernel-ops/ROUTER.md` |
| Kernel hardware profiling, profile_kernels.py, perf_report, do_bench, benchmark | profile_kernels, perf_report, do_bench, benchmark, latency, NPU hardware, torch_ref vs baseline | Read `./kernel-ops/ROUTER.md` |
| Kernel code generation, tiling strategy, implement new kernel | tiling, kernel code, codegen, implement kernel, block size, grid, tl.dot, UB, BLOCK_M, persistent grid | Read `./kernel-ops/ROUTER.md` |
| Triton-Ascend extension prefetching/preloading for mixed Cube/Vector kernels | prefetch, prefetching, preload, preloading, al.scope, bl.alloc, ping-pong, cube scope, vector scope, CV pipeline | Read `./kernel-ops/ROUTER.md` |
| Episode recording/retrieval, optimization knowledge base, past episodes | episode_write, episode_retrieve, episode_update, episode_list, optimization episodes | Read `./kernel-ops/ROUTER.md` |
| Full operator development workflow, end-to-end development, unsure which skill to use | full-process, end-to-end, development orchestration, operator dev | Read `./triton-operator/ROUTER.md` |
| Static code review, code review, P0/P1/P2 bugs | code review, static analysis, review, P0, P1, mask missing, core type mismatch | Read `./triton-operator/ROUTER.md` |
| Environment setup, install CANN/triton/torch_npu, conda setup | environment, install, CANN, conda, triton-ascend version, torch_npu, setup | Read `./triton-operator/ROUTER.md` |
| Cross-skill workflows, pipeline, full end-to-end flow | workflow, pipeline, cross-skill, batch, end-to-end | Read `./cross-cutting/SKILL.md` |
| Other / not specified | — | List all skills and let the user choose |

---

## Mandatory Dual-Tree Rule — Kernel Optimization

**Any request to optimize a Triton kernel for Ascend MUST load skills from BOTH trees:**

1. `kernel-ops/` tree — for the technical optimization work:
   - `episode-memory/` → retrieve past patterns before writing any code
   - `simulation/` → cannsim traces with sub-kernel hosts
   - `optimization/` → bottleneck analysis and fix patterns

2. `triton-operator/` tree — for deliverables and quality gates:
   - `code-review/` → static review of the optimized kernel (P0/P1/P2)
   - `orchestration/` → required output files checklist

**Required output files for every kernel optimization task** (from orchestration/SKILL.md):

| File | Content |
|------|---------|
| `opt_{kernel_name}.py` | Optimized kernel + ModelNew host interface |
| `profile_kernels.py` | `@perf_report` benchmark, all dispatch paths, unit test |
| `Optimizations.md` | Each optimization applied, with code snippets and rationale |
| `performance_report.md` | cannsim trace tables (baseline vs optimized), hardware latency (TBD if no NPU) |
| `review.md` | Static P0/P1/P2 review of the optimized kernel |

An optimization task is **not complete** until all five files exist in the kernel directory.

---

## Phase 2: Select Capability
Follow the ROUTER.md in the selected skill sub-tree, continuing to route until you reach a file marked `[LEAF NODE]`.

## Disambiguation Rules
- **"optimize" + kernel name** → Mandatory Dual-Tree Rule (both `kernel-ops/` and `triton-operator/`)
- Mentions **cannsim / trace_core0 / npubin / instr.bin / rt*** (no optimize intent) → `kernel-ops/` → `simulation/`
- Mentions **profile_kernels / do_bench / perf_report / NPU hardware latency** → `kernel-ops/` → `profiling/`
- Mentions **episode_write / episode_retrieve / optimization episodes** → `kernel-ops/` → `episode-memory/`
- Mentions **tiling / codegen / implement kernel / UB budget / BLOCK** without optimize → `kernel-ops/` → `codegen/`
- Mentions **plugin / hook / OTel** → `hermes-infra/`
- Mentions **code review / static analysis / P0 P1** → `triton-operator/` → `code-review/`
- Mentions **full-process / end-to-end / orchestration** → `triton-operator/` → `orchestration/`
- Mentions **CANN install / conda / environment** → `triton-operator/` → `env-config/`
- Skill context already established in conversation → do not re-select skill, route directly into its sub-tree

### Signal Priority

| Priority | Signal Type | Routing Behavior | Example |
|----------|-------------|-----------------|---------|
| **P1 Highest** | Skill/tool name | Route directly, no asking | "cannsim_remote_run" → kernel-ops/simulation |
| **P2** | Unique domain term | Route directly, no asking | "trace_core0.json" → kernel-ops/simulation |
| **P3 Lowest** | Cross-domain generic term | Ask only when no P1/P2 present | "optimization" alone → ask |

**Priority rule**: P1 > P2 > P3. When P2 and P3 both appear, P2 overrides P3, no asking.

## Route Tracing [Optional]

When the user prompt contains **"debug routing"** or **"route trace"**, activate route-tracing mode:

1. After Phase 1 decision, output: `[Route] ROOT → <skill-name> (P<level>: <matched signal>)`
2. After each ROUTER.md decision, append: `[Route]   → <capability-name> [LEAF]`
3. For cross-skill tasks, output one line per skill
4. Once the leaf node is reached, begin execution and stop outputting routing info

**Normal mode** (default): output no routing information, execute directly.

## Important Constraints
- **Must** route before executing — do not skip routing to guess the skill
- **Context first**: when a specific skill name was mentioned in the conversation, route directly to that sub-tree
- **Cumulative context**: routing decisions must consider all context constraints established in the conversation
- If a task spans multiple categories, read multiple leaf nodes in parallel
