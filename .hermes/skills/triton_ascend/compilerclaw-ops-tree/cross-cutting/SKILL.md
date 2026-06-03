# Cross-Skill Workflows [LEAF NODE]

Workflows that combine multiple skills in this tree. Use when a task spans multiple
capabilities (e.g., full kernel development, optimization loop with memory, plugin + kernel
instrumentation).

## Target Skills

hermes-plugin-development + kernel-episode-memory + triton-ascend-cannsim + triton-ascend-kernel-profiling + triton-operator-code-gen + triton-operator-code-review + triton-operator-dev + triton-operator-env-config + triton-operator-performance-optim

## When to Use
- Cross-skill workflows, pipelines, batch processing, cross-domain tasks
- Combining different skill capabilities to complete complex tasks
- Output of one skill feeds as input to another skill

## When NOT to Use
- Only one skill involved → route to that skill's sub-tree
- Simple single-step task → use corresponding leaf node directly

---

## Predefined Workflows

| Workflow | Skills Involved | Description |
|----------|----------------|-------------|
| Full Operator Development | env-config → codegen → code-review → simulation → optimization → profiling → episode-memory | Complete kernel from scratch to benchmarked result |
| Optimization Loop | simulation + optimization + episode-memory | Iterative cannsim → trace analysis → fix → re-measure loop |
| New Developer Onboarding | env-config + codegen + code-review | Set up environment, generate first kernel, validate it |
| Kernel Quality Gate | code-review + simulation | Static review then cannsim validation before merging |
| Knowledge-Driven Optimization | episode-memory + optimization + simulation | Retrieve past learnings, apply patterns, record new episode |
| Plugin-Instrumented Kernel Dev | hermes-plugin-development + codegen + simulation | Add Hermes tracing plugin to monitor cannsim tool calls during kernel development |
| Benchmark + Record | profiling + episode-memory | Run hardware benchmark, record results as episode for future reference |

---

## Workflow 1: Full Operator Development

**Trigger**: "Implement a new Triton operator for Ascend NPU from scratch"

**Steps**:
1. Environment check → `triton-operator/env-config/SKILL.md`
   - Verify CANN, torch_npu, triton-ascend versions match official docs
2. Retrieve episodes → `kernel-ops/episode-memory/SKILL.md`
   - `episode_retrieve(query="<kernel_type> <known_constraints>", target="ascend950", limit=5)`
3. Kernel code generation → `kernel-ops/codegen/SKILL.md`
   - Design tiling strategy, implement kernel, generate smoke test
4. Static review → `triton-operator/code-review/SKILL.md`
   - Fix all P0 issues before proceeding
5. cannsim simulation → `kernel-ops/simulation/SKILL.md`
   - Run with `cannsim_remote_run(gen_report=True)`, get trace
6. Performance optimization loop (repeat until target met) → `kernel-ops/optimization/SKILL.md`
   - Analyze trace → identify bottleneck → apply fix → re-run cannsim
7. Hardware profiling (if NPU available) → `kernel-ops/profiling/SKILL.md`
   - Generate `profile_kernels.py`, cover all dispatch paths
8. Record final episode → `kernel-ops/episode-memory/SKILL.md`
   - `episode_write(...)` with full observation/thoughts/action/result

---

## Workflow 2: Optimization Loop

**Trigger**: "Optimize this kernel / the cannsim trace shows bottleneck X"

**Steps**:
1. Read `triton-operator/orchestration/SKILL.md` — note required deliverables checklist
2. Retrieve past episodes → `kernel-ops/episode-memory/SKILL.md`
   - `episode_retrieve(query="<bottleneck_type> <kernel_type>", target="ascend950")`
3. Run cannsim to get baseline trace → `kernel-ops/simulation/SKILL.md`
   - `cannsim_remote_run(gen_report=True)` → `trace_core0.json`
   - Run `aggregate_trace.py` to get `trace_summary.txt`
4. Apply optimization → `kernel-ops/optimization/SKILL.md`
   - Identify dominant bottleneck from trace summary
   - Apply matching optimization rule
5. Re-run cannsim → compare traces → verify improvement
6. Static review of optimized kernel → `triton-operator/code-review/SKILL.md`
   - Fix all P0 issues; document P1/P2 in `review.md`
7. Write episode → `kernel-ops/episode-memory/SKILL.md`
   - Record what worked, latency before → after, key insight
8. Write required output files (all MANDATORY):
   - `opt_{kernel_name}.py` — optimized kernel + ModelNew host interface
   - `profile_kernels.py` — `@perf_report` benchmark covering all dispatch paths + unit test
   - `Optimizations.md` — each optimization applied with code snippets and rationale
   - `performance_report.md` — cannsim trace tables (baseline vs optimized), hardware latency TBD
   - `review.md` — static P0/P1/P2 review report of the optimized kernel

**Concrete example** (softmax kernel, `aiv_scalar > 80%`):
1. `episode_retrieve(query="softmax scalar overhead two-pass single-pass", target="ascend950")`
   → finds pattern: two-pass loads x multiple times, single-pass keeps x in UB
2. `cannsim_remote_run(local_dir="./softmax_v1/", gen_report=True)` → trace shows aiv_scalar=85%
3. Rewrite kernel: replace two-pass with `x = tl.load(...).to(tl.float32); tl.sum(x, 1)` single-pass
4. Re-run cannsim → compare: aiv_scalar drops to 15%, aiv_mte2 now dominant
5. `episode_write(kernel_name="softmax", observation="two-pass 3605µs aiv_scalar=85%", ..., result="752µs 4.8×")`

---

## Workflow 3: New Developer Onboarding

**Trigger**: "I want to start developing Triton kernels for Ascend NPU"

**Steps**:
1. Environment setup → `triton-operator/env-config/SKILL.md`
   - Check official docs for current version requirements
   - Install: conda → CANN → torch/torch_npu → triton-ascend
   - Verify all components work
2. First kernel: implement a simple elementwise kernel → `kernel-ops/codegen/SKILL.md`
   - Use Template 3 (activation/elementwise)
   - Apply universal rules: masks, alignment hints, fp32 upcast
3. Static review the first kernel → `triton-operator/code-review/SKILL.md`
   - Learn the P0/P1/P2 classification
   - Fix any issues found

---

## Workflow 4: Plugin-Instrumented Kernel Development

**Trigger**: "Add tracing/monitoring while developing a kernel"

**Steps**:
1. Develop or update Hermes plugin → `hermes-infra/plugin-development/SKILL.md`
   - Register `pre_tool_call` and `post_tool_call` hooks
   - Log `cannsim_remote_run` invocations with their `gen_report` parameter
   - Use `get_hermes_home() / "logs" / "kernel_dev.log"`
2. Enable the plugin in config.yaml:
   ```
   plugins:
     enabled:
       - my-kernel-tracer
   ```
3. Develop kernel → `kernel-ops/codegen/SKILL.md` (plugin logs every cannsim invocation)
4. Review plugin logs to understand the full cannsim call history

---

## Custom Workflow Combination (No Predefined Match — Fallback Mechanism)

When user intent involves multiple skills but doesn't match any predefined workflow above:

### Step 1: Identify Involved Skills
From the user intent, extract each sub-task and its corresponding skill
(refer to ROOT.md Phase 1 keyword signals):
- Mentions cannsim/trace/npubin → `kernel-ops/simulation`
- Mentions profile_kernels/perf_report → `kernel-ops/profiling`
- Mentions kernel code/tiling/UB → `kernel-ops/codegen`
- Mentions optimization/bottleneck/patterns → `kernel-ops/optimization`
- Mentions episode_write/retrieve → `kernel-ops/episode-memory`
- Mentions plugin/hook/OTel → `hermes-infra/plugin-development`
- Mentions code review/P0/P1 → `triton-operator/code-review`
- Mentions environment/CANN/conda → `triton-operator/env-config`
- Mentions full-process/orchestration → `triton-operator/orchestration`

### Step 2: Determine Execution Order
Arrange by data flow — which skill's output is the next skill's input:
- Dependencies present → serial execution
- No dependencies → parallel execution

**Typical dependency chain:**
```
env-config → episode-memory (retrieve) → codegen → code-review → simulation → optimization → episode-memory (write) → profiling
```

### Step 3: Route to Specific Leaf Nodes
For each sub-task, read the corresponding skill leaf SKILL.md to get exact execution instructions.
Do not guess instructions — always route through the tree.

### Step 4: Execute Serially
Follow the order from Step 2. Pass each step's output as input to the next step.

**Example custom workflow**: "I have a matmul kernel that passes code review but cannsim shows low cube utilization. I want to fix it and record what I learned."

1. Identify skills: simulation + optimization + episode-memory
2. Order: simulation first (get baseline trace) → optimization (apply cube fix) → simulation again (verify) → episode-memory (record)
3. Route:
   - `kernel-ops/simulation/SKILL.md` → run cannsim, get trace_core0.json
   - `kernel-ops/optimization/SKILL.md` → look up `aic_cube_ratio < 50%` fix: add `al.compile_hint(a, "dot_pad_only_k")`, ensure BLOCK multiples of 16
   - `kernel-ops/simulation/SKILL.md` → re-run cannsim, compare traces
   - `kernel-ops/episode-memory/SKILL.md` → `episode_write(kernel_name="matmul", ...)`
4. Execute each in sequence

## Constraints
- This leaf is a workflow combiner — always delegate actual work to individual skill leaves
- Custom workflow fallback must follow Steps 1-4 above, with real routing (not guessing)
- Any workflow involving optimization MUST end with `episode_write`
- Any workflow involving cannsim MUST use `gen_report=True`
