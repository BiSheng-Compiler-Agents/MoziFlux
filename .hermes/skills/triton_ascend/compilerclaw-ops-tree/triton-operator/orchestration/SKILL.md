---
name: orchestration
description: Ascend Triton operator full-process development orchestration. Use when developing a Triton operator from scratch, performing an end-to-end development process, or unsure which sub-skill to use. Automatically orchestrates all stages.
---

# Triton Operator Full-Process Development Orchestration [LEAF NODE]

Ascend Triton operator full-process development orchestration. Use when developing a Triton
operator from scratch, performing an end-to-end development process, or unsure which
sub-skill to use. Automatically orchestrates all stages.

## Workflow Overview

Building a Triton operator consists of 6 stages (including 2 conditional stages):

| # | Stage | Output | Skill | Skippable |
|---|-------|--------|-------|-----------|
| 1 | Environment Configuration | Environment verification report | triton-operator/env-config | Yes: if torch/torch_npu/triton already available |
| 2 | Code Generation | kernel + smoke test | kernel-ops/codegen | **No** |
| 3 | Static Review | Review report | triton-operator/code-review | **No** |
| 4 | Performance Optimization | Optimized code | kernel-ops/optimization + kernel-ops/simulation | No |
| 5 | Profiling + Hardware Verification | Profile results + correctness + benchmark | kernel-ops/profiling + remote-verify plugin | Conditional: if NPU hardware available |
| 6 | Episode Recording | Episode in episodes.db | kernel-ops/episode-memory | **No** |

---

## Orchestration Instructions

### Step 1: Create Task Tracking

Create one Task per stage and set to `in_progress` when entering, `completed` when done.
**NEVER skip stage tracking** — stages that aren't tracked tend to get omitted.

### Step 2: Check Environment (Stage 1)

- If torch, torch_npu, and triton-ascend are already available and working → skip to Stage 2
- Otherwise → follow `triton-operator/env-config` to set up environment

### Step 3: Code Generation (Stage 2 — MANDATORY)

Follow `kernel-ops/codegen` workflow:
1. Retrieve past episodes: `episode_retrieve(query="<kernel_type> <problem>", target="ascend950")`
2. Understand compute logic, tiling strategy, and stated shape regime from problem name/docstring/`get_inputs()` (large-K, small-K, tall-skinny, irregular, etc.)
3. Generate kernel code with all universal rules applied and preserve that regime in tiling/dispatch choices
4. Generate correctness smoke test covering all relevant dimensions inside the required regime

### Step 4: Static Review (Stage 3 — MANDATORY)

Follow `triton-operator/code-review` workflow:
1. Phase 1: Host Side review (grid, block config, core type)
2. Phase 2: Device Side review (masks, dtypes, precision, control flow)
3. Phase 3: Performance hazards review
4. Output review.md with P0/P1/P2 classification

**Must fix all P0 issues before proceeding.**

### Step 5: Performance Optimization (Stage 4 — MANDATORY)

Follow `kernel-ops/optimization` workflow:
1. Retrieve past episodes for relevant patterns
2. Run kernel via `cannsim_local_run(gen_report=True)` with a **sub-kernel host**
   (grid=1, M=BLOCK_M, K=1×BLOCK_K — see simulation skill Rule 1 and pitfall #18)
3. Analyze trace with `aggregate_trace.py`
4. Identify dominant bottleneck
5. Apply optimization patterns
6. Iterate: re-run cannsim, compare traces
7. Write episode after each successful optimization

### Step 6: Profiling + Hardware Verification (Stage 5 — MANDATORY when hardware available)

This stage has two parts:

#### Part A: Generate profile_kernels.py

Follow `kernel-ops/profiling` workflow:
1. Generate `profile_kernels.py` using `@triton.testing.perf_report`
2. Choose `_BENCH_SHAPES` from the kernel requirement/regime and original `get_inputs()`; e.g. large-K stays large-K, small-K stays small-K, tall-skinny stays `M >> N` or `N >> M`, irregular includes non-power-of-two/boundary shapes
3. Cover all dispatch paths and include the exact required benchmark shape
4. Include unit test with weight-init matching for every provider on every `_BENCH_SHAPES` entry
5. Use lazy NPU model init (no module-level ModelNew)
6. Reference column name is **always** `"PyTorch / ACL"`

#### Part B: Remote Hardware Verification

Use the `remote_verify` tool to run on real NPU hardware. The **pass/fail
verification RESULT must come from `remote_verify`** — simulation is not a
substitute for physical hardware when deciding correctness.

`cannsim_local_run` remains available throughout (including during the verify
stage) for bottleneck analysis, trace inspection, and further optimization.
It simply does not *count* as the verification result. Use cannsim to
understand and improve the kernel; use remote_verify to validate it.

```python
result = remote_verify(
    local_dir="/path/to/kernel_dir",
    run_test=True,    # correctness check first
    run_bench=True,   # benchmark if test passes
    timeout=600,
)
```

The tool uploads files via SFTP, runs correctness then benchmark on the remote NPU machine, and downloads results back to `local_dir/remote_results/`.

**Status timing:** if `kernel_status` is still in `optimize`, first finish deliverables and call `kernel_status(..., action="advance")` to enter `verify`; then run `remote_verify`. A remote run performed before the pipeline reaches `verify` may not satisfy stage validation and can require a second run.

### Step 7: Episode Recording (Stage 6 — MANDATORY)

After getting remote results, record the full optimization journey:

```python
episode_write(
    kernel_name="<op_type>",
    target="ascend950",
    observation="Baseline: <bottleneck>, <latency>. Remote: <hardware_latency>",
    thoughts="<what patterns were tried, what worked, what didn't>",
    action="<key optimizations applied>",
    result="<before/after latency>, <vs_torch_npu>, <hardware_verified: PASS/FAIL>",
)
```

**Record-stage completion rule:** when `kernel_status` reports `stage="record"`, do not stop after deliverables or verification. First record the episode, then set `kernel_status(..., key="recorded", value=True)`, then call `kernel_status(..., action="advance")`, and finally read status to confirm `stage="done"`. A response that reports `record` status without advancing to `done` is incomplete.

---

## Core Constraints

1. **Must complete full process**: Stages 3-6 cannot be skipped
2. **Track progress with ToDo**: One Task per stage
3. **No performance optimization before precision passes**
4. **Output final report**: Correctness results + performance ratio + optimization history
5. **Save findings to episode memory**: Use `episode_write` after every optimization

---

## Final Deliverables

After full process completion, **must** output the following files in the operator directory:

| File | Content | Required |
|------|---------|----------|
| `opt_{name}.py` | Optimized kernel code + ModelNew host interface | Yes |
| `profile_kernels.py` | Performance evaluation script (generated last) | Yes |
| `review.md` | Static review report (P0/P1/P2) | Yes |
| `performance_report.md` | Performance report with hardware latency | Yes |
| `Optimizations.md` | Each optimization applied with code snippets and rationale | Yes |
| `remote_results/` | Downloaded results from remote hardware verification | If hardware available |

---

## Common Traps

| Trap | Symptom | Correct Approach |
|------|---------|------------------|
| Stop after code generation | No validation | Force execution of Stages 3-6 |
| Skip ToDo | Stages omitted | Create Task for each stage |
| Generate profile before optimization | Profile tests wrong code | Generate profile_kernels.py AFTER optimization |
| Run profile locally without NPU | RuntimeError | Use remote_verify tool for hardware |
| Skip episode recording | Knowledge lost | Write episode after every optimization |
| Write deliverables outside workspace | Files scattered | Write inside the workspace directory |
| Set kernel status flags too early | Flags reset by pipeline | Only set `verified=True` when stage=`verify`, `recorded=True` when stage=`record`. Setting them early gets reset on stage advance. |
| Cannsim output exceeds tool limit | 242K+ char output truncated | Read trace files from disk (`/tmp/cannsim_local/<job_name>/report/trace_core0.json`) instead of relying on tool return |
| Stop after code generation | No validation | Force execution of Stages 3-6 |
| Skip ToDo | Stages omitted | Create Task for each stage |
| Generate profile before optimization | Profile tests wrong code | Generate profile_kernels.py AFTER optimization |
| Run profile locally without NPU | RuntimeError | Use remote_verify tool for hardware |
| Skip episode recording | Knowledge lost | Write episode after every optimization |

---

## Routing Guide

When you need to perform a specific sub-task, route to the appropriate leaf:

| Task | Route |
|------|-------|
| Environment setup | `./triton-operator/env-config/SKILL.md` |
| Generate kernel code | `./kernel-ops/codegen/SKILL.md` |
| Static code review | `./triton-operator/code-review/SKILL.md` |
| Run cannsim simulation | `./kernel-ops/simulation/SKILL.md` |
| Analyze trace and optimize | `./kernel-ops/optimization/SKILL.md` |
| Write profile_kernels.py | `./kernel-ops/profiling/SKILL.md` |
| Remote hardware verification | `remote_verify` tool (remote-verify plugin) |
| Record/retrieve episodes | `./kernel-ops/episode-memory/SKILL.md` |

---

## Anti-Pattern Checklist (NEVER)

- ❌ Start execution without creating task tracking
- ❌ Skip stages without updating task status
- ❌ Only generate code and claim "development complete"
- ❌ Generate profile_kernels.py before optimization is complete
- ❌ Run profile_kernels.py locally without NPU hardware
- ❌ Skip writing episodes after successful optimizations
- ❌ Write deliverables outside the kernel workspace directory
