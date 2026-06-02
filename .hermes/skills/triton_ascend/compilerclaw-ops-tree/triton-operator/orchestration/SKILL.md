# Triton Operator Full-Process Development Orchestration [LEAF NODE]

Ascend Triton operator full-process development orchestration. Use when developing a Triton
operator from scratch, performing an end-to-end development process, or unsure which
sub-skill to use. Automatically orchestrates all stages.

## Workflow Overview

Building a Triton operator consists of 5 stages:

| # | Stage | Output | Skill | Skippable |
|---|-------|--------|-------|-----------|
| 1 | Environment Configuration | Environment verification report | kernel-ops/env-config | Yes: if torch/torch_npu/triton already available |
| 2 | Code Generation | kernel + smoke test | kernel-ops/codegen | **No** |
| 3 | Static Review | Review report | triton-operator/code-review | **No** |
| 4 | Performance Optimization | Optimized code | kernel-ops/optimization + kernel-ops/simulation | No |
| 5 | Profiling Evaluation | Profile results + performance report | kernel-ops/profiling | Conditional: if NPU hardware available |

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
2. Understand compute logic and tiling strategy
3. Generate kernel code with all universal rules applied
4. Generate correctness smoke test

### Step 4: Static Review (Stage 3 — MANDATORY)

Follow `triton-operator/code-review` workflow:
1. Phase 1: Host Side review (grid, block config, core type)
2. Phase 2: Device Side review (masks, dtypes, precision, control flow)
3. Phase 3: Performance hazards review
4. Output review.md with P0/P1/P2 classification

**Must fix all P0 issues before proceeding.**

### Step 5: Performance Optimization (Stage 4 — MANDATORY)

Follow `kernel-ops/optimization` workflow:
1. Run kernel via `cannsim_remote_run(gen_report=True)` — get trace
2. Analyze trace with `aggregate_trace.py`
3. Identify dominant bottleneck
4. Apply optimization patterns (retrieve episodes first)
5. Iterate: re-run cannsim, compare traces
6. Write episode after each successful optimization

### Step 6: Profiling Evaluation (Stage 5 — when NPU hardware available)

Follow `kernel-ops/profiling` workflow:
1. Generate `profile_kernels.py` using `@triton.testing.perf_report`
2. Cover all dispatch paths (small/large/non-pow2/benchmark shapes)
3. Run unit test first, then benchmark
4. Save performance report

---

## Core Constraints

1. **Must complete full process**: Stages 3-5 cannot be skipped; cannot stop after code generation
2. **Track progress with ToDo**: One Task per stage, set `in_progress` when entering, `completed` when done
3. **No performance optimization before precision passes** (core principle)
4. **Output final report**: Correctness results + performance ratio + optimization history + conclusion
5. **Save findings to episode memory**: Use `episode_write` after every optimization

---

## Final Deliverables

After full process completion, **must** output the following files in the operator directory:

| File | Content | Required |
|------|---------|----------|
| `{operator_name}.py` | Kernel code + Host interface | Yes |
| `opt_{operator_name}.py` | Optimized kernel code + Host interface | Yes |
| `test_{operator_name}.cpp` | Host script to run with cannsim | Yes |
| `profile_kernels.py` | Performance evaluation script | Yes |
| `review.md` | Static review report | Yes |
| `performance_report.md` | Performance report | Yes |
| `Optimizations.md` | Optimizations applied | Yes |

---

## Common Traps

| Trap | Symptom | Correct Approach |
|------|---------|------------------|
| Stop after code generation | User thinks development is complete but no validation | Force execution of Stages 3-5 |
| Skip ToDo | Stages omitted, not traceable | Create Task for each stage |
| Confusing "generate tests" with "run tests" | Test files exist but never executed | Stage 5 must actually run tests |
| Skip episode recording | Knowledge lost for future optimizations | Write episode after every optimization |

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
| Record/retrieve episodes | `./kernel-ops/episode-memory/SKILL.md` |

---

## Anti-Pattern Checklist (NEVER)

- ❌ Start execution without creating task tracking
- ❌ Skip stages without updating task status
- ❌ Only generate code and claim "development complete"
- ❌ Use "just need code" as excuse to skip validation process
- ❌ Skip writing episodes after successful optimizations

## Constraints
- This skill is an orchestrator — it delegates actual work to sub-skills
- All 5 stages must complete for a kernel to be considered done
- Performance ratio (torch_npu time / Triton time) > 1.0 is the target
- cannsim simulation is the primary validation tool (not real NPU hardware)
