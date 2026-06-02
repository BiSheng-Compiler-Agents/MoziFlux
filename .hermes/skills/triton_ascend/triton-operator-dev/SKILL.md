---
name: triton-operator-dev
description: Ascend Triton operator full-process development orchestration. Use when developing a Triton operator from scratch, performing an end-to-end development process, or unsure which sub-skill to use. Automatically orchestrates: environment configuration → requirements design → code generation → static review → precision evaluation → performance evaluation → performance optimization. Keywords: full-process, development orchestration, end-to-end, workflow orchestration.
---

# Triton Operator Full-Process Development

## Workflow Overview

Building a Triton operator consists of 7 stages (including 1 conditional stage):

| # | Stage | Output | Skill | Skippable |
|---|-------|--------|-------|------------|
| 1 | Environment Configuration | Environment verification report | `triton-operator-env-config` | Yes: torch/torch_npu/triton already available |
| 2 | Code Generation | kernel + smoke test | `triton-operator-code-gen` | **No** |
| 3 | Static Review | Review report | `triton-operator-code-review` | **No** |
| 4 | Performance Optimization | Optimized code | `triton-operator-performance-optim` | No |
| 5 | Correctness Evaluation | Evaluation results | `triton-ascend-cannsim` & `triton-ascend-kernel-profiling` | Conditional: correctness is verified at stage 4 |

## ⚠️ Core Constraints

1. **Must complete full process**: Stages 3-5 cannot be skipped; cannot stop after code generation
2. **Track progress with ToDo**: One Task per stage, set `in_progress` when entering, `completed` when done
3. **No performance optimization before precision passes** (core principle of precision-eval)
4. **Output final report**: Correctness results + performance ratio + optimization history + conclusion and save them using `kernel-episode-memory` skill.

## Common Traps

| Trap | Symptom | Correct Approach |
|------|---------|------------------|
| Stop after code generation | User thinks development is complete but no validation | Force execution of Stages 5 |
| Skip ToDo | Stages omitted, not traceable | Create Task for each stage |
| Confusing "generate tests" with "run tests" | Test files exist but never executed | Stage 5 must actually run |

## Anti-Pattern Checklist (NEVER)

- ❌ Start execution without creating task tracking
- ❌ Skip stages without updating task status
- ❌ Only generate code and claim "development complete"
- ❌ Use "just need code" as excuse to skip validation process

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