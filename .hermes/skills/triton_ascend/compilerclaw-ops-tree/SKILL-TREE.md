# CompilerClaw Ops Tree Overview

## Overview
This skill-tree organizes 9 skills into a cross-skill hierarchical routing tree, covering 3 skill domains.

## Routing Principle
```
User Intent → ROOT.md (Phase 1: Select Skill) → ROUTER.md (Phase 2: Select Capability) → LEAF SKILL.md
```

## Directory Structure
```
compilerclaw-ops-tree/
├── ROOT.md                                 # Phase 1: Select Skill (9 keyword groups)
├── SKILL-TREE.md                           # This file
├── GENERATION-REPORT.md                    # Generation evidence
│
├── hermes-infra/                           # Hermes plugin development domain
│   ├── ROUTER.md                           # Phase 2: plugin capabilities
│   └── plugin-development/SKILL.md        # → plugin development, lifecycle hooks, OTel tracing
│
├── kernel-ops/                             # Triton-Ascend kernel operations domain
│   ├── ROUTER.md                           # Phase 2: simulation vs profiling vs codegen vs optimization vs episodes
│   ├── simulation/SKILL.md                # → cannsim simulation, host C++ launcher, trace analysis
│   ├── profiling/SKILL.md                 # → profile_kernels.py, perf_report, NPU hardware benchmarking
│   ├── codegen/SKILL.md                   # → kernel code generation, tiling strategy, templates
│   ├── optimization/SKILL.md              # → bottleneck optimization, hardware constraints, patterns
│   └── episode-memory/SKILL.md            # → episode_write/retrieve, optimization knowledge base
│
├── triton-operator/                        # Triton operator full-process domain
│   ├── ROUTER.md                           # Phase 2: env vs review vs orchestration
│   ├── env-config/SKILL.md                # → CANN/torch_npu/triton-ascend installation and config
│   ├── code-review/SKILL.md               # → static code review, P0/P1/P2 classification
│   └── orchestration/SKILL.md             # → full-process development orchestration
│
├── shared/                                 # Shared reference files (copied from source skills)
│   └── references/
│       ├── triton-api-reference.md         # Complete Triton-Ascend API (779 lines): tl, al, bl extensions, NPUOptions
│       ├── tiling-strategies.md            # Detailed tiling methodology with UB budget formulas
│       ├── ascend-terminology.md           # Ascend hardware terminology, HIVM IR mapping
│       ├── hardware-architecture.md        # AI Core architecture, memory hierarchy, platform diffs (A2/A3 vs 910_95)
│       ├── templates.md                    # Kernel templates T1-T12 with full runnable code (749 lines)
│       ├── optimization-patterns.md        # Optimization rules, pitfalls G1-G7, case studies (487 lines)
│       ├── ascend-api-dtype-matrix.md      # Complete dtype support matrix for all ops
│       ├── ascend-triton-api-constraints.md # API constraints for static review (masking, grid, atomics, control flow)
│       ├── ascend-test-patterns.md         # Official test case patterns (FlashAttn, RoPE, Reduction, Precision)
│       ├── code-review-checklist.md        # Full static review checklist
│       └── code-review-report-template.md  # Review report output format
│
└── cross-cutting/
    └── SKILL.md                            # Cross-skill workflow combiner
```

## Capability → Leaf Node Mapping Table

| Skill | Capability | Leaf Node Path |
|-------|-----------|----------------|
| `hermes-plugin-development` | Plugin development, lifecycle hooks, tool/command registration | `hermes-infra/plugin-development/SKILL.md` |
| `hermes-plugin-development` | OTel/Phoenix tracing, multi-turn session span grouping | `hermes-infra/plugin-development/SKILL.md` |
| `hermes-plugin-development` | SSH plugin pitfalls, paramiko, built-in tool registration | `hermes-infra/plugin-development/SKILL.md` |
| `kernel-episode-memory` | episode_write, episode_retrieve, episode_update, episode_list | `kernel-ops/episode-memory/SKILL.md` |
| `kernel-episode-memory` | FTS5 query sanitization, episode schema | `kernel-ops/episode-memory/SKILL.md` |
| `triton-ascend-cannsim` | cannsim simulation, cannsim_remote_run, gen_report | `kernel-ops/simulation/SKILL.md` |
| `triton-ascend-cannsim` | Host C++ launcher, RT-only APIs, CMakeLists.txt | `kernel-ops/simulation/SKILL.md` |
| `triton-ascend-cannsim` | triton patches (get_arch, TRITON_ASCEND_ARCH) | `kernel-ops/simulation/SKILL.md` |
| `triton-ascend-cannsim` | trace_core0.json analysis, aggregate_trace.py, pipeline breakdown | `kernel-ops/simulation/SKILL.md` |
| `triton-ascend-cannsim` | AIV hardware findings (fp16 vs fp32, tl.zeros, UB limits) | `kernel-ops/simulation/SKILL.md` |
| `triton-ascend-kernel-profiling` | profile_kernels.py authoring, @perf_report | `kernel-ops/profiling/SKILL.md` |
| `triton-ascend-kernel-profiling` | Three-way comparison (torch_ref vs baseline vs optimized) | `kernel-ops/profiling/SKILL.md` |
| `triton-ascend-kernel-profiling` | do_bench, grid overflow guard, shape coverage | `kernel-ops/profiling/SKILL.md` |
| `triton-operator-code-gen` | Kernel code generation, tiling strategy, UB budget | `kernel-ops/codegen/SKILL.md` |
| `triton-operator-code-gen` | Templates 1-12 (reduction, GEMM, activation, attention, etc.) | `kernel-ops/codegen/SKILL.md` |
| `triton-operator-code-gen` | cannsim profiling step, bottleneck identification | `kernel-ops/codegen/SKILL.md` |
| `triton-operator-code-gen` | Episode-driven optimization loop, episode recording | `kernel-ops/codegen/SKILL.md` |
| `triton-operator-code-review` | Static code review, P0/P1/P2 classification | `triton-operator/code-review/SKILL.md` |
| `triton-operator-code-review` | Mask completeness, dtype compliance, precision handling | `triton-operator/code-review/SKILL.md` |
| `triton-operator-code-review` | Control flow constraints, tensor indexing constraints | `triton-operator/code-review/SKILL.md` |
| `triton-operator-code-review` | Atomic operation constraints, performance hazards | `triton-operator/code-review/SKILL.md` |
| `triton-operator-dev` | Full-process orchestration, 5-stage workflow | `triton-operator/orchestration/SKILL.md` |
| `triton-operator-dev` | Stage tracking, final deliverables list | `triton-operator/orchestration/SKILL.md` |
| `triton-operator-env-config` | CANN, torch_npu, triton-ascend installation | `triton-operator/env-config/SKILL.md` |
| `triton-operator-env-config` | Conda setup, version compatibility, verification | `triton-operator/env-config/SKILL.md` |
| `triton-operator-performance-optim` | Optimization workflow, hardware constraints | `kernel-ops/optimization/SKILL.md` |
| `triton-operator-performance-optim` | Contiguity rule, single-pass, compile hints | `kernel-ops/optimization/SKILL.md` |
| `triton-operator-performance-optim` | RoPE case study, GroupNorm case study | `kernel-ops/optimization/SKILL.md` |
| All cross-skill | Cross-skill workflow combination | `cross-cutting/SKILL.md` |

## Skill Coverage Statistics

| Skill | Capabilities | Leaf Nodes |
|-------|-------------|-----------|
| `hermes-plugin-development` | 3 | 1 |
| `kernel-episode-memory` | 2 | 1 |
| `triton-ascend-cannsim` | 5 | 1 |
| `triton-ascend-kernel-profiling` | 3 | 1 |
| `triton-operator-code-gen` | 4 | 1 |
| `triton-operator-code-review` | 4 | 1 |
| `triton-operator-dev` | 2 | 1 |
| `triton-operator-env-config` | 2 | 1 |
| `triton-operator-performance-optim` | 3 | 1 |
| shared | 0 | 0 |
| cross-cutting | 7 workflows | 1 |
| **Total** | **35** | **10** |

## Adding a New Skill Guide

1. Create `{new-skill}/ROUTER.md` + leaf node(s)
2. Update ROOT.md Phase 1 routing table + disambiguation rules
3. Update cross-cutting/SKILL.md (workflow definitions + dependencies)
4. Update this file's mapping table and coverage statistics
