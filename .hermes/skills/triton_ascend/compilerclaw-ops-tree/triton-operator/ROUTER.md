# Triton Operator Router [L1]

You have reached the triton-operator sub-tree (full-process operator development). Based on the current task, determine the next step:

| Condition | Next Hop |
|-----------|----------|
| Kernel optimization task (any phrasing) — load for deliverables checklist and code review | Read `./orchestration/SKILL.md` and `./code-review/SKILL.md` |
| Environment setup, CANN/torch_npu/triton-ascend installation, conda setup, version check | Read `./env-config/SKILL.md` |
| Static code review, code review, P0/P1/P2 issues, API misuse, mask check | Read `./code-review/SKILL.md` |
| Full-process operator development, end-to-end development, development orchestration, unsure which sub-skill to use | Read `./orchestration/SKILL.md` |
| Other / not specified | Read `./orchestration/SKILL.md` |

Note: for kernel optimization tasks, this router is entered in parallel with `kernel-ops/ROUTER.md` per the Mandatory Dual-Tree Rule in ROOT.md. The role of this sub-tree in optimization is:
- `orchestration/SKILL.md` → provides the required deliverables checklist (review.md, opt_*.py, profile_kernels.py, Optimizations.md, performance_report.md)
- `code-review/SKILL.md` → provides P0/P1/P2 static review of the optimized kernel

**Route tracing**: when tracing mode is active, output `[Route]   → <matched capability> [LEAF]`.
