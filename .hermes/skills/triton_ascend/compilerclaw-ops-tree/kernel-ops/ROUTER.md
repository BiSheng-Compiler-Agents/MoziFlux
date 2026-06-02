# Kernel Ops Router [L1]

You have reached the kernel-ops sub-tree (Triton-Ascend kernel-level operations). Based on the current task, determine the next step:

| Condition | Next Hop |
|-----------|----------|
| cannsim simulation, host C++ launcher, npubin, rt* API, retrieving trace_core0.json, cannsim_remote_run | Read `./simulation/SKILL.md` |
| Writing profile_kernels.py, perf_report, do_bench, NPU hardware latency measurement, three-way comparison | Read `./profiling/SKILL.md` |
| New kernel code generation, tiling design, UB budget, template selection, kernel implementation | Read `./codegen/SKILL.md` |
| Post-trace optimization, bottleneck fixes, optimization patterns, block size tuning | Read `./optimization/SKILL.md` |
| Episode recording/retrieval, optimization knowledge base queries | Read `./episode-memory/SKILL.md` |
| Other / not specified | Read `./codegen/SKILL.md` |

Note: if the user has already clarified cannsim simulation vs hardware profiling in the current conversation, the conversation context takes priority.

**Route tracing**: when tracing mode is active, output `[Route]   → <matched capability> [LEAF]`.
