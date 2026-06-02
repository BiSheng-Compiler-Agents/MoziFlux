# Triton Operator Router [L1]

You have reached the triton-operator sub-tree (full-process operator development). Based on the current task, determine the next step:

| Condition | Next Hop |
|-----------|----------|
| Environment setup, CANN/torch_npu/triton-ascend installation, conda setup, version check | Read `./env-config/SKILL.md` |
| Static code review, code review, P0/P1/P2 issues, API misuse, mask check | Read `./code-review/SKILL.md` |
| Full-process operator development, end-to-end development, development orchestration, unsure which sub-skill to use | Read `./orchestration/SKILL.md` |
| Other / not specified | Read `./orchestration/SKILL.md` |

Note: if the user has already clarified code review vs full dev flow in the current conversation, the conversation context takes priority.

**Route tracing**: when tracing mode is active, output `[Route]   → <matched capability> [LEAF]`.
