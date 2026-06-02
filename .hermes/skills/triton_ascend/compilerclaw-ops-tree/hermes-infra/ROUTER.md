# Hermes Infra Router [L1]

You have reached the hermes-infra sub-tree. Based on the current task, determine the next step:

| Condition | Next Hop |
|-----------|----------|
| Any Hermes plugin development, lifecycle hooks, tool/command registration, OTel tracing | Read `./plugin-development/SKILL.md` |
| Other / not specified | Read `./plugin-development/SKILL.md` |

Note: if the user has already specified the plugin type (tracer, logger, SSH tool, etc.) in the current conversation, the conversation context takes priority.

**Route tracing**: when tracing mode is active, output `[Route]   → plugin-development [LEAF]`.
