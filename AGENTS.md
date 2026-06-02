# CRITICAL — DO NOT SKIP
You MUST follow this routing protocol BEFORE responding to any user request, no exceptions. This includes the very first message in a conversation.
Failure to read ROOT.md first is a critical error. Do not proceed with any task, tool call, or response until routing is complete.

## Routing Protocol
1. Glob `.hermes/skills/triton_ascend/*-tree/ROOT.md` to discover all skill trees. If glob fails, list `.hermes/skills/triton_ascend/` and check each `*-tree` subdirectory for `ROOT.md`.
2. Read every ROOT.md file found
3. Follow the routing logic in ROOT.md to select the correct skill
4. Only after routing is complete, proceed with the user's task

This applies to ALL tasks: research, code, editing, questions — everything.
