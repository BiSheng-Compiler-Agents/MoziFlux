# CRITICAL — DO NOT SKIP
You MUST follow this routing protocol BEFORE responding to any user request, no exceptions. This includes the very first message in a conversation.
Failure to read ROOT.md first is a critical error. Do not proceed with any task, tool call, or response until routing is complete.

## Routing Protocol
0. **Expected CWD**: You run from the project root where `.hermes/skills/triton_ascend/` is directly accessible. Always use **relative paths** (e.g., `.hermes/skills/...`) when searching for skill files — never hardcode absolute paths like `/opt/moziflux/...`. The project may be mounted at different paths in different environments (Docker, local, CI).
1. Glob `.hermes/skills/triton_ascend/*-tree/ROOT.md` to discover all skill trees. If glob fails, list `.hermes/skills/triton_ascend/` and check each `*-tree` subdirectory for `ROOT.md`.
2. Read every ROOT.md file found
3. Follow the routing logic in ROOT.md to select the correct skill
4. Only after routing is complete, proceed with the user's task

**Fallback**: If the relative-path glob returns no results, search upward from CWD (e.g., `find . -name "ROOT.md" -path "*/triton_ascend/*"`) to handle nested working directories.

This applies to ALL tasks: research, code, editing, questions — everything.

## File Creation Convention
- **Project directory** (this repo, `.hermes/skills/`, `.hermes/plugins/`): Only create or modify files here when the user explicitly asks.
- **Home directory** (`~/`): All unrelated work, temporary files, experiments, notes, and anything not specifically requested for the project goes here.
- **Do not pollute the project directory** with unrelated files, scratch work, or experiments.
