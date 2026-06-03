# Session End vs Finalize — Which Hook to Use

## The Problem

Hermes has two session-end hooks that fire in different contexts:

| Hook | Fires when | AIAgent API? | CLI? |
|------|-----------|:-:|:-:|
| `on_session_end` | Session ends (any mode) | **Yes** | Yes |
| `on_session_finalize` | Clean CLI exit only | **No** | Yes |

## The Mistake

Registering only `on_session_finalize` and expecting it to fire when using
`AIAgent.run_conversation()` programmatically. It does NOT — the AIAgent API
never fires `on_session_finalize`.

## The Fix

**For plugins that need to run cleanup on session end:**
- Register `on_session_end` — it fires for both CLI and AIAgent API
- If you also need CLI-specific cleanup, register both and have
  `on_session_finalize` delegate to the same handler

**For Hermes-internal code (CLI only):**
- `on_session_finalize` is fine — it fires at clean CLI exit

## Real-World Impact

The `kernel-sandbox` plugin initially registered `on_session_finalize` for its
session-end logging. This meant the plugin's cleanup never ran when the agent
was invoked via `AIAgent.run_conversation()` (the programmatic API used by
`optimize_kernels.py`). Fixed by switching to `on_session_end`.

## How to Invoke Manually

If a plugin depends on these hooks and the entrypoint doesn't fire them
automatically, invoke them manually:

```python
from hermes_cli.plugins import invoke_hook
invoke_hook("on_session_end", session_id=session_id, platform="python")
```
