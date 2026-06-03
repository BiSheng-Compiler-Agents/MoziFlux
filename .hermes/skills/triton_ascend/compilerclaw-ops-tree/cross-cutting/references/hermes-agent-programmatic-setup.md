# Hermes Agent Programmatic Setup

When using `AIAgent` from `run_agent` programmatically (outside the Hermes CLI), the module's import-time code calls `load_hermes_dotenv()` to discover the `.env` file. This has implications for where your credentials live.

## How `.env` Discovery Works

In `run_agent.py` (at module import time):

```python
_hermes_home = get_hermes_home()                # from HERMES_HOME env var or ~/.hermes
_project_env = Path(__file__).parent / '.env'   # the Hermes install dir
load_hermes_dotenv(hermes_home=_hermes_home, project_env=_project_env)
```

`load_hermes_dotenv()` looks for:
1. `<HERMES_HOME>/.env` — the user env (overrides existing vars)
2. `<project_env>` — the Hermes install `.env` (fallback, usually absent)

## The Pitfall

If `HERMES_HOME` is **not set** in the environment, `get_hermes_home()` returns `Path.home() / ".hermes"`. This means your `.env` must be at `~/.hermes/.env`, which may not be where you keep it (e.g., in a Docker deployment where `.env` is at `/opt/data/.env`).

## The Fix

Set `HERMES_HOME` **before** any import of `run_agent` or `AIAgent`:

```python
import os
os.environ.setdefault("HERMES_HOME", "/opt/data")  # or your actual path

# Safe to import now — it reads HERMES_HOME at import time
from run_agent import AIAgent
```

`setdefault` only writes when the variable is unset, so the shell environment takes priority when present.

## When It Matters

- **Docker/container deployments** where `HOME` points to one path but Hermes data and `.env` live at a different mount point.
- **Batch pipelines** like `optimize_kernels.py` that create `AIAgent` instances programmatically.
- **Any script** that does `from run_agent import AIAgent` outside the `hermes` CLI context.
