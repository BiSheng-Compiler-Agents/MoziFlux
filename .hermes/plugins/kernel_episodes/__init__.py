"""
kernel_episodes plugin
======================
SQLite-backed episode memory for recording and retrieving Triton kernel
optimization experiences on Ascend NPUs.

Each episode captures one optimization attempt:
  observation  : what the kernel looked like and what the problem was
  thoughts     : reasoning that led to the chosen optimization approach
  action       : what was changed, how, and in what form
  result       : speedup achieved, what worked, what to try next time

Namespace:
  kernel_name  : the operation being optimized (e.g. "softmax", "matmul", "layer_norm")
  target       : the hardware target (e.g. "ascend910_9589", "ascend950")

Database: $KERNEL_EPISODES_DB (default: ~/CompilerClaw/episodes.db)

Tools registered:
  episode_write    — record a new optimization episode
  episode_update   — update fields of an episode by id
  episode_delete   — delete an episode by id
  episode_retrieve — full-text search to recall relevant past episodes
  episode_list     — list episodes, optionally filtered by kernel_name / target
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Database connection
# ---------------------------------------------------------------------------


def _db_path() -> str:
    return os.environ.get(
        "KERNEL_EPISODES_DB",
        os.path.expanduser("~/CompilerClaw/episodes.db"),
    )


@contextmanager
def _db():
    path = _db_path()
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Episodes database not found at {path}. "
            "Set KERNEL_EPISODES_DB or ensure ~/CompilerClaw/episodes.db exists."
        )
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _row_to_dict(row: sqlite3.Row) -> dict:
    return dict(row)


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------


def _write_episode(
    kernel_name: str,
    target: str,
    observation: str,
    thoughts: str,
    action: str,
    result: str,
) -> dict[str, Any]:
    with _db() as conn:
        cur = conn.execute(
            """
            INSERT INTO episodes (kernel_name, target, observation, thoughts, action, result)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (kernel_name, target, observation, thoughts, action, result),
        )
        row = conn.execute("SELECT * FROM episodes WHERE id = ?",
                           (cur.lastrowid, )).fetchone()
        return _row_to_dict(row)


def _update_episode(
    episode_id: int,
    kernel_name: str | None = None,
    target: str | None = None,
    observation: str | None = None,
    thoughts: str | None = None,
    action: str | None = None,
    result: str | None = None,
) -> dict[str, Any]:
    updates = {
        k: v
        for k, v in {
            "kernel_name": kernel_name,
            "target": target,
            "observation": observation,
            "thoughts": thoughts,
            "action": action,
            "result": result,
        }.items() if v is not None
    }
    if not updates:
        raise ValueError("At least one field must be provided to update.")

    set_clause = ", ".join(f"{k} = ?" for k in updates)
    with _db() as conn:
        cur = conn.execute(
            f"UPDATE episodes SET {set_clause} WHERE id = ?",
            list(updates.values()) + [episode_id],
        )
        if cur.rowcount == 0:
            raise LookupError(f"No episode found with id={episode_id}.")
        row = conn.execute("SELECT * FROM episodes WHERE id = ?",
                           (episode_id, )).fetchone()
        return _row_to_dict(row)


def _delete_episode(episode_id: int) -> dict[str, Any]:
    with _db() as conn:
        row = conn.execute("SELECT * FROM episodes WHERE id = ?",
                           (episode_id, )).fetchone()
        if row is None:
            raise LookupError(f"No episode found with id={episode_id}.")
        conn.execute("DELETE FROM episodes WHERE id = ?", (episode_id, ))
        return _row_to_dict(row)


def _sanitize_fts_query(query: str) -> str:
    """Convert a free-form user query into a safe FTS5 MATCH expression.

    FTS5 treats bare tokens like ``foo:bar`` as column filters and bare special
    characters (``:`` ``-`` ``"`` ``(`` ``)`` etc.) as syntax. We tokenise on
    whitespace, strip those characters out of each token, and wrap each
    resulting token in double quotes so it's matched as a literal phrase.
    Tokens are joined with OR for recall-friendly retrieval.
    """
    import re

    tokens: list[str] = []
    for raw in query.split():
        cleaned = re.sub(r'[":\-()*^]', " ", raw).strip()
        for part in cleaned.split():
            if part:
                tokens.append(f'"{part}"')
    if not tokens:
        return '""'
    return " OR ".join(tokens)


def _retrieve_episodes(
    query: str,
    kernel_name: str | None = None,
    target: str | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Full-text search over all episode content fields."""
    query = _sanitize_fts_query(query)
    where_parts: list[str] = []
    where_vals: list[Any] = []
    if kernel_name:
        where_parts.append("e.kernel_name = ?")
        where_vals.append(kernel_name)
    if target:
        where_parts.append("e.target = ?")
        where_vals.append(target)

    where_clause = ("AND " + " AND ".join(where_parts)) if where_parts else ""

    with _db() as conn:
        rows = conn.execute(
            f"""
            SELECT e.*
            FROM episodes_fts
            JOIN episodes e ON episodes_fts.rowid = e.id
            WHERE episodes_fts MATCH ?
            {where_clause}
            ORDER BY rank
            LIMIT ?
            """,
            [query] + where_vals + [limit],
        ).fetchall()
        return [_row_to_dict(r) for r in rows]


def _list_episodes(
    kernel_name: str | None = None,
    target: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> list[dict[str, Any]]:
    where_parts: list[str] = []
    where_vals: list[Any] = []
    if kernel_name:
        where_parts.append("kernel_name = ?")
        where_vals.append(kernel_name)
    if target:
        where_parts.append("target = ?")
        where_vals.append(target)

    where_clause = ("WHERE " +
                    " AND ".join(where_parts)) if where_parts else ""

    with _db() as conn:
        rows = conn.execute(
            f"""
            SELECT * FROM episodes
            {where_clause}
            ORDER BY id DESC
            LIMIT ? OFFSET ?
            """,
            where_vals + [limit, offset],
        ).fetchall()
        return [_row_to_dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


def _handle_write(args: dict, **_) -> str:
    try:
        ep = _write_episode(
            kernel_name=args["kernel_name"],
            target=args["target"],
            observation=args["observation"],
            thoughts=args["thoughts"],
            action=args["action"],
            result=args["result"],
        )
        return json.dumps({"success": True, "episode": ep})
    except Exception as e:
        logger.exception("episode_write failed")
        return json.dumps({"success": False, "error": str(e)})


def _handle_update(args: dict, **_) -> str:
    try:
        ep = _update_episode(
            episode_id=int(args["id"]),
            kernel_name=args.get("kernel_name"),
            target=args.get("target"),
            observation=args.get("observation"),
            thoughts=args.get("thoughts"),
            action=args.get("action"),
            result=args.get("result"),
        )
        return json.dumps({"success": True, "episode": ep})
    except Exception as e:
        logger.exception("episode_update failed")
        return json.dumps({"success": False, "error": str(e)})


def _handle_delete(args: dict, **_) -> str:
    try:
        ep = _delete_episode(int(args["id"]))
        return json.dumps({"success": True, "deleted": ep})
    except Exception as e:
        logger.exception("episode_delete failed")
        return json.dumps({"success": False, "error": str(e)})


def _handle_retrieve(args: dict, **_) -> str:
    try:
        episodes = _retrieve_episodes(
            query=args["query"],
            kernel_name=args.get("kernel_name"),
            target=args.get("target"),
            limit=int(args.get("limit", 5)),
        )
        if not episodes:
            return f"No episodes found for query: {args['query']}."

        parts = [
            f"\n\n===== EPISODE {i} =====\n{json.dumps(ep, indent=2)}"
            for i, ep in enumerate(episodes)
        ]
        return "Retrieved episodes:" + "".join(parts)
    except Exception as e:
        logger.exception("episode_retrieve failed")
        return json.dumps({"success": False, "error": str(e)})


def _handle_list(args: dict, **_) -> str:
    try:
        episodes = _list_episodes(
            kernel_name=args.get("kernel_name"),
            target=args.get("target"),
            limit=int(args.get("limit", 20)),
            offset=int(args.get("offset", 0)),
        )
        return json.dumps({
            "success": True,
            "count": len(episodes),
            "episodes": episodes
        })
    except Exception as e:
        logger.exception("episode_list failed")
        return json.dumps({"success": False, "error": str(e)})


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


def register(ctx) -> None:

    _requires_env = ["KERNEL_EPISODES_DB"]

    def _check_fn():
        return True  # DB is created on first use; no pre-check needed

    ctx.register_tool(
        name="episode_write",
        toolset="triton_ascend",
        schema={
            "name":
            "episode_write",
            "description":
            ("Record a new Triton kernel optimization episode. "
             "Use this after completing an optimization attempt to save the experience "
             "for future recall. Write in first person ('I ...')."),
            "parameters": {
                "type":
                "object",
                "properties": {
                    "kernel_name": {
                        "type":
                        "string",
                        "description":
                        "The operation being optimized, e.g. 'softmax', 'matmul', 'layer_norm', 'relu'.",
                    },
                    "target": {
                        "type":
                        "string",
                        "description":
                        "The hardware target, e.g. 'ascend910_9589', 'ascend950'.",
                    },
                    "observation": {
                        "type":
                        "string",
                        "description":
                        "What the kernel looked like and what the problem was — baseline shape, latency, failure mode.",
                    },
                    "thoughts": {
                        "type":
                        "string",
                        "description":
                        "The reasoning that led to the chosen optimization approach. What did I notice? What alternatives did I consider?",
                    },
                    "action": {
                        "type":
                        "string",
                        "description":
                        "What optimization was applied: which pattern, what block sizes, what code changes, and how.",
                    },
                    "result": {
                        "type":
                        "string",
                        "description":
                        "Outcome: speedup achieved (before → after latency), what worked, what to try differently next time.",
                    },
                },
                "required": [
                    "kernel_name", "target", "observation", "thoughts",
                    "action", "result"
                ],
            },
        },
        handler=_handle_write,
        requires_env=_requires_env,
        check_fn=_check_fn,
    )

    ctx.register_tool(
        name="episode_update",
        toolset="triton_ascend",
        schema={
            "name": "episode_update",
            "description":
            "Update one or more fields of an existing episode by its id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "integer",
                        "description": "Episode id to update."
                    },
                    "kernel_name": {
                        "type": "string",
                        "description": "Updated kernel name."
                    },
                    "target": {
                        "type": "string",
                        "description": "Updated target hardware."
                    },
                    "observation": {
                        "type": "string",
                        "description": "Updated observation."
                    },
                    "thoughts": {
                        "type": "string",
                        "description": "Updated thoughts."
                    },
                    "action": {
                        "type": "string",
                        "description": "Updated action."
                    },
                    "result": {
                        "type": "string",
                        "description": "Updated result."
                    },
                },
                "required": ["id"],
            },
        },
        handler=_handle_update,
        requires_env=_requires_env,
        check_fn=_check_fn,
    )

    ctx.register_tool(
        name="episode_delete",
        toolset="triton_ascend",
        schema={
            "name": "episode_delete",
            "description": "Delete an episode by its id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "integer",
                        "description": "Episode id to delete."
                    },
                },
                "required": ["id"],
            },
        },
        handler=_handle_delete,
        requires_env=_requires_env,
        check_fn=_check_fn,
    )

    ctx.register_tool(
        name="episode_retrieve",
        toolset="triton_ascend",
        schema={
            "name":
            "episode_retrieve",
            "description":
            ("Search past kernel optimization episodes by relevance. "
             "Use this before starting an optimization to recall what worked on similar kernels. "
             "Use affirmative form for the query, e.g. 'softmax wide rows MTE stall fix' "
             "rather than 'how do I fix softmax?'. "
             "Optionally filter by kernel_name or target to narrow results."),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type":
                        "string",
                        "description":
                        "What to search for. Use keywords describing the kernel, problem, or optimization pattern.",
                    },
                    "kernel_name": {
                        "type":
                        "string",
                        "description":
                        "Restrict search to a specific kernel operation.",
                    },
                    "target": {
                        "type":
                        "string",
                        "description":
                        "Restrict search to a specific hardware target.",
                    },
                    "limit": {
                        "type":
                        "integer",
                        "description":
                        "Maximum number of episodes to return (default: 5).",
                    },
                },
                "required": ["query"],
            },
        },
        handler=_handle_retrieve,
        requires_env=_requires_env,
        check_fn=_check_fn,
    )

    ctx.register_tool(
        name="episode_list",
        toolset="triton_ascend",
        schema={
            "name": "episode_list",
            "description":
            "List recorded optimization episodes, newest first. Filter by kernel_name and/or target.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kernel_name": {
                        "type": "string",
                        "description": "Filter by kernel operation."
                    },
                    "target": {
                        "type": "string",
                        "description": "Filter by hardware target."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max episodes to return (default: 20)."
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Pagination offset (default: 0)."
                    },
                },
                "required": [],
            },
        },
        handler=_handle_list,
        requires_env=_requires_env,
        check_fn=_check_fn,
    )
