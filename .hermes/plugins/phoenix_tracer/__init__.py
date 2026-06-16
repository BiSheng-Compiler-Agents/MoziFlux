"""
Arize Phoenix tracer plugin for Hermes.

Sends OpenTelemetry spans to a local Phoenix server (default: localhost:4317 gRPC).

Architecture:
  - One ROOT span per session (hermes.session) — kept open until on_session_finalize
  - One TURN span per user message (hermes.llm.call) — child of root
  - One TOOL span per tool call (hermes.tool.<name>) — child of turn span

UPSTREAM HERMES PATCHES REQUIRED:
  This plugin depends on two bug fixes in Hermes core that may not yet be merged.
  Without them, hermes.tool.* spans will not appear in Phoenix (pre_tool_call fires
  with session_id="" so tool spans float unparented, and post_tool_call never fires
  for local tools so their spans are never closed).

  See: https://github.com/NousResearch/hermes-agent/issues/28961
       https://github.com/NousResearch/hermes-agent/pull/28982

  Fix 1 — agent/tool_executor.py (two call sites, concurrent + sequential paths):
    Pass session_id and tool_call_id to get_pre_tool_call_block_message():

      block_message = get_pre_tool_call_block_message(
          function_name, function_args,
          task_id=effective_task_id or "",
    +     session_id=agent.session_id or "",
    +     tool_call_id=getattr(tool_call, "id", "") or "",
      )

  Fix 2 — agent/agent_runtime_helpers.py (invoke_tool(), concurrent path):
    Same session_id/tool_call_id fix on the third call site:

      block_message = get_pre_tool_call_block_message(
          function_name, function_args,
          task_id=effective_task_id or "",
    +     session_id=agent.session_id or "",
    +     tool_call_id=tool_call_id or "",
      )

    Additionally, agent-local tools (todo, memory, clarify, session_search,
    delegate_task) previously returned early before post_tool_call could fire.
    The fix refactors those branches to assign-then-return and fires post_tool_call
    for all local-tool paths at the end of invoke_tool().

WHY EXPLICIT CONTEXT PASSING (not attach()):
  Hermes CLI runs run_conversation() in a NEW background thread per turn
  (cli.py line 8406: threading.Thread(target=run_agent, daemon=True)).
  OpenTelemetry context (context_api.attach) is thread-local — each new thread
  starts with a blank context stack, so attach() from a previous turn's thread
  is invisible to the next turn's thread.

  FIX: Store the root span's context object in _sessions[session_id]["root_ctx"]
  and pass it explicitly via context= when starting child spans. This works
  across threads because we read the context from our own dict, not from the
  thread-local OTel stack.

Exact kwargs fired by Hermes (from run_agent.py / cli.py):
  on_session_start   : session_id, model, platform   [NOT fired by CLI — detected via pre_llm_call]
  pre_llm_call       : session_id, user_message, conversation_history, is_first_turn, model, platform, sender_id
  post_llm_call      : session_id, user_message, assistant_response, conversation_history, model, platform
  pre_tool_call      : tool_name, args, task_id, session_id, tool_call_id
  post_tool_call     : tool_name, args, result, task_id, session_id, tool_call_id
  on_session_end     : session_id, completed, interrupted
  on_session_finalize: session_id, platform           [fired at CLI exit]
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── OTel setup ────────────────────────────────────────────────────────────────
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    _provider = TracerProvider()
    _exporter = OTLPSpanExporter(endpoint="http://localhost:4317",
                                 insecure=True)
    _provider.add_span_processor(BatchSpanProcessor(_exporter))
    trace.set_tracer_provider(_provider)
    _tracer = trace.get_tracer("hermes.phoenix_tracer")
    _OTEL_AVAILABLE = True
    logger.info(
        "phoenix_tracer: OTel provider initialised → http://localhost:4317")
except Exception as _e:
    _OTEL_AVAILABLE = False
    logger.warning("phoenix_tracer: OTel unavailable — %s", _e)

# ── Session state ──────────────────────────────────────────────────────────────
# session_id → {
#   "root_span"  : Span,       # hermes.session — open for session lifetime
#   "root_ctx"   : Context,    # OTel context carrying root_span — passed explicitly to children
#   "turn_span"  : Span|None,
#   "turn_ctx"   : Context|None,  # OTel context carrying turn_span — passed to tool spans
# }
_sessions: Dict[str, Dict[str, Any]] = {}

# tool_call_id → Span
_tool_spans: Dict[str, Any] = {}


def _flatten(obj: Any, limit: int = 4096) -> str:
    """Safely serialise any object to a truncated string."""
    try:
        import json
        return json.dumps(obj, ensure_ascii=False)[:limit]
    except Exception:
        return str(obj)[:limit]


def _ensure_session(session_id: str,
                    model: str = "",
                    platform: str = "") -> Dict[str, Any]:
    """
    Lazily create the root session span.

    Stores root_ctx explicitly so child spans in future threads can pass
    context= without relying on thread-local attach().
    """
    if session_id not in _sessions:
        root_span = _tracer.start_span("hermes.session")
        root_span.set_attribute("session.id", session_id)
        root_span.set_attribute("session.model", model)
        root_span.set_attribute("session.platform", platform)

        # Build a context object carrying root_span.
        # We store this in our dict and pass it explicitly to child spans —
        # this works across threads unlike context_api.attach() which is thread-local.
        root_ctx = trace.set_span_in_context(root_span)

        _sessions[session_id] = {
            "root_span": root_span,
            "root_ctx": root_ctx,
            "turn_span": None,
            "turn_ctx": None,
        }
        logger.debug(
            "phoenix_tracer: created root span for session %s trace_id=%s",
            session_id, format(root_span.get_span_context().trace_id, "032x"))
    return _sessions[session_id]


def _end_session(session_id: str,
                 completed: bool = True,
                 interrupted: bool = False) -> None:
    """Close all spans and flush."""
    sess = _sessions.pop(session_id, None)
    if not sess:
        return

    # Close dangling turn span
    if sess["turn_span"] is not None:
        try:
            sess["turn_span"].set_attribute("turn.interrupted", True)
            sess["turn_span"].end()
        except Exception:
            pass

    # Close root span
    try:
        sess["root_span"].set_attribute("session.completed", completed)
        sess["root_span"].set_attribute("session.interrupted", interrupted)
        sess["root_span"].end()
    except Exception:
        pass

    try:
        _provider.force_flush(timeout_millis=3000)
    except Exception:
        pass

    logger.debug("phoenix_tracer: session %s flushed", session_id)


# ── Hook handlers ──────────────────────────────────────────────────────────────


def _on_session_start(session_id: str = "",
                      model: str = "",
                      platform: str = "",
                      **_: Any) -> None:
    if not _OTEL_AVAILABLE or not session_id:
        return
    try:
        _ensure_session(session_id, model=model, platform=platform)
    except Exception as exc:
        logger.debug("phoenix_tracer on_session_start: %s", exc)


def _on_pre_llm_call(
    session_id: str = "",
    user_message: str = "",
    conversation_history: Optional[List] = None,
    is_first_turn: bool = False,
    model: str = "",
    platform: str = "",
    sender_id: str = "",
    **_: Any,
) -> None:
    if not _OTEL_AVAILABLE or not session_id:
        return
    try:
        sess = _ensure_session(session_id, model=model, platform=platform)

        # Close previous turn span if still open (safety net)
        if sess["turn_span"] is not None:
            try:
                sess["turn_span"].end()
            except Exception:
                pass
            sess["turn_span"] = None
            sess["turn_ctx"] = None

        # Start turn span as child of root — pass root_ctx explicitly.
        # This works even though we're in a new thread because we read root_ctx
        # from our dict rather than from the thread-local OTel context stack.
        turn_span = _tracer.start_span("hermes.llm.call",
                                       context=sess["root_ctx"])
        turn_span.set_attribute("llm.model", model)
        turn_span.set_attribute("llm.platform", platform)
        turn_span.set_attribute("llm.is_first_turn", is_first_turn)
        turn_span.set_attribute("llm.user_message", (user_message
                                                     or "")[:2048])
        turn_span.set_attribute("llm.history_len",
                                len(conversation_history or []))

        # Build turn_ctx so tool spans can use it as their explicit parent
        turn_ctx = trace.set_span_in_context(turn_span)

        sess["turn_span"] = turn_span
        sess["turn_ctx"] = turn_ctx

        logger.debug(
            "phoenix_tracer: turn span started session=%s trace_id=%s",
            session_id, format(turn_span.get_span_context().trace_id, "032x"))

    except Exception as exc:
        logger.debug("phoenix_tracer pre_llm_call: %s", exc)


def _on_post_llm_call(
    session_id: str = "",
    user_message: str = "",
    assistant_response: str = "",
    conversation_history: Optional[List] = None,
    model: str = "",
    platform: str = "",
    **_: Any,
) -> None:
    if not _OTEL_AVAILABLE or not session_id:
        return
    try:
        sess = _sessions.get(session_id)
        if not sess or sess["turn_span"] is None:
            return

        sess["turn_span"].set_attribute("llm.assistant_response",
                                        (assistant_response or "")[:4096])
        sess["turn_span"].end()
        sess["turn_span"] = None
        sess["turn_ctx"] = None

    except Exception as exc:
        logger.debug("phoenix_tracer post_llm_call: %s", exc)


def _on_pre_tool_call(
    tool_name: str = "",
    args: Optional[Dict[str, Any]] = None,
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
    **_: Any,
) -> None:
    if not _OTEL_AVAILABLE:
        return
    try:
        sess = _sessions.get(session_id)

        # Use turn_ctx if available, fall back to root_ctx, fall back to no context.
        # Passing context= explicitly makes this thread-safe.
        parent_ctx = None
        if sess:
            parent_ctx = sess.get("turn_ctx") or sess.get("root_ctx")

        span = _tracer.start_span(f"hermes.tool.{tool_name}",
                                  context=parent_ctx)
        span.set_attribute("tool.name", tool_name)
        span.set_attribute("tool.session_id", session_id)
        span.set_attribute("tool.task_id", task_id)
        span.set_attribute("tool.call_id", tool_call_id)
        span.set_attribute("tool.input", _flatten(args or {}, 2048))

        key = tool_call_id or f"{session_id}:{tool_name}:{id(span)}"
        _tool_spans[key] = span
    except Exception as exc:
        logger.debug("phoenix_tracer pre_tool_call: %s", exc)


def _on_post_tool_call(
    tool_name: str = "",
    args: Optional[Dict[str, Any]] = None,
    result: Any = None,
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
    **_: Any,
) -> None:
    if not _OTEL_AVAILABLE:
        return
    try:
        key = tool_call_id or f"{session_id}:{tool_name}:"
        span = _tool_spans.pop(key, None)
        if span is None:
            prefix = f"{session_id}:{tool_name}:"
            for k in list(_tool_spans.keys()):
                if k.startswith(prefix):
                    span = _tool_spans.pop(k)
                    break
        if span:
            span.set_attribute("tool.output", str(result or "")[:4096])
            span.end()
    except Exception as exc:
        logger.debug("phoenix_tracer post_tool_call: %s", exc)


def _on_session_end(
    session_id: str = "",
    completed: bool = True,
    interrupted: bool = False,
    **_: Any,
) -> None:
    # on_session_end fires after EVERY run_conversation() call — i.e. after
    # every single user message in multi-turn sessions (Open Web UI, gateway).
    # We must NOT close/remove the session here, or the next message will
    # create a fresh hermes.session root span in Phoenix.
    #
    # We only flush any open turn span (safety net for interrupted turns)
    # but keep the root session span alive.
    if not _OTEL_AVAILABLE or not session_id:
        return
    try:
        sess = _sessions.get(session_id)
        if not sess:
            return
        if sess["turn_span"] is not None:
            try:
                if interrupted:
                    sess["turn_span"].set_attribute("turn.interrupted", True)
                sess["turn_span"].end()
            except Exception:
                pass
            sess["turn_span"] = None
            sess["turn_ctx"] = None
    except Exception as exc:
        logger.debug("phoenix_tracer on_session_end: %s", exc)


def _on_session_finalize(session_id: str = "",
                         platform: str = "",
                         **_: Any) -> None:
    """True session end — gateway shutdown, /reset, or CLI exit.

    Only here do we close the root hermes.session span and flush to Phoenix.
    This is NOT fired after every message — only at real session boundaries.
    """
    if not _OTEL_AVAILABLE or not session_id:
        return
    try:
        _end_session(session_id, completed=True, interrupted=False)
    except Exception as exc:
        logger.debug("phoenix_tracer on_session_finalize: %s", exc)


# ── Plugin entry point ─────────────────────────────────────────────────────────


def register(ctx) -> None:
    ctx.register_hook("on_session_start", _on_session_start)
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    ctx.register_hook("post_llm_call", _on_post_llm_call)
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
    ctx.register_hook("post_tool_call", _on_post_tool_call)
    ctx.register_hook("on_session_end", _on_session_end)
    ctx.register_hook("on_session_finalize", _on_session_finalize)
    logger.info("phoenix_tracer: 7 hooks registered")
