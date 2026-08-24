"""
Arize Phoenix tracer plugin for Hermes.

Sends OpenTelemetry spans to a local Phoenix server (default: localhost:4317 gRPC).

Architecture:
  - One ROOT span per session (hermes.session), closed at finalization.
  - One TURN span per outer user turn (hermes.turn), child of the root.
  - One LLM span per actual provider request inside the tool loop.
  - One TOOL span per tool call.

The tracer is the final llm_execution middleware, so it captures the exact effective
request after guided-search/kernel-sandbox rewriting. API hooks correlate responses,
retries, and errors. OpenInference message/tool/token attributes let Phoenix render
conversation boxes instead of only raw JSON.
API retries/errors are separate spans. Payloads are complete by default;
PHOENIX_TRACE_MAX_CHARS may set an explicit cap for constrained OTLP deployments.

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

import copy
import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── OTel setup ────────────────────────────────────────────────────────────────
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import SpanLimits, TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    try:
        _max_span_attributes = max(
            128, int(os.getenv("PHOENIX_MAX_SPAN_ATTRIBUTES", "20000")))
    except ValueError:
        _max_span_attributes = 20000
    _provider = TracerProvider(span_limits=SpanLimits(
        max_span_attributes=_max_span_attributes,
        max_span_attribute_length=None,
    ))
    _otlp_endpoint = os.getenv("PHOENIX_OTLP_ENDPOINT",
                               "http://localhost:4317")
    _exporter = OTLPSpanExporter(endpoint=_otlp_endpoint, insecure=True)
    try:
        _max_export_batch_size = max(
            1, int(os.getenv("PHOENIX_MAX_EXPORT_BATCH_SIZE", "1")))
    except ValueError:
        _max_export_batch_size = 1
    _provider.add_span_processor(
        BatchSpanProcessor(_exporter,
                           max_export_batch_size=_max_export_batch_size))
    trace.set_tracer_provider(_provider)
    _tracer = trace.get_tracer("hermes.phoenix_tracer")
    _OTEL_AVAILABLE = True
    logger.info(
        "phoenix_tracer: OTel provider initialised → %s "
        "(max_span_attributes=%d, max_export_batch_size=%d)",
        _otlp_endpoint,
        _max_span_attributes,
        _max_export_batch_size,
    )
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

# Correlation maps for inner provider calls and tools.
_tool_spans: Dict[Tuple[str, str], Any] = {}
_api_spans: Dict[Tuple[str, str, int], Any] = {}
_effective_requests: Dict[Tuple[str, str, int], Any] = {}
_STATE_LOCK = threading.RLock()

try:
    _MAX_PAYLOAD_CHARS = max(0, int(os.getenv("PHOENIX_TRACE_MAX_CHARS", "0")))
except ValueError:
    _MAX_PAYLOAD_CHARS = 0


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict()
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        try:
            return vars(obj)
        except Exception:
            pass
    return str(obj)


def _set_error_status(span: Any, message: str) -> None:
    span.set_attribute("span.status", "ERROR")
    try:
        from opentelemetry.trace import Status, StatusCode
        span.set_status(Status(StatusCode.ERROR, message))
    except Exception:
        pass


def _flatten(obj: Any, limit: Optional[int] = None) -> str:
    """Serialize a payload completely unless an explicit cap is configured."""
    try:
        value = json.dumps(obj, ensure_ascii=False, default=_json_default)
    except Exception:
        value = str(obj)
    effective_limit = _MAX_PAYLOAD_CHARS if limit is None else max(0, limit)
    if effective_limit and len(value) > effective_limit:
        return value[:effective_limit] + (
            f"\n[PHOENIX PAYLOAD TRUNCATED: {len(value) - effective_limit} chars omitted]"
        )
    return value


def _set_payload(span: Any, prefix: str, value: Any) -> None:
    span.set_attribute(f"{prefix}.value", _flatten(value))
    span.set_attribute(f"{prefix}.mime_type", "application/json")


def _message_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                parts.append(str(text) if text is not None else _flatten(item))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if value is None:
        return ""
    return _flatten(value)


def _request_body(request: Any) -> Dict[str, Any]:
    if not isinstance(request, dict):
        return {}
    body = request.get("body")
    return body if isinstance(body, dict) else request


def _request_messages(request: Any) -> List[Dict[str, Any]]:
    body = _request_body(request)
    messages: List[Dict[str, Any]] = []
    instructions = body.get("instructions")
    if instructions:
        messages.append({"role": "system", "content": instructions})
    raw = body.get("messages")
    if raw is None:
        raw = body.get("input")
    if isinstance(raw, str):
        raw = [{"role": "user", "content": raw}]
    if not isinstance(raw, list):
        return messages
    for item in raw:
        if not isinstance(item, dict):
            messages.append({"role": "user", "content": str(item)})
            continue
        item_type = str(item.get("type") or "")
        if item_type == "function_call":
            messages.append({
                "role":
                "assistant",
                "content":
                "",
                "tool_calls": [{
                    "id": item.get("call_id") or item.get("id"),
                    "name": item.get("name"),
                    "arguments": item.get("arguments"),
                }],
            })
        elif item_type == "function_call_output":
            messages.append({
                "role": "tool",
                "tool_call_id": item.get("call_id"),
                "content": item.get("output"),
            })
        elif item.get("role"):
            messages.append(dict(item))
    return messages


def _set_message_attributes(span: Any, prefix: str,
                            messages: List[Dict[str, Any]]) -> None:
    for index, message in enumerate(messages):
        base = f"{prefix}.{index}.message"
        span.set_attribute(f"{base}.role", str(message.get("role") or "user"))
        span.set_attribute(f"{base}.content",
                           _message_content(message.get("content")))
        if message.get("name") is not None:
            span.set_attribute(f"{base}.name", str(message["name"]))
        if message.get("tool_call_id") is not None:
            span.set_attribute(f"{base}.tool_call_id",
                               str(message["tool_call_id"]))
        tool_calls = message.get("tool_calls") or []
        if not isinstance(tool_calls, list):
            continue
        for tool_index, tool_call in enumerate(tool_calls):
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            function = function if isinstance(function, dict) else tool_call
            tool_base = f"{base}.tool_calls.{tool_index}.tool_call"
            if tool_call.get("id") is not None:
                span.set_attribute(f"{tool_base}.id", str(tool_call["id"]))
            if function.get("name") is not None:
                span.set_attribute(f"{tool_base}.function.name",
                                   str(function["name"]))
            if function.get("arguments") is not None:
                span.set_attribute(
                    f"{tool_base}.function.arguments",
                    _message_content(function["arguments"]),
                )


def _set_request_semantics(span: Any, request: Any) -> None:
    body = _request_body(request)
    messages = _request_messages(request)
    _set_message_attributes(span, "llm.input_messages", messages)
    tools = body.get("tools")
    if isinstance(tools, list):
        for index, tool in enumerate(tools):
            span.set_attribute(f"llm.tools.{index}.tool.json_schema",
                               _flatten(tool))
    excluded = {"messages", "input", "instructions", "tools"}
    invocation = {
        key: value
        for key, value in body.items() if key not in excluded
    }
    span.set_attribute("llm.invocation_parameters", _flatten(invocation))


def _assistant_messages(response: Any,
                        assistant_message: Any) -> List[Dict[str, Any]]:
    message = assistant_message
    if (not isinstance(message, dict) or not message) and isinstance(
            response, dict):
        message = (response.get("assistant_message")
                   or response.get("assistant") or response.get("message"))
    if not isinstance(message, dict):
        return []
    normalized = dict(message)
    normalized.setdefault("role", "assistant")
    return [normalized]


def _set_usage_attributes(span: Any, usage: Any) -> None:
    if not isinstance(usage, dict):
        return
    prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
    completion = usage.get("completion_tokens", usage.get("output_tokens"))
    total = usage.get("total_tokens")
    if prompt is not None:
        span.set_attribute("llm.token_count.prompt", int(prompt))
    if completion is not None:
        span.set_attribute("llm.token_count.completion", int(completion))
    if total is None and prompt is not None and completion is not None:
        total = int(prompt) + int(completion)
    if total is not None:
        span.set_attribute("llm.token_count.total", int(total))


def _set_common_attributes(span: Any, **values: Any) -> None:
    for key, value in values.items():
        if value is not None:
            span.set_attribute(
                key,
                value if isinstance(value, (str, bool, int,
                                            float)) else _flatten(value),
            )


def _api_key(session_id: str, api_request_id: str,
             api_call_count: int) -> Tuple[str, str, int]:
    return session_id, api_request_id, int(api_call_count or 0)


def _tool_key(session_id: str,
              tool_call_id: str,
              tool_name: str = "") -> Tuple[str, str]:
    return session_id, tool_call_id or f"unidentified:{tool_name}"


def _ensure_session(session_id: str,
                    model: str = "",
                    platform: str = "") -> Dict[str, Any]:
    """Lazily create a cross-thread-safe root session span."""
    with _STATE_LOCK:
        if session_id not in _sessions:
            root_span = _tracer.start_span("hermes.session")
            _set_common_attributes(
                root_span,
                **{
                    "openinference.span.kind": "CHAIN",
                    "session.id": session_id,
                    "session.model": model,
                    "session.platform": platform,
                },
            )
            root_ctx = trace.set_span_in_context(root_span)
            _sessions[session_id] = {
                "root_span": root_span,
                "root_ctx": root_ctx,
                "turn_span": None,
                "turn_ctx": None,
                "turn_id": "",
            }
            logger.debug(
                "phoenix_tracer: created root span for session %s trace_id=%s",
                session_id,
                format(root_span.get_span_context().trace_id, "032x"))
        return _sessions[session_id]


def _end_session(session_id: str,
                 completed: bool = True,
                 interrupted: bool = False) -> None:
    """Close all outstanding spans for a session and flush them."""
    with _STATE_LOCK:
        sess = _sessions.pop(session_id, None)
        api_items = [(key, _api_spans.pop(key)) for key in list(_api_spans)
                     if key[0] == session_id]
        tool_items = [(key, _tool_spans.pop(key)) for key in list(_tool_spans)
                      if key[0] == session_id]
        for key in list(_effective_requests):
            if key[0] == session_id:
                _effective_requests.pop(key, None)
    for _, span in api_items + tool_items:
        try:
            span.set_attribute("span.interrupted", True)
            span.end()
        except Exception:
            pass
    if not sess:
        return
    if sess["turn_span"] is not None:
        try:
            sess["turn_span"].set_attribute("turn.interrupted", True)
            sess["turn_span"].end()
        except Exception:
            pass
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
    **extra: Any,
) -> None:
    """Open the outer-turn parent span; provider calls are traced separately."""
    if not _OTEL_AVAILABLE or not session_id:
        return
    try:
        sess = _ensure_session(session_id, model=model, platform=platform)
        with _STATE_LOCK:
            if sess["turn_span"] is not None:
                sess["turn_span"].set_attribute("turn.interrupted", True)
                sess["turn_span"].end()
            turn_index = int(sess.get("turn_index", 0)) + 1
            turn_id = str(
                extra.get("turn_id") or f"{session_id}:turn:{turn_index}")
            turn_span = _tracer.start_span("hermes.turn",
                                           context=sess["root_ctx"])
            _set_common_attributes(
                turn_span,
                **{
                    "openinference.span.kind": "CHAIN",
                    "session.id": session_id,
                    "turn.id": turn_id,
                    "turn.index": turn_index,
                    "llm.model": model,
                    "llm.platform": platform,
                    "llm.is_first_turn": is_first_turn,
                    "sender.id": sender_id,
                },
            )
            _set_payload(
                turn_span, "input", {
                    "user_message": user_message,
                    "conversation_history": conversation_history or [],
                })
            sess["turn_span"] = turn_span
            sess["turn_ctx"] = trace.set_span_in_context(turn_span)
            sess["turn_id"] = turn_id
            sess["turn_index"] = turn_index
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
    """Close the outer-turn span after the inner provider/tool loop finishes."""
    if not _OTEL_AVAILABLE or not session_id:
        return
    try:
        with _STATE_LOCK:
            sess = _sessions.get(session_id)
            if not sess or sess["turn_span"] is None:
                return
            turn_span = sess["turn_span"]
            _set_payload(
                turn_span, "output", {
                    "assistant_response": assistant_response,
                    "conversation_history": conversation_history or [],
                })
            turn_span.set_attribute("turn.completed", True)
            turn_span.end()
            sess["turn_span"] = None
            sess["turn_ctx"] = None
    except Exception as exc:
        logger.debug("phoenix_tracer post_llm_call: %s", exc)


def _on_llm_execution(
    request: Dict[str, Any],
    next_call,
    session_id: str = "",
    api_request_id: str = "",
    api_call_count: int = 0,
    **_: Any,
) -> Any:
    """Capture the exact request after all preceding execution middleware."""
    key = _api_key(session_id, api_request_id, api_call_count)
    try:
        exact_request = copy.deepcopy(request)
    except Exception:
        exact_request = dict(request)
    with _STATE_LOCK:
        _effective_requests[key] = exact_request
        span = _api_spans.get(key)
    if span is not None:
        _set_payload(span, "input", {"effective_request": exact_request})
        _set_request_semantics(span, exact_request)
        span.set_attribute("llm.request.capture_source",
                           "llm_execution_post_middleware")
    try:
        return next_call(request)
    finally:
        with _STATE_LOCK:
            _effective_requests.pop(key, None)


def _on_pre_api_request(
    session_id: str = "",
    task_id: str = "",
    turn_id: str = "",
    api_request_id: str = "",
    api_call_count: int = 0,
    model: str = "",
    provider: str = "",
    base_url: str = "",
    api_mode: str = "",
    platform: str = "",
    request: Any = None,
    request_messages: Any = None,
    conversation_history: Any = None,
    user_message: Any = None,
    middleware_trace: Any = None,
    approx_input_tokens: int = 0,
    request_char_count: int = 0,
    message_count: int = 0,
    tool_count: int = 0,
    started_at: Any = None,
    **_: Any,
) -> None:
    """Start one LLM span for every actual provider request."""
    if not _OTEL_AVAILABLE or not session_id:
        return
    try:
        key = _api_key(session_id, api_request_id, api_call_count)
        with _STATE_LOCK:
            effective_request = _effective_requests.get(key)
        captured_request = (effective_request
                            if effective_request is not None else request)
        sess = _ensure_session(session_id, model=model, platform=platform)
        parent_ctx = sess.get("turn_ctx") or sess.get("root_ctx")
        span = _tracer.start_span(
            f"hermes.llm.request.{int(api_call_count or 0)}",
            context=parent_ctx,
        )
        _set_common_attributes(
            span,
            **{
                "openinference.span.kind":
                "LLM",
                "session.id":
                session_id,
                "task.id":
                task_id,
                "turn.id":
                turn_id or sess.get("turn_id", ""),
                "llm.api_request_id":
                api_request_id,
                "llm.api_call_count":
                int(api_call_count or 0),
                "llm.model_name":
                model,
                "llm.provider":
                provider,
                "llm.base_url":
                base_url,
                "llm.api_mode":
                api_mode,
                "llm.approx_input_tokens":
                int(approx_input_tokens or 0),
                "llm.request_char_count":
                int(request_char_count or 0),
                "llm.message_count":
                int(message_count or 0),
                "llm.tool_count":
                int(tool_count or 0),
                "llm.started_at":
                started_at,
                "llm.middleware_trace":
                middleware_trace or [],
                "llm.request.capture_source":
                ("llm_execution_post_middleware" if effective_request
                 is not None else "pre_api_request_observer"),
            },
        )
        _set_payload(
            span, "input", {
                "effective_request": captured_request or {},
                "observer_request": request or {},
                "request_messages": request_messages or [],
                "conversation_history": conversation_history or [],
                "user_message": user_message,
            })
        span.set_attribute("llm.observer_request", _flatten(request or {}))
        _set_request_semantics(span, captured_request or {})
        with _STATE_LOCK:
            previous = _api_spans.pop(key, None)
            if previous is not None:
                previous.set_attribute("span.interrupted", True)
                previous.end()
            _api_spans[key] = span
    except Exception as exc:
        logger.debug("phoenix_tracer pre_api_request: %s", exc)


def _on_post_api_request(
    session_id: str = "",
    api_request_id: str = "",
    api_call_count: int = 0,
    response: Any = None,
    assistant_message: Any = None,
    usage: Any = None,
    finish_reason: str = "",
    api_duration: float = 0.0,
    response_model: Any = None,
    started_at: Any = None,
    ended_at: Any = None,
    **_: Any,
) -> None:
    """Attach the complete sanitized provider response and close its span."""
    if not _OTEL_AVAILABLE or not session_id:
        return
    try:
        key = _api_key(session_id, api_request_id, api_call_count)
        with _STATE_LOCK:
            span = _api_spans.pop(key, None)
        if span is None:
            sess = _ensure_session(session_id)
            span = _tracer.start_span(
                f"hermes.llm.response.{int(api_call_count or 0)}",
                context=sess.get("turn_ctx") or sess.get("root_ctx"),
            )
            span.set_attribute("span.unmatched_start", True)
        _set_common_attributes(
            span,
            **{
                "openinference.span.kind": "LLM",
                "session.id": session_id,
                "llm.api_request_id": api_request_id,
                "llm.api_call_count": int(api_call_count or 0),
                "llm.response_model": response_model,
                "llm.finish_reason": finish_reason,
                "llm.duration_seconds": float(api_duration or 0.0),
                "llm.started_at": started_at,
                "llm.ended_at": ended_at,
                "llm.usage": usage or {},
            },
        )
        _set_payload(
            span, "output", {
                "response": response or {},
                "assistant_message": assistant_message,
                "usage": usage or {},
                "finish_reason": finish_reason,
            })
        _set_message_attributes(
            span,
            "llm.output_messages",
            _assistant_messages(response, assistant_message),
        )
        _set_usage_attributes(span, usage)
        span.end()
    except Exception as exc:
        logger.debug("phoenix_tracer post_api_request: %s", exc)


def _on_api_request_error(
    session_id: str = "",
    api_request_id: str = "",
    api_call_count: int = 0,
    request: Any = None,
    error: Any = None,
    status_code: Any = None,
    retry_count: Any = None,
    max_retries: Any = None,
    retryable: Any = None,
    reason: Any = None,
    api_duration: float = 0.0,
    started_at: Any = None,
    ended_at: Any = None,
    **_: Any,
) -> None:
    """Record failed provider attempts, including their request and retry metadata."""
    if not _OTEL_AVAILABLE or not session_id:
        return
    try:
        key = _api_key(session_id, api_request_id, api_call_count)
        with _STATE_LOCK:
            span = _api_spans.pop(key, None)
        if span is None:
            sess = _ensure_session(session_id)
            span = _tracer.start_span(
                f"hermes.llm.error.{int(api_call_count or 0)}",
                context=sess.get("turn_ctx") or sess.get("root_ctx"),
            )
            _set_payload(span, "input", {"request": request or {}})
            span.set_attribute("span.unmatched_start", True)
        _set_common_attributes(
            span,
            **{
                "openinference.span.kind":
                "LLM",
                "session.id":
                session_id,
                "llm.api_request_id":
                api_request_id,
                "llm.api_call_count":
                int(api_call_count or 0),
                "error.type": (error or {}).get("type") if isinstance(
                    error, dict) else type(error).__name__,
                "error.message": (error or {}).get("message") if isinstance(
                    error, dict) else str(error or ""),
                "error.status_code":
                status_code,
                "error.retry_count":
                retry_count,
                "error.max_retries":
                max_retries,
                "error.retryable":
                retryable,
                "error.reason":
                reason,
                "llm.duration_seconds":
                float(api_duration or 0.0),
                "llm.started_at":
                started_at,
                "llm.ended_at":
                ended_at,
            },
        )
        _set_payload(span, "output", {"error": error or {}})
        _set_error_status(span, str(error or reason or "API error"))
        span.end()
    except Exception as exc:
        logger.debug("phoenix_tracer api_request_error: %s", exc)


def _on_pre_tool_call(
    tool_name: str = "",
    args: Optional[Dict[str, Any]] = None,
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
    **extra: Any,
) -> None:
    if not _OTEL_AVAILABLE or not session_id:
        return
    try:
        sess = _ensure_session(session_id)
        parent_ctx = sess.get("turn_ctx") or sess.get("root_ctx")
        span = _tracer.start_span(f"hermes.tool.{tool_name}",
                                  context=parent_ctx)
        _set_common_attributes(
            span,
            **{
                "openinference.span.kind": "TOOL",
                "session.id": session_id,
                "turn.id": extra.get("turn_id") or sess.get("turn_id", ""),
                "tool.name": tool_name,
                "tool.session_id": session_id,
                "tool.task_id": task_id,
                "tool.call_id": tool_call_id,
                "tool.api_request_id": extra.get("api_request_id"),
            },
        )
        _set_payload(span, "input", args or {})
        correlation = tool_call_id or f"unidentified:{tool_name}:{id(span)}"
        with _STATE_LOCK:
            key = (session_id, correlation)
            previous = _tool_spans.pop(key, None)
            if previous is not None:
                previous.set_attribute("span.interrupted", True)
                previous.end()
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
    **extra: Any,
) -> None:
    if not _OTEL_AVAILABLE or not session_id:
        return
    try:
        with _STATE_LOCK:
            span = _tool_spans.pop(
                _tool_key(session_id, tool_call_id, tool_name), None)
            if span is None and not tool_call_id:
                prefix = f"unidentified:{tool_name}:"
                fallback = next(
                    (key for key in _tool_spans
                     if key[0] == session_id and key[1].startswith(prefix)),
                    None,
                )
                if fallback is not None:
                    span = _tool_spans.pop(fallback)
        if span is None:
            sess = _ensure_session(session_id)
            span = _tracer.start_span(
                f"hermes.tool.{tool_name}",
                context=sess.get("turn_ctx") or sess.get("root_ctx"),
            )
            span.set_attribute("span.unmatched_start", True)
            _set_payload(span, "input", args or {})
        _set_common_attributes(
            span,
            **{
                "openinference.span.kind": "TOOL",
                "session.id": session_id,
                "tool.name": tool_name,
                "tool.task_id": task_id,
                "tool.call_id": tool_call_id,
                "tool.status": extra.get("status"),
                "tool.duration_ms": extra.get("duration_ms"),
                "tool.error_type": extra.get("error_type"),
                "tool.error_message": extra.get("error_message"),
                "tool.middleware_trace": extra.get("middleware_trace") or [],
            },
        )
        _set_payload(span, "output", result)
        parsed_result = result
        if isinstance(result, str):
            try:
                parsed_result = json.loads(result)
            except json.JSONDecodeError:
                parsed_result = None
        structured_error = bool(
            isinstance(parsed_result, dict)
            and (parsed_result.get("success") is False
                 or bool(parsed_result.get("error"))
                 or parsed_result.get("action") == "block"
                 or parsed_result.get("status") in {"error", "failed"}))
        if (extra.get("error_type")
                or extra.get("status") in {"error", "failed"}
                or structured_error):
            message = (str(
                parsed_result.get("error") or parsed_result.get("message")
                or "tool error") if isinstance(parsed_result, dict) else str(
                    extra.get("error_message") or "tool error"))
            _set_error_status(span, message)
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
        with _STATE_LOCK:
            sess = _sessions.get(session_id)
            dangling_api = [
                _api_spans.pop(key) for key in list(_api_spans)
                if key[0] == session_id
            ]
            dangling_tools = [
                _tool_spans.pop(key) for key in list(_tool_spans)
                if key[0] == session_id
            ]
        for span in dangling_api + dangling_tools:
            span.set_attribute("span.interrupted", True)
            span.end()
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


def _on_session_reset(session_id: str = "", **_: Any) -> None:
    if _OTEL_AVAILABLE and session_id:
        _end_session(session_id, completed=False, interrupted=True)


# ── Plugin entry point ─────────────────────────────────────────────────────────


def register(ctx) -> None:
    ctx.register_hook("on_session_start", _on_session_start)
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    ctx.register_hook("post_llm_call", _on_post_llm_call)
    ctx.register_hook("pre_api_request", _on_pre_api_request)
    ctx.register_hook("post_api_request", _on_post_api_request)
    ctx.register_hook("api_request_error", _on_api_request_error)
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
    ctx.register_hook("post_tool_call", _on_post_tool_call)
    ctx.register_hook("on_session_end", _on_session_end)
    ctx.register_hook("on_session_finalize", _on_session_finalize)
    ctx.register_hook("on_session_reset", _on_session_reset)
    ctx.register_middleware("llm_execution", _on_llm_execution)
    logger.info(
        "phoenix_tracer: 11 hooks and exact-request middleware registered")
