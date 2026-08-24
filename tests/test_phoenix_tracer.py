from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


class FakeSpan:
    _next_trace_id = 1

    def __init__(self, name, context=None):
        self.name = name
        self.context = context
        self.attributes = {}
        self.ended = False
        self.trace_id = FakeSpan._next_trace_id
        self.status = None
        FakeSpan._next_trace_id += 1

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def set_status(self, status):
        self.status = status

    def end(self):
        self.ended = True

    def get_span_context(self):
        return types.SimpleNamespace(trace_id=self.trace_id)


class FakeTracer:

    def __init__(self):
        self.spans = []

    def start_span(self, name, context=None):
        span = FakeSpan(name, context=context)
        self.spans.append(span)
        return span


class FakeProvider:

    def __init__(self, *args, **kwargs):
        self.processors = []
        self.flushes = 0

    def add_span_processor(self, processor):
        self.processors.append(processor)

    def force_flush(self, timeout_millis=0):
        self.flushes += 1
        return True


@pytest.fixture
def phoenix(monkeypatch):
    tracer = FakeTracer()
    trace_module = types.ModuleType("opentelemetry.trace")
    trace_module.set_tracer_provider = lambda provider: None
    trace_module.get_tracer = lambda name: tracer
    trace_module.set_span_in_context = lambda span: ("context", span)
    setattr(trace_module, "StatusCode", types.SimpleNamespace(ERROR="ERROR"))
    setattr(trace_module,
            "Status",
            lambda code, description="": (code, description))

    sdk_trace = types.ModuleType("opentelemetry.sdk.trace")
    sdk_trace.TracerProvider = FakeProvider
    setattr(sdk_trace, "SpanLimits", lambda **kwargs: ("span_limits", kwargs))
    sdk_export = types.ModuleType("opentelemetry.sdk.trace.export")
    sdk_export.BatchSpanProcessor = lambda exporter: ("processor", exporter)
    grpc_export = types.ModuleType(
        "opentelemetry.exporter.otlp.proto.grpc.trace_exporter")
    grpc_export.OTLPSpanExporter = lambda **kwargs: ("exporter", kwargs)

    modules = {
        "opentelemetry":
        types.ModuleType("opentelemetry"),
        "opentelemetry.trace":
        trace_module,
        "opentelemetry.sdk":
        types.ModuleType("opentelemetry.sdk"),
        "opentelemetry.sdk.trace":
        sdk_trace,
        "opentelemetry.sdk.trace.export":
        sdk_export,
        "opentelemetry.exporter":
        types.ModuleType("opentelemetry.exporter"),
        "opentelemetry.exporter.otlp":
        types.ModuleType("opentelemetry.exporter.otlp"),
        "opentelemetry.exporter.otlp.proto":
        types.ModuleType("opentelemetry.exporter.otlp.proto"),
        "opentelemetry.exporter.otlp.proto.grpc":
        types.ModuleType("opentelemetry.exporter.otlp.proto.grpc"),
        "opentelemetry.exporter.otlp.proto.grpc.trace_exporter":
        grpc_export,
    }
    modules["opentelemetry"].trace = trace_module
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delenv("PHOENIX_TRACE_MAX_CHARS", raising=False)

    path = Path(".hermes/plugins/phoenix_tracer/__init__.py").resolve()
    spec = importlib.util.spec_from_file_location("phoenix_tracer_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._tracer = tracer
    yield module, tracer
    module._sessions.clear()
    module._api_spans.clear()
    module._tool_spans.clear()
    module._effective_requests.clear()


def test_traces_every_provider_request_with_complete_payloads(phoenix):
    plugin, tracer = phoenix
    long_input = "request-" + "x" * 10000
    long_output = "response-" + "y" * 10000

    plugin._on_pre_llm_call(
        session_id="session-1",
        user_message="optimize",
        conversation_history=[{
            "role": "user",
            "content": "history"
        }],
        model="model",
        platform="python",
    )
    for count in (1, 2):
        plugin._on_pre_api_request(
            session_id="session-1",
            task_id="task-1",
            turn_id="turn-1",
            api_request_id=f"request-{count}",
            api_call_count=count,
            model="model",
            provider="provider",
            api_mode="chat_completions",
            request={
                "body": {
                    "messages": [{
                        "role": "user",
                        "content": long_input
                    }]
                }
            },
            middleware_trace=[{
                "plugin": "guided-search"
            }],
        )
        plugin._on_post_api_request(
            session_id="session-1",
            api_request_id=f"request-{count}",
            api_call_count=count,
            response={"assistant": {
                "content": long_output
            }},
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 20
            },
            finish_reason="tool_calls" if count == 1 else "stop",
        )

    llm_spans = [
        span for span in tracer.spans
        if span.name.startswith("hermes.llm.request")
    ]
    assert len(llm_spans) == 2
    for span in llm_spans:
        assert long_input in span.attributes["input.value"]
        assert long_output in span.attributes["output.value"]
        assert "TRUNCATED" not in span.attributes["input.value"]
        assert span.attributes["openinference.span.kind"] == "LLM"
        assert span.ended is True
        assert span.context[1].name == "hermes.turn"
        assert span.attributes["llm.input_messages.0.message.role"] == "user"
        assert long_input in span.attributes[
            "llm.input_messages.0.message.content"]
        assert span.attributes[
            "llm.output_messages.0.message.role"] == "assistant"
        assert long_output in span.attributes[
            "llm.output_messages.0.message.content"]
        assert span.attributes["llm.token_count.prompt"] == 10
        assert span.attributes["llm.token_count.completion"] == 20
        assert span.attributes["llm.token_count.total"] == 30


def test_exact_post_middleware_request_and_message_tool_metadata(phoenix):
    plugin, tracer = phoenix
    plugin._on_pre_llm_call(session_id="session-1", user_message="run")
    exact = {
        "body": {
            "model":
            "model",
            "instructions":
            "system instructions",
            "input": [{
                "role": "user",
                "content": "[GUIDED SEARCH] exact middleware context",
            }],
            "tools": [{
                "type": "function",
                "name": "remote_verify",
                "parameters": {
                    "type": "object"
                },
            }],
        }
    }

    plugin._on_pre_api_request(
        session_id="session-1",
        api_request_id="exact-1",
        api_call_count=1,
        request={
            "_truncated": True,
            "preview": "observer payload"
        },
        model="model",
        provider="provider",
    )

    def provider(request):
        plugin._on_post_api_request(
            session_id="session-1",
            api_request_id="exact-1",
            api_call_count=1,
            response={
                "assistant_message": {
                    "role":
                    "assistant",
                    "content":
                    "",
                    "tool_calls": [{
                        "id":
                        "call-1",
                        "name":
                        "remote_verify",
                        "arguments":
                        "{\"local_dir\":\"/tmp/kernel\"}",
                    }],
                }
            },
            assistant_message={},
            usage={
                "input_tokens": 12,
                "output_tokens": 3
            },
            finish_reason="tool_calls",
        )
        return "ok"

    assert plugin._on_llm_execution(
        request=exact,
        next_call=provider,
        session_id="session-1",
        api_request_id="exact-1",
        api_call_count=1,
    ) == "ok"

    span = next(item for item in tracer.spans
                if item.name == "hermes.llm.request.1")
    assert "[GUIDED SEARCH] exact middleware context" in span.attributes[
        "input.value"]
    assert "observer payload" in span.attributes["llm.observer_request"]
    assert span.attributes[
        "llm.request.capture_source"] == "llm_execution_post_middleware"
    assert span.attributes["llm.input_messages.0.message.role"] == "system"
    assert span.attributes["llm.input_messages.1.message.role"] == "user"
    assert "exact middleware context" in span.attributes[
        "llm.input_messages.1.message.content"]
    assert "remote_verify" in span.attributes["llm.tools.0.tool.json_schema"]
    assert span.attributes[
        "llm.output_messages.0.message.tool_calls.0.tool_call.function.name"] == "remote_verify"
    assert span.attributes["llm.token_count.total"] == 15
    assert plugin._effective_requests == {}


def test_traces_complete_tool_io_and_provider_errors(phoenix):
    plugin, tracer = phoenix
    plugin._on_pre_llm_call(session_id="session-1", user_message="run")
    tool_input = {"command": "x" * 9000}
    tool_output = {"output": "y" * 9000}
    plugin._on_pre_tool_call(
        tool_name="terminal",
        args=tool_input,
        task_id="task-1",
        session_id="session-1",
        tool_call_id="tool-1",
    )
    plugin._on_post_tool_call(
        tool_name="terminal",
        args=tool_input,
        result=tool_output,
        task_id="task-1",
        session_id="session-1",
        tool_call_id="tool-1",
        status="success",
        duration_ms=12.5,
    )
    plugin._on_pre_tool_call(
        tool_name="remote_verify",
        args={"local_dir": "/tmp/kernel"},
        session_id="session-1",
        tool_call_id="tool-error",
    )
    plugin._on_post_tool_call(
        tool_name="remote_verify",
        result={
            "success": False,
            "error": "remote timeout"
        },
        session_id="session-1",
        tool_call_id="tool-error",
    )
    plugin._on_pre_api_request(
        session_id="session-1",
        api_request_id="request-error",
        api_call_count=3,
        request={"body": {
            "messages": []
        }},
    )
    plugin._on_api_request_error(
        session_id="session-1",
        api_request_id="request-error",
        api_call_count=3,
        error={
            "type": "timeout",
            "message": "provider timed out"
        },
        retry_count=1,
        max_retries=3,
        retryable=True,
    )

    tool_span = next(span for span in tracer.spans
                     if span.name == "hermes.tool.terminal")
    assert "x" * 9000 in tool_span.attributes["input.value"]
    assert "y" * 9000 in tool_span.attributes["output.value"]
    assert tool_span.attributes["tool.duration_ms"] == 12.5
    assert tool_span.ended is True

    failed_tool_span = next(span for span in tracer.spans
                            if span.name == "hermes.tool.remote_verify")
    assert failed_tool_span.attributes["span.status"] == "ERROR"
    assert failed_tool_span.status is not None

    error_span = next(span for span in tracer.spans
                      if span.name == "hermes.llm.request.3")
    assert error_span.attributes["error.type"] == "timeout"
    assert error_span.attributes["span.status"] == "ERROR"
    assert error_span.ended is True


def test_registers_request_tool_and_session_hooks(phoenix):
    plugin, _ = phoenix

    class Context:

        def __init__(self):
            self.hooks = {}
            self.middleware = {}

        def register_hook(self, name, callback):
            self.hooks[name] = callback

        def register_middleware(self, name, callback):
            self.middleware[name] = callback

    context = Context()
    plugin.register(context)
    assert set(context.hooks) == {
        "on_session_start",
        "pre_llm_call",
        "post_llm_call",
        "pre_api_request",
        "post_api_request",
        "api_request_error",
        "pre_tool_call",
        "post_tool_call",
        "on_session_end",
        "on_session_finalize",
        "on_session_reset",
    }
    assert context.middleware == {"llm_execution": plugin._on_llm_execution}
