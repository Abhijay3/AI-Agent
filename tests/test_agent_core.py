import os
from types import SimpleNamespace

os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("TAVILY_API_KEY", "test-tavily-key")

import agent_core  # noqa: E402


def make_chunk(content=None, tool_calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content, tool_calls=tool_calls))])


def make_tool_call_delta(index, call_id=None, name=None, arguments=None):
    return SimpleNamespace(index=index, id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


def test_stream_turn_plain_text_response(monkeypatch):
    chunks = [make_chunk(content="Hello"), make_chunk(content=" world")]
    monkeypatch.setattr(agent_core, "call_model_stream", lambda messages: iter(chunks))

    messages = [{"role": "user", "content": "hi"}]
    events = list(agent_core.stream_turn(messages))

    assert events == [{"event": "token", "text": "Hello"}, {"event": "token", "text": " world"}]
    assert messages[-1] == {"role": "assistant", "content": "Hello world"}


def test_stream_turn_executes_tool_calls_then_streams_answer(monkeypatch):
    first_call_chunks = [
        make_chunk(tool_calls=[make_tool_call_delta(0, call_id="call1", name="calculator", arguments='{"operation":')]),
        make_chunk(tool_calls=[make_tool_call_delta(0, arguments='"add","a":2,"b":3}')]),
    ]
    second_call_chunks = [make_chunk(content="The answer is 5.")]
    responses = [iter(first_call_chunks), iter(second_call_chunks)]
    monkeypatch.setattr(agent_core, "call_model_stream", lambda messages: responses.pop(0))

    messages = [{"role": "user", "content": "what is 2+3"}]
    events = list(agent_core.stream_turn(messages))

    assert events[0] == {"event": "tool_call", "tool": "calculator"}
    assert events[-1] == {"event": "token", "text": "The answer is 5."}

    tool_messages = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["content"] == "5"


def test_stream_turn_detects_leaked_pseudo_tool_call(monkeypatch):
    # Llama/Groq occasionally emits a malformed function-call attempt as
    # plain text instead of a structured tool_calls delta.
    chunks = [
        make_chunk(content='<function.run_sql_query{"query": "SELECT 1"}'),
        make_chunk(content="</function>"),
    ]
    monkeypatch.setattr(agent_core, "call_model_stream", lambda messages: iter(chunks))

    messages = [{"role": "user", "content": "list products"}]
    events = list(agent_core.stream_turn(messages))

    assert len(events) == 1
    assert events[0]["event"] == "content_replace"
    # the raw leaked syntax must never reach the caller
    assert "<function" not in events[0]["text"]
    assert messages[-1]["content"] == events[0]["text"]


def test_stream_turn_retries_silently_on_early_stream_failure(monkeypatch):
    call_count = {"n": 0}

    def fake_call_model_stream(api_messages):
        call_count["n"] += 1
        if call_count["n"] == 1:
            def bad_gen():
                raise RuntimeError("Failed to call a function.")
                yield  # pragma: no cover - unreachable, makes this a generator
            return bad_gen()
        return iter([make_chunk(content="ok now")])

    monkeypatch.setattr(agent_core, "call_model_stream", fake_call_model_stream)

    messages = [{"role": "user", "content": "hi"}]
    events = list(agent_core.stream_turn(messages))

    assert call_count["n"] == 2
    assert events == [{"event": "token", "text": "ok now"}]


def test_stream_turn_gives_up_after_max_retries(monkeypatch):
    def always_fails(api_messages):
        def bad_gen():
            raise RuntimeError("still broken")
            yield  # pragma: no cover
        return bad_gen()

    monkeypatch.setattr(agent_core, "call_model_stream", always_fails)

    messages = [{"role": "user", "content": "hi"}]
    events = list(agent_core.stream_turn(messages))

    assert events == [{"event": "error", "message": "still broken"}]


def test_run_turn_wraps_stream_turn(monkeypatch):
    chunks = [make_chunk(content="plain answer")]
    monkeypatch.setattr(agent_core, "call_model_stream", lambda messages: iter(chunks))

    messages = [{"role": "user", "content": "hi"}]
    reply = agent_core.run_turn(messages)

    assert reply == "plain answer"
