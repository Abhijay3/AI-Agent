import os
import time
from datetime import date
from types import SimpleNamespace

import httpx

os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("TAVILY_API_KEY", "test-tavily-key")

import agent_core  # noqa: E402
from openai import RateLimitError  # noqa: E402


def make_chunk(content=None, tool_calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content, tool_calls=tool_calls))])


def make_tool_call_delta(index, call_id=None, name=None, arguments=None):
    return SimpleNamespace(index=index, id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


def test_build_system_message_injects_remembered_facts(monkeypatch):
    monkeypatch.setattr(
        agent_core,
        "get_all_memories",
        lambda user_id: {"name": "Abhi", "role": "full stack developer"} if user_id == "u1" else {},
    )
    monkeypatch.setattr(agent_core, "retrieve", lambda query: [])

    system_message = agent_core._build_system_message([{"role": "user", "content": "hi"}], "u1")

    assert "- name: Abhi" in system_message["content"]
    assert "- role: full stack developer" in system_message["content"]


def test_build_system_message_handles_no_memories(monkeypatch):
    monkeypatch.setattr(agent_core, "get_all_memories", lambda user_id: {})
    monkeypatch.setattr(agent_core, "retrieve", lambda query: [])

    system_message = agent_core._build_system_message([{"role": "user", "content": "hi"}], "u1")

    assert "(nothing remembered yet)" in system_message["content"]


def test_stream_turn_plain_text_response(monkeypatch):
    chunks = [make_chunk(content="Hello"), make_chunk(content=" world")]
    monkeypatch.setattr(agent_core, "call_model_stream", lambda messages: iter(chunks))

    messages = [{"role": "user", "content": "hi"}]
    events = list(agent_core.stream_turn(messages, "u1"))

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
    events = list(agent_core.stream_turn(messages, "u1"))

    assert events[0] == {"event": "tool_call", "tool": "calculator"}
    assert events[-1] == {"event": "token", "text": "The answer is 5."}

    tool_messages = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["content"] == "5"


def test_stream_turn_runs_multiple_tool_calls_in_parallel(monkeypatch):
    first_call_chunks = [
        make_chunk(
            tool_calls=[
                make_tool_call_delta(0, call_id="call1", name="calculator", arguments='{"operation":"add","a":1,"b":1}'),
                make_tool_call_delta(1, call_id="call2", name="get_weather", arguments='{"city":"Paris"}'),
            ]
        ),
    ]
    second_call_chunks = [make_chunk(content="done")]
    responses = [iter(first_call_chunks), iter(second_call_chunks)]
    monkeypatch.setattr(agent_core, "call_model_stream", lambda messages: responses.pop(0))

    intervals = {}

    def make_slow_tool(name, delay):
        def fn(**kwargs):
            start = time.monotonic()
            time.sleep(delay)
            intervals[name] = (start, time.monotonic())
            return "ok"

        return fn

    monkeypatch.setitem(agent_core.TOOL_FUNCTIONS, "calculator", make_slow_tool("calculator", 0.2))
    monkeypatch.setitem(agent_core.TOOL_FUNCTIONS, "get_weather", make_slow_tool("get_weather", 0.2))

    messages = [{"role": "user", "content": "do two things"}]
    list(agent_core.stream_turn(messages, "u1"))

    (start1, end1), (start2, end2) = intervals["calculator"], intervals["get_weather"]
    # If the two tool calls ran one after another, one interval would only
    # start once the other had already finished. Overlapping start/end
    # windows is the signal that they actually ran concurrently.
    assert max(start1, start2) < min(end1, end2)


def test_run_tool_catches_unexpected_exceptions(monkeypatch):
    def broken(**kwargs):
        raise TypeError("boom")

    monkeypatch.setitem(agent_core.TOOL_FUNCTIONS, "calculator", broken)

    result = agent_core._run_tool("calculator", '{"operation":"add","a":1,"b":1}', "u1")

    assert result == "Error: boom"


def test_build_system_message_includes_todays_date(monkeypatch):
    monkeypatch.setattr(agent_core, "get_all_memories", lambda user_id: {})
    monkeypatch.setattr(agent_core, "retrieve", lambda query: [])

    system_message = agent_core._build_system_message([{"role": "user", "content": "hi"}], "u1")

    assert date.today().isoformat() in system_message["content"]


def test_build_system_message_omits_docs_block_when_nothing_relevant(monkeypatch):
    monkeypatch.setattr(agent_core, "get_all_memories", lambda user_id: {})
    monkeypatch.setattr(agent_core, "retrieve", lambda query: [])

    system_message = agent_core._build_system_message([{"role": "user", "content": "hi"}], "u1")

    assert "Relevant company documents" not in system_message["content"]


def test_build_system_message_includes_docs_block_when_relevant(monkeypatch):
    monkeypatch.setattr(agent_core, "get_all_memories", lambda user_id: {})
    monkeypatch.setattr(agent_core, "retrieve", lambda query: ["Some policy text"])

    system_message = agent_core._build_system_message([{"role": "user", "content": "hi"}], "u1")

    assert "Relevant company documents" in system_message["content"]
    assert "Some policy text" in system_message["content"]


def test_stream_turn_detects_leaked_pseudo_tool_call(monkeypatch):
    # Llama/Groq occasionally emits a malformed function-call attempt as
    # plain text instead of a structured tool_calls delta.
    chunks = [
        make_chunk(content='<function.run_sql_query{"query": "SELECT 1"}'),
        make_chunk(content="</function>"),
    ]
    monkeypatch.setattr(agent_core, "call_model_stream", lambda messages: iter(chunks))

    messages = [{"role": "user", "content": "list products"}]
    events = list(agent_core.stream_turn(messages, "u1"))

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
    events = list(agent_core.stream_turn(messages, "u1"))

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
    events = list(agent_core.stream_turn(messages, "u1"))

    # The raw exception text isn't shown to the user (it can contain API
    # error internals) — only a generic, friendly message is.
    assert events == [
        {"event": "error", "message": "Sorry, something went wrong generating a response. Please try again."}
    ]


def test_run_turn_wraps_stream_turn(monkeypatch):
    chunks = [make_chunk(content="plain answer")]
    monkeypatch.setattr(agent_core, "call_model_stream", lambda messages: iter(chunks))

    messages = [{"role": "user", "content": "hi"}]
    reply = agent_core.run_turn(messages, "u1")

    assert reply == "plain answer"


def make_rate_limit_error(message="rate limited"):
    response = httpx.Response(429, request=httpx.Request("POST", "https://api.groq.com/x"))
    return RateLimitError(message, response=response, body=None)


def test_stream_turn_gives_friendly_message_on_rate_limit(monkeypatch):
    def always_rate_limited(api_messages):
        raise make_rate_limit_error()

    monkeypatch.setattr(agent_core, "call_model_stream", always_rate_limited)

    messages = [{"role": "user", "content": "hi"}]
    events = list(agent_core.stream_turn(messages, "u1"))

    assert events == [
        {
            "event": "token",
            "text": "I'm getting a lot of requests right now and hit the free plan's rate limit. Please wait about a minute and try again.",
        }
    ]
    assert messages[-1]["role"] == "assistant"


def test_stream_turn_gives_friendly_message_on_mid_stream_rate_limit(monkeypatch):
    def bad_gen():
        raise make_rate_limit_error()
        yield  # pragma: no cover

    monkeypatch.setattr(agent_core, "call_model_stream", lambda api_messages: bad_gen())

    messages = [{"role": "user", "content": "hi"}]
    events = list(agent_core.stream_turn(messages, "u1"))

    assert events == [
        {
            "event": "error",
            "message": "I'm getting a lot of requests right now and hit the free plan's rate limit. Please wait about a minute and try again.",
        }
    ]


def test_trim_history_leaves_short_history_untouched():
    messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    assert agent_core._trim_history(messages) == messages


def test_trim_history_caps_long_history_at_a_user_turn_boundary():
    # Build more than MAX_HISTORY_MESSAGES turns; each turn is a user message
    # followed by an assistant reply.
    messages = []
    for i in range(15):
        messages.append({"role": "user", "content": f"question {i}"})
        messages.append({"role": "assistant", "content": f"answer {i}"})

    trimmed = agent_core._trim_history(messages)

    assert len(trimmed) <= agent_core.MAX_HISTORY_MESSAGES
    assert trimmed[0]["role"] == "user"
    assert trimmed == messages[-len(trimmed):]


def test_trim_history_does_not_split_a_tool_call_sequence():
    # A user turn whose assistant reply used a tool spans 3 messages
    # (user, assistant-with-tool_calls, tool) before the final assistant
    # answer — trimming must never start mid-sequence.
    messages = []
    for i in range(8):
        messages.append({"role": "user", "content": f"question {i}"})
        messages.append({"role": "assistant", "content": None, "tool_calls": [{"id": f"c{i}"}]})
        messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": "result"})
        messages.append({"role": "assistant", "content": f"answer {i}"})

    trimmed = agent_core._trim_history(messages)

    assert trimmed[0]["role"] == "user"
