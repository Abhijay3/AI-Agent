import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date

from dotenv import load_dotenv
from openai import BadRequestError, OpenAI, RateLimitError

from rag import retrieve_with_sources
from tool_registry import Tool, all_tools, get_tool, openai_tool_schemas, register_tool
from tools import (
    browse_webpage,
    calculator,
    check_order_status,
    create_support_ticket,
    forget_about_me,
    get_all_memories,
    get_current_time,
    get_weather,
    open_url,
    read_pdf,
    remember_about_me,
    run_sql_query,
    web_search_with_sources,
)

load_dotenv()

logger = logging.getLogger("acme_support_agent")

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
)

MODEL = "llama-3.3-70b-versatile"

# Every tool is declared once, here, with everything the rest of the app
# needs to know about it: name, description, parameter schema, execution
# function, risk level, and whether it requires confirmation before running.
# agent_core's tool-calling loop (below) reads all of this through
# tool_registry rather than hardcoding per-tool behavior, so a future tool
# module (macOS tools, file tools, ...) can add tools by calling
# register_tool() from wherever it's defined — no edits needed here.
#
# Nothing below is above "safe"/"low" risk today, since nothing destructive
# or irreversible exists in this app yet. requires_confirmation is the
# extension point for when that changes (e.g. deleting a file, shutting
# down a machine): mark it True and the confirmation UX (not yet built,
# since there's nothing to protect yet) plugs in without touching the loop.
register_tool(Tool(
    name="calculator",
    description="Perform a basic arithmetic operation (add, subtract, multiply, divide) between two numbers.",
    parameters={
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]},
            "a": {"type": "number"},
            "b": {"type": "number"},
        },
        "required": ["operation", "a", "b"],
        "additionalProperties": False,
    },
    handler=calculator,
))
register_tool(Tool(
    name="get_weather",
    description="Get the current weather for a given city.",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
        "additionalProperties": False,
    },
    handler=get_weather,
))
register_tool(Tool(
    name="run_sql_query",
    description=(
        "Run a read-only SQL SELECT query against the 'products' table "
        "(columns: id, name, price, stock) to answer questions about inventory."
    ),
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
    },
    handler=run_sql_query,
))
register_tool(Tool(
    name="check_order_status",
    description=(
        "Look up the status of a customer's order by order ID and the "
        "email address it was placed under. Both must match."
    ),
    parameters={
        "type": "object",
        "properties": {
            "order_id": {"type": "integer"},
            "email": {"type": "string"},
        },
        "required": ["order_id", "email"],
        "additionalProperties": False,
    },
    handler=check_order_status,
))
register_tool(Tool(
    name="create_support_ticket",
    description=(
        "File a support ticket for an issue the agent can't resolve directly "
        "(e.g. a complaint, a return request, a bug report). Only use this "
        "after collecting the customer's name, email, and a clear description "
        "of the issue."
    ),
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "email": {"type": "string"},
            "subject": {"type": "string"},
            "description": {"type": "string"},
        },
        "required": ["name", "email", "subject", "description"],
        "additionalProperties": False,
    },
    handler=create_support_ticket,
    risk_level="low",
))
register_tool(Tool(
    name="remember_about_me",
    description=(
        "Save a durable fact about the person you're talking to, so "
        "you remember it in every future conversation with them, "
        "not just this one — e.g. their name, role, company, or "
        "preferences. Use a short lowercase key like 'name' or "
        "'role'. Calling this again with the same key overwrites "
        "the old value, so use it to correct facts too."
    ),
    parameters={
        "type": "object",
        "properties": {
            "key": {"type": "string"},
            "value": {"type": "string"},
            "category": {
                "type": "string",
                "enum": ["name", "preference", "project", "other"],
                "description": "How this shows up grouped in the memory panel. Use 'name' only for their actual name.",
            },
        },
        "required": ["key", "value"],
        "additionalProperties": False,
    },
    handler=remember_about_me,
    needs_user_id=True,
))
register_tool(Tool(
    name="forget_about_me",
    description="Remove a previously remembered fact about the user by its key.",
    parameters={
        "type": "object",
        "properties": {"key": {"type": "string"}},
        "required": ["key"],
        "additionalProperties": False,
    },
    handler=forget_about_me,
    risk_level="low",
    needs_user_id=True,
))
register_tool(Tool(
    name="web_search",
    description=(
        "Search the web for current/factual information not reliably known "
        "from training data (news, prices, weather, recent releases, etc). "
        "Always pass a clear, effective English query capturing the user's "
        "intent, even if they asked in another language — translate/rewrite "
        "it first, don't pass their raw text through unless it's already "
        "a good English query."
    ),
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
    },
    handler=web_search_with_sources,
    returns_sources=True,
))
register_tool(Tool(
    name="read_pdf",
    description="Extract and return the text content of a PDF file given its local file path.",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
        "additionalProperties": False,
    },
    handler=read_pdf,
))
register_tool(Tool(
    name="browse_webpage",
    description="Open a URL in a real browser and return the visible page text (works on JS-rendered pages).",
    parameters={
        "type": "object",
        "properties": {"url": {"type": "string"}},
        "required": ["url"],
        "additionalProperties": False,
    },
    handler=browse_webpage,
))
register_tool(Tool(
    name="get_current_time",
    description="Get the current date and time, optionally in a specific IANA timezone (e.g. 'Asia/Kolkata', 'America/New_York'). Defaults to UTC.",
    parameters={
        "type": "object",
        "properties": {"timezone": {"type": "string"}},
        "required": [],
        "additionalProperties": False,
    },
    handler=get_current_time,
))
register_tool(Tool(
    name="open_url",
    description=(
        "Open a URL for the user in a new browser tab (e.g. a page found via "
        "web_search, or a well-known site they asked for like 'open YouTube'). "
        "This opens it in their current browser tab, not a separate application — "
        "say so naturally if that distinction matters to what they asked."
    ),
    parameters={
        "type": "object",
        "properties": {"url": {"type": "string"}},
        "required": ["url"],
        "additionalProperties": False,
    },
    handler=open_url,
    client_action=lambda args: {"action": "open_url", "url": args["url"]},
))

tools = openai_tool_schemas()
TOOL_FUNCTIONS = {t.name: t.handler for t in all_tools()}
TOOL_RISK = {t.name: {"risk_level": t.risk_level, "requires_confirmation": t.requires_confirmation} for t in all_tools()}


MAX_RETRIES = 3

# Groq's free tier caps tokens-per-minute per model. Resending the *entire*
# conversation on every turn (as we used to) makes token usage grow with
# session length, so long chats would hit that cap more and more often. Cap
# how much history is replayed to the model — the full history still stays
# in Redis/the UI, this only bounds what gets sent as context.
MAX_HISTORY_MESSAGES = 20

# A model could in principle request many tool calls in a single round.
# Running them all concurrently with no cap would spin up a thread (and,
# for browse_webpage, potentially a whole browser) per call at once —
# capping the pool bounds worst-case resource use per request regardless
# of how many tool calls one round asks for.
MAX_PARALLEL_TOOL_CALLS = 4


def _trim_history(messages: list) -> list:
    if len(messages) <= MAX_HISTORY_MESSAGES:
        return messages
    trimmed = messages[-MAX_HISTORY_MESSAGES:]
    # A "user" message always starts a fresh turn, so cutting there can't
    # split an assistant/tool_calls sequence from its tool results — Groq
    # rejects a tool message whose triggering assistant message is missing.
    for i, m in enumerate(trimmed):
        if m.get("role") == "user":
            return trimmed[i:]
    return trimmed


def _run_tool(name: str, arguments_json: str, user_id: str) -> tuple:
    """Returns (content_for_model, sources, client_action).

    Which of these apply for a given tool comes from its registration in
    tool_registry (needs_user_id, returns_sources, client_action) rather
    than being hardcoded per tool name here — see Tool for what each flag
    means. sources is only ever non-None for tools with returns_sources=True
    (currently just web_search), letting the UI show real citations instead
    of parsing them back out of the prose the model reads. client_action
    signals something the frontend must do itself — the backend is a cloud
    container, it cannot open a tab in the user's actual browser, only ask
    it to; open_url is the first tool that needs this, and the same
    extension point a future "run this on the desktop agent" action would
    use once one exists.

    The actual callable is still looked up through TOOL_FUNCTIONS (a plain,
    mutable dict derived from the registry at import time) rather than the
    Tool object's own handler reference, so tests can substitute a fake
    implementation via monkeypatch.setitem without re-registering a tool.
    """
    args = json.loads(arguments_json) if arguments_json else {}
    tool = get_tool(name)
    if tool.needs_user_id:
        args["user_id"] = user_id
    try:
        function = TOOL_FUNCTIONS[name]
        if tool.returns_sources:
            content, sources = function(**args)
            return content, sources, None
        content = str(function(**args))
        client_action = tool.client_action(args) if tool.client_action else None
        return content, None, client_action
    except Exception as e:
        # Tools raise ValueError for expected, already-friendly failures
        # (bad input, search/browse unavailable, etc). Anything else is an
        # unexpected bug — catch it too so one tool failing can't take down
        # the whole turn; either way the model sees "Error: ..." as its tool
        # result and can react gracefully instead of the request just dying.
        logger.warning("tool %s failed: %s", name, e)
        return f"Error: {e}", None, None


def call_model_stream(messages: list):
    last_error = None
    for _ in range(MAX_RETRIES):
        try:
            return client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tools,
                stream=True,
            )
        except BadRequestError as e:
            last_error = e
    raise last_error


def _build_system_message(messages: list, user_id: str) -> tuple:
    """Returns (system_message, rag_sources) — rag_sources lets the caller
    show which company document(s) actually informed the answer, the same
    way web_search results already get cited."""
    last_user_message = messages[-1]["content"]

    t0 = time.monotonic()
    # retrieve_with_sources() already filters out low-relevance chunks (see
    # rag.py), so an empty result here means nothing in the company docs is
    # actually about this question — omit the section entirely rather than
    # handing the model a block of unrelated policy text to "ignore."
    context_docs, rag_sources = retrieve_with_sources(last_user_message)
    logger.info("[RAG] %.2fs (%d chunk%s)", time.monotonic() - t0, len(context_docs), "" if len(context_docs) == 1 else "s")
    context_block = (
        "\n\nRelevant company documents (these were retrieved because they "
        "matched the question — answer using them, don't second-guess "
        "whether they're relevant):\n\n" + "\n\n---\n\n".join(context_docs)
        if context_docs
        else ""
    )

    memories = get_all_memories(user_id)
    memory_text = (
        "\n".join(f"- {key}: {value}" for key, value in memories.items())
        if memories
        else "(nothing remembered yet)"
    )
    remembered_name = memories.get("name")

    today = date.today().isoformat()

    system_message = {
        "role": "system",
        "content": (
            "You are Abhijay's AI — a personal AI assistant, also answering "
            "to the nickname 'Jarvis' if someone greets you with it (e.g. "
            "'Hi Jarvis') — respond naturally to either, don't insist on "
            "being called by your full name. Your tone is calm, confident, "
            "concise, and a little futuristic: intelligent and professional, "
            "not chatty or overly enthusiastic. Keep replies tight — a "
            "sentence or two for simple things, more only when the question "
            "actually needs it. Don't perform helpfulness with filler "
            "('Great question!', 'I'd be happy to help!') — just help.\n"
            + (
                f"The person you're talking to is named {remembered_name} — address "
                "them by name naturally sometimes, and occasionally (not every "
                "message, that gets tiresome) as 'boss' for warmth.\n\n"
                if remembered_name
                else "You don't know this person's name yet — ask naturally if it "
                "comes up, and once you learn it, save it with remember_about_me "
                "so you address them by name in future replies. Until then, "
                "'boss' works fine as an occasional warm address.\n\n"
            )
            + f"Today's date is {today} — use it to judge what's current and "
            "to answer date-relative questions correctly. Use get_current_time "
            "for the actual current time or a specific timezone rather than "
            "guessing.\n\n"
            "How to decide whether to search the web (web_search tool):\n"
            "- For current events, news, prices, weather, sports results, "
            "stocks, or anything about a specific person/product/event that "
            "could have changed since your training — ALWAYS use web_search "
            "first. Don't answer these from memory even if you think you "
            "know the answer; your training data has a cutoff and can be "
            "stale.\n"
            "- For general knowledge, definitions, explanations, or coding "
            "help (e.g. 'What is Python?', 'Explain recursion', 'write a "
            "React component') that doesn't change over time — answer "
            "directly, immediately, without searching. Searching for these "
            "only adds delay for no benefit.\n"
            "- For this customer's own orders use check_order_status; for "
            "catalog inventory/pricing use run_sql_query. Never web-search "
            "for data that lives in our own database.\n\n"
            "When you do search the web:\n"
            "- Always search with a clear, effective English query capturing "
            "the intent, even if the user asked in Hinglish or another "
            "language (e.g. for 'bhai latest iphone ka price kya hai India "
            "me', search 'latest iPhone price in India', not the literal "
            "text).\n"
            "- One good search is usually enough — don't repeat near-"
            "identical searches. Only reach for browse_webpage if the "
            "search snippets genuinely don't have enough detail; it's much "
            "slower, so don't use it by default.\n"
            "- Only state facts the results actually support — never invent "
            "numbers, dates, or details that aren't in what you retrieved. "
            "If sources disagree, say so instead of picking one silently.\n"
            "- When your answer relies on a search, briefly name the source "
            "(e.g. the site name) so the user knows where it came from.\n"
            "- If they ask you to open a page (search results, a site they "
            "name), use open_url — it opens a new tab in their browser. You "
            "cannot launch a native application (e.g. the actual Chrome.app "
            "or VS Code) — that requires a local desktop agent this app "
            "doesn't have yet. If asked to do that, say so plainly instead "
            "of pretending to.\n\n"
            "Language: reply in whatever language/style the user wrote in. "
            "If they wrote in Hinglish, reply naturally in Hinglish — don't "
            "switch to formal Hindi or pure English. If they wrote in "
            "English, reply in English.\n\n"
            "You can look up real order status with check_order_status, and "
            "file a real support ticket with create_support_ticket. "
            "create_support_ticket only needs a name, email, subject, and "
            "description — nothing else. Once you have those four things, "
            "file the ticket immediately; don't ask for an order ID or any "
            "other detail it doesn't require.\n\n"
            "You have persistent memory about the person you're talking to, "
            "shared across every conversation with them, not just this one. "
            "What you currently remember about them:\n" + memory_text + "\n\n"
            "Whenever they share something durable and ordinary about "
            "themselves (name, role, company, preferences, ongoing "
            "projects, etc.), call remember_about_me to save it "
            "immediately, without being asked — tag it with the closest "
            "category ('name', 'preference', 'project', or 'other'). Use "
            "what's already remembered naturally instead of asking them to "
            "repeat it. Do NOT proactively save sensitive personal details "
            "(health, finances, relationships, religion, politics, and "
            "similar) even if mentioned in passing — only remember those if "
            "the person explicitly asks you to remember them." + context_block
        ),
    }
    return system_message, rag_sources


def stream_turn(messages: list, user_id: str):
    """Runs one user turn against the model, yielding events as they
    happen so a caller can show live progress instead of waiting for the
    whole reply:
      {"event": "tool_call", "tool": name}
      {"event": "token", "text": delta}
      {"event": "error", "message": str}
    Appends the resulting assistant/tool messages to `messages` in place,
    same as the old non-streaming run_turn did.

    `user_id` scopes remember_about_me/forget_about_me/get_all_memories so
    one person's remembered facts never leak into another person's chats —
    it is injected server-side into those two tool calls below, never
    exposed to the model as a callable parameter, so the model can't be
    steered into reading or writing another user's memory.
    """
    turn_start = time.monotonic()
    system_message, rag_sources = _build_system_message(messages, user_id)
    if rag_sources:
        yield {"event": "sources", "sources": rag_sources}
    # Counts consecutive stream failures that happened before anything was
    # shown to the user, so they can be retried transparently (distinct
    # from the outer loop's normal tool-calling round trips, which reset
    # this back to 0).
    retry_count = 0

    while True:
        api_messages = [system_message] + _trim_history(messages)
        llm_start = time.monotonic()
        try:
            stream = call_model_stream(api_messages)
        except BadRequestError:
            fallback = "Sorry, I couldn't process that. Could you rephrase, or be more specific (e.g. name a city)?"
            messages.append({"role": "assistant", "content": fallback})
            yield {"event": "token", "text": fallback}
            logger.info("[TOTAL] %.2fs", time.monotonic() - turn_start)
            return
        except RateLimitError:
            # Groq's free tier has a strict per-minute token budget shared
            # across all requests. The raw error body names the limit and
            # token counts, which reads like garbled text to a customer —
            # give them something actionable instead.
            logger.warning("Groq rate limit hit")
            fallback = "I'm getting a lot of requests right now and hit the free plan's rate limit. Please wait about a minute and try again."
            messages.append({"role": "assistant", "content": fallback})
            yield {"event": "token", "text": fallback}
            logger.info("[TOTAL] %.2fs", time.monotonic() - turn_start)
            return

        content = ""
        tool_calls_acc = {}  # index -> {"id": str, "name": str, "arguments": str}
        # Llama 3.3 70B on Groq occasionally leaks a malformed pseudo tool
        # call as plain text content (e.g. "<function.run_sql_query{...}")
        # instead of using the structured tool_calls delta, especially when
        # a prompt combines tool use with formatting requests. Detect it on
        # the first chunk and suppress streaming it verbatim to the user.
        leaked_tool_call = False
        first_chunk_seen = False

        try:
            for chunk in stream:
                delta = chunk.choices[0].delta

                if delta.content:
                    content += delta.content
                    if not first_chunk_seen:
                        first_chunk_seen = True
                        if re.match(r"^\s*<function[.\s]", content, re.IGNORECASE):
                            leaked_tool_call = True
                    if not leaked_tool_call:
                        yield {"event": "token", "text": delta.content}

                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        acc = tool_calls_acc.setdefault(
                            tc.index, {"id": None, "name": None, "arguments": ""}
                        )
                        if tc.id:
                            acc["id"] = tc.id
                        if tc.function and tc.function.name:
                            acc["name"] = tc.function.name
                        if tc.function and tc.function.arguments:
                            acc["arguments"] += tc.function.arguments
        except Exception as e:
            # Groq sometimes rejects a malformed function-call generation
            # entirely (a different failure mode than the text-leak above).
            # If nothing was shown to the user yet, it's safe to quietly
            # retry rather than surfacing a scary error for a transient
            # generation hiccup.
            if not content and not tool_calls_acc and retry_count < MAX_RETRIES - 1:
                retry_count += 1
                continue
            # str(e) on an API error is a raw error body (status codes, limit
            # numbers, etc.) — log it for debugging but never show it as-is,
            # it reads like garbled text in the chat.
            logger.warning("stream_turn mid-stream failure: %s", e)
            message = (
                "I'm getting a lot of requests right now and hit the free plan's rate limit. Please wait about a minute and try again."
                if isinstance(e, RateLimitError)
                else "Sorry, something went wrong generating a response. Please try again."
            )
            yield {"event": "error", "message": message}
            logger.info("[TOTAL] %.2fs", time.monotonic() - turn_start)
            return

        logger.info("[LLM] %.2fs", time.monotonic() - llm_start)
        retry_count = 0

        if leaked_tool_call and not tool_calls_acc:
            fallback = "Sorry, I wasn't able to complete that request. Could you try rephrasing it?"
            messages.append({"role": "assistant", "content": fallback})
            yield {"event": "content_replace", "text": fallback}
            logger.info("[TOTAL] %.2fs", time.monotonic() - turn_start)
            return

        if not tool_calls_acc:
            messages.append({"role": "assistant", "content": content})
            logger.info("[TOTAL] %.2fs", time.monotonic() - turn_start)
            return

        tool_calls_list = [
            {
                "id": acc["id"],
                "type": "function",
                "function": {"name": acc["name"], "arguments": acc["arguments"]},
            }
            for acc in tool_calls_acc.values()
        ]
        messages.append(
            {"role": "assistant", "content": content or None, "tool_calls": tool_calls_list}
        )

        for acc in tool_calls_acc.values():
            yield {"event": "tool_call", "tool": acc["name"]}

        # Independent tool calls from the same round (e.g. two searches, or
        # a search + a weather lookup) used to run one at a time in a plain
        # for-loop, each waiting on the previous one's network I/O. Running
        # them concurrently means the round takes as long as the *slowest*
        # call instead of the sum of all of them.
        tools_start = time.monotonic()
        with ThreadPoolExecutor(max_workers=min(len(tool_calls_acc), MAX_PARALLEL_TOOL_CALLS)) as executor:
            futures = [
                (acc["id"], acc["name"], executor.submit(_run_tool, acc["name"], acc["arguments"], user_id))
                for acc in tool_calls_acc.values()
            ]
            results = [(tool_call_id, name, future.result()) for tool_call_id, name, future in futures]
        logger.info(
            "[TOOLS] %.2fs (%d call%s)", time.monotonic() - tools_start, len(results), "" if len(results) == 1 else "s"
        )

        for tool_call_id, name, (output, sources, client_action) in results:
            if sources:
                yield {"event": "sources", "sources": sources}
            if client_action:
                yield {"event": "client_action", **client_action}
            # Persistent, visible record of what actually ran (not just the
            # transient "Searching the web…"-style status text) — lets the
            # user see, after the fact, which tool was used and whether it
            # succeeded, the same way a real assistant would show its work.
            yield {"event": "tool_result", "tool": name, "ok": not output.startswith("Error:")}
            messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": output})
        # loop back for the next model call now that tool results are in


def run_turn(messages: list, user_id: str) -> str:
    """Non-streaming convenience wrapper over stream_turn, for the CLI."""
    text = ""
    for event in stream_turn(messages, user_id):
        if event["event"] == "token":
            text += event["text"]
        elif event["event"] == "content_replace":
            text = event["text"]
        elif event["event"] == "error":
            raise RuntimeError(event["message"])
    return text
