import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date

from dotenv import load_dotenv
from openai import BadRequestError, OpenAI, RateLimitError

from rag import retrieve
from tools import (
    browse_webpage,
    calculator,
    check_order_status,
    create_support_ticket,
    forget_about_me,
    get_all_memories,
    get_weather,
    read_pdf,
    remember_about_me,
    run_sql_query,
    web_search,
    web_search_with_sources,
)

load_dotenv()

logger = logging.getLogger("acme_support_agent")

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
)

MODEL = "llama-3.3-70b-versatile"

tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform a basic arithmetic operation (add, subtract, multiply, divide) between two numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                    },
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["operation", "a", "b"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                },
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_sql_query",
            "description": (
                "Run a read-only SQL SELECT query against the 'products' table "
                "(columns: id, name, price, stock) to answer questions about inventory."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_order_status",
            "description": (
                "Look up the status of a customer's order by order ID and the "
                "email address it was placed under. Both must match."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "integer"},
                    "email": {"type": "string"},
                },
                "required": ["order_id", "email"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_support_ticket",
            "description": (
                "File a support ticket for an issue the agent can't resolve directly "
                "(e.g. a complaint, a return request, a bug report). Only use this "
                "after collecting the customer's name, email, and a clear description "
                "of the issue."
            ),
            "parameters": {
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
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember_about_me",
            "description": (
                "Save a durable fact about the person you're talking to, so "
                "you remember it in every future conversation with them, "
                "not just this one — e.g. their name, role, company, or "
                "preferences. Use a short lowercase key like 'name' or "
                "'role'. Calling this again with the same key overwrites "
                "the old value, so use it to correct facts too."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["key", "value"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forget_about_me",
            "description": "Remove a previously remembered fact about the user by its key.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                },
                "required": ["key"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current/factual information not reliably known "
                "from training data (news, prices, weather, recent releases, etc). "
                "Always pass a clear, effective English query capturing the user's "
                "intent, even if they asked in another language — translate/rewrite "
                "it first, don't pass their raw text through unless it's already "
                "a good English query."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_pdf",
            "description": "Extract and return the text content of a PDF file given its local file path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browse_webpage",
            "description": "Open a URL in a real browser and return the visible page text (works on JS-rendered pages).",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                },
                "required": ["url"],
                "additionalProperties": False,
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "calculator": calculator,
    "get_weather": get_weather,
    "run_sql_query": run_sql_query,
    "web_search": web_search,
    "read_pdf": read_pdf,
    "browse_webpage": browse_webpage,
    "check_order_status": check_order_status,
    "create_support_ticket": create_support_ticket,
    "remember_about_me": remember_about_me,
    "forget_about_me": forget_about_me,
}


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
    """Returns (content_for_model, sources). sources is only ever non-None
    for web_search, letting the UI show real citations instead of parsing
    them back out of the prose the model reads."""
    args = json.loads(arguments_json) if arguments_json else {}
    if name in ("remember_about_me", "forget_about_me"):
        args["user_id"] = user_id
    try:
        if name == "web_search":
            content, sources = web_search_with_sources(**args)
            return content, sources
        function = TOOL_FUNCTIONS[name]
        return str(function(**args)), None
    except Exception as e:
        # Tools raise ValueError for expected, already-friendly failures
        # (bad input, search/browse unavailable, etc). Anything else is an
        # unexpected bug — catch it too so one tool failing can't take down
        # the whole turn; either way the model sees "Error: ..." as its tool
        # result and can react gracefully instead of the request just dying.
        logger.warning("tool %s failed: %s", name, e)
        return f"Error: {e}", None


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


def _build_system_message(messages: list, user_id: str) -> dict:
    last_user_message = messages[-1]["content"]

    t0 = time.monotonic()
    # retrieve() already filters out low-relevance chunks (see rag.py), so an
    # empty result here means nothing in the company docs is actually about
    # this question — omit the section entirely rather than handing the
    # model a block of unrelated policy text to "ignore."
    context_docs = retrieve(last_user_message)
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

    today = date.today().isoformat()

    return {
        "role": "system",
        "content": (
            "You are a helpful customer support agent for Acme Corp. "
            f"Today's date is {today} — use it to judge what's current and "
            "to answer date-relative questions correctly.\n\n"
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
            "(e.g. the site name) so the user knows where it came from.\n\n"
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
            "Whenever they share something durable about themselves (name, "
            "role, company, preferences, ongoing projects, etc.), call "
            "remember_about_me to save it immediately, without being asked. "
            "Use what's already remembered naturally instead of asking them "
            "to repeat it." + context_block
        ),
    }


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
    system_message = _build_system_message(messages, user_id)
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
                (acc["id"], executor.submit(_run_tool, acc["name"], acc["arguments"], user_id))
                for acc in tool_calls_acc.values()
            ]
            results = [(tool_call_id, future.result()) for tool_call_id, future in futures]
        logger.info(
            "[TOOLS] %.2fs (%d call%s)", time.monotonic() - tools_start, len(results), "" if len(results) == 1 else "s"
        )

        for tool_call_id, (output, sources) in results:
            if sources:
                yield {"event": "sources", "sources": sources}
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
