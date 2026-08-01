import json
import os
import re

from dotenv import load_dotenv
from openai import BadRequestError, OpenAI

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
)

load_dotenv()

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
            "description": "Search the web for current information not known from training data.",
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


def _build_system_message(messages: list) -> dict:
    last_user_message = messages[-1]["content"]
    context_docs = retrieve(last_user_message)
    context_text = "\n\n---\n\n".join(context_docs)

    memories = get_all_memories()
    memory_text = (
        "\n".join(f"- {key}: {value}" for key, value in memories.items())
        if memories
        else "(nothing remembered yet)"
    )

    return {
        "role": "system",
        "content": (
            "You are a helpful customer support agent for Acme Corp. Use the "
            "following company policy documents and the available tools to "
            "answer customer questions accurately. If the documents aren't "
            "relevant to the question, ignore them.\n\n"
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
            "to repeat it.\n\n" + context_text
        ),
    }


def stream_turn(messages: list):
    """Runs one user turn against the model, yielding events as they
    happen so a caller can show live progress instead of waiting for the
    whole reply:
      {"event": "tool_call", "tool": name}
      {"event": "token", "text": delta}
      {"event": "error", "message": str}
    Appends the resulting assistant/tool messages to `messages` in place,
    same as the old non-streaming run_turn did.
    """
    system_message = _build_system_message(messages)
    # Counts consecutive stream failures that happened before anything was
    # shown to the user, so they can be retried transparently (distinct
    # from the outer loop's normal tool-calling round trips, which reset
    # this back to 0).
    retry_count = 0

    while True:
        api_messages = [system_message] + messages
        try:
            stream = call_model_stream(api_messages)
        except BadRequestError:
            fallback = "Sorry, I couldn't process that. Could you rephrase, or be more specific (e.g. name a city)?"
            messages.append({"role": "assistant", "content": fallback})
            yield {"event": "token", "text": fallback}
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
            yield {"event": "error", "message": str(e)}
            return

        retry_count = 0

        if leaked_tool_call and not tool_calls_acc:
            fallback = "Sorry, I wasn't able to complete that request. Could you try rephrasing it?"
            messages.append({"role": "assistant", "content": fallback})
            yield {"event": "content_replace", "text": fallback}
            return

        if not tool_calls_acc:
            messages.append({"role": "assistant", "content": content})
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
            args = json.loads(acc["arguments"]) if acc["arguments"] else {}
            function = TOOL_FUNCTIONS[acc["name"]]
            try:
                output = str(function(**args))
            except ValueError as e:
                output = f"Error: {e}"

            messages.append(
                {"role": "tool", "tool_call_id": acc["id"], "content": output}
            )
        # loop back for the next model call now that tool results are in


def run_turn(messages: list) -> str:
    """Non-streaming convenience wrapper over stream_turn, for the CLI."""
    text = ""
    for event in stream_turn(messages):
        if event["event"] == "token":
            text += event["text"]
        elif event["event"] == "content_replace":
            text = event["text"]
        elif event["event"] == "error":
            raise RuntimeError(event["message"])
    return text
