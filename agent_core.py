import json
import os

from dotenv import load_dotenv
from openai import BadRequestError, OpenAI

from rag import retrieve
from tools import (
    browse_webpage,
    calculator,
    get_weather,
    read_pdf,
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
}


MAX_RETRIES = 3


def call_model(messages: list):
    last_error = None
    for _ in range(MAX_RETRIES):
        try:
            return client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tools,
            )
        except BadRequestError as e:
            last_error = e
    raise last_error


def run_turn(messages: list) -> str:
    last_user_message = messages[-1]["content"]
    context_docs = retrieve(last_user_message)
    context_text = "\n\n---\n\n".join(context_docs)
    system_message = {
        "role": "system",
        "content": (
            "You are a helpful customer support agent for Acme Corp. Use the "
            "following company policy documents and the available tools to "
            "answer customer questions accurately. If the documents aren't "
            "relevant to the question, ignore them.\n\n" + context_text
        ),
    }

    while True:
        api_messages = [system_message] + messages
        try:
            response = call_model(api_messages)
        except BadRequestError:
            fallback = "Sorry, I couldn't process that. Could you rephrase, or be more specific (e.g. name a city)?"
            messages.append({"role": "assistant", "content": fallback})
            return fallback

        message = response.choices[0].message
        messages.append(message)

        if not message.tool_calls:
            return message.content

        for tool_call in message.tool_calls:
            args = json.loads(tool_call.function.arguments)
            function = TOOL_FUNCTIONS[tool_call.function.name]
            try:
                output = str(function(**args))
            except ValueError as e:
                output = f"Error: {e}"

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": output,
            })
