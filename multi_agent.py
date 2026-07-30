import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from tools import run_sql_query

load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
)

MODEL = "llama-3.3-70b-versatile"


def ask(system_prompt: str, user_prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content


def planner_agent(task: str) -> str:
    return ask(
        "You are a planning agent. Break the user's task into a short numbered "
        "list of concrete steps. Be concise, 3-5 steps max.",
        task,
    )


def research_agent(task: str) -> str:
    sql_tool = {
        "type": "function",
        "function": {
            "name": "run_sql_query",
            "description": (
                "Run a read-only SQL SELECT query against the 'products' table "
                "(columns: id, name, price, stock)."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    }

    messages = [
        {
            "role": "system",
            "content": (
                "You are a research agent. Use the run_sql_query tool to find "
                "any facts needed for the task, then summarize what you found "
                "in one or two sentences."
            ),
        },
        {"role": "user", "content": task},
    ]

    response = client.chat.completions.create(
        model=MODEL, messages=messages, tools=[sql_tool]
    )
    message = response.choices[0].message
    messages.append(message)

    if message.tool_calls:
        for tool_call in message.tool_calls:
            args = json.loads(tool_call.function.arguments)
            try:
                output = run_sql_query(args["query"])
            except ValueError as e:
                output = f"Error: {e}"
            messages.append(
                {"role": "tool", "tool_call_id": tool_call.id, "content": output}
            )

        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=[sql_tool]
        )
        return response.choices[0].message.content

    return message.content


def coding_agent(task: str, research_findings: str) -> str:
    return ask(
        "You are a coding agent. Write clean, minimal Python code that "
        "accomplishes the task, using the given research findings as input "
        "data. Output ONLY a Python code block, no extra prose.",
        f"Task: {task}\n\nResearch findings: {research_findings}",
    )


def reviewer_agent(code: str) -> str:
    return ask(
        "You are a code reviewer. Check the given Python code for correctness "
        "and bugs. Reply with 'APPROVED' if it's correct, or a short list of "
        "specific issues if not.",
        code,
    )


def run_pipeline(task: str) -> None:
    print(f"\n=== TASK ===\n{task}")

    plan = planner_agent(task)
    print(f"\n=== PLAN (Planner Agent) ===\n{plan}")

    findings = research_agent(task)
    print(f"\n=== FINDINGS (Research Agent) ===\n{findings}")

    code = coding_agent(task, findings)
    print(f"\n=== CODE (Coding Agent) ===\n{code}")

    review = reviewer_agent(code)
    print(f"\n=== REVIEW (Reviewer Agent) ===\n{review}")


if __name__ == "__main__":
    run_pipeline(
        "Write a Python function that calculates the discounted price of the "
        "Laptop after a 15% discount, using its current price from the products database."
    )
