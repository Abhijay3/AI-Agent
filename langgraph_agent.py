import os
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from tools import calculator as _calculator
from tools import get_weather as _get_weather
from tools import run_sql_query as _run_sql_query

load_dotenv()


@tool
def calculator(operation: str, a: float, b: float) -> str:
    """Perform a basic arithmetic operation (add, subtract, multiply, divide) between two numbers."""
    try:
        return str(_calculator(operation, a, b))
    except ValueError as e:
        return f"Error: {e}"


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    try:
        return _get_weather(city)
    except ValueError as e:
        return f"Error: {e}"


@tool
def run_sql_query(query: str) -> str:
    """Run a read-only SQL SELECT query against the 'products' table (columns: id, name, price, stock)."""
    try:
        return _run_sql_query(query)
    except ValueError as e:
        return f"Error: {e}"


all_tools = [calculator, get_weather, run_sql_query]

llm = ChatOpenAI(
    model="llama-3.3-70b-versatile",
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
)
llm_with_tools = llm.bind_tools(all_tools)


class State(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(all_tools))
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")

graph = graph_builder.compile()


def main():
    state = {"messages": []}
    print("LangGraph chat. Type 'exit' to quit.")

    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == "exit":
            break

        state["messages"].append({"role": "user", "content": user_input})
        state = graph.invoke(state)
        print(f"AI: {state['messages'][-1].content}")


if __name__ == "__main__":
    main()
