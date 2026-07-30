from mcp.server import MCPServer

from tools import calculator as _calculator
from tools import get_weather as _get_weather
from tools import run_sql_query as _run_sql_query

mcp = MCPServer("agent-tools")


@mcp.tool()
def calculator(operation: str, a: float, b: float) -> str:
    """Perform a basic arithmetic operation (add, subtract, multiply, divide) between two numbers."""
    try:
        return str(_calculator(operation, a, b))
    except ValueError as e:
        return f"Error: {e}"


@mcp.tool()
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    try:
        return _get_weather(city)
    except ValueError as e:
        return f"Error: {e}"


@mcp.tool()
def run_sql_query(query: str) -> str:
    """Run a read-only SQL SELECT query against the 'products' table (columns: id, name, price, stock)."""
    try:
        return _run_sql_query(query)
    except ValueError as e:
        return f"Error: {e}"


if __name__ == "__main__":
    mcp.run()
