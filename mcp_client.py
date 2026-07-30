import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

server_params = StdioServerParameters(
    command="python",
    args=["mcp_server.py"],
)


async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("Available tools:", [t.name for t in tools.tools])

            result = await session.call_tool(
                "calculator", {"operation": "multiply", "a": 12, "b": 7}
            )
            print("calculator(12 * 7) ->", result.content[0].text)

            result = await session.call_tool(
                "run_sql_query",
                {"query": "SELECT name, stock FROM products WHERE stock < 30"},
            )
            print("run_sql_query(stock < 30) ->", result.content[0].text)


if __name__ == "__main__":
    asyncio.run(main())
