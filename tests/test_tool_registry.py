import os

os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("TAVILY_API_KEY", "test-tavily-key")

import pytest

import agent_core  # noqa: E402 — importing this registers the built-in tools
from tool_registry import Tool, get_tool, register_tool  # noqa: E402


def test_get_tool_returns_registered_tool():
    tool = get_tool("calculator")
    assert tool.name == "calculator"
    assert tool.handler is agent_core.calculator
    assert tool.risk_level == "safe"
    assert tool.requires_confirmation is False


def test_get_tool_raises_for_unknown_name():
    with pytest.raises(KeyError):
        get_tool("does_not_exist")


def test_register_tool_rejects_duplicate_name():
    with pytest.raises(ValueError):
        register_tool(Tool(name="calculator", description="dupe", parameters={}, handler=lambda: None))


def test_openai_tool_schemas_shape_matches_tools_list():
    for schema in agent_core.tools:
        assert schema["type"] == "function"
        assert "name" in schema["function"]
        assert "description" in schema["function"]
        assert "parameters" in schema["function"]


def test_every_registered_tool_has_a_handler_in_tool_functions():
    for tool in [get_tool(name) for name in agent_core.TOOL_FUNCTIONS]:
        assert callable(agent_core.TOOL_FUNCTIONS[tool.name])


def test_needs_user_id_tools_are_the_memory_tools():
    memory_tools = {name for name in agent_core.TOOL_FUNCTIONS if get_tool(name).needs_user_id}
    assert memory_tools == {"remember_about_me", "forget_about_me"}


def test_open_url_is_the_only_tool_with_a_client_action():
    client_action_tools = {name for name in agent_core.TOOL_FUNCTIONS if get_tool(name).client_action}
    assert client_action_tools == {"open_url"}


def test_web_search_is_the_only_tool_that_returns_sources():
    sourced_tools = {name for name in agent_core.TOOL_FUNCTIONS if get_tool(name).returns_sources}
    assert sourced_tools == {"web_search"}


# "Must NOT allow arbitrary shell execution" (explicit product requirement):
# no registered tool's parameter schema should expose a raw command/shell
# string for the model to fill in and have executed verbatim. This won't
# catch every way a tool could be unsafe, but it catches the specific,
# concrete mistake of accidentally adding a run_shell_command-style tool.
BANNED_PARAMETER_NAMES = {"command", "cmd", "shell", "shell_command", "script"}


def test_no_tool_exposes_a_raw_command_parameter():
    for schema in agent_core.tools:
        properties = schema["function"]["parameters"].get("properties", {})
        offending = BANNED_PARAMETER_NAMES & set(properties)
        assert not offending, f"{schema['function']['name']} exposes raw command parameter(s): {offending}"
