import os

os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("TAVILY_API_KEY", "test-tavily-key")

import pytest

import agent_core  # noqa: E402 — importing this registers all tools, including desktop_agent_tools
import desktop_agent_hub  # noqa: E402
from tool_registry import get_tool  # noqa: E402

DESKTOP_TOOL_NAMES = [
    "open_application",
    "close_application",
    "get_running_applications",
    "get_active_application",
    "get_mac_system_info",
    "get_battery_status",
    "get_mac_memory_usage",
    "get_mac_cpu_usage",
    "get_mac_disk_usage",
    "open_finder",
    "take_screenshot",
]


def test_all_desktop_tools_are_registered():
    for name in DESKTOP_TOOL_NAMES:
        tool = get_tool(name)
        assert tool.name == name
        assert callable(tool.handler)


def test_desktop_tools_are_all_safe_or_low_risk():
    # This is the "first safe, low-risk set" — nothing here should be
    # registered at a risk level implying it needs confirmation yet.
    for name in DESKTOP_TOOL_NAMES:
        tool = get_tool(name)
        assert tool.risk_level in ("safe", "low")
        assert tool.requires_confirmation is False


def test_open_url_capability_not_duplicated_as_a_desktop_tool():
    # open_url already exists as a browser-tab client_action — Phase 4/5
    # deliberately didn't re-implement it as a desktop capability.
    assert "open_url" not in DESKTOP_TOOL_NAMES


def test_handler_returns_result_on_success(monkeypatch):
    monkeypatch.setattr(
        desktop_agent_hub.hub, "call_sync",
        lambda capability, params=None: {"ok": True, "result": {"percent": 82}},
    )
    handler = get_tool("get_battery_status").handler
    assert handler() == {"percent": 82}


def test_handler_forwards_params_to_call_sync(monkeypatch):
    seen = {}

    def fake_call_sync(capability, params=None):
        seen["capability"] = capability
        seen["params"] = params
        return {"ok": True, "result": {"opened": "Calculator"}}

    monkeypatch.setattr(desktop_agent_hub.hub, "call_sync", fake_call_sync)
    handler = get_tool("open_application").handler
    result = handler(name="Calculator")

    assert seen == {"capability": "open_application", "params": {"name": "Calculator"}}
    assert result == {"opened": "Calculator"}


def test_handler_raises_value_error_when_capability_fails(monkeypatch):
    monkeypatch.setattr(
        desktop_agent_hub.hub, "call_sync",
        lambda capability, params=None: {"ok": False, "error": "application not found"},
    )
    handler = get_tool("open_application").handler
    with pytest.raises(ValueError, match="application not found"):
        handler(name="NotARealApp")


def test_handler_raises_friendly_error_when_agent_not_connected(monkeypatch):
    def raise_connection_error(capability, params=None):
        raise ConnectionError("Desktop agent is not connected")

    monkeypatch.setattr(desktop_agent_hub.hub, "call_sync", raise_connection_error)
    handler = get_tool("get_mac_system_info").handler
    with pytest.raises(ValueError, match="isn't connected"):
        handler()


def test_handler_raises_friendly_error_on_timeout(monkeypatch):
    def raise_timeout(capability, params=None):
        raise TimeoutError("too slow")

    monkeypatch.setattr(desktop_agent_hub.hub, "call_sync", raise_timeout)
    handler = get_tool("get_mac_cpu_usage").handler
    with pytest.raises(ValueError, match="didn't respond in time"):
        handler()


def test_take_screenshot_summary_and_client_action():
    tool = get_tool("take_screenshot")
    result = {"image_base64": "abcd1234", "format": "jpeg", "size_bytes": 42}

    summary = tool.summarize_for_model(result)
    assert "42" in summary
    assert "jpeg" in summary
    assert "abcd1234" not in summary  # the model must never see the raw image data

    action = tool.client_action({}, result)
    assert action == {"action": "show_image", "image_base64": "abcd1234", "format": "jpeg"}


def test_run_tool_uses_summarize_for_model_and_client_action_for_screenshot(monkeypatch):
    monkeypatch.setitem(
        agent_core.TOOL_FUNCTIONS, "take_screenshot",
        lambda **params: {"image_base64": "zzz", "format": "jpeg", "size_bytes": 999},
    )
    content, sources, client_action = agent_core._run_tool("take_screenshot", "{}", "u1")

    assert content == "Screenshot captured (999 bytes, jpeg)."
    assert sources is None
    assert client_action == {"action": "show_image", "image_base64": "zzz", "format": "jpeg"}


def test_run_tool_json_encodes_dict_results_for_tools_without_a_summarizer(monkeypatch):
    monkeypatch.setitem(
        agent_core.TOOL_FUNCTIONS, "get_mac_memory_usage",
        lambda **params: {"percent_used": 57.0},
    )
    content, sources, client_action = agent_core._run_tool("get_mac_memory_usage", "{}", "u1")

    assert content == '{"percent_used": 57.0}'
    assert sources is None
    assert client_action is None


def test_run_tool_surfaces_desktop_tool_errors_to_the_model(monkeypatch):
    def raise_connection_error(capability, params=None):
        raise ConnectionError("Desktop agent is not connected")

    monkeypatch.setattr(desktop_agent_hub.hub, "call_sync", raise_connection_error)
    content, sources, client_action = agent_core._run_tool("open_application", '{"name": "Calculator"}', "u1")

    assert content.startswith("Error:")
    assert "isn't connected" in content
    assert sources is None
    assert client_action is None
