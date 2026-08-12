"""Registers the desktop agent's macOS capabilities (desktop_agent/
capabilities.py) as tools the AI can call during a normal conversation —
the "Tool Router" step in Browser -> FastAPI backend -> AI Orchestrator ->
Tool Router -> Authenticated Local Desktop Agent -> macOS.

Every handler here goes through desktop_agent_hub.call_sync(), which only
ever reaches a capability name the connected agent itself explicitly
registered (see desktop_agent/capabilities.py) — there's no path from a
chat message to an arbitrary command on the user's Mac, and this backend
process (a cloud container) never touches the local machine it runs on.

Importing this module is what registers these tools — agent_core.py does
that import before deriving `tools`/TOOL_FUNCTIONS/TOOL_RISK from the
registry, so nothing else needs to change to pick them up.
"""

import desktop_agent_hub
from tool_registry import Tool, register_tool


def _desktop_capability_handler(capability_name: str):
    def handler(**params) -> object:
        try:
            response = desktop_agent_hub.hub.call_sync(capability_name, params)
        except ConnectionError as e:
            raise ValueError(
                "The desktop agent isn't connected right now — the user needs to "
                "run it on their Mac (python3 desktop_agent/agent.py) first."
            ) from e
        except TimeoutError as e:
            raise ValueError("The desktop agent didn't respond in time.") from e
        if not response.get("ok"):
            raise ValueError(response.get("error") or f"'{capability_name}' failed on the desktop agent.")
        return response["result"]

    return handler


def _screenshot_summary(result: dict) -> str:
    return f"Screenshot captured ({result.get('size_bytes', '?')} bytes, {result.get('format', 'jpeg')})."


def _screenshot_client_action(_args: dict, result: dict) -> dict:
    return {
        "action": "show_image",
        "image_base64": result["image_base64"],
        "format": result.get("format", "jpeg"),
    }


NO_PARAMS = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}


register_tool(Tool(
    name="open_application",
    description=(
        "Open/launch an application on the user's Mac by name (e.g. 'Safari', "
        "'Visual Studio Code', 'Calculator'). Requires the desktop agent to be "
        "connected — if it isn't, say so plainly instead of pretending to."
    ),
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "The application's name, as it appears in the Dock or Applications folder."},
        },
        "required": ["name"],
        "additionalProperties": False,
    },
    handler=_desktop_capability_handler("open_application"),
    risk_level="low",
))
register_tool(Tool(
    name="close_application",
    description="Quit a running application on the user's Mac by name. This may lose unsaved work in that application.",
    parameters={
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "additionalProperties": False,
    },
    handler=_desktop_capability_handler("close_application"),
    risk_level="low",
))
register_tool(Tool(
    name="get_running_applications",
    description="List the applications currently running (visible, non-background) on the user's Mac.",
    parameters=NO_PARAMS,
    handler=_desktop_capability_handler("get_running_applications"),
))
register_tool(Tool(
    name="get_active_application",
    description="Get the name of whichever application is currently in the foreground on the user's Mac.",
    parameters=NO_PARAMS,
    handler=_desktop_capability_handler("get_active_application"),
))
register_tool(Tool(
    name="get_mac_system_info",
    description="Get the user's Mac's hostname, macOS version, and CPU architecture.",
    parameters=NO_PARAMS,
    handler=_desktop_capability_handler("get_system_info"),
))
register_tool(Tool(
    name="get_battery_status",
    description=(
        "Get the user's Mac's battery percentage and charging state. Returns "
        "has_battery: false on desktop Macs (Mac mini, Mac Studio, etc.) that have no battery."
    ),
    parameters=NO_PARAMS,
    handler=_desktop_capability_handler("get_battery_status"),
))
register_tool(Tool(
    name="get_mac_memory_usage",
    description="Get the user's Mac's current RAM usage.",
    parameters=NO_PARAMS,
    handler=_desktop_capability_handler("get_memory_usage"),
))
register_tool(Tool(
    name="get_mac_cpu_usage",
    description="Get the user's Mac's current CPU usage percentage.",
    parameters=NO_PARAMS,
    handler=_desktop_capability_handler("get_cpu_usage"),
))
register_tool(Tool(
    name="get_mac_disk_usage",
    description="Get the user's Mac's disk space usage (total/used/free) for its main volume.",
    parameters=NO_PARAMS,
    handler=_desktop_capability_handler("get_disk_usage"),
))
register_tool(Tool(
    name="open_finder",
    description="Open a Finder window on the user's Mac at a given folder path. Defaults to the home folder if no path is given.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute or ~-relative folder path. Omit for the home folder."},
        },
        "required": [],
        "additionalProperties": False,
    },
    handler=_desktop_capability_handler("open_finder"),
))
register_tool(Tool(
    name="take_screenshot",
    description="Capture a screenshot of the user's Mac screen right now and show it to them in the chat.",
    parameters=NO_PARAMS,
    handler=_desktop_capability_handler("take_screenshot"),
    risk_level="low",
    summarize_for_model=_screenshot_summary,
    client_action=_screenshot_client_action,
))
