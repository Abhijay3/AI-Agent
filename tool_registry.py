"""Central tool registry.

Every tool declares its name, description, parameter schema, execution
function, risk level, and confirmation requirement in exactly one place —
instead of that information being hand-duplicated across an OpenAI function
schema list, a name-to-function lookup dict, and a separate risk-level dict
that all have to be kept in sync by hand (which is how this worked before;
nothing enforced the three stayed consistent except code review).

Future tool modules (macOS tools, file tools, etc.) register into this same
registry via register_tool() from wherever they're defined. agent_core.py's
tool-calling loop reads everything through this module, so adding a new
tool never requires editing that loop — only registering the tool.

Tools must never accept a raw shell/command string from the model and exec
it verbatim — parameters should be specific and structured (e.g. an enum of
allowed values, a validated path), not a free-text command. See
test_tool_registry.py for the regression check that guards against this.
"""

from dataclasses import dataclass
from typing import Callable, Literal, Optional

RiskLevel = Literal["safe", "low", "medium", "high", "critical"]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict
    handler: Callable[..., object]
    risk_level: RiskLevel = "safe"
    requires_confirmation: bool = False
    # True for tools whose handler needs the caller's user_id injected
    # server-side (e.g. to scope memory reads/writes to that person) —
    # never exposed to the model as a callable parameter, so it can't be
    # steered into acting on another user's data.
    needs_user_id: bool = False
    # True for tools that return (content, sources) instead of a plain
    # string — lets the UI show real citations (e.g. web_search).
    returns_sources: bool = False
    # Optional builder for a "client_action" the frontend must carry out
    # itself (e.g. opening a URL in the user's actual browser tab — the
    # backend is a cloud container, it can't do that directly). Called with
    # the tool's parsed arguments; returns the action dict, or None.
    client_action: Optional[Callable[[dict], Optional[dict]]] = None


_REGISTRY: dict = {}


def register_tool(tool: Tool) -> None:
    if tool.name in _REGISTRY:
        raise ValueError(f"Tool '{tool.name}' is already registered.")
    _REGISTRY[tool.name] = tool


def get_tool(name: str) -> Tool:
    return _REGISTRY[name]


def all_tools() -> list:
    return list(_REGISTRY.values())


def openai_tool_schemas() -> list:
    """The `tools` param OpenAI-compatible function calling expects —
    derived from the registry so it can never drift out of sync with what's
    actually registered."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in _REGISTRY.values()
    ]
