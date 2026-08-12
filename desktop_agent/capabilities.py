"""Everything this desktop agent is willing to do when the backend asks.

Just like tool_registry.py on the server side, every capability the agent
can perform must be explicitly registered here by name. The backend can
only ever request one of these registered names with a fixed set of
keyword arguments — there is no path from a chat message to an arbitrary
shell command running on this Mac. Real macOS-affecting capabilities
(opening an app, reading system info, etc.) get added here in a later
phase; for now there's exactly one, used to prove the connection actually
works end to end.
"""

import platform

CAPABILITY_RISK = {
    "ping": "safe",
}


def ping(**_params) -> dict:
    return {
        "message": "pong",
        "hostname": platform.node(),
        "platform": platform.platform(),
    }


CAPABILITIES = {
    "ping": ping,
}

assert set(CAPABILITIES) == set(CAPABILITY_RISK), "every capability needs a declared risk level"
