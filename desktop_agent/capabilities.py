"""Everything this desktop agent is willing to do when the backend asks.

Just like tool_registry.py on the server side, every capability the agent
can perform must be explicitly registered here by name, with a declared
risk level. The backend can only ever request one of these registered
names with a fixed set of keyword arguments — there is no path from a
chat message to an arbitrary shell command running on this Mac. Every
subprocess call below passes a fixed argv list (never shell=True, never
string-interpolated shell text), so nothing here is vulnerable to shell
injection. AppleScript-based capabilities (close_application) validate
their string argument before splicing it into the script text, since
AppleScript injection is a separate concern from shell injection.

This is the "first safe, low-risk set" (per the product spec) — nothing
here is destructive or hard to reverse. Higher-risk system control
(sleep/restart/shutdown) is a deliberately separate, later phase that
will need real confirmation UX, not just a risk label.
"""

import base64
import os
import platform
import re
import subprocess
import tempfile

import psutil

CAPABILITY_RISK = {
    "ping": "safe",
    "open_application": "low",
    "close_application": "low",
    "get_running_applications": "safe",
    "get_active_application": "safe",
    "get_system_info": "safe",
    "get_battery_status": "safe",
    "get_memory_usage": "safe",
    "get_cpu_usage": "safe",
    "get_disk_usage": "safe",
    "open_finder": "safe",
    "take_screenshot": "low",
}

SUBPROCESS_TIMEOUT_SECONDS = 10


def ping(**_params) -> dict:
    return {
        "message": "pong",
        "hostname": platform.node(),
        "platform": platform.platform(),
    }


def open_application(name: str, **_params) -> dict:
    # `open -a <name>` resolves an application by name via Launch Services —
    # there's no shell involved, so there's nothing here for a stray quote
    # or space in `name` to break out of.
    result = subprocess.run(
        ["open", "-a", name], capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT_SECONDS
    )
    if result.returncode != 0:
        raise ValueError(f"Couldn't open '{name}': {result.stderr.strip() or 'application not found'}")
    return {"opened": name}


_UNSAFE_APPLESCRIPT_CHARS = re.compile(r'["\n\r\\]')


def close_application(name: str, **_params) -> dict:
    # AppleScript injection (not shell injection) is the actual risk here —
    # `name` gets spliced directly into a script string, so anything that
    # could break out of the quoted literal (a quote, a newline, a
    # backslash) is rejected outright rather than escaped, since a
    # rejected request is easier to reason about than an escaped one.
    if _UNSAFE_APPLESCRIPT_CHARS.search(name):
        raise ValueError("Application name contains characters that aren't allowed.")
    script = f'tell application "{name}" to quit'
    result = subprocess.run(
        ["osascript", "-e", script], capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT_SECONDS
    )
    if result.returncode != 0:
        raise ValueError(f"Couldn't close '{name}': {result.stderr.strip() or 'application not found or not running'}")
    return {"closed": name}


def get_running_applications(**_params) -> dict:
    result = subprocess.run(
        ["osascript", "-e", 'tell application "System Events" to get name of every application process whose background only is false'],
        capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        raise ValueError(f"Couldn't list running applications: {result.stderr.strip()}")
    apps = [name.strip() for name in result.stdout.strip().split(",") if name.strip()]
    return {"applications": apps}


def get_active_application(**_params) -> dict:
    result = subprocess.run(
        ["osascript", "-e", 'tell application "System Events" to get name of first application process whose frontmost is true'],
        capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        raise ValueError(f"Couldn't determine the active application: {result.stderr.strip()}")
    return {"active_application": result.stdout.strip()}


def get_system_info(**_params) -> dict:
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "macos_version": platform.mac_ver()[0],
        "architecture": platform.machine(),
    }


# pmset prints a line like "82%; discharging; 8:57 remaining present: true" —
# capture the state word directly rather than substring-matching "charging"
# against the whole line, which would also match inside "discharging".
_BATTERY_LINE_RE = re.compile(r"(\d+)%;\s*([a-zA-Z ]+?);")


def get_battery_status(**_params) -> dict:
    result = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT_SECONDS)
    if result.returncode != 0:
        raise ValueError(f"Couldn't read battery status: {result.stderr.strip()}")
    output = result.stdout
    match = _BATTERY_LINE_RE.search(output)
    if not match:
        # Desktop Macs (Mac mini, Mac Studio, some iMacs) have no battery —
        # that's a legitimate, expected answer, not a failure.
        return {"has_battery": False}
    state = match.group(2).strip().lower()
    return {
        "has_battery": True,
        "percent": int(match.group(1)),
        "charging": state in ("charging", "finishing charging"),
        "on_ac_power": "AC Power" in output,
    }


def get_memory_usage(**_params) -> dict:
    vm = psutil.virtual_memory()
    return {
        "total_bytes": vm.total,
        "used_bytes": vm.used,
        "available_bytes": vm.available,
        "percent_used": vm.percent,
    }


def get_cpu_usage(**_params) -> dict:
    # A short blocking sample — psutil measures over this interval rather
    # than returning an instantaneous (and much noisier) reading.
    return {"percent_used": psutil.cpu_percent(interval=0.5)}


def get_disk_usage(**_params) -> dict:
    du = psutil.disk_usage("/")
    return {
        "total_bytes": du.total,
        "used_bytes": du.used,
        "free_bytes": du.free,
        "percent_used": du.percent,
    }


def open_finder(path: str = None, **_params) -> dict:
    target = os.path.expanduser(path) if path else os.path.expanduser("~")
    if not os.path.isdir(target):
        raise ValueError(f"'{target}' is not a directory that exists on this Mac.")
    result = subprocess.run(["open", target], capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT_SECONDS)
    if result.returncode != 0:
        raise ValueError(f"Couldn't open Finder at '{target}': {result.stderr.strip()}")
    return {"opened": target}


# Screenshots are transported as base64 JSON over the same WebSocket used
# for every other capability, which has a real (~1MB) frame size limit —
# so a raw full-resolution PNG (often several MB) won't fit. Downscaling
# and re-encoding as JPEG keeps a genuinely useful screenshot comfortably
# under that limit instead of silently truncating or failing.
SCREENSHOT_MAX_WIDTH = 1280
SCREENSHOT_JPEG_QUALITY = 65


def take_screenshot(**_params) -> dict:
    with tempfile.TemporaryDirectory() as tmpdir:
        png_path = os.path.join(tmpdir, "shot.png")
        jpg_path = os.path.join(tmpdir, "shot.jpg")

        capture = subprocess.run(
            ["screencapture", "-x", png_path], capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT_SECONDS
        )
        if capture.returncode != 0 or not os.path.exists(png_path):
            raise ValueError(
                "Couldn't capture the screen — this Mac likely hasn't granted Screen "
                "Recording permission to whatever runs this agent. Enable it in "
                "System Settings > Privacy & Security > Screen Recording, then "
                "restart the agent. (" + capture.stderr.strip() + ")"
            )

        resize = subprocess.run(
            ["sips", "-Z", str(SCREENSHOT_MAX_WIDTH), "-s", "format", "jpeg",
             "-s", "formatOptions", str(SCREENSHOT_JPEG_QUALITY), png_path, "--out", jpg_path],
            capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT_SECONDS,
        )
        if resize.returncode != 0 or not os.path.exists(jpg_path):
            raise ValueError(f"Captured the screen but couldn't downscale it: {resize.stderr.strip()}")

        with open(jpg_path, "rb") as f:
            image_bytes = f.read()

    return {
        "image_base64": base64.b64encode(image_bytes).decode("ascii"),
        "format": "jpeg",
        "size_bytes": len(image_bytes),
    }


CAPABILITIES = {
    "ping": ping,
    "open_application": open_application,
    "close_application": close_application,
    "get_running_applications": get_running_applications,
    "get_active_application": get_active_application,
    "get_system_info": get_system_info,
    "get_battery_status": get_battery_status,
    "get_memory_usage": get_memory_usage,
    "get_cpu_usage": get_cpu_usage,
    "get_disk_usage": get_disk_usage,
    "open_finder": open_finder,
    "take_screenshot": take_screenshot,
}

assert set(CAPABILITIES) == set(CAPABILITY_RISK), "every capability needs a declared risk level"
