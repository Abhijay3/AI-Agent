import base64
from subprocess import CompletedProcess

import pytest

from desktop_agent import capabilities as c


def _proc(returncode=0, stdout="", stderr=""):
    return CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def test_every_capability_has_a_declared_risk_level():
    assert set(c.CAPABILITIES) == set(c.CAPABILITY_RISK)


def test_ping_returns_hostname_and_platform():
    result = c.ping()
    assert result["message"] == "pong"
    assert result["hostname"]
    assert result["platform"]


def test_open_application_success(monkeypatch):
    monkeypatch.setattr(c.subprocess, "run", lambda *a, **k: _proc(0))
    assert c.open_application("Calculator") == {"opened": "Calculator"}


def test_open_application_not_found_raises_friendly_error(monkeypatch):
    monkeypatch.setattr(c.subprocess, "run", lambda *a, **k: _proc(1, stderr="Unable to find application"))
    with pytest.raises(ValueError, match="Couldn't open"):
        c.open_application("NotARealApp")


def test_close_application_success(monkeypatch):
    monkeypatch.setattr(c.subprocess, "run", lambda *a, **k: _proc(0))
    assert c.close_application("Calculator") == {"closed": "Calculator"}


@pytest.mark.parametrize("unsafe_name", ['Bad"app', "line\nbreak", "back\\slash"])
def test_close_application_rejects_applescript_injection_attempts(unsafe_name):
    # Must be rejected before subprocess.run is ever called — these
    # characters could otherwise break out of the quoted AppleScript
    # literal and splice in arbitrary script text.
    with pytest.raises(ValueError, match="allowed"):
        c.close_application(unsafe_name)


def test_get_running_applications_parses_comma_separated_output(monkeypatch):
    monkeypatch.setattr(c.subprocess, "run", lambda *a, **k: _proc(0, stdout="Finder, Safari, Code\n"))
    assert c.get_running_applications() == {"applications": ["Finder", "Safari", "Code"]}


def test_get_active_application_strips_output(monkeypatch):
    monkeypatch.setattr(c.subprocess, "run", lambda *a, **k: _proc(0, stdout="Safari\n"))
    assert c.get_active_application() == {"active_application": "Safari"}


def test_get_system_info_returns_real_platform_fields():
    result = c.get_system_info()
    assert result["hostname"]
    assert result["platform"]
    assert "architecture" in result


def test_get_battery_status_parses_discharging(monkeypatch):
    monkeypatch.setattr(
        c.subprocess, "run",
        lambda *a, **k: _proc(0, stdout="Now drawing from 'Battery Power'\n -InternalBattery-0\t82%; discharging; 8:57 remaining present: true"),
    )
    result = c.get_battery_status()
    assert result == {"has_battery": True, "percent": 82, "charging": False, "on_ac_power": False}


def test_get_battery_status_parses_charging_not_confused_with_discharging(monkeypatch):
    # Regression: a naive `"charging" in output` substring check also
    # matches inside the word "discharging" — this must not happen.
    monkeypatch.setattr(
        c.subprocess, "run",
        lambda *a, **k: _proc(0, stdout="Now drawing from 'AC Power'\n -InternalBattery-0\t95%; charging; 0:20 remaining present: true"),
    )
    result = c.get_battery_status()
    assert result["charging"] is True
    assert result["on_ac_power"] is True


def test_get_battery_status_parses_charged(monkeypatch):
    monkeypatch.setattr(
        c.subprocess, "run",
        lambda *a, **k: _proc(0, stdout="Now drawing from 'AC Power'\n -InternalBattery-0\t100%; charged; 0:00 remaining present: true"),
    )
    assert c.get_battery_status()["charging"] is False


def test_get_battery_status_reports_no_battery_on_desktop_macs(monkeypatch):
    monkeypatch.setattr(c.subprocess, "run", lambda *a, **k: _proc(0, stdout="No batteries available.\n"))
    assert c.get_battery_status() == {"has_battery": False}


def test_get_memory_usage_returns_real_values():
    result = c.get_memory_usage()
    assert result["total_bytes"] > 0
    assert 0 <= result["percent_used"] <= 100


def test_get_cpu_usage_returns_a_percentage():
    result = c.get_cpu_usage()
    assert 0 <= result["percent_used"] <= 100


def test_get_disk_usage_returns_real_values():
    result = c.get_disk_usage()
    assert result["total_bytes"] > 0
    assert 0 <= result["percent_used"] <= 100


def test_open_finder_rejects_a_nonexistent_path():
    with pytest.raises(ValueError, match="not a directory"):
        c.open_finder("/definitely/does/not/exist/anywhere")


def test_open_finder_defaults_to_home_directory(monkeypatch):
    calls = []
    monkeypatch.setattr(c.subprocess, "run", lambda args, **k: calls.append(args) or _proc(0))
    result = c.open_finder()
    assert result["opened"] == c.os.path.expanduser("~")
    assert calls[0] == ["open", c.os.path.expanduser("~")]


def test_take_screenshot_raises_friendly_error_when_capture_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(c.tempfile, "TemporaryDirectory", lambda: __import__("contextlib").nullcontext(str(tmp_path)))
    monkeypatch.setattr(c.subprocess, "run", lambda *a, **k: _proc(1, stderr="could not create image from display"))
    with pytest.raises(ValueError, match="Screen Recording"):
        c.take_screenshot()


def test_take_screenshot_returns_base64_jpeg_on_success(monkeypatch, tmp_path):
    fake_jpeg_bytes = b"\xff\xd8\xff\xe0fake-jpeg-data"

    def fake_run(args, **k):
        if args[0] == "screencapture":
            with open(args[2], "wb") as f:
                f.write(b"fake-png-data")
            return _proc(0)
        if args[0] == "sips":
            out_path = args[-1]
            with open(out_path, "wb") as f:
                f.write(fake_jpeg_bytes)
            return _proc(0)
        raise AssertionError(f"unexpected command: {args}")

    monkeypatch.setattr(c.tempfile, "TemporaryDirectory", lambda: __import__("contextlib").nullcontext(str(tmp_path)))
    monkeypatch.setattr(c.subprocess, "run", fake_run)

    result = c.take_screenshot()
    assert result["format"] == "jpeg"
    assert base64.b64decode(result["image_base64"]) == fake_jpeg_bytes
    assert result["size_bytes"] == len(fake_jpeg_bytes)
