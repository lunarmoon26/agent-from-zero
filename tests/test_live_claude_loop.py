"""Test the CLI adapter without launching Claude or spending credits."""

import json
from types import SimpleNamespace

from examples import live_claude_loop


def test_live_adapter_and_injected_recovery(monkeypatch):
    requests = []

    def run(command, **kwargs):
        requests.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=json.dumps({
            "subtype": "success", "is_error": False,
            "result": '{"A100": 12, "B200": 5}', "total_cost_usd": 0.01,
        }))

    monkeypatch.setattr(live_claude_loop.subprocess, "run", run)
    report = live_claude_loop.run_experiment()
    assert report["normal"]["status"] == "complete"
    assert report["recovery"]["status"] == "complete"
    assert len(requests) == len(report["calls"]) == 3
    assert json.loads(report["recovery"]["attempts"][0]["artifact"]) == {"A100": 12}
    assert all(json.loads(call["artifact"]) == {"A100": 12, "B200": 5} for call in report["calls"])
    assert "Required SKUs: A100, B200." in requests[2][1]["input"]
    for command, kwargs in requests:
        assert command[command.index("--tools") + 1] == ""
        assert "--safe-mode" in command and "--strict-mcp-config" in command
        assert command[command.index("--max-budget-usd") + 1] == "0.15"
        assert kwargs["timeout"] == 90


def test_cli_error_blocks_without_exposing_output(monkeypatch):
    monkeypatch.setattr(live_claude_loop.subprocess, "run", lambda *a, **kw: SimpleNamespace(
        returncode=1, stdout="private diagnostic", stderr="private diagnostic",
    ))
    report = live_claude_loop.run_experiment()
    assert report["normal"]["status"] == "blocked"
    assert report["recovery"] is None
    assert "private diagnostic" not in json.dumps(report)


def test_successful_cli_exit_with_error_payload_is_blocked(monkeypatch):
    monkeypatch.setattr(live_claude_loop.subprocess, "run", lambda *a, **kw: SimpleNamespace(
        returncode=0, stdout=json.dumps({"subtype": "error_max_budget_usd", "is_error": True}),
    ))
    report = live_claude_loop.run_experiment()
    assert report["normal"]["status"] == "blocked"
    assert report["recovery"] is None


def test_timeout_is_not_retried(monkeypatch):
    calls = []

    def run(*a, **kw):
        calls.append(1)
        raise live_claude_loop.subprocess.TimeoutExpired("claude", 90)

    monkeypatch.setattr(live_claude_loop.subprocess, "run", run)
    report = live_claude_loop.run_experiment()
    assert report["normal"]["status"] == "blocked"
    assert len(calls) == 1
