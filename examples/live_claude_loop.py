"""Bounded live test through the signed-in Claude Code CLI, with no tools.

Run from the repository root: python -m examples.live_claude_loop
Makes up to three paid model calls. Does not read or extract credentials.
"""

import json
import subprocess
import tempfile

from examples.goal_loop import inventory_verifier, supervise


def run_experiment() -> dict:
    snapshot = {"A100": 12, "B200": 5}
    calls = []
    system = (
        "Compile inventory reports. Return only a JSON object mapping requested "
        "SKUs to integer stock counts, without Markdown fences. "
        f"Authoritative teaching snapshot: {json.dumps(snapshot)}"
    )

    def worker(prompt):
        # An empty directory and explicit system prompt avoid loading repo context.
        with tempfile.TemporaryDirectory(prefix="goal-loop-") as cwd:
            result = subprocess.run(
                [
                    "claude", "-p", "--safe-mode", "--restricted",
                    "--tools", "", "--strict-mcp-config",
                    "--mcp-config", '{"mcpServers":{}}',
                    "--disable-slash-commands", "--no-session-persistence",
                    "--output-format", "json", "--model", "claude-opus-5",
                    "--effort", "low", "--max-budget-usd", "0.15",
                    "--system-prompt", system,
                ],
                input=prompt, text=True, capture_output=True, cwd=cwd, timeout=90,
            )
        # Do not expose raw errors: auth diagnostics may contain account details.
        if result.returncode != 0:
            raise RuntimeError("Claude Code call failed; inspect authentication privately.")
        payload = json.loads(result.stdout)
        if payload.get("is_error") or payload.get("subtype") != "success":
            raise RuntimeError("Claude Code did not report success.")
        artifact = payload["result"]
        calls.append({
            "artifact": artifact,
            "usage": payload.get("usage"),
            "model_usage": payload.get("modelUsage"),
            "reported_cost_usd": payload.get("total_cost_usd"),
            "duration_ms": payload.get("duration_ms"),
        })
        return artifact

    objective = "Report inventory for A100 and B200 from the supplied snapshot."
    normal = supervise(objective, worker, inventory_verifier(snapshot), max_turns=1)
    injected = []

    def corrupt_first_artifact(prompt):
        artifact = worker(prompt)
        if not injected:
            report = json.loads(artifact)
            # Controlled delivery fault, NOT an observed spontaneous model error.
            report.pop("B200", None)
            artifact = json.dumps(report)
            injected.append(artifact)
        return artifact

    recovery = None
    if normal.status == "complete":
        recovery = supervise(
            objective, corrupt_first_artifact, inventory_verifier(snapshot), max_turns=2,
        )

    def summarize(outcome):
        if outcome is None:
            return None
        return {
            "status": outcome.status,
            "reason": outcome.reason,
            "attempts": [vars(attempt) for attempt in outcome.attempts],
        }

    return {
        "provider_surface": "Claude Code CLI (existing login), tools disabled",
        "model_requested": "claude-opus-5",
        "fault": "Host removes B200 from the first recovery artifact after generation",
        "normal": summarize(normal),
        "recovery": summarize(recovery),
        "calls": calls,
    }


if __name__ == "__main__":
    print(json.dumps(run_experiment(), indent=2))
