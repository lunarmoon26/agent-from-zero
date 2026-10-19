"""Up to three live, read-only DeepSeek calls using an exported API key."""

import json
import os
import time

from openai import OpenAI

from examples.goal_loop import inventory_verifier, supervise
from src.agent.llm import DEFAULT_MODEL, DEEPSEEK_BASE_URL


def run_experiment(client) -> dict:
    snapshot = {"A100": 12, "B200": 5}
    calls = []
    system = (
        "Compile inventory reports. Return only a JSON object mapping requested "
        "SKUs to integer stock counts, without Markdown fences. "
        f"Authoritative teaching snapshot: {json.dumps(snapshot)}"
    )

    def worker(prompt):
        started = time.perf_counter()
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            temperature=0.0,
            max_tokens=128,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        choice = response.choices[0]
        usage = response.usage
        calls.append({
            "model_returned": response.model,
            "artifact": choice.message.content,
            "finish_reason": choice.finish_reason,
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "cache_hit_input_tokens": getattr(usage, "prompt_cache_hit_tokens", None),
            "cache_miss_input_tokens": getattr(usage, "prompt_cache_miss_tokens", None),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        })
        if choice.finish_reason != "stop":
            raise RuntimeError("Model did not finish normally.")
        return choice.message.content

    objective = "Report inventory for A100 and B200 from the supplied snapshot."
    normal = supervise(objective, worker, inventory_verifier(snapshot), max_turns=1)
    injected = []

    def corrupt_first_artifact(prompt):
        artifact = worker(prompt)
        if not injected:
            report = json.loads(artifact)
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
        "provider_surface": "DeepSeek API via OpenAI SDK, no tools",
        "model_requested": DEFAULT_MODEL,
        "max_worker_calls": 3,
        "max_output_tokens_per_call": 128,
        "fault": "Host removes B200 from the first recovery artifact after generation",
        "normal": summarize(normal),
        "recovery": summarize(recovery),
        "calls": calls,
        "cost_note": "Token usage only; no current-price estimate or invoice verified.",
    }


if __name__ == "__main__":
    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"], base_url=DEEPSEEK_BASE_URL,
        timeout=30.0, max_retries=0,
    )
    with client:
        print(json.dumps(run_experiment(client), indent=2))
