"""A synchronous, single-owner goal supervisor. No network calls in the demo."""

from dataclasses import dataclass
import argparse
import json
from typing import Callable, Literal


@dataclass(frozen=True)
class Check:
    passed: bool
    feedback: str


@dataclass(frozen=True)
class Attempt:
    number: int
    artifact: str
    feedback: str


@dataclass(frozen=True)
class Outcome:
    status: Literal["complete", "limited", "stalled", "blocked", "cancelled"]
    attempts: tuple[Attempt, ...]
    reason: str


def supervise(
    objective: str,
    worker: Callable[[str], str],
    verify: Callable[[str], Check],
    *,
    max_turns: int = 4,
    repeat_limit: int = 2,
    cancelled: Callable[[], bool] = lambda: False,
) -> Outcome:
    """Admit bounded turns; only the host verifier can accept an artifact.

    Cancellation is cooperative between calls. There is no deadline, durable
    checkpoint, tool sandbox, or guarantee against duplicate external effects.
    """
    if max_turns < 1 or repeat_limit < 1:
        raise ValueError("limits must be positive")
    attempts: list[Attempt] = []
    feedback = "No artifact has been checked yet."
    previous: str | None = None
    repeated = 0

    def stop(status, reason):
        return Outcome(status, tuple(attempts), reason)

    for number in range(1, max_turns + 1):
        if cancelled():
            return stop("cancelled", "Cancelled before admitting another turn.")
        prompt = (
            f"Objective: {objective}\n"
            f"Host acceptance feedback: {feedback}\n"
            "Return a fresh JSON artifact for the complete objective."
        )
        try:
            artifact = worker(prompt)
            if not isinstance(artifact, str):
                raise TypeError("worker must return text")
        except Exception:
            # No automatic retry: the failed call might have changed external state.
            return stop("blocked", "Worker failed; inspect before retrying.")
        if cancelled():
            attempts.append(Attempt(number, artifact, "Not checked: cancelled."))
            return stop("cancelled", "Cancelled after the worker returned.")
        try:
            check = verify(artifact)
            if not isinstance(check, Check) or type(check.passed) is not bool:
                raise TypeError("verifier must return Check with a boolean verdict")
        except Exception:
            attempts.append(Attempt(number, artifact, "Verification unavailable."))
            return stop("blocked", "Cannot verify; completion is not established.")
        feedback = check.feedback
        attempts.append(Attempt(number, artifact, feedback))
        if cancelled():
            return stop("cancelled", "Cancelled before accepting the result.")
        if check.passed:
            return stop("complete", feedback)
        repeated = repeated + 1 if artifact == previous else 1
        previous = artifact
        if repeated >= repeat_limit:
            return stop("stalled", "Identical rejected artifact repeated.")
    return stop("limited", "Turn limit reached without passing acceptance.")


def inventory_verifier(snapshot: dict[str, int]) -> Callable[[str], Check]:
    """Capture a host-owned acceptance snapshot, isolated from caller mutation."""
    expected = dict(snapshot)
    if not expected or any(type(v) is not int or v < 0 for v in expected.values()):
        raise ValueError("snapshot must contain nonnegative integer counts")

    def verify(artifact: str) -> Check:
        try:
            report = json.loads(artifact)
        except (ValueError, TypeError):
            return Check(False, "Return a JSON object containing every requested SKU.")
        if not isinstance(report, dict):
            return Check(False, "Report must be a JSON object.")
        if set(report) != set(expected):
            return Check(False, f"Required SKUs: {', '.join(sorted(expected))}.")
        incorrect = [
            sku for sku, count in expected.items()
            if type(report[sku]) is not int or report[sku] != count
        ]
        if incorrect:
            return Check(False, f"Recheck inventory for: {', '.join(incorrect)}.")
        return Check(True, "All requested counts match the captured snapshot.")

    return verify


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Call DeepSeek (requires credentials and incurs API charges).")
    args = parser.parse_args()
    snapshot = {"A100": 12, "B200": 5}
    # Scripted worker deliberately ends its first turn too early.
    responses = iter(['{"A100": 12}', '{"A100": 12, "B200": 5}'])
    worker = lambda prompt: next(responses)
    if args.live:
        # A text-only worker keeps the demo read-only. No upstream write tools.
        from src.agent.llm import chat

        def worker(prompt):
            return chat(
                prompt,
                system=(
                    "Compile inventory reports. Return only a JSON object mapping "
                    "requested SKUs to integer stock counts; no Markdown fences. "
                    f"Authoritative teaching snapshot: {json.dumps(snapshot)}"
                ),
            ).text

    outcome = supervise(
        "Report inventory for A100 and B200 from the supplied snapshot.",
        worker,
        inventory_verifier(snapshot),
    )
    for attempt in outcome.attempts:
        print(f"turn {attempt.number}: {attempt.artifact}")
        print(f"check: {attempt.feedback}")
    print(f"status: {outcome.status}")


if __name__ == "__main__":
    main()
