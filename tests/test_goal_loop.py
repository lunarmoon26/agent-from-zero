"""Offline controller tests, not model-quality evaluations."""

import pytest

from examples.goal_loop import Check, inventory_verifier, supervise


def test_premature_final_answer_receives_feedback_and_another_turn():
    prompts = []
    responses = iter(['{"A100": 12}', '{"A100": 12, "B200": 5}'])

    def worker(prompt):
        prompts.append(prompt)
        return next(responses)

    result = supervise("Both SKUs", worker, inventory_verifier({"A100": 12, "B200": 5}))
    assert result.status == "complete"
    assert len(result.attempts) == 2
    assert "Objective: Both SKUs" in prompts[1]
    assert "Required SKUs: A100, B200" in prompts[1]


@pytest.mark.parametrize("artifact", [
    "Done!", "[]", '{"A100": true}', '{"A100": 12.0}',
    '{"A100": 11}', '{"A100": 12, "EXTRA": 0}',
])
def test_invalid_artifact_is_rejected(artifact):
    assert not inventory_verifier({"A100": 12})(artifact).passed


def test_snapshot_is_copied():
    snapshot = {"A100": 12}
    verify = inventory_verifier(snapshot)
    snapshot["A100"] = 0
    assert verify('{"A100": 12}').passed


def test_identical_rejected_artifact_stalls():
    result = supervise("Counts", lambda _: "Done!", inventory_verifier({"A100": 12}))
    assert result.status == "stalled"
    assert len(result.attempts) == 2


def test_different_wrong_answers_exhaust_turn_limit():
    responses = iter(['{"A100": 1}', '{"A100": 2}', '{"A100": 3}'])
    result = supervise("Counts", lambda _: next(responses), inventory_verifier({"A100": 12}), max_turns=3)
    assert result.status == "limited"
    assert len(result.attempts) == 3


def test_success_on_last_admitted_turn():
    result = supervise("Counts", lambda _: '{"A100": 12}', inventory_verifier({"A100": 12}), max_turns=1)
    assert result.status == "complete"


def fail(_):
    raise RuntimeError("simulated outage")


def test_verifier_failure_blocks_instead_of_accepting():
    result = supervise("Counts", lambda _: "Done!", fail)
    assert result.status == "blocked"
    assert result.attempts[0].feedback == "Verification unavailable."


def test_worker_failure_is_not_automatically_retried():
    result = supervise("Counts", fail, inventory_verifier({"A100": 12}))
    assert result.status == "blocked"
    assert not result.attempts


def test_cancel_before_admission_never_calls_worker():
    result = supervise("Counts", fail, fail, cancelled=lambda: True)
    assert result.status == "cancelled"
    assert not result.attempts


def test_cancel_during_worker_prevents_verification():
    stopped = False

    def worker(_):
        nonlocal stopped
        stopped = True
        return "Done!"

    result = supervise("Counts", worker, fail, cancelled=lambda: stopped)
    assert result.status == "cancelled"
    assert len(result.attempts) == 1


def test_cancel_during_verification_prevents_completion():
    stopped = False

    def verify(_):
        nonlocal stopped
        stopped = True
        return Check(True, "passed")

    result = supervise("Counts", lambda _: "{}", verify, cancelled=lambda: stopped)
    assert result.status == "cancelled"


@pytest.mark.parametrize("limits", [{"max_turns": 0}, {"repeat_limit": 0}])
def test_invalid_limits(limits):
    with pytest.raises(ValueError):
        supervise("Counts", fail, fail, **limits)


def test_malformed_verdict_fails_closed():
    result = supervise("Counts", lambda _: "{}", lambda _: Check("false", "wrong type"))
    assert result.status == "blocked"
