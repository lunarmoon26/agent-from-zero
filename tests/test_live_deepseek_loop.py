"""Offline tests: no credentials or paid requests."""

from types import SimpleNamespace

from examples.live_deepseek_loop import run_experiment


def client_with(create):
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def response(content='{"A100": 12, "B200": 5}', finish_reason="stop"):
    return SimpleNamespace(
        model="test-model",
        choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish_reason)],
        usage=SimpleNamespace(prompt_tokens=84, completion_tokens=14),
    )


def test_api_adapter_and_injected_recovery():
    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        return response()

    report = run_experiment(client_with(create))
    assert report["normal"]["status"] == "complete"
    assert report["recovery"]["status"] == "complete"
    assert len(requests) == len(report["calls"]) == 3
    assert report["recovery"]["attempts"][0]["artifact"] == '{"A100": 12}'
    assert "Required SKUs: A100, B200." in requests[2]["messages"][1]["content"]
    assert all(request["max_tokens"] == 128 and "tools" not in request for request in requests)
    assert all(call["cache_hit_input_tokens"] is None for call in report["calls"])


def test_request_failure_is_not_retried_or_exposed():
    calls = []

    def create(**kwargs):
        calls.append(1)
        raise RuntimeError("private diagnostic")

    report = run_experiment(client_with(create))
    assert report["normal"]["status"] == "blocked"
    assert report["recovery"] is None
    assert len(calls) == 1
    assert "private diagnostic" not in str(report)


def test_truncated_response_does_not_complete():
    report = run_experiment(client_with(lambda **kwargs: response(finish_reason="length")))
    assert report["normal"]["status"] == "blocked"
    assert report["recovery"] is None
    assert report["calls"][0]["finish_reason"] == "length"


def test_null_content_does_not_complete():
    report = run_experiment(client_with(lambda **kwargs: response(content=None)))
    assert report["normal"]["status"] == "blocked"
    assert report["recovery"] is None
