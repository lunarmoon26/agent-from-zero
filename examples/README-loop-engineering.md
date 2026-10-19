# A small goal supervisor

Original companion example extending the teaching approach of
[Bange's Agent From Zero](https://www.bange.life/tech/agent-course-01-overview).
Upstream baseline: `sleepworm/agent-from-zero` at
`55c55ba5fd2b88a6d6de3e35c3cc96cbb310695c`.
Upstream source files are unchanged. No upstream license was found at that
revision; this example does not assign a license to the upstream project.

## Run without a model

Python 3.11+; use the repository root:

```sh
uv run --no-project python examples/goal_loop.py
uv run --with pytest python -m pytest tests/test_goal_loop.py -q
```

The scripted first turn returns only A100. The host verifier requests both
SKUs. The second turn passes. This reproduces an incomplete **final answer**,
not a claim that a real model exhibited that failure.

The example uses its own frozen-in-time teaching data (A100: 12, B200: 5),
not the upstream mutable inventory fixture. `inventory_verifier` captures a
copy. Its verdict proves only exact agreement with that captured snapshot,
not current warehouse inventory, tool use, provenance, or a correct business
decision. Duplicate JSON keys follow Python's default last-value behavior.

## Optional real-model worker

Only run this if you intend to spend API credits. Export `DEEPSEEK_API_KEY`
in your shell first; this example does not load `.env` automatically.

```sh
uv run python -m examples.goal_loop --live
```

The adapter uses the pinned upstream `chat` function and its default model.
It puts the snapshot in the system message and exposes **no tools**. This is
deliberately simpler than the upstream agent: the outer controller works with
either a text-only worker or a complete tool-driven turn. Upstream `ask()`
exposes stock-changing tools, so do not blindly wrap and retry it.

Live output can pass on the first turn or fail; the deterministic two-turn
transcript applies only to the default scripted demo. The basic `--live` command
is separate from the bounded DeepSeek experiment below. Upstream SDK retries can occur inside one
worker call. A four-turn bound is not a dollar cap or wall-clock deadline.

## Live Claude Code experiment

`live_claude_loop.py` uses the signed-in Claude Code CLI, without reading or
extracting its credentials. It starts each worker in an empty temporary directory
with customizations and tools disabled and a fixed inventory system prompt.
It requests `claude-opus-5`, allows up to three worker calls, configures a $0.15
budget per CLI invocation, and has a 90-second subprocess timeout per call.
These are controls, not a guarantee that a provider request is cancelled or that
budget accounting cannot overshoot when the host stops waiting.

Run only if you intend to use your account's model allowance or API credits:

```sh
uv run --no-project python -m examples.live_claude_loop
```

The October 19, 2026 run passed the normal task in one turn. In the recovery
scenario, the host removed B200 from an otherwise correct first model response.
Acceptance rejected that delivered artifact, and the next real model call passed.
All three raw model responses were correct. This tests controller recovery with
real generation, not a spontaneous model failure or the causal value of feedback.
It does not test live tools or writes.

The CLI reported $0.011055 total list-price-equivalent cost; this is not a verified
subscription charge. Sanitized observations are saved in
`examples/live-results-2026-10-19.json`. Adapter tests mock subprocess calls and do
not spend credits. The DeepSeek adapters are separate from this CLI adapter.

## Live DeepSeek experiment

With `DEEPSEEK_API_KEY` exported, run the bounded experiment only if you intend to
spend API credits:

```sh
uv run python -m examples.live_deepseek_loop
```

This adapter calls DeepSeek directly through the project's OpenAI SDK, using the
same system prompt and controller as the CLI experiment. Unlike the original
upstream wrapper, it explicitly caps output at 128 tokens per call, sets a
30-second SDK timeout, and disables SDK retries. It admits at most three worker
calls. No tools are exposed. These controls do not establish a hard dollar cap
or guarantee that a timed-out request was never processed by the provider.

The October 19 run requested `deepseek-chat`; the API reported `deepseek-flash`.
The normal task passed in one turn. The recovery task rejected the host-injected
omission and passed on turn two. All three raw responses contained both correct
counts. The API reported 255 input and 42 output tokens in total; no current-price
cost estimate or invoice was verified. Observations are saved in
`examples/live-deepseek-results-2026-10-19.json`.

This verifies a small text-only integration with real generation, not natural
model failure, the causal value of feedback, or the separate upstream `chat()`
wrapper. Tests of the adapter use a fake client and never load credentials.

## Contract

`supervise(objective, worker, verify)` calls a synchronous worker with the
original objective and latest acceptance feedback. `Check` is the host's
verdict. `Outcome` has a reason and an in-memory attempt trace:

| Status | Meaning |
| --- | --- |
| `complete` | Current artifact passed the verifier |
| `limited` | All admitted turns used without passing |
| `stalled` | Identical rejected output repeated (default: twice) |
| `blocked` | Worker or verifier failed; inspect before retrying |
| `cancelled` | Cooperative cancellation observed between calls |

Completion on the last allowed turn still succeeds. A verifier outage never
counts as success. Worker exceptions do not trigger automatic retries because
an exception does not establish that an external effect never happened.
Exceptions become generic blocked outcomes here; a production host needs
private diagnostic logging and request/effect identities for reconciliation.

## Deliberate boundaries

- One synchronous owner; no distributed locking, steering revisions, or event replay.
- The controller does not preserve conversation history; each live `chat` is fresh.
- Exact repeated text is a crude stall guard, not a semantic progress measure.
- Cancellation is checked before work, after work, and after verification. It
  cannot interrupt a blocked synchronous call or undo completed effects.
- State and traces are in memory; restart loses them. Failed worker calls are
  not included as completed attempts.
- No time/token/spend enforcement, durable recovery, approval flow, or sandbox.
- A copied dict is a teaching boundary inside one process, not security
  isolation against arbitrary code with access to the host.

Tests exercise controller behavior with fakes. They do not establish model
accuracy, live compatibility, or safe unattended production use.
