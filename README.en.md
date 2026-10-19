# Agent From Zero

[简体中文](README.md) | **English**

Build an inventory and quote assistant in Python, starting with one model call. Work through tools, agent loops, errors, context, and acceptance checks by adding one requirement at a time, then inspecting minimal code, failure experiments, and tests.

## Original course and this fork

The original project and Chinese-language course are by **Bange (半格记录 / sleepworm)**:

- [Original repository](https://github.com/sleepworm/agent-from-zero)
- [Course overview](https://www.bange.life/tech/agent-course-01-overview)
- [Complete article collection](https://www.bange.life/tech/collections/agent-from-zero)

This is [Haochuan Zhang's fork](https://github.com/lunarmoon26/agent-from-zero). It preserves the upstream implementation and teaching notes while adding a separate loop-engineering example, offline tests, and read-only DeepSeek / Claude Code experiments. The extensions are not part of Bange's original course. This English README does **not** imply that the Chinese chapters in `docs/` have been translated.

As checked on October 19, 2026, Bange's web collection contains an overview and seven lessons. The repository continues beyond those articles into loops, step limits, and conversation history. The two indexes below distinguish published articles from later repository chapters.

## Quick start: run offline tests first

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/lunarmoon26/agent-from-zero.git
cd agent-from-zero
uv sync --extra dev
uv run python -m pytest -q
```

The current fork has **79 offline tests**. They require no API key and make no real model requests. Passing tests establish behavior under controlled inputs, not general model accuracy or production reliability.

### Optional: call DeepSeek

Real requests consume API credits. Copy the configuration example, edit it locally, and never commit credentials:

```sh
cp .env.example .env
# Edit .env to set DEEPSEEK_API_KEY, then load it explicitly:
uv run --env-file .env python -m src.agent.llm "Hello"
```

The application does not automatically read `.env`; `uv run --env-file .env` loads it here. Alternatively, export the environment variable yourself and omit `--env-file`.

## Bange's seven published lessons

Start with the [course overview](https://www.bange.life/tech/agent-course-01-overview), then follow the table. Article labels below are English descriptions; the linked original articles and repository chapters are in Chinese.

| Lesson | Bange's article | Repository chapter | Code tag |
| --- | --- | --- | --- |
| 01 · A minimal model call | [Make the first call](https://www.bange.life/tech/agent-course-01-part-01-basic-chat) | [00.1](docs/00-01-basic-chat.md) | `v0.0.1-basic-chat` |
| 02 · Error classification | [What the application should see when a call fails](https://www.bange.life/tech/agent-course-01-part-02-error-handling) | [00.2](docs/00-02-error-handling.md) | `v0.0.2-error-handling` |
| 03 · Tokens, latency, and cost | [Measure a model call](https://www.bange.life/tech/agent-course-01-part-03-cost-latency-tracking) | [00.3](docs/00-03-cost-latency-tracking.md) | `v0.0.3-cost-latency-tracking` |
| 04 · Tool contracts | [Describe an inventory tool to the model](https://www.bange.life/tech/agent-course-01-part-04-inventory-tool-schema) | [01.1](docs/01-01-inventory-tool-schema.md) | `v0.1.1-inventory-tool-schema` |
| 05 · One tool round trip | [The model requests; the program executes](https://www.bange.life/tech/agent-course-01-part-05-single-tool-call) | [01.2](docs/01-02-single-tool-call.md) | `v0.1.2-single-tool-call` |
| 06 · Unknown products | [Will the model invent a missing stock count?](https://www.bange.life/tech/agent-course-01-part-06-hallucination-guard) | [01.3](docs/01-03-hallucination-guard.md) | `v0.1.3-hallucination-guard` |
| 07 · A multi-tool registry | [Add stock-changing tools](https://www.bange.life/tech/agent-course-01-part-07-multi-tool-registry) | [02.1](docs/02-01-multi-tool-registry.md) | `v0.2.1-multi-tool-registry` |

## Later chapters already in the repository

These documents exist in the repository but are not in the eight-page web collection above. Numbering follows upstream; gaps do not imply that the missing phases have been implemented.

| Topic | Repository chapter | Code tag |
| --- | --- | --- |
| Agent loop and multiple tool calls in one response | [02.2 · Agent Loop](docs/02-02-agent-loop.md) | `v0.2.2-agent-loop` |
| Maximum-step guard | [02.3 · Max-Step Guard](docs/02-03-max-step-guard.md) | `v0.2.3-max-step-guard` |
| Context, memory, and cache concepts | [06.0 · Concepts](docs/06-00-context-memory-cache-concepts.md) | No separate tag |
| Business context injection | [06.1 · Business Context](docs/06-01-business-context.md) | `v0.6.1-business-context` |
| Cross-turn conversation history | [06.2 · Conversation History](docs/06-02-conversation-history.md) | `v0.6.2-conversation-history` |

The additional `v0.2.2.1-heterogeneous-tool-calls` tag covers heterogeneous tool-call tests discussed in chapter 02.2. The current implementation executes calls from the same response sequentially, not concurrently.

Use `git switch --detach <tag>` to inspect a stage and `git diff <tag1> <tag2>` to compare stages. Earlier tags do not include later features, fork extensions, or all 79 tests.

## Fork extension: loop engineering

[Loop Engineering From First Principles](https://blog.haochuanz.net/posts/loop-engineering-from-first-principles) starts with an answer that omits a requested product, then builds acceptance checks, feedback, continuation, and stopping rules. The [companion guide](examples/README-loop-engineering.md) documents the full examples and their limits.

```sh
# Deterministic offline demo: no model calls.
uv run python -m examples.goal_loop

# Optional paid DeepSeek experiment: at most three requests, no tools.
uv run --env-file .env python -m examples.live_deepseek_loop

# Optional Claude Code experiment: requires an authenticated Claude Code CLI.
uv run python -m examples.live_claude_loop
```

Live observations: [DeepSeek](examples/live-deepseek-results-2026-10-19.json) · [Claude Code](examples/live-results-2026-10-19.json). Both passed a normal task and a controlled recovery scenario. All raw model answers were correct; the host deliberately removed a product after generation. This exercises continuation, not spontaneous model failure, the causal value of feedback, or production reliability.

## Current behavior and limits

- `chat()` makes one model call and returns text, token usage, latency, and a cost estimate based on a historical price snapshot. The estimate is not a current invoice.
- `ask()` runs a step-bounded tool loop. `Conversation.send()` retains cross-turn history in memory, without durable recovery.
- Inventory tools can query, add, and remove stock. Writes mutate in-memory data without positive-quantity validation, approval, idempotency, or transactions. Do not connect them to real business data or run them unattended.
- The fork's outer supervisor uses a separate read-only teaching snapshot. Turn limits are not time, token, or dollar limits; cooperative cancellation cannot interrupt blocked calls or undo effects.
- This is an evolving teaching project. Roadmap goals such as quoting, recovery, evaluation, and production deployment should not be read as fully implemented features.

## Repository layout

```text
agent-from-zero/
├── src/
│   ├── tools/             # In-memory inventory, tool functions, and schemas
│   └── agent/             # Model client, tool loop, and conversation history
├── tests/                 # Upstream and fork offline tests
├── docs/                  # Original Chinese teaching chapters
├── examples/              # Fork supervisor, read-only experiments, sanitized results
├── TODO.md                # Open research questions
└── failure_taxonomy.md    # Failure classification notes
```

## Attribution and licensing

The original course, upstream code, and teaching chapters belong to their respective author; fork extensions and explanations identify their sources separately. No license declaration was found in the inspected upstream version. Do not assume MIT or another open-source license. This fork does not relicense upstream content.
