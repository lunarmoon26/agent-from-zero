# Agent From Zero

**简体中文** | [English](README.en.md)

从一次模型调用开始，用 Python 逐步搭建库存与报价助手，理解工具调用、Agent Loop、错误处理、上下文和验收。每一步先提出一个具体需求，再用最小代码、失败实验和测试说明机制的边界。

## 原教程与这个 Fork

原项目及中文教程由 **Bange（半格记录 / sleepworm）**创作：

- [原始仓库](https://github.com/sleepworm/agent-from-zero)
- [课程总览](https://www.bange.life/tech/agent-course-01-overview)
- [完整文章合集](https://www.bange.life/tech/collections/agent-from-zero)

这里是 [Haochuan Zhang 维护的 Fork](https://github.com/lunarmoon26/agent-from-zero)。在保留上游实现和教材的基础上，增加独立的 Loop Engineering 示例、离线测试，以及 DeepSeek / Claude Code 的只读实验记录。扩展示例不属于 Bange 的原课程；英文 README 也不意味着 `docs/` 中的中文教材已经全部翻译。

截至 2026-10-19，Bange 的网页合集包含课程总览和七篇正文，仓库中的教材已经继续推进到循环、步数限制和会话历史。下面将网页文章与后续本地章节分开列出，避免把尚未发布的网页文章当成已有内容。

## 快速开始：先跑离线测试

需要 Python 3.11+ 和 [uv](https://docs.astral.sh/uv/)。

```sh
git clone https://github.com/lunarmoon26/agent-from-zero.git
cd agent-from-zero
uv sync --extra dev
uv run python -m pytest -q
```

当前 Fork 的完整离线测试共 **79 项**，不需要 API Key，也不会发起真实模型请求。测试通过说明受控输入下的程序行为符合断言，不代表真实模型或生产系统的普遍可靠性。

### 可选：调用 DeepSeek

真实请求会消耗 API 额度。先复制配置示例，在本地填入自己的 Key，不要提交凭证：

```sh
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY，然后显式加载它：
uv run --env-file .env python -m src.agent.llm "Hello"
```

应用代码不会自动读取 `.env`；这里由 `uv run --env-file .env` 加载。也可以自行导出环境变量，再省略 `--env-file`。

## Bange 已发布的七篇教程

从[课程总览](https://www.bange.life/tech/agent-course-01-overview)开始，再按下表阅读。网页正文包含背景和实验过程；本地文档与 Tag 便于对照对应阶段的代码。

| 篇目 | Bange 原文 | 仓库教材 | 对应 Tag |
| --- | --- | --- | --- |
| 01 · 最小模型调用 | [先让程序和模型说上第一句话](https://www.bange.life/tech/agent-course-01-part-01-basic-chat) | [00.1](docs/00-01-basic-chat.md) | `v0.0.1-basic-chat` |
| 02 · 错误分类 | [模型调用失败时，程序应该看见什么](https://www.bange.life/tech/agent-course-01-part-02-error-handling) | [00.2](docs/00-02-error-handling.md) | `v0.0.2-error-handling` |
| 03 · Token、延迟与成本 | [一次回答到底花了多少时间和钱](https://www.bange.life/tech/agent-course-01-part-03-cost-latency-tracking) | [00.3](docs/00-03-cost-latency-tracking.md) | `v0.0.3-cost-latency-tracking` |
| 04 · 工具契约 | [先给模型一份查库存的说明书](https://www.bange.life/tech/agent-course-01-part-04-inventory-tool-schema) | [01.1](docs/01-01-inventory-tool-schema.md) | `v0.1.1-inventory-tool-schema` |
| 05 · 一次工具往返 | [模型提出调用，程序执行工具](https://www.bange.life/tech/agent-course-01-part-05-single-tool-call) | [01.2](docs/01-02-single-tool-call.md) | `v0.1.2-single-tool-call` |
| 06 · 未知商品 | [查不到商品时，模型会不会编一个数字](https://www.bange.life/tech/agent-course-01-part-06-hallucination-guard) | [01.3](docs/01-03-hallucination-guard.md) | `v0.1.3-hallucination-guard` |
| 07 · 多工具注册表 | [让销售助手学会入库和出库](https://www.bange.life/tech/agent-course-01-part-07-multi-tool-registry) | [02.1](docs/02-01-multi-tool-registry.md) | `v0.2.1-multi-tool-registry` |

## 仓库中已有的后续章节

这些文档存在于仓库中，但不在上述八篇网页合集里。章节编号沿用上游，编号空缺不表示相应功能已经实现。

| 主题 | 仓库教材 | 对应 Tag |
| --- | --- | --- |
| Agent Loop 与同一响应中的多个工具调用 | [02.2 · Agent Loop](docs/02-02-agent-loop.md) | `v0.2.2-agent-loop` |
| 最大步数保护 | [02.3 · Max-Step Guard](docs/02-03-max-step-guard.md) | `v0.2.3-max-step-guard` |
| Context、Memory 与 Cache 概念 | [06.0 · 概念说明](docs/06-00-context-memory-cache-concepts.md) | 无独立 Tag |
| 注入业务上下文 | [06.1 · Business Context](docs/06-01-business-context.md) | `v0.6.1-business-context` |
| 跨轮会话历史 | [06.2 · Conversation History](docs/06-02-conversation-history.md) | `v0.6.2-conversation-history` |

`v0.2.2.1-heterogeneous-tool-calls` 另有异构工具调用测试，对应说明已补入 02.2。当前实现按顺序执行同一响应中的工具调用，并非并发执行。

可以用 `git switch --detach <tag>` 查看某一步，用 `git diff <tag1> <tag2>` 对比变化。较早的 Tag 不包含后来的功能、Fork 扩展或 79 项测试。

## Fork 扩展：Loop Engineering

[Loop Engineering From First Principles](https://blog.haochuanz.net/posts/loop-engineering-from-first-principles) 从一个遗漏商品的回答出发，介绍目标验收、反馈、继续执行和停止条件。配套的[示例说明](examples/README-loop-engineering.md)包含完整运行方式和边界。

```sh
# 确定性的离线演示，不调用模型：
uv run python -m examples.goal_loop

# 可选的付费 DeepSeek 实验，最多三次请求，不开放工具：
uv run --env-file .env python -m examples.live_deepseek_loop

# 可选的 Claude Code 实验，需要已登录的 Claude Code CLI：
uv run python -m examples.live_claude_loop
```

真实实验记录：[DeepSeek](examples/live-deepseek-results-2026-10-19.json) · [Claude Code](examples/live-results-2026-10-19.json)。两者都通过了正常任务与受控故障恢复场景；所有原始模型回答均正确，缺失商品由宿主在生成后主动删除。这验证了继续执行链路，不证明模型自然出错、反馈的因果效果或生产可靠性。

## 当前实现与使用边界

- `chat()` 进行单次模型调用，返回文本、Token 用量、延迟和历史价格下的估算成本；估算值不是当前账单。
- `ask()` 执行带步数限制的工具循环；`Conversation.send()` 在内存中保留跨轮历史，不提供持久化恢复。
- 库存工具包含查询、入库和出库。写工具直接修改内存数据，尚无正数量校验、权限审批、幂等或事务保护，不应连接真实业务数据或无人值守运行。
- Fork 的外层监督器使用独立的只读教学快照。轮数上限不等于时间、Token 或金额上限；合作式取消不能中断阻塞调用或撤销已发生的副作用。
- 项目仍在教学演进中，报价、恢复、评估与生产部署等路线目标不应被当成全部已经实现的功能。

## 目录结构

```text
agent-from-zero/
├── src/
│   ├── tools/             # 内存库存数据、工具实现和 Schema
│   └── agent/             # 模型客户端、工具循环和会话历史
├── tests/                 # 上游和 Fork 的离线测试
├── docs/                  # 上游中文教材
├── examples/              # Fork 的监督器、只读实验和脱敏结果
├── TODO.md                # 尚待研究的问题
└── failure_taxonomy.md    # 失败分类记录
```

## 署名与许可

原教程、上游代码和教材归原作者；Fork 扩展与说明另行标注来源。在当前检视的上游版本中未发现许可证声明，不应假定项目采用 MIT 或其他开源许可证。本 Fork 不为上游内容重新指定许可。
