# 02.3 — Add Max-Step Guard Against Infinite Loops

> Commit: `feat: add max-step guard against infinite loops`
> Tag: `v0.2.3-max-step-guard`

## 1. Problem

Commit 2.2 的 `while True` 循环没有任何退出条件保护——这是当时刻意留下的问题。如果模型陷入"总觉得还需要再调用一次工具"的状态（真实世界里会发生：工具结果含糊、模型对同一个问题反复确认、甚至工具本身返回的数据暗示"还差点什么"），程序会真的死循环，永远不返回。这一步要给循环加一个硬性上限。

## 2. Minimal Example

`while True` 改成 `for _ in range(max_steps)`，跑满还没结束就抛异常：

```python
DEFAULT_MAX_STEPS = 10


class LoopError(Exception):
    """Agent Loop 本身的控制流失败——不是模型调用失败（那是 llm.py 的
    ChatError 家族），是"循环这个机制自己出了问题"，两者不共用基类。"""


class MaxStepsExceeded(LoopError):
    def __init__(self, max_steps: int):
        super().__init__(f"Agent loop did not reach a final answer within {max_steps} steps.")
        self.max_steps = max_steps


def ask(user_message: str, system: str = llm.DEFAULT_SYSTEM, max_steps: int = DEFAULT_MAX_STEPS) -> ChatResult:
    messages = [...]

    results: list[ChatResult] = []
    for _ in range(max_steps):
        response, latency_ms = llm._complete(messages, tools=TOOLS)
        results.append(llm._to_chat_result(response, latency_ms))

        message = response.choices[0].message
        if not message.tool_calls:
            return _sum_chat_results(results)

        messages.append(_assistant_message(message))
        for tool_call in message.tool_calls:
            messages.append(_execute_tool_call(tool_call))

    raise MaxStepsExceeded(max_steps)
```

两个设计决定：

- **抛异常，不是返回一个"未完成"的字符串**。原因和 Commit 0.2 的 `ChatError` 家族一致——一个字符串结果和一个正常的最终答案长得一样，调用方很容易忘记检查"这是不是真的答案还是一个失败提示"；异常会强制调用方要么处理、要么让它继续往上冒，不会被悄悄当成正常结果使用。
- **`LoopError` 和 `ChatError` 是两条不同的失败家族，不共用基类**。`ChatError`（Commit 0.2）描述的是"这次 API 调用本身失败了"（认证/网络/请求格式）；`MaxStepsExceeded` 描述的是"API 调用都成功了，但循环控制流自己失控了"——性质不同，以后 Phase 5（Retry）大概率要对它们采取完全不同的策略：`NetworkFailure` 值得重试，`MaxStepsExceeded` 重试大概率还是会撞到同样的上限，不该被同一套重试逻辑不加区分地处理。

## 3. Break It

**这一次的 Break It 故意不打真实 API**。要验证的是"我们自己的循环控制代码在跑满 `max_steps` 后会不会正确停下来"——这是我们自己写的逻辑，不是模型的真实行为，用 mock 反复模拟"模型永远还要再调用一次工具"完全够用，而且免费。真的花钱让模型在生产环境里空转 10 次以上，不会比 mock 更有说服力，只会更贵（这条原则在这个项目里越来越明确：**真实调用用来验证模型的真实行为，mock 用来验证我们自己的代码逻辑，两者不能互相替代，但也不要用真实调用去验证本该用 mock 验证的东西**）。

```python
def _always_wants_another_tool_call(self, **_kwargs):
    return _response(
        content=None,
        tool_calls=[_tool_call("call_loop", "get_inventory", {"product_id": "A100"})],
    )

def test_infinite_loop_is_stopped_at_max_steps(self, monkeypatch):
    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = self._always_wants_another_tool_call
    monkeypatch.setattr(llm, "get_client", lambda: fake_client)

    with pytest.raises(agent.MaxStepsExceeded) as exc_info:
        agent.ask("随便问点什么", max_steps=10)

    assert exc_info.value.max_steps == 10
    assert fake_client.chat.completions.create.call_count == 10  # 跑满 10 步就停手，不多不少
```

真实结果：`MaxStepsExceeded` 被正确抛出，`create()` 被调用了精确 10 次（不是 9 次也不是 11 次），第 11 次调用真的没有发生。

## 4. Implement the Fix

见第 2 节。

## 5. Introduce Industrial Solution

不适用，第一次框架对比在 Phase 17。

## 6. Evaluate

```bash
pytest -v
```

实际输出：

```text
tests/test_agent.py::TestMaxStepsGuard::test_normal_task_is_not_affected_by_the_guard PASSED
tests/test_agent.py::TestMaxStepsGuard::test_infinite_loop_is_stopped_at_max_steps PASSED
tests/test_agent.py::TestMaxStepsGuard::test_max_steps_exceeded_message_is_readable PASSED
tests/test_agent.py::TestMaxStepsGuard::test_max_steps_is_configurable PASSED
tests/test_agent.py::TestMaxStepsGuard::test_max_steps_exceeded_is_a_loop_error_not_a_chat_error PASSED
...（加上此前的 35 个）

40 passed in 0.64s
```

DoD 明确要求两件事都要有测试：**正常任务不会被 guard 误伤**（`test_normal_task_is_not_affected_by_the_guard`，两轮就结束的任务，`max_steps=10` 完全不影响它）；**死循环任务会被正确拦截并给出可读的错误信息**（`test_infinite_loop_is_stopped_at_max_steps` + `test_max_steps_exceeded_message_is_readable`）。这两条缺一不可——只测"能拦住死循环"，没测"不会误伤正常任务"，guard 可能被设计得太激进（比如 `max_steps=1`）也能通过测试。

另外补了一次真实调用，确认默认的 `max_steps=10` 不会影响任何正常场景：

```bash
python3 -c "from src.agent.agent import ask; print(ask('A100 还有多少库存？').text)"
```

真实输出（`deepseek-chat`，2026-09-20）：

```text
A100（工业传感器 A100）目前库存为 20 件，单价 100。
```

正常返回真实答案，一次工具调用就结束，远没有触达 10 步上限。

## 7. Exercises

1. 把 `max_steps` 设成 `1`，重新跑一次"A100 库存够不够给 B200 客户凑单"这个需要两次工具调用的真实问题（Commit 2.2 验证过的场景），观察它会不会被误伤——这是"guard 设置得太激进"这种新失败模式的一个具体例子，`max_steps` 这个参数本身也需要根据真实场景校准，不是随便定一个数字。
2. `MaxStepsExceeded` 现在只携带 `max_steps` 这一个信息。如果要让调用方能看到"跑到第几步的时候在做什么"（比如最后一次工具调用是什么），需要在异常里加什么字段？这和 Phase 8（Tracing）要解决的问题有什么关系？
3. 想一个"最大步数"以外的死循环判定方式（比如"连续两次调用了完全相同的工具和参数"），思考它能不能比单纯数步数更早发现问题，以及会不会有误伤正常任务的风险。

## 8. What We Learned

给循环加边界这件事，代码改动很小（`while True` 变成 `for _ in range(max_steps)`，加一个异常类），但验证这件事需要两个方向都测到，缺一不可；`LoopError` 和 `ChatError` 分成两条独立的异常家族，是在为 Phase 5 的 Retry 逻辑预留"不同失败类型需要不同处理策略"的空间；以及这一章第一次明确说出了"什么时候该花钱调真实 API、什么时候该用 mock"的判断标准——验证我们自己的控制流逻辑，永远优先选 mock。
