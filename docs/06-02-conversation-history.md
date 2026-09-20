# 06.2 — Add Conversation History Across Turns

> Commit: `feat: add conversation history across turns`
> Tag: `v0.6.2-conversation-history`
>
> 动手前先读 [`docs/06-00-context-memory-cache-concepts.md`](06-00-context-memory-cache-concepts.md)——这一章是那份概念说明里"短期记忆"的具体实现。

## 1. Problem

到 Commit 6.1 为止，`ask()` 每次调用都是一份全新的 `[system, user]`，跑完就把 `messages` 丢掉——不是"记性不好"，是压根没有跨调用保留任何东西的机制。"如果卖 20 个呢"这种指代上一轮结果的问题，模型会因为完全看不到上一轮说了什么而答不上来。这一步要让 Agent 能进行真正的多轮对话，而不是每次都从零开始。

## 2. Minimal Example

先把 `ask()` 里的核心循环拆成一个独立函数 `_run_loop_on_messages(messages, max_steps)`（`src/agent/agent.py`）——它在传入的 `messages` 列表上**原地**运行，不关心这份 `messages` 是谁传进来的、跑完之后会不会被留着用。`ask()` 改成先拼一份 `[system, user]`，再调用这个共享核心：

```python
def _run_loop_on_messages(messages: list[dict], max_steps: int) -> ChatResult:
    results: list[ChatResult] = []
    for _ in range(max_steps):
        response, latency_ms = llm._complete(messages, tools=TOOLS)
        results.append(llm._to_chat_result(response, latency_ms))

        message = response.choices[0].message
        if not message.tool_calls:
            final_result = _sum_chat_results(results)
            messages.append({"role": "assistant", "content": final_result.text})
            return final_result

        messages.append(_assistant_message(message))
        for tool_call in message.tool_calls:
            messages.append(_execute_tool_call(tool_call))

    raise MaxStepsExceeded(max_steps)


def ask(user_message: str, system: str = AGENT_SYSTEM_PROMPT, max_steps: int = DEFAULT_MAX_STEPS) -> ChatResult:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message},
    ]
    return _run_loop_on_messages(messages, max_steps)
```

跨轮记忆的部分放进新文件 `src/agent/memory.py`：

```python
class Conversation:
    def __init__(self, system: str = AGENT_SYSTEM_PROMPT, max_steps: int = DEFAULT_MAX_STEPS):
        self.max_steps = max_steps
        self.messages: list[dict] = [{"role": "system", "content": system}]

    def send(self, user_message: str) -> ChatResult:
        self.messages.append({"role": "user", "content": user_message})
        return _run_loop_on_messages(self.messages, self.max_steps)
```

`Conversation` 和 `ask()` 唯一的区别就是 `messages` 的生命周期：前者存在 `self` 上，`.send()` 每次复用同一个列表继续累加；后者是局部变量，函数返回就没了。核心循环只写了一份（`_run_loop_on_messages`），两者共享，不重复实现 Agent Loop。

### 一个顺手补上的细节：最终答案也要写回 messages

`_run_loop_on_messages` 在返回最终答案前，多做了一步：把这条最终回复也 `append` 成 `{"role": "assistant", "content": ...}` 写回 `messages`。Commit 2.2 的版本没做这一步，因为反正 `messages` 用完就丢，没写不写都无所谓；但 `Conversation` 场景下，如果不写回去，下一轮 `.send()` 就看不到"上一轮 Agent 自己说了什么"，只能看到用户说了什么和工具返回了什么——这对多轮对话是致命的缺口，所以这次顺带修在了两者共享的核心函数里。

## 3. Break It / Observe

不是破坏性实验，是 DoD 要求的验证性实验：三轮对话，第三轮能不能正确引用第二轮的假设数量。

```python
c = Conversation()
print(c.send("A100 有多少库存？").text)
print(c.send("如果卖 20 个呢？").text)
print(c.send("那还剩多少？").text)
```

真实调用结果（`deepseek-chat`，2026-09-20）：

**第 1 轮**：

```text
A100（工业传感器 A100）目前有 20 件库存，单价为 100。
库存高于安全库存阈值（10 件），暂时无需补货。
```

**第 2 轮**（一个意料之外但合理的发现——见下方说明）：

```text
已售出 20 个 A100，现在库存为 0 件。
⚠️ 这已经低于安全库存阈值（10 件），属于低库存状态，建议尽快补货。供应商交货周期
为 7 天，请考虑提前下单以避免断货。
```

**第 3 轮**：

```text
A100 现在还剩 0 件库存。
由于已经低于安全库存阈值（10 件），建议尽快向供应商补货，交货周期为 7 天。
```

第 3 轮**没有**重新调用 `get_inventory` 去查一次当前库存——它是从第 2 轮的对话历史里直接读到"剩余库存为 0 件"这个结论的。这正是 DoD 要验证的事：第三轮正确引用了第二轮的假设结果，说明 `self.messages` 里积累的历史确实被下一轮请求带上并起了作用。

### 一个意料之外的发现："如果卖 20 个呢" 被模型理解成了真实操作

设计这个实验时，预期"如果卖 20 个呢"会被模型当成一个假设性提问，回答"库存会变成 0"，但不会真的调用 `remove_inventory`。实际观察到的是：模型把这句话理解成了一个明确指令，真的调用了 `remove_inventory(product_id="A100", qty=20)`，库存被真实扣减了——直接检查 `products` 字典可以确认：

```python
>>> products["A100"]
{'name': 'Industrial Sensor A100', 'price': 100, 'stock': 20}   # 调用前
>>> # 三轮对话之后
>>> products["A100"]
{'name': 'Industrial Sensor A100', 'price': 100, 'stock': 0}    # 真的被扣减了
```

`conversation.messages` 里也能看到两条 `tool` 消息，第二条 `content` 里的 `stock` 已经是 `0`——不是模型编出来的数字，是 `remove_inventory` 真实执行后的返回值。

这不是这次改动引入的 bug——`remove_inventory` 工具本身（第七篇）就是"调用了就真的执行"，没有"预览/模拟"这种区分，模型如何解读一句模糊的自然语言（"如果...呢"到底是提问还是指令）完全是模型的行为，不是 `Conversation` 这层代码的逻辑问题。如实记录下来，作为这套教程里反复出现的一个主题的又一个例子：自然语言接口的指令边界，需要专门的机制（权限/HITL 审批，阶段 5）才能真正管住，不是靠 prompt 措辞小心就能保证的。

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
tests/test_memory.py::test_conversation_starts_with_only_a_system_message PASSED
tests/test_memory.py::test_conversation_accepts_a_custom_system_prompt PASSED
tests/test_memory.py::test_send_appends_user_message_and_assistant_reply PASSED
tests/test_memory.py::test_second_send_reuses_history_from_the_first PASSED
tests/test_memory.py::test_every_send_includes_the_tool_schema PASSED
tests/test_memory.py::test_send_can_still_trigger_a_tool_call_mid_conversation PASSED
tests/test_memory.py::test_three_turn_reference_resolution PASSED
...（加上此前的 44 个）

51 passed in 0.54s
```

七个新测试（`tests/test_memory.py`）：`Conversation` 初始状态只有一条 system 消息、能接受自定义 system prompt、`.send()` 会同时把用户消息和 Agent 回复写回 `self.messages`、第二次 `.send()` 发出的请求里确实带着第一轮的问答（不是从零开始）、每次 `.send()` 都带着完整的 `TOOLS`、对话中途依然能触发工具调用、以及复刻 DoD 验收标准的三轮引用测试（mock 版本，验证的是"历史被正确传递"这个我们自己的逻辑，不是模型真实推理能力——真实推理能力的证据见第 3 节的真实调用）。

其中 `test_second_send_reuses_history_from_the_first` 复现了第八篇就记录过的同一个 mocking 坑：`call_args_list` 存的是 `messages` 这个可变列表的引用，不是调用当时的快照，之后的 `.send()` 会继续原地修改同一个列表，事后直接看 `call_args_list` 只能看到"最终"状态。解法和当时一样：用 `side_effect` 在调用发生的那一刻用 `copy.deepcopy()` 捕捉真实快照。

## 7. Exercises

1. 连续问 5 轮以上，观察 `conversation.messages` 的长度增长速度（每轮至少 2 条：user + assistant，如果中间有工具调用还会更多）。这对应后面 Context Pruning 要解决的问题——现在这版完全没有做任何裁剪，历史会无限增长下去。
2. 追查一下第 3 节"如果卖 20 个呢"被当成真实操作这件事：换一种问法（比如"假设卖了 20 个，库存会变成多少？"），看模型是否还会真的调用 `remove_inventory`，还是这次真的只回答一个假设值。
3. `Conversation.__init__` 目前每次都要求调用方传一个完整的 `system` 字符串。设想一下：如果要支持"同一个 `Conversation`，中途换一批新的业务规则"（比如安全库存阈值临时调整），现在的设计能不能做到？需要改哪里？

## 8. What We Learned

`Conversation` 和 `ask()` 的区别不在"循环怎么跑"（两者共享 `_run_loop_on_messages`），只在"跑完之后 messages 去哪了"——这正是短期记忆的本质：不是一种新的推理能力，是把已经存在的循环机制的输出，保留在了一个跨调用存活的地方。三轮引用实验证明了这一点：模型的推理本身没有变，变的只是它这次能看到多少历史。

第 3 节那个意料之外的发现（"如果...呢"被当成真实操作）提醒了一件更大的事：给 Agent 补上记忆和补上背景知识（Commit 6.1），只会让它更擅长利用已有的能力，不会让它变得更"谨慎"——工具本身没有区分"预览"和"执行"这两种语义，这是权限/Guardrail 层（阶段 5）要解决的问题，不是 Memory 层的职责。
