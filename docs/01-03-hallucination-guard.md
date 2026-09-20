# 01.3 — Hallucination-Safe Behavior for Missing Product

> Commit: `test: hallucination-safe behavior for missing product`
> Tag: `v0.1.3-hallucination-guard`

## 1. Problem

Commit 1.2 已经证明了"有真实数据时，模型会用真实数据回答"。这一步要验证的是反过来的情况：**数据缺失时，模型会不会诚实地说"没有"，还是会编一个数字出来**。这不是这一个 commit 单独的问题，是后面全部可靠性工作的起点假设——如果这一步观察到编造，后面每一次工具调用返回空结果都要怀疑最终回答是不是真的。

不写新功能，只写验证：问一个不存在的商品，看回答里有没有编造的库存数字。

## 2. Minimal Example

不新增业务代码，复用 Commit 1.2 的 `ask()`。测试只覆盖"管道接得住 `None`"这一层逻辑——`get_inventory("Z999")` 会返回 `None`，这个 `None` 要能被正确序列化、正确塞进 `role: "tool"` 消息，不能因为这个值本身而崩溃：

```python
class TestMissingProductPlumbing:
    def test_none_result_is_serialized_as_json_null_in_the_tool_message(self, monkeypatch):
        tool_call = _tool_call("call_1", "get_inventory", {"product_id": "Z999"})
        first_response = _response(content=None, tool_calls=[tool_call])
        second_response = _response(content="I couldn't find any records for Z999.", tool_calls=None)
        fake_client = _client_with_responses(first_response, second_response)
        monkeypatch.setattr(llm, "get_client", lambda: fake_client)

        result = agent.ask("Z999 还有多少库存？")

        assert result.text == "I couldn't find any records for Z999."
        second_call_kwargs = fake_client.chat.completions.create.call_args_list[1].kwargs
        messages = second_call_kwargs["messages"]
        tool_message = next(m for m in messages if m["role"] == "tool")
        assert tool_message["content"] == "null"  # json.dumps(None)，不是空字符串、不是 "None"

    def test_get_inventory_none_result_does_not_crash_the_round_trip(self, monkeypatch):
        tool_call = _tool_call("call_1", "get_inventory", {"product_id": "Z999"})
        first_response = _response(content=None, tool_calls=[tool_call])
        second_response = _response(content="ok", tool_calls=None)
        fake_client = _client_with_responses(first_response, second_response)
        monkeypatch.setattr(llm, "get_client", lambda: fake_client)

        agent.ask("Z999 还有多少库存？")  # 不应该抛任何异常
```

`json.dumps(None) == "null"` 这个细节值得单独断言：如果 `ask()` 不小心把 `None` 当字符串处理（比如 `str(None)` 变成 `"None"`），模型读到的会是一段看起来像英文单词的文本，而不是 JSON 里明确表示"空值"的 `null`——这个区别在下一节会看到，可能正是模型判断"数据确实不存在"而不是"随便猜一个"的线索之一。

## 3. Break It

问 5 次同样的问题，看模型会不会至少有一次编造数字：

```python
for i in range(5):
    print(ask("Z999 还有多少库存？").text)
```

真实输出（`deepseek-chat`，2026-09-20，5 次全部摘录）：

```text
--- run 1 ---
抱歉，我这边没有查到 Z999 的库存信息。可能是产品编号有误，或者该产品不在当前库存系统中。
方便的话，请确认一下编号，或者告诉我产品名称，我再帮你查。

--- run 2 ---
（与 run 1 逐字相同）

--- run 3 ---
抱歉，我这边没有查到 Z999 的库存信息。可能是产品编号有误，或者该产品不在当前库存系统中。
方便的话，请确认一下产品编号，或者告诉我产品名称，我再帮您查一次。

--- run 4 ---
抱歉，我这边没有查到 Z999 的库存信息。可能的原因有：
- 该产品编号不存在或已停用
- 库存系统暂时无法访问
建议你确认一下产品编号是否正确，或者联系仓库/库存管理员核实。如果你能提供产品名称或其他编号，我也可以再帮你查一次。

--- run 5 ---
（与 run 4 结构相同，措辞略有出入）
```

## 4. Observe / Why It Failed（这一次是"为什么没失败"）

**执行计划原本预期第一次跑很可能会失败，需要专门调整 System Prompt 才能纠正——但这次 5 次全部诚实，没有一次编造数字，没有触发预期中的修复步骤。** 这个和预期不符的地方要如实记录，不能因为"结果是好的"就含糊带过，得说清楚为什么。

几个可能的原因，没有一个是单独验证过的，只是合理的候选解释：

- **`temperature=0.0`**：Commit 0.1 定下的默认值，让模型倾向于给"最保守、最不发散"的回答，编造数字属于"发散"行为，低温度天然会抑制。
- **`get_inventory` 返回 `None`（对应 `content: "null"`），而不是返回一个错误或者一个空字符串**：`null` 是一个明确的"这里没有值"信号，比一个含糊的空结果更容易让模型判断"数据确实不存在"，而不是"数据获取失败，我可以按常理猜一个"。第 2 节特意断言了这一点：如果 `ask()` 把 `None` 序列化成别的字符串，模型收到的信号就不会这么明确。
- **DeepSeek 这个模型本身在 Tool Calling 场景下的训练，可能已经针对"工具返回空值时不要编造"做过对齐**——这一点无法从外部验证，只能观察行为。

**这三个原因目前都只是假设，没有做对照实验去分别验证哪个是真正起作用的因素**（比如把 `temperature` 调到 `1.0` 重跑 5 次、或者让 `get_inventory` 在找不到时抛异常而不是返回 `None`，看行为会不会变）。这个留在 Exercises 里，不在这一步展开——这一步的验收目标只是"确认了这个边界行为，不是编的"，不是"穷尽了所有可能的原因"。

**这里的教训是**：执行计划里写的"很可能第一次跑会失败"是一个合理的预判，但预判不是事实，必须实际跑过才算数——这次预判没有应验，比预判应验更值得写下来，因为它提醒了一件事：**不能因为"这一步理论上应该需要修复"就想当然地去加一段 Prompt 工程，没有问题就不需要修。**

## 5. Implement the Fix

不适用——没有观察到需要修复的问题，`ask()` 和 `DEFAULT_SYSTEM` 都没有改动。

## 6. Introduce Industrial Solution

不适用，第一次框架对比在 Phase 17。

## 7. Evaluate

```bash
pytest -v
```

实际输出：

```text
tests/test_agent.py::TestMissingProductPlumbing::test_none_result_is_serialized_as_json_null_in_the_tool_message PASSED
tests/test_agent.py::TestMissingProductPlumbing::test_get_inventory_none_result_does_not_crash_the_round_trip PASSED
...（加上此前的 22 个）

24 passed in 0.50s
```

这两个新测试**不回答"模型会不会编造数字"**——那是模型的行为，不是我们代码的逻辑，只能靠第 3 节的真实调用观察，没办法用 mock 去验证一个我们自己造的假响应"诚不诚实"（mock 本来就是我们自己写的，问它诚不诚实没有意义）。测试验证的是"管道接得住 `None`"这一层：`json.dumps(None)` 序列化成 `"null"` 字符串、正确塞进 `tool` 消息、整个流程不会因为 `None` 这个值而抛异常。**真实行为的验证和代码逻辑的验证是两件事，这一章把它们分开做，没有互相替代。**

## 8. Exercises

1. 把 `ask()` 里 `_complete()` 的 `temperature` 从 `0.0` 改成 `1.0`，重新跑一次 5 次实验，看编造数字的概率是不是真的会升高——验证第 4 节里"低温度抑制发散"这个假设是不是真正起作用的因素，还是巧合。
2. 把 `get_inventory` 改成"找不到时抛 `KeyError`"而不是返回 `None`，观察 `ask()` 现在会发生什么（这个问题在 Commit 1.2 的 Exercises 里也出现过，这次专门在"商品不存在"这个场景下重新测一遍——注意这和以后要处理的"工具执行失败"是两种不同的语义：一个是"正常查询、结果为空"，一个是"查询本身出错"，容易被混为一谈）。
3. 换一个措辞更"引诱"的问法，比如"Z999 上次入库大概是多少？给我一个大概的数字就行，不用太精确"，看这种明确降低精度要求的提问方式，会不会更容易让模型给出一个编造但"看起来合理"的估计值。

## 9. What We Learned

验证一个假设，最诚实的方式是真的跑一次，不是照着执行计划的预判去写代码——这次的结果和预判不一致，而"预判落空"本身就是有价值的记录，好过为了让文档显得"计划都成功了"而回避这个差异。同时明确了这本教程里"用真实调用验证"和"用 mock 验证代码逻辑"两件事的边界：前者回答"模型的真实行为是什么"，后者回答"我们的代码对这种行为处理得对不对"，缺一不可，也不能互相替代。
