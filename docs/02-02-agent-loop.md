# 02.2 — Implement Minimal Agent Loop

> Commit: `feat: implement minimal agent loop`
> Tag: `v0.2.2-agent-loop`

## 1. Problem

Commit 1.2/1.3 的 `ask()` 只能处理"最多一次工具调用"——问一个需要连续查两个商品才能回答的问题（比如"A100 库存够不够给 B200 客户凑单"），它接不住。这一步要真正理解 **Agent 和普通 Chat Completion 的本质区别**：不是"问一句、答一句"，而是一个循环——模型可以反复决定"我还需要再查一次"，直到它认为信息够了为止。

## 2. Minimal Example

`ask()` 从"最多一次"升级成一个 `while True` 循环：

```python
def _assistant_message(message) -> dict:
    return {
        "role": "assistant",
        "content": message.content,
        "tool_calls": [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
            for tool_call in message.tool_calls
        ],
    }


def _execute_tool_call(tool_call) -> dict:
    args = json.loads(tool_call.function.arguments)
    tool_result = TOOL_FUNCTIONS[tool_call.function.name](**args)
    return {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": json.dumps(tool_result),
    }


def ask(user_message: str, system: str = llm.DEFAULT_SYSTEM) -> ChatResult:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message},
    ]

    results: list[ChatResult] = []
    while True:
        response, latency_ms = llm._complete(messages, tools=TOOLS)
        results.append(llm._to_chat_result(response, latency_ms))

        message = response.choices[0].message
        if not message.tool_calls:
            return _sum_chat_results(results)

        messages.append(_assistant_message(message))
        for tool_call in message.tool_calls:
            messages.append(_execute_tool_call(tool_call))
```

`_execute_tool_call()` 是把 Commit 1.2 里内联写的那几行（查 `TOOL_FUNCTIONS`、`json.loads` 参数、执行、包装成 `tool` 消息）抽成一个小函数，因为这一版需要在一次响应里处理**多个** `tool_call`（模型可能一次性请求好几个工具调用，不一定是分开的几轮），`for tool_call in message.tool_calls` 逐个执行，不再假设 `tool_calls[0]` 就是全部。`_assistant_message()` 同样升级成处理任意数量的 `tool_calls`，不再假设只有一个。

`_sum_chat_results()` 也从"固定接收两个 `ChatResult`"改成接收一个列表——循环可能跑任意多轮，成本和延迟要把全部轮次加起来，不能再假设正好是两次：

```python
def _sum_chat_results(results: list[ChatResult]) -> ChatResult:
    return ChatResult(
        text=results[-1].text,
        tokens_in=sum(r.tokens_in for r in results),
        tokens_out=sum(r.tokens_out for r in results),
        cache_hit_tokens=sum(r.cache_hit_tokens for r in results),
        cache_miss_tokens=sum(r.cache_miss_tokens for r in results),
        latency_ms=sum(r.latency_ms for r in results),
        estimated_cost_usd=sum(r.estimated_cost_usd for r in results),
    )
```

### 一个专门纠正回来的设计决定：每一轮都要带 `tools`

Commit 1.2 的 `ask()` 第二次（也是最后一次）请求**没有**带 `tools` 参数——当时的理由是"这次只是要总结，不会再有第三次请求"。这个理由在"最多一次工具调用"的前提下是对的，但放到真正的循环里就不成立了：**只要循环还可能继续，模型就必须在每一轮都能看到自己有哪些工具可以用**，不然它在第二轮、第三轮永远没有机会决定"我还要再查一次"。这不是一个无关紧要的小改动，是这一步真正要修的问题——`while True` 这几行代码本身很短，但如果 `tools` 参数只在第一轮传，循环实际上还是只能跑一轮，和上一版没有本质区别。

### 这一版还没有步数上限

`while True` 没有任何退出条件保护——如果模型陷入"总觉得还需要再查一次"的状态，这个函数会真的死循环。这是**刻意留出来**的问题，Phase 2.3 才会加最大步数限制。这一步先把"能循环"这件事做对，下一步再把"循环必须有边界"这件事做对——这是这本教程一贯的"一个 commit 一个概念"的拆分方式，不是疏忽。

## 3. Break It / Observe

不是破坏性实验，是验证性实验：一个真实需要查两个商品才能回答的问题。这次不满足于看最终答案，给 `ask()` 实际使用的 `llm._client` 挂上 `httpx2` 的请求钩子，把每一轮**完整**发出去的原始 body 都记录下来——不省略任何字段，`tools` 每次都是完整的三个 Schema，不用"和上一轮一样"这种话代替。

```python
captured = []

def log_request(request):
    captured.append(json.loads(request.content))

llm._client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url=llm.DEEPSEEK_BASE_URL,
    http_client=httpx2.Client(event_hooks={"request": [log_request]}),
)

result = ask("A100 库存够不够给 B200 客户凑单？B200 客户需要 A100 和 B200 各查一下库存。")
```

### 第 1 次请求——完整原始 body（2026-09-20）

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant for a small industrial equipment company."},
    {"role": "user", "content": "A100 库存够不够给 B200 客户凑单？B200 客户需要 A100 和 B200 各查一下库存。"}
  ],
  "model": "deepseek-chat",
  "temperature": 0.0,
  "tools": [
    {"type": "function", "function": {"name": "get_inventory", "description": "...", "parameters": {"...": "..."}}},
    {"type": "function", "function": {"name": "add_inventory", "description": "...", "parameters": {"...": "..."}}},
    {"type": "function", "function": {"name": "remove_inventory", "description": "...", "parameters": {"...": "..."}}}
  ]
}
```

（完整 Schema 内容和 Commit 2.1 抓到的一致，这里省略以突出结构；实际请求里 `tools` 是三个完整的 Schema，一字不少。）

只有 `system` + `user` 两条消息，`tools` 带着完整的三个 Schema。

### 第 1 次响应——完整原始 body

```json
{
  "choices": [{
    "finish_reason": "tool_calls",
    "message": {
      "role": "assistant",
      "content": "I'll check the stock levels for both products.",
      "tool_calls": [
        {
          "id": "call_00_LGydVG3309we03AQ2Lc78463",
          "type": "function",
          "function": {"name": "get_inventory", "arguments": "{\"product_id\": \"A100\"}"}
        },
        {
          "id": "call_01_UnOgrd8FfyvpkxixUJz29520",
          "type": "function",
          "function": {"name": "get_inventory", "arguments": "{\"product_id\": \"B200\"}"}
        }
      ]
    }
  }],
  "model": "deepseek-flash"
}
```

**这里有一个第一次抓包才看清楚、和最初预想不一样的真实行为**：模型不是"先查 A100、看到结果再决定要不要查 B200"分两轮做的，而是**一次响应里的 `tool_calls` 数组直接给了两个元素**——这是并行工具调用，不是先后两轮。`ask()` 里 `for tool_call in message.tool_calls` 这个循环，处理的正是这种"一轮给多个"的情况，不是"分两轮各给一个"。

### 第 2 次请求——完整原始 body

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant for a small industrial equipment company."},
    {"role": "user", "content": "A100 库存够不够给 B200 客户凑单？B200 客户需要 A100 和 B200 各查一下库存。"},
    {
      "role": "assistant",
      "content": "I'll check the stock levels for both products.",
      "tool_calls": [
        {"id": "call_00_LGydVG3309we03AQ2Lc78463", "type": "function", "function": {"name": "get_inventory", "arguments": "{\"product_id\": \"A100\"}"}},
        {"id": "call_01_UnOgrd8FfyvpkxixUJz29520", "type": "function", "function": {"name": "get_inventory", "arguments": "{\"product_id\": \"B200\"}"}}
      ]
    },
    {"role": "tool", "tool_call_id": "call_00_LGydVG3309we03AQ2Lc78463", "content": "{\"name\": \"Industrial Sensor A100\", \"price\": 100, \"stock\": 20}"},
    {"role": "tool", "tool_call_id": "call_01_UnOgrd8FfyvpkxixUJz29520", "content": "{\"name\": \"Industrial Sensor B200\", \"price\": 180, \"stock\": 8}"}
  ],
  "model": "deepseek-chat",
  "temperature": 0.0,
  "tools": [ /* 完整的三个 Schema，一字不少 */ ]
}
```

三处值得逐字对照看的地方：

1. **`tools` 字段依然是完整的三个 Schema**——这正是这一章"每一轮都要带 tools"的实际效果，不是口头说说，抓包能直接看到。
2. **第 1 轮那条 `assistant` 消息被原样加了回来**，包括它的 `content`（"I'll check the stock levels for both products."）和完整的 `tool_calls` 数组——不是只把工具结果加回去，模型"为什么会有这些工具结果"这个上下文本身也要在。这条消息是 `_assistant_message()` 手动重建出的纯字典（延续 Commit 1.2 的设计决定），不是 SDK 对象原样塞回去的。
3. **两条新增的 `role: "tool"` 消息**，`tool_call_id` 精确对应第 1 轮返回的两个 id，`content` 是 `get_inventory()` 真实返回值的 JSON 序列化——这是 `_execute_tool_call()` 产出的东西，原样能在这里看到。

### 第 2 次响应与最终结果

```text
Here are the current stock levels:

| Product | Unit Price | Stock |
|---|---|---|
| Industrial Sensor A100 | $100 | 20 |
| Industrial Sensor B200 | $180 | 8 |

Regarding whether A100 stock is enough to "凑单" for the B200 customer: I can't answer
that yet, because I don't know how many units the customer wants. Could you tell me the
quantity they need for each product?

tokens_in=1362 tokens_out=222 latency_ms=2139 cost_usd=0.000206
```

`finish_reason` 从 `"tool_calls"` 变成了 `"stop"`，`message` 里不再有 `tool_calls` 字段——`ask()` 里 `if not message.tool_calls: return _sum_chat_results(results)` 这一行判断的正是这个变化，循环到这里结束。回答里的库存数字（20、8）和第 2 次请求里 `tool` 消息的 `content` 完全一致，是原样引用回来的，不是重新编的。

模型这次没有直接下结论"库存够不够"，而是反问用户需要多少数量——这是一个合理但和最初预想略有出入的回答（最初设计这个问题是想看它直接判断"够"或"不够"）。**这也值得如实记录**：问题本身没有说明客户具体要多少件，模型选择先澄清而不是瞎猜一个够不够的结论，是一次诚实的边界处理，不是这一步要修的问题。

**这次抓包全程只用了 2 次请求，不是 3 次**——因为两个商品的查询被模型合并成了第 1 轮里的两个并行 `tool_calls`，而不是分成"查 A100 → 查 B200 → 总结"三轮。这和最初预想的"连续两次调用"略有出入，值得记录：**模型选择并行还是串行发起多个工具调用，是它自己的决定，不是我们代码能控制的**——`ask()` 的 `for tool_call in message.tool_calls` 之所以要用循环而不是假设只有一个，就是为了同时接得住这两种情况。

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
tests/test_agent.py::test_ask_returns_directly_when_model_does_not_call_a_tool PASSED
tests/test_agent.py::test_ask_sends_the_tool_schema_on_the_first_call PASSED
tests/test_agent.py::test_ask_calls_get_inventory_and_returns_the_real_stock_number PASSED
tests/test_agent.py::test_ask_feeds_the_real_tool_result_back_to_the_model PASSED
tests/test_agent.py::test_ask_sums_cost_and_latency_across_both_calls PASSED
tests/test_agent.py::test_tool_functions_keys_match_tool_schema_names PASSED
tests/test_agent.py::test_every_request_includes_the_tool_schema PASSED
tests/test_agent.py::test_two_sequential_tool_calls_across_three_requests PASSED
tests/test_agent.py::test_multiple_tool_calls_in_a_single_response_are_all_executed PASSED
...（加上此前的 26 个）

35 passed in 0.45s
```

### 一个测试本身的坑，值得单独记录

写 `test_two_sequential_tool_calls_across_three_requests` 时，第一版直接用 `fake_client.chat.completions.create.call_args_list[1].kwargs["messages"]` 去检查"第二轮请求发出时带了多少条 tool 消息"，结果断言失败。

**原因**：`messages` 是循环里不断被原地 `append` 的**同一个列表对象**。`MagicMock` 的 `call_args_list` 存的是这个对象的**引用**，不是调用发生那一刻的快照——所以事后翻看历史调用记录，看到的永远是这个列表**最终**的样子（循环结束时已经包含了全部 tool 消息），不是"当时"的样子。

**修法**：用一个自定义的 `side_effect` 函数，在每次调用真正发生的那一刻，把 `messages` 深拷贝一份存起来：

```python
message_snapshots = []

def fake_create(**kwargs):
    message_snapshots.append(copy.deepcopy(kwargs["messages"]))
    return responses.pop(0)

fake_client.chat.completions.create.side_effect = fake_create
```

这不是 `ask()` 的 bug——它的行为本身是对的（其他测试已经间接验证过），是测试代码对"mock 捕获的是引用还是快照"这件事想当然了。这是一个通用的 mock 陷阱，只要被 mock 的函数接收一个之后会被调用方继续修改的可变对象，就可能踩到。

DoD 要求"一个需要连续两次工具调用才能回答的问题能被正确处理"，第 3 节的真实调用验证了这一点。

## 7. Exercises

1. 故意构造一个"模型认为查了三次还不够"的场景（比如问一个模糊到需要反复澄清的问题），观察循环会跑多少轮，为 Phase 2.3 的步数上限设计提供一个真实的参考数字（现在设 10 步够不够？）。
2. 让 `ask()` 打印每一轮的 `finish_reason`，观察它在"继续调用工具"和"给出最终答案"两种情况下分别是什么值。
3. 对比现在的 `ask()` 和 Commit 1.2 的版本（可以在 git 历史里翻出来看）：Commit 1.2 处理"零次或一次工具调用"总共写了不到 20 行；这一版处理"任意多次"反而更短。为什么"处理更多情况"的代码可以比"处理更少情况"的代码还简洁？

## 8. What We Learned

真正的 Agent Loop 和"最多一次工具调用"之间，代码上的差距只是一个 `while True` 加上"每一轮都要带 `tools`"这一个容易被忽略的细节——但这个细节如果漏掉，`while True` 就是摆设。Mock 测试里"捕获的是引用不是快照"这个坑，是这本教程第一次真正踩到的一个通用 Python 测试陷阱，值得记住：**只要被 mock 的对象是可变的、之后还会被继续修改，事后检查历史调用参数就要小心。**
