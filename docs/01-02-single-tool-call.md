# 01.2 — Wire Tool into a Single LLM Call

> Commit: `feat: wire tool into a single LLM call`
> Tag: `v0.1.2-single-tool-call`

## 1. Problem

Commit 1.1 把"查库存"写成了模型能读懂的 Tool Schema，也验证过模型能正确复述这个工具的用途——但那次验证只是"模型知道这个工具存在"，从没有真的让模型**决定**调用它、程序**执行**调用、把结果**喂回去**。`get_inventory("A100")` 目前只能被 Python 代码直接调用，用户问"A100 还有多少库存？"，`chat()` 完全不知道这个问题该查数据，只能凭训练数据编一个听起来合理的回答。这一步要把 Tool Calling 协议的完整单轮闭环打通：模型看到 Schema → 决定调用 → 程序执行 → 结果喂回 → 模型给出基于真实数据的最终回答。

还不是循环（那是 Commit 2.2 的 Agent Loop）——这一步只处理"最多一次工具调用"：模型要么不调用工具直接回答，要么调用一次 `get_inventory` 后必须给出最终回答。

## 2. Minimal Example

先把 `src/agent/llm.py` 里原来揉在 `chat()` 一个函数里的"发请求"和"把 response 拍扁成 `ChatResult`"拆成两个可复用的部分：

```python
def _complete(messages, temperature=0.0, tools=None):
    started_at = time.perf_counter()
    try:
        kwargs = dict(model=DEFAULT_MODEL, temperature=temperature, messages=messages)
        if tools:
            kwargs["tools"] = tools
        response = get_client().chat.completions.create(**kwargs)
    except AuthenticationError as e:
        raise AuthenticationFailure(str(e)) from e
    except APIConnectionError as e:
        raise NetworkFailure(str(e)) from e
    except BadRequestError as e:
        raise BadRequestFailure(str(e)) from e
    latency_ms = (time.perf_counter() - started_at) * 1000
    return response, latency_ms


def _to_chat_result(response, latency_ms) -> ChatResult:
    usage = response.usage
    cache_hit_tokens = usage.prompt_cache_hit_tokens or 0
    cache_miss_tokens = usage.prompt_cache_miss_tokens or 0
    tokens_out = usage.completion_tokens
    return ChatResult(
        text=response.choices[0].message.content,
        tokens_in=usage.prompt_tokens,
        tokens_out=tokens_out,
        cache_hit_tokens=cache_hit_tokens,
        cache_miss_tokens=cache_miss_tokens,
        latency_ms=latency_ms,
        estimated_cost_usd=_estimate_cost_usd(cache_hit_tokens, cache_miss_tokens, tokens_out),
    )


def chat(user_message, system=DEFAULT_SYSTEM, temperature=0.0) -> ChatResult:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message},
    ]
    response, latency_ms = _complete(messages, temperature=temperature)
    return _to_chat_result(response, latency_ms)
```

`chat()` 对外的行为完全没变（Commit 0.1-0.3 的测试原样通过），只是内部把"发一次请求"和"读出成本/延迟"两件事抽出来，好让新写的 tool-calling 逻辑复用，而不是把这套失败分类和计费逻辑再抄一遍。

真正新增的部分在 `src/agent/agent.py`：

```python
TOOLS = [GET_INVENTORY_SCHEMA]
TOOL_FUNCTIONS = {"get_inventory": get_inventory}


def ask(user_message: str, system: str = llm.DEFAULT_SYSTEM) -> ChatResult:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message},
    ]

    response, latency_ms = llm._complete(messages, tools=TOOLS)
    first_result = llm._to_chat_result(response, latency_ms)

    message = response.choices[0].message
    if not message.tool_calls:
        return first_result  # 模型没打算用工具，直接回答，只花了一次请求

    tool_call = message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    tool_result = TOOL_FUNCTIONS[tool_call.function.name](**args)

    messages.append({
        "role": "assistant",
        "content": message.content,
        "tool_calls": [{
            "id": tool_call.id,
            "type": "function",
            "function": {"name": tool_call.function.name, "arguments": tool_call.function.arguments},
        }],
    })
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": json.dumps(tool_result),
    })

    response, latency_ms = llm._complete(messages)
    second_result = llm._to_chat_result(response, latency_ms)
    return _sum_chat_results(first_result, second_result)
```

几个设计点：

- **`messages` 里必须先原样放回模型那条"我要调用工具"的 assistant message，再放工具结果**——这不是这个项目自己发明的约定，是 OpenAI 兼容协议本身的要求：`role: "tool"` 的消息靠 `tool_call_id` 和前一条 assistant message 里的 `tool_calls[].id` 对上号，模型才知道"这条结果是回答我刚才哪一次调用请求的"。漏掉 assistant 那条消息，或者 `tool_call_id` 对不上，协议层面就是不合法的请求。
- **工具执行结果用 `json.dumps()` 序列化成字符串再放进 `content`**——`role: "tool"` 消息的 `content` 只能是字符串，不能直接塞一个 Python `dict`。`get_inventory` 返回的 `{"name": ..., "price": ..., "stock": ...}` 要先序列化。
- **触发了工具调用的一次问答，实际是两次真实的模型请求**——`text` 只取第二次（那才是给用户看的最终回答），但 `tokens_in`/`tokens_out`/`latency_ms`/`estimated_cost_usd` 必须把两次请求的数字加起来（`_sum_chat_results`），否则 Phase 14 的 Dashboard 会看到一个比实际发生的便宜一半、快一半的假数字。
- **这一步只处理 `message.tool_calls[0]`，不处理模型一次返回多个 tool_calls 的情况**——目前只有一个工具可选，模型没有理由在一条消息里请求调用两次 `get_inventory`；等 Commit 2.1 加入多个工具、Commit 2.2 写出真正的 Agent Loop，才需要认真处理"一次响应里带多个 tool_calls"和"工具调用触发新一轮工具调用"这些情况。
- **没有做成通用的 Tool Registry（`name -> function` 之外更复杂的东西）**——`TOOL_FUNCTIONS` 就是一个最直接的字典，够这一步用。Commit 2.1 加入第二、第三个工具时，如果这个字典还够用，就不动它；如果分发逻辑变复杂了，到那时候再重构，不提前设计。
- **回填 `messages` 时手动重建了一个精简的 assistant dict，没有把 `response.choices[0].message` 这个 SDK 对象原样塞回去**——`openai` SDK 其实允许直接塞对象（下一次请求前会自己序列化），但那样一来，这一次问答的完整历史里就会混进一个不是纯 JSON、绑定着 `openai` 这个具体 SDK 版本的对象。手动重建成 `{"role": ..., "content": ..., "tool_calls": [...]}` 这样的纯字典，多写了几行代码，换来的是 `messages` 列表在任何时候都可以直接 `json.dumps()`——Commit 8.x 要把完整对话历史存成 Trace，Commit 13.x 要把它落盘做 Checkpoint，都要求这份历史是普通数据，不能夹带一个具体 SDK 版本才认识的对象。

### 2.0 谁真正"调用"了这个工具

看 `ask()` 这段代码，容易冒出一个误解：既然叫"Tool Calling"，会不会是模型自己在某个地方直接执行了 `get_inventory()`？

不是。**在所有主流 LLM Provider（OpenAI、Anthropic、DeepSeek……）的协议里，模型从来没有、也不可能真正执行任何工具**——这不是这个项目简化出来的说法，是整个行业统一的协议设计，生产环境同样如此。

`ask()` 里，工作被拆成互不重叠的两半，边界就在这一行：

```python
tool_result = TOOL_FUNCTIONS[tool_call.function.name](**args)
```

- **这一行左边，是模型已经做完的事——"决定"**：它读到 `TOOLS`（也就是 `GET_INVENTORY_SCHEMA`）里的 `description` 和 `parameters`，输出一段结构化 JSON，表达"我认为应该调用 `get_inventory`、带 `product_id=A100`"。这段 JSON 到达程序这一侧之前，没有触发过任何真实的数据库查询——`response.choices[0].message.tool_calls` 就是模型交出的全部产物。
- **这一行本身，才是真正"执行"的那一半**：`TOOL_FUNCTIONS[tool_call.function.name]` 在本地字典里查到真实的 Python 函数，`(**args)` 才让它在本地进程里真正跑起来、真正读一次 `products` 字典。执行权、要不要真的执行、执行前要不要校验参数，这些决定权始终留在这一行所在的本地代码里，不在模型那一侧。

为什么协议要这样切割：模型运行在 Provider 的服务器上，没有理由、也不应该有权限直接访问企业内部的数据库、支付系统或第三方 API——如果"模型直接执行"，就意味着 Provider 的服务器能够触达全世界所有客户的私有系统，这在安全上是不可接受的。这也是为什么 Commit 4.x（主动破坏）能够专门针对"模型生成了一个恶意参数"做实验：一次 Prompt Injection 最多能做到"诱导模型在 `tool_call` 里生成一个指向别人订单的 `order_id`"，而不可能让模型绕过这行代码直接作恶——它压根没有直接作恶的能力。

### 2.1 抓包看这一轮真正发生的两次请求

第 3 节的"Break It / Observe"只记录了最终文本和汇总后的 token/延迟，还没有把两次真实 HTTP 请求的原始 body 展开看过。给 `httpx2` 加一个请求钩子，把 `ask("A100 还有多少库存？")` 触发的两次请求分别记录下来：

```python
captured = []

def log_request(request):
    captured.append(json.loads(request.content))

llm._client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url=llm.DEEPSEEK_BASE_URL,
    http_client=httpx2.Client(event_hooks={"request": [log_request]}),
)

result = ask("A100 还有多少库存？")
```

真实抓到的第一次请求体（2026-09-20）：

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant for a small industrial equipment company."},
    {"role": "user", "content": "A100 还有多少库存？"}
  ],
  "model": "deepseek-chat",
  "temperature": 0.0,
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_inventory",
        "description": "Look up the current stock level and unit price for a product, given its product ID. Use this whenever the user asks about stock, availability, or price for a specific product.",
        "parameters": {
          "type": "object",
          "properties": {"product_id": {"type": "string", "description": "The product ID, e.g. 'A100' or 'B200'."}},
          "required": ["product_id"]
        }
      }
    }
  ]
}
```

对应的第一次响应（同一次请求）：

```json
{
  "choices": [{
    "finish_reason": "tool_calls",
    "message": {
      "content": "",
      "role": "assistant",
      "tool_calls": [{
        "id": "call_00_bej7uwqVwn3nm4gNtEQg4078",
        "function": {"name": "get_inventory", "arguments": "{\"product_id\": \"A100\"}"},
        "type": "function"
      }]
    }
  }],
  "model": "deepseek-flash",
  "usage": {
    "completion_tokens": 40,
    "prompt_tokens": 335,
    "prompt_cache_hit_tokens": 128,
    "prompt_cache_miss_tokens": 207
  }
}
```

真实抓到的第二次请求体——这次不带 `tools`，但多了 `ask()` 手动重建的 assistant 消息和一条 `role: "tool"` 消息：

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant for a small industrial equipment company."},
    {"role": "user", "content": "A100 还有多少库存？"},
    {
      "role": "assistant",
      "content": "",
      "tool_calls": [{
        "id": "call_00_luvHOu8Y1L4ze41twVA08026",
        "type": "function",
        "function": {"name": "get_inventory", "arguments": "{\"product_id\": \"A100\"}"}
      }]
    },
    {
      "role": "tool",
      "tool_call_id": "call_00_luvHOu8Y1L4ze41twVA08026",
      "content": "{\"name\": \"Industrial Sensor A100\", \"price\": 100, \"stock\": 20}"
    }
  ],
  "model": "deepseek-chat",
  "temperature": 0.0
}
```

几个只有抓包才看得清楚的细节：

1. **第一次响应里 `content` 是空字符串，不是 `null`**——这一次模型没有像 Commit 1.1 抓到的另一次调用那样先说一句过渡语，`tool_calls` 是这条 assistant 消息里唯一有实质内容的部分。`ask()` 把这个空字符串原样存进重建的 assistant dict，不会展示给用户。
2. **`tool_call_id` 是两次请求之间唯一的关联线索**——第二次请求里的 assistant 消息和 tool 消息，靠同一个 `call_00_luvHOu8Y1L4ze41twVA08026` 才能被模型认出"这条工具结果是在回答哪一次调用请求"。如果 `ask()` 在拼第二条消息时把这个 id 写错，协议层面这次请求就是不合法的。
3. **`role: "tool"` 消息的 `content` 是 `get_inventory("A100")` 真实返回值序列化后的字符串**——`{"name": "Industrial Sensor A100", "price": 100, "stock": 20}`，和 `products` 字典里的原始数据逐字段对得上，不是模型自己编出来再让程序确认的结果。
4. **第一次调用命中的模型是 `deepseek-flash`，`prompt_cache_hit_tokens=128`**——请求发的是 `deepseek-chat`，服务端解析后实际路由到 `deepseek-flash`（延续 Commit 0.3 的观察），这次 335 个输入 token 里有 128 个命中了缓存，说明 `system` 加 `tools` 这段固定内容已经具备被复用的条件，具体命中窗口和粒度仍然是 `TODO.md` 里待深入的问题。

## 3. Break It / Observe

不是破坏性实验（那是 Commit 2.3 的死循环、Commit 4.x 的故障注入），是 DoD 要求的验证性实验：发两类问题，一类应该触发工具调用，一类不该。

```python
from src.agent.agent import ask

r1 = ask("A100 还有多少库存？")
print(r1.text)

r2 = ask("What is 2+2?")
print(r2.text)
```

真实调用结果（`deepseek-chat`，2026-09-19）：

```text
--- "A100 还有多少库存？" ---
A100（工业传感器 A100）目前还有 20 件库存。
tokens_in=431 tokens_out=57 latency_ms=1668 cost_usd=0.000099

--- "What is 2+2?" ---
2 + 2 = 4.
（只发生了一次请求，call_count == 1）
```

两件事都验证到了：第一个问题里的"20"是 `products["A100"]["stock"]` 的真实值，不是模型编的——`git blame` 得到的数据和回答里的数字对得上；第二个问题模型完全没有尝试调用工具，`_complete` 只被调用了一次，说明 Schema 里 `description` 写的"Use this whenever the user asks about stock..."确实起了限定作用，模型不会把这个工具用在不相关的问题上。

## 4. Implement the Fix

不适用，理由同上——这一步没有需要修复的破坏性发现。

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
...（加上此前的 17 个）

22 passed in 0.44s
```

五个新测试：模型不调用工具时只发一次请求、直接回答；第一次请求确实带上了 `tools=[GET_INVENTORY_SCHEMA]`；模型调用工具时会发生两次请求，最终文本是第二次的；喂回模型的 `role: "tool"` 消息里 `tool_call_id` 和真实的 `get_inventory` 返回值对得上（不是随便拼的字符串）；两次请求的 token 数被正确相加，不是只取了其中一次。第 3 节那次真实调用没有写进自动化测试——"模型会不会决定调用工具、最终回答措辞怎么样"是模型这次的真实行为，不是我们代码的逻辑，断言会让测试变得脆弱且不可控。

## 7. Exercises

1. 把 `GET_INVENTORY_SCHEMA` 的 `description` 临时改回 Commit 1.1 练习题 1 里那个含糊版本（`"Get inventory."`），重新跑一次第 3 节的两个问题，观察第二个问题（"What is 2+2?"）会不会也触发一次不必要的工具调用——直接感受"描述质量"这次不只是影响模型怎么"说"，还影响它怎么"做决定"。
2. 故意构造一个 `product_id` 传参错误的场景（比如在 `system` 里加一句误导性指令，让模型倾向于把商品名"Industrial Sensor A100"整个填进 `product_id`），观察 `get_inventory` 会不会因为查不到而返回 `None`，模型最终会怎么回应这个 `None`——这是在为 Commit 1.3 的 Hallucination 实验预热，这里先只观察，不用急着修。
3. 想一想：如果这一步不是"最多一次工具调用"，而是模型在拿到第一次 `get_inventory` 结果后，又想再查一次别的商品（比如"A100 和 B200 都查一下"），现在的 `ask()` 会发生什么？（提示：`message.tool_calls[0]` 之外的调用请求会被直接忽略，第二次 `_complete(messages)` 也没有再传 `tools`，模型这次物理上不可能再发起新的工具调用）——这个限制正是 Commit 2.2 要解决的问题。

## 8. What We Learned

Tool Calling 协议的完整单轮闭环，比"模型返回一个函数名和参数"要多两步容易被忽略的细节：喂回结果前必须先把模型那条"我要调用工具"的 assistant message 原样放回历史（`tool_call_id` 靠它才能对上号），工具的返回值进 `content` 前必须先序列化成字符串。一次触发了工具调用的问答，背后是两次真实的模型请求，衡量成本和延迟时必须把两次加起来，不能只看其中一次——这是延续 Commit 0.3"衡量机制要在用到之前搭好"这个习惯,在多请求场景下的第一次实践。以及：`chat()` 和这里新加的 `ask()` 共用同一套失败分类和计费逻辑，靠的是先把 `chat()` 内部拆成 `_complete`/`_to_chat_result` 两个可复用的部分，而不是复制粘贴一份改一半。
