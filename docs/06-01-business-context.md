# 06.1 — Add Business Context Injection

> Commit: `feat: add business context injection`
> Tag: `v0.6.1-business-context`
>
> 动手前先读 [`docs/06-00-context-memory-cache-concepts.md`](06-00-context-memory-cache-concepts.md)——这一章是那份概念说明里"固定层"的具体实现。

## 1. Problem

到 Commit 2.3 为止，Agent 只会查原始数字（"A100 库存 20"），不会做业务判断（"20 算不算够"）。这一步给它补上判断所需要的背景知识——不是新的工具，是让它"知道"一些从不查库存表也该知道的规则。

## 2. Minimal Example

`src/agent/agent.py` 新增：

```python
SAFETY_STOCK = 10

BUSINESS_CONTEXT = (
    "Business context for this company:\n"
    "- Warehouse location: Singapore\n"
    f"- Safety stock threshold: {SAFETY_STOCK} units, applies to every product. "
    "If a product's stock is at or below this threshold, treat it as low stock "
    "and mention that a reorder should be considered.\n"
    "- Supplier lead time: 7 days.\n"
)

AGENT_SYSTEM_PROMPT = f"{llm.DEFAULT_SYSTEM}\n\n{BUSINESS_CONTEXT}"
```

`ask()` 的默认 `system` 参数从 `llm.DEFAULT_SYSTEM` 改成 `AGENT_SYSTEM_PROMPT`。

### 三个动手前就定下来的范围决定

- **只做 System Context，不做 Runtime/User Context**。目前系统里没有 session、没有用户身份（Phase 12 才有权限概念），Runtime/User Context 现在填不出任何真实内容——硬塞会违反"不加没有测试覆盖的能力"这条一直在守的规矩。这两类留到它们真正有内容可测的时候再加。
- **Safety Stock 是全局一个数，不是每个商品各自的字段**。`SAFETY_STOCK = 10` 对所有商品统一生效，够用（A100 库存 20 高于阈值，B200 库存 8 低于阈值，两种情况都能被测到），不提前引入"每个商品单独配置"这种当前不需要的复杂度。
- **`AGENT_SYSTEM_PROMPT` 是叠加在 `llm.DEFAULT_SYSTEM` 之上，不是替换掉它**。`chat()`（Commit 0.1 的通用单次对话函数）不需要认识这家公司的库存政策，它的默认值应该保持通用；`ask()` 这一层才需要业务背景，所以在 `agent.py` 里单独拼一份，`llm.py` 的 `DEFAULT_SYSTEM` 一个字都没改。

## 3. Break It / Observe

不是破坏性实验，是两个方向的验证性实验：DoD 要求的业务判断，以及 `docs/06-00` 里提出但没验证过的一个假设——**业务上下文加进 System Prompt 之后，还能不能被 Cache 命中**。

```python
print(ask("A100 库存是不是低于安全水平？").text)  # stock=20 > 10
print(ask("B200 库存是不是低于安全水平？").text)  # stock=8  <= 10
```

### DoD 验证：业务判断是否正确

真实调用结果（`deepseek-chat`，2026-09-20）：

**A100（真实库存 20，高于阈值 10）**：

```text
A100（Industrial Sensor A100）目前库存为 20 件，高于安全库存阈值 10 件，所以不属于
低库存，暂时无需补货。

补充信息：
- 单价：100
- 供应商交货周期：7 天

如果后续库存降到 10 件或以下，就建议考虑下单补货了。
```

**B200（真实库存 8，低于阈值 10）**：

```text
是的，B200 目前库存偏低。

- 产品：Industrial Sensor B200
- 当前库存：8 件
- 安全库存阈值：10 件
- 单价：180

由于库存（8 件）已低于安全库存水平（10 件），建议考虑补货。供应商交货周期为 7 天，
建议尽快下单以避免缺货。
```

两个方向都判断对了，而且**两次回答都主动**把 `BUSINESS_CONTEXT` 里第三条"Supplier lead time: 7 days"用上了——这条信息问题本身根本没提到，是模型自己从背景知识里调出来、用在了"要不要补货、什么时候补"这个判断上。这正是这一章想要的效果：不是"背景知识被复述一遍"，是"背景知识参与了推理"。

### Cache 实验：业务上下文是否进了可缓存的前缀

四次请求（两个问题，每个问题内部一次工具调用 + 一次总结，共 4 次真实请求）的 `usage`：

```text
请求 1（A100，第一轮）  prompt_tokens=651  cache_hit=512  cache_miss=139
请求 2（A100，第二轮）  prompt_tokens=723  cache_hit=512  cache_miss=211
请求 3（B200，第一轮）  prompt_tokens=651  cache_hit=512  cache_miss=139
请求 4（B200，第二轮）  prompt_tokens=733  cache_hit=512  cache_miss=221
```

**四次请求的 `cache_hit_tokens` 全部是 512，包括脚本里的"第一次"请求**——这和 `docs/06-00` 原本设想的"第一次冷启动应该是 0"不一致，值得如实记录：这段 `AGENT_SYSTEM_PROMPT + TOOLS` 前缀在这次实验之前，已经被本章前面 DoD 验证那两次真实调用发送过了，DeepSeek 的缓存显然**按内容命中，不是按进程或脚本运行边界重置**——只要同一段前缀最近发送过，换一个全新的 Python 进程重新跑，一样能命中。这比"第一次必然是冷启动"更准确地说明了缓存的生效范围。

更重要的验证在于：**`cache_hit_tokens` 稳定停在 512，不受后面问的是 A100 还是 B200 影响**——`cache_miss_tokens` 随着 `user` 消息和工具结果变化（139/211/139/221），但命中的那部分始终是同一个数字。这证实了 `docs/06-00` 提出的假设：业务上下文只要保持稳定、放在前面，确实会被算进可缓存的前缀，不受后面用户具体问什么影响。

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
tests/test_agent.py::TestBusinessContext::test_agent_system_prompt_contains_the_generic_default_system PASSED
tests/test_agent.py::TestBusinessContext::test_agent_system_prompt_contains_business_context PASSED
tests/test_agent.py::TestBusinessContext::test_ask_defaults_to_the_agent_system_prompt PASSED
tests/test_agent.py::TestBusinessContext::test_ask_still_accepts_a_custom_system PASSED
...（加上此前的 40 个）

44 passed in 0.58s
```

四个新测试：`AGENT_SYSTEM_PROMPT` 同时包含通用默认值和业务背景（不是二选一）、三条业务规则的关键字都在（Singapore/10/7 days）、`ask()` 默认真的用了这份新 prompt、自定义 `system` 参数依然能覆盖默认值（不会被这次改动锁死）。

Cache 命中率这条**没有**写进自动化测试——命中率是服务端的真实行为，不是我们代码的逻辑，套用 Commit 2.3 定下的原则：验证我们自己的逻辑用 mock，验证真实（尤其是服务端黑盒）行为必须用真实调用，两者不能互相替代，也不该用测试断言去锁死一个我们控制不了的服务端数字。

## 7. Exercises

1. 把 `SAFETY_STOCK` 改成 `25`（这样 A100 的库存 20 也会变成"低于安全水平"），重新跑一次针对 A100 的问题，验证判断确实跟着阈值变了，而不是模型自己记了一个固定的心智模型。
2. 现在 `BUSINESS_CONTEXT` 是三条规则拼在一个字符串里。设想一下：如果以后要支持"不同地区仓库、不同安全库存阈值"，这个纯文本拼接的方式会在哪里开始变得难维护？和把它做成结构化数据（比如一个 dict，再在运行时格式化成文本）相比，各自的取舍是什么？
3. 追查一下"`cache_hit_tokens` 在脚本第一次请求就已经是 512"这个现象：连续运行两次这个实验脚本（中间不做任何操作），第二次的四个数字会不会和第一次完全一样？如果隔几分钟再跑呢？这能帮助摸清 DeepSeek 缓存的失效窗口大概有多长，补进 `TODO.md`。

## 8. What We Learned

Business Context 不是"背景资料"，是能被模型主动调用、参与推理的信息——A100/B200 两次都主动用上 Supplier Lead Time 就是证据；`docs/06-00` 里提出但没验证的"业务上下文能不能被缓存"这个假设，这一章第一次有了真实数据支撑：可以，而且不受后续具体问题内容影响；顺带发现了一个比预想更值得记录的现象——缓存命中是按内容匹配的服务端行为，不是按"这是不是这个脚本/进程里的第一次请求"来判断冷热，这一点原计划没有预料到。三个待定的范围问题（Runtime/User Context 要不要做、Safety Stock 要不要分商品、要不要改 `DEFAULT_SYSTEM`）最终都选择了更小的范围，这不是偷懒，是延续这本教程一直在守的"不为了凑全而加没有真实用途的代码"这条原则。
