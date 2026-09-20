import json

from src.agent import llm
from src.agent.llm import ChatResult
from src.tools.inventory import (
    ADD_INVENTORY_SCHEMA,
    GET_INVENTORY_SCHEMA,
    REMOVE_INVENTORY_SCHEMA,
    add_inventory,
    get_inventory,
    remove_inventory,
)

# 从一个只读工具扩展到三个（一读两写）——ask() 本身一行都不用改，靠
# tool_call.function.name 去这两个字典/列表里按名字查，加一个新工具只需要
# 在这两个地方各加一条。
TOOLS = [GET_INVENTORY_SCHEMA, ADD_INVENTORY_SCHEMA, REMOVE_INVENTORY_SCHEMA]
TOOL_FUNCTIONS = {
    "get_inventory": get_inventory,
    "add_inventory": add_inventory,
    "remove_inventory": remove_inventory,
}

DEFAULT_MAX_STEPS = 10


class LoopError(Exception):
    """Agent Loop 本身的控制流失败——不是模型调用失败（那是 llm.py 的
    ChatError 家族），是"循环这个机制自己出了问题"，两者不共用基类。"""


class MaxStepsExceeded(LoopError):
    def __init__(self, max_steps: int):
        super().__init__(f"Agent loop did not reach a final answer within {max_steps} steps.")
        self.max_steps = max_steps


def _sum_chat_results(results: list[ChatResult]) -> ChatResult:
    """一次问答只要触发了工具调用，就会发生不止一次真实的模型请求——`text`
    只取最后一次（那才是给用户看的最终回答），但成本/延迟必须把全部请求
    的数字加起来，否则 Phase 14 的 Dashboard 会看到一个远比实际发生便宜、
    快的假数字。"""
    return ChatResult(
        text=results[-1].text,
        tokens_in=sum(r.tokens_in for r in results),
        tokens_out=sum(r.tokens_out for r in results),
        cache_hit_tokens=sum(r.cache_hit_tokens for r in results),
        cache_miss_tokens=sum(r.cache_miss_tokens for r in results),
        latency_ms=sum(r.latency_ms for r in results),
        estimated_cost_usd=sum(r.estimated_cost_usd for r in results),
    )


def _assistant_message(message) -> dict:
    """把模型返回的 message 对象重建成一个纯字典再放回历史，而不是把 SDK
    对象原样塞回去——这样 messages 列表在任何时候都可以直接 json.dumps()，
    不会夹带一个只有这套 SDK 才认识的对象（见 Commit 1.2 的说明）。"""
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
    """执行一次工具调用，返回可以直接塞进下一轮 messages 的那条 tool 消息。"""
    args = json.loads(tool_call.function.arguments)
    tool_result = TOOL_FUNCTIONS[tool_call.function.name](**args)
    return {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": json.dumps(tool_result),
    }


def ask(user_message: str, system: str = llm.DEFAULT_SYSTEM, max_steps: int = DEFAULT_MAX_STEPS) -> ChatResult:
    """
    真正的 Agent Loop：持续循环，直到模型这一轮不再要求调用任何工具为止。

    每一轮都带上 `TOOLS`——循环可能要跑任意多轮，只要还没到"模型不再需要
    工具"的那一轮，模型就必须始终能看到自己还有哪些工具可以用，否则它在
    第二轮、第三轮永远没有机会决定再调用一次。

    模型一次响应里也可能同时请求多个工具调用（不只是一个），用 for 循环
    逐个执行、逐条把结果加回去，不假设 `tool_calls` 只有一个元素。

    `max_steps` 是这一版新加的硬上限——Commit 2.2 的 `while True` 没有任何
    退出条件保护，如果模型陷入"总觉得还需要再查一次"的状态，会真的死循环。
    跑满 `max_steps` 轮还没拿到最终答案，就抛 `MaxStepsExceeded`，不返回一个
    看起来像正常回答、实际是"未完成"的字符串——字符串结果和真正的最终答案
    长得一样，调用方很容易忘记检查"这是不是真的答案"；异常会强制调用方
    要么处理、要么让它继续往上冒，不会被悄悄当成正常结果使用。
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message},
    ]

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
