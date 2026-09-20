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


def ask(user_message: str, system: str = llm.DEFAULT_SYSTEM) -> ChatResult:
    """
    真正的 Agent Loop：取代 Commit 1.2/1.3 那个"最多一次工具调用"的版本——
    不再假设只会有零次或一次调用，而是持续循环，直到模型这一轮不再要求调用
    任何工具为止。

    和上一版相比，有一个刻意纠正的设计决定：上一版最后一次请求没有带
    `tools` 参数，因为那时候假设已经不会再有第三次请求。这一版每一轮都带上
    `TOOLS`——循环可能要跑任意多轮，只要还没到"模型不再需要工具"的那一轮，
    模型就必须始终能看到自己还有哪些工具可以用，否则它在第二轮、第三轮
    永远没有机会决定再调用一次。

    这一版还没有步数上限（Phase 2.3 才加）——如果模型陷入"总觉得还需要
    再查一次"的状态，这个函数会真的死循环，这是刻意留给下一步解决的问题，
    不是这一步的疏漏。

    模型一次响应里也可能同时请求多个工具调用（不只是一个），这一版用
    for 循环逐个执行、逐条把结果加回去，不假设 tool_calls 只有一个元素。
    """
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
