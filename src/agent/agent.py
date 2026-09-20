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


def _sum_chat_results(first: ChatResult, second: ChatResult) -> ChatResult:
    """一次问答如果触发了工具调用，实际发生了两次真实的模型请求——`text` 只取
    最后一次（那才是给用户看的最终回答），但成本/延迟必须两次相加，否则
    Dashboard（Phase 14）会看到一个比实际便宜一半的数字。"""
    return ChatResult(
        text=second.text,
        tokens_in=first.tokens_in + second.tokens_in,
        tokens_out=first.tokens_out + second.tokens_out,
        cache_hit_tokens=first.cache_hit_tokens + second.cache_hit_tokens,
        cache_miss_tokens=first.cache_miss_tokens + second.cache_miss_tokens,
        latency_ms=first.latency_ms + second.latency_ms,
        estimated_cost_usd=first.estimated_cost_usd + second.estimated_cost_usd,
    )


def ask(user_message: str, system: str = llm.DEFAULT_SYSTEM) -> ChatResult:
    """把 Tool Schema 带给模型，让它自己决定要不要调用；如果调用了，就真的执行
    对应的 Python 函数，把结果喂回去，再问模型一次拿最终自然语言回答。

    这里还不是 Agent Loop（Commit 2.2 才是）——只处理"最多一次工具调用"：模型
    要么不调用工具直接回答，要么调用一次 `get_inventory` 后就必须给出最终回答，
    不会再触发第二轮工具调用。
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message},
    ]

    response, latency_ms = llm._complete(messages, tools=TOOLS)
    first_result = llm._to_chat_result(response, latency_ms)

    message = response.choices[0].message
    if not message.tool_calls:
        return first_result

    tool_call = message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    tool_result = TOOL_FUNCTIONS[tool_call.function.name](**args)

    messages.append(
        {
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
            ],
        }
    )
    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(tool_result),
        }
    )

    response, latency_ms = llm._complete(messages)
    second_result = llm._to_chat_result(response, latency_ms)

    return _sum_chat_results(first_result, second_result)
