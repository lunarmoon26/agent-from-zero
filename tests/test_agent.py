import copy
import json
from unittest.mock import MagicMock

import pytest

from src.agent import agent, llm


def _tool_call(call_id: str, name: str, arguments: dict) -> MagicMock:
    tool_call = MagicMock()
    tool_call.id = call_id
    tool_call.function.name = name
    tool_call.function.arguments = json.dumps(arguments)
    return tool_call


def _response(*, content: str | None, tool_calls: list | None, prompt_tokens: int = 10, completion_tokens: int = 5):
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls
    choice = MagicMock()
    choice.message = message
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.prompt_cache_hit_tokens = 0
    usage.prompt_cache_miss_tokens = prompt_tokens
    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


def _client_with_responses(*responses) -> MagicMock:
    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = list(responses)
    return fake_client


class TestBusinessContext:
    def test_agent_system_prompt_contains_the_generic_default_system(self):
        assert llm.DEFAULT_SYSTEM in agent.AGENT_SYSTEM_PROMPT

    def test_agent_system_prompt_contains_business_context(self):
        assert "Singapore" in agent.AGENT_SYSTEM_PROMPT
        assert str(agent.SAFETY_STOCK) in agent.AGENT_SYSTEM_PROMPT
        assert "7 days" in agent.AGENT_SYSTEM_PROMPT

    def test_ask_defaults_to_the_agent_system_prompt(self, monkeypatch):
        fake_client = _client_with_responses(_response(content="ok", tool_calls=None))
        monkeypatch.setattr(llm, "get_client", lambda: fake_client)

        agent.ask("hi")

        call_kwargs = fake_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["messages"][0] == {"role": "system", "content": agent.AGENT_SYSTEM_PROMPT}

    def test_ask_still_accepts_a_custom_system(self, monkeypatch):
        fake_client = _client_with_responses(_response(content="ok", tool_calls=None))
        monkeypatch.setattr(llm, "get_client", lambda: fake_client)

        agent.ask("hi", system="custom system")

        call_kwargs = fake_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["messages"][0] == {"role": "system", "content": "custom system"}


def test_ask_returns_directly_when_model_does_not_call_a_tool(monkeypatch):
    fake_client = _client_with_responses(_response(content="Hi there!", tool_calls=None))
    monkeypatch.setattr(llm, "get_client", lambda: fake_client)

    result = agent.ask("hello")

    assert result.text == "Hi there!"
    assert fake_client.chat.completions.create.call_count == 1


def test_ask_sends_the_tool_schema_on_the_first_call(monkeypatch):
    fake_client = _client_with_responses(_response(content="Hi there!", tool_calls=None))
    monkeypatch.setattr(llm, "get_client", lambda: fake_client)

    agent.ask("A100 还有多少库存？")

    call_kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["tools"] == agent.TOOLS


def test_ask_calls_get_inventory_and_returns_the_real_stock_number(monkeypatch):
    tool_call = _tool_call("call_1", "get_inventory", {"product_id": "A100"})
    first_response = _response(content=None, tool_calls=[tool_call])
    second_response = _response(content="A100 still has 20 units in stock.", tool_calls=None)
    fake_client = _client_with_responses(first_response, second_response)
    monkeypatch.setattr(llm, "get_client", lambda: fake_client)

    result = agent.ask("A100 还有多少库存？")

    assert result.text == "A100 still has 20 units in stock."
    assert fake_client.chat.completions.create.call_count == 2


def test_ask_feeds_the_real_tool_result_back_to_the_model(monkeypatch):
    tool_call = _tool_call("call_1", "get_inventory", {"product_id": "A100"})
    first_response = _response(content=None, tool_calls=[tool_call])
    second_response = _response(content="ok", tool_calls=None)
    fake_client = _client_with_responses(first_response, second_response)
    monkeypatch.setattr(llm, "get_client", lambda: fake_client)

    agent.ask("A100 还有多少库存？")

    second_call_kwargs = fake_client.chat.completions.create.call_args_list[1].kwargs
    messages = second_call_kwargs["messages"]
    tool_message = next(m for m in messages if m["role"] == "tool")
    assert tool_message["tool_call_id"] == "call_1"
    assert json.loads(tool_message["content"]) == {
        "name": "Industrial Sensor A100",
        "price": 100,
        "stock": 20,
    }


def test_ask_sums_cost_and_latency_across_both_calls(monkeypatch):
    tool_call = _tool_call("call_1", "get_inventory", {"product_id": "A100"})
    first_response = _response(content=None, tool_calls=[tool_call], prompt_tokens=10, completion_tokens=5)
    second_response = _response(content="ok", tool_calls=None, prompt_tokens=20, completion_tokens=8)
    fake_client = _client_with_responses(first_response, second_response)
    monkeypatch.setattr(llm, "get_client", lambda: fake_client)

    result = agent.ask("A100 还有多少库存？")

    assert result.tokens_in == 30
    assert result.tokens_out == 13


def test_tool_functions_keys_match_tool_schema_names():
    schema_names = {schema["function"]["name"] for schema in agent.TOOLS}
    assert schema_names == set(agent.TOOL_FUNCTIONS.keys())


def test_every_request_includes_the_tool_schema(monkeypatch):
    # 和上一版不一样：这一版每一轮都要带 tools，因为循环可能不止两轮，
    # 模型需要在每一轮都知道自己还能调用什么。
    tool_call = _tool_call("call_1", "get_inventory", {"product_id": "A100"})
    first_response = _response(content=None, tool_calls=[tool_call])
    second_response = _response(content="ok", tool_calls=None)
    fake_client = _client_with_responses(first_response, second_response)
    monkeypatch.setattr(llm, "get_client", lambda: fake_client)

    agent.ask("A100 还有多少库存？")

    for call in fake_client.chat.completions.create.call_args_list:
        assert call.kwargs["tools"] == agent.TOOLS


def test_two_sequential_tool_calls_across_three_requests(monkeypatch):
    # "A100 库存够不够给 B200 客户凑单" 这类问题：先查 A100，再查 B200，
    # 两轮都是模型自己决定要不要继续调用，第三轮才给出最终总结。
    first = _response(content=None, tool_calls=[_tool_call("call_1", "get_inventory", {"product_id": "A100"})])
    second = _response(content=None, tool_calls=[_tool_call("call_2", "get_inventory", {"product_id": "B200"})])
    third = _response(content="A100 库存 20，B200 库存 8，足够凑单。", tool_calls=None)
    responses = [first, second, third]

    # 不能直接用 side_effect=[...] 之后再看 call_args_list 里的 messages——
    # messages 是循环里同一个列表对象，会被原地不断 append，call_args_list
    # 存的是这个对象的引用，不是调用当时的快照。事后翻看历史调用参数，看到
    # 的永远是这个列表"最终"的样子，不是"当时"的样子。这里用一个包装函数，
    # 在每次调用真正发生的那一刻就把 messages 深拷贝下来，才能拿到真实的
    # 历史快照。
    message_snapshots = []

    def fake_create(**kwargs):
        message_snapshots.append(copy.deepcopy(kwargs["messages"]))
        return responses.pop(0)

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = fake_create
    monkeypatch.setattr(llm, "get_client", lambda: fake_client)

    result = agent.ask("A100 库存够不够给 B200 客户凑单？")

    assert result.text == "A100 库存 20，B200 库存 8，足够凑单。"
    assert fake_client.chat.completions.create.call_count == 3

    # 第二轮请求发出那一刻，应该已经包含第一轮的 tool 调用记录和结果
    tool_messages = [m for m in message_snapshots[1] if isinstance(m, dict) and m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert json.loads(tool_messages[0]["content"])["name"] == "Industrial Sensor A100"

    # 第三轮请求发出那一刻，两次工具结果应该都在
    tool_messages = [m for m in message_snapshots[2] if isinstance(m, dict) and m.get("role") == "tool"]
    assert len(tool_messages) == 2


def test_multiple_tool_calls_in_a_single_response_are_all_executed(monkeypatch):
    # 模型在同一轮里一次性请求两个工具调用（不是分两轮），验证 for 循环
    # 逐个执行、逐条把结果加回去，而不是只处理第一个就丢掉其余的。
    first_response = _response(
        content=None,
        tool_calls=[
            _tool_call("call_a", "get_inventory", {"product_id": "A100"}),
            _tool_call("call_b", "get_inventory", {"product_id": "B200"}),
        ],
    )
    second_response = _response(content="两个都查到了。", tool_calls=None)
    fake_client = _client_with_responses(first_response, second_response)
    monkeypatch.setattr(llm, "get_client", lambda: fake_client)

    result = agent.ask("A100 和 B200 分别还有多少库存？")

    assert result.text == "两个都查到了。"
    second_call_messages = fake_client.chat.completions.create.call_args_list[1].kwargs["messages"]
    tool_messages = [m for m in second_call_messages if isinstance(m, dict) and m.get("role") == "tool"]
    assert {m["tool_call_id"] for m in tool_messages} == {"call_a", "call_b"}


class TestMissingProductPlumbing:
    """
    这几个测试不回答"模型会不会编造数字"——那是模型的行为，不是我们代码的逻辑，
    只能靠真实调用观察（见 docs/01-03-hallucination-guard.md 的 Break It 实验）。
    这里只验证 get_inventory 返回 None 时，ask() 的管道本身接得住，不会因为
    None 这个值而崩溃或传错东西。
    """

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

        # 不应该抛任何异常——None 是 get_inventory 的正常返回值之一（Commit 1.1
        # 就是这么设计的），不是需要特殊 catch 的错误。
        agent.ask("Z999 还有多少库存？")


class TestMaxStepsGuard:
    """
    这一组的 Break It 实验故意不打真实 API——要验证的是"我们自己的循环控制
    代码在跑满 max_steps 之后会不会正确停下来"，这是我们自己的逻辑，不是模型
    的真实行为，用 mock 反复模拟"模型永远还要再调用一次工具"完全够用，而且
    免费。真的花钱去让模型在生产环境里死循环 10+ 次，不会比这更有说服力。
    """

    def _always_wants_another_tool_call(self, **_kwargs) -> MagicMock:
        # 每次都返回一个新的 tool_call，模拟"模型永远觉得还需要再查一次"
        # 这种（真实世界里确实会发生的）状态。
        return _response(
            content=None,
            tool_calls=[_tool_call("call_loop", "get_inventory", {"product_id": "A100"})],
        )

    def test_normal_task_is_not_affected_by_the_guard(self, monkeypatch):
        # 正常两轮就结束的任务，加了 max_steps 之后不应该被误伤。
        tool_call = _tool_call("call_1", "get_inventory", {"product_id": "A100"})
        first_response = _response(content=None, tool_calls=[tool_call])
        second_response = _response(content="A100 库存 20。", tool_calls=None)
        fake_client = _client_with_responses(first_response, second_response)
        monkeypatch.setattr(llm, "get_client", lambda: fake_client)

        result = agent.ask("A100 还有多少库存？", max_steps=10)

        assert result.text == "A100 库存 20。"

    def test_infinite_loop_is_stopped_at_max_steps(self, monkeypatch):
        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = self._always_wants_another_tool_call
        monkeypatch.setattr(llm, "get_client", lambda: fake_client)

        with pytest.raises(agent.MaxStepsExceeded) as exc_info:
            agent.ask("随便问点什么", max_steps=10)

        assert exc_info.value.max_steps == 10
        # 跑满 10 步就应该停手，不多不少——第 11 次调用不应该发生。
        assert fake_client.chat.completions.create.call_count == 10

    def test_max_steps_exceeded_message_is_readable(self, monkeypatch):
        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = self._always_wants_another_tool_call
        monkeypatch.setattr(llm, "get_client", lambda: fake_client)

        with pytest.raises(agent.MaxStepsExceeded) as exc_info:
            agent.ask("随便问点什么", max_steps=3)

        assert "3" in str(exc_info.value)
        assert "steps" in str(exc_info.value)

    def test_max_steps_is_configurable(self, monkeypatch):
        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = self._always_wants_another_tool_call
        monkeypatch.setattr(llm, "get_client", lambda: fake_client)

        with pytest.raises(agent.MaxStepsExceeded):
            agent.ask("随便问点什么", max_steps=2)

        assert fake_client.chat.completions.create.call_count == 2

    def test_max_steps_exceeded_is_a_loop_error_not_a_chat_error(self):
        # LoopError 和 llm.py 的 ChatError 是两条不同的失败家族——循环控制流
        # 失败和模型调用失败是不同性质的问题，不应该共用同一个基类，方便以后
        # 调用方分别处理（比如：ChatError 值得 retry，LoopError 通常不值得）。
        assert issubclass(agent.MaxStepsExceeded, agent.LoopError)
        assert issubclass(agent.LoopError, Exception)
        assert not issubclass(agent.LoopError, llm.ChatError)
