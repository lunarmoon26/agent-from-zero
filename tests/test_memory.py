import copy
import json
from unittest.mock import MagicMock

from src.agent import agent, llm, memory
from tests.test_agent import _client_with_responses, _response, _tool_call


def test_conversation_starts_with_only_a_system_message():
    conversation = memory.Conversation()

    assert conversation.messages == [{"role": "system", "content": agent.AGENT_SYSTEM_PROMPT}]


def test_conversation_accepts_a_custom_system_prompt():
    conversation = memory.Conversation(system="a totally different system prompt")

    assert conversation.messages[0] == {"role": "system", "content": "a totally different system prompt"}


def test_send_appends_user_message_and_assistant_reply(monkeypatch):
    fake_client = _client_with_responses(_response(content="A100 库存 20。", tool_calls=None))
    monkeypatch.setattr(llm, "get_client", lambda: fake_client)

    conversation = memory.Conversation()
    result = conversation.send("A100 还有多少库存？")

    assert result.text == "A100 库存 20。"
    assert conversation.messages[1] == {"role": "user", "content": "A100 还有多少库存？"}
    assert conversation.messages[2] == {"role": "assistant", "content": "A100 库存 20。"}


def test_second_send_reuses_history_from_the_first(monkeypatch):
    # 第二轮请求发出的那一刻，messages 里应该已经带着第一轮的问答，不是
    # 又从一份全新的 [system, user] 开始——这正是 Conversation 和 ask()
    # 的核心区别。
    #
    # 不能事后直接看 call_args_list 里的 messages——它存的是同一个列表对象
    # 的引用，self.messages 会在这之后继续被 append，事后看到的永远是"最终"
    # 状态。这里用 side_effect 在调用发生的那一刻就深拷贝下来（同
    # test_agent.py 的 test_two_sequential_tool_calls_across_three_requests
    # 用的方法）。
    responses = [
        _response(content="A100 库存 20。", tool_calls=None),
        _response(content="卖 20 个之后剩 0。", tool_calls=None),
    ]
    message_snapshots = []

    def fake_create(**kwargs):
        message_snapshots.append(copy.deepcopy(kwargs["messages"]))
        return responses.pop(0)

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = fake_create
    monkeypatch.setattr(llm, "get_client", lambda: fake_client)

    conversation = memory.Conversation()
    conversation.send("A100 还有多少库存？")
    conversation.send("如果卖 20 个呢？")

    second_call_messages = message_snapshots[1]
    assert {"role": "user", "content": "A100 还有多少库存？"} in second_call_messages
    assert {"role": "assistant", "content": "A100 库存 20。"} in second_call_messages
    assert second_call_messages[-1] == {"role": "user", "content": "如果卖 20 个呢？"}


def test_every_send_includes_the_tool_schema(monkeypatch):
    fake_client = _client_with_responses(
        _response(content="ok", tool_calls=None),
        _response(content="ok again", tool_calls=None),
    )
    monkeypatch.setattr(llm, "get_client", lambda: fake_client)

    conversation = memory.Conversation()
    conversation.send("hi")
    conversation.send("hi again")

    for call in fake_client.chat.completions.create.call_args_list:
        assert call.kwargs["tools"] == agent.TOOLS


def test_send_can_still_trigger_a_tool_call_mid_conversation(monkeypatch):
    first_reply = _response(content="好的，有什么可以帮你？", tool_calls=None)
    tool_call_reply = _response(
        content=None,
        tool_calls=[_tool_call("call_1", "get_inventory", {"product_id": "A100"})],
    )
    final_reply = _response(content="A100 库存 20。", tool_calls=None)
    fake_client = _client_with_responses(first_reply, tool_call_reply, final_reply)
    monkeypatch.setattr(llm, "get_client", lambda: fake_client)

    conversation = memory.Conversation()
    conversation.send("你好")
    result = conversation.send("A100 还有多少库存？")

    assert result.text == "A100 库存 20。"
    tool_messages = [m for m in conversation.messages if isinstance(m, dict) and m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert json.loads(tool_messages[0]["content"])["name"] == "Industrial Sensor A100"


def test_three_turn_reference_resolution(monkeypatch):
    # 复刻 Commit 6.2 的验收标准：三轮对话，第三轮要能正确引用第二轮提到
    # 的假设数量（"卖 20 个"），不是重新去查一次原始库存。
    fake_client = _client_with_responses(
        _response(content="A100 目前库存 20 台。", tool_calls=None),
        _response(content="卖 20 个之后，剩余 0 台。", tool_calls=None),
        _response(content="按上一轮的假设，还剩 0 台。", tool_calls=None),
    )
    monkeypatch.setattr(llm, "get_client", lambda: fake_client)

    conversation = memory.Conversation()
    conversation.send("A100 有多少库存？")
    conversation.send("如果卖 20 个呢？")
    result = conversation.send("那还剩多少？")

    assert "0" in result.text
    third_call_messages = fake_client.chat.completions.create.call_args_list[2].kwargs["messages"]
    assert {"role": "assistant", "content": "卖 20 个之后，剩余 0 台。"} in third_call_messages
