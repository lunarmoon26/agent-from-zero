import os
import sys
import time
from dataclasses import dataclass

from openai import (
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
)

DEFAULT_MODEL = "deepseek-chat"
DEFAULT_SYSTEM = "You are a helpful assistant for a small industrial equipment company."
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# USD / 1M tokens，非高峰时段价格（DeepSeek 官方定价页，2026-09-13 抓取）。
# "deepseek-chat" 这个模型名目前实际解析到 deepseek-flash（用真实请求的
# response.model 字段确认过，见 docs/00-03），价格按 deepseek-flash 这一档算。
# 故意不做高峰/非高峰时段判断——高峰时段价格是这里的两倍，是一个真实存在但
# 这一步先不处理的简化，见文档"我们故意没有解决的问题"。
PRICE_PER_MILLION_TOKENS_USD = {
    "input_cache_hit": 0.003,
    "input_cache_miss": 0.15,
    "output": 0.6,
}

_client: OpenAI | None = None


class ChatError(Exception):
    """chat() 所有已分类失败的公共基类。调用方既可以精确 catch 某一个子类，
    也可以只认这一个基类，一网打尽——这一层存在的意义是让后面的 Agent Loop
    /Retry 逻辑不需要认识 openai SDK 自己的异常体系，只需要认识这三种。"""


class AuthenticationFailure(ChatError):
    """API Key 缺失或错误——重试没有意义，必须先修好凭证。"""


class NetworkFailure(ChatError):
    """连不上或超时——本质都是"这次没打通"，见 chat() 里对
    APIConnectionError 的说明，超时被归进了同一类。"""


class BadRequestFailure(ChatError):
    """请求本身不合法（模型名不存在、参数超范围、prompt 太长等）——
    重试没有意义，必须先改请求内容。"""


@dataclass
class ChatResult:
    """chat() 从这一版开始返回这个结构，不是裸字符串——`text` 是真正的回答，
    其余字段是这次调用的成本/延迟画像。这不是为了这一个 commit 好看，是给
    Phase 14（Observability）的 MetricsCollector 攒素材：那时候要做的是把这里
    已经产出的单次数据聚合成趋势，不是重新发明"怎么衡量一次调用"这件事。"""

    text: str
    tokens_in: int
    tokens_out: int
    cache_hit_tokens: int
    cache_miss_tokens: int
    latency_ms: float
    estimated_cost_usd: float


def get_client() -> OpenAI:
    """惰性单例：只有真正调用 chat() 时才会去读 DEEPSEEK_API_KEY、建立客户端——
    这样 import 这个模块本身不需要环境变量存在，测试可以直接 monkeypatch 掉
    这个函数，绕过真实的网络客户端。"""
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url=DEEPSEEK_BASE_URL)
    return _client


def _estimate_cost_usd(cache_hit_tokens: int, cache_miss_tokens: int, tokens_out: int) -> float:
    return (
        cache_hit_tokens / 1_000_000 * PRICE_PER_MILLION_TOKENS_USD["input_cache_hit"]
        + cache_miss_tokens / 1_000_000 * PRICE_PER_MILLION_TOKENS_USD["input_cache_miss"]
        + tokens_out / 1_000_000 * PRICE_PER_MILLION_TOKENS_USD["output"]
    )


def _complete(messages: list[dict], temperature: float = 0.0, tools: list[dict] | None = None):
    """真正发请求的那一层，被 chat() 和 Commit 1.2 的 tool-calling 逻辑共用——
    后者需要拿到原始 response（读 message.tool_calls），不能只要 chat() 那样
    拍扁成纯文本的 ChatResult。`tools` 不传时完全不带这个字段，行为和 Commit 0.1
    时一模一样。三种失败分类逻辑只在这一层做一次，两个调用方都不用重复处理。
    """
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


def _to_chat_result(response, latency_ms: float) -> ChatResult:
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


def chat(user_message: str, system: str = DEFAULT_SYSTEM, temperature: float = 0.0) -> ChatResult:
    """
    最基础的一次 Chat Completion，不涉及工具、不涉及历史。DeepSeek 走的是
    OpenAI 兼容协议：system 不是单独的参数，而是 messages 列表里的第一条记录。

    temperature=0.0 是这里的默认值——Agent 场景要的是"同样的输入尽量得到同样
    的行为"，不是聊天场景那种"每次都想要点新意"。

    三种失败模式被重新分类成我们自己的异常类型，而不是让调用方直接接住
    openai SDK 原始的异常：AuthenticationFailure（凭证问题）、NetworkFailure
    （连接失败或超时——openai.APITimeoutError 本身就是 APIConnectionError
    的子类，一个 except 分支同时覆盖两种情况）、BadRequestFailure（请求本身
    不合法）。

    返回值是 ChatResult，用 time.perf_counter() 量出这次请求真正花了多久，
    从 response.usage 里读出真实的 token 计费明细，再折算成估算成本。
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message},
    ]
    response, latency_ms = _complete(messages, temperature=temperature)
    return _to_chat_result(response, latency_ms)


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or input("You: ")
    result = chat(question)
    print(result.text)
    print(
        f"[chat] tokens_in={result.tokens_in} tokens_out={result.tokens_out} "
        f"latency_ms={result.latency_ms:.0f} cost_usd={result.estimated_cost_usd:.6f}"
    )
