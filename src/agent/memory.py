from src.agent.agent import AGENT_SYSTEM_PROMPT, DEFAULT_MAX_STEPS, _run_loop_on_messages
from src.agent.llm import ChatResult


class Conversation:
    """短期记忆的具体实现——和 `ask()` 唯一的区别是 `messages` 的生命周期：
    这里存在 `self` 上，`.send()` 每次复用同一个列表继续累加；`ask()` 里
    `messages` 是局部变量，函数返回就没了。核心循环只写了一份
    （`_run_loop_on_messages`），两者共享，不重复实现 Agent Loop。

    这不是一种新的推理能力，只是把已经存在的循环机制的输出，保留在了一个
    跨调用存活的地方——模型的推理本身没有变，变的只是它这次能看到多少历史。
    """

    def __init__(self, system: str = AGENT_SYSTEM_PROMPT, max_steps: int = DEFAULT_MAX_STEPS):
        self.max_steps = max_steps
        self.messages: list[dict] = [{"role": "system", "content": system}]

    def send(self, user_message: str) -> ChatResult:
        self.messages.append({"role": "user", "content": user_message})
        return _run_loop_on_messages(self.messages, self.max_steps)
