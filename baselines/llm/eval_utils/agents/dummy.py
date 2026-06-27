import logging
from collections import namedtuple

try:
    from .base import BaseAgent
except ImportError:
    from eval_utils.agents.base import BaseAgent

LLMResponse = namedtuple(
    "LLMResponse",
    ["model_id", "completion", "stop_reason", "input_tokens", "output_tokens", "reasoning"],
)


def make_dummy_action(text):
    return LLMResponse(
        model_id="dummy",
        completion="wait",
        stop_reason="none",
        input_tokens=1,
        output_tokens=1,
        reasoning=None,
    )


class DummyAgent(BaseAgent):
    """Agent for debugging purposes."""

    def __init__(self, client_factory, prompt_builder):
        super().__init__(client_factory, prompt_builder)

    def act(self, obs, prev_action=None):
        return make_dummy_action("dummy_action")
