"""Tests for how a past turn's reasoning is fed back to the model.

Chat templates that re-render history write a thinking block for each past
assistant turn. Leaving that block empty is harmless for most models, but Laguna
S 2.1 reads it as the mode marker, so an empty one tells it to stop reasoning.
"inline" mode keeps the reasoning in the turn's text, which is what the board
has always run; "structured" mode sends it as reasoning_content instead. Either
way it reaches the model exactly once.
"""

import unittest

from omegaconf import OmegaConf

from baselines.llm.eval_utils.agents import AgentFactory
from baselines.llm.eval_utils.client import OpenAIWrapper
from baselines.llm.eval_utils.prompt_builder import HistoryPromptBuilder, Message


def _observation(text):
    return {
        "text": {"long_term_context": text, "short_term_context": ""},
        "image": None,
    }


def _builder(mode="inline"):
    return HistoryPromptBuilder(
        max_text_history=4,
        max_image_history=0,
        max_cot_history=1,
        reasoning_history_mode=mode,
    )


def _two_turns(builder):
    builder.update_observation(_observation("Step: 0/10"))
    builder.update_reasoning("First plan")
    builder.update_action("Move East")
    builder.update_observation(_observation("Step: 1/10"))
    builder.update_reasoning("Latest plan")
    builder.update_action("Do")
    builder.update_observation(_observation("Step: 2/10"))
    return [m for m in builder.get_prompt() if m.role == "assistant"]


def _wrapper(client_name="vllm", enable_thinking=True, mode="structured"):
    wrapper = OpenAIWrapper.__new__(OpenAIWrapper)
    wrapper.alternate_roles = False
    wrapper.client_name = client_name
    wrapper.enable_thinking = enable_thinking
    wrapper.reasoning_history_mode = mode
    return wrapper


def _config(agent_mode=None, client_modes=(None, None, None)):
    clients = []
    for m in client_modes:
        c = {
            "client_name": "vllm",
            "model_id": "some/model",
            "base_url": "http://127.0.0.1:8000/v1",
            "generate_kwargs": {"max_tokens": 16},
            "timeout": 5,
            "max_retries": 1,
            "delay": 0,
            "alternate_roles": False,
        }
        if m is not None:
            c["reasoning_history_mode"] = m
        clients.append(c)
    agent = {
        "type": "robust_all",
        "reasoning": True,
        "max_text_history": 4,
        "max_image_history": 0,
        "max_cot_history": 1,
    }
    if agent_mode is not None:
        agent["reasoning_history_mode"] = agent_mode
    return OmegaConf.create(
        {"agent": agent, "clients": clients, "alem": {"num_agents": len(clients)}}
    )


class TestInlineMode(unittest.TestCase):
    def test_reasoning_goes_into_the_turn_text(self):
        assistant = _two_turns(_builder("inline"))

        # max_cot_history=1, so only the newest turn keeps its reasoning.
        self.assertEqual(len(assistant), 2)
        self.assertIsNone(assistant[0].reasoning)
        self.assertEqual(assistant[0].content, "Move East")
        self.assertEqual(assistant[1].reasoning, "Latest plan")
        self.assertEqual(assistant[1].content, "Previous plan:\nLatest plan\n\nAction taken: Do")

    def test_no_reasoning_content_field_is_sent(self):
        message = Message(role="assistant", content="x", reasoning="Plan")

        converted = _wrapper(mode="inline").convert_messages([message])

        self.assertNotIn("reasoning_content", converted[0])


class TestStructuredMode(unittest.TestCase):
    def test_turn_text_holds_the_action_only(self):
        assistant = _two_turns(_builder("structured"))

        self.assertEqual(assistant[0].content, "Move East")
        self.assertEqual(assistant[1].reasoning, "Latest plan")
        self.assertEqual(assistant[1].content, "Action taken: Do")

    def test_reasoning_reaches_the_model_exactly_once(self):
        assistant = _two_turns(_builder("structured"))
        converted = _wrapper().convert_messages(assistant)

        newest = converted[-1]
        self.assertEqual(newest["reasoning_content"], "Latest plan")
        # The whole point: not also sitting in the visible text.
        text = "".join(part["text"] for part in newest["content"])
        self.assertNotIn("Latest plan", text)
        self.assertEqual(text, "Action taken: Do")

    def test_turn_without_reasoning_sends_no_field(self):
        converted = _wrapper().convert_messages([Message(role="assistant", content="Do")])

        self.assertNotIn("reasoning_content", converted[0])


class TestModePlumbing(unittest.TestCase):
    """The config path from yaml to the live client.

    A previous draft ran the config value through bool() before handing it over,
    which silently turned every non-empty string into True. Nothing downstream
    could tell, so these assert the value that actually lands on the client.
    """

    def _client(self, config, agent_idx=0):
        return AgentFactory(config).create_agent(agent_idx=agent_idx).client

    def test_default_is_inline(self):
        self.assertEqual(self._client(_config()).reasoning_history_mode, "inline")

    def test_agent_level_setting_reaches_the_client(self):
        client = self._client(_config(agent_mode="structured"))
        self.assertEqual(client.reasoning_history_mode, "structured")

    def test_client_level_setting_wins_for_a_mixed_team(self):
        config = _config(agent_mode="inline", client_modes=("structured", None, None))
        self.assertEqual(self._client(config, 0).reasoning_history_mode, "structured")
        self.assertEqual(self._client(config, 1).reasoning_history_mode, "inline")

    def test_prompt_builder_gets_the_same_mode_as_the_client(self):
        agent = AgentFactory(_config(agent_mode="structured")).create_agent(agent_idx=0)
        self.assertEqual(agent.prompt_builder.reasoning_history_mode, "structured")
        self.assertEqual(agent.client.reasoning_history_mode, "structured")

    def test_an_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            self._client(_config(agent_mode="auto"))

    def test_structured_is_refused_when_nothing_can_carry_the_field(self):
        """Structured strips the inline copy, so a client that cannot send
        reasoning_content would drop the reasoning entirely. Refuse instead of
        falling back, which would change the protocol a run is scored under."""
        hosted = _config(agent_mode="structured")
        for client in hosted.clients:
            client.client_name = "openai"
        with self.assertRaises(ValueError):
            self._client(hosted)

        no_thinking = _config(agent_mode="structured")
        no_thinking.agent.reasoning = False
        with self.assertRaises(ValueError):
            self._client(no_thinking)

    def test_structured_keeps_the_reasoning_reachable(self):
        """The pair that matters: text without the plan, field with it."""
        agent = AgentFactory(_config(agent_mode="structured")).create_agent(agent_idx=0)
        builder = agent.prompt_builder
        builder.update_observation(_observation("Step 0"))
        builder.update_reasoning("Latest plan")
        builder.update_action("Do")
        builder.update_observation(_observation("Step 1"))
        newest = [m for m in builder.get_prompt() if m.role == "assistant"][-1]

        converted = agent.client.convert_messages([newest])[0]
        text = "".join(part["text"] for part in converted["content"])
        self.assertEqual(text, "Action taken: Do")
        self.assertEqual(converted["reasoning_content"], "Latest plan")


class TestPromptCapture(unittest.TestCase):
    """What debug.html gets to show.

    In structured mode the plan is only in reasoning_content, so a capture of
    content alone would render a past turn as a bare action and look exactly
    like the failure we were investigating.
    """

    def test_capture_keeps_reasoning_content(self):
        from baselines.llm.eval_utils.agents.base import _ClientProxy

        proxy = _ClientProxy(client=None)
        proxy._capture(
            [
                Message(role="user", content="Observation"),
                Message(role="assistant", content="Action taken: Do", reasoning="Latest plan"),
            ]
        )

        self.assertNotIn("reasoning_content", proxy.last_prompt_messages[0])
        self.assertEqual(
            proxy.last_prompt_messages[1]["reasoning_content"], "Latest plan"
        )
        self.assertEqual(proxy.last_prompt_messages[1]["content"], "Action taken: Do")


if __name__ == "__main__":
    unittest.main()
