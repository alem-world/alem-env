import copy
import logging
import re

try:
    from ..client import LLMClientWrapper
    from .base import BaseAgent
except ImportError:
    from eval_utils.agents.base import BaseAgent
    from eval_utils.client import LLMClientWrapper

logger = logging.getLogger(__name__)


class ChainOfThoughtAgent(BaseAgent):
    """An agent that performs actions using a chain-of-thought reasoning process."""

    def __init__(self, client_factory: LLMClientWrapper, prompt_builder, config):
        super().__init__(client_factory, prompt_builder)
        self.remember_cot = config.agent.remember_cot
        self.step_count = 0

    def act(self, obs, prev_action=None):
        if prev_action:
            self.prompt_builder.update_action(prev_action)

        self.prompt_builder.update_observation(obs)
        messages = self.prompt_builder.get_prompt()

        cot_instructions = """
First think about what's the best course of action step by step.
Finally, provide a single output action at the end of the message in the form of: ACTION: <action>
        """.strip()

        messages[-1].content += "\n\n" + cot_instructions
        cot_reasoning = self.client.generate(messages)
        self.step_count += 1

        # Save raw outputs before extraction (for debug logging)
        self._last_raw_completion = cot_reasoning.completion
        self._last_raw_reasoning = cot_reasoning.reasoning

        if self.step_count % 50 == 0:
            long_obs = obs.get("text", {}).get("long_term_context", "")
            short_obs = obs.get("text", {}).get("short_term_context", "")
            obs_text = f"{long_obs}\n\n{short_obs}" if long_obs else short_obs
            logger.info(f"ChainOfThoughtAgent step {self.step_count}")
            logger.info(f" Observation:\n{obs_text}")
            logger.info(
                f" Raw LLM Response: model={cot_reasoning.model_id}, completion='{cot_reasoning.completion}', "
                f"stop_reason={cot_reasoning.stop_reason}, input_tokens={cot_reasoning.input_tokens}, "
                f"output_tokens={cot_reasoning.output_tokens}, reasoning={cot_reasoning.reasoning}"
            )

        final_answer = self._extract_final_answer(cot_reasoning)

        if self.step_count % 50 == 0:
            logger.info(f" Extracted action: '{final_answer.completion}'")

        return final_answer

    def _extract_final_answer(self, reasoning):
        def filter_letters(input_string):
            return re.sub(r"[^a-zA-Z\s:]", "", input_string)

        answer = copy.deepcopy(reasoning)
        self.prompt_builder.update_reasoning(reasoning.completion)
        answer = answer._replace(reasoning=answer.completion)
        answer = answer._replace(
            completion=filter_letters(answer.completion).split("ACTION:")[-1].strip()
        )
        return answer

    def reset(self):
        super().reset()
        self.step_count = 0
