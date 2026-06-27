import copy
import logging
import re

try:
    from .base import BaseAgent
except ImportError:
    from eval_utils.agents.base import BaseAgent

logger = logging.getLogger(__name__)


class NaiveAgent(BaseAgent):
    """An agent that generates actions based on observations without complex reasoning."""

    def __init__(self, client_factory, prompt_builder):
        super().__init__(client_factory, prompt_builder)
        self.step_count = 0

    def act(self, obs, prev_action=None):
        if prev_action:
            self.prompt_builder.update_action(prev_action)

        self.prompt_builder.update_observation(obs)
        messages = self.prompt_builder.get_prompt()

        naive_instruction = """
You always have to output one of the above actions at a time and no other text. You always have to output an action until the episode terminates.
        """.strip()

        if messages and messages[-1].role == "user":
            messages[-1].content += "\n\n" + naive_instruction

        response = self.client.generate(messages)
        self.step_count += 1

        # Save raw outputs before extraction (for debug logging)
        self._last_raw_completion = response.completion
        self._last_raw_reasoning = response.reasoning

        if self.step_count % 50 == 0:
            long_obs = obs.get("text", {}).get("long_term_context", "")
            short_obs = obs.get("text", {}).get("short_term_context", "")
            obs_text = f"{long_obs}\n\n{short_obs}" if long_obs else short_obs
            agent_label = (
                f"Agent {self.agent_id}"
                if hasattr(self, "agent_id") and self.agent_id is not None
                else "NaiveAgent"
            )
            logger.info(f"{agent_label} step {self.step_count}: selected '{response.completion}'")
            logger.info(f"  Observation:\n{obs_text}")
            logger.info(
                f"  Raw LLM Response: model={response.model_id}, completion='{response.completion}', "
                f"stop_reason={response.stop_reason}, input_tokens={response.input_tokens}, "
                f"output_tokens={response.output_tokens}, reasoning={response.reasoning}"
            )

        final_answer = self._extract_final_answer(response)
        return final_answer

    def _extract_final_answer(self, answer):
        def filter_letters(input_string):
            return re.sub(r"[^a-zA-Z\s:]", "", input_string)

        filtered = filter_letters(answer.completion)
        lines = filtered.strip().split("\n")
        if len(lines) > 1:
            for line in lines:
                line = line.strip()
                if line:
                    filtered = line
                    break

        final_answer = copy.deepcopy(answer)
        final_answer = final_answer._replace(completion=filtered.strip())
        return final_answer

    def reset(self):
        super().reset()
        self.step_count = 0
