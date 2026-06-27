"""
Robust Chain-of-Thought Agent for Alem LLM Evaluation.

Based on BALROG's RobustCoTAgent scaffolding with additional robustness:
  - Structured output tags (<action>...</action>) for reliable action extraction
  - Multi-strategy fallback extraction: tag → ACTION: prefix → substring → fuzzy
  - Retry with error feedback when extraction fails (up to MAX_RETRIES)
  - Action validation against the valid action list before returning
  - Clean separation of reasoning (stored in .reasoning) and action (.completion)
  - Configurable CoT memory (remember_cot) for prompt history
  - Retry statistics tracking for analysis
"""

import copy
import logging
import re

try:
    from .base import BaseAgent
    from .robust_naive import extract_action_multistrategy
except ImportError:
    from eval_utils.agents.base import BaseAgent
    from eval_utils.agents.robust_naive import extract_action_multistrategy

logger = logging.getLogger(__name__)


class RobustCoTAgent(BaseAgent):
    """An agent that performs actions using a chain-of-thought reasoning process.

    Uses <action>...</action> XML tags for reliable action extraction,
    with multi-strategy fallback parsing and retry-with-feedback on failure.
    The entire chain-of-thought is stored in `reasoning` and the extracted
    action in `completion`.
    """

    MAX_RETRIES = 0  # Additional attempts after first failure

    def __init__(self, client_factory, prompt_builder, config, client_config):
        """Initialize the RobustCoTAgent with a client, prompt builder, and configuration.

        Args:
            client_factory: A factory for creating the LLM client instance.
            prompt_builder: Object to build prompts for the agent.
            config: Configuration object containing settings for the agent.
        """
        super().__init__(client_factory, prompt_builder)
        self.remember_cot = config.agent.remember_cot
        self.max_tokens = client_config.generate_kwargs.get("max_tokens", 2048)
        self.step_count = 0
        self.total_retries = 0
        self.total_parse_failures = 0

    def _get_cot_instructions(self):
        return (
            f"You have a total budget of {self.max_tokens} tokens for your entire response (reasoning + output combined). Be concise.\n"
            "You must output your response in the following order:\n"
            "1. Your reasoning about the best course of action (keep it brief).\n"
            "2. (Required) Exactly one action from the available action list:\n"
            "<action>YOUR_CHOSEN_ACTION</action>"
        )

    def build_prompt(self, obs, prev_action=None):
        """Build the full prompt messages for this step (without calling the LLM)."""
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)
        messages = self.prompt_builder.get_prompt()
        if messages and messages[-1].role == "user":
            messages[-1].content += "\n\n" + self._get_cot_instructions()
        return messages

    def act(self, obs, prev_action=None):
        """Generate the next action using chain-of-thought reasoning.

        Args:
            obs (dict): The current observation in the environment.
            prev_action (str, optional): The previous action taken.

        Returns:
            LLMResponse: The response with the extracted action in `completion`
                and the entire chain-of-thought in `reasoning`.
        """
        if prev_action:
            self.prompt_builder.update_action(prev_action)

        self.prompt_builder.update_observation(obs)

        messages = self.prompt_builder.get_prompt()

        if messages and messages[-1].role == "user":
            messages[-1].content += "\n\n" + self._get_cot_instructions()

        # Generate with validation — retries are handled by the client
        first_response, last_response, extracted, retries = self.client.generate_with_validation(
            messages,
            validate_fn=lambda r: extract_action_multistrategy(r.completion),
            error_message=(
                "Your output did not contain a valid action. "
                "You must output exactly one action from the game's action list "
                "using the format: <action>YOUR_CHOSEN_ACTION</action>\n"
                "Keep your reasoning brief and focus on outputting a valid action."
            ),
            max_parse_retries=self.MAX_RETRIES,
        )
        self.step_count += 1
        self.total_retries += retries

        if extracted is None:
            self.total_parse_failures += 1
            extracted = "Noop"
            logger.warning(
                f"RobustCoTAgent step {self.step_count}: failed to parse after "
                f"{retries + 1} attempts. Raw: '{last_response.completion[:200]}'. "
                f"Defaulting to Noop."
            )

        # Save raw outputs before extraction/cleaning (for debug logging)
        self._last_raw_completion = last_response.completion
        self._last_raw_reasoning = last_response.reasoning

        final_answer = self._extract_final_answer(last_response, extracted)

        # Store reasoning in prompt history for CoT memory.
        # Clean out model-invented tags and bare action echoes.
        # Pass None if no genuine reasoning exists.
        if self.remember_cot:
            if first_response.reasoning:
                # Thinking-mode model: reasoning already separated into .reasoning
                raw_cot = first_response.reasoning
            else:
                # Standard model: reasoning is inline before the action tag
                raw_cot = first_response.completion.split("<action>")[0].strip()
            self.prompt_builder.update_reasoning(self._clean_reasoning(raw_cot, extracted))

        reasoning_preview = (final_answer.reasoning or "")[:300]
        logger.debug(
            f"[step {self.step_count}] action='{final_answer.completion}' retries={retries}\n"
            f"  reasoning: {reasoning_preview!r}"
        )

        if self.step_count % 50 == 0:
            logger.info(
                f"RobustCoTAgent step {self.step_count}: action='{final_answer.completion}', "
                f"retries_total={self.total_retries}, failures={self.total_parse_failures}"
            )

        return final_answer

    @staticmethod
    def _clean_reasoning(raw_text, extracted_action):
        """Clean reasoning text: strip output tags and bare action echoes.

        Handles common failure modes from small models:
          - Model outputs action tags inside reasoning text
          - Bare action echo: model outputs just "Move South" with no reasoning
          - ACTION: prefix leftovers
        """
        if not raw_text:
            return None
        # Strip any XML output tags
        cleaned = re.sub(r"</?(?:action|communication|scratchpad)>", "", raw_text).strip()
        if not cleaned:
            return None
        # If what remains is just the action name, there's no actual reasoning
        if cleaned.lower() == extracted_action.lower():
            return None
        return cleaned

    def _extract_final_answer(self, response, extracted_action):
        """Build the final LLMResponse with the extracted action and CoT reasoning.

        Args:
            response (LLMResponse): The raw response from the LLM.
            extracted_action (str): The validated action name.

        Returns:
            LLMResponse: A copy with `completion` set to extracted_action
                         and `reasoning` set to the model's reasoning text.
        """
        if response.reasoning:
            # Internal thinking mode (enable_thinking=true + --reasoning-parser on server):
            # vLLM already separated the <think> block into reasoning_content.
            reasoning_text = response.reasoning
        else:
            # Explicit CoT mode: take text before <action>, then clean.
            raw = response.completion.split("<action>")[0].strip()
            reasoning_text = self._clean_reasoning(raw, extracted_action)
        final_answer = copy.deepcopy(response)
        final_answer = final_answer._replace(
            reasoning=reasoning_text,
            completion=extracted_action,
        )
        return final_answer

    def reset(self):
        """Reset the agent state for a new episode."""
        super().reset()
        self.step_count = 0
        self.total_retries = 0
        self.total_parse_failures = 0

    def get_retry_stats(self):
        """Return retry/parse statistics for logging and analysis."""
        return {
            "total_steps": self.step_count,
            "total_retries": self.total_retries,
            "total_parse_failures": self.total_parse_failures,
            "retry_rate": round(self.total_retries / max(self.step_count, 1), 4),
            "parse_failure_rate": round(self.total_parse_failures / max(self.step_count, 1), 4),
        }
