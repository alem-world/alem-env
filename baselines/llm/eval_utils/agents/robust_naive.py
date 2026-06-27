"""
Robust Naive Agent for Alem LLM Evaluation.

Based on BALROG's RobustNaiveAgent scaffolding with additional robustness:
  - Structured output tags (<action>...</action>) for reliable parsing
  - Multi-strategy fallback extraction: tag → ACTION: prefix → substring → fuzzy
  - Retry with error feedback when extraction fails (up to MAX_RETRIES)
  - Action validation against the valid action list before returning
  - Retry statistics tracking for analysis
"""

import copy
import logging
import re
from difflib import get_close_matches

try:
    from .base import BaseAgent
except ImportError:
    from eval_utils.agents.base import BaseAgent

logger = logging.getLogger(__name__)

# ============================================================================
# Valid actions & helpers (shared with robust_cot.py)
# ============================================================================

# Source of truth for action names.
try:
    from alem.llm.alem_language_wrapper import ACTIONS as _WRAPPER_ACTIONS
except ImportError:
    # Fallback for executions that import via package path.
    from baselines.llm.alem_language_wrapper import ACTIONS as _WRAPPER_ACTIONS

VALID_ACTIONS = list(_WRAPPER_ACTIONS)

_VALID_ACTIONS_LOWER = {a.lower(): a for a in VALID_ACTIONS}


def validate_action(candidate):
    """Return canonical action name if candidate is valid, else None.

    Performs exact match first, then case-insensitive match.
    """
    if not candidate or not candidate.strip():
        return None
    candidate = candidate.strip()
    # Exact match
    if candidate in VALID_ACTIONS:
        return candidate
    # Case-insensitive match
    lower = candidate.lower()
    if lower in _VALID_ACTIONS_LOWER:
        return _VALID_ACTIONS_LOWER[lower]
    return None


def fuzzy_match_action(candidate, cutoff=0.6):
    """Fuzzy-match candidate against valid actions using difflib.

    Args:
        candidate: Raw text to match.
        cutoff: Minimum similarity ratio (0-1). Defaults to 0.6.

    Returns:
        Canonical action name or None.
    """
    if not candidate or not candidate.strip():
        return None
    matches = get_close_matches(
        candidate.strip().lower(),
        [a.lower() for a in VALID_ACTIONS],
        n=1,
        cutoff=cutoff,
    )
    if matches:
        return _VALID_ACTIONS_LOWER[matches[0]]
    return None


_GIVE_PATTERN = re.compile(r"^give\s+(?:to\s+)?(?:agent|teammate)[_\s-]*\d+$", re.IGNORECASE)
_GIVE_EXTRACT_PATTERN = re.compile(
    r"^\s*give\s+(?:to\s+)?(?:agent|teammate)[_\s-]*(\d+)\s*$",
    re.IGNORECASE,
)


def _clean_extracted(text):
    """Strip common model formatting noise from extracted tag content."""
    return text.strip().strip("\"'`").rstrip(".,;:!?").strip()


def _normalize_give_target(text):
    """Return canonical 'Give to Agent X' form, or None if not a Give-target action."""
    if not text:
        return None
    m = _GIVE_EXTRACT_PATTERN.match(text)
    if not m:
        return None
    return f"Give to Agent {int(m.group(1))}"


def _try_extract_strict(text):
    """Strict extraction: Give-target normalization + exact/case-insensitive action match.

    Intentionally excludes fuzzy matching to avoid false positives in contexts
    where we only want explicit action selections.
    """
    if not text:
        return None
    give = _normalize_give_target(text)
    if give:
        return give
    return validate_action(text)


def _try_extract_valid(text):
    """Try exact, colon-split, and fuzzy matching against VALID_ACTIONS.

    Returns canonical action name or None.
    Also canonicalizes Give target actions to "Give to Agent X".
    """
    if not text:
        return None
    # Give to Agent X — return canonical target form for env-level slot mapping.
    give = _normalize_give_target(text)
    if give:
        return give
    # Exact / case-insensitive match
    valid = validate_action(text)
    if valid:
        return valid
    # "ActionName: description" — try before colon
    if ":" in text:
        before = text.split(":")[0].strip()
        give = _normalize_give_target(before)
        if give:
            return give
        valid = validate_action(before)
        if valid:
            return valid
        fuzzy = fuzzy_match_action(before)
        if fuzzy:
            return fuzzy
    # Fuzzy on full text (handles minor typos)
    return fuzzy_match_action(text)


def extract_action_multistrategy(completion_text):
    """Try multiple parsing strategies to extract a valid action from LLM output.

    Strict extraction — only recognizes actions the model explicitly selected,
    not action names mentioned in reasoning text (following BALROG's approach).

    Strategies (in priority order):
      1a. <action>...</action> XML tags (primary format)
      1b. <action>... without end tag — grab until next tag or end of string
      2.  ACTION: prefix (common CoT/instruction-following output)
      3.  Entire completion is exactly a valid action name (no extra text)
      4.  Text before the first XML tag is a valid action — handles both
          "Move South <communication>..." and "Noop\n<communication>..."
      5.  First line is a valid action with no XML tags anywhere

    Returns:
        Canonical action name, or None if all strategies fail.
    """
    if not completion_text:
        return None

    # Strategy 1a: <action>...</action>
    match = re.search(r"<action>(.*?)</action>", completion_text, re.DOTALL)
    if match:
        extracted = _clean_extracted(match.group(1))
        result = _try_extract_valid(extracted)
        if result:
            return result

    # Strategy 1b: opening tag present but no end tag — grab until next tag or end of string.
    if not match and "<action>" in completion_text:
        m2 = re.search(r"<action>(.*?)(?=<[a-z]|$)", completion_text, re.DOTALL)
        if m2:
            extracted = _clean_extracted(m2.group(1))
            result = _try_extract_valid(extracted)
            if result:
                return result

    # Strategy 2: ACTION: prefix (e.g. from CoT "ACTION: Move North")
    action_match = re.search(r"ACTION:\s*(.+?)(?:\n|$)", completion_text, re.IGNORECASE)
    if action_match:
        extracted = _clean_extracted(action_match.group(1))
        result = _try_extract_valid(extracted)
        if result:
            return result

    # Strategy 3: Entire completion is exactly a valid action (no extra text).
    # This catches models that output just "Move North" or "Give to Agent 0"
    # without tags. Uses strict matching only — no fuzzy — so we still avoid
    # accidentally extracting action words from free-form reasoning.
    exact = _try_extract_strict(completion_text.strip())
    if exact:
        return exact

    # Strategy 3b: "ActionName: description" — colon-split, exact match only.
    # LLMs sometimes echo the action with its description from the prompt, e.g.
    # "Move North: move north on flat ground". Extract the part before the colon.
    if ":" in completion_text and "\n" not in completion_text.strip():
        before_colon = completion_text.split(":")[0].strip()
        result = _try_extract_strict(before_colon)
        if result:
            return result

    # Strategy 4: Text before the first XML tag is a valid action.
    # Handles models that omit <action> tags but place the action before their
    # communication/scratchpad blocks, either inline or newline-separated:
    #   "Move South <communication>..."   (same line, no newline)
    #   "Noop\n<communication>..."        (newline before tag)
    # Uses exact match only — no fuzzy — to avoid false positives.
    if "<" in completion_text:
        before_tag = completion_text[: completion_text.index("<")].strip()
        if before_tag:
            # Keep this path strict-only to avoid fuzzy false positives from
            # incidental prose before other tags.
            result = _try_extract_strict(before_tag)
            if result:
                return result

    # Strategy 5: First line is a valid action with no XML tags anywhere.
    # Handles "Move North\nnon-tagged reasoning text".
    # Uses exact match only.
    first_line = completion_text.strip().split("\n")[0].strip()
    if first_line and first_line != completion_text.strip():
        result = _try_extract_strict(first_line)
        if result:
            return result

    return None


# ============================================================================
# RobustNaiveAgent
# ============================================================================


class RobustNaiveAgent(BaseAgent):
    """An agent that generates actions based on observations without complex reasoning.

    Uses <action>...</action> XML tags for reliable action extraction,
    with multi-strategy fallback parsing and retry-with-feedback on failure.
    """

    MAX_RETRIES = 0  # Additional attempts after first failure

    def __init__(self, client_factory, prompt_builder):
        """Initialize the RobustNaiveAgent with a client and prompt builder."""
        super().__init__(client_factory, prompt_builder)
        self.step_count = 0
        self.total_retries = 0
        self.total_parse_failures = 0

    NAIVE_INSTRUCTION = (
        "You must choose exactly one action from the action list and output it in the following format:\n"
        "<action>YOUR_CHOSEN_ACTION</action>\n"
        "Output no other text, explanation, or reasoning."
    )

    def build_prompt(self, obs, prev_action=None):
        """Build the full prompt messages for this step (without calling the LLM)."""
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)
        messages = self.prompt_builder.get_prompt()
        if messages and messages[-1].role == "user":
            messages[-1].content += "\n\n" + self.NAIVE_INSTRUCTION
        return messages

    def act(self, obs, prev_action=None):
        """Generate the next action based on the observation and previous action.

        Args:
            obs (dict): The current observation in the environment.
            prev_action (str, optional): The previous action taken.

        Returns:
            LLMResponse: The response with the extracted action in `completion`.
        """
        if prev_action:
            self.prompt_builder.update_action(prev_action)

        self.prompt_builder.update_observation(obs)

        messages = self.prompt_builder.get_prompt()

        naive_instruction = self.NAIVE_INSTRUCTION

        if messages and messages[-1].role == "user":
            messages[-1].content += "\n\n" + naive_instruction

        _, last_response, extracted, retries = self.client.generate_with_validation(
            messages,
            validate_fn=lambda r: extract_action_multistrategy(r.completion),
            error_message=(
                "Your output did not contain a valid action. "
                "You must output exactly one action from the game's action list "
                "using the format: <action>YOUR_CHOSEN_ACTION</action>\n"
                "Try again."
            ),
            max_parse_retries=self.MAX_RETRIES,
        )
        self.step_count += 1
        self.total_retries += retries

        if extracted is None:
            self.total_parse_failures += 1
            extracted = "Noop"
            logger.warning(
                f"RobustNaiveAgent step {self.step_count}: failed to parse after "
                f"{retries + 1} attempts. Raw: '{last_response.completion[:200]}'. "
                f"Defaulting to Noop."
            )

        # Save raw outputs before extraction/cleaning (for debug logging)
        self._last_raw_completion = last_response.completion
        self._last_raw_reasoning = last_response.reasoning

        final_answer = self._extract_final_answer(last_response, extracted)

        if self.step_count % 50 == 0:
            logger.info(
                f"RobustNaiveAgent step {self.step_count}: action='{final_answer.completion}', "
                f"retries_total={self.total_retries}, failures={self.total_parse_failures}"
            )

        return final_answer

    def _extract_final_answer(self, answer, extracted_action):
        """Build the final LLMResponse with the extracted action.

        Args:
            answer (LLMResponse): The raw response from the LLM.
            extracted_action (str): The validated action name.

        Returns:
            LLMResponse: A copy of answer with `completion` set to extracted_action
                and `reasoning` set to the raw LLM output (for evaluator diagnostics).
        """
        final_answer = copy.deepcopy(answer)
        final_answer = final_answer._replace(
            reasoning=None,  # robust_naive doesn't reason; raw API data in _last_raw_reasoning
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
