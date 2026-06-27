"""
Robust Agent for Alem LLM Evaluation.

Implements:
- Chain-of-Thought: use_cot: true in config to enable CoT reasoning with thinking blocks.
- Memory: use_scratchpad: true to store scratchpad memory
- Communication: use_communication: true to enable inter-agent communication with history.

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


def _extract_tagged(text, tag_name):
    """Extract content from <tag_name>...</tag_name> XML tags.

    Handles two patterns (in priority order):
      1. End tag present: <tag>content</tag>
      2. No end tag: content runs until the next opening tag or end of string
    """
    if not text:
        return None
    open_tag = rf"<{tag_name}\b[^>]*>"
    close_tag = rf"</{tag_name}\s*>"
    m = re.search(rf"{open_tag}(.*?){close_tag}", text, re.DOTALL | re.IGNORECASE)
    if m and m.group(1).strip():
        return m.group(1).strip()
    # Fallback: no end tag — grab until next opening tag or end of string
    m = re.search(rf"{open_tag}(.*?)(?=<[a-zA-Z/]|$)", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip() or None
    return None


def _strip_tagged(text, tag_name):
    """Remove <tag_name>...</tag_name> blocks (and unclosed variants) from text."""
    if not text:
        return text
    open_tag = rf"<{tag_name}\b[^>]*>"
    close_tag = rf"</{tag_name}\s*>"
    text = re.sub(rf"{open_tag}.*?{close_tag}", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(rf"{open_tag}.*?(?=<[a-zA-Z/]|$)", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text


class RobustAllAgent(BaseAgent):
    """An agent that can toggle between Chain-of-Thought and Naive execution."""

    def __init__(self, client_factory, prompt_builder, config, client_config):
        agent_id = config.agent.get("agent_id", None)
        super().__init__(client_factory, prompt_builder, agent_id=agent_id)
        # Toggles between CoT and Naive
        self.use_cot = config.agent.get("use_cot", True)
        self.prompt_mode = config.agent.get("prompt_mode", "specific_collaborative")
        self.remember_cot = config.agent.get("remember_cot", True)
        self.max_tokens = client_config.generate_kwargs.get("max_tokens", 8192)
        self.use_scratchpad = config.agent.get("use_scratchpad", False)
        self.structured_scratchpad = config.agent.get("structured_scratchpad", False)
        self.max_scratchpad_history = config.agent.get("max_scratchpad_history", 1)
        self.max_scratchpad_length = config.agent.get("max_scratchpad_length", 1000)
        self.use_communication = config.agent.get("use_communication", False)
        self.structured_communication = config.agent.get("structured_communication", False)
        self.max_communication_history = config.agent.get("max_communication_history", 4)
        self.max_communication_length = config.agent.get("max_communication_length", 400)
        # How many times to re-prompt when action parsing fails (0 = no retries).
        # Each retry sends the model's raw output back with a format error message.
        self.max_parse_retries = config.agent.get("max_parse_retries", 0)

        if self.use_communication:
            assert self.max_communication_history <= config.agent.max_text_history, (
                "Communication history must be less than or equal to overall text history to ensure it is included in prompts."
            )
        if self.use_scratchpad:
            assert self.max_scratchpad_history <= config.agent.max_text_history, (
                "Scratchpad history must be less than or equal to overall text history to ensure it is included in prompts."
            )

        # Thinking-mode: only for models with a separate reasoning field
        # (e.g. Qwen3 on vLLM with --reasoning-parser). Must be explicitly
        # enabled via config.agent.reasoning=True. GPT/Claude models do not
        # have .reasoning so they should always use the standard CoT path.
        self.enable_thinking = bool(config.agent.get("reasoning", False))

        # When True, output format instructions are placed in the system prompt
        # once rather than appended to every observation. Saves tokens over long
        # episodes. False preserves the legacy per-turn instruction behavior.
        self.instructions_in_system_prompt = bool(
            config.agent.get("instructions_in_system_prompt", False)
        )
        # Format instructions are injected via set_instruction_prompt() after the
        # game instruction prompt is set by the evaluator, NOT here at init.
        # Injecting here would be immediately overwritten when the evaluator calls
        # prompt_builder.update_instruction_prompt() with the game instructions.

        self.scratchpad_history = []
        self.communication_history = []
        self.current_communication = None
        self._was_inactive = False  # Tracks prior step's inactive state for revival detection

        self.step_count = 0
        self.total_retries = 0
        self.total_parse_failures = 0
        self._last_parse_failed = False  # set each act() call; read by evaluator for feedback
        # Track comm/scratchpad tag quality: of the steps where the model
        # attempted a tag, how often did it close it properly?
        # (comm/scratchpad are optional so "expected" = every step is wrong)
        self.comm_attempted = 0
        self.comm_parsed = 0
        self.scratchpad_attempted = 0
        self.scratchpad_parsed = 0
        self._warned_naive_mode = False

    def receive_communication(self, communication):
        """Receive a communication messages from all other agents. Needs to be called after all agents have acted for the step to ensure messages are included in the next step's prompt."""
        if self.use_communication:
            self.communication_history.append(communication)
            if len(self.communication_history) > self.max_communication_history:
                self.communication_history.pop(0)

    def _build_format_instructions(self):
        """Build the output format instruction block.

        Returns the full instruction text describing the expected response format
        (reasoning, communication, scratchpad, action tags). Used both for system
        prompt injection and per-turn appending depending on the mode.
        """
        parts = []
        step = 1

        parts.append("<output_format>")
        parts.append(
            "Each turn you receive an observation showing what you see, your inventory, teammates, and available actions."
        )

        # -- Preamble (mode-dependent) --
        if self.use_cot:
            parts.append("Think first, then output strictly in the following format:")
            if not self.enable_thinking:
                parts.append(
                    f"{step}. (Required) Brief reasoning wrapped in tags:\n"
                    "<think>YOUR_REASONING</think>"
                )
                step += 1
        else:
            if self.use_communication or self.use_scratchpad:
                parts.append("Respond in this order:")

        # -- Action (always required, always first) --
        parts.append(
            f"{step}. (Required) Exactly one action from the available action list:\n"
            "<action>YOUR_CHOSEN_ACTION</action>"
        )
        step += 1

        # -- Communication --
        if self.use_communication:
            _collab = self.prompt_mode == "specific_collaborative"
            _coord_note = (
                " Teammates can only act on what you tell them. "
                "Be specific (e.g. 'Dig on tree next turn', 'Ladder at 5NE', 'Need 2 wood')."
                " Reply to teammates' requests."
                if _collab
                else ""
            )
            if self.structured_communication:
                parts.append(
                    f"{step}. (Optional) Broadcast to teammates, up to {self.max_communication_length} chars.{_coord_note}\n"
                    "Suggested structure (use what's relevant, skip the rest):\n"
                    "  DOING: what you're doing this turn / next turn.\n"
                    "  FOUND: new info teammates can't see (locations, loot, potion effects).\n"
                    "  NEED: coordination requests — who, what, where, when.\n"
                    "<communication>YOUR_MESSAGE</communication>"
                )
            else:
                parts.append(
                    f"{step}. (Optional) Broadcast to teammates, up to {self.max_communication_length} chars.{_coord_note}\n"
                    "<communication>YOUR_MESSAGE</communication>"
                )
            step += 1

        if self.use_scratchpad:
            _collab = self.prompt_mode == "specific_collaborative"
            if self.structured_scratchpad:
                _team_section = (
                    "  TEAM: each teammate's last stated intent and relevant resources.\n"
                    if _collab and self.use_communication
                    else ""
                )
                parts.append(
                    f"{step}. (Optional) Private notes, up to {self.max_scratchpad_length} chars — not shared with teammates. "
                    "Your context resets each turn — this is your only memory. "
                    "Don't repeat what's already in your observation; store what you'll need later.\n"
                    "Suggested structure (use what's relevant, skip the rest):\n"
                    "  PLAN: your goal + next 2-3 steps.\n"
                    "  LEARNED: discovered facts (off-screen locations, what worked/failed).\n"
                    + _team_section
                    + "<scratchpad>YOUR_NOTES</scratchpad>"
                )
            else:
                _team_note = (
                    " Record teammates' plans and any facts you'll need after they scroll out of view."
                    if _collab and self.use_communication
                    else ""
                )
                parts.append(
                    f"{step}. (Optional) Private notes, up to {self.max_scratchpad_length} chars — not shared with teammates. "
                    "Your context resets each turn — this is your only memory. "
                    f"Don't repeat what's already in your observation; store what you'll need later.{_team_note}\n"
                    "<scratchpad>YOUR_NOTES</scratchpad>"
                )
            step += 1

        if not self.use_cot:
            if self.use_communication or self.use_scratchpad:
                parts.append("Output only the tags listed above in the order shown.")
            else:
                parts.append("Output no other text, explanation, or reasoning.")

        # talor example to action type.
        if self.use_communication:
            parts.append(
                "Important: every tag you open must be closed (e.g. <communication>...</communication>)."
            )
        elif self.use_scratchpad:
            parts.append(
                "Important: every tag you open must be closed (e.g. <scratchpad>...</scratchpad>)."
            )
        else:
            parts.append(
                "Important: every tag you open must be closed (e.g. <action>...</action>)."
            )

        if self.use_cot:
            parts.append(
                f"Token budget: {self.max_tokens} tokens for your full response (including reasoning). "
                "Keep reasoning concise and stop thinking early enough to emit every required tag — "
                "if you exhaust the budget mid-reasoning, no action is produced and your turn fails."
            )
        else:
            parts.append(
                f"Token budget: {self.max_tokens} tokens for your full response. "
                "Keep the response concise and emit every required tag."
            )
        parts.append("</output_format>")

        return "\n".join(parts)

    def _build_turn_reminder(self):
        """Build a short per-turn reminder when instructions are in the system prompt."""
        tags = ["<action>"]
        if self.use_cot and not self.enable_thinking:
            tags = ["<think>"] + tags
        if self.use_communication:
            tags.append("<communication>")
        if self.use_scratchpad:
            tags.append("<scratchpad>")
        return (
            "\n\n---\nResponse format (in this order): "
            + " , ".join(tags)
            + ". Close every tag you open."
        )

    def build_prompt(self, obs, prev_action=None):
        """Build the full prompt messages for this step (without calling the LLM).

        Updates prompt history with the previous action and current observation,
        then returns the complete message list including agent-specific
        instructions. Useful for debugging / human-play inspection.
        """
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)

        # Inactive agents (dead or sleeping) can only Noop. Sending their full
        # observation history and scratchpad wastes tokens with zero decision value.
        # Reduce to a single observation and skip scratchpad; keep the last
        # communication entry so they can still coordinate with alive teammates.
        is_inactive = obs.get("is_inactive", False)

        # Resume annotation (inactive→active): is_inactive covers death, sleep, and
        # rest — all three gate out decision-making but only death resets position.
        # Use a generic resume note so the annotation is correct for every path.
        if self._was_inactive and not is_inactive and self.scratchpad_history:
            self.scratchpad_history[-1] = (
                "[Resuming after idle period (death, sleep, rest)."
                "Re-read your prior notes — position and health may have changed. Strategic plans and teammate knowledge are still valid."
                "]\n\n" + self.scratchpad_history[-1]
            )

        kwargs = {}
        if not is_inactive:
            if self.use_scratchpad and self.scratchpad_history:
                kwargs["scratchpad_history"] = self.scratchpad_history
            if self.use_communication and self.communication_history:
                kwargs["communication_history"] = self.communication_history
        elif self.use_communication and self.communication_history:
            kwargs["communication_history"] = self.communication_history[-1:]

        # Dead agents can only Noop but may be revived later.
        # Use a small history (quarter of normal, min 2) rather than 1 so the
        # agent isn't completely context-free on revival: the pre-death scratchpad
        # survives intact, and only a handful of dead-state observations fill the
        # window before alive observations start replacing them.
        dead_history = max(2, self.prompt_builder.max_text_history // 4)
        messages = self.prompt_builder.get_prompt(
            **kwargs,
            max_text_history=dead_history if is_inactive else None,
        )

        if self.instructions_in_system_prompt:
            if messages and messages[-1].role == "user":
                messages[-1].content += self._build_turn_reminder()
        else:
            self._append_instructions(messages)

        return messages

    def set_instruction_prompt(self, new_prompt):
        """Update the base instruction prompt and re-inject format instructions if needed.

        Use this instead of prompt_builder.update_instruction_prompt() directly so
        that format instructions survive system prompt refreshes (e.g. progressive
        disclosure on level change).
        """
        self.prompt_builder.update_instruction_prompt(new_prompt)
        if self.instructions_in_system_prompt:
            self._inject_system_instructions()

    def _inject_system_instructions(self):
        instructions = self._build_format_instructions()
        sp = self.prompt_builder.system_prompt or ""
        sp = re.sub(r"\n*<output_format>.*?</output_format>", "", sp, flags=re.DOTALL)
        self.prompt_builder.system_prompt = sp.rstrip() + "\n\n" + instructions

    def _append_instructions(self, messages):
        """Append agent-specific instructions to the last user message.

        Builds a single, consistently-formatted instruction block with numbered
        steps.  The order is always: reasoning → action → communication → scratchpad.
        Only enabled steps are included; numbering adjusts automatically.
        """
        if not self.use_cot and not self._warned_naive_mode:
            logger.warning(
                "RobustAllAgent is running in Naive mode without CoT reasoning, "
                "this might lead to the model not using messages or scratchpad "
                "effectively due to cutting off after action."
            )
            self._warned_naive_mode = True

        instruction = self._build_format_instructions()
        if messages and messages[-1].role == "user":
            messages[-1].content += "\n" + instruction

    def act(self, obs, prev_action=None):
        messages = self.build_prompt(obs, prev_action)

        error_msg = (
            "Your output did not contain a valid action. "
            "You must include exactly one action using: <action>YOUR_CHOSEN_ACTION</action>\n"
        )
        if self.use_cot:
            if self.enable_thinking:
                error_msg += (
                    "Keep your reasoning brief and focus on outputting a valid action. "
                    "Important: your latest retry response is used for memory/communication/scratchpad extraction."
                )
            else:
                error_msg += (
                    "Use this exact visible format in your latest retry response: "
                    "<think>brief reasoning</think> then <action>YOUR_CHOSEN_ACTION</action>. "
                    "If you include optional tags, close them properly. "
                    "Important: your latest retry response is used for memory/communication/scratchpad extraction."
                )
        else:
            error_msg += "Try again. Output no other text."

        # Generate with action-parse retries; comm/scratchpad failures are tracked below.
        _, last_response, extracted, retries = self.client.generate_with_validation(
            messages,
            validate_fn=lambda r: extract_action_multistrategy(r.completion),
            error_message=error_msg,
            max_parse_retries=self.max_parse_retries,
        )

        self.step_count += 1
        self.total_retries += retries

        self._last_parse_failed = extracted is None
        if self._last_parse_failed:
            self.total_parse_failures += 1
            logger.warning(
                f"RobustAllAgent {self.agent_id} step {self.step_count}: "
                f"failed to parse action after {retries + 1} attempt(s). "
                f"Raw: '{last_response.completion[:200]}'. Defaulting to Noop."
            )
            extracted = "Noop"

        # Save raw outputs for debug logging
        self._last_raw_completion = last_response.completion
        self._last_raw_reasoning = last_response.reasoning

        # 4. Handle Memory based on flag
        if self.use_cot and self.remember_cot:
            if last_response.reasoning:
                # Thinking-mode model (e.g. Qwen3 with --reasoning-parser):
                # reasoning is already separated into the .reasoning field.
                # Only trust .reasoning when enable_thinking is on — some models
                # return residual reasoning_content even with enable_thinking=false.
                reasoning_for_memory = last_response.reasoning
            else:
                # Standard model fallback: reasoning can appear anywhere in the
                # completion when comm/scratchpad tags are enabled.
                reasoning_for_memory = self._extract_reasoning_from_completion(
                    last_response.completion, extracted
                )
            self.prompt_builder.update_reasoning(reasoning_for_memory)

        # Extract comm/scratchpad, track misses, apply results.
        # Skip for inactive agents (dead/sleeping) — they can only Noop, so
        # storing scratchpad entries would just pollute history with stale state.
        is_inactive = obs.get("is_inactive", False)

        self._was_inactive = is_inactive

        communication, scratchpad_entry = self._extract_comm_scratchpad(last_response)

        if self.use_communication:
            self.current_communication = communication

        if self.use_scratchpad and not is_inactive:
            if scratchpad_entry:
                self.scratchpad_history.append(scratchpad_entry)
                if len(self.scratchpad_history) > self.max_scratchpad_history:
                    self.scratchpad_history.pop(0)

        final_answer = self._extract_final_answer(last_response, extracted)

        reasoning_preview = (final_answer.reasoning or "")[:300]
        logger.debug(
            f"[step {self.step_count}] action='{final_answer.completion}' retries={retries}\n"
            f"  reasoning: {reasoning_preview!r}"
        )

        if self.step_count % 50 == 0:
            logger.info(
                f"RobustAllAgent {self.agent_id} step {self.step_count}: action='{final_answer.completion}', "
                f"retries_total={self.total_retries}, failures={self.total_parse_failures}"
            )

        return final_answer

    def _extract_comm_scratchpad(self, response):
        """Extract communication and scratchpad from the completion text.

        Only searches completion (not reasoning) — these tags belong in visible
        output. Uses _extract_tagged() which tolerates missing angle brackets
        on end tags. Also tracks attempted vs parsed counts for parse rate stats.

        Returns:
            (communication, scratchpad_entry) — either may be None if the tag
            was absent or if the feature is disabled.
        """
        communication = None
        scratchpad_entry = None
        raw = response.completion or ""

        # Per-step flags for evaluator feedback
        self._last_comm_failed = False
        self._last_scratchpad_failed = False

        if self.use_communication:
            # Count as "attempted" if the model produced an opening tag at all
            if "<communication>" in raw:
                self.comm_attempted += 1
                communication = _extract_tagged(raw, "communication")
                if communication:
                    self.comm_parsed += 1
                    if self.max_communication_length > 0:
                        communication = communication[: self.max_communication_length]
                else:
                    self._last_comm_failed = True

        if self.use_scratchpad:
            if "<scratchpad>" in raw:
                self.scratchpad_attempted += 1
                scratchpad_entry = _extract_tagged(raw, "scratchpad")
                if scratchpad_entry:
                    self.scratchpad_parsed += 1
                    if self.max_scratchpad_length > 0:
                        scratchpad_entry = scratchpad_entry[: self.max_scratchpad_length]
                else:
                    self._last_scratchpad_failed = True

        return communication, scratchpad_entry

    @staticmethod
    def _clean_reasoning(raw_text, extracted_action):
        """Clean reasoning text: strip output tags and bare action echoes.

        Handles common failure modes from small models:
          - Model outputs action/comm/scratchpad tags inside reasoning text
          - Bare action echo: model outputs just "Move South" with no reasoning
          - ACTION: prefix leftovers
        """
        if not raw_text:
            return None
        cleaned = re.sub(r"</?think\b[^>]*>", "", raw_text, flags=re.IGNORECASE).strip()
        if not cleaned:
            return None
        cleaned = _strip_tagged(cleaned, "communication")
        cleaned = _strip_tagged(cleaned, "scratchpad")

        # Strip any remaining XML output tags
        cleaned = re.sub(r"</?(?:action|communication|scratchpad)>", "", cleaned).strip()
        if not cleaned:
            return None
        # If what remains is just the action name, there's no actual reasoning
        if cleaned.lower() == extracted_action.lower():
            return None
        return cleaned

    def _extract_reasoning_from_completion(self, completion_text, extracted_action):
        """Extract reasoning from completion regardless of output tag order."""
        if not completion_text:
            return None
        tagged_think = _extract_tagged(completion_text, "think")
        if tagged_think is not None:
            return self._clean_reasoning(tagged_think, extracted_action)
        raw = _strip_tagged(completion_text, "action")
        raw = _strip_tagged(raw, "communication")
        raw = _strip_tagged(raw, "scratchpad")
        return self._clean_reasoning(raw, extracted_action)

    def _extract_final_answer(self, response, extracted_action):
        final_answer = copy.deepcopy(response)

        if self.use_cot:
            if response.reasoning:
                # Preserve model-provided reasoning as-is when available.
                # e.g. Qwen3 models use .reasoning for reasoning output
                reasoning_text = response.reasoning
            else:
                # Standard CoT fallback: strip tagged outputs and keep the
                # remaining free-form reasoning text.
                reasoning_text = self._extract_reasoning_from_completion(
                    response.completion, extracted_action
                )
        else:
            # Naive fallback: store raw output as diagnostic "reasoning"
            reasoning_text = ""

        final_answer = final_answer._replace(
            reasoning=reasoning_text,
            completion=extracted_action,
        )
        return final_answer

    def build_debrief_prompt(
        self,
        agent_id: int,
        role: str,
        final_scratchpad: str,
        achievements: list,
        total_achievements: int,
        max_achievements: int,
        steps_survived: int,
        max_steps: int,
        deepest_level: int,
        death_cause: str,
        inventory_at_death: str,
        stats_at_death: dict,
        communication_log_sample: list = None,
    ) -> str:
        """Build a structured post-episode debrief prompt.

        Returns a prompt string that asks the LLM to reflect on its actual
        experience in a specific, non-generic way. Anchors the reflection to
        the final scratchpad (accumulated episode knowledge) to avoid recency
        bias and generic platitudes.
        """
        achieved_str = "\n".join(f"  - {a}" for a in achievements) if achievements else "  (none)"
        stats_str = (
            ", ".join(f"{k}: {v}" for k, v in stats_at_death.items()) if stats_at_death else ""
        )

        comms_section = ""
        if communication_log_sample:
            comms_str = "\n".join(f"  {msg}" for msg in communication_log_sample[-10:])
            comms_section = f"\nLast messages before death:\n{comms_str}\n"

        scratchpad_section = (
            f"\nYour final scratchpad:\n{final_scratchpad}\n"
            if final_scratchpad
            else "\n(No scratchpad recorded.)\n"
        )

        return f"""The game is over. All agents have died.

You were Agent {agent_id} ({role}). Here is your final state:

Survived: {steps_survived}/{max_steps} steps
Deepest level reached: {deepest_level}/9
Achievements: {total_achievements}/{max_achievements}
{achieved_str}
Cause of death: {death_cause}
Inventory at death: {inventory_at_death}
Stats at death: {stats_str}
{comms_section}{scratchpad_section}
Answer each question in 1-3 sentences. Be specific — name exact situations, \
step ranges, items, or locations. Do not give generic advice.

INSTRUCTIONS FEEDBACK:
1. Were any instructions in the system prompt wrong, misleading, or \
contradictory? (e.g. "it said X but actually Y happened")
2. Was anything important missing from the instructions that you had to \
figure out by trial and error?
3. Were the crafting recipes, coordination rules, or role descriptions \
accurate and clear?

OBSERVATION FEEDBACK:
4. Was there information you needed but couldn't see in your observations? \
(e.g. exact coordinates, teammate inventories, enemy stats)
5. Did you struggle with spatial reasoning? If so, what specifically was \
hard? (e.g. navigating to a location, understanding relative positions, \
knowing which direction to face)

COORDINATION FEEDBACK:
6. Describe one specific failed coordination attempt: what were you trying \
to do, what did each agent do, and why did it fail?
7. Were teammate messages useful? What information did you wish teammates \
would share but didn't?

GAME DIFFICULTY:
8. What was the hardest bottleneck that blocked your progression? \
(e.g. couldn't find resource X, couldn't craft Y, kept dying to Z)
9. At what step range did you feel "stuck" with no clear next action? \
What were you trying to do?
10. If you could change ONE thing about the game interface or information \
you receive, what would have the biggest impact on your performance?

Format your response as:
<debrief>
1. ...
2. ...
...
10. ...
</debrief>"""

    def get_debrief(self, **kwargs) -> str | None:
        """Call the LLM with a debrief prompt and return the extracted <debrief> content.

        kwargs are forwarded directly to build_debrief_prompt(). Returns None
        if the client is unavailable or the model produces no parseable debrief.
        """
        if self.client is None:
            return None
        try:
            prompt = self.build_debrief_prompt(**kwargs)
            # Single-turn: system prompt is the game instructions, user is the debrief
            try:
                from ..prompt_builder import Message
            except ImportError:
                from eval_utils.prompt_builder import Message
            messages = [Message(role="user", content=prompt)]
            response = self.client.generate(messages)
            logger.info(
                f"Agent {self.agent_id} debrief raw: stop_reason={response.stop_reason!r} "
                f"input_tokens={response.input_tokens} output_tokens={response.output_tokens} "
                f"reasoning_chars={len(response.reasoning) if response.reasoning else 0} "
                f"completion_chars={len(response.completion)}"
            )
            text = response.completion.strip()
            # Try tag first (in case model wraps it), fall back to full completion
            return _extract_tagged(text, "debrief") or (text if text else None)
        except Exception as e:
            logger.warning(f"Agent {self.agent_id} debrief failed: {e}")
            return None

    def reset(self):
        """Reset the agent state for a new episode."""
        super().reset()
        self.scratchpad_history = []
        self.communication_history = []
        self.current_communication = None
        self._was_inactive = False
        self.step_count = 0
        self.total_retries = 0
        self.total_parse_failures = 0
        self._last_parse_failed = False
        self.comm_attempted = 0
        self.comm_parsed = 0
        self.scratchpad_attempted = 0
        self.scratchpad_parsed = 0

    def get_retry_stats(self):
        """Return retry/parse statistics for logging and analysis."""
        stats = {
            "total_steps": self.step_count,
            "total_retries": self.total_retries,
            "total_parse_failures": self.total_parse_failures,
            "retry_rate": round(self.total_retries / max(self.step_count, 1), 4),
            "action_parse_rate": round(
                1.0 - self.total_parse_failures / max(self.step_count, 1), 4
            ),
        }
        if self.use_communication:
            stats["comm_attempted"] = self.comm_attempted
            stats["comm_parsed"] = self.comm_parsed
            stats["comm_parse_rate"] = round(self.comm_parsed / max(self.comm_attempted, 1), 4)
        if self.use_scratchpad:
            stats["scratchpad_attempted"] = self.scratchpad_attempted
            stats["scratchpad_parsed"] = self.scratchpad_parsed
            stats["scratchpad_parse_rate"] = round(
                self.scratchpad_parsed / max(self.scratchpad_attempted, 1), 4
            )
        return stats
