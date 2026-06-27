"""
Prompt Builder for LLM Agents
Based on BALROG prompt_builder implementation
"""

import re
import warnings
from collections import deque
from typing import List, Optional


class Message:
    """Represents a conversation message with role, content, and optional attachment."""

    def __init__(self, role: str, content: str, attachment: object | None = None):
        self.role = role  # 'system', 'user', 'assistant'
        self.content = content  # String content of the message
        self.attachment = attachment

    def __repr__(self):
        return f"Message(role={self.role}, content={self.content}, attachment={self.attachment})"


class HistoryPromptBuilder:
    """Builds a prompt with a history of observations, actions, and reasoning."""

    def __init__(
        self,
        max_text_history: int = 16,
        max_image_history: int = 1,
        system_prompt: str | None = None,
        max_cot_history: int = 1,
    ):
        self.max_text_history = max_text_history
        self.max_image_history = max_image_history
        self.max_history = max(max_text_history, max_image_history)
        self.system_prompt = system_prompt
        self._events = deque(maxlen=self.max_history * 2)
        self._last_short_term_obs = None
        self.previous_reasoning = None
        self.max_cot_history = max_cot_history

    def update_instruction_prompt(self, instruction: str):
        """Set the system-level instruction prompt."""
        self.system_prompt = instruction

    def _extract_step_line(self, text):
        """Extract the 'Step: N/M ...' line from observation text.

        Returns (step_line, remaining_text). step_line is None if not found.
        Placing the step line first in the user message lets agents immediately
        see temporal context before the observation body.
        """
        if not text:
            return None, text
        lines = text.split("\n")
        step_line = None
        remaining = []
        for line in lines:
            if step_line is None and re.match(r"^Step:\s+\d+/\d+", line):
                step_line = line
            else:
                remaining.append(line)
        remaining_text = "\n".join(remaining).strip()
        return step_line, remaining_text

    def update_observation(self, obs: dict):
        """Add an observation to the prompt history."""
        long_term_context = obs["text"].get("long_term_context", "")
        self._last_short_term_obs = obs["text"].get("short_term_context", "")
        text = long_term_context
        image = obs.get("image", None)
        self._events.append(
            {
                "type": "observation",
                "text": text,
                "image": image,
            }
        )

    def update_action(self, action: str):
        """Add an action to the prompt history."""
        self._events.append(
            {
                "type": "action",
                "action": action,
                "reasoning": self.previous_reasoning,
            }
        )

    def update_reasoning(self, reasoning: str):
        """Set the reasoning text to be included with subsequent actions."""
        self.previous_reasoning = reasoning

    def reset(self):
        """Clear the event history."""
        self._events.clear()

    def get_prompt(
        self,
        icl_episodes=False,
        scratchpad_history=None,
        communication_history=None,
        max_text_history=None,
    ) -> list[Message]:
        """Generate a list of Message objects representing the prompt."""
        messages = []

        if self.system_prompt and not icl_episodes:
            messages.append(Message(role="system", content=self.system_prompt))

        # Determine which text observations to include
        text_needed = max_text_history if max_text_history is not None else self.max_text_history
        for event in reversed(self._events):
            if event["type"] == "observation":
                if text_needed > 0 and event.get("text") is not None:
                    event["include_text"] = True
                    text_needed -= 1
                else:
                    event["include_text"] = False

        # Determine which image observations to include
        images_needed = self.max_image_history
        for event in reversed(self._events):
            if event["type"] == "observation":
                if images_needed > 0 and event.get("image") is not None:
                    event["include_image"] = True
                    images_needed -= 1
                else:
                    event["include_image"] = False

        # Determine the reasoning to include
        reasoning_needed = self.max_cot_history
        for event in reversed(self._events):
            if event["type"] == "action":
                if reasoning_needed > 0 and event.get("reasoning") is not None:
                    reasoning_needed -= 1
                else:
                    event["reasoning"] = None

        # Process events to create messages
        for idx, event in enumerate(self._events):
            if event["type"] == "observation":
                message_parts = []

                idx_relative = len(self._events) - idx - 1
                is_current = idx_relative == 0
                step_relative = (
                    idx_relative // 2
                )  # Each step has an observation and an action, so divide by 2 to get step index
                step_line = None
                observation_text = event.get("text", "")
                if event.get("include_text", False):
                    step_line, observation_text = self._extract_step_line(observation_text)

                if step_line:
                    message_parts.append(step_line)

                if is_current:
                    message_parts.append("Current Observation:")
                else:
                    message_parts.append(f"Observation from {step_relative} step(s) ago:")

                # Environment state first (what changed, what you see)
                if event.get("include_text", False):
                    if observation_text:
                        message_parts.append(observation_text)

                # Inventory/status after (stable context)
                if is_current and self._last_short_term_obs:
                    message_parts.append("")
                    message_parts.append(self._last_short_term_obs)

                image = None
                if event.get("include_image", False):
                    image = event["image"]
                    message_parts.append("Image observation provided.")

                if scratchpad_history and step_relative < len(scratchpad_history):
                    scratchpad_entry = scratchpad_history[-(step_relative + 1)]
                    # enough space before scratchpad
                    if message_parts and message_parts[-1] != "":
                        message_parts.append("")
                    message_parts.append("Scratchpad:")
                    if scratchpad_entry:
                        message_parts.append(scratchpad_entry)
                    else:
                        message_parts.append("[Empty]")

                if communication_history is not None:
                    communication_entry = None
                    if step_relative < len(communication_history):
                        communication_entry = communication_history[-(step_relative + 1)]
                    if communication_entry:
                        # enough space before coms
                        if message_parts and message_parts[-1] != "":
                            message_parts.append("")
                        message_parts.append("Communication from other agent(s):")
                        for agent_idx in sorted(communication_entry):
                            comm = communication_entry[agent_idx]
                            message_parts.append(f"- Agent {agent_idx}: {comm}")

                content = "\n".join(message_parts)
                message = Message(role="user", content=content, attachment=image)

                # Clean up temporary flags
                for flag in ["include_text", "include_image"]:
                    if flag in event:
                        del event[flag]
            elif event["type"] == "action":
                if event.get("reasoning") is not None:
                    content = (
                        "Previous plan:\n"
                        + event["reasoning"]
                        + "\n\nAction taken: "
                        + event["action"]
                    )
                else:
                    content = event["action"]
                message = Message(role="assistant", content=content)
            messages.append(message)

        return messages


def create_prompt_builder(config):
    """Creates an instance of a prompt builder based on the provided configuration."""
    max_history = config.get("max_history", None)
    if max_history is not None:
        warnings.warn(
            "The 'max_history' parameter is deprecated. Please use 'max_text_history' instead."
        )

    max_text_history = max_history
    if max_text_history is None:
        max_text_history = config.max_text_history

    return HistoryPromptBuilder(
        max_text_history=max_text_history,
        max_image_history=config.max_image_history,
        max_cot_history=config.max_cot_history,
    )
