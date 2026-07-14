"""Constrained-decoding client wrapper for OpenAI-compatible backends.

Small models (<=1B) fail Alem not because they cannot reason about the game, but
because they cannot reliably emit the `<action>...</action>` envelope the parser
expects. Observed sub-1B failures are near-misses on *syntax*:

    llama3.2:1b   -> "ACTION Move West"                (sensible move, no tags)
    qwen2.5:0.5b  -> "<action>Action 1: Do</action>"   (tags right, payload wrong)
    smollm2:360m  -> tutorial prose about how to descend
    gemma3:270m   -> "Okay, I'm ready to play!"

All four are scored as Noop, so the benchmark measures format compliance before
it measures competence.

This wrapper removes that confound. Generation is constrained to a JSON schema
whose `action` field is an enum of the game's real action list, making an invalid
action *unemittable* rather than merely discouraged. The constrained JSON is then
re-emitted as the exact XML the existing agents parse, so agent code, parse
statistics, and the memory/communication path all stay untouched.

Enabled by giving a client a `client_name` containing "constrained", e.g.
`clients.0.client_name=vllm_constrained`.
"""

import json
import logging

try:
    from .client import OpenAIWrapper
except ImportError:
    from eval_utils.client import OpenAIWrapper

logger = logging.getLogger(__name__)

# Source of truth for action names — same import the agents use.
try:
    from alem.llm.alem_language_wrapper import ACTIONS as _WRAPPER_ACTIONS
except ImportError:
    from baselines.llm.alem_language_wrapper import ACTIONS as _WRAPPER_ACTIONS

VALID_ACTIONS = list(_WRAPPER_ACTIONS)


def build_action_schema(want_communication=True, want_scratchpad=True):
    """Build a JSON schema constraining `action` to the valid action enum.

    Communication/scratchpad fields are only requested when the prompt actually
    asks for them, so the constrained response stays faithful to the agent's
    prompt rather than forcing tags the agent will never read.
    """
    properties = {"action": {"type": "string", "enum": VALID_ACTIONS}}
    required = ["action"]
    if want_communication:
        properties["communication"] = {"type": "string"}
        required.append("communication")
    if want_scratchpad:
        properties["scratchpad"] = {"type": "string"}
        required.append("scratchpad")
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def json_to_tagged(payload):
    """Re-emit a constrained JSON action as the XML envelope agents parse."""
    parts = [f"<action>{payload['action']}</action>"]
    if payload.get("communication"):
        parts.append(f"<communication>{payload['communication']}</communication>")
    if payload.get("scratchpad"):
        parts.append(f"<scratchpad>{payload['scratchpad']}</scratchpad>")
    return "\n".join(parts)


class ConstrainedOpenAIWrapper(OpenAIWrapper):
    """OpenAI-compatible wrapper that constrains actions via JSON schema.

    Behaves exactly like OpenAIWrapper except that every generation is forced to
    match `build_action_schema`, and the resulting JSON is translated back into
    `<action>`/`<communication>`/`<scratchpad>` tags before returning.
    """

    def __init__(self, client_config):
        super().__init__(client_config)
        # Strip the "_constrained" suffix so the parent's backend-specific paths
        # (base_url/api-key selection, extra_body) still match on the real name.
        self.client_name = self.client_name.lower().replace("_constrained", "").replace(
            "constrained_", ""
        )
        self._constrained_failures = 0

    def _wants(self, messages):
        """Detect which optional blocks the prompt actually asked for."""
        text = " ".join(m.content for m in messages if getattr(m, "content", None))
        return "<communication>" in text, "<scratchpad>" in text

    def generate(self, messages):
        want_comm, want_scratch = self._wants(messages)
        schema = build_action_schema(want_comm, want_scratch)

        prev = self.client_kwargs.get("response_format")
        self.client_kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "alem_action", "strict": True, "schema": schema},
        }
        try:
            response = super().generate(messages)
        finally:
            if prev is None:
                self.client_kwargs.pop("response_format", None)
            else:
                self.client_kwargs["response_format"] = prev

        raw = (response.completion or "").strip()
        if not raw:
            return response
        try:
            payload = json.loads(raw)
            tagged = json_to_tagged(payload)
        except (ValueError, KeyError, TypeError) as exc:
            # Schema-constrained output should always parse; if a backend ignores
            # response_format, fall through to the raw text so the agent's normal
            # multi-strategy parser still gets its chance.
            self._constrained_failures += 1
            logger.warning(f"constrained decode did not yield schema JSON ({exc}): {raw[:160]!r}")
            return response

        return response._replace(completion=tagged)
