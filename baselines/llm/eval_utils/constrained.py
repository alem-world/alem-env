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

Enabled by appending "_constrained" to an OpenAI-compatible client name, e.g.
`clients.0.client_name=vllm_constrained`. The suffix is exact: a name that merely
contains the word (e.g. "vllm_unconstrained") is NOT constrained.
"""

import json
import logging

try:
    from .client import CONSTRAINED_SUFFIX, OpenAIWrapper
except ImportError:
    from eval_utils.client import CONSTRAINED_SUFFIX, OpenAIWrapper

logger = logging.getLogger(__name__)

# Free-text bounds, calibrated against the real corpus rather than assumed: across every
# episode logged in outputs/ (n=6,913 communications, n=6,528 scratchpads), communication
# runs median 55 / p99 310 chars and scratchpad median 73 / p99 504. These caps clear the
# 99th percentile with headroom while guaranteeing the object closes well inside a default
# max_tokens=768 budget (worst case ~1.5k chars, ~400 tokens). They exist to make the JSON
# TERMINATE — see build_action_schema for the failure they prevent.
COMMUNICATION_MAX_CHARS = 512
SCRATCHPAD_MAX_CHARS = 1024

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

    The free-text fields are LENGTH-BOUNDED, and that bound is load-bearing rather
    than cosmetic: constraining `action` alone still lets a small model pick a valid
    action instantly and then run away in `communication` until it hits max_tokens,
    leaving the JSON unterminated. json.loads then fails, the re-emission never runs,
    and the raw JSON leaks out untagged — so the constrained arm scores a PARSE FAILURE
    on a response whose action was perfectly valid. Observed on gemma3:270m: 58 of 100
    steps died exactly this way ('{"action": "Noop", "communication": "I am trying to
    get a better understanding of..." <768 tokens later, still going>'), dragging the
    constrained parse rate from 0.98 down to 0.65 on that seed. Bounding the enum is
    not enough; the tail has to terminate too.
    """
    properties = {"action": {"type": "string", "enum": VALID_ACTIONS}}
    required = ["action"]
    if want_communication:
        properties["communication"] = {
            "type": "string",
            "maxLength": COMMUNICATION_MAX_CHARS,
        }
        required.append("communication")
    if want_scratchpad:
        properties["scratchpad"] = {"type": "string", "maxLength": SCRATCHPAD_MAX_CHARS}
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
        # removesuffix, not replace(): replace() would also delete an occurrence in
        # the MIDDLE of a name, and it is the same suffix the factory routes on —
        # so the two can never disagree about what "constrained" means.
        self.client_name = self.client_name.lower().removesuffix(CONSTRAINED_SUFFIX)
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
