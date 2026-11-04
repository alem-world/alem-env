"""Unit tests for the constrained-decoding client wrapper.

Tests cover:
- build_action_schema: action enum is the real action list, not a hand-copied one
- build_action_schema: optional blocks requested only when the prompt asks for them
- json_to_tagged: re-emits the XML envelope the existing agents parse
- json_to_tagged: omits optional blocks the model left empty
- create_llm_client: "vllm_constrained" routes to the constrained wrapper rather
  than falling through to the plain OpenAI wrapper (it also matches "vllm")
- ConstrainedOpenAIWrapper: strips the suffix so the parent's backend-specific
  paths still match on the real client name
"""

import unittest

from omegaconf import OmegaConf

from baselines.llm.eval_utils.client import OpenAIWrapper, create_llm_client
from baselines.llm.eval_utils.constrained import (
    VALID_ACTIONS,
    ConstrainedOpenAIWrapper,
    build_action_schema,
    json_to_tagged,
)


def _client_config(client_name):
    return OmegaConf.create(
        {
            "client_name": client_name,
            "model_id": "test-model",
            "base_url": "http://localhost:11434/v1",
            "timeout": 60,
            "generate_kwargs": {"max_tokens": 768},
            "max_retries": 1,
            "delay": 0,
            "alternate_roles": False,
        }
    )


class TestBuildActionSchema(unittest.TestCase):
    def test_action_enum_is_the_real_action_list(self):
        """The enum must come from the wrapper's ACTIONS, not a copy that can drift."""
        from alem.llm.alem_language_wrapper import ACTIONS

        schema = build_action_schema()
        self.assertEqual(schema["properties"]["action"]["enum"], list(ACTIONS))
        self.assertEqual(VALID_ACTIONS, list(ACTIONS))

    def test_enum_contains_known_actions(self):
        enum = build_action_schema()["properties"]["action"]["enum"]
        for action in ("Noop", "Do", "Move West"):
            self.assertIn(action, enum)

    def test_schema_is_strict(self):
        schema = build_action_schema()
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["type"], "object")

    def test_free_text_fields_are_length_bounded(self):
        """Regression: constraining `action` alone is not enough — the tail must terminate.

        gemma3:270m picked a valid action instantly, then rambled in `communication`
        until max_tokens, leaving the JSON unterminated: json.loads failed, the XML
        re-emission never ran, and the raw JSON leaked out untagged, scoring a PARSE
        FAILURE on a response whose action was perfectly valid. 58/100 steps died this
        way, pulling the constrained parse rate to 0.65 vs 0.98 unconstrained on the
        same seed — i.e. the "fix" measured WORSE than doing nothing.

        Bounds are calibrated to the real corpus (p99 = 310 / 504 chars), not invented,
        and must stay generous enough not to truncate normal play.
        """
        schema = build_action_schema()
        for field, p99 in (("communication", 310), ("scratchpad", 504)):
            prop = schema["properties"][field]
            self.assertIn("maxLength", prop, f"{field} is unbounded — the JSON can run away")
            self.assertGreater(prop["maxLength"], p99, f"{field} cap truncates normal play")
        total = sum(schema["properties"][f]["maxLength"] for f in ("communication", "scratchpad"))
        self.assertLess(total, 2048, "bounds must close the object inside max_tokens=768")

    def test_optional_blocks_requested_when_wanted(self):
        schema = build_action_schema(want_communication=True, want_scratchpad=True)
        self.assertEqual(sorted(schema["properties"]), ["action", "communication", "scratchpad"])
        self.assertEqual(sorted(schema["required"]), ["action", "communication", "scratchpad"])

    def test_optional_blocks_omitted_when_not_wanted(self):
        schema = build_action_schema(want_communication=False, want_scratchpad=False)
        self.assertEqual(list(schema["properties"]), ["action"])
        self.assertEqual(schema["required"], ["action"])

    def test_only_communication_wanted(self):
        schema = build_action_schema(want_communication=True, want_scratchpad=False)
        self.assertEqual(sorted(schema["properties"]), ["action", "communication"])


class TestJsonToTagged(unittest.TestCase):
    def test_emits_all_blocks(self):
        out = json_to_tagged(
            {"action": "Move West", "communication": "heading west", "scratchpad": "plan"}
        )
        self.assertIn("<action>Move West</action>", out)
        self.assertIn("<communication>heading west</communication>", out)
        self.assertIn("<scratchpad>plan</scratchpad>", out)

    def test_action_only(self):
        self.assertEqual(json_to_tagged({"action": "Do"}), "<action>Do</action>")

    def test_empty_optional_blocks_are_omitted(self):
        """An empty string must not produce a stray empty tag."""
        out = json_to_tagged({"action": "Do", "communication": "", "scratchpad": ""})
        self.assertEqual(out, "<action>Do</action>")

    def test_every_valid_action_round_trips_through_the_real_parser(self):
        """The core claim: a schema-valid action is always parseable by the agent.

        If the enum can emit an action the existing parser cannot read back, the
        wrapper would trade one silent failure for another.
        """
        from eval_utils.agents.robust_naive import extract_action_multistrategy

        for action in VALID_ACTIONS:
            with self.subTest(action=action):
                tagged = json_to_tagged(
                    {"action": action, "communication": "hi", "scratchpad": "note"}
                )
                self.assertEqual(extract_action_multistrategy(tagged), action)


class TestClientFactoryRouting(unittest.TestCase):
    def test_vllm_constrained_routes_to_constrained_wrapper(self):
        """Regression: "vllm_constrained" also matches "vllm" — order matters."""
        client = create_llm_client(_client_config("vllm_constrained"))()
        self.assertIsInstance(client, ConstrainedOpenAIWrapper)

    def test_plain_vllm_is_not_constrained(self):
        client = create_llm_client(_client_config("vllm"))()
        self.assertIsInstance(client, OpenAIWrapper)
        self.assertNotIsInstance(client, ConstrainedOpenAIWrapper)

    def test_unconstrained_name_is_not_routed_as_constrained(self):
        """Regression: "vllm_unconstrained" CONTAINS "constrained".

        A substring test routed it to the constrained wrapper — silently giving a
        caller who explicitly asked for UNCONSTRAINED decoding the constrained one.
        It compounded: the old chained-replace() strip left the name as
        "vllm_unconstrained" (no "_constrained" substring to remove), no branch in
        _initialize_client matched "vllm", and self.client was never assigned while
        _initialized was still set True — so it failed at generate() with an
        AttributeError rather than at config time.
        """
        client = create_llm_client(_client_config("vllm_unconstrained"))()
        self.assertNotIsInstance(client, ConstrainedOpenAIWrapper)
        self.assertIsInstance(client, OpenAIWrapper)

    def test_suffix_stripped_so_backend_paths_still_match(self):
        client = create_llm_client(_client_config("vllm_constrained"))()
        self.assertEqual(client.client_name, "vllm")

    def test_stripped_name_is_one_the_parent_actually_initializes(self):
        """The strip and the routing must agree, or we get a client with no .client.

        Pins the compound failure rather than just the routing: whatever name the
        factory routes as constrained must strip to a backend _initialize_client
        recognises, so the wrapper ends up with a live self.client.
        """
        client = create_llm_client(_client_config("vllm_constrained"))()
        client._initialize_client()
        self.assertTrue(
            hasattr(client, "client"),
            "constrained wrapper initialized without a backend client",
        )


if __name__ == "__main__":
    unittest.main()
