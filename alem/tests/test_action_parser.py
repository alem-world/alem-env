"""Tests for action_parser.py — action extraction from LLM text output.

Covers all extraction strategies: xml tags, ACTION: format, exact match,
colon-split, fuzzy match, and Give-target normalization.
"""

import unittest

from alem.llm.action_parser import (
    _clean_extracted,
    _normalize_give_target,
    _try_extract_strict,
    _try_extract_valid,
    extract_action_multistrategy,
    fuzzy_match_action,
    validate_action,
)

_VALID = [
    "Noop",
    "Move North",
    "Move South",
    "Move East",
    "Move West",
    "Do",
    "Place Table",
    "Chop Tree",
    "Mine Stone",
]


class TestValidateAction(unittest.TestCase):
    def test_exact_match_returns_canonical(self):
        self.assertEqual(validate_action("Move North", _VALID), "Move North")

    def test_case_insensitive_match(self):
        self.assertEqual(validate_action("move north", _VALID), "Move North")

    def test_all_caps_match(self):
        self.assertEqual(validate_action("MOVE NORTH", _VALID), "Move North")

    def test_unknown_action_returns_none(self):
        self.assertIsNone(validate_action("Jump", _VALID))

    def test_empty_string_returns_none(self):
        self.assertIsNone(validate_action("", _VALID))

    def test_whitespace_only_returns_none(self):
        self.assertIsNone(validate_action("   ", _VALID))

    def test_none_input_returns_none(self):
        self.assertIsNone(validate_action(None, _VALID))

    def test_whitespace_stripped_before_match(self):
        self.assertEqual(validate_action("  Move North  ", _VALID), "Move North")

    def test_noop_exact(self):
        self.assertEqual(validate_action("Noop", _VALID), "Noop")


class TestFuzzyMatchAction(unittest.TestCase):
    def test_close_typo_match_returned(self):
        result = fuzzy_match_action("Move Norht", _VALID)
        self.assertEqual(result, "Move North")

    def test_no_match_below_cutoff_returns_none(self):
        self.assertIsNone(fuzzy_match_action("xyzabc123", _VALID))

    def test_empty_string_returns_none(self):
        self.assertIsNone(fuzzy_match_action("", _VALID))

    def test_none_returns_none(self):
        self.assertIsNone(fuzzy_match_action(None, _VALID))

    def test_high_cutoff_rejects_close_match(self):
        self.assertIsNone(fuzzy_match_action("Move Norht", _VALID, cutoff=0.99))

    def test_exact_match_at_low_cutoff(self):
        self.assertEqual(fuzzy_match_action("Move North", _VALID, cutoff=0.1), "Move North")


class TestCleanExtracted(unittest.TestCase):
    def test_strips_leading_whitespace(self):
        self.assertEqual(_clean_extracted("  Move North"), "Move North")

    def test_strips_double_quotes(self):
        self.assertEqual(_clean_extracted('"Move North"'), "Move North")

    def test_strips_single_quotes(self):
        self.assertEqual(_clean_extracted("'Move North'"), "Move North")

    def test_strips_backticks(self):
        self.assertEqual(_clean_extracted("`Move North`"), "Move North")

    def test_strips_trailing_period(self):
        self.assertEqual(_clean_extracted("Move North."), "Move North")

    def test_strips_trailing_comma(self):
        self.assertEqual(_clean_extracted("Move North,"), "Move North")

    def test_strips_trailing_exclamation(self):
        self.assertEqual(_clean_extracted("Move North!"), "Move North")

    def test_strips_multiple_trailing_punct(self):
        self.assertEqual(_clean_extracted("Move North.,;"), "Move North")

    def test_plain_text_unchanged(self):
        self.assertEqual(_clean_extracted("Move North"), "Move North")


class TestNormalizeGiveTarget(unittest.TestCase):
    def test_give_to_agent_0(self):
        self.assertEqual(_normalize_give_target("Give to Agent 0"), "Give to Agent 0")

    def test_give_to_agent_2(self):
        self.assertEqual(_normalize_give_target("give to agent 2"), "Give to Agent 2")

    def test_give_teammate(self):
        self.assertEqual(_normalize_give_target("give teammate 1"), "Give to Agent 1")

    def test_give_agent_hyphen(self):
        self.assertEqual(_normalize_give_target("give agent-1"), "Give to Agent 1")

    def test_give_agent_underscore(self):
        self.assertEqual(_normalize_give_target("give agent_2"), "Give to Agent 2")

    def test_non_give_action_returns_none(self):
        self.assertIsNone(_normalize_give_target("Move North"))

    def test_empty_string_returns_none(self):
        self.assertIsNone(_normalize_give_target(""))

    def test_none_returns_none(self):
        self.assertIsNone(_normalize_give_target(None))

    def test_extra_whitespace_ok(self):
        self.assertEqual(_normalize_give_target("  give  to  agent  3  "), "Give to Agent 3")


class TestTryExtractStrict(unittest.TestCase):
    def test_exact_action(self):
        self.assertEqual(_try_extract_strict("Move North", _VALID), "Move North")

    def test_case_insensitive(self):
        self.assertEqual(_try_extract_strict("move north", _VALID), "Move North")

    def test_give_form_recognized(self):
        self.assertEqual(_try_extract_strict("give agent 1", _VALID), "Give to Agent 1")

    def test_unknown_returns_none(self):
        self.assertIsNone(_try_extract_strict("FlyAway", _VALID))

    def test_empty_returns_none(self):
        self.assertIsNone(_try_extract_strict("", _VALID))

    def test_none_returns_none(self):
        self.assertIsNone(_try_extract_strict(None, _VALID))


class TestTryExtractValid(unittest.TestCase):
    def test_exact_match(self):
        self.assertEqual(_try_extract_valid("Move North", _VALID), "Move North")

    def test_give_form_direct(self):
        self.assertEqual(_try_extract_valid("give to agent 2", _VALID), "Give to Agent 2")

    def test_colon_split_exact_before_colon(self):
        result = _try_extract_valid("Move North: because I need to explore", _VALID)
        self.assertEqual(result, "Move North")

    def test_colon_split_give_before_colon(self):
        result = _try_extract_valid("give agent 0: share item", _VALID)
        self.assertEqual(result, "Give to Agent 0")

    def test_fuzzy_fallback_on_typo(self):
        result = _try_extract_valid("Move Norht", _VALID)
        self.assertEqual(result, "Move North")

    def test_colon_split_with_fuzzy(self):
        result = _try_extract_valid("Move Norht: reason here", _VALID)
        self.assertEqual(result, "Move North")

    def test_nothing_matches_returns_none(self):
        self.assertIsNone(_try_extract_valid("xyzabc123456789", _VALID))

    def test_empty_returns_none(self):
        self.assertIsNone(_try_extract_valid("", _VALID))

    def test_none_returns_none(self):
        self.assertIsNone(_try_extract_valid(None, _VALID))


class TestExtractActionMultistrategy(unittest.TestCase):
    def test_closed_xml_tag_exact(self):
        self.assertEqual(
            extract_action_multistrategy("<action>Move North</action>", _VALID),
            "Move North",
        )

    def test_closed_xml_tag_case_insensitive(self):
        self.assertEqual(
            extract_action_multistrategy("<action>move north</action>", _VALID),
            "Move North",
        )

    def test_closed_xml_tag_with_trailing_punct(self):
        self.assertEqual(
            extract_action_multistrategy("<action>Move North.</action>", _VALID),
            "Move North",
        )

    def test_closed_xml_tag_with_give(self):
        result = extract_action_multistrategy("<action>give agent 1</action>", _VALID)
        self.assertEqual(result, "Give to Agent 1")

    def test_unclosed_xml_tag(self):
        result = extract_action_multistrategy("Thinking... <action>Move North", _VALID)
        self.assertEqual(result, "Move North")

    def test_action_colon_format(self):
        result = extract_action_multistrategy("I should move. ACTION: Move South\n", _VALID)
        self.assertEqual(result, "Move South")

    def test_action_colon_case_insensitive(self):
        result = extract_action_multistrategy("action: move east\n", _VALID)
        self.assertEqual(result, "Move East")

    def test_bare_exact_text(self):
        self.assertEqual(extract_action_multistrategy("Move East", _VALID), "Move East")

    def test_single_line_colon_split(self):
        result = extract_action_multistrategy("Move West: moving towards water", _VALID)
        self.assertEqual(result, "Move West")

    def test_text_before_html_tag(self):
        result = extract_action_multistrategy("Move North<br>ignored", _VALID)
        self.assertEqual(result, "Move North")

    def test_first_line_fallback(self):
        result = extract_action_multistrategy("Move East\nExplanation line\nMore text", _VALID)
        self.assertEqual(result, "Move East")

    def test_fuzzy_inside_xml_tag(self):
        result = extract_action_multistrategy("<action>Move Norht</action>", _VALID)
        self.assertEqual(result, "Move North")

    def test_no_match_returns_none(self):
        self.assertIsNone(extract_action_multistrategy("xyzabc123456789", _VALID))

    def test_empty_string_returns_none(self):
        self.assertIsNone(extract_action_multistrategy("", _VALID))

    def test_none_returns_none(self):
        self.assertIsNone(extract_action_multistrategy(None, _VALID))

    def test_xml_tag_with_surrounding_text(self):
        text = "I will move north. <action>Move North</action> That is my plan."
        self.assertEqual(extract_action_multistrategy(text, _VALID), "Move North")

    def test_give_via_action_colon(self):
        result = extract_action_multistrategy("ACTION: give agent 2\n", _VALID)
        self.assertEqual(result, "Give to Agent 2")


if __name__ == "__main__":
    unittest.main()
