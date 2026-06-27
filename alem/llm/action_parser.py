"""Action parsing helpers shared by ALEM text wrappers and LLM agents."""

import re
from difflib import get_close_matches

_GIVE_EXTRACT_PATTERN = re.compile(
    r"^\s*give\s+(?:to\s+)?(?:agent|teammate)[_\s-]*(\d+)\s*$",
    re.IGNORECASE,
)


def _valid_actions_lower(valid_actions):
    return {action.lower(): action for action in valid_actions}


def validate_action(candidate, valid_actions):
    """Return the canonical action name when a candidate is valid.

    Args:
        candidate: Candidate action text.
        valid_actions: Canonical action names accepted by the environment.

    Returns:
        Canonical action name, or ``None`` when no exact match exists.
    """
    if not candidate or not candidate.strip():
        return None
    candidate = candidate.strip()
    if candidate in valid_actions:
        return candidate
    return _valid_actions_lower(valid_actions).get(candidate.lower())


def fuzzy_match_action(candidate, valid_actions, cutoff=0.6):
    """Fuzzy-match a candidate against valid actions.

    Args:
        candidate: Candidate action text.
        valid_actions: Canonical action names accepted by the environment.
        cutoff: Minimum similarity accepted by ``get_close_matches``.

    Returns:
        Closest canonical action, or ``None`` when similarity is too low.
    """
    if not candidate or not candidate.strip():
        return None
    valid_actions_lower = _valid_actions_lower(valid_actions)
    matches = get_close_matches(
        candidate.strip().lower(),
        list(valid_actions_lower.keys()),
        n=1,
        cutoff=cutoff,
    )
    if matches:
        return valid_actions_lower[matches[0]]
    return None


def _clean_extracted(text):
    """Strip common model formatting noise from extracted tag content."""
    return text.strip().strip("\"'`").rstrip(".,;:!?").strip()


def _normalize_give_target(text):
    """Return canonical 'Give to Agent X' form, or None if not a Give-target action."""
    if not text:
        return None
    match = _GIVE_EXTRACT_PATTERN.match(text)
    if not match:
        return None
    return f"Give to Agent {int(match.group(1))}"


def _try_extract_strict(text, valid_actions):
    """Strict extraction: Give-target normalization plus exact/case-insensitive action match."""
    if not text:
        return None
    give = _normalize_give_target(text)
    if give:
        return give
    return validate_action(text, valid_actions)


def _try_extract_valid(text, valid_actions):
    """Try exact, colon-split, and fuzzy matching against valid actions."""
    if not text:
        return None

    give = _normalize_give_target(text)
    if give:
        return give

    valid = validate_action(text, valid_actions)
    if valid:
        return valid

    if ":" in text:
        before = text.split(":")[0].strip()
        give = _normalize_give_target(before)
        if give:
            return give
        valid = validate_action(before, valid_actions)
        if valid:
            return valid
        fuzzy = fuzzy_match_action(before, valid_actions)
        if fuzzy:
            return fuzzy

    return fuzzy_match_action(text, valid_actions)


def extract_action_multistrategy(completion_text, valid_actions):
    """Extract a canonical action from model output using ordered fallbacks.

    Args:
        completion_text: Raw model completion to parse.
        valid_actions: Canonical action names accepted by the environment.

    Returns:
        Parsed canonical action, or ``None`` when every strategy fails.
    """
    if not completion_text:
        return None

    match = re.search(r"<action>(.*?)</action>", completion_text, re.DOTALL)
    if match:
        extracted = _clean_extracted(match.group(1))
        result = _try_extract_valid(extracted, valid_actions)
        if result:
            return result

    if not match and "<action>" in completion_text:
        match_open = re.search(r"<action>(.*?)(?=<[a-z]|$)", completion_text, re.DOTALL)
        if match_open:
            extracted = _clean_extracted(match_open.group(1))
            result = _try_extract_valid(extracted, valid_actions)
            if result:
                return result

    action_match = re.search(r"ACTION:\s*(.+?)(?:\n|$)", completion_text, re.IGNORECASE)
    if action_match:
        extracted = _clean_extracted(action_match.group(1))
        result = _try_extract_valid(extracted, valid_actions)
        if result:
            return result

    exact = _try_extract_strict(completion_text.strip(), valid_actions)
    if exact:
        return exact

    if ":" in completion_text and "\n" not in completion_text.strip():
        before_colon = completion_text.split(":")[0].strip()
        result = _try_extract_strict(before_colon, valid_actions)
        if result:
            return result

    if "<" in completion_text:
        before_tag = completion_text[: completion_text.index("<")].strip()
        if before_tag:
            result = _try_extract_strict(before_tag, valid_actions)
            if result:
                return result

    first_line = completion_text.strip().split("\n")[0].strip()
    if first_line and first_line != completion_text.strip():
        result = _try_extract_strict(first_line, valid_actions)
        if result:
            return result

    return None
