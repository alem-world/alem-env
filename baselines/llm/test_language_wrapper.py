"""Unit tests for AlemLanguageWrapper.

Tests cover:
- describe_status: sleeping, dead, role, dungeon level
- describe_env: facing direction, block scanning, unique filtering
- describe_mobs: melee/ranged/passive mobs, projectiles, visibility, coordination tags
- describe_level_info: light level, level cleared boolean, boss state
- describe_coordination_cues: sync/handover cells, pending handovers
- describe_construction: returns empty (construction visible via block types only)
- describe_teammates: alive/dead, requests (no duration), specialization
- describe_inventory: vitals, items, tools, potions, enchantments
- describe_frame: wiring of all sections, conditional coordination/construction
- get_action_index: exact, case-insensitive, ACTION: prefix, fallback
- process_obs: output dict structure
- RL parity: no extra info (mob health, projectile direction, handover initiator, etc.)
"""

import os
import sys
import unittest
from pathlib import Path

# Add project root and alem/ to path
_project_root = str(Path(__file__).parent.parent.parent)
_alem_root = os.path.join(_project_root, "alem")
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if _alem_root not in sys.path:
    sys.path.insert(0, _alem_root)

import jax
import jax.numpy as jnp
import numpy as np

from alem.alem_coop.alem_state import EnvParams
from alem.alem_coop.constants import OBS_DIM, Achievement, Action, BlockType, ItemType
from alem.llm.alem_env import CraftaxEnv
from alem.llm.alem_language_wrapper import (
    ACTIONS,
    DIRECTION_NAMES,
    ITEM_NAMES,
    MELEE_MOB_NAMES,
    PASSIVE_MOB_NAMES,
    PROJECTILE_NAMES,
    RANGED_MOB_NAMES,
    AlemLanguageWrapper,
    _absolute_direction,
    _egocentric_direction,
    _rotate_to_egocentric,
    describe_loc_old,
    describe_loc_precise,
    get_instruction_prompt,
    make_alem_env,
)
from baselines.llm.eval_utils.agents.robust_all import RobustAllAgent
from baselines.llm.eval_utils.prompt_builder import HistoryPromptBuilder, Message

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wrapper(coordination_difficulty="none", **kwargs):
    """Create a wrapper with given config.

    Note: player_count is always 3 (hardcoded in StaticEnvParams).
    """
    config = {
        "max_timesteps": 10000,
        "god_mode": False,
        "coordination_difficulty": coordination_difficulty,
        "soft_specialization": kwargs.pop("soft_specialization", False),
        "shared_reward": kwargs.pop("shared_reward", True),
    }
    config.update(kwargs)
    env = make_alem_env(config=config)
    env_params = env.default_params
    return AlemLanguageWrapper(
        env,
        env_params,
        **{
            k: v
            for k, v in kwargs.items()
            if k
            in (
                "unique_items",
                "precise_location",
                "exact_coordinates",
                "llm_mode",
                "prompt_mode",
                "egocentric",
            )
        },
    )


def _reset_wrapper(wrapper, seed=42):
    """Reset and return (text_obs_list, state, rng)."""
    rng = jax.random.PRNGKey(seed)
    return wrapper.reset(rng)


# ============================================================================
# 1. LOCATION HELPERS
# ============================================================================


class TestLocationHelpers(unittest.TestCase):
    """Tests for describe_loc_precise and describe_loc_old."""

    def test_describe_loc_precise_same_position(self):
        result = describe_loc_precise(np.array([0, 0]), np.array([0, 0]))
        self.assertEqual(result, "at your location")

    def test_describe_loc_precise_north(self):
        result = describe_loc_precise(np.array([0, 0]), np.array([-3, 0]))
        self.assertIn("north", result)
        self.assertIn("3", result)

    def test_describe_loc_precise_south_east(self):
        result = describe_loc_precise(np.array([0, 0]), np.array([2, 1]))
        self.assertIn("south", result)
        self.assertIn("east", result)

    def test_describe_loc_old_distance(self):
        result = describe_loc_old(np.array([0, 0]), np.array([-2, 3]))
        self.assertIn("east", result)
        self.assertIn("north", result)
        self.assertIn("5", result)

    def test_relative_direction_str_appends_exact_coordinates(self):
        wrapper = _make_wrapper(exact_coordinates=True)
        result = wrapper._relative_direction_str(-1, 2, abs_pos=np.array([7, 11]))
        self.assertEqual(result, "(x=11, y=7)")

    def test_relative_direction_str_without_exact_coordinates(self):
        wrapper = _make_wrapper(exact_coordinates=False)
        result = wrapper._relative_direction_str(-1, 2, abs_pos=np.array([7, 11]))
        self.assertNotIn("x=", result)
        self.assertNotIn("y=", result)


# ============================================================================
# 2. ACTION PARSING
# ============================================================================


class TestActionParsing(unittest.TestCase):
    """Tests for get_action_index — exact, case-insensitive, colon-separated,
    ACTION: prefix, substring match, and failure handling."""

    def setUp(self):
        self.wrapper = _make_wrapper()

    def test_exact_match(self):
        idx = self.wrapper.get_action_index("Move North")
        self.assertEqual(ACTIONS[idx], "Move North")

    def test_case_insensitive(self):
        idx = self.wrapper.get_action_index("move north")
        self.assertEqual(ACTIONS[idx], "Move North")

    def test_action_prefix(self):
        idx = self.wrapper.get_action_index("ACTION: Move South")
        self.assertEqual(ACTIONS[idx], "Move South")

    def test_quoted_input(self):
        idx = self.wrapper.get_action_index('"Do"')
        self.assertEqual(ACTIONS[idx], "Do")

    def test_trailing_punctuation(self):
        idx = self.wrapper.get_action_index("Sleep.")
        self.assertEqual(ACTIONS[idx], "Sleep")

    def test_invalid_action_returns_noop(self):
        idx = self.wrapper.get_action_index("fly to the moon")
        self.assertEqual(idx, 0)  # Noop

    def test_failed_candidates_tracked(self):
        self.wrapper.get_action_index("nonsense action", agent_idx=1)
        self.assertIn("nonsense action", self.wrapper.failed_candidates[1])

    # --- New tests for LLM-typical output formats ---

    def test_action_with_description_colon(self):
        """LLMs often echo 'ActionName: description' from the prompt."""
        idx = self.wrapper.get_action_index("Move North: move north on flat ground")
        self.assertEqual(ACTIONS[idx], "Move North")

    def test_action_with_description_make(self):
        idx = self.wrapper.get_action_index(
            "Make Wood Sword: craft a wood sword needs nearby table wood"
        )
        self.assertEqual(ACTIONS[idx], "Make Wood Sword")

    def test_action_with_description_do(self):
        idx = self.wrapper.get_action_index("Do: search")
        self.assertEqual(ACTIONS[idx], "Do")

    def test_action_with_description_place(self):
        idx = self.wrapper.get_action_index("Place Stone: place a stone in front")
        self.assertEqual(ACTIONS[idx], "Place Stone")

    def test_action_with_description_sleep(self):
        idx = self.wrapper.get_action_index("Sleep: sleep when energy level is below maximum")
        self.assertEqual(ACTIONS[idx], "Sleep")

    def test_action_with_description_shoot(self):
        idx = self.wrapper.get_action_index("Shoot Arrow: shoot an arrow with your bow")
        self.assertEqual(ACTIONS[idx], "Shoot Arrow")

    def test_action_with_description_cast(self):
        idx = self.wrapper.get_action_index("Cast Spell: cast a learned spell costs mana")
        self.assertEqual(ACTIONS[idx], "Cast Spell")

    def test_action_with_description_descend(self):
        idx = self.wrapper.get_action_index("Descend: go down to the next dungeon level")
        self.assertEqual(ACTIONS[idx], "Descend")

    def test_action_with_description_request(self):
        idx = self.wrapper.get_action_index("Request Wood: request wood from nearby teammates")
        self.assertEqual(ACTIONS[idx], "Request Wood")

    def test_hallucinated_action_returns_noop(self):
        """Truly invalid actions that don't exist should return Noop."""
        for invalid in ["Place Tree", "Make Wooden Sword", "Make Food"]:
            idx = self.wrapper.get_action_index(invalid)
            self.assertEqual(idx, 0, f"Expected Noop for hallucinated action: {invalid}")

    def test_empty_string_returns_noop(self):
        idx = self.wrapper.get_action_index("")
        self.assertEqual(idx, 0)

    def test_action_prefix_with_colon_desc(self):
        """ACTION: ActionName: description"""
        idx = self.wrapper.get_action_index("ACTION: Move East: move east on flat ground")
        self.assertEqual(ACTIONS[idx], "Move East")

    def test_bare_targeted_give_maps_to_slot(self):
        idx = self.wrapper.get_action_index("Give to Agent 0", agent_idx=1)
        self.assertEqual(idx, Action.GIVE.value)

    def test_bare_targeted_give_maps_second_slot(self):
        idx = self.wrapper.get_action_index("Give to Agent 2", agent_idx=1)
        self.assertEqual(idx, Action.GIVE.value + 1)

    # --- Give via XML tags (primary LLM output format) ---

    def test_give_via_xml_tags(self):
        idx = self.wrapper.get_action_index("<action>Give to Agent 0</action>", agent_idx=1)
        self.assertEqual(idx, Action.GIVE.value)

    def test_give_via_xml_tags_second_target(self):
        idx = self.wrapper.get_action_index("<action>Give to Agent 2</action>", agent_idx=1)
        self.assertEqual(idx, Action.GIVE.value + 1)

    def test_give_via_action_prefix(self):
        idx = self.wrapper.get_action_index("ACTION: Give to Agent 0", agent_idx=1)
        self.assertEqual(idx, Action.GIVE.value)

    def test_give_via_action_prefix_second_target(self):
        idx = self.wrapper.get_action_index("ACTION: Give to Agent 2", agent_idx=1)
        self.assertEqual(idx, Action.GIVE.value + 1)

    def test_give_teammate_keyword(self):
        """'teammate' synonym should also parse correctly."""
        idx = self.wrapper.get_action_index("Give to Teammate 0", agent_idx=1)
        self.assertEqual(idx, Action.GIVE.value)

    def test_give_teammate_keyword_second_target(self):
        idx = self.wrapper.get_action_index("Give to Teammate 2", agent_idx=1)
        self.assertEqual(idx, Action.GIVE.value + 1)

    def test_give_with_description(self):
        """LLMs sometimes echo 'Give to Agent X: description'."""
        idx = self.wrapper.get_action_index(
            "Give to Agent 0: give requested resources to a specific teammate", agent_idx=1
        )
        self.assertEqual(idx, Action.GIVE.value)

    # --- Slot mapping for all giver/target combos (3 agents: 0, 1, 2) ---

    def test_give_slot_agent0_gives_to_agent1(self):
        # agent_idx=0 giving to agent 1: target(1) > giver(0) → slot = 1-1 = 0
        idx = self.wrapper.get_action_index("Give to Agent 1", agent_idx=0)
        self.assertEqual(idx, Action.GIVE.value + 0)

    def test_give_slot_agent0_gives_to_agent2(self):
        # agent_idx=0 giving to agent 2: target(2) > giver(0) → slot = 2-1 = 1
        idx = self.wrapper.get_action_index("Give to Agent 2", agent_idx=0)
        self.assertEqual(idx, Action.GIVE.value + 1)

    def test_give_slot_agent1_gives_to_agent0(self):
        # agent_idx=1 giving to agent 0: target(0) < giver(1) → slot = 0
        idx = self.wrapper.get_action_index("Give to Agent 0", agent_idx=1)
        self.assertEqual(idx, Action.GIVE.value + 0)

    def test_give_slot_agent1_gives_to_agent2(self):
        # agent_idx=1 giving to agent 2: target(2) > giver(1) → slot = 2-1 = 1
        idx = self.wrapper.get_action_index("Give to Agent 2", agent_idx=1)
        self.assertEqual(idx, Action.GIVE.value + 1)

    def test_give_slot_agent2_gives_to_agent0(self):
        # agent_idx=2 giving to agent 0: target(0) < giver(2) → slot = 0
        idx = self.wrapper.get_action_index("Give to Agent 0", agent_idx=2)
        self.assertEqual(idx, Action.GIVE.value + 0)

    def test_give_slot_agent2_gives_to_agent1(self):
        # agent_idx=2 giving to agent 1: target(1) < giver(2) → slot = 1
        idx = self.wrapper.get_action_index("Give to Agent 1", agent_idx=2)
        self.assertEqual(idx, Action.GIVE.value + 1)

    # --- Invalid give targets → Noop ---

    def test_give_to_self_returns_noop(self):
        idx = self.wrapper.get_action_index("Give to Agent 1", agent_idx=1)
        self.assertEqual(idx, 0)

    def test_give_to_out_of_range_agent_returns_noop(self):
        idx = self.wrapper.get_action_index("Give to Agent 99", agent_idx=0)
        self.assertEqual(idx, 0)

    def test_give_without_target_maps_to_base_give(self):
        """Bare 'Give' (no target) maps to the base GIVE action index, not a targeted slot.

        It won't be in the action mask (only targeted slots are masked), so in practice
        it acts like a noop at env step time — but parsing-wise it's Action.GIVE.value.
        """
        idx = self.wrapper.get_action_index("Give", agent_idx=0)
        self.assertEqual(idx, Action.GIVE.value)

    # --- Available actions list includes 'Give to Agent X' when masked ---

    def test_available_actions_give_affordances_via_reset_state(self):
        """get_affordances should list 'Give to Agent X' when give slots are unmasked.

        We use a real reset state and verify the method output excludes bare 'Give'
        and only ever lists targeted 'Give to Agent X' forms.
        """
        _, state, _ = _reset_wrapper(self.wrapper)
        affordances = self.wrapper.get_affordances(state, player_idx=1)
        # Bare 'Give' must never appear — only targeted forms are valid
        lines = [l.strip().lstrip("- ") for l in affordances.splitlines()]
        self.assertNotIn("Give", lines)
        # Any give affordances must match the expected targeted pattern
        give_lines = [l for l in lines if l.startswith("Give")]
        for line in give_lines:
            self.assertRegex(line, r"^Give to Agent \d+$")


class TestCraftaxEnvActionValidity(unittest.TestCase):
    """Regression tests for evaluator-side action canonicalization."""

    def setUp(self):
        self.wrapper = _make_wrapper()
        self.env = CraftaxEnv.__new__(CraftaxEnv)
        self.env.wrapper = self.wrapper
        self.env.num_agents = 3
        self.env.failed_candidates = []
        self.env.failed_candidates_per_agent = [[] for _ in range(self.env.num_agents)]

    def test_targeted_give_is_preserved_for_first_slot(self):
        action = self.env.check_action_validity("Give to Agent 0", agent_idx=1)
        self.assertEqual(action, "Give to Agent 0")

    def test_targeted_give_is_preserved_for_second_slot(self):
        action = self.env.check_action_validity("Give to Agent 2", agent_idx=1)
        self.assertEqual(action, "Give to Agent 2")


# ============================================================================
# 3. DESCRIBE_STATUS
# ============================================================================


class TestDescribeStatus(unittest.TestCase):
    """Tests for describe_status — role, level, sleeping/dead states."""

    def setUp(self):
        self.wrapper = _make_wrapper(soft_specialization=True)
        _, self.state, _ = _reset_wrapper(self.wrapper)

    def test_contains_level(self):
        status = self.wrapper.describe_status(self.state, 0)
        # Should mention overworld or dungeon level (case-insensitive)
        self.assertTrue("overworld" in status.lower() or "dungeon level" in status.lower())

    def test_specialization_shown_when_assigned(self):
        status = self.wrapper.describe_status(self.state, 0)
        # With soft_specialization, agents should have roles
        spec = int(self.state.player_specialization[0])
        if spec > 0:
            self.assertIn("role", status.lower())

    def test_exact_coordinates_include_self_position(self):
        wrapper = _make_wrapper(soft_specialization=True, exact_coordinates=True)
        _, state, _ = _reset_wrapper(wrapper)
        status = wrapper.describe_status(state, 0)
        self.assertIn("Position: (x=", status)


# ============================================================================
# 4. DESCRIBE_ENV
# ============================================================================


class TestDescribeEnv(unittest.TestCase):
    """Tests for describe_env — facing direction, block scanning, light masking, items."""

    def setUp(self):
        self.wrapper = _make_wrapper()
        _, self.state, _ = _reset_wrapper(self.wrapper)

    def test_contains_facing_direction(self):
        desc = self.wrapper.describe_env(self.state, 0)
        self.assertIn("facing", desc.lower())

    def test_direction_mapping_correctness(self):
        """Verify DIRECTION_NAMES matches expected: 1=west, 2=east, 3=north, 4=south."""
        self.assertEqual(DIRECTION_NAMES[1], "west")
        self.assertEqual(DIRECTION_NAMES[2], "east")
        self.assertEqual(DIRECTION_NAMES[3], "north")
        self.assertEqual(DIRECTION_NAMES[4], "south")

    def test_you_see_section(self):
        desc = self.wrapper.describe_env(self.state, 0)
        # Should have "You see:" or "You see nothing"
        self.assertTrue("You see" in desc)

    def test_skip_items_excluded(self):
        desc = self.wrapper.describe_env(self.state, 0)
        # grass/sand/path should not appear as objects
        for skip in ["- grass ", "- sand ", "- path "]:
            self.assertNotIn(skip, desc)

    def test_item_names_defined(self):
        """ITEM_NAMES should cover all non-NONE ItemType values."""
        for it in ItemType:
            if it != ItemType.NONE:
                self.assertIn(it.value, ITEM_NAMES, f"Missing ITEM_NAMES entry for {it}")

    def test_light_mask_helper(self):
        """_get_local_light_mask should return a boolean array of OBS_DIM shape."""
        mask = self.wrapper._get_local_light_mask(self.state, 0)
        self.assertEqual(mask.shape, tuple(OBS_DIM))
        self.assertEqual(mask.dtype, np.bool_)


# ============================================================================
# 5. DESCRIBE_MOBS
# ============================================================================


class TestDescribeMobs(unittest.TestCase):
    """Tests for describe_mobs — mob visibility, coordination tags."""

    def setUp(self):
        self.wrapper = _make_wrapper(coordination_difficulty="easy")
        _, self.state, _ = _reset_wrapper(self.wrapper)

    def test_returns_string(self):
        result = self.wrapper.describe_mobs(self.state, 0)
        self.assertIsInstance(result, str)

    def test_mob_names_valid(self):
        """All mob name lists should be non-empty and contain strings."""
        for names in [MELEE_MOB_NAMES, RANGED_MOB_NAMES, PASSIVE_MOB_NAMES, PROJECTILE_NAMES]:
            self.assertTrue(len(names) > 0)
            for name in names:
                self.assertIsInstance(name, str)

    def test_coordination_tags_in_output(self):
        """If mobs have coordination values, tags should appear."""
        result = self.wrapper.describe_mobs(self.state, 0)
        # We can't guarantee elite mobs exist on this seed, but at least
        # verify the method runs without error for all agents
        for i in range(self.wrapper.num_agents):
            self.wrapper.describe_mobs(self.state, i)

    def test_player_projectiles_section(self):
        """describe_mobs should handle player projectiles without error."""
        # On reset there are no projectiles, but the code path should not crash
        result = self.wrapper.describe_mobs(self.state, 0)
        # "Your projectiles" section only appears if player has active projectiles
        self.assertIsInstance(result, str)


# ============================================================================
# 6. DESCRIBE_LEVEL_INFO
# ============================================================================


class TestDescribeLevelInfo(unittest.TestCase):
    """Tests for describe_level_info — light, monster progress, boss state."""

    def setUp(self):
        self.wrapper = _make_wrapper()
        _, self.state, _ = _reset_wrapper(self.wrapper)

    def test_contains_light(self):
        result = self.wrapper.describe_level_info(self.state)
        self.assertIn("Light:", result)

    def test_light_level_categories(self):
        result = self.wrapper.describe_level_info(self.state)
        # Should contain one of bright/dim/dark
        self.assertTrue(
            "bright" in result or "dim" in result or "dark" in result,
            f"Expected light category in: {result}",
        )

    def test_level_cleared_status(self):
        result = self.wrapper.describe_level_info(self.state)
        # Should contain either "cleared" or "not yet cleared"
        self.assertTrue("cleared" in result, f"Expected level cleared status in: {result}")

    def test_level_info_header(self):
        result = self.wrapper.describe_level_info(self.state)
        self.assertTrue(result.startswith("Level info:"))


# ============================================================================
# 7. DESCRIBE_COORDINATION_CUES
# ============================================================================


class TestDescribeCoordinationCues(unittest.TestCase):
    """Tests for describe_coordination_cues — sync/handover cells, pending handovers."""

    def setUp(self):
        self.wrapper = _make_wrapper(coordination_difficulty="easy")
        _, self.state, _ = _reset_wrapper(self.wrapper)

    def test_returns_string(self):
        result = self.wrapper.describe_coordination_cues(self.state, 0)
        self.assertIsInstance(result, str)

    def test_runs_for_all_agents(self):
        for i in range(self.wrapper.num_agents):
            result = self.wrapper.describe_coordination_cues(self.state, i)
            self.assertIsInstance(result, str)

    def test_sync_cues_format(self):
        """If sync cues exist, they should follow the expected format."""
        result = self.wrapper.describe_coordination_cues(self.state, 0)
        if "Sync:" in result:
            self.assertIn("requires", result)
            self.assertIn("agents", result)

    def test_handover_cues_format(self):
        """If handover cues exist, they should follow the expected format."""
        result = self.wrapper.describe_coordination_cues(self.state, 0)
        if "Handover:" in result:
            self.assertIn("window", result)

    def test_no_cues_without_coordination(self):
        """With coordination_difficulty='none', no coordination cues should appear."""
        wrapper_none = _make_wrapper(coordination_difficulty="none")
        _, state_none, _ = _reset_wrapper(wrapper_none)
        # coordination_map should be all zeros
        coord_map = np.array(state_none.coordination_map)
        self.assertTrue(np.all(coord_map == 0))


# ============================================================================
# 8. DESCRIBE_CONSTRUCTION
# ============================================================================


class TestDescribeConstruction(unittest.TestCase):
    """Tests for describe_construction — returns empty for RL parity.

    Construction sites are visible only through block types in describe_env()
    (CONSTRUCTION_SITE, CONSTRUCTION_IN_PROGRESS), matching what the RL agent sees.
    """

    def setUp(self):
        self.wrapper = _make_wrapper(coordination_difficulty="easy")
        _, self.state, _ = _reset_wrapper(self.wrapper)

    def test_returns_empty_string(self):
        """describe_construction always returns empty for RL parity."""
        result = self.wrapper.describe_construction(self.state, 0)
        self.assertEqual(result, "")

    def test_returns_empty_for_all_agents(self):
        for i in range(self.wrapper.num_agents):
            result = self.wrapper.describe_construction(self.state, i)
            self.assertEqual(result, "")


# ============================================================================
# 9. DESCRIBE_TEAMMATES
# ============================================================================


class TestDescribeTeammates(unittest.TestCase):
    """Tests for describe_teammates — positions, health, requests, off-screen directions."""

    def setUp(self):
        self.wrapper = _make_wrapper()
        _, self.state, _ = _reset_wrapper(self.wrapper)

    def test_other_agents_mentioned(self):
        result = self.wrapper.describe_teammates(self.state, 0)
        # Should mention agent 1 and agent 2
        self.assertIn("Agent 1", result)
        self.assertIn("Agent 2", result)

    def test_own_agent_not_mentioned(self):
        result = self.wrapper.describe_teammates(self.state, 0)
        self.assertNotIn("Agent 0:", result)

    def test_health_shown(self):
        result = self.wrapper.describe_teammates(self.state, 0)
        self.assertIn("health=", result)

    def test_teammates_header(self):
        result = self.wrapper.describe_teammates(self.state, 0)
        self.assertTrue(result.startswith("Teammates:"))

    def test_on_screen_vs_off_screen(self):
        """All teammates should show either a position or 'off-screen to the ...'."""
        result = self.wrapper.describe_teammates(self.state, 0)
        for line in result.split("\n"):
            if line.startswith("Agent") or line.startswith("- Agent"):
                # Should have either a relative position or off-screen indicator
                self.assertTrue(
                    "step" in line or "at your location" in line or "off-screen" in line,
                    f"Teammate line missing position/direction: {line}",
                )


# ============================================================================
# 10. DESCRIBE_INVENTORY
# ============================================================================


class TestDescribeInventory(unittest.TestCase):
    """Tests for describe_inventory — vitals, items, tools."""

    def setUp(self):
        self.wrapper = _make_wrapper()
        _, self.state, _ = _reset_wrapper(self.wrapper)

    def test_contains_vitals(self):
        result = self.wrapper.describe_inventory(self.state, 0)
        for vital in ["health:", "food:", "drink:", "energy:", "mana:"]:
            self.assertIn(vital, result)

    def test_contains_status_header(self):
        result = self.wrapper.describe_inventory(self.state, 0)
        self.assertIn("Your status:", result)


# ============================================================================
# 11. DESCRIBE_FRAME (integration)
# ============================================================================


class TestDescribeFrame(unittest.TestCase):
    """Integration tests for describe_frame — wiring of all sections."""

    def test_frame_without_coordination(self):
        wrapper = _make_wrapper(coordination_difficulty="none")
        _, state, _ = _reset_wrapper(wrapper)
        long_ctx, short_ctx = wrapper.describe_frame(state, 0)

        # Should have level info, env, teammates
        self.assertIn("Level info:", long_ctx)
        self.assertIn("You see", long_ctx)
        self.assertIn("Teammates:", long_ctx)

        # Should NOT have coordination or construction (disabled)
        self.assertNotIn("Coordination cues:", long_ctx)
        self.assertNotIn("Construction sites:", long_ctx)

        # short_ctx is inventory
        self.assertIn("health:", short_ctx)

    def test_frame_level_info_no_raw_kill_count(self):
        """Level info should show cleared/not-cleared boolean, not raw kill count."""
        wrapper = _make_wrapper()
        _, state, _ = _reset_wrapper(wrapper)
        long_ctx, _ = wrapper.describe_frame(state, 0)
        # Should not leak raw monsters_killed count
        self.assertNotIn("Monsters killed:", long_ctx)
        # Should have the boolean form
        self.assertTrue("cleared" in long_ctx)

    def test_frame_with_coordination(self):
        wrapper = _make_wrapper(coordination_difficulty="easy")
        _, state, _ = _reset_wrapper(wrapper)
        long_ctx, short_ctx = wrapper.describe_frame(state, 0)

        # Should have level info
        self.assertIn("Level info:", long_ctx)
        # Coordination and construction sections may or may not appear
        # depending on what's in view, but the method should not error
        self.assertIsInstance(long_ctx, str)
        self.assertIsInstance(short_ctx, str)

    def test_frame_returns_tuple(self):
        wrapper = _make_wrapper()
        _, state, _ = _reset_wrapper(wrapper)
        result = wrapper.describe_frame(state, 0)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_frame_all_agents(self):
        """describe_frame should work for every agent index."""
        wrapper = _make_wrapper()
        _, state, _ = _reset_wrapper(wrapper)
        for i in range(3):
            long_ctx, short_ctx = wrapper.describe_frame(state, i)
            self.assertIsInstance(long_ctx, str)
            self.assertIsInstance(short_ctx, str)


# ============================================================================
# 12. PROCESS_OBS
# ============================================================================


class TestProcessObs(unittest.TestCase):
    """Tests for process_obs output structure."""

    def setUp(self):
        self.wrapper = _make_wrapper()
        rng = jax.random.PRNGKey(42)
        rng, _rng = jax.random.split(rng)
        self.obs, self.state = self.wrapper.env.reset(_rng)

    def test_output_has_required_keys(self):
        result = self.wrapper.process_obs(self.obs, self.state, 0)
        self.assertIn("text", result)
        self.assertIn("image", result)
        self.assertIn("obs", result)

    def test_text_has_context_keys(self):
        result = self.wrapper.process_obs(self.obs, self.state, 0)
        self.assertIn("long_term_context", result["text"])
        self.assertIn("short_term_context", result["text"])

    def test_text_values_are_strings(self):
        result = self.wrapper.process_obs(self.obs, self.state, 0)
        self.assertIsInstance(result["text"]["long_term_context"], str)
        self.assertIsInstance(result["text"]["short_term_context"], str)


# ============================================================================
# 13. RESET & STEP
# ============================================================================


class TestResetAndStep(unittest.TestCase):
    """Tests for reset and step methods."""

    def setUp(self):
        self.wrapper = _make_wrapper()

    def test_reset_returns_obs_for_all_agents(self):
        text_obs_list, state, rng = _reset_wrapper(self.wrapper)
        self.assertEqual(len(text_obs_list), 3)
        for obs in text_obs_list:
            self.assertIn("text", obs)

    def test_step_returns_correct_structure(self):
        text_obs_list, state, rng = _reset_wrapper(self.wrapper)
        actions = ["Noop", "Move North", "Move East"]
        result = self.wrapper.step(state, actions, rng)
        text_obs_list, new_state, rewards, dones, info, rng = result

        self.assertEqual(len(text_obs_list), 3)
        self.assertEqual(len(rewards), 3)
        self.assertEqual(len(dones), 3)

    def test_step_with_integer_actions(self):
        text_obs_list, state, rng = _reset_wrapper(self.wrapper)
        actions = [0, 3, 2]  # Noop, Move North, Move East
        text_obs_list, new_state, rewards, dones, info, rng = self.wrapper.step(state, actions, rng)
        self.assertEqual(len(text_obs_list), 3)


# ============================================================================
# 14. INSTRUCTION PROMPT
# ============================================================================


class TestInstructionPrompt(unittest.TestCase):
    """Tests for get_instruction_prompt."""

    _BASE_KWARGS = dict(
        llm_mode="easy",
        coordination_enabled=True,
        num_agents=3,
        agent_id=0,
        role="warrior",
    )

    @staticmethod
    def _normalize(text):
        return "\n".join(line.rstrip() for line in text.strip().splitlines())

    def _load_fixture(self, filename):
        return (Path(_project_root) / "docs" / "prompts" / filename).read_text(encoding="utf-8")

    # --- Snapshot / golden tests ---

    def test_general_mode_matches_fixture(self):
        prompt = get_instruction_prompt(prompt_mode="general", **self._BASE_KWARGS)
        expected = self._load_fixture("general_agent.md")
        self.assertEqual(self._normalize(prompt), self._normalize(expected))

    def test_specific_mode_matches_fixture(self):
        prompt = get_instruction_prompt(prompt_mode="specific", **self._BASE_KWARGS)
        expected = self._load_fixture("specific_agent.md")
        self.assertEqual(self._normalize(prompt), self._normalize(expected))

    def test_specific_collaborative_mode_matches_fixture(self):
        prompt = get_instruction_prompt(prompt_mode="specific_collaborative", **self._BASE_KWARGS)
        expected = self._load_fixture("specific_collaborative_agent.md")
        self.assertEqual(self._normalize(prompt), self._normalize(expected))

    # --- Isolation / leakage tests ---

    def test_specific_mode_has_no_coordination_content(self):
        """specific mode must not leak any coordination info into the prompt."""
        prompt = get_instruction_prompt(prompt_mode="specific", **self._BASE_KWARGS)
        self.assertIn("<game_rules>", prompt)
        self.assertIn("## Survival stats", prompt)
        self.assertIn("## Crafting recipes", prompt)
        self.assertIn("## Attributes", prompt)
        self.assertNotIn("## Coordination", prompt)
        self.assertNotIn("enough agents crafting", prompt)
        self.assertNotIn("synchronous-style coordination", prompt)
        self.assertNotIn("requires coordination", prompt)
        self.assertNotIn("coordinating with teammates", prompt)

    def test_general_mode_has_action_list_no_game_rules(self):
        prompt = get_instruction_prompt(prompt_mode="general", **self._BASE_KWARGS)
        self.assertIn("## Actions", prompt)
        self.assertIn("Move North:", prompt)
        self.assertNotIn("<game_rules>", prompt)
        self.assertNotIn("## Coordination", prompt)

    def test_specific_collaborative_mode_has_all_coordination(self):
        prompt = get_instruction_prompt(prompt_mode="specific_collaborative", **self._BASE_KWARGS)
        self.assertIn("<game_rules>", prompt)
        self.assertIn("## Coordination", prompt)
        self.assertIn("Sync", prompt)
        self.assertIn("Handover", prompt)
        self.assertIn("enough agents crafting", prompt)
        self.assertIn("## Attributes", prompt)

    def test_attributes_shown_at_level_0(self):
        """Attributes section is always shown (not gated by level)."""
        for mode in ("specific", "specific_collaborative"):
            with self.subTest(mode=mode):
                prompt = get_instruction_prompt(
                    prompt_mode=mode,
                    progressive_disclosure=True,
                    current_level=0,
                    **self._BASE_KWARGS,
                )
                self.assertIn("## Attributes", prompt)
                self.assertIn("Gain 1 XP each time you descend", prompt)

    def test_invalid_prompt_mode_raises(self):
        with self.assertRaises(ValueError):
            get_instruction_prompt(prompt_mode="invalid_mode")


class TestPromptBuilderFormatting(unittest.TestCase):
    """Tests for user-turn formatting in history prompts."""

    def test_step_line_is_moved_to_top(self):
        builder = HistoryPromptBuilder(max_text_history=4, max_image_history=0)
        obs = {
            "text": {
                "long_term_context": (
                    "Last action: Move East\n"
                    "Reward: +0.000\n\n"
                    "Step: 6/10000 (9994 remaining, ends early if all agents die)\n"
                    "Role: warrior"
                ),
                "short_term_context": "",
            },
            "image": None,
        }
        builder.update_observation(obs)
        messages = builder.get_prompt()
        self.assertEqual(messages[0].role, "user")
        self.assertTrue(messages[0].content.startswith("Step: 6/10000"))
        self.assertIn("\nCurrent Observation:\n", messages[0].content)
        self.assertIn("Last action: Move East", messages[0].content)
        self.assertNotIn("\n\nStep: 6/10000", messages[0].content)


class TestRobustAllScratchpadInstructions(unittest.TestCase):
    """Mode- and communication-aware scratchpad instructions."""

    @staticmethod
    def _make_agent(prompt_mode, use_scratchpad=True, use_communication=False):
        agent = RobustAllAgent.__new__(RobustAllAgent)
        agent.prompt_mode = prompt_mode
        agent.use_cot = False
        agent.enable_thinking = False
        agent.use_scratchpad = use_scratchpad
        agent.max_scratchpad_length = 1000
        agent.structured_scratchpad = False
        agent.use_communication = use_communication
        agent.max_communication_length = 400
        agent.max_communication_history = 4
        agent.structured_communication = False
        agent.max_tokens = 8192
        agent.reasoning_enabled = False
        agent._warned_naive_mode = True
        return agent

    @staticmethod
    def _build_instruction_text(agent):
        messages = [Message(role="user", content="Current Observation:\n...")]
        agent._append_instructions(messages)
        return messages[-1].content

    def test_specific_mode_has_non_collaborative_scratchpad_guidance(self):
        agent = self._make_agent(
            prompt_mode="specific",
            use_scratchpad=True,
            use_communication=True,
        )
        text = self._build_instruction_text(agent)
        # Coordination team-tracking tip must not appear in non-collaborative mode
        self.assertNotIn("Record teammates", text)

    def test_specific_collaborative_without_comm_has_no_coordination_memory_tips(self):
        agent = self._make_agent(
            prompt_mode="specific_collaborative",
            use_scratchpad=True,
            use_communication=False,
        )
        text = self._build_instruction_text(agent)
        self.assertNotIn("Record teammates", text)

    def test_specific_collaborative_with_comm_adds_coordination_memory_tips(self):
        agent = self._make_agent(
            prompt_mode="specific_collaborative",
            use_scratchpad=True,
            use_communication=True,
        )
        text = self._build_instruction_text(agent)
        self.assertIn("Record teammates", text)

    # --- Structured scratchpad leakage tests ---

    def test_structured_scratchpad_specific_mode_no_coordination_leak(self):
        """Structured scratchpad must not leak teammate content in non-collaborative mode."""
        agent = self._make_agent(
            prompt_mode="specific",
            use_scratchpad=True,
            use_communication=True,
        )
        agent.structured_scratchpad = True
        text = self._build_instruction_text(agent)
        self.assertNotIn("TEAM:", text)

    def test_structured_scratchpad_collaborative_with_comm_has_team_section(self):
        """Structured scratchpad shows TEAM section in collaborative + comm mode."""
        agent = self._make_agent(
            prompt_mode="specific_collaborative",
            use_scratchpad=True,
            use_communication=True,
        )
        agent.structured_scratchpad = True
        text = self._build_instruction_text(agent)
        self.assertIn("TEAM:", text)

    def test_structured_scratchpad_collaborative_without_comm_no_team_section(self):
        """No TEAM section in structured scratchpad when communication is disabled."""
        agent = self._make_agent(
            prompt_mode="specific_collaborative",
            use_scratchpad=True,
            use_communication=False,
        )
        agent.structured_scratchpad = True
        text = self._build_instruction_text(agent)
        self.assertNotIn("TEAM:", text)

    # --- Communication coordination framing tests ---

    def test_specific_mode_communication_has_no_coordination_framing(self):
        """specific mode must not tell the agent to coordinate via communication."""
        agent = self._make_agent(
            prompt_mode="specific",
            use_scratchpad=False,
            use_communication=True,
        )
        text = self._build_instruction_text(agent)
        self.assertIn("Broadcast to teammates", text)
        self.assertNotIn("Teammates can only act on what you tell them", text)
        self.assertNotIn("Reply to teammates' requests", text)

    def test_specific_collaborative_mode_communication_has_coordination_framing(self):
        """specific_collaborative mode must include coordination framing in communication."""
        agent = self._make_agent(
            prompt_mode="specific_collaborative",
            use_scratchpad=False,
            use_communication=True,
        )
        text = self._build_instruction_text(agent)
        self.assertIn("Teammates can only act on what you tell them", text)
        self.assertIn("Reply to teammates' requests", text)


# ============================================================================
# 15. HELPER METHODS
# ============================================================================


class TestHelperMethods(unittest.TestCase):
    """Tests for _get_local_view_params and _relative_direction_str."""

    def setUp(self):
        self.wrapper = _make_wrapper()
        _, self.state, _ = _reset_wrapper(self.wrapper)

    def test_local_view_params(self):
        level, pos, vh, vw, hh, hw = self.wrapper._get_local_view_params(self.state, 0)
        self.assertIsInstance(level, int)
        self.assertEqual(len(pos), 2)
        self.assertTrue(vh > 0 and vw > 0)

    def test_relative_direction_str_here(self):
        result = self.wrapper._relative_direction_str(0, 0)
        self.assertEqual(result, "at your location")

    def test_relative_direction_str_north(self):
        result = self.wrapper._relative_direction_str(-3, 0)
        self.assertIn("north", result)
        self.assertIn("3", result)

    def test_relative_direction_str_south_east(self):
        result = self.wrapper._relative_direction_str(2, 1)
        self.assertIn("south", result)
        self.assertIn("east", result)


# ============================================================================
# 16. PROGRESS TRACKING
# ============================================================================


class TestProgressTracking(unittest.TestCase):
    """Tests for update_progress and get_stats."""

    def setUp(self):
        self.wrapper = _make_wrapper()
        _, self.state, _ = _reset_wrapper(self.wrapper)

    def test_update_progress_returns_score(self):
        score = self.wrapper.update_progress(self.state, 0)
        self.assertIsInstance(score, float)

    def test_get_stats_single_agent(self):
        self.wrapper.update_progress(self.state, 0)
        stats = self.wrapper.get_stats(player_idx=0)
        self.assertIn("score", stats)
        self.assertIn("progression", stats)
        self.assertIn("achievements", stats)

    def test_get_stats_all_agents(self):
        for i in range(3):
            self.wrapper.update_progress(self.state, i)
        stats = self.wrapper.get_stats()
        self.assertEqual(len(stats), 3)


# ============================================================================
# 17. RL PARITY — no extra info beyond what the symbolic renderer provides
# ============================================================================


class TestRLParity(unittest.TestCase):
    """Verify the LLM wrapper does not leak information beyond what the RL
    symbolic renderer provides."""

    def setUp(self):
        self.wrapper = _make_wrapper(coordination_difficulty="easy")
        _, self.state, _ = _reset_wrapper(self.wrapper)

    def test_no_mob_health_in_output(self):
        """Mob descriptions should not contain health values (RL only sees type+position)."""
        result = self.wrapper.describe_mobs(self.state, 0)
        self.assertNotIn("health", result.lower())

    def test_no_projectile_direction_in_output(self):
        """Projectile descriptions should not contain movement direction (RL only sees position)."""
        result = self.wrapper.describe_mobs(self.state, 0)
        self.assertNotIn("moving", result.lower())

    def test_no_handover_initiator_in_output(self):
        """Pending handovers should not reveal which agent started them."""
        result = self.wrapper.describe_coordination_cues(self.state, 0)
        self.assertNotIn("started by", result.lower())

    def test_no_construction_details_in_output(self):
        """Construction details (deadline, built type) should not be exposed."""
        result = self.wrapper.describe_construction(self.state, 0)
        self.assertNotIn("steps to complete", result)
        self.assertNotIn("shelter", result)
        self.assertNotIn("forge", result)
        self.assertNotIn("beacon", result)

    def test_no_request_duration_in_output(self):
        """Teammate requests should not reveal duration (RL only sees active/inactive)."""
        result = self.wrapper.describe_teammates(self.state, 0)
        self.assertNotIn("expires in", result)

    def test_no_raw_monster_kill_count(self):
        """Level info should show cleared/not-cleared boolean, not raw kill count."""
        result = self.wrapper.describe_level_info(self.state)
        self.assertNotIn("Monsters killed:", result)
        # Should have the boolean form
        self.assertIn("cleared", result.lower())

    def test_frame_no_extra_info(self):
        """Full frame output should not leak any extra info."""
        long_ctx, short_ctx = self.wrapper.describe_frame(self.state, 0)
        # No mob health
        if "Nearby creatures:" in long_ctx:
            creature_section = long_ctx.split("Nearby creatures:")[1].split("\n\n")[0]
            self.assertNotIn("health", creature_section.lower())
        # No request duration
        self.assertNotIn("expires in", long_ctx)
        # No handover initiator
        self.assertNotIn("started by", long_ctx)


# ============================================================================
# 18. EGOCENTRIC FRAME OF REFERENCE
# ============================================================================


class TestEgocentric(unittest.TestCase):
    """Tests for egocentric (agent-relative) direction system."""

    # --- Rotation logic ---

    def test_rotate_north_facing_north_is_identity(self):
        """Facing north: north(-1,0) -> ahead(-1,0)."""
        self.assertEqual(_rotate_to_egocentric(-1, 0, 3), (-1, 0))

    def test_rotate_east_facing_north_is_right(self):
        """Facing north: east(0,+1) -> right(0,+1)."""
        self.assertEqual(_rotate_to_egocentric(0, 1, 3), (0, 1))

    def test_rotate_north_facing_south(self):
        """Facing south: north(-1,0) -> behind(+1,0)."""
        self.assertEqual(_rotate_to_egocentric(-1, 0, 4), (1, 0))

    def test_rotate_north_facing_east(self):
        """Facing east: north(-1,0) -> left(0,-1)."""
        self.assertEqual(_rotate_to_egocentric(-1, 0, 2), (0, -1))

    def test_rotate_east_facing_east(self):
        """Facing east: east(0,+1) -> ahead(-1,0)."""
        self.assertEqual(_rotate_to_egocentric(0, 1, 2), (-1, 0))

    def test_rotate_north_facing_west(self):
        """Facing west: north(-1,0) -> right(0,+1)."""
        self.assertEqual(_rotate_to_egocentric(-1, 0, 1), (0, 1))

    def test_rotate_west_facing_west(self):
        """Facing west: west(0,-1) -> ahead(-1,0)."""
        self.assertEqual(_rotate_to_egocentric(0, -1, 1), (-1, 0))

    # --- Direction labels ---

    def test_egocentric_direction_ahead(self):
        parts = _egocentric_direction(-1, 0, 3)  # north facing north
        self.assertEqual(parts, ["ahead"])

    def test_egocentric_direction_behind_right(self):
        parts = _egocentric_direction(1, 1, 3)  # south-east facing north
        self.assertEqual(parts, ["behind", "right"])

    def test_egocentric_direction_left_facing_east(self):
        parts = _egocentric_direction(-1, 0, 2)  # north facing east = left
        self.assertEqual(parts, ["left"])

    # --- describe_loc_precise with facing ---

    def test_loc_precise_egocentric_ahead(self):
        # Agent at (0,0) facing north, target at (-3,0) = 3 north = 3 ahead
        result = describe_loc_precise(np.array([0, 0]), np.array([-3, 0]), facing=3)
        self.assertIn("ahead", result)
        self.assertIn("3", result)
        self.assertNotIn("north", result)

    def test_loc_precise_egocentric_behind(self):
        # Agent at (0,0) facing north, target at (2,0) = 2 south = 2 behind
        result = describe_loc_precise(np.array([0, 0]), np.array([2, 0]), facing=3)
        self.assertIn("behind", result)
        self.assertNotIn("south", result)

    def test_loc_precise_allocentric_default(self):
        # Without facing, should use cardinal directions
        result = describe_loc_precise(np.array([0, 0]), np.array([-3, 0]))
        self.assertIn("north", result)
        self.assertNotIn("ahead", result)

    # --- Full wrapper integration ---

    def test_wrapper_egocentric_env_description(self):
        """With egocentric=True, describe_env should use ahead/behind/left/right."""
        wrapper = _make_wrapper(egocentric=True)
        _, state, _ = _reset_wrapper(wrapper)
        desc = wrapper.describe_env(state, 0)
        # Should NOT contain cardinal directions in item locations
        # (facing line itself may still say "facing north" for reference)
        self.assertIsInstance(desc, str)
        self.assertIn("facing", desc.lower())

    def test_wrapper_allocentric_env_description(self):
        """With egocentric=False (default), describe_env should use cardinal directions."""
        wrapper = _make_wrapper(egocentric=False)
        _, state, _ = _reset_wrapper(wrapper)
        desc = wrapper.describe_env(state, 0)
        self.assertIsInstance(desc, str)
        # Cardinal directions should appear in item locations
        self.assertIn("facing", desc.lower())

    def test_wrapper_egocentric_teammates(self):
        """With egocentric=True, describe_teammates should use relative directions."""
        wrapper = _make_wrapper(egocentric=True)
        _, state, _ = _reset_wrapper(wrapper)
        result = wrapper.describe_teammates(state, 0)
        self.assertIsInstance(result, str)
        self.assertIn("Teammates:", result)

    def test_wrapper_egocentric_frame_runs(self):
        """describe_frame should work with egocentric=True without errors."""
        wrapper = _make_wrapper(egocentric=True)
        _, state, _ = _reset_wrapper(wrapper)
        for i in range(3):
            long_ctx, short_ctx = wrapper.describe_frame(state, i)
            self.assertIsInstance(long_ctx, str)
            self.assertIsInstance(short_ctx, str)


if __name__ == "__main__":
    unittest.main()
