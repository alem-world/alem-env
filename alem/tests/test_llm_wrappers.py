"""Tests for the LLM language wrappers — state-to-text correctness.

Critical invariant: every value in the symbolic JAX state must translate
correctly to the text that an LLM reads. A mismatch (wrong number, wrong
name, wrong condition) is a silent correctness bug — the LLM gets stale or
wrong information and behaves incorrectly.

Each test below:
  1. Takes the post-reset base state.
  2. Applies a targeted .replace() to set exactly the field being tested.
  3. Calls the relevant describe method.
  4. Asserts the exact substring the LLM would read.

Test classes:
  TestInventoryToText      — item counts, vitals, tool tiers, armour, potions
  TestStatusToText         — level names, conditions, role, step counter
  TestTeammatesToText      — dead/alive/off-screen/requesting teammates
  TestStepFeedbackToText   — reward, new achievements, action-failed flag
  TestDescribeFrameRouting — dead/sleeping/active state routing
  TestSingleWrapperOverrides — single-agent text differences

All integration classes use setUpClass to compile JAX once per class.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from alem.alem_coop.alem_state import StaticEnvParams
from alem.alem_coop.constants import MONSTERS_KILLED_TO_CLEAR_LEVEL, OBS_DIM, Action, Specialization
from alem.alem_coop.envs.alem_symbolic_env import AlemCoopSymbolicEnv
from alem.alem_coop.envs.alem_symbolic_env_debug import AlemCoopSymbolicEnvDebug
from alem.alem_env import make_alem_env_from_name
from alem.environment_base.environment_bases import EnvironmentNoAutoReset
from alem.llm.alem_language_wrapper import (
    ACTIONS,
    AlemLanguageWrapper,
    _achievements_for_level,
    _rotate_to_egocentric,
    describe_loc_precise,
    get_instruction_prompt,
    make_alem_env,
)
from alem.llm.alem_language_wrapper_single import (
    AlemLanguageWrapperSingle,
)

# ---------------------------------------------------------------------------
# Helpers shared across integration test classes
# ---------------------------------------------------------------------------


def _make_wrapper(env_name="Alem-Coop-Symbolic", **wrapper_kwargs):
    env = make_alem_env_from_name(env_name)
    return AlemLanguageWrapper(env, env.default_params, **wrapper_kwargs), env


def _reset(wrapper):
    _, state = wrapper.env.reset(jax.random.PRNGKey(0))
    return state


def _ascii_grid_map_rows(rendered):
    """Extract the map-only portion of each ASCII grid row.

    The grid is the OBS_DIM[0] lines after the "N" compass header. Each row is
    a 2-space indent + space-joined cells, optionally followed by an inline
    legend after 3+ spaces. Slicing the fixed map width drops the legend (so
    e.g. the "@=you" legend entry is not mistaken for a second player marker).
    """
    lines = rendered.split("\n")
    n_idx = next(i for i, l in enumerate(lines) if l.strip() == "N")
    grid_rows = lines[n_idx + 1 : n_idx + 1 + OBS_DIM[0]]
    indent = 2
    map_width = 2 * OBS_DIM[1] - 1  # cells joined by single spaces
    return [r[indent : indent + map_width] for r in grid_rows]


# ---------------------------------------------------------------------------
# 1. Inventory → text
# ---------------------------------------------------------------------------


class TestInventoryToText(unittest.TestCase):
    """describe_inventory() must faithfully render every item count and vital."""

    @classmethod
    def setUpClass(cls):
        cls.wrapper, cls.env = _make_wrapper()
        cls.base = _reset(cls.wrapper)

    def _inv(self, **kwargs):
        """Return a state with inventory fields overridden via keyword args."""
        inv = self.base.inventory
        for field, val in kwargs.items():
            inv = inv.replace(**{field: val})
        return self.base.replace(inventory=inv)

    # --- Raw materials ---

    def test_wood_count_rendered_exactly(self):
        state = self._inv(wood=self.base.inventory.wood.at[0].set(7))
        text = self.wrapper.describe_inventory(state, 0)
        self.assertIn("- wood: 7", text)

    def test_zero_wood_omitted(self):
        state = self._inv(wood=self.base.inventory.wood.at[0].set(0))
        text = self.wrapper.describe_inventory(state, 0)
        self.assertNotIn("wood", text)

    def test_multiple_items_each_rendered(self):
        inv = self.base.inventory
        inv = inv.replace(wood=inv.wood.at[0].set(3))
        inv = inv.replace(stone=inv.stone.at[0].set(5))
        inv = inv.replace(coal=inv.coal.at[0].set(2))
        state = self.base.replace(inventory=inv)
        text = self.wrapper.describe_inventory(state, 0)
        self.assertIn("- wood: 3", text)
        self.assertIn("- stone: 5", text)
        self.assertIn("- coal: 2", text)

    def test_zero_items_not_listed_at_all(self):
        """After a fresh start, any zero-count material must be absent."""
        inv = self.base.inventory
        inv = inv.replace(diamond=inv.diamond.at[0].set(0))
        inv = inv.replace(ruby=inv.ruby.at[0].set(0))
        state = self.base.replace(inventory=inv)
        text = self.wrapper.describe_inventory(state, 0)
        self.assertNotIn("- diamond:", text)
        self.assertNotIn("- ruby:", text)

    # --- Tools ---

    def test_pickaxe_tier_names(self):
        tier_names = {1: "wood", 2: "stone", 3: "iron", 4: "diamond"}
        for tier, name in tier_names.items():
            state = self._inv(pickaxe=self.base.inventory.pickaxe.at[0].set(tier))
            text = self.wrapper.describe_inventory(state, 0)
            self.assertIn(f"- pickaxe: {name}", text, msg=f"tier {tier}")

    def test_no_pickaxe_omitted(self):
        state = self._inv(pickaxe=self.base.inventory.pickaxe.at[0].set(0))
        text = self.wrapper.describe_inventory(state, 0)
        self.assertNotIn("pickaxe", text)

    def test_sword_tier_names(self):
        tier_names = {1: "wood", 2: "stone", 3: "iron", 4: "diamond"}
        for tier, name in tier_names.items():
            state = self._inv(sword=self.base.inventory.sword.at[0].set(tier))
            text = self.wrapper.describe_inventory(state, 0)
            self.assertIn(f"- sword: {name}", text, msg=f"tier {tier}")

    def test_bow_present(self):
        state = self._inv(bow=self.base.inventory.bow.at[0].set(1))
        text = self.wrapper.describe_inventory(state, 0)
        self.assertIn("- bow: yes", text)

    def test_bow_absent_omitted(self):
        state = self._inv(bow=self.base.inventory.bow.at[0].set(0))
        text = self.wrapper.describe_inventory(state, 0)
        self.assertNotIn("bow", text)

    # --- Armour ---

    def test_armour_single_tier_iron(self):
        # 2x iron armour
        new_armour = self.base.inventory.armour.at[0].set(jnp.array([1, 1, 0, 0]))
        state = self._inv(armour=new_armour)
        text = self.wrapper.describe_inventory(state, 0)
        self.assertIn("2x iron", text)

    def test_armour_mixed_tiers(self):
        # 1 diamond + 2 iron
        new_armour = self.base.inventory.armour.at[0].set(jnp.array([2, 1, 1, 0]))
        state = self._inv(armour=new_armour)
        text = self.wrapper.describe_inventory(state, 0)
        self.assertIn("1x diamond", text)
        self.assertIn("2x iron", text)

    def test_armour_all_zero_omitted(self):
        new_armour = self.base.inventory.armour.at[0].set(jnp.zeros(4, dtype=jnp.int32))
        state = self._inv(armour=new_armour)
        text = self.wrapper.describe_inventory(state, 0)
        self.assertNotIn("armour:", text)

    # --- Potions ---

    def test_potion_red_count(self):
        new_potions = self.base.inventory.potions.at[0, 0].set(3)
        state = self._inv(potions=new_potions)
        text = self.wrapper.describe_inventory(state, 0)
        self.assertIn("- potion_red: 3", text)

    def test_potion_blue_count(self):
        new_potions = self.base.inventory.potions.at[0, 2].set(2)
        state = self._inv(potions=new_potions)
        text = self.wrapper.describe_inventory(state, 0)
        self.assertIn("- potion_blue: 2", text)

    def test_zero_potions_omitted(self):
        new_potions = self.base.inventory.potions.at[0].set(jnp.zeros(6, dtype=jnp.int32))
        state = self._inv(potions=new_potions)
        text = self.wrapper.describe_inventory(state, 0)
        self.assertNotIn("potion_red", text)
        self.assertNotIn("potion_blue", text)

    # --- Vitals ---

    def test_health_value_exact(self):
        state = self.base.replace(player_health=self.base.player_health.at[0].set(6.0))
        text = self.wrapper.describe_inventory(state, 0)
        self.assertIn("- health: 6", text)

    def test_food_value_exact(self):
        state = self.base.replace(player_food=self.base.player_food.at[0].set(4))
        text = self.wrapper.describe_inventory(state, 0)
        self.assertIn("- food: 4", text)

    def test_energy_value_exact(self):
        state = self.base.replace(player_energy=self.base.player_energy.at[0].set(3))
        text = self.wrapper.describe_inventory(state, 0)
        self.assertIn("- energy: 3", text)

    def test_empty_inventory_message(self):
        """A player with no items gets a clear 'nothing' message."""
        inv = self.base.inventory
        for field in (
            "wood",
            "stone",
            "coal",
            "iron",
            "diamond",
            "ruby",
            "sapphire",
            "sapling",
            "arrows",
            "torches",
        ):
            inv = inv.replace(**{field: getattr(inv, field).at[0].set(0)})
        inv = inv.replace(pickaxe=inv.pickaxe.at[0].set(0))
        inv = inv.replace(sword=inv.sword.at[0].set(0))
        inv = inv.replace(bow=inv.bow.at[0].set(0))
        inv = inv.replace(armour=inv.armour.at[0].set(jnp.zeros(4, dtype=jnp.int32)))
        inv = inv.replace(potions=inv.potions.at[0].set(jnp.zeros(6, dtype=jnp.int32)))
        inv = inv.replace(books=inv.books.at[0].set(0))
        state = self.base.replace(inventory=inv)
        state = state.replace(
            sword_enchantment=state.sword_enchantment.at[0].set(0),
            bow_enchantment=state.bow_enchantment.at[0].set(0),
            learned_spells=state.learned_spells.at[0].set(False),
        )
        text = self.wrapper.describe_inventory(state, 0)
        self.assertIn("nothing in your inventory", text)

    def test_correct_player_index_used(self):
        """Items belonging to player 1 must not appear in player 0's inventory."""
        state = self._inv(wood=self.base.inventory.wood.at[1].set(9))
        text = self.wrapper.describe_inventory(state, 0)
        self.assertNotIn("- wood: 9", text)


# ---------------------------------------------------------------------------
# 2. Status → text
# ---------------------------------------------------------------------------


class TestStatusToText(unittest.TestCase):
    """describe_status() must reflect player level, condition, role, and step."""

    @classmethod
    def setUpClass(cls):
        cls.wrapper, cls.env = _make_wrapper()
        cls.base = _reset(cls.wrapper)

    # --- Location ---

    def test_level_0_shows_overworld(self):
        state = self.base.replace(player_level=jnp.int32(0))
        text = self.wrapper.describe_status(state, 0)
        self.assertIn("Overworld", text)
        self.assertIn("surface", text)

    def test_level_1_shows_gnomish_mines(self):
        state = self.base.replace(player_level=jnp.int32(1))
        text = self.wrapper.describe_status(state, 0)
        self.assertIn("Gnomish Mines", text)
        self.assertIn("dungeon level 1", text)

    def test_level_2_shows_dungeon(self):
        state = self.base.replace(player_level=jnp.int32(2))
        text = self.wrapper.describe_status(state, 0)
        self.assertIn("Dungeon", text)

    def test_level_3_shows_sewers(self):
        state = self.base.replace(player_level=jnp.int32(3))
        text = self.wrapper.describe_status(state, 0)
        self.assertIn("Sewers", text)

    # --- Conditions ---

    def test_dead_condition(self):
        state = self.base.replace(player_health=self.base.player_health.at[0].set(0.0))
        text = self.wrapper.describe_status(state, 0)
        self.assertIn("dead", text)

    def test_sleeping_condition(self):
        state = self.base.replace(is_sleeping=self.base.is_sleeping.at[0].set(True))
        text = self.wrapper.describe_status(state, 0)
        self.assertIn("sleeping", text)

    def test_resting_condition(self):
        state = self.base.replace(is_resting=self.base.is_resting.at[0].set(True))
        text = self.wrapper.describe_status(state, 0)
        self.assertIn("resting", text)

    def test_sleeping_takes_priority_over_resting(self):
        state = self.base.replace(
            is_sleeping=self.base.is_sleeping.at[0].set(True),
            is_resting=self.base.is_resting.at[0].set(True),
        )
        text = self.wrapper.describe_status(state, 0)
        self.assertIn("sleeping", text)
        self.assertNotIn("resting", text)

    # --- Role ---

    def test_warrior_role_shown(self):
        state = self.base.replace(
            player_specialization=self.base.player_specialization.at[0].set(
                Specialization.WARRIOR.value
            )
        )
        text = self.wrapper.describe_status(state, 0)
        self.assertIn("warrior", text.lower())

    def test_forager_role_shown(self):
        state = self.base.replace(
            player_specialization=self.base.player_specialization.at[0].set(
                Specialization.FORAGER.value
            )
        )
        text = self.wrapper.describe_status(state, 0)
        self.assertIn("forager", text.lower())

    def test_no_role_when_unspecialized(self):
        state = self.base.replace(
            player_specialization=self.base.player_specialization.at[0].set(0)
        )
        text = self.wrapper.describe_status(state, 0)
        self.assertNotIn("Role:", text)

    # --- Step counter ---

    def test_timestep_in_status(self):
        state = self.base.replace(timestep=jnp.int32(123))
        text = self.wrapper.describe_status(state, 0)
        self.assertIn("123/", text)

    def test_remaining_steps_correct(self):
        max_ts = int(self.wrapper.env_params.max_timesteps)
        state = self.base.replace(timestep=jnp.int32(100))
        text = self.wrapper.describe_status(state, 0)
        remaining = max_ts - 100
        self.assertIn(str(remaining), text)


# ---------------------------------------------------------------------------
# 3. Teammates → text
# ---------------------------------------------------------------------------


class TestTeammatesToText(unittest.TestCase):
    """describe_teammates() must correctly report each teammate's status."""

    @classmethod
    def setUpClass(cls):
        cls.wrapper, cls.env = _make_wrapper()
        cls.base = _reset(cls.wrapper)

    def test_dead_teammate_shows_dead(self):
        state = self.base.replace(player_alive=self.base.player_alive.at[1].set(False))
        text = self.wrapper.describe_teammates(state, 0)
        self.assertIn("Agent 1 is dead", text)

    def test_alive_teammate_health_rendered(self):
        state = self.base.replace(player_health=self.base.player_health.at[1].set(5.0))
        text = self.wrapper.describe_teammates(state, 0)
        self.assertIn("health=5", text)

    def test_off_screen_teammate_shows_direction(self):
        """Move agent 1 far south so they leave the 9×11 view window."""
        agent0_row = int(self.base.player_position[0][0])
        far_row = agent0_row + 30
        far_pos = self.base.player_position.at[1].set(
            jnp.array([far_row, int(self.base.player_position[1][1])])
        )
        state = self.base.replace(player_position=far_pos)
        text = self.wrapper.describe_teammates(state, 0)
        self.assertIn("off-screen", text)
        self.assertIn("south", text)

    def test_requesting_teammate_shows_request_type(self):
        state = self.base.replace(
            request_type=self.base.request_type.at[1].set(Action.REQUEST_WOOD.value),
            request_duration=self.base.request_duration.at[1].set(10),
        )
        text = self.wrapper.describe_teammates(state, 0)
        self.assertIn("Requesting wood", text)

    def test_self_not_in_teammate_text(self):
        text = self.wrapper.describe_teammates(self.base, 0)
        # Player 0 describes teammates — "Agent 0" should not appear
        self.assertNotIn("Agent 0", text)

    def test_teammate_specialization_shown(self):
        state = self.base.replace(
            player_specialization=self.base.player_specialization.at[1].set(
                Specialization.WARRIOR.value
            )
        )
        text = self.wrapper.describe_teammates(state, 0)
        self.assertIn("warrior", text.lower())

    def test_no_teammates_when_solo(self):
        """Single-agent wrapper must always return empty string."""
        env = make_alem_env_from_name("Alem-SingleAgent-Symbolic")
        wrapper = AlemLanguageWrapperSingle(env, env.default_params)
        _, state = env.reset(jax.random.PRNGKey(0))
        self.assertEqual(wrapper.describe_teammates(state, 0), "")


# ---------------------------------------------------------------------------
# 4. Step feedback → text
# ---------------------------------------------------------------------------


class TestStepFeedbackToText(unittest.TestCase):
    """describe_step_feedback() must render reward, achievements, and failures."""

    @classmethod
    def setUpClass(cls):
        cls.wrapper, cls.env = _make_wrapper()
        cls.base = _reset(cls.wrapper)

    def test_reward_formatted_correctly(self):
        text = self.wrapper.describe_step_feedback(reward=0.5, new_achievements=[])
        self.assertIn("+0.500", text)

    def test_negative_reward_formatted(self):
        text = self.wrapper.describe_step_feedback(reward=-0.25, new_achievements=[])
        self.assertIn("-0.250", text)

    def test_new_achievement_shown(self):
        text = self.wrapper.describe_step_feedback(reward=None, new_achievements=["COLLECT_WOOD"])
        self.assertIn("Collect Wood", text)

    def test_multiple_achievements_shown(self):
        text = self.wrapper.describe_step_feedback(
            reward=None, new_achievements=["COLLECT_WOOD", "PLACE_TABLE"]
        )
        self.assertIn("Collect Wood", text)
        self.assertIn("Place Table", text)

    def test_action_failed_message(self):
        text = self.wrapper.describe_step_feedback(
            reward=None,
            new_achievements=[],
            last_action="Make Iron Sword",
            action_failed=True,
        )
        self.assertIn("Make Iron Sword", text)
        self.assertIn("had no effect", text)

    def test_successful_action_no_failure_note(self):
        text = self.wrapper.describe_step_feedback(
            reward=None,
            new_achievements=[],
            last_action="Move North",
            action_failed=False,
        )
        self.assertIn("Move North", text)
        self.assertNotIn("had no effect", text)

    def test_no_feedback_returns_empty(self):
        text = self.wrapper.describe_step_feedback(
            reward=None, new_achievements=None, last_action=None
        )
        self.assertEqual(text, "")


# ---------------------------------------------------------------------------
# 5. describe_frame routing
# ---------------------------------------------------------------------------


class TestDescribeFrameRouting(unittest.TestCase):
    """describe_frame() must route to the correct compressed/full observation."""

    @classmethod
    def setUpClass(cls):
        cls.wrapper, cls.env = _make_wrapper()
        cls.base = _reset(cls.wrapper)

    def test_active_frame_contains_location(self):
        long_term, _ = self.wrapper.describe_frame(self.base, 0)
        self.assertIn("Location", long_term)

    def test_active_frame_inventory_contains_health(self):
        _, inv_text = self.wrapper.describe_frame(self.base, 0)
        self.assertIn("health", inv_text.lower())

    def test_dead_frame_long_term_says_dead(self):
        state = self.base.replace(player_health=self.base.player_health.at[0].set(0.0))
        long_term, _ = self.wrapper.describe_frame(state, 0)
        self.assertIn("dead", long_term.lower())

    def test_sleeping_frame_long_term_says_sleeping(self):
        state = self.base.replace(is_sleeping=self.base.is_sleeping.at[0].set(True))
        long_term, _ = self.wrapper.describe_frame(state, 0)
        self.assertIn("sleeping", long_term.lower())

    def test_resting_frame_long_term_says_resting(self):
        state = self.base.replace(is_resting=self.base.is_resting.at[0].set(True))
        long_term, _ = self.wrapper.describe_frame(state, 0)
        self.assertIn("resting", long_term.lower())

    def test_achievement_feedback_in_frame(self):
        long_term, _ = self.wrapper.describe_frame(self.base, 0, new_achievements=["COLLECT_WOOD"])
        self.assertIn("Collect Wood", long_term)

    def test_failed_action_feedback_in_frame(self):
        long_term, _ = self.wrapper.describe_frame(
            self.base, 0, last_action="Make Diamond Pickaxe", action_failed=True
        )
        self.assertIn("Make Diamond Pickaxe", long_term)
        self.assertIn("had no effect", long_term)

    def test_reward_feedback_in_frame(self):
        long_term, _ = self.wrapper.describe_frame(self.base, 0, reward=1.0)
        self.assertIn("+1.000", long_term)

    def test_frame_returns_two_strings(self):
        result = self.wrapper.describe_frame(self.base, 0)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], str)
        self.assertIsInstance(result[1], str)


# ---------------------------------------------------------------------------
# 6. Single-agent wrapper overrides
# ---------------------------------------------------------------------------


class TestSingleWrapperOverrides(unittest.TestCase):
    """AlemLanguageWrapperSingle must differ from the base wrapper in specific ways."""

    @classmethod
    def setUpClass(cls):
        env = make_alem_env_from_name("Alem-SingleAgent-Symbolic")
        cls.wrapper = AlemLanguageWrapperSingle(env, env.default_params)
        _, cls.base = env.reset(jax.random.PRNGKey(0))

    def test_describe_teammates_always_empty(self):
        self.assertEqual(self.wrapper.describe_teammates(self.base, 0), "")

    def test_describe_status_says_ends_early_if_you_die(self):
        text = self.wrapper.describe_status(self.base, 0)
        self.assertIn("ends early if you die", text)

    def test_describe_status_not_if_all_agents_die(self):
        text = self.wrapper.describe_status(self.base, 0)
        self.assertNotIn("if all agents die", text)

    def test_dead_frame_says_game_over(self):
        state = self.base.replace(player_health=self.base.player_health.at[0].set(0.0))
        long_term, _ = self.wrapper.describe_frame(state, 0)
        self.assertIn("game over", long_term.lower())

    def test_dead_frame_no_coordinate_with_teammates(self):
        state = self.base.replace(player_health=self.base.player_health.at[0].set(0.0))
        long_term, _ = self.wrapper.describe_frame(state, 0)
        self.assertNotIn("communicate", long_term.lower())

    def test_prompt_mode_collaborative_coerced_to_specific(self):
        env = make_alem_env_from_name("Alem-SingleAgent-Symbolic")
        w = AlemLanguageWrapperSingle(env, env.default_params, prompt_mode="specific_collaborative")
        self.assertEqual(w.prompt_mode, "specific")

    def test_system_prompt_excludes_request_and_give(self):
        prompt = self.wrapper.get_instruction_prompt()
        self.assertNotIn("Request Food", prompt)
        self.assertNotIn("Give", prompt)


# ---------------------------------------------------------------------------
# 7. Pure helper tests (no env, no JAX)
# ---------------------------------------------------------------------------


class TestGetInstructionPrompt(unittest.TestCase):
    def test_invalid_prompt_mode_raises(self):
        with self.assertRaises(ValueError):
            get_instruction_prompt(prompt_mode="bad_mode")

    def test_all_valid_modes_return_string(self):
        for mode in ("general", "specific", "specific_collaborative"):
            p = get_instruction_prompt(prompt_mode=mode)
            self.assertIsInstance(p, str)
            self.assertGreater(len(p), 200)

    def test_coordination_section_only_in_collaborative(self):
        p_collab = get_instruction_prompt(
            prompt_mode="specific_collaborative", coordination_enabled=True
        )
        p_specific = get_instruction_prompt(prompt_mode="specific")
        self.assertIn("Coordination", p_collab)
        self.assertNotIn("Coordination", p_specific)

    def test_boss_section_gated_by_progressive_disclosure(self):
        p0 = get_instruction_prompt(
            prompt_mode="specific", progressive_disclosure=True, current_level=0
        )
        p5 = get_instruction_prompt(
            prompt_mode="specific", progressive_disclosure=True, current_level=5
        )
        self.assertNotIn("Boss", p0)
        self.assertIn("Boss", p5)


class TestAchievementsForLevel(unittest.TestCase):
    def test_overworld_excludes_sapphire(self):
        ach = _achievements_for_level(0, coordination_enabled=False)
        self.assertNotIn("Collect Sapphire", ach)

    def test_level1_includes_sapphire(self):
        ach = _achievements_for_level(1, coordination_enabled=False)
        self.assertIn("Collect Sapphire", ach)

    def test_coord_gated_by_flag(self):
        with_coord = _achievements_for_level(0, coordination_enabled=True)
        without_coord = _achievements_for_level(0, coordination_enabled=False)
        self.assertIn("Coord 2 Agents Soft", with_coord)
        self.assertNotIn("Coord 2 Agents Soft", without_coord)


class TestDirectionHelpers(unittest.TestCase):
    def test_same_position(self):
        self.assertEqual(describe_loc_precise([5, 5], [5, 5]), "at your location")

    def test_north_cardinal(self):
        result = describe_loc_precise([5, 5], [2, 5])
        self.assertIn("north", result)

    def test_east_cardinal(self):
        result = describe_loc_precise([5, 5], [5, 8])
        self.assertIn("east", result)

    def test_facing_north_is_identity(self):
        self.assertEqual(_rotate_to_egocentric(-1, 0, 3), (-1, 0))
        self.assertEqual(_rotate_to_egocentric(0, 1, 3), (0, 1))

    def test_facing_south_flips_180(self):
        self.assertEqual(_rotate_to_egocentric(-1, 0, 4), (1, 0))
        self.assertEqual(_rotate_to_egocentric(0, 1, 4), (0, -1))

    def test_tile_ahead_always_negative_ego_row(self):
        face_to_delta = {3: (-1, 0), 4: (1, 0), 2: (0, 1), 1: (0, -1)}
        for facing, (dr, dc) in face_to_delta.items():
            ego_dr, _ = _rotate_to_egocentric(dr, dc, facing)
            self.assertLess(ego_dr, 0, msg=f"facing={facing}: tile ahead should have ego_dr<0")


# ---------------------------------------------------------------------------
# 10. num_agents parameter correctly routes to static env params
# ---------------------------------------------------------------------------


class TestEnvNumAgentsParam(unittest.TestCase):
    """The num_agents constructor arg must wire through to StaticEnvParams.player_count.
    Previously it was silently ignored, leaving the env with the default (3)
    regardless of what the caller requested.
    """

    def test_symbolic_two_agents_player_count_is_2(self):
        env = AlemCoopSymbolicEnv(num_agents=2)
        self.assertEqual(env.static_env_params.player_count, 2)

    def test_symbolic_two_agents_num_agents_attr_is_2(self):
        env = AlemCoopSymbolicEnv(num_agents=2)
        self.assertEqual(env.num_agents, 2)

    def test_symbolic_two_agents_agents_list_has_two_entries(self):
        env = AlemCoopSymbolicEnv(num_agents=2)
        self.assertEqual(len(env.agents), 2)
        self.assertIn("agent_0", env.agents)
        self.assertIn("agent_1", env.agents)
        self.assertNotIn("agent_2", env.agents)

    def test_symbolic_explicit_static_params_wins_over_num_agents(self):
        # If caller passes both, static_env_params takes precedence.
        env = AlemCoopSymbolicEnv(num_agents=2, static_env_params=StaticEnvParams(player_count=4))
        self.assertEqual(env.num_agents, 4)

    def test_symbolic_default_is_3(self):
        env = AlemCoopSymbolicEnv()
        self.assertEqual(env.num_agents, 3)

    def test_debug_two_agents_player_count_is_2(self):
        env = AlemCoopSymbolicEnvDebug(num_agents=2)
        self.assertEqual(env.static_env_params.player_count, 2)

    def test_debug_two_agents_num_agents_attr_is_2(self):
        env = AlemCoopSymbolicEnvDebug(num_agents=2)
        self.assertEqual(env.num_agents, 2)

    def test_debug_two_agents_agents_list_has_two_entries(self):
        env = AlemCoopSymbolicEnvDebug(num_agents=2)
        self.assertEqual(len(env.agents), 2)

    def test_debug_explicit_static_params_wins(self):
        env = AlemCoopSymbolicEnvDebug(
            num_agents=2, static_env_params=StaticEnvParams(player_count=4, num_levels=1)
        )
        self.assertEqual(env.num_agents, 4)


# ---------------------------------------------------------------------------
# 11. make_alem_env factory
# ---------------------------------------------------------------------------


class TestMakeAlemEnv(unittest.TestCase):
    def test_coordination_enabled_flag_set(self):
        env = make_alem_env({"coordination_difficulty": "easy", "num_agents": 3})
        self.assertTrue(env.default_params.coordination_enabled)

    def test_coordination_disabled_by_default(self):
        env = make_alem_env({})
        self.assertFalse(env.default_params.coordination_enabled)

    def test_max_timesteps_applied(self):
        env = make_alem_env({"max_timesteps": 500})
        self.assertEqual(int(env.default_params.max_timesteps), 500)

    def test_num_agents_routes_to_player_count(self):
        env = make_alem_env({"num_agents": 2})
        self.assertEqual(int(env.static_env_params.player_count), 2)

    def test_god_mode_applied(self):
        env = make_alem_env({"god_mode": True})
        self.assertTrue(env.default_params.god_mode)


# ---------------------------------------------------------------------------
# 12. Wrapper lifecycle — reset() and step()
# ---------------------------------------------------------------------------


class TestWrapperLifecycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wrapper, cls.env = _make_wrapper()
        cls.rng = jax.random.PRNGKey(42)

    def test_reset_zeroes_all_score_trackers(self):
        self.wrapper.reset(self.rng)
        self.assertTrue(all(s == 0.0 for s in self.wrapper.score_trackers))

    def test_reset_obs_contains_location_text(self):
        obs_list, _, _ = self.wrapper.reset(self.rng)
        for obs in obs_list:
            self.assertIn("Location:", obs["text"]["long_term_context"])

    def test_step_advances_timestep_by_one(self):
        obs_list, state, rng = self.wrapper.reset(self.rng)
        before = int(state.timestep)
        _, new_state, _, _, _, _ = self.wrapper.step(state, ["Noop"] * self.wrapper.num_agents, rng)
        self.assertEqual(int(new_state.timestep), before + 1)

    def test_first_step_all_agents_alive(self):
        obs_list, state, rng = self.wrapper.reset(self.rng)
        _, _, _, dones, _, _ = self.wrapper.step(state, ["Noop"] * self.wrapper.num_agents, rng)
        self.assertTrue(all(not d for d in dones))

    def test_step_obs_shows_incremented_step_counter(self):
        obs_list, state, rng = self.wrapper.reset(self.rng)
        obs_after, _, _, _, _, _ = self.wrapper.step(state, ["Noop"] * self.wrapper.num_agents, rng)
        for obs in obs_after:
            self.assertIn("Step: 1/", obs["text"]["long_term_context"])

    def test_integer_action_advances_timestep(self):
        obs_list, state, rng = self.wrapper.reset(self.rng)
        _, new_state, _, _, _, _ = self.wrapper.step(state, [0] * self.wrapper.num_agents, rng)
        self.assertEqual(int(new_state.timestep), 1)


# ---------------------------------------------------------------------------
# 12. get_action_index
# ---------------------------------------------------------------------------


class TestGetActionIndex(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wrapper, _ = _make_wrapper()
        cls.wrapper.reset(jax.random.PRNGKey(0))  # initialise failed_candidates list

    def test_noop_returns_0(self):
        self.assertEqual(self.wrapper.get_action_index("Noop", 0), 0)

    def test_move_north_returns_correct_idx(self):
        idx = self.wrapper.get_action_index("Move North", 0)
        self.assertEqual(idx, ACTIONS.index("Move North"))

    def test_move_west_returns_correct_idx(self):
        idx = self.wrapper.get_action_index("Move West", 0)
        self.assertEqual(idx, ACTIONS.index("Move West"))

    def test_invalid_action_returns_noop(self):
        self.assertEqual(self.wrapper.get_action_index("FlyAway", 0), 0)

    def test_empty_string_returns_noop(self):
        self.assertEqual(self.wrapper.get_action_index("", 0), 0)

    def test_none_returns_noop(self):
        self.assertEqual(self.wrapper.get_action_index(None, 0), 0)

    def test_case_insensitive_match(self):
        idx = self.wrapper.get_action_index("move north", 0)
        self.assertEqual(idx, ACTIONS.index("Move North"))

    def test_give_to_agent_1_from_agent_0(self):
        # agent 0 giving to agent 1: slot=0 (target 1 >= giver 0 → slot = target-1 = 0)
        idx = self.wrapper.get_action_index("Give to Agent 1", 0)
        self.assertEqual(idx, Action.GIVE.value + 0)

    def test_give_to_agent_2_from_agent_0(self):
        # agent 0 giving to agent 2: slot=1
        idx = self.wrapper.get_action_index("Give to Agent 2", 0)
        self.assertEqual(idx, Action.GIVE.value + 1)

    def test_give_to_self_returns_noop(self):
        # Agent 0 giving to Agent 0 is invalid
        idx = self.wrapper.get_action_index("Give to Agent 0", 0)
        self.assertEqual(idx, 0)

    def test_invalid_action_tracked_in_failed_candidates(self):
        before = len(self.wrapper.failed_candidates[0])
        self.wrapper.get_action_index("NotAnAction", 0)
        self.assertEqual(len(self.wrapper.failed_candidates[0]), before + 1)


# ---------------------------------------------------------------------------
# 13. describe_level_info branches
# ---------------------------------------------------------------------------


class TestDescribeLevelInfo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wrapper, cls.env = _make_wrapper()
        cls.base = _reset(cls.wrapper)

    def test_bright_light(self):
        state = self.base.replace(light_level=jnp.float32(0.9))
        text = self.wrapper.describe_level_info(state)
        self.assertIn("bright", text)

    def test_dim_light(self):
        state = self.base.replace(light_level=jnp.float32(0.5))
        text = self.wrapper.describe_level_info(state)
        self.assertIn("dim", text)

    def test_dark_light(self):
        state = self.base.replace(light_level=jnp.float32(0.1))
        text = self.wrapper.describe_level_info(state)
        self.assertIn("dark", text)

    def test_light_value_shown(self):
        state = self.base.replace(light_level=jnp.float32(0.5))
        text = self.wrapper.describe_level_info(state)
        self.assertIn("0.50", text)

    def test_level_not_cleared(self):
        state = self.base.replace(monsters_killed=self.base.monsters_killed.at[0].set(0))
        text = self.wrapper.describe_level_info(state)
        self.assertIn("not yet cleared", text)

    def test_level_cleared(self):
        state = self.base.replace(
            monsters_killed=self.base.monsters_killed.at[0].set(MONSTERS_KILLED_TO_CLEAR_LEVEL)
        )
        text = self.wrapper.describe_level_info(state)
        self.assertIn("cleared", text)

    def test_returns_level_info_header(self):
        text = self.wrapper.describe_level_info(self.base)
        self.assertTrue(text.startswith("Level info:"))


# ---------------------------------------------------------------------------
# 14. describe_mobs and describe_coordination_cues
# ---------------------------------------------------------------------------


class TestDescribeMobsAndCoord(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wrapper, cls.env = _make_wrapper()
        cls.base = _reset(cls.wrapper)

    def test_no_mobs_outside_view_gives_empty_string(self):
        # Clear every mob mask so nothing is alive anywhere, then verify silence.
        level = int(self.base.player_level)
        cleared = self.base.melee_mobs.replace(
            mask=self.base.melee_mobs.mask.at[level].set(
                jnp.zeros_like(self.base.melee_mobs.mask[level])
            )
        )
        state = self.base.replace(melee_mobs=cleared)
        # Similarly clear ranged and passive
        state = state.replace(
            ranged_mobs=state.ranged_mobs.replace(
                mask=state.ranged_mobs.mask.at[level].set(
                    jnp.zeros_like(state.ranged_mobs.mask[level])
                )
            ),
            passive_mobs=state.passive_mobs.replace(
                mask=state.passive_mobs.mask.at[level].set(
                    jnp.zeros_like(state.passive_mobs.mask[level])
                )
            ),
        )
        result = self.wrapper.describe_mobs(state, 0)
        self.assertEqual(result, "")

    def test_visible_melee_mob_appears_in_output(self):
        level = int(self.base.player_level)
        player_pos = np.array(self.base.player_position[0])
        # Place mob one step east, guaranteed in the 9×11 view
        mob_r, mob_c = int(player_pos[0]), int(player_pos[1]) + 1
        state = self.base.replace(
            melee_mobs=self.base.melee_mobs.replace(
                mask=self.base.melee_mobs.mask.at[level, 0].set(True),
                position=self.base.melee_mobs.position.at[level, 0].set(jnp.array([mob_r, mob_c])),
            ),
            # Ensure the mob tile is lit so it isn't hidden by darkness
            light_map=self.base.light_map.at[level, mob_r, mob_c].set(1.0),
        )
        result = self.wrapper.describe_mobs(state, 0)
        self.assertIn("Nearby creatures:", result)

    def test_coordination_cues_empty_on_zero_coord_map(self):
        # coordination_map is all zeros after a plain reset; nothing to report.
        result = self.wrapper.describe_coordination_cues(self.base, 0)
        self.assertEqual(result, "")

    def test_coordination_cue_visible_tile_appears_in_output(self):
        # Place a coord cue (value=2) one step east of the player and check text.
        level = int(self.base.player_level)
        player_pos = np.array(self.base.player_position[0])
        cr, cc = int(player_pos[0]), int(player_pos[1]) + 1
        state = self.base.replace(
            coordination_map=self.base.coordination_map.at[level, cr, cc].set(2),
            light_map=self.base.light_map.at[level, cr, cc].set(1.0),
        )
        result = self.wrapper.describe_coordination_cues(state, 0)
        self.assertIn("Coordination:", result)
        self.assertIn("2 agents", result)

    def test_ascii_describe_frame_places_player_at_center(self):
        ascii_wrapper, _ = _make_wrapper(use_ascii=True)
        state = _reset(ascii_wrapper)
        long_term, _ = ascii_wrapper.describe_frame(state, 0)
        # describe_frame prepends status/level-info before the grid, so anchor
        # on the "N" header rather than assuming the grid leads the output.
        grid_rows = _ascii_grid_map_rows(long_term)
        self.assertIn("@", grid_rows[OBS_DIM[0] // 2])  # centre row


# ---------------------------------------------------------------------------
# 15. update_progress, get_stats, get_affordances, check_action_validity
# ---------------------------------------------------------------------------


class TestWrapperUtilMethods(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wrapper, cls.env = _make_wrapper()
        cls.base = _reset(cls.wrapper)
        cls.wrapper.reset(jax.random.PRNGKey(0))

    def test_score_zero_after_reset(self):
        self.assertEqual(self.wrapper.score_trackers[0], 0.0)

    def test_update_progress_records_all_achievement_entries(self):
        from alem.alem_coop.constants import Achievement

        self.wrapper.update_progress(self.base, 0)
        self.assertEqual(
            set(self.wrapper.achievements[0].keys()),
            {a.name for a in Achievement},
        )

    def test_achievement_values_are_0_after_reset(self):
        self.wrapper.update_progress(self.base, 0)
        self.assertTrue(all(v == 0 for v in self.wrapper.achievements[0].values()))

    def test_get_stats_score_zero_after_reset(self):
        self.wrapper.update_progress(self.base, 0)
        self.assertEqual(self.wrapper.get_stats(0)["score"], 0.0)

    def test_get_stats_progression_is_zero_after_reset(self):
        self.wrapper.update_progress(self.base, 0)
        self.assertEqual(self.wrapper.get_stats(0)["progression"], 0.0)

    def test_get_stats_all_agents_length_matches_num_agents(self):
        for i in range(self.wrapper.num_agents):
            self.wrapper.update_progress(self.base, i)
        self.assertEqual(len(self.wrapper.get_stats()), self.wrapper.num_agents)

    def test_noop_always_in_affordances(self):
        # Action.NOOP (index 0) is always unmasked — Noop must always appear.
        result = self.wrapper.get_affordances(self.base, 0)
        self.assertIn("Noop", result)

    def test_affordances_header_present_when_actions_available(self):
        result = self.wrapper.get_affordances(self.base, 0)
        self.assertIn("Available actions:", result)

    def test_check_action_validity_exact_match(self):
        self.assertEqual(self.wrapper.check_action_validity("Noop"), "Noop")

    def test_check_action_validity_unknown_returns_noop(self):
        self.assertEqual(self.wrapper.check_action_validity("FlyAway"), "Noop")

    def test_instruction_prompt_agent_0_is_warrior(self):
        self.assertIn("warrior", self.wrapper.get_instruction_prompt(agent_idx=0))

    def test_instruction_prompt_agent_1_is_forager(self):
        self.assertIn("forager", self.wrapper.get_instruction_prompt(agent_idx=1))

    def test_instruction_prompt_agent_2_is_miner(self):
        self.assertIn("miner", self.wrapper.get_instruction_prompt(agent_idx=2))


# ---------------------------------------------------------------------------
# 16. ascii_map via wrapper.render_ascii_map
# ---------------------------------------------------------------------------


class TestAsciiMapViaWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wrapper, cls.env = _make_wrapper(use_ascii=True)
        cls.base = _reset(cls.wrapper)
        cls.rendered = cls.wrapper.render_ascii_map(cls.base, 0)
        cls.lines = cls.rendered.split("\n")

    def test_grid_has_nine_data_rows(self):
        grid_rows = _ascii_grid_map_rows(self.rendered)
        self.assertEqual(len(grid_rows), OBS_DIM[0])
        self.assertTrue(all(r.strip() for r in grid_rows))

    def test_player_at_center_row(self):
        # First line is "N" header; rows 1-9 are the grid.
        grid_rows = [l for l in self.lines if l.strip()][1:10]  # skip N header
        center = grid_rows[4]  # row 4 (0-indexed) = centre of 9-row grid
        self.assertIn("@", center)

    def test_north_compass_header(self):
        self.assertIn("N", self.lines[0])

    def test_player_symbol_appears_exactly_once(self):
        # The player marker appears once in the grid. (The legend also carries an
        # "@=you" entry to explain the symbol; that is intentional and excluded
        # here by counting only the map portion of the grid rows.)
        grid_map = "\n".join(_ascii_grid_map_rows(self.rendered))
        self.assertEqual(grid_map.count("@"), 1)

    def test_facing_line_present(self):
        self.assertIn("Facing:", self.rendered)

    def test_each_player_sees_themselves_at_center(self):
        for i in range(self.wrapper.num_agents):
            r = self.wrapper.render_ascii_map(self.base, i)
            grid_rows = [l for l in r.split("\n") if l.strip()][1:10]
            self.assertIn("@", grid_rows[4])


# ---------------------------------------------------------------------------
# 17. EnvironmentNoAutoReset abstract interface
# ---------------------------------------------------------------------------


class TestEnvironmentBases(unittest.TestCase):
    def setUp(self):
        self.env = EnvironmentNoAutoReset()

    def test_step_env_raises_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.env.step_env(None, None, None, None)

    def test_reset_env_raises_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.env.reset_env(None, None)

    def test_get_obs_raises_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.env.get_obs(None)

    def test_is_terminal_raises_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.env.is_terminal(None, None)

    def test_num_actions_raises_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            _ = self.env.num_actions

    def test_action_space_raises_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.env.action_space(None)

    def test_observation_space_raises_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.env.observation_space(None)

    def test_state_space_raises_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.env.state_space(None)

    def test_default_params_raises_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            _ = self.env.default_params

    def test_discount_returns_one_when_not_terminal(self):
        class ConcreteEnv(EnvironmentNoAutoReset):
            @property
            def default_params(self):
                return None

            def is_terminal(self, state, params):
                return jnp.array(False)

        self.assertAlmostEqual(float(ConcreteEnv().discount(None, None)), 1.0)

    def test_discount_returns_zero_when_terminal(self):
        class ConcreteEnv(EnvironmentNoAutoReset):
            @property
            def default_params(self):
                return None

            def is_terminal(self, state, params):
                return jnp.array(True)

        self.assertAlmostEqual(float(ConcreteEnv().discount(None, None)), 0.0)


if __name__ == "__main__":
    unittest.main()
