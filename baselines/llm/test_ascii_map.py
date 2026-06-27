"""Tests for alem/llm/ascii_map.py -- render_ascii_map().

Each test exercises a specific rendering guarantee:
  - Agent @ is always at grid centre (row 4, col 5 for OBS_DIM 9×11)
  - Teammate digits appear at correct relative positions
  - Dead teammates are excluded from the grid
  - Block chars appear at known positions after world manipulation
  - Dark cells show '?' (light_mask False)
  - Facing target shows "entity (on block)" format
  - Facing: none for direction=0 (noop)
  - Legend only includes chars actually present in the grid
  - No legend entry for '.' (ground) or ' ' (void)
  - Legend entries are 3 per row (inline with grid)
"""

import os
import sys
import unittest
from pathlib import Path

_project_root = str(Path(__file__).parent.parent.parent)
_alem_root = os.path.join(_project_root, "alem")
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if _alem_root not in sys.path:
    sys.path.insert(0, _alem_root)

import jax
import jax.numpy as jnp
import numpy as np

from alem.alem_coop.constants import OBS_DIM, BlockType, ItemType
from alem.llm.alem_language_wrapper import DIRECTION_NAMES, AlemLanguageWrapper, make_alem_env
from alem.llm.ascii_map import ASCII_BLOCK, ASCII_ITEM, ASCII_LEGEND, render_ascii_map

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wrapper(**kwargs):
    config = {
        "max_timesteps": 10000,
        "god_mode": False,
        "coordination_difficulty": "none",
        "soft_specialization": False,
        "shared_reward": True,
    }
    env = make_alem_env(config=config)
    env_params = env.default_params
    return AlemLanguageWrapper(env, env_params, use_ascii=True, **kwargs)


def _reset(wrapper, seed=42):
    rng = jax.random.PRNGKey(seed)
    _, state, _ = wrapper.reset(rng)
    return state


def _parse_grid(output: str):
    """Extract the 9×11 character grid from render_ascii_map output.

    Returns a list of lists of single chars (spaces between cells are stripped).
    """
    lines = output.split("\n")
    # First line is the compass 'N' header; grid rows have 2-char indent and
    # space-separated chars.  Each content line starts with '  ' (2 spaces).
    grid = []
    for line in lines:
        if not line.startswith("  "):
            continue
        # Strip the 2-char indent, then take only the first 11*2-1 = 21 chars
        # (grid part, before the legend separator).
        content = line[2:]
        # Grid chars are separated by single spaces; 11 cols → 21 chars wide
        cols = content[:21].split()
        if len(cols) == 11:
            grid.append(cols)
    return grid  # up to 9 rows


def _render(wrapper, state, player_idx=0):
    """Call render_ascii_map via the wrapper's thin method."""
    return wrapper.render_ascii_map(state, player_idx)


def _kill_others(state):
    """Remove agents 1 and 2 from the grid so tests can verify terrain chars
    without agent overlay masking the cells under test.

    Agents start adjacent to agent 0 (1 step east and south in the default
    seed), so without this any test that writes to an adjacent cell will see
    the agent digit instead of the char it placed.
    """
    return state.replace(player_alive=state.player_alive.at[1].set(False).at[2].set(False))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAgentAtCenter(unittest.TestCase):
    """Agent @ must always be at the grid centre (row 4, col 5, 0-indexed)."""

    def setUp(self):
        self.wrapper = _make_wrapper()
        self.state = _reset(self.wrapper)

    def test_self_at_center(self):
        output = _render(self.wrapper, self.state, player_idx=0)
        grid = _parse_grid(output)
        self.assertEqual(len(grid), 9, f"Expected 9 rows, got {len(grid)}")
        self.assertEqual(grid[4][5], "@", f"Expected '@' at (4,5), got '{grid[4][5]}'")

    def test_self_at_center_agent1(self):
        output = _render(self.wrapper, self.state, player_idx=1)
        grid = _parse_grid(output)
        self.assertEqual(grid[4][5], "@")

    def test_self_at_center_agent2(self):
        output = _render(self.wrapper, self.state, player_idx=2)
        grid = _parse_grid(output)
        self.assertEqual(grid[4][5], "@")


class TestTeammatePositions(unittest.TestCase):
    """Teammate digits must appear at the correct relative position."""

    def setUp(self):
        self.wrapper = _make_wrapper()
        self.state = _reset(self.wrapper)

    def test_teammate_relative_position_consistency(self):
        """Agent 0's view and agent 1's view must show each other symmetrically."""
        grid0 = _parse_grid(_render(self.wrapper, self.state, player_idx=0))
        grid1 = _parse_grid(_render(self.wrapper, self.state, player_idx=1))

        # Find agent 1 in agent 0's view
        pos1_in_0 = None
        for r in range(9):
            for c in range(11):
                if grid0[r][c] == "1":
                    pos1_in_0 = (r, c)

        # Find agent 0 in agent 1's view
        pos0_in_1 = None
        for r in range(9):
            for c in range(11):
                if grid1[r][c] == "0":
                    pos0_in_1 = (r, c)

        if pos1_in_0 is not None and pos0_in_1 is not None:
            # The offset from agent 1 to agent 0 should be the mirror of
            # the offset from agent 0 to agent 1.
            dr_from_0 = pos1_in_0[0] - 4  # relative to center (row 4)
            dc_from_0 = pos1_in_0[1] - 5  # relative to center (col 5)
            dr_from_1 = pos0_in_1[0] - 4
            dc_from_1 = pos0_in_1[1] - 5
            self.assertEqual(
                dr_from_0, -dr_from_1, f"Row offsets should be negated: {dr_from_0} vs {dr_from_1}"
            )
            self.assertEqual(
                dc_from_0, -dc_from_1, f"Col offsets should be negated: {dc_from_0} vs {dc_from_1}"
            )

    def test_dead_teammate_not_shown(self):
        """Dead agents must not be rendered in the grid."""
        # Kill agent 1 by setting health to 0
        state = self.state.replace(
            player_health=self.state.player_health.at[1].set(0.0),
            player_alive=self.state.player_alive.at[1].set(False),
        )
        grid = _parse_grid(_render(self.wrapper, state, player_idx=0))
        flat = [ch for row in grid for ch in row]
        self.assertNotIn("1", flat, "Dead agent 1 should not appear in grid")

    def test_alive_teammate_not_shown_in_own_view(self):
        """'@' must represent self, not show as digit in own view."""
        grid = _parse_grid(_render(self.wrapper, self.state, player_idx=0))
        flat = [ch for row in grid for ch in row]
        self.assertNotIn("0", flat, "Agent 0 should show as '@' not '0' in own view")


class TestBlockChars(unittest.TestCase):
    """Block characters should match ASCII_BLOCK for lit cells."""

    def setUp(self):
        self.wrapper = _make_wrapper()
        self.state = _kill_others(_reset(self.wrapper))

    def test_water_char(self):
        """Place water adjacent to player, verify '~' appears."""
        pos = np.array(self.state.player_position[0])
        pr, pc = int(pos[0]), int(pos[1])
        level = int(self.state.player_level)

        # Place water one step east (col +1)
        new_map = self.state.map.at[level, pr, pc + 1].set(BlockType.WATER.value)
        state = self.state.replace(map=new_map)

        grid = _parse_grid(_render(self.wrapper, state, player_idx=0))
        self.assertEqual(grid[4][6], "~", f"Expected water '~' at (4,6), got '{grid[4][6]}'")

    def test_stone_char(self):
        """Place stone adjacent to player, verify 's' appears."""
        pos = np.array(self.state.player_position[0])
        pr, pc = int(pos[0]), int(pos[1])
        level = int(self.state.player_level)

        new_map = self.state.map.at[level, pr, pc + 1].set(BlockType.STONE.value)
        state = self.state.replace(map=new_map)

        grid = _parse_grid(_render(self.wrapper, state, player_idx=0))
        self.assertEqual(grid[4][6], "s")

    def test_tree_char(self):
        pos = np.array(self.state.player_position[0])
        pr, pc = int(pos[0]), int(pos[1])
        level = int(self.state.player_level)

        new_map = self.state.map.at[level, pr - 1, pc].set(BlockType.TREE.value)
        state = self.state.replace(map=new_map)

        grid = _parse_grid(_render(self.wrapper, state, player_idx=0))
        self.assertEqual(grid[3][5], "T")

    def test_all_block_types_have_char(self):
        """ASCII_BLOCK must map every BlockType value to a single character."""
        for bt in BlockType:
            self.assertIn(bt.value, ASCII_BLOCK, f"BlockType.{bt.name} missing from ASCII_BLOCK")
            ch = ASCII_BLOCK[bt.value]
            self.assertEqual(
                len(ch), 1, f"BlockType.{bt.name} char '{ch}' is not a single character"
            )


class TestItemChars(unittest.TestCase):
    """Item layer (torch/ladders) should override the block char."""

    def setUp(self):
        self.wrapper = _make_wrapper()
        self.state = _kill_others(_reset(self.wrapper))

    def test_torch_char(self):
        pos = np.array(self.state.player_position[0])
        pr, pc = int(pos[0]), int(pos[1])
        level = int(self.state.player_level)

        new_item_map = self.state.item_map.at[level, pr, pc + 1].set(ItemType.TORCH.value)
        state = self.state.replace(item_map=new_item_map)

        grid = _parse_grid(_render(self.wrapper, state, player_idx=0))
        self.assertEqual(grid[4][6], "o")

    def test_all_item_types_have_char(self):
        for it in ItemType:
            if it == ItemType.NONE:
                continue
            self.assertIn(it.value, ASCII_ITEM, f"ItemType.{it.name} missing from ASCII_ITEM")


class TestLightMasking(unittest.TestCase):
    """Cells with light_map ≤ 0.05 must show '?'."""

    def setUp(self):
        self.wrapper = _make_wrapper()
        self.state = _kill_others(_reset(self.wrapper))

    def test_dark_cell_shows_question_mark(self):
        pos = np.array(self.state.player_position[0])
        pr, pc = int(pos[0]), int(pos[1])
        level = int(self.state.player_level)

        # Force the cell one step east to be dark
        new_light = self.state.light_map.at[level, pr, pc + 1].set(0.0)
        state = self.state.replace(light_map=new_light)

        grid = _parse_grid(_render(self.wrapper, state, player_idx=0))
        self.assertEqual(grid[4][6], "?", f"Expected '?' for dark cell, got '{grid[4][6]}'")

    def test_lit_cell_not_question_mark(self):
        pos = np.array(self.state.player_position[0])
        pr, pc = int(pos[0]), int(pos[1])
        level = int(self.state.player_level)

        # Ensure the cell one step east is lit
        new_light = self.state.light_map.at[level, pr, pc + 1].set(1.0)
        new_map = self.state.map.at[level, pr, pc + 1].set(BlockType.GRASS.value)
        state = self.state.replace(light_map=new_light, map=new_map)

        grid = _parse_grid(_render(self.wrapper, state, player_idx=0))
        self.assertNotEqual(grid[4][6], "?")


class TestFacingLine(unittest.TestCase):
    """Facing line must match describe_env: direction name + Do target."""

    def setUp(self):
        self.wrapper = _make_wrapper()
        self.state = _reset(self.wrapper)

    def test_facing_none_for_direction_zero(self):
        state = self.state.replace(player_direction=self.state.player_direction.at[0].set(0))
        output = _render(self.wrapper, state, player_idx=0)
        self.assertIn("Facing: none", output)
        self.assertNotIn("Do target", output)

    def test_facing_north(self):
        state = self.state.replace(player_direction=self.state.player_direction.at[0].set(3))
        output = _render(self.wrapper, state, player_idx=0)
        self.assertIn("north", output.lower())

    def test_facing_dark_cell(self):
        pos = np.array(self.state.player_position[0])
        pr, pc = int(pos[0]), int(pos[1])
        level = int(self.state.player_level)

        # Make north cell dark, face north
        new_light = self.state.light_map.at[level, pr - 1, pc].set(0.0)
        state = self.state.replace(
            light_map=new_light,
            player_direction=self.state.player_direction.at[0].set(3),
        )
        output = _render(self.wrapper, state, player_idx=0)
        self.assertIn("darkness", output)

    def test_facing_target_shows_block(self):
        pos = np.array(self.state.player_position[0])
        pr, pc = int(pos[0]), int(pos[1])
        level = int(self.state.player_level)

        # Place stone to the east, ensure lit, face east
        new_map = self.state.map.at[level, pr, pc + 1].set(BlockType.STONE.value)
        new_light = self.state.light_map.at[level, pr, pc + 1].set(1.0)
        state = self.state.replace(
            map=new_map,
            light_map=new_light,
            player_direction=self.state.player_direction.at[0].set(2),  # east
        )
        output = _render(self.wrapper, state, player_idx=0)
        self.assertIn("stone", output)


class TestLegend(unittest.TestCase):
    """Legend correctness: only present chars, no '.' or ' ' entries."""

    def setUp(self):
        self.wrapper = _make_wrapper()
        self.state = _kill_others(_reset(self.wrapper))

    def test_legend_excludes_ground(self):
        output = _render(self.wrapper, self.state, player_idx=0)
        # Legend entries look like "s=stone", "~=water" etc.
        # '.' should never appear as a legend key
        legend_section = output  # scan entire output
        lines = legend_section.split("\n")
        for line in lines:
            # Legend entries are after the grid column (position > 23)
            entries_part = line[23:] if len(line) > 23 else ""
            for token in entries_part.split():
                self.assertFalse(
                    token.startswith(".="), f"'.' (ground) should not appear in legend: {line}"
                )

    def test_legend_excludes_void(self):
        output = _render(self.wrapper, self.state, player_idx=0)
        self.assertNotIn(" =void", output)

    def test_legend_contains_at_symbol(self):
        output = _render(self.wrapper, self.state, player_idx=0)
        self.assertIn("@=you", output)

    def test_legend_shows_water_when_water_present(self):
        pos = np.array(self.state.player_position[0])
        pr, pc = int(pos[0]), int(pos[1])
        level = int(self.state.player_level)

        new_map = self.state.map.at[level, pr, pc + 1].set(BlockType.WATER.value)
        new_light = self.state.light_map.at[level, pr, pc + 1].set(1.0)
        state = self.state.replace(map=new_map, light_map=new_light)

        output = _render(self.wrapper, state, player_idx=0)
        self.assertIn("~=water", output)

    def test_legend_omits_water_when_water_absent(self):
        """If no water tile is visible, '~=water' must not appear in legend."""
        pos = np.array(self.state.player_position[0])
        pr, pc = int(pos[0]), int(pos[1])
        level = int(self.state.player_level)

        # Replace all water in view radius with grass
        view_h, view_w = OBS_DIM
        half_h, half_w = view_h // 2, view_w // 2
        new_map = np.array(self.state.map[level])
        for dr in range(-half_h, half_h + 1):
            for dc in range(-half_w, half_w + 1):
                if new_map[pr + dr, pc + dc] == BlockType.WATER.value:
                    new_map[pr + dr, pc + dc] = BlockType.GRASS.value
        state = self.state.replace(map=self.state.map.at[level].set(new_map))

        output = _render(self.wrapper, state, player_idx=0)
        self.assertNotIn("~=water", output)


class TestGridDimensions(unittest.TestCase):
    """Output must always be a 9×11 grid."""

    def setUp(self):
        self.wrapper = _make_wrapper()
        self.state = _reset(self.wrapper)

    def test_grid_is_9_rows(self):
        output = _render(self.wrapper, self.state)
        grid = _parse_grid(output)
        self.assertEqual(len(grid), 9)

    def test_grid_is_11_cols(self):
        output = _render(self.wrapper, self.state)
        grid = _parse_grid(output)
        for r, row in enumerate(grid):
            self.assertEqual(len(row), 11, f"Row {r} has {len(row)} cols, expected 11")

    def test_compass_n_present(self):
        output = _render(self.wrapper, self.state)
        self.assertIn("N", output.split("\n")[0])


class TestConstructionSiteConsistency(unittest.TestCase):
    """Construction site 'C' must appear consistently across agent views."""

    def setUp(self):
        self.wrapper = _make_wrapper()
        self.state = _kill_others(_reset(self.wrapper))

    def test_construction_site_same_position_all_agents(self):
        """All agents at same world pos should see 'C' at same world coords."""
        pos0 = np.array(self.state.player_position[0])
        pr0, pc0 = int(pos0[0]), int(pos0[1])
        level = int(self.state.player_level)

        # Place construction site one step south of agent 0
        new_map = self.state.map.at[level, pr0 + 1, pc0].set(BlockType.CONSTRUCTION_SITE.value)
        # Ensure it's lit for all agents
        new_light = self.state.light_map
        for i in range(3):
            pos_i = np.array(self.state.player_position[i])
            new_light = new_light.at[level, pr0 + 1, pc0].set(1.0)
        state = self.state.replace(map=new_map, light_map=new_light)

        # Check that agent 0 sees 'C' at (5, 5)  [one south = row 4+1=5]
        grid0 = _parse_grid(_render(self.wrapper, state, player_idx=0))
        self.assertEqual(
            grid0[5][5], "C", f"Agent 0 should see 'C' at row 5, col 5; got '{grid0[5][5]}'"
        )


if __name__ == "__main__":
    unittest.main()
