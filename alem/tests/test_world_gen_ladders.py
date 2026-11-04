"""Regression tests for ladder placement in world generation.

Ladders are the only way between levels: ``change_floor`` permits DESCEND solely from a
``LADDER_DOWN`` tile and ASCEND solely from a ``LADDER_UP`` tile. Nothing else in the world can
substitute for one, so a level that generates without its full row of down-ladders is sealed —
every level below it is unreachable for the whole episode, and the team's achievable score is
capped by the generator rather than by play.

``get_ladder_positions`` samples one anchor and lays a level's ladders out in a row from it.
Down-ladders and up-ladders are drawn independently and the up-ladders are written to the item
map second, so the two draws can land on the same row and the second write deletes the first.
These tests pin the three things generation has to leave intact:

  - levels that are meant to have a way down have a full row of down-ladders (the seal),
  - levels that also have a way up have a full row of those too, on different tiles,
  - the item map only ever holds valid ``ItemType`` ids.

Level order comes from the splice in ``generate_world``: 0 overworld, 1/3/4 dungeons, 2 gnomish
mines, 5 troll mines, 6 fire, 7 ice, 8 boss. The overworld has no way up and the boss level has
neither ladder kind, which is why those two are called out separately below.
"""

import unittest

import jax
import jax.numpy as jnp

from alem.alem_coop.alem_state import EnvParams, StaticEnvParams
from alem.alem_coop.constants import BlockType, ItemType
from alem.alem_coop.util.game_logic_utils import get_ladder_positions
from alem.alem_coop.world_gen.world_gen import generate_world, resolve_ladder_collision
from alem.alem_coop.world_gen.world_gen_configs import OVERWORLD_CONFIG

# The bug seals the overworld on roughly 7% of seeds and a deeper level on roughly another 3%,
# so a few dozen seeds is enough to make a regression essentially certain to show up while
# keeping the suite quick.
NUM_SEEDS = 64

# Levels that must offer a way down, and the subset that must also offer a way back up.
# Level 8 (boss) is the bottom of the world and has neither.
LEVELS_WITH_DOWN_LADDER = tuple(range(8))
LEVELS_WITH_UP_LADDER = tuple(range(1, 8))
LEVEL_BOSS = 8


def _generate_worlds(num_seeds: int):
    """Generate ``num_seeds`` worlds and return their stacked item maps and ladder coordinates."""
    static_params = StaticEnvParams()
    params = EnvParams()

    def _one(seed):
        state = generate_world(jax.random.PRNGKey(seed), params, static_params)
        return state.item_map, state.down_ladders, state.up_ladders

    item_map, down_ladders, up_ladders = jax.jit(jax.vmap(_one))(jnp.arange(num_seeds))
    return static_params, item_map, down_ladders, up_ladders


class TestLadderGeneration(unittest.TestCase):
    """Whole-world checks: every generated level keeps the ladders it is supposed to have."""

    @classmethod
    def setUpClass(cls):
        cls.static_params, cls.item_map, cls.down_ladders, cls.up_ladders = _generate_worlds(
            NUM_SEEDS
        )
        # (num_seeds, num_levels) counts of each ladder kind on the item map.
        cls.num_down = (cls.item_map == ItemType.LADDER_DOWN.value).sum(axis=(2, 3))
        cls.num_up = (cls.item_map == ItemType.LADDER_UP.value).sum(axis=(2, 3))

    def test_every_descendable_level_has_a_full_row_of_down_ladders(self):
        """A missing down-ladder seals the level and every level under it."""
        expected = self.static_params.player_count
        for level in LEVELS_WITH_DOWN_LADDER:
            counts = self.num_down[:, level]
            bad = jnp.nonzero(counts != expected)[0]
            self.assertEqual(
                bad.size,
                0,
                f"level {level} is missing down-ladders on seeds {bad.tolist()} "
                f"(counts {counts[bad].tolist()}, expected {expected})",
            )

    def test_levels_with_an_up_ladder_have_a_full_row(self):
        expected = self.static_params.player_count
        for level in LEVELS_WITH_UP_LADDER:
            counts = self.num_up[:, level]
            bad = jnp.nonzero(counts != expected)[0]
            self.assertEqual(
                bad.size,
                0,
                f"level {level} is missing up-ladders on seeds {bad.tolist()} "
                f"(counts {counts[bad].tolist()}, expected {expected})",
            )

    def test_overworld_has_no_up_ladder(self):
        """Level 0 is the top of the world; nothing should be written above it."""
        self.assertEqual(int(self.num_up[:, 0].sum()), 0)

    def test_boss_level_has_neither_ladder_kind(self):
        """Level 8 is the bottom of the world and is configured with both flags off."""
        self.assertEqual(int(self.num_down[:, LEVEL_BOSS].sum()), 0)
        self.assertEqual(int(self.num_up[:, LEVEL_BOSS].sum()), 0)

    def test_up_and_down_ladders_never_share_a_tile(self):
        """The overlap is what deletes a level's way down, so check the coordinates directly."""
        # (num_seeds, num_levels, player_count, player_count): every down/up ladder pair.
        same = (self.down_ladders[:, :, :, None, :] == self.up_ladders[:, :, None, :, :]).all(-1)
        # Only levels carrying both kinds can overlap; the rest hold placeholder coordinates.
        for level in LEVELS_WITH_UP_LADDER:
            bad = jnp.nonzero(same[:, level].any(axis=(1, 2)))[0]
            self.assertEqual(
                bad.size,
                0,
                f"level {level} places an up-ladder on a down-ladder for seeds {bad.tolist()}",
            )

    def test_item_map_only_holds_valid_item_types(self):
        """A level with a ladder flag off must leave the item map alone, not stamp a block id."""
        invalid = self.item_map >= len(ItemType)
        bad = jnp.nonzero(invalid.any(axis=(1, 2, 3)))[0]
        self.assertEqual(
            bad.size,
            0,
            f"item map holds ids outside ItemType on seeds {bad.tolist()} "
            f"(values {jnp.unique(self.item_map[invalid]).tolist()})",
        )


class TestResolveLadderCollision(unittest.TestCase):
    """Unit tests for the collision resolver, on hand-built inputs rather than sampled worlds."""

    def setUp(self):
        self.static_params = StaticEnvParams()
        self.rng = jax.random.PRNGKey(0)
        # A map that is entirely valid ladder terrain, so any row of tiles is a legal draw.
        self.map = jnp.full(self.static_params.map_size, BlockType.PATH.value, dtype=jnp.int32)
        self.ladders_down = get_ladder_positions(
            self.rng, self.static_params, OVERWORLD_CONFIG, self.map
        )

    def test_colliding_draw_is_moved_off_the_down_ladders(self):
        ladders_up = self.ladders_down
        resolved = resolve_ladder_collision(
            self.rng,
            self.static_params,
            OVERWORLD_CONFIG,
            self.map,
            self.ladders_down,
            ladders_up,
        )
        overlap = (self.ladders_down[None, :, :] == resolved[:, None, :]).all(-1).any()
        self.assertFalse(bool(overlap), f"resolver left up-ladders on {resolved.tolist()}")

    def test_non_colliding_draw_is_left_untouched(self):
        """Levels that drew cleanly must be bit-identical to what the unfixed generator made."""
        ladders_up = self.ladders_down + jnp.array([1, 0])
        resolved = resolve_ladder_collision(
            self.rng,
            self.static_params,
            OVERWORLD_CONFIG,
            self.map,
            self.ladders_down,
            ladders_up,
        )
        self.assertTrue(bool((resolved == ladders_up).all()))

    def test_disabled_leaves_the_draw_untouched_even_on_a_collision(self):
        """Levels missing one ladder kind cannot suffer the overwrite, so they are not redrawn."""
        ladders_up = self.ladders_down
        resolved = resolve_ladder_collision(
            self.rng,
            self.static_params,
            OVERWORLD_CONFIG,
            self.map,
            self.ladders_down,
            ladders_up,
            enabled=False,
        )
        self.assertTrue(bool((resolved == ladders_up).all()))


class TestGetLadderPositionsExclusion(unittest.TestCase):
    """The ``exclude`` argument added for the resolver, tested on its own."""

    def setUp(self):
        self.static_params = StaticEnvParams()
        self.rng = jax.random.PRNGKey(1)
        self.map = jnp.full(self.static_params.map_size, BlockType.PATH.value, dtype=jnp.int32)

    def test_excluded_tiles_are_avoided(self):
        unrestricted = get_ladder_positions(
            self.rng, self.static_params, OVERWORLD_CONFIG, self.map
        )
        exclude = (
            jnp.zeros(self.static_params.map_size, dtype=bool)
            .at[unrestricted[:, 0], unrestricted[:, 1]]
            .set(True)
        )
        restricted = get_ladder_positions(
            self.rng, self.static_params, OVERWORLD_CONFIG, self.map, exclude=exclude
        )
        overlap = (unrestricted[None, :, :] == restricted[:, None, :]).all(-1).any()
        self.assertFalse(bool(overlap))

    def test_exclusion_is_ignored_when_it_rules_out_every_tile(self):
        """Better a colliding ladder than no ladder: an impossible exclusion falls back."""
        exclude = jnp.ones(self.static_params.map_size, dtype=bool)
        positions = get_ladder_positions(
            self.rng, self.static_params, OVERWORLD_CONFIG, self.map, exclude=exclude
        )
        unrestricted = get_ladder_positions(
            self.rng, self.static_params, OVERWORLD_CONFIG, self.map
        )
        self.assertTrue(bool((positions == unrestricted).all()))


if __name__ == "__main__":
    unittest.main()
