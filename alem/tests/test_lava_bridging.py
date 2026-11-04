"""Regression coverage for placing stone on lava and water.

Fixtures arrange valid standing positions in a generated state. These tests cover
placement, mining, movement, and existing placement restrictions.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alem.alem_coop.alem_state import EnvParams, StaticEnvParams
from alem.alem_coop.constants import Action, BlockType, ItemType, Specialization
from alem.alem_coop.game_logic import alem_step, place_block
from alem.alem_coop.world_gen.world_gen import generate_world
from alem.llm.alem_language_wrapper import get_instruction_prompt
from alem.llm.alem_language_wrapper_single import get_instruction_prompt_single


def test_language_instructions_explain_lava_bridging():
    for prompt in (get_instruction_prompt(), get_instruction_prompt_single()):
        assert "stone can also be placed into water or lava" in prompt
        assert "Mine the placed stone with a pickaxe to leave a walkable path" in prompt


@pytest.fixture(scope="module")
def clean_world():
    # A zero spawn/despawn radius keeps unrelated combat out of these fixtures;
    # movement, placement, mining, inventories, and coordination remain enabled.
    params = EnvParams(coordination_enabled=True, mob_despawn_distance=0)
    static = StaticEnvParams()
    state = jax.jit(lambda key: generate_world(key, params, static))(jax.random.PRNGKey(9999))
    state = state.replace(
        map=jnp.full_like(state.map, BlockType.PATH.value),
        item_map=jnp.zeros_like(state.item_map),
        mob_map=jnp.zeros_like(state.mob_map),
        coordination_map=jnp.zeros_like(state.coordination_map),
        soft_coordination_mask=jnp.zeros_like(state.soft_coordination_mask),
        player_level=jnp.int32(0),
        player_position=jnp.array([[10, 9], [20, 30], [30, 30]], dtype=jnp.int32),
        player_direction=jnp.full(3, Action.RIGHT.value, dtype=jnp.int32),
        player_specialization=jnp.array(
            [
                Specialization.MINER.value,
                Specialization.FORAGER.value,
                Specialization.WARRIOR.value,
            ],
            dtype=jnp.int32,
        ),
        inventory=state.inventory.replace(
            stone=jnp.array([1, 0, 0], dtype=jnp.int32),
            wood=jnp.array([2, 0, 0], dtype=jnp.int32),
            pickaxe=jnp.array([3, 0, 0], dtype=jnp.int32),
            diamond=jnp.zeros(3, dtype=jnp.int32),
        ),
        melee_mobs=state.melee_mobs.replace(mask=jnp.zeros_like(state.melee_mobs.mask)),
        ranged_mobs=state.ranged_mobs.replace(mask=jnp.zeros_like(state.ranged_mobs.mask)),
        passive_mobs=state.passive_mobs.replace(mask=jnp.zeros_like(state.passive_mobs.mask)),
    )
    return state, params, static


def _actions(agent, action):
    return jnp.full(3, Action.NOOP.value, dtype=jnp.int32).at[agent].set(action.value)


def test_stone_bridges_lava_and_water_through_complete_steps(clean_world):
    base, params, static = clean_world
    step = jax.jit(
        lambda state, actions: alem_step(jax.random.PRNGKey(42), state, actions, params, static)[0]
    )
    target = (10, 10)

    for terrain in (BlockType.LAVA, BlockType.WATER):
        # A full column of liquid separates the starting tile from the far side.
        state = base.replace(
            player_level=jnp.int32(6),
            map=base.map.at[6, :, target[1]].set(terrain.value),
        )
        start = np.asarray(state.player_position[0])
        state = step(state, _actions(0, Action.RIGHT))
        np.testing.assert_array_equal(state.player_position[0], start)

        state = step(state, _actions(0, Action.PLACE_STONE))
        assert int(state.map[6, *target]) == BlockType.STONE.value, terrain
        assert int(state.inventory.stone[0]) == 0

        state = step(state, _actions(0, Action.DO))
        assert int(state.map[6, *target]) == BlockType.PATH.value
        assert int(state.inventory.stone[0]) == 1

        state = step(state, _actions(0, Action.RIGHT))
        np.testing.assert_array_equal(state.player_position[0], target)
        assert bool(state.player_alive[0])


def test_stone_placement_retains_existing_restrictions(clean_world):
    base, params, static = clean_world
    place = jax.jit(
        lambda state, actions: place_block(jax.random.PRNGKey(42), state, actions, params, static)
    )
    target = (10, 10)

    for terrain in (BlockType.LAVA, BlockType.WATER):
        state = base.replace(map=base.map.at[0, *target].set(terrain.value))
        blocked = [
            (
                "no stone",
                state.replace(
                    inventory=state.inventory.replace(stone=jnp.zeros(3, dtype=jnp.int32))
                ),
            ),
            (
                "not a miner",
                state.replace(
                    player_specialization=state.player_specialization.at[0].set(
                        Specialization.FORAGER.value
                    )
                ),
            ),
            ("solid block", state.replace(map=state.map.at[0, *target].set(BlockType.STONE.value))),
            ("mob", state.replace(mob_map=state.mob_map.at[0, *target].set(True))),
            (
                "teammate",
                state.replace(player_position=state.player_position.at[1].set(jnp.array(target))),
            ),
            (
                "out of bounds",
                state.replace(
                    player_position=state.player_position.at[0].set(jnp.array([0, 10])),
                    player_direction=state.player_direction.at[0].set(Action.UP.value),
                ),
            ),
        ]
        blocked.extend(
            (item.name, state.replace(item_map=state.item_map.at[0, *target].set(item.value)))
            for item in ItemType
            if item != ItemType.NONE
        )
        for reason, fixture in blocked:
            result = place(fixture, _actions(0, Action.PLACE_STONE))
            np.testing.assert_array_equal(result.map, fixture.map, err_msg=reason)
            np.testing.assert_array_equal(result.item_map, fixture.item_map, err_msg=reason)
            np.testing.assert_array_equal(
                result.inventory.stone, fixture.inventory.stone, err_msg=reason
            )

        # Allowing bridges must not allow building tables/furnaces directly in
        # either hazardous terrain type.
        for action in (Action.PLACE_TABLE, Action.PLACE_FURNACE):
            result = place(state, _actions(0, action))
            np.testing.assert_array_equal(result.map, state.map)
            np.testing.assert_array_equal(result.inventory.stone, state.inventory.stone)
            np.testing.assert_array_equal(result.inventory.wood, state.inventory.wood)
