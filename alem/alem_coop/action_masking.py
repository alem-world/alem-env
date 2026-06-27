"""Action masking for Alem-Coop environments.

Computes a boolean mask over all actions for each player, indicating which
actions are valid given the current game state. Conservative: never masks
a valid action (may allow some no-ops).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:
    from jaxtyping import Array, Bool

    from .alem_state import EnvParams, EnvState, StaticEnvParams

from .constants import (
    BEACON_COST_COAL,
    BEACON_COST_IRON,
    CLOSE_BLOCKS,
    FORGE_COST_COAL,
    FORGE_COST_IRON,
    FORGE_COST_STONE,
    MONSTERS_KILLED_TO_CLEAR_LEVEL,
    SHELTER_COST_STONE,
    SHELTER_COST_WOOD,
    Action,
    BlockType,
    ItemType,
    Specialization,
)
from .util.game_logic_utils import get_max_drink, get_max_food, is_near_block


def compute_action_mask(
    state: EnvState, env_params: EnvParams, static_params: StaticEnvParams
) -> Bool[Array, "player_count num_actions"]:
    """Compute which actions are valid for each player.

    Args:
        state: EnvState
        env_params: EnvParams
        static_params: StaticEnvParams

    Returns:
        jnp.ndarray of shape (player_count, num_actions) with True = valid.
        num_actions = len(Action) + (player_count - 2) to account for
        extra GIVE target slots.
    """
    pc = static_params.player_count
    inv = state.inventory
    num_actions = len(Action) + max(0, pc - 2) + static_params.num_comm_channels

    # --- Role checks ---
    is_miner = state.player_specialization == Specialization.MINER.value
    is_warrior = state.player_specialization == Specialization.WARRIOR.value
    soft = env_params.soft_specialization
    role_miner = jnp.where(soft, True, is_miner)
    role_warrior = jnp.where(soft, True, is_warrior)

    # --- Proximity checks ---
    near_table = is_near_block(state, BlockType.CRAFTING_TABLE.value, static_params)
    near_furnace = is_near_block(state, BlockType.FURNACE.value, static_params)
    near_epic_forge = is_near_block(state, BlockType.EPIC_FORGE.value, static_params)
    near_enchant = is_near_block(
        state, BlockType.ENCHANTMENT_TABLE_FIRE.value, static_params
    ) | is_near_block(state, BlockType.ENCHANTMENT_TABLE_ICE.value, static_params)
    near_construction_site = is_near_block(state, BlockType.CONSTRUCTION_SITE.value, static_params)

    # --- Item/position checks ---
    has_gems = (inv.ruby >= 1) | (inv.sapphire >= 1)

    at_pos = state.item_map[
        state.player_level,
        state.player_position[:, 0],
        state.player_position[:, 1],
    ]
    at_ladder_down = at_pos == ItemType.LADDER_DOWN.value
    at_ladder_up = at_pos == ItemType.LADDER_UP.value
    level_cleared = state.monsters_killed[state.player_level] >= MONSTERS_KILLED_TO_CLEAR_LEVEL

    # Diamond crafting location depends on coordination config
    diamond_craft_loc = jnp.where(
        env_params.crafting_coordination_enabled, near_epic_forge, near_table
    )

    # --- Build mask column by column ---
    ones = jnp.ones(pc, dtype=jnp.bool_)

    # Start with all-False mask
    mask = jnp.zeros((pc, num_actions), dtype=jnp.bool_)

    def s(action_enum):
        return action_enum.value

    # Always valid: NOOP, movement, DO, SLEEP, REST
    for a in [
        Action.NOOP,
        Action.LEFT,
        Action.RIGHT,
        Action.UP,
        Action.DOWN,
        Action.DO,
        Action.SLEEP,
        Action.REST,
    ]:
        mask = mask.at[:, s(a)].set(ones)

    # Request actions are always valid
    for a in [
        Action.REQUEST_FOOD,
        Action.REQUEST_DRINK,
        Action.REQUEST_WOOD,
        Action.REQUEST_STONE,
        Action.REQUEST_IRON,
        Action.REQUEST_COAL,
        Action.REQUEST_DIAMOND,
        Action.REQUEST_RUBY,
        Action.REQUEST_SAPPHIRE,
    ]:
        mask = mask.at[:, s(a)].set(ones)

    # --- Ladder ---
    mask = mask.at[:, s(Action.DESCEND)].set(at_ladder_down & level_cleared)
    mask = mask.at[:, s(Action.ASCEND)].set(at_ladder_up)

    # --- Placement (inventory + role) ---
    mask = mask.at[:, s(Action.PLACE_STONE)].set(role_miner & (inv.stone >= 1))
    mask = mask.at[:, s(Action.PLACE_TABLE)].set(inv.wood >= 2)
    mask = mask.at[:, s(Action.PLACE_FURNACE)].set(inv.stone >= 1)
    mask = mask.at[:, s(Action.PLACE_PLANT)].set(inv.sapling >= 1)
    mask = mask.at[:, s(Action.PLACE_TORCH)].set(inv.torches >= 1)

    # --- Crafting (inventory + proximity + role) ---
    # Pickaxes
    mask = mask.at[:, s(Action.MAKE_WOOD_PICKAXE)].set(
        role_miner & (inv.wood >= 1) & near_table & (inv.pickaxe < 1)
    )
    mask = mask.at[:, s(Action.MAKE_STONE_PICKAXE)].set(
        role_miner & (inv.wood >= 1) & (inv.stone >= 1) & near_table & (inv.pickaxe < 2)
    )
    mask = mask.at[:, s(Action.MAKE_IRON_PICKAXE)].set(
        role_miner
        & (inv.wood >= 1)
        & (inv.stone >= 1)
        & (inv.iron >= 1)
        & (inv.coal >= 1)
        & near_table
        & near_furnace
        & (inv.pickaxe < 3)
    )
    mask = mask.at[:, s(Action.MAKE_DIAMOND_PICKAXE)].set(
        role_miner & (inv.wood >= 1) & (inv.diamond >= 3) & diamond_craft_loc & (inv.pickaxe < 4)
    )

    # Swords
    mask = mask.at[:, s(Action.MAKE_WOOD_SWORD)].set((inv.wood >= 1) & near_table & (inv.sword < 1))
    mask = mask.at[:, s(Action.MAKE_STONE_SWORD)].set(
        role_warrior & (inv.wood >= 1) & (inv.stone >= 1) & near_table & (inv.sword < 2)
    )
    mask = mask.at[:, s(Action.MAKE_IRON_SWORD)].set(
        role_warrior
        & (inv.wood >= 1)
        & (inv.stone >= 1)
        & (inv.iron >= 1)
        & (inv.coal >= 1)
        & near_table
        & near_furnace
        & (inv.sword < 3)
    )
    mask = mask.at[:, s(Action.MAKE_DIAMOND_SWORD)].set(
        role_warrior & (inv.wood >= 1) & (inv.diamond >= 2) & diamond_craft_loc & (inv.sword < 4)
    )

    # Armour
    mask = mask.at[:, s(Action.MAKE_IRON_ARMOUR)].set(
        (inv.iron >= 3) & (inv.coal >= 3) & near_table & near_furnace
    )
    mask = mask.at[:, s(Action.MAKE_DIAMOND_ARMOUR)].set((inv.diamond >= 3) & diamond_craft_loc)

    # Arrows and torches
    mask = mask.at[:, s(Action.MAKE_ARROW)].set(
        role_warrior & (inv.wood >= 1) & (inv.stone >= 1) & near_table
    )
    mask = mask.at[:, s(Action.MAKE_TORCH)].set(
        role_miner & (inv.wood >= 1) & (inv.coal >= 1) & near_table
    )

    # --- Combat ---
    mask = mask.at[:, s(Action.SHOOT_ARROW)].set((inv.bow >= 1) & (inv.arrows >= 1))
    # Note: mana cost varies by role (Fireball=2, Heal=6) so we conservatively
    # allow casting whenever a spell is known and trust the agent to manage mana.
    mask = mask.at[:, s(Action.CAST_SPELL)].set(state.learned_spells)

    # --- Potions and books ---
    mask = mask.at[:, s(Action.DRINK_POTION_RED)].set(inv.potions[:, 0] > 0)
    mask = mask.at[:, s(Action.DRINK_POTION_GREEN)].set(inv.potions[:, 1] > 0)
    mask = mask.at[:, s(Action.DRINK_POTION_BLUE)].set(inv.potions[:, 2] > 0)
    mask = mask.at[:, s(Action.DRINK_POTION_PINK)].set(inv.potions[:, 3] > 0)
    mask = mask.at[:, s(Action.DRINK_POTION_CYAN)].set(inv.potions[:, 4] > 0)
    mask = mask.at[:, s(Action.DRINK_POTION_YELLOW)].set(inv.potions[:, 5] > 0)
    mask = mask.at[:, s(Action.READ_BOOK)].set(inv.books > 0)

    # --- Enchanting ---
    has_mana = state.player_mana >= 9
    mask = mask.at[:, s(Action.ENCHANT_SWORD)].set(
        role_warrior & near_enchant & has_gems & (inv.sword > 0) & has_mana
    )
    mask = mask.at[:, s(Action.ENCHANT_ARMOUR)].set(
        near_enchant & has_gems & (inv.armour.sum(axis=1) > 0) & has_mana
    )
    mask = mask.at[:, s(Action.ENCHANT_BOW)].set(
        role_warrior & near_enchant & has_gems & (inv.bow > 0) & has_mana
    )

    # --- Level up ---
    has_xp = state.player_xp >= 1
    mask = mask.at[:, s(Action.LEVEL_UP_DEXTERITY)].set(has_xp)
    mask = mask.at[:, s(Action.LEVEL_UP_STRENGTH)].set(has_xp)
    mask = mask.at[:, s(Action.LEVEL_UP_INTELLIGENCE)].set(has_xp)

    # --- GIVE ---
    # Strict GIVE mask: valid iff selecting this slot would transfer at least one
    # requested resource in trade_materials().
    num_give_slots = pc - 1  # can give to pc-1 other players
    max_food = get_max_food(state)
    max_drink = get_max_drink(state)

    def _give_mask_for_slot(k: int) -> Bool[Array, "player_count"]:
        """For GIVE slot k: exact per-giver transfer feasibility."""
        # For giver i, target = k if k < i else k+1.
        target_indices = jnp.where(jnp.arange(pc) <= k, k + 1, k)

        target_is_requesting = (state.request_duration[target_indices] > 0) & state.player_alive[
            target_indices
        ]
        request_type = state.request_type[target_indices]

        can_give_food = (
            target_is_requesting
            & (request_type == Action.REQUEST_FOOD.value)
            & (state.player_food[target_indices] < max_food[target_indices])
            & (state.player_food > 0)
        )
        can_give_drink = (
            target_is_requesting
            & (request_type == Action.REQUEST_DRINK.value)
            & (state.player_drink[target_indices] < max_drink[target_indices])
            & (state.player_drink > 0)
        )
        can_give_wood = (
            target_is_requesting
            & (request_type == Action.REQUEST_WOOD.value)
            & (inv.wood[target_indices] < 99)
            & (inv.wood > 0)
        )
        can_give_stone = (
            target_is_requesting
            & (request_type == Action.REQUEST_STONE.value)
            & (inv.stone[target_indices] < 99)
            & (inv.stone > 0)
        )
        can_give_iron = (
            target_is_requesting
            & (request_type == Action.REQUEST_IRON.value)
            & (inv.iron[target_indices] < 99)
            & (inv.iron > 0)
        )
        can_give_coal = (
            target_is_requesting
            & (request_type == Action.REQUEST_COAL.value)
            & (inv.coal[target_indices] < 99)
            & (inv.coal > 0)
        )
        can_give_diamond = (
            target_is_requesting
            & (request_type == Action.REQUEST_DIAMOND.value)
            & (inv.diamond[target_indices] < 99)
            & (inv.diamond > 0)
        )
        can_give_ruby = (
            target_is_requesting
            & (request_type == Action.REQUEST_RUBY.value)
            & (inv.ruby[target_indices] < 99)
            & (inv.ruby > 0)
        )
        can_give_sapphire = (
            target_is_requesting
            & (request_type == Action.REQUEST_SAPPHIRE.value)
            & (inv.sapphire[target_indices] < 99)
            & (inv.sapphire > 0)
        )

        return (
            can_give_food
            | can_give_drink
            | can_give_wood
            | can_give_stone
            | can_give_iron
            | can_give_coal
            | can_give_diamond
            | can_give_ruby
            | can_give_sapphire
        )

    for k in range(num_give_slots):
        mask = mask.at[:, Action.GIVE.value + k].set(_give_mask_for_slot(k))

    # --- Construction ---
    # Handover completers at a CONSTRUCTION_IN_PROGRESS site don't need materials,
    # but they should only see the structure type that is actually pending there.
    has_shelter_materials = (inv.wood >= SHELTER_COST_WOOD) & (inv.stone >= SHELTER_COST_STONE)
    has_forge_materials = (
        (inv.stone >= FORGE_COST_STONE)
        & (inv.iron >= FORGE_COST_IRON)
        & (inv.coal >= FORGE_COST_COAL)
    )
    has_beacon_materials = (inv.iron >= BEACON_COST_IRON) & (inv.coal >= BEACON_COST_COAL)
    # Sync construction only spends materials once, so helpers should still see
    # BUILD_* if any teammate can pay for that structure type.
    team_has_shelter_materials = has_shelter_materials.any()
    team_has_forge_materials = has_forge_materials.any()
    team_has_beacon_materials = has_beacon_materials.any()
    # Construction handover completion should only expose the structure type
    # actually pending at a nearby IN_PROGRESS site.
    close_blocks = state.player_position[:, None, :] + CLOSE_BLOCKS[None, :, :]
    pending = state.pending_handovers
    nearby_pending = (
        (pending[None, None, :, 0] == 1)
        & (pending[None, None, :, 3] > state.timestep)
        & (pending[None, None, :, 4] != jnp.arange(pc)[:, None, None])
        & (pending[None, None, :, 1] == close_blocks[:, :, None, 0])
        & (pending[None, None, :, 2] == close_blocks[:, :, None, 1])
    )
    near_in_progress_shelter = (nearby_pending & (pending[None, None, :, 5] == 1)).any(axis=(1, 2))
    near_in_progress_forge = (nearby_pending & (pending[None, None, :, 5] == 2)).any(axis=(1, 2))
    near_in_progress_beacon = (nearby_pending & (pending[None, None, :, 5] == 3)).any(axis=(1, 2))

    mask = mask.at[:, s(Action.BUILD_SHELTER)].set(
        (near_construction_site & team_has_shelter_materials) | near_in_progress_shelter
    )
    mask = mask.at[:, s(Action.BUILD_FORGE)].set(
        (near_construction_site & team_has_forge_materials) | near_in_progress_forge
    )
    mask = mask.at[:, s(Action.BUILD_BEACON)].set(
        (near_construction_site & team_has_beacon_materials) | near_in_progress_beacon
    )

    # --- Communication (always valid) ---
    comm_base = len(Action) + (pc - 2)
    for i in range(static_params.num_comm_channels):
        mask = mask.at[:, comm_base + i].set(ones)

    # --- Dead / sleeping / resting: only Noop is valid ---
    # The game engine forces all actions to NOOP for these agents.
    # Reflect that here so LLM and RL agents both see an accurate action mask.
    cant_act = jnp.logical_not(state.player_alive) | state.is_sleeping | state.is_resting
    noop_only = jnp.zeros((pc, num_actions), dtype=jnp.bool_)
    noop_only = noop_only.at[:, Action.NOOP.value].set(jnp.ones(pc, dtype=jnp.bool_))
    mask = jnp.where(cant_act[:, None], noop_only, mask)

    return mask


def compute_action_mask_single_agent(
    state: EnvState, env_params: EnvParams, static_params: StaticEnvParams
) -> Bool[Array, "player_count sa_action_dim"]:
    """Action mask for single-agent: only the 42 solo-playable actions (NOOP … ENCHANT_BOW).

    REQUEST_*, BUILD_*, and GIVE are excluded — they require teammates and are
    meaningless for player_count=1.

    Args:
        state: Current environment state.
        env_params: Dynamic parameters used for action requirements.
        static_params: Static parameters defining action-space dimensions.

    Returns:
        Boolean action mask with shape ``(1, 42)``.
    """
    full = compute_action_mask(state, env_params, static_params)
    return full[:, : Action.ENCHANT_BOW.value + 1]
