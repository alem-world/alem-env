from __future__ import annotations

from typing import TYPE_CHECKING

import chex
import jax
import jax.numpy as jnp

from .alem_state import (
    EnvParams,
    EnvState,
    InventorySlice,
    Mobs,
    MobScanState,
    StaticEnvParams,
)
from .constants import (
    ACHIEVEMENT_REWARD_MAP,
    BEACON_COST_COAL,
    BEACON_COST_IRON,
    BEACON_LIGHT_MAP,
    BOSS_FIGHT_SPAWN_TURNS,
    CAN_PLACE_ITEM_MAPPING,
    COLLISION_LAND_CREATURE,
    DIRECTIONS,
    FLOOR_MOB_MAPPING,
    FLOOR_MOB_SPAWN_CHANCE,
    FORGE_COST_COAL,
    FORGE_COST_IRON,
    FORGE_COST_STONE,
    LEVEL_ACHIEVEMENT_MAP,
    MOB_TYPE_COLLISION_MAPPING,
    MOB_TYPE_DAMAGE_MAPPING,
    MOB_TYPE_HEALTH_MAPPING,
    MONSTERS_KILLED_TO_CLEAR_LEVEL,
    RANGED_MOB_TYPE_TO_PROJECTILE_TYPE_MAPPING,
    REQUEST_MAX_DURATION,
    SHELTER_COST_STONE,
    SHELTER_COST_WOOD,
    TORCH_LIGHT_MAP,
    Achievement,
    Action,
    BlockType,
    DeathCause,
    ItemType,
    MobType,
    ProjectileType,
    Specialization,
)
from .util.game_logic_utils import (
    attack_mob,
    calculate_light_level,
    clip_inventory_and_intrinsics,
    get_damage,
    get_damage_between_players,
    get_damage_done_to_player,
    get_max_drink,
    get_max_energy,
    get_max_food,
    get_max_health,
    get_nearest_block_pos,
    get_player_damage_vector,
    get_player_defense_vector,
    has_beaten_boss,
    in_bounds,
    is_boss_spawn_wave,
    is_boss_vulnerable,
    is_fighting_boss,
    is_in_mob,
    is_in_other_player,
    is_in_solid_block,
    is_near_block,
    is_position_in_bounds_not_in_mob_not_colliding,
    is_position_not_colliding_other_player,
    spawn_projectile,
)
from .util.maths_utils import get_all_players_distance_map

if TYPE_CHECKING:
    from jaxtyping import Array, Float, Int


def soft_gate_specialization(rng, is_specialist, params):
    """Apply soft gating to a specialization boolean.

    When soft_specialization=False: Returns is_specialist unchanged (hard gate)
    When soft_specialization=True: Returns probabilistic boolean where:
        - Specialists succeed with probability specialist_efficiency (1.0)
        - Non-specialists succeed with probability non_specialist_efficiency (0.2)

    Args:
        rng: JAX random key used to sample non-specialist success.
        is_specialist: Boolean mask indicating players with the required role.
        params: Gameplay parameters containing specialization efficiencies.

    Returns:
        Gated success mask and the advanced random key.
    """
    if not params.soft_specialization:
        return is_specialist, rng

    rng, _rng = jax.random.split(rng)
    roll = jax.random.uniform(_rng, shape=is_specialist.shape)

    threshold = jnp.where(
        is_specialist,
        params.specialist_efficiency,
        params.non_specialist_efficiency,
    )

    return roll < threshold, rng


# Helper functions for cleaner code


def set_achievement(achievements, achievement_type, condition):
    """Set an achievement for agents matching a condition.

    Args:
        achievements: Existing per-agent achievement matrix.
        achievement_type: Achievement enum member to update.
        condition: Per-agent boolean mask for earning the achievement.

    Returns:
        Achievement matrix with the condition ORed into the target column.
    """
    return achievements.at[:, achievement_type.value].set(
        achievements[:, achievement_type.value] | condition
    )


def get_block_at(state, block_position):
    """Get the block type at each agent's target position.

    Args:
        state: Current environment state.
        block_position: Per-agent row/column target coordinates.

    Returns:
        Integer block identifier for each agent.
    """
    return state.map[state.player_level, block_position[:, 0], block_position[:, 1]]


def mine_resource(
    new_map,
    block_position,
    equal_block_placement,
    doing_action,
    block_type,
    replacement_type,
    can_mine,
):
    """Handle mining a resource block type.

    Args:
        new_map: Mutable level-map value being assembled for the next state.
        block_position: Per-agent mining target coordinates.
        equal_block_placement: Matrix identifying agents sharing targets.
        doing_action: Mask of agents attempting the mining action.
        block_type: Resource block identifier required at the target.
        replacement_type: Block identifier left after successful mining.
        can_mine: Mask of agents meeting tool and specialization requirements.

    Returns:
        new_map: Updated map with mined blocks replaced
        is_mining: Boolean array of which agents are mining this resource
    """
    current_blocks = new_map[block_position[:, 0], block_position[:, 1]]
    is_block = current_blocks == block_type
    is_mining = is_block & can_mine & doing_action

    is_any_player_mining = (equal_block_placement & is_mining[:, None]).any(axis=0)
    mined_block = jnp.where(is_any_player_mining, replacement_type, current_blocks)
    new_map = new_map.at[block_position[:, 0], block_position[:, 1]].set(mined_block)

    return new_map, is_mining


def check_sync_coordination(
    rng, state, block_positions, is_acting, equal_block_placement, params, static_params
):
    """Check if synchronous coordination requirements are met.

    For blocks requiring multiple agents (positive coordination_map value):
    - Hard coordination: requires N agents or action fails
    - Soft coordination: solo succeeds with P(1 - α), coordinated always succeeds.
      This interpolates between trivial (α=0) and hard-sync (α=1), scaling the
      resource slack without changing opportunity count.

    Args:
        rng: JAX random key used for soft solo success rolls.
        state: Current environment state and coordination maps.
        block_positions: Per-agent target coordinates.
        is_acting: Mask of agents attempting the action.
        equal_block_placement: Matrix identifying agents sharing targets.
        params: Gameplay parameters controlling soft coordination.
        static_params: Static parameters containing player count.

    Returns:
        succeeds: Boolean array of which agents' actions succeed
        is_sync_success: Boolean array of successful sync coordination
        coord_multiplier: Reward multiplier based on agents coordinating
        is_soft_sync: Boolean array of soft sync coordination blocks
        agents_at_same_pos: Count of acting agents at each agent's target position
    """
    if not params.coordination_enabled:
        zeros = jnp.zeros_like(is_acting, dtype=jnp.bool_)
        return (
            is_acting,
            zeros,
            jnp.ones(static_params.player_count, dtype=jnp.float32),
            zeros,
            jnp.zeros(static_params.player_count, dtype=jnp.int32),
        )

    # Get coordination value at each agent's target position (positive = sync requirement)
    coord_req = state.coordination_map[
        state.player_level, block_positions[:, 0], block_positions[:, 1]
    ]
    is_sync = coord_req > 0

    # Check if this is soft coordination
    is_soft = state.soft_coordination_mask[
        state.player_level, block_positions[:, 0], block_positions[:, 1]
    ]

    # Count agents targeting each position (including self)
    agents_at_same_pos = (equal_block_placement & is_acting[:, None]).sum(axis=0)

    # Check if requirement met for each agent's target
    coord_met = agents_at_same_pos >= coord_req

    # Hard coordination: must meet requirement or action fails
    hard_succeeds = jnp.where(is_sync, coord_met, True)

    # Soft coordination: coordinated always succeeds; solo fails with P(α).
    # At α=0 soft blocks are trivial; at α=1 they behave like hard blocks.
    # Scales resource slack — same opportunities, less yield without coordination.
    soft_roll = jax.random.uniform(rng, shape=(static_params.player_count,))
    soft_solo_succeeds = soft_roll >= params.soft_solo_fail_prob  # P(succeed) = 1 - α
    soft_succeeds = jnp.where(coord_met, True, soft_solo_succeeds)

    # Use soft or hard based on mask
    sync_succeeds = jnp.where(is_soft, soft_succeeds, hard_succeeds)
    succeeds = is_acting & sync_succeeds

    # Track successful sync coordinations (action succeeded on a sync block)
    is_sync_success = succeeds & is_sync

    # Per-agent soft/hard sync flag (for split logging)
    is_soft_sync = is_sync_success & is_soft

    # Pairwise complementarity coordination multiplier:
    # multiplier = k(k+1)/2
    # k=1→1x, k=2→3x, k=3→6x — superlinear, no free parameters.
    #
    # Hard sync: multiplier always applies when succeeding (gated by coord_met).
    # Soft sync: action succeeds regardless, but multiplier only applies when
    # agents_at_same_pos >= coord_req (threshold coordination game).
    is_on_sync = is_sync & succeeds
    bonus_threshold_met = agents_at_same_pos >= coord_req
    k = agents_at_same_pos.astype(jnp.float32)
    coord_multiplier = jnp.where(
        is_on_sync & (~is_soft | bonus_threshold_met),
        k * (k + 1.0) / 2.0,
        jnp.ones(static_params.player_count, dtype=jnp.float32),
    )

    return succeeds, is_sync_success, coord_multiplier, is_soft_sync, agents_at_same_pos


def interplayer_interaction(state, block_position, is_doing_action, env_params, static_params):
    """Resolve revive or friendly-fire interactions at target positions.

    Args:
        state: Current environment state.
        block_position: Per-player interaction target coordinates.
        is_doing_action: Mask of players performing the interaction action.
        env_params: Gameplay parameters including friendly-fire behavior.
        static_params: Static parameters including player count.

    Returns:
        State with player health and achievements updated.
    """
    # If other player is down revive them, otherwise damage (if friendly fire is enabled)
    in_other_player = (
        (jnp.expand_dims(state.player_position, axis=1) == jnp.expand_dims(block_position, axis=0))
        .all(axis=2)
        .T
    )
    player_interacting_with = jnp.argmax(in_other_player, axis=-1)

    is_interacting_with_other_player = jnp.logical_and(
        in_other_player.any(axis=-1),
        is_doing_action,
    )
    is_player_being_interacted_with = jnp.any(
        jnp.logical_and(
            jnp.arange(static_params.player_count)[:, None] == player_interacting_with,
            is_interacting_with_other_player[None, :],
        ),
        axis=-1,
    )
    is_player_being_revived = jnp.logical_and(
        is_player_being_interacted_with,
        jnp.logical_not(state.player_alive),
    )

    # Per-attacker damage attribution: is_interacting_with_other_player is indexed
    # by the attacker; player_interacting_with is the victim index for that attacker.
    per_attacker_damage = (
        is_interacting_with_other_player
        * get_damage_between_players(state, player_interacting_with)
        * env_params.friendly_fire
    )
    # Per-victim damage (scatter attacker damage onto victim index).
    damage_taken = (
        jnp.zeros(static_params.player_count).at[player_interacting_with].add(per_attacker_damage)
    )

    new_player_health = jnp.where(
        is_player_being_revived,
        1.0,
        state.player_health - damage_taken,
    )
    state = state.replace(
        player_health=new_player_health,
        revives=state.revives + is_player_being_revived.sum(),
        ff_damage_dealt=state.ff_damage_dealt + per_attacker_damage,
    )
    return state


def update_plants_with_eat(state, plant_position, is_eating_plant):
    """Reset the age of growing plants consumed by players.

    Args:
        state: Current environment state containing growing plants.
        plant_position: Per-player coordinates of consumed plants.
        is_eating_plant: Mask of players successfully eating a plant.

    Returns:
        Updated growing-plant age array.
    """
    # is_plant shape: (num_players, num_plants)
    is_plant = jax.vmap(jnp.equal, in_axes=(0, None))(
        plant_position, state.growing_plants_positions
    ).all(axis=-1)

    # Only care about agents actually eating a plant they are at
    should_update = is_eating_plant[:, None] & is_plant
    # any_eaten shape: (num_plants,)
    any_eaten = should_update.any(axis=0)

    return jnp.where(any_eaten, 0, state.growing_plants_age)


def add_items_from_chest(rng, state, inventory, is_opening_chest, env_params):
    """Sample chest loot and add role-gated items to player inventories.

    Args:
        rng: JAX random key used to sample loot.
        state: Current state used for player specializations.
        inventory: Inventory structure to update.
        is_opening_chest: Mask of players that opened a chest.
        env_params: Gameplay parameters controlling specialization gates.

    Returns:
        Inventory with sampled chest rewards applied.
    """
    is_miner = state.player_specialization == Specialization.MINER.value
    is_warrior = state.player_specialization == Specialization.WARRIOR.value
    # Apply soft gating to specializations
    is_miner, rng = soft_gate_specialization(rng, is_miner, env_params)
    is_warrior, rng = soft_gate_specialization(rng, is_warrior, env_params)

    # Wood (60%)
    rng, _rng = jax.random.split(rng)
    is_looting_wood = jax.random.uniform(_rng) < 0.6 * is_opening_chest * is_miner
    rng, _rng = jax.random.split(rng)
    wood_loot_amount = jax.random.randint(_rng, shape=(), minval=1, maxval=6) * is_looting_wood

    # Torch (60%)
    rng, _rng = jax.random.split(rng)
    collect_prob = 0.6 * is_miner
    is_looting_torch = jax.random.uniform(_rng) < collect_prob * is_opening_chest
    rng, _rng = jax.random.split(rng)
    torch_loot_amount = jax.random.randint(_rng, shape=(), minval=4, maxval=8) * is_looting_torch

    # Ores (60%)
    rng, _rng = jax.random.split(rng)
    is_looting_ore = jax.random.uniform(_rng) < collect_prob * is_opening_chest
    rng, _rng = jax.random.split(rng)
    ore_loot_id = jax.random.choice(
        _rng,
        jnp.arange(5, dtype=jnp.int32),
        shape=(),
        p=jnp.array([0.3, 0.3, 0.15, 0.125, 0.125]),
    )
    rng, _rng = jax.random.split(rng)

    # Use the same rng as events are mutually exclusive
    coal_loot_amount = (
        jax.random.randint(_rng, shape=(), minval=1, maxval=4) * (ore_loot_id == 0) * is_looting_ore
    )
    iron_loot_amount = (
        jax.random.randint(_rng, shape=(), minval=1, maxval=3) * (ore_loot_id == 1) * is_looting_ore
    )
    diamond_loot_amount = (
        jax.random.randint(_rng, shape=(), minval=1, maxval=2) * (ore_loot_id == 2) * is_looting_ore
    )
    sapphire_loot_amount = (
        jax.random.randint(_rng, shape=(), minval=1, maxval=2) * (ore_loot_id == 3) * is_looting_ore
    )
    ruby_loot_amount = (
        jax.random.randint(_rng, shape=(), minval=1, maxval=2) * (ore_loot_id == 4) * is_looting_ore
    )

    # Potion (50%)
    rng, _rng = jax.random.split(rng)
    is_looting_potion = jax.random.uniform(_rng) < 0.5 * is_opening_chest
    rng, _rng = jax.random.split(rng)
    potion_loot_index = jax.random.randint(_rng, shape=(), minval=0, maxval=6)
    rng, _rng = jax.random.split(rng)
    potion_loot_amount = jax.random.randint(_rng, shape=(), minval=1, maxval=3)

    # Arrows (50%)
    rng, _rng = jax.random.split(rng)
    is_looting_arrows = jax.random.uniform(_rng) < 0.5 * is_opening_chest * is_warrior
    rng, _rng = jax.random.split(rng)
    arrows_loot_amount = jax.random.randint(_rng, shape=(), minval=4, maxval=9) * is_looting_arrows

    # Tools (20%)
    rng, _rng = jax.random.split(rng)
    is_looting_tool = jax.random.uniform(_rng) < 0.2
    rng, _rng = jax.random.split(rng)
    tool_id = jax.random.randint(_rng, shape=(), minval=0, maxval=2)

    is_looting_pickaxe = jnp.logical_and(
        jnp.logical_and(is_miner, jnp.logical_and(is_looting_tool, tool_id == 0)), is_opening_chest
    )
    rng, _rng = jax.random.split(rng)
    pickaxe_loot_level = (
        jax.random.choice(
            _rng,
            (jnp.arange(4) + 1).astype(int),
            shape=(),
            p=jnp.array([0.4, 0.3, 0.2, 0.1]),
        )
        * is_looting_pickaxe
    )
    pickaxe_loot_level = jnp.maximum(pickaxe_loot_level, inventory.pickaxe)
    new_pickaxe_level = (
        is_looting_pickaxe * pickaxe_loot_level + (1 - is_looting_pickaxe) * inventory.pickaxe
    )

    # Special chests
    is_looting_bow = jnp.logical_and(
        jnp.logical_and(
            is_opening_chest,
            is_warrior,
        ),
        jnp.logical_and(
            state.player_level == 1,
            jnp.logical_not(state.chests_opened[state.player_level]),
        ),
    )
    new_bow_level = is_looting_bow * 1 + (1 - is_looting_bow) * inventory.bow

    can_loot_book = jnp.logical_and(
        jnp.logical_not(state.chests_opened[state.player_level]),
        jnp.logical_or(state.player_level == 3, state.player_level == 4),
    )
    is_looting_book = jnp.logical_and(can_loot_book, is_opening_chest)

    # Update inventory
    return inventory.replace(
        wood=inventory.wood + wood_loot_amount * is_miner,
        torches=inventory.torches + torch_loot_amount,
        coal=inventory.coal + coal_loot_amount * is_miner,
        iron=inventory.iron + iron_loot_amount * is_miner,
        diamond=inventory.diamond + diamond_loot_amount * is_miner,
        sapphire=inventory.sapphire + sapphire_loot_amount * is_miner,
        ruby=inventory.ruby + ruby_loot_amount * is_miner,
        arrows=inventory.arrows + arrows_loot_amount,
        pickaxe=new_pickaxe_level,
        potions=inventory.potions.at[:, potion_loot_index].set(
            inventory.potions[:, potion_loot_index]
            + potion_loot_amount * is_looting_potion * is_opening_chest
        ),
        bow=new_bow_level,
        books=inventory.books + 1 * is_looting_book,
    )


def do_action(
    rng: chex.PRNGKey,
    state: EnvState,
    action: Int[Array, "n_agents"],
    env_params: EnvParams,
    static_params: StaticEnvParams,
) -> EnvState:
    """Resolve contextual DO actions such as mining, eating, and opening.

    Args:
        rng: JAX random key used by stochastic action outcomes.
        state: Current environment state.
        action: Per-player action identifiers.
        env_params: Gameplay and reward parameters.
        static_params: Static map and player parameters.

    Returns:
        State after all contextual interactions are resolved.
    """
    is_forager = state.player_specialization == Specialization.FORAGER.value
    is_forager, rng = soft_gate_specialization(rng, is_forager, env_params)

    block_position = state.player_position + DIRECTIONS[state.player_direction]
    equal_block_placement = (block_position[:, None] == block_position[None, :]).all(axis=2)
    doing_action = in_bounds(block_position, static_params) & (action == Action.DO.value)

    # Exclude construction blocks from mining coordination gates.
    current_blocks = get_block_at(state, block_position)
    is_at_construction = (current_blocks == BlockType.CONSTRUCTION_SITE.value) | (
        current_blocks == BlockType.CONSTRUCTION_IN_PROGRESS.value
    )
    doing_action_non_construction = doing_action & ~is_at_construction

    # Track who attempted mining before coordination gates
    attempted_mining = doing_action_non_construction

    # Apply coordination gates only to non-construction blocks
    rng, _rng = jax.random.split(rng)
    (
        doing_action_non_construction,
        is_sync_success,
        coord_multiplier,
        is_soft_sync,
        agents_at_same_pos,
    ) = check_sync_coordination(
        _rng,
        state,
        block_position,
        doing_action_non_construction,
        equal_block_placement,
        env_params,
        static_params,
    )
    doing_action_non_construction, state = process_handover(
        state, block_position, doing_action_non_construction, env_params, static_params
    )

    # Recombine: construction block positions pass through ungated
    doing_action = doing_action_non_construction | (doing_action & is_at_construction)

    # Coordination achievements
    # Soft achievements require actual coordination (2+ agents), not just solo on a soft block.
    # Hard achievements already require N agents (action fails otherwise).
    coord_req = state.coordination_map[
        state.player_level, block_position[:, 0], block_position[:, 1]
    ]

    # Sync attempt tracking (event-level, for coordination_success_rate denominator)
    is_sync = coord_req > 0
    is_sync_attempt = attempted_mining & is_sync  # ANY agent acting on a sync block (incl. solo)

    # Coordination success: coord_req met (enough agents present)
    is_multi_agent_success = is_sync_attempt & (agents_at_same_pos >= coord_req)

    # Solo soft tracking: agent on soft sync block without meeting coord_req
    is_soft = state.soft_coordination_mask[
        state.player_level, block_position[:, 0], block_position[:, 1]
    ]
    is_solo_on_soft = attempted_mining & is_sync & is_soft & (agents_at_same_pos < coord_req)
    is_solo_soft_success = is_solo_on_soft & doing_action_non_construction

    achievements = state.achievements
    is_2 = is_sync_success & (coord_req == 2)
    is_3_plus = is_sync_success & (coord_req >= 3)
    achievements = set_achievement(
        achievements,
        Achievement.COORD_2_AGENTS_SOFT,
        is_2 & is_soft_sync & (agents_at_same_pos >= 2),
    )
    achievements = set_achievement(
        achievements, Achievement.COORD_2_AGENTS_HARD, is_2 & ~is_soft_sync
    )
    achievements = set_achievement(
        achievements,
        Achievement.COORD_3_AGENTS_SOFT,
        is_3_plus & is_soft_sync & (agents_at_same_pos >= 3),
    )
    achievements = set_achievement(
        achievements, Achievement.COORD_3_AGENTS_HARD, is_3_plus & ~is_soft_sync
    )

    new_sync_coord_by_agents = state.sync_coord_by_agents
    new_sync_coord_by_agents = new_sync_coord_by_agents.at[0].add(is_2.any().astype(jnp.int32))
    new_sync_coord_by_agents = new_sync_coord_by_agents.at[1].add(is_3_plus.any().astype(jnp.int32))
    state = state.replace(achievements=achievements, sync_coord_by_agents=new_sync_coord_by_agents)

    # Combat
    (
        state,
        did_attack_mob,
        did_kill_mob,
        elite_melee_kills,
        elite_ranged_kills,
        large_passive_kills,
        elite_melee_killed_per_agent,
        elite_ranged_killed_per_agent,
        large_passive_killed_per_agent,
        elite_attempt,
        elite_success,
    ) = attack_mob(
        state,
        doing_action,
        block_position,
        get_player_damage_vector(state),
        is_forager,
        equal_block_placement,
        env_params,
        static_params,
    )
    # Track elite mob kills and coordination attempts
    state = state.replace(
        coord_elite_melee_kills=state.coord_elite_melee_kills + elite_melee_kills,
        coord_elite_ranged_kills=state.coord_elite_ranged_kills + elite_ranged_kills,
        coord_large_passive_kills=state.coord_large_passive_kills + large_passive_kills,
        coord_elite_attempts=state.coord_elite_attempts + elite_attempt,
        coord_elite_successes=state.coord_elite_successes + elite_success,
    )
    achievements = state.achievements
    achievements = set_achievement(
        achievements, Achievement.COORD_ELITE_MELEE_KILL, elite_melee_killed_per_agent
    )
    achievements = set_achievement(
        achievements, Achievement.COORD_ELITE_RANGED_KILL, elite_ranged_killed_per_agent
    )
    achievements = set_achievement(
        achievements, Achievement.COORD_LARGE_PASSIVE_KILL, large_passive_killed_per_agent
    )
    state = state.replace(achievements=achievements)

    # Interact with other players (Damage/Revive)
    state = interplayer_interaction(state, block_position, doing_action, env_params, static_params)

    # BATCHED MINING & INTERACTIONS
    # Key insight: each map position holds exactly one block type, so at most one
    # mining/interaction condition is true per position. This lets us compute the
    # replacement block and all mining flags in parallel, then apply a SINGLE map
    # write and a SINGLE inventory update instead of ~12 sequential ones.

    resource_yield = coord_multiplier.astype(jnp.int32)
    pickaxe = state.inventory.pickaxe

    # Block type identification
    is_tree = current_blocks == BlockType.TREE.value
    is_fire_tree = current_blocks == BlockType.FIRE_TREE.value
    is_ice_shrub = current_blocks == BlockType.ICE_SHRUB.value
    is_any_tree = is_tree | is_fire_tree | is_ice_shrub
    is_stone = current_blocks == BlockType.STONE.value
    is_coal = current_blocks == BlockType.COAL.value
    is_iron = current_blocks == BlockType.IRON.value
    is_diamond = current_blocks == BlockType.DIAMOND.value
    is_sapphire = current_blocks == BlockType.SAPPHIRE.value
    is_ruby = current_blocks == BlockType.RUBY.value
    is_stalagmite = current_blocks == BlockType.STALAGMITE.value
    is_chest = current_blocks == BlockType.CHEST.value
    is_ripe_plant = current_blocks == BlockType.RIPE_PLANT.value

    # Per-player mining/interaction conditions
    is_mining_tree = is_any_tree & doing_action
    is_mining_stone = is_stone & (pickaxe >= 1) & doing_action
    is_mining_coal = is_coal & (pickaxe >= 1) & doing_action
    is_mining_iron = is_iron & (pickaxe >= 2) & doing_action
    is_mining_diamond = is_diamond & (pickaxe >= 3) & doing_action
    is_mining_sapphire = is_sapphire & (pickaxe >= 4) & doing_action
    is_mining_ruby = is_ruby & (pickaxe >= 4) & doing_action
    is_mining_stalagmite = is_stalagmite & (pickaxe >= 1) & doing_action
    is_opening_chest = is_chest & doing_action
    is_eating_plant = is_ripe_plant & doing_action

    # Combined flag: is any player changing the block at each position?
    is_changing_map = (
        is_mining_tree
        | is_mining_stone
        | is_mining_coal
        | is_mining_iron
        | is_mining_diamond
        | is_mining_sapphire
        | is_mining_ruby
        | is_mining_stalagmite
        | is_opening_chest
        | is_eating_plant
    )
    is_any_player_changing = (equal_block_placement & is_changing_map[:, None]).any(axis=0)

    # Replacement block per position (exactly one type matches per position, so sum is unambiguous)
    replacement_block = (
        is_tree * BlockType.GRASS.value
        + is_fire_tree * BlockType.FIRE_GRASS.value
        + is_ice_shrub * BlockType.ICE_GRASS.value
        + (
            is_stone
            | is_coal
            | is_iron
            | is_diamond
            | is_sapphire
            | is_ruby
            | is_stalagmite
            | is_chest
        )
        * BlockType.PATH.value
        + is_ripe_plant * BlockType.PLANT.value
    )

    # Update whole map at once
    # process_handover might have updated the map for construction handovers, so we need to read the current block types again to avoid overwriting those updates
    post_handover_blocks = state.map[state.player_level, block_position[:, 0], block_position[:, 1]]
    new_map = (
        state.map[state.player_level]
        .at[block_position[:, 0], block_position[:, 1]]
        .set(jnp.where(is_any_player_changing, replacement_block, post_handover_blocks))
    )

    # SPECIAL INTERACTIONS
    # Sapling collection (foragers speciality, random chance)
    rng, _rng = jax.random.split(rng)
    is_grass = current_blocks == BlockType.GRASS.value
    sapling_roll = jax.random.uniform(_rng, (static_params.player_count,))
    is_mining_sapling = is_grass & (sapling_roll < 0.2 * is_forager) & doing_action

    # Update whole inventory at once
    new_inventory = state.inventory.replace(
        wood=state.inventory.wood + resource_yield * is_mining_tree,
        stone=state.inventory.stone + resource_yield * is_mining_stone + is_mining_stalagmite,
        coal=state.inventory.coal + resource_yield * is_mining_coal,
        iron=state.inventory.iron + resource_yield * is_mining_iron,
        diamond=state.inventory.diamond + resource_yield * is_mining_diamond,
        sapphire=state.inventory.sapphire + resource_yield * is_mining_sapphire,
        ruby=state.inventory.ruby + resource_yield * is_mining_ruby,
        sapling=state.inventory.sapling + is_mining_sapling,
    )

    # Chest loot (must come after base inventory update since it adds on top)
    rng, _rng = jax.random.split(rng)
    new_inventory = add_items_from_chest(_rng, state, new_inventory, is_opening_chest, env_params)

    # Water/Fountain (foragers speciality)
    is_water = (current_blocks == BlockType.WATER.value) | (
        current_blocks == BlockType.FOUNTAIN.value
    )
    is_drinking_water = is_water & doing_action & is_forager
    new_drink = jnp.where(
        is_drinking_water,
        jnp.minimum(get_max_drink(state), state.player_drink + 4),
        state.player_drink,
    )
    new_thirst = jnp.where(is_drinking_water, 0.0, state.player_thirst)

    # Ripe Plant (eat for food)
    new_food = jnp.where(
        is_eating_plant, jnp.minimum(get_max_food(state), state.player_food + 4), state.player_food
    )
    new_hunger = jnp.where(is_eating_plant, 0.0, state.player_hunger)
    is_any_eating_plant = (equal_block_placement & is_eating_plant[:, None]).any(axis=0)
    new_growing_plants_age = update_plants_with_eat(state, block_position, is_any_eating_plant)

    # Chest tracking
    new_chests_opened = state.chests_opened.at[state.player_level].set(
        state.chests_opened[state.player_level] | is_opening_chest
    )

    # Boss (Necromancer)
    is_attacking_boss = (current_blocks == BlockType.NECROMANCER.value) & doing_action
    can_damage_boss = is_boss_vulnerable(state) & is_fighting_boss(state, static_params)
    is_damaging_boss = is_attacking_boss & can_damage_boss
    new_boss_progress = state.boss_progress + is_damaging_boss.any().astype(jnp.int32)
    new_boss_timesteps_to_spawn_this_round = jnp.where(
        is_damaging_boss.any(), BOSS_FIGHT_SPAWN_TURNS, state.boss_timesteps_to_spawn_this_round
    )

    # ACHIEVEMENTS & METRICS
    new_achievements = state.achievements
    new_achievements = set_achievement(
        new_achievements, Achievement.COLLECT_DRINK, is_drinking_water
    )
    new_achievements = set_achievement(new_achievements, Achievement.EAT_PLANT, is_eating_plant)
    new_achievements = set_achievement(new_achievements, Achievement.OPEN_CHEST, is_opening_chest)
    new_achievements = set_achievement(
        new_achievements, Achievement.DAMAGE_NECROMANCER, is_damaging_boss
    )

    # Mining coordination tracking
    is_mining_resource = (
        is_mining_tree
        | is_mining_stone
        | is_mining_coal
        | is_mining_iron
        | is_mining_diamond
        | is_mining_sapphire
        | is_mining_ruby
    )
    actually_coordinated = agents_at_same_pos >= 2
    agent_mining_sync_soft = is_soft_sync & is_mining_resource & actually_coordinated
    agent_mining_sync_hard = is_sync_success & is_mining_resource & ~is_soft_sync

    # Resource-specific coordinated mining achievements
    for is_mining, soft_ach, hard_ach in [
        (is_mining_stone, Achievement.COORD_MINE_STONE_SOFT, Achievement.COORD_MINE_STONE_HARD),
        (is_mining_coal, Achievement.COORD_MINE_COAL_SOFT, Achievement.COORD_MINE_COAL_HARD),
        (is_mining_iron, Achievement.COORD_MINE_IRON_SOFT, Achievement.COORD_MINE_IRON_HARD),
        (
            is_mining_diamond,
            Achievement.COORD_MINE_DIAMOND_SOFT,
            Achievement.COORD_MINE_DIAMOND_HARD,
        ),
        (
            is_mining_sapphire,
            Achievement.COORD_MINE_SAPPHIRE_SOFT,
            Achievement.COORD_MINE_SAPPHIRE_HARD,
        ),
        (is_mining_ruby, Achievement.COORD_MINE_RUBY_SOFT, Achievement.COORD_MINE_RUBY_HARD),
    ]:
        new_achievements = set_achievement(
            new_achievements, soft_ach, is_soft_sync & is_mining & actually_coordinated
        )
        new_achievements = set_achievement(
            new_achievements, hard_ach, is_sync_success & is_mining & ~is_soft_sync
        )

    # UPDATE STATE
    # Clear coordination_map at positions where resources were successfully mined
    is_any_mining = (equal_block_placement & is_mining_resource[:, None]).any(axis=0)
    bp0, bp1 = block_position[:, 0], block_position[:, 1]
    new_coordination_map = state.coordination_map.at[state.player_level, bp0, bp1].set(
        jnp.where(is_any_mining, 0, state.coordination_map[state.player_level, bp0, bp1])
    )
    new_soft_mask = state.soft_coordination_mask.at[state.player_level, bp0, bp1].set(
        jnp.where(is_any_mining, False, state.soft_coordination_mask[state.player_level, bp0, bp1])
    )

    state = state.replace(
        map=state.map.at[state.player_level].set(new_map),
        inventory=new_inventory,
        player_drink=new_drink,
        player_thirst=new_thirst,
        player_food=new_food,
        player_hunger=new_hunger,
        growing_plants_age=new_growing_plants_age,
        achievements=new_achievements,
        chests_opened=new_chests_opened,
        boss_progress=new_boss_progress,
        boss_timesteps_to_spawn_this_round=new_boss_timesteps_to_spawn_this_round,
        coord_mine_sync_soft_count=state.coord_mine_sync_soft_count
        + agent_mining_sync_soft.any().astype(jnp.int32),
        coord_mine_sync_hard_count=state.coord_mine_sync_hard_count
        + agent_mining_sync_hard.any().astype(jnp.int32),
        coord_sync_attempts=state.coord_sync_attempts + is_sync_attempt.any().astype(jnp.int32),
        coord_sync_successes=state.coord_sync_successes
        + is_multi_agent_success.any().astype(jnp.int32),
        soft_sync_events=state.soft_sync_events + is_soft_sync.any().astype(jnp.int32),
        soft_sync_bonus_events=state.soft_sync_bonus_events
        + (is_soft_sync & (agents_at_same_pos >= coord_req)).any().astype(jnp.int32),
        coord_solo_soft_attempts=state.coord_solo_soft_attempts
        + is_solo_on_soft.any().astype(jnp.int32),
        coord_solo_soft_successes=state.coord_solo_soft_successes
        + is_solo_soft_success.any().astype(jnp.int32),
        coordination_map=new_coordination_map,
        soft_coordination_mask=new_soft_mask,
    )

    return state


def do_crafting(rng, state, actions, params, static_params):
    """Resolve all crafting actions and their material costs.

    Args:
        rng: JAX random key used by specialization gates.
        state: Current environment state.
        actions: Per-player action identifiers.
        params: Gameplay parameters controlling crafting behavior.
        static_params: Static map and player parameters.

    Returns:
        State with crafted equipment, inventories, and achievements updated.
    """
    is_at_crafting_table = is_near_block(state, BlockType.CRAFTING_TABLE.value, static_params)
    is_at_furnace = is_near_block(state, BlockType.FURNACE.value, static_params)
    is_at_epic_forge = is_near_block(state, BlockType.EPIC_FORGE.value, static_params)

    # Compute per-forge proximity matrix for diamond crafting coordination
    _has_adj_forge, _forge_pos = get_nearest_block_pos(
        state, BlockType.EPIC_FORGE.value, static_params
    )
    # Two agents share a forge if both are near one and their forge positions match
    # i.e. both near same forge
    same_forge = (
        _has_adj_forge[:, None]
        & _has_adj_forge[None, :]
        & (_forge_pos[:, None, :] == _forge_pos[None, :, :]).all(axis=-1)
    )

    is_miner = state.player_specialization == Specialization.MINER.value
    is_warrior = state.player_specialization == Specialization.WARRIOR.value
    # Apply soft gating to specializations
    is_miner, rng = soft_gate_specialization(rng, is_miner, params)
    is_warrior, rng = soft_gate_specialization(rng, is_warrior, params)

    new_achievements = state.achievements

    # Wood pickaxe
    can_craft_wood_pickaxe = jnp.logical_and(state.inventory.wood >= 1, is_miner)

    is_crafting_wood_pickaxe = jnp.logical_and(
        actions == Action.MAKE_WOOD_PICKAXE.value,
        jnp.logical_and(
            can_craft_wood_pickaxe,
            jnp.logical_and(is_at_crafting_table, state.inventory.pickaxe < 1),
        ),
    )

    new_inventory = state.inventory.replace(
        wood=state.inventory.wood - 1 * is_crafting_wood_pickaxe,
        pickaxe=state.inventory.pickaxe * (1 - is_crafting_wood_pickaxe)
        + 1 * is_crafting_wood_pickaxe,
    )

    # Stone pickaxe
    can_craft_stone_pickaxe = jnp.logical_and(
        is_miner, jnp.logical_and(new_inventory.wood >= 1, new_inventory.stone >= 1)
    )
    is_crafting_stone_pickaxe = jnp.logical_and(
        actions == Action.MAKE_STONE_PICKAXE.value,
        jnp.logical_and(
            can_craft_stone_pickaxe,
            jnp.logical_and(is_at_crafting_table, new_inventory.pickaxe < 2),
        ),
    )

    new_inventory = new_inventory.replace(
        stone=new_inventory.stone - 1 * is_crafting_stone_pickaxe,
        wood=new_inventory.wood - 1 * is_crafting_stone_pickaxe,
        pickaxe=new_inventory.pickaxe * (1 - is_crafting_stone_pickaxe)
        + 2 * is_crafting_stone_pickaxe,
    )

    # Iron pickaxe
    can_craft_iron_pickaxe = jnp.logical_and(
        new_inventory.wood >= 1,
        jnp.logical_and(
            new_inventory.stone >= 1,
            jnp.logical_and(
                new_inventory.iron >= 1,
                new_inventory.coal >= 1,
            ),
        ),
    )
    can_craft_iron_pickaxe = jnp.logical_and(
        is_miner,
        can_craft_iron_pickaxe,
    )
    is_crafting_iron_pickaxe = jnp.logical_and(
        actions == Action.MAKE_IRON_PICKAXE.value,
        jnp.logical_and(
            can_craft_iron_pickaxe,
            jnp.logical_and(
                is_at_furnace,
                jnp.logical_and(is_at_crafting_table, new_inventory.pickaxe < 3),
            ),
        ),
    )

    new_inventory = new_inventory.replace(
        iron=new_inventory.iron - 1 * is_crafting_iron_pickaxe,
        wood=new_inventory.wood - 1 * is_crafting_iron_pickaxe,
        stone=new_inventory.stone - 1 * is_crafting_iron_pickaxe,
        coal=new_inventory.coal - 1 * is_crafting_iron_pickaxe,
        pickaxe=new_inventory.pickaxe * (1 - is_crafting_iron_pickaxe)
        + 3 * is_crafting_iron_pickaxe,
    )

    # Diamond pickaxe
    can_craft_diamond_pickaxe = jnp.logical_and(new_inventory.wood >= 1, new_inventory.diamond >= 3)
    can_craft_diamond_pickaxe = jnp.logical_and(
        is_miner,
        can_craft_diamond_pickaxe,
    )
    # Diamond crafting location
    # if coordination enabled you need to use an epic forge for diamond items, else you can use simple crafting table
    diamond_craft_location = jnp.where(
        params.crafting_coordination_enabled, is_at_epic_forge, is_at_crafting_table
    )
    wants_diamond_pickaxe = jnp.logical_and(
        actions == Action.MAKE_DIAMOND_PICKAXE.value,
        jnp.logical_and(
            can_craft_diamond_pickaxe,
            jnp.logical_and(diamond_craft_location, new_inventory.pickaxe < 4),
        ),
    )
    # Coordination check: count agents at the SAME forge attempting to craft diamond pickaxe
    agents_at_same_forge_pickaxe = (same_forge & wants_diamond_pickaxe[None, :]).sum(axis=1)
    coord_met_diamond_pickaxe = jnp.where(
        params.crafting_coordination_enabled,
        agents_at_same_forge_pickaxe >= params.diamond_crafting_agents_required,
        True,
    )
    is_crafting_diamond_pickaxe = wants_diamond_pickaxe & coord_met_diamond_pickaxe

    new_inventory = new_inventory.replace(
        diamond=new_inventory.diamond - 3 * is_crafting_diamond_pickaxe,
        wood=new_inventory.wood - 1 * is_crafting_diamond_pickaxe,
        pickaxe=new_inventory.pickaxe * (1 - is_crafting_diamond_pickaxe)
        + 4 * is_crafting_diamond_pickaxe,
    )

    # Wood sword
    can_craft_wood_sword = new_inventory.wood >= 1
    is_crafting_wood_sword = jnp.logical_and(
        actions == Action.MAKE_WOOD_SWORD.value,
        jnp.logical_and(
            can_craft_wood_sword,
            jnp.logical_and(is_at_crafting_table, new_inventory.sword < 1),
        ),
    )

    new_inventory = new_inventory.replace(
        wood=new_inventory.wood - 1 * is_crafting_wood_sword,
        sword=new_inventory.sword * (1 - is_crafting_wood_sword) + 1 * is_crafting_wood_sword,
    )

    # Stone sword
    can_craft_stone_sword = jnp.logical_and(new_inventory.stone >= 1, new_inventory.wood >= 1)
    can_craft_stone_sword = jnp.logical_and(can_craft_stone_sword, is_warrior)
    is_crafting_stone_sword = jnp.logical_and(
        actions == Action.MAKE_STONE_SWORD.value,
        jnp.logical_and(
            can_craft_stone_sword,
            jnp.logical_and(is_at_crafting_table, new_inventory.sword < 2),
        ),
    )

    new_inventory = new_inventory.replace(
        wood=new_inventory.wood - 1 * is_crafting_stone_sword,
        stone=new_inventory.stone - 1 * is_crafting_stone_sword,
        sword=new_inventory.sword * (1 - is_crafting_stone_sword) + 2 * is_crafting_stone_sword,
    )

    # Iron sword
    can_craft_iron_sword = jnp.logical_and(
        new_inventory.iron >= 1,
        jnp.logical_and(
            new_inventory.wood >= 1,
            jnp.logical_and(new_inventory.stone >= 1, new_inventory.coal >= 1),
        ),
    )
    can_craft_iron_sword = jnp.logical_and(is_warrior, can_craft_iron_sword)
    is_crafting_iron_sword = jnp.logical_and(
        actions == Action.MAKE_IRON_SWORD.value,
        jnp.logical_and(
            can_craft_iron_sword,
            jnp.logical_and(
                is_at_furnace,
                jnp.logical_and(is_at_crafting_table, new_inventory.sword < 3),
            ),
        ),
    )

    new_inventory = new_inventory.replace(
        wood=new_inventory.wood - 1 * is_crafting_iron_sword,
        iron=new_inventory.iron - 1 * is_crafting_iron_sword,
        stone=new_inventory.stone - 1 * is_crafting_iron_sword,
        coal=new_inventory.coal - 1 * is_crafting_iron_sword,
        sword=new_inventory.sword * (1 - is_crafting_iron_sword) + 3 * is_crafting_iron_sword,
    )

    # Diamond sword
    can_craft_diamond_sword = jnp.logical_and(new_inventory.diamond >= 2, new_inventory.wood >= 1)
    can_craft_diamond_sword = jnp.logical_and(is_warrior, can_craft_diamond_sword)
    wants_diamond_sword = jnp.logical_and(
        actions == Action.MAKE_DIAMOND_SWORD.value,
        jnp.logical_and(
            can_craft_diamond_sword,
            jnp.logical_and(diamond_craft_location, new_inventory.sword < 4),
        ),
    )
    # Coordination check: count agents at the SAME forge attempting to craft diamond sword
    agents_at_same_forge_sword = (same_forge & wants_diamond_sword[None, :]).sum(axis=1)
    coord_met_diamond_sword = jnp.where(
        params.crafting_coordination_enabled,
        agents_at_same_forge_sword >= params.diamond_crafting_agents_required,
        True,
    )
    is_crafting_diamond_sword = wants_diamond_sword & coord_met_diamond_sword

    new_inventory = new_inventory.replace(
        wood=new_inventory.wood - 1 * is_crafting_diamond_sword,
        diamond=new_inventory.diamond - 2 * is_crafting_diamond_sword,
        sword=new_inventory.sword * (1 - is_crafting_diamond_sword) + 4 * is_crafting_diamond_sword,
    )

    # Iron armour
    can_craft_iron_armour = (new_inventory.armour < 1).sum(axis=1) > 0
    can_craft_iron_armour = jnp.logical_and(
        can_craft_iron_armour,
        jnp.logical_and(new_inventory.iron >= 3, new_inventory.coal >= 3),
    )

    iron_armour_index_to_craft = jnp.argmax(new_inventory.armour < 1, axis=1)

    is_crafting_iron_armour = jnp.logical_and(
        actions == Action.MAKE_IRON_ARMOUR.value,
        jnp.logical_and(
            can_craft_iron_armour,
            jnp.logical_and(is_at_crafting_table, is_at_furnace),
        ),
    )

    new_inventory = new_inventory.replace(
        iron=new_inventory.iron - 3 * is_crafting_iron_armour,
        coal=new_inventory.coal - 3 * is_crafting_iron_armour,
        armour=new_inventory.armour.at[
            jnp.arange(0, len(new_inventory.armour)), iron_armour_index_to_craft
        ].set(
            is_crafting_iron_armour * 1
            + (1 - is_crafting_iron_armour)
            * new_inventory.armour[
                jnp.arange(0, len(new_inventory.armour)), iron_armour_index_to_craft
            ]
        ),
    )
    new_achievements = new_achievements.at[:, Achievement.MAKE_IRON_ARMOUR.value].set(
        jnp.logical_or(
            new_achievements[:, Achievement.MAKE_IRON_ARMOUR.value],
            is_crafting_iron_armour,
        )
    )

    # Diamond armour
    can_craft_diamond_armour = (new_inventory.armour < 2).sum(axis=1) > 0
    can_craft_diamond_armour = jnp.logical_and(can_craft_diamond_armour, new_inventory.diamond >= 3)

    diamond_armour_index_to_craft = jnp.argmax(new_inventory.armour < 2, axis=1)

    wants_diamond_armour = jnp.logical_and(
        actions == Action.MAKE_DIAMOND_ARMOUR.value,
        jnp.logical_and(
            can_craft_diamond_armour,
            diamond_craft_location,
        ),
    )
    # Coordination check: count agents at the SAME forge attempting to craft diamond armour
    agents_at_same_forge_armour = (same_forge & wants_diamond_armour[None, :]).sum(axis=1)
    coord_met_diamond_armour = jnp.where(
        params.crafting_coordination_enabled,
        agents_at_same_forge_armour >= params.diamond_crafting_agents_required,
        True,
    )
    is_crafting_diamond_armour = wants_diamond_armour & coord_met_diamond_armour

    new_inventory = new_inventory.replace(
        diamond=new_inventory.diamond - 3 * is_crafting_diamond_armour,
        armour=new_inventory.armour.at[
            jnp.arange(0, len(new_inventory.armour)), diamond_armour_index_to_craft
        ].set(
            is_crafting_diamond_armour * 2
            + (1 - is_crafting_diamond_armour)
            * new_inventory.armour[
                jnp.arange(0, len(new_inventory.armour)), diamond_armour_index_to_craft
            ]
        ),
    )
    new_achievements = new_achievements.at[:, Achievement.MAKE_DIAMOND_ARMOUR.value].set(
        jnp.logical_or(
            new_achievements[:, Achievement.MAKE_DIAMOND_ARMOUR.value],
            is_crafting_diamond_armour,
        )
    )

    # Arrow
    can_craft_arrow = jnp.logical_and(new_inventory.stone >= 1, new_inventory.wood >= 1)
    can_craft_arrow = jnp.logical_and(can_craft_arrow, is_warrior)
    is_crafting_arrow = jnp.logical_and(
        actions == Action.MAKE_ARROW.value,
        jnp.logical_and(
            can_craft_arrow,
            jnp.logical_and(is_at_crafting_table, new_inventory.arrows < 99),
        ),
    )
    new_inventory = new_inventory.replace(
        wood=new_inventory.wood - 1 * is_crafting_arrow,
        stone=new_inventory.stone - 1 * is_crafting_arrow,
        arrows=new_inventory.arrows + 2 * is_crafting_arrow,
    )

    # Torch
    can_craft_torch = jnp.logical_and(new_inventory.coal >= 1, new_inventory.wood >= 1)
    can_craft_torch = jnp.logical_and(can_craft_torch, is_miner)
    is_crafting_torch = jnp.logical_and(
        actions == Action.MAKE_TORCH.value,
        jnp.logical_and(
            can_craft_torch,
            jnp.logical_and(is_at_crafting_table, new_inventory.torches < 99),
        ),
    )
    new_inventory = new_inventory.replace(
        wood=new_inventory.wood - 1 * is_crafting_torch,
        coal=new_inventory.coal - 1 * is_crafting_torch,
        torches=new_inventory.torches + 4 * is_crafting_torch,
    )

    # Track diamond crafting coordination attempts (agent wants to craft, has materials, at forge)
    craft_attempt = jnp.where(
        params.crafting_coordination_enabled,
        (wants_diamond_pickaxe | wants_diamond_sword | wants_diamond_armour)
        .any()
        .astype(jnp.int32),
        0,
    )
    craft_success = jnp.where(
        params.crafting_coordination_enabled,
        (is_crafting_diamond_pickaxe | is_crafting_diamond_sword | is_crafting_diamond_armour)
        .any()
        .astype(jnp.int32),
        0,
    )

    # Track diamond crafting coordination (only when crafting coordination is enabled)
    diamond_pickaxe_crafted = jnp.where(
        params.crafting_coordination_enabled, is_crafting_diamond_pickaxe.any().astype(jnp.int32), 0
    )
    diamond_sword_crafted = jnp.where(
        params.crafting_coordination_enabled, is_crafting_diamond_sword.any().astype(jnp.int32), 0
    )
    diamond_armour_crafted = jnp.where(
        params.crafting_coordination_enabled, is_crafting_diamond_armour.any().astype(jnp.int32), 0
    )

    # Update diamond crafting coordination achievements only when coordination is enabled
    new_achievements = new_achievements.at[:, Achievement.COORD_DIAMOND_PICKAXE.value].set(
        jnp.logical_or(
            new_achievements[:, Achievement.COORD_DIAMOND_PICKAXE.value],
            is_crafting_diamond_pickaxe & params.crafting_coordination_enabled,
        )
    )
    new_achievements = new_achievements.at[:, Achievement.COORD_DIAMOND_SWORD.value].set(
        jnp.logical_or(
            new_achievements[:, Achievement.COORD_DIAMOND_SWORD.value],
            is_crafting_diamond_sword & params.crafting_coordination_enabled,
        )
    )
    new_achievements = new_achievements.at[:, Achievement.COORD_DIAMOND_ARMOUR.value].set(
        jnp.logical_or(
            new_achievements[:, Achievement.COORD_DIAMOND_ARMOUR.value],
            is_crafting_diamond_armour & params.crafting_coordination_enabled,
        )
    )

    state = state.replace(
        inventory=new_inventory,
        achievements=new_achievements,
        coord_craft_attempts=state.coord_craft_attempts + craft_attempt,
        coord_craft_successes=state.coord_craft_successes + craft_success,
        coord_diamond_pickaxe_count=state.coord_diamond_pickaxe_count + diamond_pickaxe_crafted,
        coord_diamond_sword_count=state.coord_diamond_sword_count + diamond_sword_crafted,
        coord_diamond_armour_count=state.coord_diamond_armour_count + diamond_armour_crafted,
    )

    return state


def add_new_growing_plant(
    growing_plant_positions, growing_plant_age, growing_plant_mask, position, is_placing_sapling
):
    """Insert a newly planted sapling into the first free plant slot.

    Args:
        growing_plant_positions: Coordinates for all plant slots.
        growing_plant_age: Age value for all plant slots.
        growing_plant_mask: Active status for all plant slots.
        position: Coordinate of the candidate sapling.
        is_placing_sapling: Whether the placement action succeeded.

    Returns:
        Updated positions, ages, mask, and a placement-success flag.
    """
    is_empty = jnp.logical_not(growing_plant_mask)
    plant_index = jnp.argmax(is_empty)
    is_an_empty_slot = is_empty.any()
    is_adding_plant = jnp.logical_and(is_an_empty_slot, is_placing_sapling)

    new_growing_plants_positions = jax.lax.select(
        is_adding_plant,
        growing_plant_positions.at[plant_index].set(position),
        growing_plant_positions,
    )
    new_growing_plants_age = jax.lax.select(
        is_adding_plant,
        growing_plant_age.at[plant_index].set(0),
        growing_plant_age,
    )
    new_growing_plants_mask = jax.lax.select(
        is_adding_plant,
        growing_plant_mask.at[plant_index].set(True),
        growing_plant_mask,
    )
    return (
        new_growing_plants_positions,
        new_growing_plants_age,
        new_growing_plants_mask,
        is_adding_plant,
    )


def place_block(rng, state, action, env_params, static_params):
    """Resolve block placement, construction, and placement coordination.

    Args:
        rng: JAX random key used by stochastic coordination gates.
        state: Current environment state.
        action: Per-player action identifiers.
        env_params: Gameplay and coordination parameters.
        static_params: Static map and player parameters.

    Returns:
        State with maps, inventories, plants, and metrics updated.
    """
    placing_block_position = state.player_position + DIRECTIONS[state.player_direction]
    equal_block_placement = (
        jnp.expand_dims(placing_block_position, axis=1)
        == jnp.expand_dims(placing_block_position, axis=0)
    ).all(axis=2)

    new_map = state.map[state.player_level]
    new_item_map = state.item_map[state.player_level]

    is_miner = state.player_specialization == Specialization.MINER.value
    # Apply soft gating to specialization
    is_miner, rng = soft_gate_specialization(rng, is_miner, env_params)

    is_block_in_other_player = is_in_other_player(state, placing_block_position)
    is_block_in_mob = is_in_mob(state, placing_block_position)
    is_block_in_bounds = in_bounds(placing_block_position, static_params)
    is_placement_in_bounds_not_in_mobs = jnp.logical_and(
        is_block_in_bounds,
        jnp.logical_not(jnp.logical_or(is_block_in_other_player, is_block_in_mob)),
    )

    # Terrain at placement position — used to reject water/lava for structures
    placement_terrain = new_map[placing_block_position[:, 0], placing_block_position[:, 1]]
    is_water_or_lava = jnp.logical_or(
        placement_terrain == BlockType.WATER.value,
        placement_terrain == BlockType.LAVA.value,
    )

    # Crafting table
    is_valid_placement = jnp.logical_and(
        is_placement_in_bounds_not_in_mobs,
        jnp.logical_and(
            jnp.logical_not(is_in_solid_block(new_map, placing_block_position)),
            jnp.logical_and(
                jnp.logical_not(is_water_or_lava),
                new_item_map[placing_block_position[:, 0], placing_block_position[:, 1]]
                == ItemType.NONE.value,
            ),
        ),
    )
    crafting_table_key_down = action == Action.PLACE_TABLE.value
    has_wood = state.inventory.wood >= 2
    is_player_placing_crafting_table = jnp.logical_and(
        crafting_table_key_down,
        jnp.logical_and(is_valid_placement, has_wood),
    )
    is_any_player_placing_crafting_table = jnp.logical_and(
        equal_block_placement, is_player_placing_crafting_table[:, None]
    ).any(axis=0)

    placed_crafting_table_block = jnp.where(
        is_any_player_placing_crafting_table,
        BlockType.CRAFTING_TABLE.value,
        new_map[placing_block_position[:, 0], placing_block_position[:, 1]],
    )
    new_map = new_map.at[placing_block_position[:, 0], placing_block_position[:, 1]].set(
        placed_crafting_table_block
    )
    new_inventory = state.inventory.replace(
        wood=state.inventory.wood - 2 * is_player_placing_crafting_table
    )
    new_achievements = state.achievements.at[:, Achievement.PLACE_TABLE.value].set(
        jnp.logical_or(
            state.achievements[:, Achievement.PLACE_TABLE.value], is_player_placing_crafting_table
        )
    )

    # Furnace
    is_valid_placement = jnp.logical_and(
        is_placement_in_bounds_not_in_mobs,
        jnp.logical_and(
            jnp.logical_not(is_in_solid_block(new_map, placing_block_position)),
            jnp.logical_and(
                jnp.logical_not(is_water_or_lava),
                new_item_map[placing_block_position[:, 0], placing_block_position[:, 1]]
                == ItemType.NONE.value,
            ),
        ),
    )

    furnace_key_down = action == Action.PLACE_FURNACE.value
    has_stone = new_inventory.stone > 0
    is_player_placing_furnace = jnp.logical_and(
        furnace_key_down,
        jnp.logical_and(is_valid_placement, has_stone),
    )
    is_any_player_placing_furnace = jnp.logical_and(
        equal_block_placement, is_player_placing_furnace[:, None]
    ).any(axis=0)
    placed_furnace_block = jnp.where(
        is_any_player_placing_furnace,
        BlockType.FURNACE.value,
        new_map[placing_block_position[:, 0], placing_block_position[:, 1]],
    )
    new_map = new_map.at[placing_block_position[:, 0], placing_block_position[:, 1]].set(
        placed_furnace_block
    )
    new_inventory = new_inventory.replace(stone=new_inventory.stone - 1 * is_player_placing_furnace)
    new_achievements = new_achievements.at[:, Achievement.PLACE_FURNACE.value].set(
        jnp.logical_or(
            new_achievements[:, Achievement.PLACE_FURNACE.value], is_player_placing_furnace
        )
    )

    # Stone
    stone_key_down = action == Action.PLACE_STONE.value
    has_stone = new_inventory.stone > 0
    is_valid_placement = jnp.logical_and(
        is_placement_in_bounds_not_in_mobs,
        new_item_map[placing_block_position[:, 0], placing_block_position[:, 1]]
        == ItemType.NONE.value,
    )
    is_valid_placement = jnp.logical_and(
        is_valid_placement,
        jnp.logical_or(
            placement_terrain == BlockType.WATER.value,
            jnp.logical_and(
                jnp.logical_not(is_in_solid_block(new_map, placing_block_position)),
                # we dont do water check here because we allow placing stone in water (bridge)
                placement_terrain != BlockType.LAVA.value,
            ),
        ),
    )
    is_player_placing_stone = jnp.logical_and(
        stone_key_down,
        jnp.logical_and(is_valid_placement, has_stone),
    )
    is_player_placing_stone = jnp.logical_and(is_player_placing_stone, is_miner)
    is_any_player_placing_stone = jnp.logical_and(
        equal_block_placement, is_player_placing_stone[:, None]
    ).any(axis=0)
    placed_stone_block = jnp.where(
        is_any_player_placing_stone,
        BlockType.STONE.value,
        new_map[placing_block_position[:, 0], placing_block_position[:, 1]],
    )
    new_map = new_map.at[placing_block_position[:, 0], placing_block_position[:, 1]].set(
        placed_stone_block
    )
    new_inventory = new_inventory.replace(stone=new_inventory.stone - 1 * is_player_placing_stone)
    new_achievements = new_achievements.at[:, Achievement.PLACE_STONE.value].set(
        jnp.logical_or(new_achievements[:, Achievement.PLACE_STONE.value], is_player_placing_stone)
    )

    # Torch
    # TODO: Make more parallelized
    def _player_place_torch(action_info, player_index):
        (
            working_item_map,
            working_padded_light_map,
        ) = action_info

        torch_key_down = action[player_index] == Action.PLACE_TORCH.value
        has_torch = new_inventory.torches[player_index] > 0

        is_valid_placement = jnp.logical_and(
            CAN_PLACE_ITEM_MAPPING[
                new_map[
                    placing_block_position[player_index, 0], placing_block_position[player_index, 1]
                ]
            ],
            working_item_map[
                placing_block_position[player_index, 0], placing_block_position[player_index, 1]
            ]
            == ItemType.NONE.value,
        )
        is_valid_placement = jnp.logical_and(
            is_placement_in_bounds_not_in_mobs[player_index], is_valid_placement
        )
        is_player_placing_torch = jnp.logical_and(
            torch_key_down,
            jnp.logical_and(is_valid_placement, has_torch),
        )
        placed_torch_item = jax.lax.select(
            is_player_placing_torch,
            ItemType.TORCH.value,
            working_item_map[
                placing_block_position[player_index, 0], placing_block_position[player_index, 1]
            ],
        )
        working_item_map = working_item_map.at[
            placing_block_position[player_index, 0], placing_block_position[player_index, 1]
        ].set(placed_torch_item)

        current_light_map = jax.lax.dynamic_slice(
            working_padded_light_map,
            placing_block_position[player_index]
            - jnp.array([4, 4])
            + jnp.array([light_map_padding, light_map_padding]),
            (9, 9),
        )
        torch_light_map = jnp.clip(TORCH_LIGHT_MAP + current_light_map, 0.0, 1.0)
        torch_light_map = torch_light_map * is_player_placing_torch + current_light_map * (
            1 - is_player_placing_torch
        )
        working_padded_light_map = jax.lax.dynamic_update_slice(
            working_padded_light_map,
            torch_light_map,
            placing_block_position[player_index]
            - jnp.array([4, 4])
            + jnp.array([light_map_padding, light_map_padding]),
        )
        return (working_item_map, working_padded_light_map), is_player_placing_torch

    light_map_padding = 6
    padded_light_map_floor = jnp.pad(
        state.light_map[state.player_level],
        (light_map_padding, light_map_padding),
        constant_values=0,
    )
    (new_item_map, padded_light_map_floor), is_player_placing_torch = jax.lax.scan(
        _player_place_torch,
        (new_item_map, padded_light_map_floor),
        jnp.arange(static_params.player_count),
    )
    new_light_map_floor = padded_light_map_floor[
        light_map_padding:-light_map_padding, light_map_padding:-light_map_padding
    ]
    new_light_map = state.light_map.at[state.player_level].set(new_light_map_floor)

    new_inventory = new_inventory.replace(
        torches=new_inventory.torches - 1 * is_player_placing_torch
    )
    new_achievements = new_achievements.at[:, Achievement.PLACE_TORCH.value].set(
        jnp.logical_or(new_achievements[:, Achievement.PLACE_TORCH.value], is_player_placing_torch)
    )

    # Plant
    # TODO: Make more parallelized
    def _player_place_plant(action_info, player_index):
        (
            working_map,
            working_growing_plants_positions,
            working_growing_plants_age,
            working_growing_plants_mask,
        ) = action_info
        sapling_key_down = action[player_index] == Action.PLACE_PLANT.value
        has_sapling = state.inventory.sapling[player_index] > 0
        is_valid_placement = jnp.logical_and(
            is_placement_in_bounds_not_in_mobs[player_index],
            jnp.logical_and(
                working_map[
                    placing_block_position[player_index, 0], placing_block_position[player_index, 1]
                ]
                == BlockType.GRASS.value,
                new_item_map[
                    placing_block_position[player_index, 0], placing_block_position[player_index, 1]
                ]
                == ItemType.NONE.value,
            ),
        )
        is_player_placing_sapling = jnp.logical_and(
            is_valid_placement,
            jnp.logical_and(
                sapling_key_down,
                has_sapling,
            ),
        )
        (
            working_growing_plants_positions,
            working_growing_plants_age,
            working_growing_plants_mask,
            is_player_placing_sapling,
        ) = add_new_growing_plant(
            working_growing_plants_positions,
            working_growing_plants_age,
            working_growing_plants_mask,
            placing_block_position[player_index],
            is_player_placing_sapling,
        )
        placed_sapling_block = jax.lax.select(
            is_player_placing_sapling,
            BlockType.PLANT.value,
            working_map[
                placing_block_position[player_index, 0], placing_block_position[player_index, 1]
            ],
        )
        working_map = working_map.at[
            placing_block_position[player_index, 0], placing_block_position[player_index, 1]
        ].set(placed_sapling_block)
        return (
            working_map,
            working_growing_plants_positions,
            working_growing_plants_age,
            working_growing_plants_mask,
        ), is_player_placing_sapling

    (
        (new_map, new_growing_plants_positions, new_growing_plants_age, new_growing_plants_mask),
        is_player_placing_sapling,
    ) = jax.lax.scan(
        _player_place_plant,
        (
            new_map,
            state.growing_plants_positions,
            state.growing_plants_age,
            state.growing_plants_mask,
        ),
        jnp.arange(static_params.player_count),
    )

    new_inventory = new_inventory.replace(
        sapling=new_inventory.sapling - 1 * is_player_placing_sapling
    )
    new_achievements = new_achievements.at[:, Achievement.PLACE_PLANT.value].set(
        jnp.logical_or(
            new_achievements[:, Achievement.PLACE_PLANT.value], is_player_placing_sapling
        )
    )

    # Do?
    new_whole_map = state.map.at[state.player_level].set(new_map)
    new_whole_item_map = state.item_map.at[state.player_level].set(new_item_map)
    state = state.replace(
        map=new_whole_map,
        item_map=new_whole_item_map,
        light_map=new_light_map,
        inventory=new_inventory,
        achievements=new_achievements,
        growing_plants_positions=new_growing_plants_positions,
        growing_plants_age=new_growing_plants_age,
        growing_plants_mask=new_growing_plants_mask,
    )

    return state


def find_pending_matches(pending, block_positions, timestep, static_params):
    """Find matching active pending handovers for each agent's target position.

    A match requires: active entry, same position, not expired, different initiator.

    Args:
        pending: Pending-handover table.
        block_positions: Per-agent target coordinates.
        timestep: Current environment timestep.
        static_params: Static parameters containing player count.

    Returns:
        has_match: Boolean per agent, shape (player_count,)
        match_idx: Index into pending array of the match, shape (player_count,)
    """

    def _check(agent_idx):
        agent_pos = block_positions[agent_idx]
        matches = (
            (pending[:, 0] == 1)
            & (pending[:, 1] == agent_pos[0])
            & (pending[:, 2] == agent_pos[1])
            & (pending[:, 3] > timestep)
            & (pending[:, 4] != agent_idx)
        )
        return matches.any(), jnp.argmax(matches)

    return jax.vmap(_check)(jnp.arange(static_params.player_count))


def clear_completed_handovers(pending, is_completing, match_idx, static_params):
    """Zero out pending entries that have been completed.

    Args:
        pending: Pending-handover table.
        is_completing: Mask of agents completing a matched handover.
        match_idx: Matched table row for each agent.
        static_params: Static parameters containing player count.

    Returns:
        Pending-handover table with completed rows cleared.
    """

    def _clear(pending_arr, agent_idx):
        should_clear = is_completing[agent_idx]
        idx = match_idx[agent_idx]
        new_row = jnp.where(should_clear, jnp.zeros(6, dtype=jnp.int32), pending_arr[idx])
        return pending_arr.at[idx].set(new_row)

    return jax.lax.fori_loop(0, static_params.player_count, lambda i, p: _clear(p, i), pending)


def add_pending_handovers(
    pending,
    block_positions,
    is_setting_up,
    handover_window,
    timestep,
    static_params,
    building_type=None,
):
    """Add new pending handover entries for agents initiating setups.

    Must be called AFTER clear_completed_handovers so freed slots are available.

    Args:
        building_type: Per-agent int array (0=mining, 1=shelter, 2=forge, 3=beacon).
            If None, defaults to 0 (mining) for all agents.

    Returns:
        new_pending: Updated pending array
        setup_count: Number of setups successfully added
    """
    if building_type is None:
        building_type = jnp.zeros(static_params.player_count, dtype=jnp.int32)

    def _add(carry, agent_idx):
        pending_arr, setup_count = carry
        is_setup = is_setting_up[agent_idx]
        pos = block_positions[agent_idx]

        # Never create a second entry for a position that already has an active handover.
        # This guard covers both an initiator re-acting on its own handover (which can't self-complete)
        # and several agents initiating the same position in one step -- either would otherwise leave
        # a ghost slot that survives completion and keeps a stale clock rendering.
        already_pending = (
            (pending_arr[:, 0] == 1) & (pending_arr[:, 1] == pos[0]) & (pending_arr[:, 2] == pos[1])
        ).any()

        empty_slots = pending_arr[:, 0] == 0
        empty_idx = jnp.argmax(empty_slots)
        has_empty = empty_slots.any()

        deadline = timestep + handover_window[agent_idx] + 1
        new_entry = jnp.array([1, pos[0], pos[1], deadline, agent_idx, building_type[agent_idx]])

        should_add = is_setup & has_empty & ~already_pending
        new_row = jnp.where(should_add, new_entry, pending_arr[empty_idx])
        pending_arr = pending_arr.at[empty_idx].set(new_row)
        setup_count = setup_count + should_add.astype(jnp.int32)
        return (pending_arr, setup_count), None

    (new_pending, setup_count), _ = jax.lax.scan(
        _add, (pending, jnp.asarray(0, dtype=jnp.int32)), jnp.arange(static_params.player_count)
    )
    return new_pending, setup_count


def process_handover(state, block_positions, is_acting, params, static_params):
    """Process handover coordination for mining: setup initiation and completion.

    Also expires old pending handovers and reverts timed-out construction sites.
    This runs before do_construction in alem_step, so construction gets clean state.

    Args:
        state: Current environment state.
        block_positions: Per-agent target coordinates.
        is_acting: Mask of agents attempting a handover-capable action.
        params: Gameplay parameters controlling coordination.
        static_params: Static parameters containing player and table sizes.

    Returns:
        succeeds: Boolean array of which agents' actions succeed
        new_state: Updated state with modified pending_handovers and metrics
    """
    if not params.coordination_enabled:
        return is_acting, state

    coord_type = state.coordination_map[
        state.player_level, block_positions[:, 0], block_positions[:, 1]
    ]
    is_handover = coord_type < 0
    window_size = jnp.abs(coord_type)

    # --- Expire old pending handovers ---
    raw_pending = state.pending_handovers
    is_expired = (raw_pending[:, 0] == 1) & (raw_pending[:, 3] <= state.timestep)
    expired_build_type = raw_pending[:, 5]  # 0=mining, 1=shelter, 2=forge, 3=beacon
    expired_construction_handovers = is_expired & (expired_build_type > 0)
    expired_mining_handovers = is_expired & (expired_build_type == 0)

    # Revert expired construction handovers: CONSTRUCTION_IN_PROGRESS -> CONSTRUCTION_SITE
    # (vectorized -- one scatter instead of scanning over all pending slots)
    expired_pos_x = raw_pending[:, 1]
    expired_pos_y = raw_pending[:, 2]
    current_blocks = state.map[state.player_level, expired_pos_x, expired_pos_y]
    should_revert = is_expired & (current_blocks == BlockType.CONSTRUCTION_IN_PROGRESS.value)
    reverted_blocks = jnp.where(should_revert, BlockType.CONSTRUCTION_SITE.value, current_blocks)
    reverted_map = (
        state.map[state.player_level].at[expired_pos_x, expired_pos_y].set(reverted_blocks)
    )
    state = state.replace(map=state.map.at[state.player_level].set(reverted_map))

    # Refund materials to initiator for expired construction handovers
    expired_initiator = raw_pending[:, 4]
    is_expired_shelter = is_expired & (expired_build_type == 1)
    is_expired_forge = is_expired & (expired_build_type == 2)
    is_expired_beacon = is_expired & (expired_build_type == 3)

    wood_refund = jnp.zeros(static_params.player_count, dtype=state.inventory.wood.dtype)
    wood_refund = wood_refund.at[expired_initiator].add(SHELTER_COST_WOOD * is_expired_shelter)
    stone_refund = jnp.zeros(static_params.player_count, dtype=state.inventory.stone.dtype)
    stone_refund = stone_refund.at[expired_initiator].add(
        SHELTER_COST_STONE * is_expired_shelter + FORGE_COST_STONE * is_expired_forge
    )
    iron_refund = jnp.zeros(static_params.player_count, dtype=state.inventory.iron.dtype)
    iron_refund = iron_refund.at[expired_initiator].add(
        FORGE_COST_IRON * is_expired_forge + BEACON_COST_IRON * is_expired_beacon
    )
    coal_refund = jnp.zeros(static_params.player_count, dtype=state.inventory.coal.dtype)
    coal_refund = coal_refund.at[expired_initiator].add(
        FORGE_COST_COAL * is_expired_forge + BEACON_COST_COAL * is_expired_beacon
    )

    state = state.replace(
        inventory=state.inventory.replace(
            wood=state.inventory.wood + wood_refund,
            stone=state.inventory.stone + stone_refund,
            iron=state.inventory.iron + iron_refund,
            coal=state.inventory.coal + coal_refund,
        )
    )

    pending = jnp.where(is_expired[:, None], 0, raw_pending)

    # --- Find matching pending handovers ---
    has_match, match_idx = find_pending_matches(
        pending, block_positions, state.timestep, static_params
    )

    # Exclude construction blocks -- those are handled by do_construction
    target_blocks = state.map[state.player_level, block_positions[:, 0], block_positions[:, 1]]
    is_construction_block = (target_blocks == BlockType.CONSTRUCTION_SITE.value) | (
        target_blocks == BlockType.CONSTRUCTION_IN_PROGRESS.value
    )

    is_completing_raw = is_acting & is_handover & has_match & ~is_construction_block

    # Deduplicate: only the first agent (by index) completing each pending slot gets credit.
    # This prevents multiple agents from receiving independent rewards for the same handover.
    # For each slot, find the minimum completing agent index; others are excluded.
    agent_ids = jnp.arange(static_params.player_count)
    slot_ids = jnp.where(
        is_completing_raw, match_idx, pending.shape[0]
    )  # non-completers get invalid slot
    first_agent_per_slot = jnp.full(pending.shape[0], static_params.player_count, dtype=jnp.int32)
    first_agent_per_slot = first_agent_per_slot.at[slot_ids].min(agent_ids)
    is_completing = is_completing_raw & (first_agent_per_slot[match_idx] == agent_ids)

    is_setting_up = is_acting & is_handover & ~has_match & ~is_construction_block

    # --- Update pending array ---
    cleared = clear_completed_handovers(pending, is_completing, match_idx, static_params)
    new_pending, handover_setups = add_pending_handovers(
        cleared, block_positions, is_setting_up, window_size, state.timestep, static_params
    )

    # For handover blocks: setup always "fails" (no immediate effect), completion succeeds
    succeeds = jnp.where(is_handover, is_completing, is_acting)

    # --- Achievements & metrics ---
    completed_slots = (
        jnp.zeros(pending.shape[0], dtype=jnp.int32)
        .at[match_idx]
        .add(is_completing.astype(jnp.int32))
    )
    handover_successes = (completed_slots > 0).sum()

    achievements = state.achievements
    achievements = achievements.at[:, Achievement.HANDOVER_COMPLETE.value].set(
        jnp.logical_or(achievements[:, Achievement.HANDOVER_COMPLETE.value], is_completing)
    )
    achievements = achievements.at[:, Achievement.COORD_MINE_HANDOVER.value].set(
        jnp.logical_or(achievements[:, Achievement.COORD_MINE_HANDOVER.value], is_completing)
    )

    # Clear coordination_map at completed positions
    is_any_completing = (
        jnp.expand_dims(is_completing, 1)
        & (block_positions[:, None] == block_positions[None, :]).all(axis=2)
    ).any(axis=0)
    new_coord_map = state.coordination_map.at[
        state.player_level, block_positions[:, 0], block_positions[:, 1]
    ].set(
        jnp.where(
            is_any_completing,
            0,
            state.coordination_map[
                state.player_level, block_positions[:, 0], block_positions[:, 1]
            ],
        )
    )
    new_soft_mask = state.soft_coordination_mask.at[
        state.player_level, block_positions[:, 0], block_positions[:, 1]
    ].set(
        jnp.where(
            is_any_completing,
            False,
            state.soft_coordination_mask[
                state.player_level, block_positions[:, 0], block_positions[:, 1]
            ],
        )
    )

    new_state = state.replace(
        pending_handovers=new_pending,
        handover_successes=state.handover_successes + handover_successes,
        handover_setups=state.handover_setups + handover_setups,
        handover_expiries=state.handover_expiries
        + expired_mining_handovers.sum()
        + expired_construction_handovers.sum(),
        coord_mine_handover_count=state.coord_mine_handover_count + handover_successes,
        coord_mine_handover_expiries=state.coord_mine_handover_expiries
        + expired_mining_handovers.sum(),
        coord_construction_handover_expiries=state.coord_construction_handover_expiries
        + expired_construction_handovers.sum(),
        achievements=achievements,
        coordination_map=new_coord_map,
        soft_coordination_mask=new_soft_mask,
    )

    return succeeds, new_state


def do_construction(rng, state, actions, params, static_params):
    """Handle building at construction sites.

    Sync sites: soft → any agent can build solo; hard → require N agents simultaneously.
    Handover sites: initiator pays materials upfront, completer finishes for free
    within a time window. If no one completes, site reverts and materials are refunded.

    Material costs (see constants.py for values):
    - Shelter: SHELTER_COST_WOOD wood, SHELTER_COST_STONE stone
    - Forge: FORGE_COST_STONE stone, FORGE_COST_IRON iron, FORGE_COST_COAL coal
    - Beacon: BEACON_COST_IRON iron, BEACON_COST_COAL coal

    Effects when built:
    - Shelter: +50% rest energy regen for all agents
    - Forge: enables diamond crafting at this location
    - Beacon: expand light_map in radius around beacon

    Args:
        rng: JAX random key used for soft coordination outcomes.
        state: Current environment state.
        actions: Per-player action identifiers.
        params: Gameplay and construction parameters.
        static_params: Static player, map, and handover limits.

    Returns:
        State with construction sites, materials, structures, and metrics updated.
    """
    if not params.construction_enabled:
        return state

    # Check for BUILD actions
    is_building_shelter = actions == Action.BUILD_SHELTER.value
    is_building_forge = actions == Action.BUILD_FORGE.value
    is_building_beacon = actions == Action.BUILD_BEACON.value
    is_building = is_building_shelter | is_building_forge | is_building_beacon

    # Get target block position for each agent
    block_position = state.player_position + DIRECTIONS[state.player_direction]
    is_building = is_building & in_bounds(block_position, static_params)

    # Check if agents are at construction sites or in-progress sites
    block_types = state.map[state.player_level, block_position[:, 0], block_position[:, 1]]
    is_at_construction_site = block_types == BlockType.CONSTRUCTION_SITE.value
    is_at_in_progress = block_types == BlockType.CONSTRUCTION_IN_PROGRESS.value

    intended_building_type = (
        1 * is_building_shelter.astype(jnp.int32)
        + 2 * is_building_forge.astype(jnp.int32)
        + 3 * is_building_beacon.astype(jnp.int32)
    )

    # Check material requirements per agent
    has_shelter_materials = (state.inventory.wood >= SHELTER_COST_WOOD) & (
        state.inventory.stone >= SHELTER_COST_STONE
    )
    has_forge_materials = (
        (state.inventory.stone >= FORGE_COST_STONE)
        & (state.inventory.iron >= FORGE_COST_IRON)
        & (state.inventory.coal >= FORGE_COST_COAL)
    )
    has_beacon_materials = (state.inventory.iron >= BEACON_COST_IRON) & (
        state.inventory.coal >= BEACON_COST_COAL
    )

    can_pay_shelter = is_building_shelter & has_shelter_materials
    can_pay_forge = is_building_forge & has_forge_materials
    can_pay_beacon = is_building_beacon & has_beacon_materials
    can_pay = can_pay_shelter | can_pay_forge | can_pay_beacon

    # Check which agents are at same position (for sync coordination)
    equal_block_placement = (
        jnp.expand_dims(block_position, axis=1) == jnp.expand_dims(block_position, axis=0)
    ).all(axis=2)

    # Count agents building the SAME structure type at the same position
    # Each agent's building type: 1=shelter, 2=forge, 3=beacon, 0=none
    agent_build_type = intended_building_type
    same_build_type = jnp.expand_dims(agent_build_type, axis=1) == jnp.expand_dims(
        agent_build_type, axis=0
    )
    builders_at_same_site = equal_block_placement & is_building[:, None] & same_build_type
    agents_building_here = builders_at_same_site.sum(axis=0)
    payer_present = (builders_at_same_site & can_pay[:, None]).any(axis=0)

    # Get coordination value at each position from the construction site's coordination map
    coord_value = state.coordination_map[
        state.player_level, block_position[:, 0], block_position[:, 1]
    ]
    is_sync = coord_value > 0
    is_handover = coord_value < 0
    required_agents = jnp.abs(coord_value)

    # Check soft coordination mask — soft sync sites are always buildable
    is_soft = state.soft_coordination_mask[
        state.player_level, block_position[:, 0], block_position[:, 1]
    ]

    # SYNC COORDINATION: 2+ agents build together
    coord_met = agents_building_here >= required_agents
    # Hard sync: must meet agent count. Soft sync: always succeeds.
    sync_succeeds = (
        is_at_construction_site & is_building & is_sync & payer_present & (coord_met | is_soft)
    )

    # Handover coordination: initiator pays materials upfront, completer finishes free (refunded on expiry)
    pending = state.pending_handovers
    has_match, match_idx = find_pending_matches(
        pending, block_position, state.timestep, static_params
    )

    # Deduplicate setups: if multiple agents target the same site, only the first one (by index) initiates
    earlier_agent_at_same_pos = (
        jnp.tril(
            jnp.ones((static_params.player_count, static_params.player_count), dtype=jnp.bool_),
            k=-1,
        )
        & equal_block_placement
    ).any(axis=1)

    # Completing: agent at IN_PROGRESS site with matching pending entry AND
    # the same structure type that the initiator started (no materials needed).
    pending_build_type = jnp.where(has_match, pending[match_idx, 5], 0)
    matches_pending_type = pending_build_type == intended_building_type
    is_completing_raw = is_building & is_at_in_progress & has_match & matches_pending_type

    # Deduplicate completion
    agent_ids = jnp.arange(static_params.player_count)
    slot_ids = jnp.where(is_completing_raw, match_idx, pending.shape[0])
    first_agent_per_slot = jnp.full(pending.shape[0], static_params.player_count, dtype=jnp.int32)
    first_agent_per_slot = first_agent_per_slot.at[slot_ids].min(agent_ids)
    is_completing = is_completing_raw & (first_agent_per_slot[match_idx] == agent_ids)

    handover_window = jnp.abs(coord_value)
    # Setup: initiator at construction site with handover, HAS materials (pays upfront, refunded on expiry)
    # Deduplicate setups to prevent double-charging materials
    is_setting_up = (
        is_building
        & is_at_construction_site
        & is_handover
        & can_pay
        & ~has_match
        & ~earlier_agent_at_same_pos
    )

    # Determine which agents succeed (build completes = structure placed)
    build_succeeds = sync_succeeds | is_completing
    # Coordinated builds get achievements
    is_coordinated_build = (sync_succeeds & coord_met) | is_completing

    building_type = intended_building_type * build_succeeds.astype(jnp.int32)

    # Apply build: update map (vectorized)
    new_map = state.map[state.player_level]

    # Determine target block type per agent
    _build_block = jnp.where(
        building_type == 1,
        BlockType.EPIC_SHELTER.value,
        jnp.where(
            building_type == 2,
            BlockType.EPIC_FORGE.value,
            jnp.where(
                building_type == 3,
                BlockType.EPIC_BEACON.value,
                new_map[block_position[:, 0], block_position[:, 1]],
            ),
        ),
    )

    # Per-position build type: propagate via max over same-position agents
    pos_build_type = (
        (equal_block_placement & build_succeeds[None, :]) * building_type[None, :]
    ).max(axis=1)

    built_block = jnp.where(
        pos_build_type == 1,
        BlockType.EPIC_SHELTER.value,
        jnp.where(
            pos_build_type == 2,
            BlockType.EPIC_FORGE.value,
            jnp.where(
                pos_build_type == 3,
                BlockType.EPIC_BEACON.value,
                new_map[block_position[:, 0], block_position[:, 1]],
            ),
        ),
    )

    is_any_building = (equal_block_placement & build_succeeds[:, None]).any(axis=0)
    is_any_setting_up = (equal_block_placement & is_setting_up[:, None]).any(axis=0)

    # Build and setup are mutually exclusive per position
    final_block = jnp.where(
        is_any_building,
        built_block,
        jnp.where(
            is_any_setting_up,
            BlockType.CONSTRUCTION_IN_PROGRESS.value,
            new_map[block_position[:, 0], block_position[:, 1]],
        ),
    )
    new_map = new_map.at[block_position[:, 0], block_position[:, 1]].set(final_block)

    # Material charging:
    # - Sync builds: first builder per position pays (avoid double-charging)
    # - Handover setup: initiator pays upfront (refunded on expiry)
    # - Handover completion: completer pays nothing
    n = static_params.player_count
    earlier_builder = (
        jnp.tril(jnp.ones((n, n), dtype=jnp.bool_), k=-1)
        & equal_block_placement
        & same_build_type
        & (sync_succeeds & can_pay)[None, :]
    ).any(axis=1)
    is_first_sync_builder = sync_succeeds & can_pay & ~earlier_builder & ~is_completing

    # Initiator building type for material deduction
    setup_building_type = intended_building_type

    # Who pays: first sync builder OR handover initiator
    pays_materials = is_first_sync_builder | is_setting_up
    charge_type = jnp.where(is_setting_up, setup_building_type, building_type)

    is_shelter_charge = (charge_type == 1) & pays_materials
    is_forge_charge = (charge_type == 2) & pays_materials
    is_beacon_charge = (charge_type == 3) & pays_materials

    new_inventory = state.inventory.replace(
        wood=state.inventory.wood - SHELTER_COST_WOOD * is_shelter_charge,
        stone=state.inventory.stone
        - SHELTER_COST_STONE * is_shelter_charge
        - FORGE_COST_STONE * is_forge_charge,
        iron=state.inventory.iron
        - FORGE_COST_IRON * is_forge_charge
        - BEACON_COST_IRON * is_beacon_charge,
        coal=state.inventory.coal
        - FORGE_COST_COAL * is_forge_charge
        - BEACON_COST_COAL * is_beacon_charge,
    )

    # Beacon effect: expand light_map in radius around beacon
    new_light_map = state.light_map[state.player_level]
    light_pad = 6
    padded_light = jnp.pad(new_light_map, (light_pad, light_pad), constant_values=0.0)

    def _apply_beacon_light(padded, agent_idx):
        pos = block_position[agent_idx]
        is_beacon = (building_type[agent_idx] == 3) & build_succeeds[agent_idx]
        current_region = jax.lax.dynamic_slice(
            padded,
            pos - jnp.array([4, 4]) + jnp.array([light_pad, light_pad]),
            (9, 9),
        )
        lit_region = jnp.clip(BEACON_LIGHT_MAP + current_region, 0.0, 1.0)
        new_region = lit_region * is_beacon + current_region * (1 - is_beacon)
        padded = jax.lax.dynamic_update_slice(
            padded,
            new_region,
            pos - jnp.array([4, 4]) + jnp.array([light_pad, light_pad]),
        )
        return padded, None

    padded_light, _ = jax.lax.scan(
        _apply_beacon_light, padded_light, jnp.arange(static_params.player_count)
    )
    new_light_map = padded_light[light_pad:-light_pad, light_pad:-light_pad]

    # Update pending handovers
    cleared = clear_completed_handovers(pending, is_completing, match_idx, static_params)
    new_pending, handover_setups_count = add_pending_handovers(
        cleared,
        block_position,
        is_setting_up,
        handover_window,
        state.timestep,
        static_params,
        building_type=setup_building_type,
    )

    # Metrics
    completed_slots = (
        jnp.zeros(pending.shape[0], dtype=jnp.int32)
        .at[match_idx]
        .add(is_completing.astype(jnp.int32))
    )
    handover_completions = (completed_slots > 0).sum()

    shelters_built = (building_type == 1).any().astype(jnp.int32)
    forges_built = (building_type == 2).any().astype(jnp.int32)
    beacons_built = (building_type == 3).any().astype(jnp.int32)

    # Achievements
    # Coordination achievements only for coordinated builds (enough agents met threshold)
    agent_built_shelter = is_coordinated_build & (intended_building_type == 1)
    agent_built_forge = is_coordinated_build & (intended_building_type == 2)
    agent_built_beacon = is_coordinated_build & (intended_building_type == 3)

    achievements = state.achievements
    achievements = set_achievement(
        achievements, Achievement.COORD_BUILD_SHELTER, agent_built_shelter
    )
    achievements = set_achievement(achievements, Achievement.COORD_BUILD_FORGE, agent_built_forge)
    achievements = set_achievement(achievements, Achievement.COORD_BUILD_BEACON, agent_built_beacon)
    # Award HANDOVER_COMPLETE to both completing and setup agents
    initiator_ids = pending[match_idx, 4]
    setup_got_complete = jnp.zeros(static_params.player_count, dtype=jnp.bool_)
    setup_got_complete = setup_got_complete.at[initiator_ids].max(is_completing)
    achievements = set_achievement(
        achievements, Achievement.HANDOVER_COMPLETE, is_completing | setup_got_complete
    )

    # Clear coordination_map at positions where construction succeeded
    new_coord_map = state.coordination_map.at[
        state.player_level, block_position[:, 0], block_position[:, 1]
    ].set(
        jnp.where(
            is_any_building,
            0,
            state.coordination_map[state.player_level, block_position[:, 0], block_position[:, 1]],
        )
    )
    new_soft_mask = state.soft_coordination_mask.at[
        state.player_level, block_position[:, 0], block_position[:, 1]
    ].set(
        jnp.where(
            is_any_building,
            False,
            state.soft_coordination_mask[
                state.player_level, block_position[:, 0], block_position[:, 1]
            ],
        )
    )

    # Update construction_sites_built
    site_positions = state.construction_site_positions[state.player_level]
    new_sites_built = state.construction_sites_built

    def _update_site_built(sites_built, agent_idx):
        pos = block_position[agent_idx]
        b_type = building_type[agent_idx]
        succeeds = build_succeeds[agent_idx]
        # Find matching site index
        matches = (site_positions[:, 0] == pos[0]) & (site_positions[:, 1] == pos[1])
        site_idx = jnp.argmax(matches)
        has_match = matches.any()
        sites_built = sites_built.at[state.player_level, site_idx].set(
            jnp.where(succeeds & has_match, b_type, sites_built[state.player_level, site_idx])
        )
        return sites_built, None

    new_sites_built, _ = jax.lax.scan(
        _update_site_built, new_sites_built, jnp.arange(static_params.player_count)
    )

    # Update state
    new_whole_map = state.map.at[state.player_level].set(new_map)
    new_whole_light_map = state.light_map.at[state.player_level].set(new_light_map)

    state = state.replace(
        map=new_whole_map,
        light_map=new_whole_light_map,
        inventory=new_inventory,
        pending_handovers=new_pending,
        handover_successes=state.handover_successes + handover_completions,
        handover_setups=state.handover_setups + handover_setups_count,
        # Construction coordination tracking:
        # - attempts count sync-site build tries where at least one participant can pay
        # - successes count completed sync-site builds, including soft solo builds
        coord_construction_attempts=state.coord_construction_attempts
        + (
            (is_building & is_at_construction_site & is_sync & payer_present)
            .any()
            .astype(jnp.int32)
        ),
        coord_construction_successes=state.coord_construction_successes
        + (sync_succeeds.any().astype(jnp.int32)),
        coord_construction_handover_count=state.coord_construction_handover_count
        + handover_completions,
        coord_construction_handover_setups=state.coord_construction_handover_setups
        + handover_setups_count,
        # Diagnostic: BUILD_X at a CONSTRUCTION_SITE with no materials on tile.
        # Funded attempts already register as coord_construction_attempts.
        construction_build_at_site_unfunded=state.construction_build_at_site_unfunded
        + (is_building & is_at_construction_site & ~payer_present).any().astype(jnp.int32),
        coord_build_shelter_count=state.coord_build_shelter_count + shelters_built,
        coord_build_forge_count=state.coord_build_forge_count + forges_built,
        coord_build_beacon_count=state.coord_build_beacon_count + beacons_built,
        achievements=achievements,
        coordination_map=new_coord_map,
        soft_coordination_mask=new_soft_mask,
        construction_sites_built=new_sites_built,
    )

    return state


def update_mobs(rng, state, params, env_params, static_params):
    """Advance melee, passive, ranged, and projectile entities one tick.

    Args:
        rng: JAX random key used for movement and attacks.
        state: Current environment state.
        params: Dynamic movement and combat parameters.
        env_params: Gameplay parameters controlling damage and behavior.
        static_params: Static entity limits and map parameters.

    Returns:
        State after all non-player entities have acted.
    """

    # Extract lightweight MobScanState to avoid copying full EnvState
    # coordination/metric arrays) through 42+ scan iterations per step.
    mob_state = MobScanState(
        map=state.map,
        mob_map=state.mob_map,
        player_level=state.player_level,
        player_position=state.player_position,
        player_alive=state.player_alive,
        player_health=state.player_health,
        is_sleeping=state.is_sleeping,
        is_resting=state.is_resting,
        achievements=state.achievements,
        melee_mobs=state.melee_mobs,
        passive_mobs=state.passive_mobs,
        ranged_mobs=state.ranged_mobs,
        mob_projectiles=state.mob_projectiles,
        mob_projectile_directions=state.mob_projectile_directions,
        mob_projectile_owners=state.mob_projectile_owners,
        player_projectiles=state.player_projectiles,
        player_projectile_directions=state.player_projectile_directions,
        player_projectile_owners=state.player_projectile_owners,
        monsters_killed=state.monsters_killed,
        inventory=InventorySlice(armour=state.inventory.armour),
        armour_enchantments=state.armour_enchantments,
        bow_enchantment=state.bow_enchantment,
        player_dexterity=state.player_dexterity,
        player_intelligence=state.player_intelligence,
        player_specialization=state.player_specialization,
        player_food=state.player_food,
        player_hunger=state.player_hunger,
    )

    # Move melee_mobs
    def _move_melee_mob(rng_and_state, melee_mob_index):
        rng, state = rng_and_state
        melee_mobs = state.melee_mobs

        # Random move
        rng, _rng = jax.random.split(rng)
        valid_random_moves = in_bounds(
            DIRECTIONS[1:5] + melee_mobs.position[state.player_level, melee_mob_index],
            static_params,
        )
        random_move_direction = jax.random.choice(_rng, DIRECTIONS[1:5], p=valid_random_moves)
        random_move_proposed_position = (
            melee_mobs.position[state.player_level, melee_mob_index] + random_move_direction
        )

        # Move towards closest player
        player_move_direction = jnp.zeros((2,), dtype=jnp.int32)
        all_players_move_direction_abs = jnp.abs(
            state.player_position - melee_mobs.position[state.player_level, melee_mob_index]
        )
        distance_to_players = all_players_move_direction_abs.sum(axis=1)
        player_targetted = jnp.argmin(jnp.where(state.player_alive, distance_to_players, jnp.inf))
        player_move_direction_abs = all_players_move_direction_abs[player_targetted]

        is_max_dist = player_move_direction_abs == player_move_direction_abs.max()
        player_move_direction_index_p = is_max_dist / jnp.maximum(is_max_dist.sum(), 1.0)
        rng, _rng = jax.random.split(rng)
        player_move_direction_index = jax.random.choice(
            _rng,
            jnp.arange(2),
            p=player_move_direction_index_p,
        )

        player_move_direction = player_move_direction.at[player_move_direction_index].set(
            jnp.sign(
                state.player_position[player_targetted, player_move_direction_index]
                - melee_mobs.position[
                    state.player_level, melee_mob_index, player_move_direction_index
                ]
            ).astype(jnp.int32)
        )
        player_move_proposed_position = (
            melee_mobs.position[state.player_level, melee_mob_index] + player_move_direction
        )

        # Choose movement
        close_to_player = distance_to_players < 10
        close_to_player = jnp.logical_and(close_to_player, state.player_alive).any()
        close_to_player = jnp.logical_or(close_to_player, is_fighting_boss(state, static_params))
        rng, _rng = jax.random.split(rng)
        close_to_player = jnp.logical_and(close_to_player, jax.random.uniform(_rng) < 0.75)
        proposed_position = jax.lax.select(
            close_to_player,
            player_move_proposed_position,
            random_move_proposed_position,
        )

        # Choose attack or not
        is_attacking_player = distance_to_players == 1
        is_attacking_player = jnp.logical_and(is_attacking_player, state.player_alive)
        is_attacking_player = jnp.logical_and(
            is_attacking_player,
            melee_mobs.attack_cooldown[state.player_level, melee_mob_index] <= 0,
        )
        is_attacking_player = jnp.logical_and(
            is_attacking_player, melee_mobs.mask[state.player_level, melee_mob_index]
        )

        proposed_position = jax.lax.select(
            is_attacking_player.any(),
            melee_mobs.position[state.player_level, melee_mob_index],
            proposed_position,
        )

        melee_mob_base_damage = MOB_TYPE_DAMAGE_MAPPING[
            melee_mobs.type_id[state.player_level, melee_mob_index], MobType.MELEE.value
        ]

        melee_mob_damage = get_damage_done_to_player(
            state, static_params, melee_mob_base_damage * (1 + 2.5 * state.is_sleeping[:, None])
        )

        new_cooldown = jnp.where(
            is_attacking_player.any(),
            5,
            melee_mobs.attack_cooldown[state.player_level, melee_mob_index] - 1,
        )

        is_waking_player = jnp.logical_and(state.is_sleeping, is_attacking_player)

        state = state.replace(
            player_health=state.player_health - melee_mob_damage * is_attacking_player,
            is_sleeping=jnp.logical_and(state.is_sleeping, jnp.logical_not(is_attacking_player)),
            is_resting=jnp.logical_and(state.is_resting, jnp.logical_not(is_attacking_player)),
            achievements=state.achievements.at[:, Achievement.WAKE_UP.value].set(
                jnp.logical_or(state.achievements[:, Achievement.WAKE_UP.value], is_waking_player)
            ),
        )

        mob_type = melee_mobs.type_id[state.player_level, melee_mob_index]
        collision_map = MOB_TYPE_COLLISION_MAPPING[mob_type, 1]
        valid_move = is_position_in_bounds_not_in_mob_not_colliding(
            state, proposed_position[None, :], collision_map, static_params
        )[0]
        in_other_player = is_in_other_player(state, proposed_position[None, :])[0]
        valid_move = jnp.logical_and(valid_move, jnp.logical_not(in_other_player))

        position = jax.lax.select(
            valid_move,
            proposed_position,
            melee_mobs.position[state.player_level, melee_mob_index],
        )

        should_not_despawn = distance_to_players < params.mob_despawn_distance
        should_not_despawn = jnp.logical_and(should_not_despawn, state.player_alive).any()
        should_not_despawn = jnp.logical_or(
            should_not_despawn, is_fighting_boss(state, static_params)
        )

        rng, _rng = jax.random.split(rng)

        # Clear our old entry if we are alive
        new_mob_map = state.mob_map.at[
            state.player_level,
            state.melee_mobs.position[state.player_level, melee_mob_index, 0],
            state.melee_mobs.position[state.player_level, melee_mob_index, 1],
        ].set(
            jnp.logical_and(
                state.mob_map[
                    state.player_level,
                    state.melee_mobs.position[state.player_level, melee_mob_index, 0],
                    state.melee_mobs.position[state.player_level, melee_mob_index, 1],
                ],
                jnp.logical_not(melee_mobs.mask[state.player_level, melee_mob_index]),
            )
        )
        new_mask = jnp.logical_and(
            state.melee_mobs.mask[state.player_level, melee_mob_index],
            should_not_despawn,
        )
        # Enter new entry if we are alive and not despawning this timestep
        new_mob_map = new_mob_map.at[state.player_level, position[0], position[1]].set(
            jnp.logical_or(new_mob_map[state.player_level, position[0], position[1]], new_mask)
        )

        state = state.replace(
            melee_mobs=state.melee_mobs.replace(
                position=state.melee_mobs.position.at[state.player_level, melee_mob_index].set(
                    position
                ),
                attack_cooldown=state.melee_mobs.attack_cooldown.at[
                    state.player_level, melee_mob_index
                ].set(new_cooldown),
                mask=state.melee_mobs.mask.at[state.player_level, melee_mob_index].set(new_mask),
            ),
            mob_map=new_mob_map,
        )

        return (_rng, state), None

    rng, _rng = jax.random.split(rng)
    (rng, mob_state), _ = jax.lax.scan(
        _move_melee_mob,
        (rng, mob_state),
        jnp.arange(static_params.max_melee_mobs * static_params.player_count),
    )

    # Move passive_mobs
    def _move_passive_mob(rng_and_state, passive_mob_index):
        rng, state = rng_and_state
        passive_mobs = state.passive_mobs

        # Random move
        rng, _rng = jax.random.split(rng)
        valid_random_moves = in_bounds(
            DIRECTIONS[1:9] + passive_mobs.position[state.player_level, passive_mob_index],
            static_params,
        )
        random_move_direction = jax.random.choice(
            _rng,
            DIRECTIONS[1:9],  # 50% chance of not moving
            p=valid_random_moves,
        )
        proposed_position = (
            passive_mobs.position[state.player_level, passive_mob_index] + random_move_direction
        )

        mob_type = passive_mobs.type_id[state.player_level, passive_mob_index]
        collision_map = MOB_TYPE_COLLISION_MAPPING[mob_type, 0]
        valid_move = is_position_in_bounds_not_in_mob_not_colliding(
            state, proposed_position[None, :], collision_map, static_params
        )[0]
        in_other_player = is_in_other_player(state, proposed_position[None, :])[0]
        valid_move = jnp.logical_and(valid_move, jnp.logical_not(in_other_player))
        position = jax.lax.select(
            valid_move,
            proposed_position,
            passive_mobs.position[state.player_level, passive_mob_index],
        )

        distance_to_players = jnp.abs(
            state.player_position - passive_mobs.position[state.player_level, passive_mob_index]
        ).sum(axis=1)
        should_not_despawn = jnp.logical_and(
            distance_to_players < params.mob_despawn_distance, state.player_alive
        ).any()

        # Clear our old entry if we are alive
        new_mob_map = state.mob_map.at[
            state.player_level,
            state.passive_mobs.position[state.player_level, passive_mob_index, 0],
            state.passive_mobs.position[state.player_level, passive_mob_index, 1],
        ].set(
            jnp.logical_and(
                state.mob_map[
                    state.player_level,
                    state.passive_mobs.position[state.player_level, passive_mob_index, 0],
                    state.passive_mobs.position[state.player_level, passive_mob_index, 1],
                ],
                jnp.logical_not(passive_mobs.mask[state.player_level, passive_mob_index]),
            )
        )
        new_mask = jnp.logical_and(
            state.passive_mobs.mask[state.player_level, passive_mob_index],
            should_not_despawn,
        )
        # Enter new entry if we are alive and not despawning this timestep
        new_mob_map = new_mob_map.at[state.player_level, position[0], position[1]].set(
            jnp.logical_or(new_mob_map[state.player_level, position[0], position[1]], new_mask)
        )

        state = state.replace(
            passive_mobs=state.passive_mobs.replace(
                position=state.passive_mobs.position.at[state.player_level, passive_mob_index].set(
                    position
                ),
                mask=state.passive_mobs.mask.at[state.player_level, passive_mob_index].set(
                    jnp.logical_and(
                        state.passive_mobs.mask[state.player_level, passive_mob_index],
                        should_not_despawn,
                    )
                ),
            ),
            mob_map=new_mob_map,
        )

        return (rng, state), None

    rng, _rng = jax.random.split(rng)
    (rng, mob_state), _ = jax.lax.scan(
        _move_passive_mob,
        (rng, mob_state),
        jnp.arange(static_params.max_passive_mobs * static_params.player_count),
    )

    # Move ranged_mobs

    def _move_ranged_mob(rng_and_state, ranged_mob_index):
        rng, state = rng_and_state
        ranged_mobs = state.ranged_mobs

        # Random move
        rng, _rng = jax.random.split(rng)
        valid_random_moves = in_bounds(
            DIRECTIONS[1:5] + ranged_mobs.position[state.player_level, ranged_mob_index],
            static_params,
        )
        random_move_direction = jax.random.choice(_rng, DIRECTIONS[1:5], p=valid_random_moves)
        random_move_proposed_position = (
            ranged_mobs.position[state.player_level, ranged_mob_index] + random_move_direction
        )

        # Move towards closest player
        player_move_direction = jnp.zeros((2,), dtype=jnp.int32)
        all_players_move_direction_abs = jnp.abs(
            state.player_position - ranged_mobs.position[state.player_level, ranged_mob_index]
        )
        distance_to_players = all_players_move_direction_abs.sum(axis=1)
        player_targetted = jnp.argmin(jnp.where(state.player_alive, distance_to_players, jnp.inf))
        player_move_direction_abs = all_players_move_direction_abs[player_targetted]
        is_max_dist = player_move_direction_abs == player_move_direction_abs.max()
        player_move_direction_index_p = is_max_dist / jnp.maximum(is_max_dist.sum(), 1.0)
        rng, _rng = jax.random.split(rng)
        player_move_direction_index = jax.random.choice(
            _rng,
            jnp.arange(2),
            p=player_move_direction_index_p,
        )

        player_move_direction = player_move_direction.at[player_move_direction_index].set(
            jnp.sign(
                state.player_position[player_targetted, player_move_direction_index]
                - ranged_mobs.position[
                    state.player_level, ranged_mob_index, player_move_direction_index
                ]
            ).astype(jnp.int32)
        )
        player_move_towards_proposed_position = (
            ranged_mobs.position[state.player_level, ranged_mob_index] + player_move_direction
        )
        player_move_away_proposed_position = (
            ranged_mobs.position[state.player_level, ranged_mob_index] - player_move_direction
        )

        # Choose movement
        far_from_player = player_move_direction_abs[player_move_direction_index] >= 6
        too_close_to_player = player_move_direction_abs[player_move_direction_index] <= 3

        proposed_position = jax.lax.select(
            far_from_player,
            player_move_towards_proposed_position,
            random_move_proposed_position,
        )
        proposed_position = jax.lax.select(
            too_close_to_player,
            player_move_away_proposed_position,
            proposed_position,
        )

        rng, _rng = jax.random.split(rng)

        proposed_position = jax.lax.select(
            jax.random.uniform(_rng) < 0.15,
            proposed_position,
            random_move_proposed_position,
        )

        # Choose attack or not
        is_attacking_player = jnp.logical_not(far_from_player)
        is_attacking_player = jnp.logical_and(
            is_attacking_player,
            ranged_mobs.attack_cooldown[state.player_level, ranged_mob_index] <= 0,
        )
        is_attacking_player = jnp.logical_and(
            is_attacking_player, ranged_mobs.mask[state.player_level, ranged_mob_index]
        )

        # Spawn projectile
        can_spawn_projectile = (
            state.mob_projectiles.mask[state.player_level].sum()
            < static_params.max_mob_projectiles * static_params.player_count
        )
        new_projectile_position = ranged_mobs.position[state.player_level, ranged_mob_index]

        is_spawning_projectile = jnp.logical_and(is_attacking_player, can_spawn_projectile)

        new_mob_projectiles, new_mob_projectile_directions, new_mob_projectile_owners = (
            spawn_projectile(
                state,
                static_params,
                state.mob_projectiles,
                state.mob_projectile_directions,
                state.mob_projectile_owners,
                new_projectile_position,
                is_spawning_projectile,
                ranged_mob_index,
                player_move_direction,
                RANGED_MOB_TYPE_TO_PROJECTILE_TYPE_MAPPING[
                    ranged_mobs.type_id[state.player_level, ranged_mob_index]
                ],
            )
        )

        state = state.replace(
            mob_projectiles=new_mob_projectiles,
            mob_projectile_directions=new_mob_projectile_directions,
            mob_projectile_owners=new_mob_projectile_owners,
        )

        proposed_position = jax.lax.select(
            is_attacking_player,
            ranged_mobs.position[state.player_level, ranged_mob_index],
            proposed_position,
        )

        new_cooldown = jax.lax.select(
            is_attacking_player,
            4,
            ranged_mobs.attack_cooldown[state.player_level, ranged_mob_index] - 1,
        )

        mob_type = ranged_mobs.type_id[state.player_level, ranged_mob_index]
        collision_map = MOB_TYPE_COLLISION_MAPPING[mob_type, 2]
        valid_move = is_position_in_bounds_not_in_mob_not_colliding(
            state, proposed_position[None, :], collision_map, static_params
        )[0]
        in_other_player = is_in_other_player(state, proposed_position[None, :])[0]
        valid_move = jnp.logical_and(valid_move, jnp.logical_not(in_other_player))

        position = jax.lax.select(
            valid_move,
            proposed_position,
            ranged_mobs.position[state.player_level, ranged_mob_index],
        )

        should_not_despawn = distance_to_players < params.mob_despawn_distance
        should_not_despawn = jnp.logical_and(should_not_despawn, state.player_alive).any()
        should_not_despawn = jnp.logical_or(
            should_not_despawn, is_fighting_boss(state, static_params)
        )

        # Clear our old entry if we are alive
        new_mob_map = state.mob_map.at[
            state.player_level,
            state.ranged_mobs.position[state.player_level, ranged_mob_index, 0],
            state.ranged_mobs.position[state.player_level, ranged_mob_index, 1],
        ].set(
            jnp.logical_and(
                state.mob_map[
                    state.player_level,
                    state.ranged_mobs.position[state.player_level, ranged_mob_index, 0],
                    state.ranged_mobs.position[state.player_level, ranged_mob_index, 1],
                ],
                jnp.logical_not(ranged_mobs.mask[state.player_level, ranged_mob_index]),
            )
        )
        new_mask = jnp.logical_and(
            state.ranged_mobs.mask[state.player_level, ranged_mob_index],
            should_not_despawn,
        )
        # Enter new entry if we are alive and not despawning this timestep
        new_mob_map = new_mob_map.at[state.player_level, position[0], position[1]].set(
            jnp.logical_or(new_mob_map[state.player_level, position[0], position[1]], new_mask)
        )

        state = state.replace(
            ranged_mobs=state.ranged_mobs.replace(
                position=state.ranged_mobs.position.at[state.player_level, ranged_mob_index].set(
                    position
                ),
                attack_cooldown=state.ranged_mobs.attack_cooldown.at[
                    state.player_level, ranged_mob_index
                ].set(new_cooldown),
                mask=state.ranged_mobs.mask.at[state.player_level, ranged_mob_index].set(
                    jnp.logical_and(
                        state.ranged_mobs.mask[state.player_level, ranged_mob_index],
                        should_not_despawn,
                    )
                ),
            ),
            mob_map=new_mob_map,
        )

        return (rng, state), None

    rng, _rng = jax.random.split(rng)
    (rng, mob_state), _ = jax.lax.scan(
        _move_ranged_mob,
        (rng, mob_state),
        jnp.arange(static_params.max_ranged_mobs * static_params.player_count),
    )

    # Move projectiles
    def _move_mob_projectile(rng_and_state, projectile_index):
        rng, state = rng_and_state
        projectiles = state.mob_projectiles

        proposed_position = (
            projectiles.position[state.player_level, projectile_index]
            + state.mob_projectile_directions[state.player_level, projectile_index]
        )

        proposed_position_in_bounds = in_bounds(proposed_position[None, :], static_params)[0]
        in_wall = is_in_solid_block(state.map[state.player_level], proposed_position[None, :])[0]
        in_wall = jnp.logical_and(
            in_wall,
            jnp.logical_not(
                state.map[state.player_level][proposed_position[0], proposed_position[1]]
                == BlockType.WATER.value
            ),
        )  # Arrows can go over water
        in_mob = is_in_mob(state, proposed_position[None, :])[0]

        continue_move = jnp.logical_and(proposed_position_in_bounds, jnp.logical_not(in_wall))
        continue_move = jnp.logical_and(continue_move, jnp.logical_not(in_mob))

        hit_player0 = jnp.logical_and(
            (
                projectiles.position[state.player_level, projectile_index] == state.player_position
            ).all(axis=1),
            projectiles.mask[state.player_level, projectile_index],
        )

        proposed_position_in_player = (proposed_position == state.player_position).all(axis=1)
        hit_player1 = jnp.logical_and(
            proposed_position_in_player,
            projectiles.mask[state.player_level, projectile_index],
        )
        hit_player = jnp.logical_or(hit_player0, hit_player1)
        hit_player = jnp.logical_and(hit_player, state.player_alive)

        continue_move = jnp.logical_and(continue_move, jnp.logical_not(hit_player.any()))

        position = proposed_position

        # Clear our old entry if we are alive
        new_mask = jnp.logical_and(
            continue_move, projectiles.mask[state.player_level, projectile_index]
        )

        hit_bench_or_furnace = jnp.logical_or(
            state.map[state.player_level, position[0], position[1]] == BlockType.FURNACE.value,
            state.map[state.player_level, position[0], position[1]]
            == BlockType.CRAFTING_TABLE.value,
        )
        removing_block = jnp.logical_and(
            hit_bench_or_furnace, projectiles.mask[state.player_level, projectile_index]
        )

        new_block = jax.lax.select(
            removing_block,
            BlockType.PATH.value,
            state.map[state.player_level, position[0], position[1]],
        )

        projectile_type = state.mob_projectiles.type_id[state.player_level, projectile_index]
        projectile_damage = get_damage_done_to_player(
            state,
            static_params,
            MOB_TYPE_DAMAGE_MAPPING[projectile_type, MobType.PROJECTILE.value][None, :],
        )

        state = state.replace(
            mob_projectiles=state.mob_projectiles.replace(
                position=state.mob_projectiles.position.at[
                    state.player_level, projectile_index
                ].set(position),
                mask=state.mob_projectiles.mask.at[state.player_level, projectile_index].set(
                    new_mask
                ),
            ),
            player_health=state.player_health - projectile_damage * hit_player,
            is_sleeping=jnp.logical_and(state.is_sleeping, jnp.logical_not(hit_player)),
            is_resting=jnp.logical_and(state.is_resting, jnp.logical_not(hit_player)),
            map=state.map.at[state.player_level, position[0], position[1]].set(new_block),
        )

        return (rng, state), None

    rng, _rng = jax.random.split(rng)
    (rng, mob_state), _ = jax.lax.scan(
        _move_mob_projectile,
        (rng, mob_state),
        jnp.arange(static_params.max_mob_projectiles * static_params.player_count),
    )

    def _move_player_projectile(rng_and_state, projectile_index):
        rng, state = rng_and_state
        projectiles = state.player_projectiles

        projectile_owner = state.player_projectile_owners[state.player_level, projectile_index]

        projectile_type = state.player_projectiles.type_id[state.player_level, projectile_index]

        projectile_damage_vector = (
            MOB_TYPE_DAMAGE_MAPPING[projectile_type, MobType.PROJECTILE.value]
            * projectiles.mask[state.player_level, projectile_index]
        )

        is_arrow = jnp.logical_or(
            projectile_type == ProjectileType.ARROW.value,
            projectile_type == ProjectileType.ARROW2.value,
        )

        # Bow enchantment
        arrow_damage_add = jnp.zeros(3, dtype=jnp.float32)
        arrow_damage_add = arrow_damage_add.at[state.bow_enchantment[projectile_owner]].set(
            projectile_damage_vector[0] / 2
        )
        arrow_damage_add = arrow_damage_add.at[0].set(0)

        projectile_damage_vector += jax.lax.select(
            is_arrow,
            arrow_damage_add,
            jnp.zeros(3, dtype=jnp.float32),
        )

        # Apply attribute scaling
        arrow_damage_coeff = 1 + 0.2 * (state.player_dexterity[projectile_owner] - 1)
        magic_damage_coeff = 1 + 0.5 * (state.player_intelligence[projectile_owner] - 1)

        projectile_damage_vector *= jax.lax.select(
            is_arrow,
            arrow_damage_coeff,
            1.0,
        )

        projectile_damage_vector *= jax.lax.select(
            projectile_type == ProjectileType.FIREBALL.value,
            magic_damage_coeff,
            1.0,
        )

        proposed_position = (
            projectiles.position[state.player_level, projectile_index]
            + state.player_projectile_directions[state.player_level, projectile_index]
        )

        proposed_position_in_bounds = in_bounds(proposed_position[None, :], static_params)[0]
        in_wall = is_in_solid_block(state.map[state.player_level], proposed_position[None, :])[0]
        in_wall = jnp.logical_and(
            in_wall,
            jnp.logical_not(
                state.map[state.player_level][proposed_position[0], proposed_position[1]]
                == BlockType.WATER.value
            ),
        )  # Arrows can go over water

        # Check if we hit a player
        deal_damage = projectiles.mask[state.player_level, projectile_index]

        per_player_contact = (state.player_position == proposed_position[None, :]).all(axis=-1)
        did_attack_player = per_player_contact.any()
        player_attack_index = jnp.argmax(per_player_contact)

        player_defense_vector = get_player_defense_vector(state)[player_attack_index]
        player_damage_dealt = (
            get_damage(projectile_damage_vector, player_defense_vector)
            * did_attack_player
            * env_params.friendly_fire
        )
        new_player_health = state.player_health.at[player_attack_index].subtract(
            player_damage_dealt
        )

        state, did_attack_mob0, did_kill_mob0, _, _, _, _, _, _, _, _ = attack_mob(
            state,
            deal_damage,
            projectiles.position[None, state.player_level, projectile_index],
            projectile_damage_vector[None, :],
            jnp.array([False]),
        )
        did_attack_mob0 = did_attack_mob0[0]

        did_attack_mob = jnp.logical_or(did_attack_player, did_attack_mob0)

        projectile_damage_vector = projectile_damage_vector * (1 - did_attack_mob0)

        state, did_attack_mob1, did_kill_mob1, _, _, _, _, _, _, _, _ = attack_mob(
            state,
            deal_damage,
            proposed_position[None, :],
            projectile_damage_vector[None, :],
            jnp.array([False]),
        )
        did_attack_mob1 = did_attack_mob1[0]

        did_attack_mob = jnp.logical_or(did_attack_mob, did_attack_mob1)

        continue_move = jnp.logical_and(proposed_position_in_bounds, jnp.logical_not(in_wall))
        continue_move = jnp.logical_and(continue_move, jnp.logical_not(did_attack_mob))
        position = proposed_position

        # Clear our old entry if we are alive
        new_mask = jnp.logical_and(
            continue_move, projectiles.mask[state.player_level, projectile_index]
        )

        state = state.replace(
            player_health=new_player_health,
            player_projectiles=state.player_projectiles.replace(
                position=state.player_projectiles.position.at[
                    state.player_level, projectile_index
                ].set(position),
                mask=state.player_projectiles.mask.at[state.player_level, projectile_index].set(
                    new_mask
                ),
            ),
        )

        return (rng, state), None

    rng, _rng = jax.random.split(rng)
    (rng, mob_state), _ = jax.lax.scan(
        _move_player_projectile,
        (rng, mob_state),
        jnp.arange(static_params.max_player_projectiles * static_params.player_count),
    )

    # Write back all modified fields from MobScanState to EnvState
    state = state.replace(
        map=mob_state.map,
        mob_map=mob_state.mob_map,
        player_health=mob_state.player_health,
        is_sleeping=mob_state.is_sleeping,
        is_resting=mob_state.is_resting,
        achievements=mob_state.achievements,
        melee_mobs=mob_state.melee_mobs,
        passive_mobs=mob_state.passive_mobs,
        ranged_mobs=mob_state.ranged_mobs,
        mob_projectiles=mob_state.mob_projectiles,
        mob_projectile_directions=mob_state.mob_projectile_directions,
        mob_projectile_owners=mob_state.mob_projectile_owners,
        player_projectiles=mob_state.player_projectiles,
        player_food=mob_state.player_food,
        player_hunger=mob_state.player_hunger,
        monsters_killed=mob_state.monsters_killed,
    )

    return state


def update_player_intrinsics(state, action, static_params):
    """Update hunger, thirst, fatigue, health, mana, and sleep state.

    Args:
        state: Current environment state.
        action: Per-player action identifiers.
        static_params: Static parameters used by boss and player logic.

    Returns:
        State with player intrinsic resources advanced one tick.
    """
    # Start sleeping?
    is_starting_sleep = jnp.logical_and(
        action == Action.SLEEP.value, state.player_energy < get_max_energy(state)
    )
    new_is_sleeping = jnp.logical_or(state.is_sleeping, is_starting_sleep)
    state = state.replace(
        is_sleeping=jnp.where(state.player_alive, new_is_sleeping, state.is_sleeping)
    )

    # Wake up?
    is_waking_up = jnp.logical_and(state.player_energy >= get_max_energy(state), state.is_sleeping)
    new_is_sleeping = jnp.logical_and(state.is_sleeping, jnp.logical_not(is_waking_up))
    new_achievements = state.achievements.at[:, Achievement.WAKE_UP.value].set(
        jnp.logical_or(state.achievements[:, Achievement.WAKE_UP.value], is_waking_up)
    )
    state = state.replace(
        is_sleeping=jnp.where(state.player_alive, new_is_sleeping, state.is_sleeping),
        achievements=jnp.where(state.player_alive[:, None], new_achievements, state.achievements),
    )

    # Start resting?
    is_starting_rest = jnp.logical_and(
        action == Action.REST.value, state.player_health < get_max_health(state)
    )
    new_is_resting = jnp.logical_or(state.is_resting, is_starting_rest)
    state = state.replace(is_resting=new_is_resting)

    # Wake up from resting
    is_waking_up = jnp.logical_and(
        state.is_resting,
        jnp.logical_or(
            state.player_health >= get_max_health(state),
            jnp.logical_or(state.player_food <= 0, state.player_drink <= 0),
        ),
    )
    new_is_resting = jnp.logical_and(state.is_resting, jnp.logical_not(is_waking_up))
    state = state.replace(
        is_resting=jnp.where(state.player_alive, new_is_resting, state.is_resting),
    )

    not_boss = jnp.logical_not(is_fighting_boss(state, static_params))

    intrinsic_decay_coeff = 1.0 - (0.125 * (state.player_dexterity - 1))

    # Hunger
    hunger_add = (
        jnp.where(
            state.is_sleeping,
            0.5,
            1.0,
        )
        * intrinsic_decay_coeff
    )
    new_hunger = state.player_hunger + hunger_add

    hungered_food = jnp.maximum(state.player_food - 1 * not_boss, 0)
    new_food = jnp.where(new_hunger > 25, hungered_food, state.player_food)
    new_hunger = jnp.where(new_hunger > 25, 0.0, new_hunger)

    state = state.replace(
        player_hunger=jnp.where(state.player_alive, new_hunger, state.player_hunger),
        player_food=jnp.where(state.player_alive, new_food, state.player_food),
    )

    # Thirst
    thirst_add = (
        jnp.where(
            state.is_sleeping,
            0.5,
            1.0,
        )
        * intrinsic_decay_coeff
    )
    new_thirst = state.player_thirst + thirst_add
    thirsted_drink = jnp.maximum(state.player_drink - 1 * not_boss, 0)
    new_drink = jnp.where(new_thirst > 20, thirsted_drink, state.player_drink)
    new_thirst = jnp.where(new_thirst > 20, 0.0, new_thirst)

    state = state.replace(
        player_thirst=jnp.where(state.player_alive, new_thirst, state.player_thirst),
        player_drink=jnp.where(state.player_alive, new_drink, state.player_drink),
    )

    # Fatigue
    # Epic Shelter bonus: +50% fatigue recovery rate when sleeping near shelter
    near_shelter = is_near_block(state, BlockType.EPIC_SHELTER.value, static_params)
    fatigue_recovery = jnp.where(near_shelter & state.is_sleeping, -1.5, -1.0)
    new_fatigue = jnp.where(
        state.is_sleeping,
        jnp.minimum(state.player_fatigue + fatigue_recovery, 0),
        state.player_fatigue + intrinsic_decay_coeff,
    )

    new_energy = jnp.where(
        new_fatigue > 30,
        jnp.maximum(state.player_energy - 1 * not_boss, 0),
        state.player_energy,
    )
    new_fatigue = jnp.where(new_fatigue > 30, 0.0, new_fatigue)

    new_energy = jnp.where(
        new_fatigue < -10,
        jnp.minimum(state.player_energy + 1, get_max_energy(state)),
        new_energy,
    )
    new_fatigue = jnp.where(new_fatigue < -10, 0.0, new_fatigue)

    state = state.replace(
        player_fatigue=jnp.where(state.player_alive, new_fatigue, state.player_fatigue),
        player_energy=jnp.where(state.player_alive, new_energy, state.player_energy),
    )

    # Health
    necessities = jnp.stack(
        [
            state.player_food > 0,
            state.player_drink > 0,
            jnp.logical_or(state.player_energy > 0, state.is_sleeping),
        ],
        axis=1,
    )

    all_necessities = necessities.all(axis=1)
    # all_necessities = jnp.full((static_params.player_count,), True, dtype=bool)

    new_all_necessities_frac = (state.all_necessities_frac * state.timestep + all_necessities) / (
        state.timestep + 1
    )

    recover_all = jnp.where(
        state.is_sleeping,
        2.0,
        1.0,
    )

    # Epic Shelter bonus: +50% health recovery rate when near a shelter
    # (fatigue recovery bonus is applied earlier in the fatigue section)
    recover_all = jnp.where(near_shelter, recover_all * 1.5, recover_all)

    recover_not_all = (
        jnp.where(
            state.is_sleeping,
            -0.5,
            -1.0,
        )
        * not_boss
    )
    recover_add = jnp.where(all_necessities, recover_all, recover_not_all)

    new_recover = state.player_recover + recover_add

    recovered_health = jnp.minimum(state.player_health + 2, get_max_health(state))
    derecovered_health = state.player_health - 1

    new_health = jnp.where(new_recover > 25, recovered_health, state.player_health)
    new_recover = jnp.where(new_recover > 25, 0.0, new_recover)
    new_health = jnp.where(new_recover < -15, derecovered_health, new_health)
    new_recover = jnp.where(new_recover < -15, 0.0, new_recover)

    # Guard health/recover updates: only apply to players alive in the previous step
    # AND still at positive health now (mob damage may have driven health <= 0 this step;
    # applying intrinsic healing to them would resurrect them and break death tracking).
    currently_alive = state.player_alive & (state.player_health > 0.0)
    state = state.replace(
        player_recover=jnp.where(currently_alive, new_recover, state.player_recover),
        player_health=jnp.where(currently_alive, new_health, state.player_health),
    )

    # Mana
    mana_recover_coeff = 1 + 0.25 * (state.player_intelligence - 1)
    mana_increment = jnp.where(state.is_sleeping, 2.0, 1.0) * mana_recover_coeff
    new_recover_mana = state.player_recover_mana + mana_increment

    new_mana = jnp.where(new_recover_mana > 30, state.player_mana + 1, state.player_mana)
    new_recover_mana = jnp.where(new_recover_mana > 30, 0.0, new_recover_mana)

    state = state.replace(
        player_recover_mana=jnp.where(
            state.player_alive, new_recover_mana, state.player_recover_mana
        ),
        player_mana=jnp.where(state.player_alive, new_mana, state.player_mana),
        all_necessities_frac=new_all_necessities_frac,
    )

    return state


def update_plants(state, static_params):
    """Age growing plants and mature ready saplings into map tiles.

    Args:
        state: Current environment state.
        static_params: Static plant limits and map parameters.

    Returns:
        State with plant slots and block maps updated.
    """
    growing_plants_age = state.growing_plants_age + 1
    growing_plants_age *= state.growing_plants_mask

    finished_growing_plants = growing_plants_age >= 500

    new_plant_blocks = jnp.where(
        finished_growing_plants,
        BlockType.RIPE_PLANT.value,
        BlockType.PLANT.value,
    )

    # Vectorized plant block update (single scatter instead of sequential scan)
    plant_rows = state.growing_plants_positions[:, 0]
    plant_cols = state.growing_plants_positions[:, 1]
    existing_blocks = state.map[0, plant_rows, plant_cols]
    updated_blocks = jnp.where(finished_growing_plants, new_plant_blocks, existing_blocks)
    new_map = state.map[0].at[plant_rows, plant_cols].set(updated_blocks)

    new_whole_map = state.map.at[0].set(new_map)

    state = state.replace(
        map=new_whole_map,
        growing_plants_age=growing_plants_age,
    )

    return state


def move_player(state, actions, params, static_params):
    """Resolve movement actions against terrain, mobs, and other players.

    Args:
        state: Current environment state.
        actions: Per-player action identifiers.
        params: Gameplay parameters controlling movement abilities.
        static_params: Static map dimensions and player count.

    Returns:
        State with valid player positions and directions applied.
    """
    proposed_position = state.player_position + DIRECTIONS[actions]

    valid_move = is_position_in_bounds_not_in_mob_not_colliding(
        state, proposed_position, COLLISION_LAND_CREATURE, static_params
    )
    valid_move = jnp.logical_and(
        valid_move, is_position_not_colliding_other_player(state, proposed_position)
    )
    valid_move = jnp.logical_or(valid_move, params.god_mode)

    position = (
        state.player_position
        + jnp.expand_dims(valid_move, axis=1).astype(jnp.int32) * DIRECTIONS[actions]
    )

    is_new_direction = jnp.sum(jnp.abs(DIRECTIONS[actions]), axis=1) != 0
    new_direction = state.player_direction * (1 - is_new_direction) + actions * is_new_direction

    state = state.replace(
        player_position=position,
        player_direction=new_direction,
    )

    return state


def spawn_mobs(state, rng, params, static_params):
    """Sample and populate free mob slots around eligible players.

    Args:
        state: Current environment state.
        rng: JAX random key used for spawn decisions and locations.
        params: Gameplay parameters controlling spawn rates.
        static_params: Static entity limits and map parameters.

    Returns:
        State with newly spawned passive, melee, and ranged mobs.
    """
    player_distance_map = get_all_players_distance_map(
        state.player_position, state.player_alive, static_params
    )
    grave_map = jnp.logical_or(
        state.map[state.player_level] == BlockType.GRAVE.value,
        jnp.logical_or(
            state.map[state.player_level] == BlockType.GRAVE2.value,
            state.map[state.player_level] == BlockType.GRAVE3.value,
        ),
    )

    floor_mob_spawn_chance = FLOOR_MOB_SPAWN_CHANCE * static_params.player_count
    monster_spawn_coeff = (
        1 + (state.monsters_killed[state.player_level] < MONSTERS_KILLED_TO_CLEAR_LEVEL) * 2
    )  # Triple spawn rate if we are on an uncleared level

    monster_spawn_coeff *= jax.lax.select(
        is_fighting_boss(state, static_params),
        is_boss_spawn_wave(state, static_params) * 1000,
        1,
    )

    # Passive mobs
    can_spawn_passive_mob = (
        state.passive_mobs.mask[state.player_level].sum()
        < static_params.max_passive_mobs * static_params.player_count
    )

    rng, _rng = jax.random.split(rng)
    can_spawn_passive_mob = jnp.logical_and(
        can_spawn_passive_mob,
        jax.random.uniform(_rng) < floor_mob_spawn_chance[state.player_level, 0],
    )

    can_spawn_passive_mob = jnp.logical_and(
        can_spawn_passive_mob, jnp.logical_not(is_fighting_boss(state, static_params))
    )

    all_valid_blocks_map = jnp.logical_or(
        state.map[state.player_level] == BlockType.GRASS.value,
        jnp.logical_or(
            state.map[state.player_level] == BlockType.PATH.value,
            jnp.logical_or(
                state.map[state.player_level] == BlockType.FIRE_GRASS.value,
                state.map[state.player_level] == BlockType.ICE_GRASS.value,
            ),
        ),
    )
    new_passive_mob_type = FLOOR_MOB_MAPPING[state.player_level, MobType.PASSIVE.value]

    passive_mobs_can_spawn_map = all_valid_blocks_map

    passive_mobs_can_spawn_map = jnp.logical_and(
        passive_mobs_can_spawn_map, player_distance_map > 3
    )
    passive_mobs_can_spawn_map = jnp.logical_and(
        passive_mobs_can_spawn_map, player_distance_map < params.mob_despawn_distance
    )
    passive_mobs_can_spawn_map = jnp.logical_and(
        passive_mobs_can_spawn_map, jnp.logical_not(state.mob_map[state.player_level])
    )

    # To avoid spawning mobs ontop of dead players
    passive_mobs_can_spawn_map = passive_mobs_can_spawn_map.at[
        state.player_position[:, 0], state.player_position[:, 1]
    ].set(False)

    can_spawn_passive_mob = jnp.logical_and(
        can_spawn_passive_mob, passive_mobs_can_spawn_map.sum() > 0
    )

    rng, _rng = jax.random.split(rng)
    passive_mob_position = jax.random.choice(
        _rng,
        jnp.arange(static_params.map_size[0] * static_params.map_size[1]),
        shape=(1,),
        p=jnp.reshape(passive_mobs_can_spawn_map, -1) / jnp.sum(passive_mobs_can_spawn_map),
    )
    passive_mob_position = jnp.array(
        [
            passive_mob_position // static_params.map_size[0],
            passive_mob_position % static_params.map_size[1],
        ]
    ).T.astype(jnp.int32)[0]

    new_passive_mob_index = jnp.argmax(jnp.logical_not(state.passive_mobs.mask[state.player_level]))

    new_passive_mob_position = jax.lax.select(
        can_spawn_passive_mob,
        passive_mob_position,
        state.passive_mobs.position[state.player_level, new_passive_mob_index],
    )

    new_passive_mob_health = jax.lax.select(
        can_spawn_passive_mob,
        MOB_TYPE_HEALTH_MAPPING[new_passive_mob_type, MobType.PASSIVE.value]
        * params.mob_health_multiplier,
        state.passive_mobs.health[state.player_level, new_passive_mob_index],
    )

    new_passive_mob_mask = jax.lax.select(
        can_spawn_passive_mob,
        True,
        state.passive_mobs.mask[state.player_level, new_passive_mob_index],
    )

    passive_mobs = Mobs(
        position=state.passive_mobs.position.at[state.player_level, new_passive_mob_index].set(
            new_passive_mob_position
        ),
        health=state.passive_mobs.health.at[state.player_level, new_passive_mob_index].set(
            new_passive_mob_health
        ),
        mask=state.passive_mobs.mask.at[state.player_level, new_passive_mob_index].set(
            new_passive_mob_mask
        ),
        attack_cooldown=state.passive_mobs.attack_cooldown,
        type_id=state.passive_mobs.type_id.at[state.player_level, new_passive_mob_index].set(
            new_passive_mob_type
        ),
    )

    # Elite/large status is pre-determined in world_gen, no random rolling here
    # Just update the mob data and mob_map

    state = state.replace(
        passive_mobs=passive_mobs,
        mob_map=state.mob_map.at[
            state.player_level, new_passive_mob_position[0], new_passive_mob_position[1]
        ].set(
            jnp.logical_or(
                state.mob_map[
                    state.player_level,
                    new_passive_mob_position[0],
                    new_passive_mob_position[1],
                ],
                new_passive_mob_mask,
            )
        ),
    )

    # Monsters
    DUNGEONS = jnp.array([1, 3, 4])
    in_dungeon = (state.player_level == DUNGEONS).any()

    monsters_can_spawn_player_range_map = player_distance_map > 9
    monsters_can_spawn_player_range_map_boss = player_distance_map <= 6

    monsters_can_spawn_player_range_map = jax.lax.select(
        is_fighting_boss(state, static_params),
        monsters_can_spawn_player_range_map_boss,
        monsters_can_spawn_player_range_map,
    )

    # Melee mobs
    can_spawn_melee_mob = state.melee_mobs.mask[state.player_level].sum() < (
        static_params.max_melee_mobs
        * (
            static_params.player_count * (1 - in_dungeon)
            + 1 * in_dungeon  # reduce number of mobs if in dungeons to avoid crowdedness
        )
    )

    new_melee_mob_type = FLOOR_MOB_MAPPING[state.player_level, MobType.MELEE.value]
    new_melee_mob_type_boss = FLOOR_MOB_MAPPING[state.boss_progress, MobType.MELEE.value]

    new_melee_mob_type = jax.lax.select(
        is_fighting_boss(state, static_params),
        new_melee_mob_type_boss,
        new_melee_mob_type,
    )

    rng, _rng = jax.random.split(rng)
    melee_mob_spawn_chance = floor_mob_spawn_chance[state.player_level, 1] + floor_mob_spawn_chance[
        state.player_level, 3
    ] * jnp.square(1 - state.light_level)
    can_spawn_melee_mob = jnp.logical_and(
        can_spawn_melee_mob,
        jax.random.uniform(_rng) < melee_mob_spawn_chance * monster_spawn_coeff,
    )

    melee_mobs_can_spawn_map = jax.lax.select(
        is_fighting_boss(state, static_params), grave_map, all_valid_blocks_map
    )

    melee_mobs_can_spawn_map = jnp.logical_and(
        melee_mobs_can_spawn_map, monsters_can_spawn_player_range_map
    )
    melee_mobs_can_spawn_map = jnp.logical_and(
        melee_mobs_can_spawn_map, player_distance_map < params.mob_despawn_distance
    )
    melee_mobs_can_spawn_map = jnp.logical_and(
        melee_mobs_can_spawn_map, jnp.logical_not(state.mob_map[state.player_level])
    )
    melee_mobs_can_spawn_map = melee_mobs_can_spawn_map.at[
        state.player_position[:, 0], state.player_position[:, 1]
    ].set(False)

    can_spawn_melee_mob = jnp.logical_and(can_spawn_melee_mob, melee_mobs_can_spawn_map.sum() > 0)

    rng, _rng = jax.random.split(rng)
    melee_mob_position = jax.random.choice(
        _rng,
        jnp.arange(static_params.map_size[0] * static_params.map_size[1]),
        shape=(1,),
        p=jnp.reshape(melee_mobs_can_spawn_map, -1) / jnp.sum(melee_mobs_can_spawn_map),
    )
    melee_mob_position = jnp.array(
        [
            melee_mob_position // static_params.map_size[0],
            melee_mob_position % static_params.map_size[1],
        ]
    ).T.astype(jnp.int32)[0]

    new_melee_mob_index = jnp.argmax(jnp.logical_not(state.melee_mobs.mask[state.player_level]))

    new_melee_mob_position = jax.lax.select(
        can_spawn_melee_mob,
        melee_mob_position,
        state.melee_mobs.position[state.player_level, new_melee_mob_index],
    )

    new_melee_mob_health = jax.lax.select(
        can_spawn_melee_mob,
        MOB_TYPE_HEALTH_MAPPING[new_melee_mob_type, MobType.MELEE.value]
        * params.mob_health_multiplier,
        state.melee_mobs.health[state.player_level, new_melee_mob_index],
    )

    new_melee_mob_mask = jax.lax.select(
        can_spawn_melee_mob,
        True,
        state.melee_mobs.mask[state.player_level, new_melee_mob_index],
    )

    melee_mobs = Mobs(
        position=state.melee_mobs.position.at[state.player_level, new_melee_mob_index].set(
            new_melee_mob_position
        ),
        health=state.melee_mobs.health.at[state.player_level, new_melee_mob_index].set(
            new_melee_mob_health
        ),
        mask=state.melee_mobs.mask.at[state.player_level, new_melee_mob_index].set(
            new_melee_mob_mask
        ),
        attack_cooldown=state.melee_mobs.attack_cooldown,
        type_id=state.melee_mobs.type_id.at[state.player_level, new_melee_mob_index].set(
            new_melee_mob_type
        ),
    )

    state = state.replace(
        melee_mobs=melee_mobs,
        mob_map=state.mob_map.at[
            state.player_level, new_melee_mob_position[0], new_melee_mob_position[1]
        ].set(
            jnp.logical_or(
                state.mob_map[
                    state.player_level,
                    new_melee_mob_position[0],
                    new_melee_mob_position[1],
                ],
                new_melee_mob_mask,
            )
        ),
    )

    # Ranged mobs
    can_spawn_ranged_mob = state.ranged_mobs.mask[state.player_level].sum() < (
        static_params.max_ranged_mobs
        * (
            static_params.player_count * (1 - in_dungeon)
            + 1 * in_dungeon  # reduce number of mobs if in dungeons to avoid crowdedness
        )
    )

    new_ranged_mob_type = FLOOR_MOB_MAPPING[state.player_level, MobType.RANGED.value]
    new_ranged_mob_type_boss = FLOOR_MOB_MAPPING[state.boss_progress, MobType.RANGED.value]

    new_ranged_mob_type = jax.lax.select(
        is_fighting_boss(state, static_params),
        new_ranged_mob_type_boss,
        new_ranged_mob_type,
    )

    rng, _rng = jax.random.split(rng)
    can_spawn_ranged_mob = jnp.logical_and(
        can_spawn_ranged_mob,
        jax.random.uniform(_rng)
        < floor_mob_spawn_chance[state.player_level, 2] * monster_spawn_coeff,
    )

    # Hack for deep thing
    ranged_mobs_can_spawn_map = jax.lax.select(
        new_ranged_mob_type == 5,
        state.map[state.player_level] == BlockType.WATER.value,
        all_valid_blocks_map,
    )
    ranged_mobs_can_spawn_map = jax.lax.select(
        is_fighting_boss(state, static_params), grave_map, ranged_mobs_can_spawn_map
    )

    ranged_mobs_can_spawn_map = jnp.logical_and(
        ranged_mobs_can_spawn_map, monsters_can_spawn_player_range_map
    )
    ranged_mobs_can_spawn_map = jnp.logical_and(
        ranged_mobs_can_spawn_map, player_distance_map < params.mob_despawn_distance
    )
    ranged_mobs_can_spawn_map = jnp.logical_and(
        ranged_mobs_can_spawn_map, jnp.logical_not(state.mob_map[state.player_level])
    )
    ranged_mobs_can_spawn_map = ranged_mobs_can_spawn_map.at[
        state.player_position[:, 0], state.player_position[:, 1]
    ].set(False)

    can_spawn_ranged_mob = jnp.logical_and(
        can_spawn_ranged_mob, ranged_mobs_can_spawn_map.sum() > 0
    )

    rng, _rng = jax.random.split(rng)
    ranged_mob_position = jax.random.choice(
        _rng,
        jnp.arange(static_params.map_size[0] * static_params.map_size[1]),
        shape=(1,),
        p=jnp.reshape(ranged_mobs_can_spawn_map, -1) / jnp.sum(ranged_mobs_can_spawn_map),
    )
    ranged_mob_position = jnp.array(
        [
            ranged_mob_position // static_params.map_size[0],
            ranged_mob_position % static_params.map_size[1],
        ]
    ).T.astype(jnp.int32)[0]

    new_ranged_mob_index = jnp.argmax(jnp.logical_not(state.ranged_mobs.mask[state.player_level]))

    new_ranged_mob_position = jax.lax.select(
        can_spawn_ranged_mob,
        ranged_mob_position,
        state.ranged_mobs.position[state.player_level, new_ranged_mob_index],
    )

    new_ranged_mob_health = jax.lax.select(
        can_spawn_ranged_mob,
        MOB_TYPE_HEALTH_MAPPING[new_ranged_mob_type, MobType.RANGED.value]
        * params.mob_health_multiplier,
        state.ranged_mobs.health[state.player_level, new_ranged_mob_index],
    )

    new_ranged_mob_mask = jax.lax.select(
        can_spawn_ranged_mob,
        True,
        state.ranged_mobs.mask[state.player_level, new_ranged_mob_index],
    )

    ranged_mobs = Mobs(
        position=state.ranged_mobs.position.at[state.player_level, new_ranged_mob_index].set(
            new_ranged_mob_position
        ),
        health=state.ranged_mobs.health.at[state.player_level, new_ranged_mob_index].set(
            new_ranged_mob_health
        ),
        mask=state.ranged_mobs.mask.at[state.player_level, new_ranged_mob_index].set(
            new_ranged_mob_mask
        ),
        attack_cooldown=state.ranged_mobs.attack_cooldown,
        type_id=state.ranged_mobs.type_id.at[state.player_level, new_ranged_mob_index].set(
            new_ranged_mob_type
        ),
    )

    state = state.replace(
        ranged_mobs=ranged_mobs,
        mob_map=state.mob_map.at[
            state.player_level, new_ranged_mob_position[0], new_ranged_mob_position[1]
        ].set(
            jnp.logical_or(
                state.mob_map[
                    state.player_level,
                    new_ranged_mob_position[0],
                    new_ranged_mob_position[1],
                ],
                new_ranged_mob_mask,
            )
        ),
    )

    return state


def change_floor(state: EnvState, actions, env_params: EnvParams, static_params: StaticEnvParams):
    """Move players through valid ladders and update level progression.

    Args:
        state: Current environment state.
        actions: Per-player action identifiers.
        env_params: Gameplay parameters controlling ladder use.
        static_params: Static level count and map parameters.

    Returns:
        State with player levels, positions, and progression updated.
    """
    is_moving_down = jnp.logical_and(
        actions == Action.DESCEND.value,
        jnp.logical_or(
            env_params.god_mode,
            jnp.logical_and(
                state.item_map[
                    state.player_level, state.player_position[:, 0], state.player_position[:, 1]
                ]
                == ItemType.LADDER_DOWN.value,
                state.monsters_killed[state.player_level] >= MONSTERS_KILLED_TO_CLEAR_LEVEL,
            ),
        ),
    )
    is_moving_down = jnp.logical_and(
        is_moving_down, state.player_level < static_params.num_levels - 1
    )
    is_moving_down = is_moving_down.any()

    moving_down_position = state.up_ladders[state.player_level + 1]

    is_moving_up = jnp.logical_and(
        actions == Action.ASCEND.value,
        jnp.logical_or(
            env_params.god_mode,
            state.item_map[
                state.player_level, state.player_position[:, 0], state.player_position[:, 1]
            ]
            == ItemType.LADDER_UP.value,
        ),
    )
    is_moving_up = jnp.logical_and(is_moving_up, state.player_level > 0)
    is_moving_up = is_moving_up.any()

    moving_up_position = state.down_ladders[state.player_level - 1]

    # prioritizes moving players down levels if two players are conflicted
    position = jax.lax.select(
        is_moving_down,
        moving_down_position,
        jax.lax.select(is_moving_up, moving_up_position, state.player_position),
    )
    delta_floor = jax.lax.select(is_moving_down, 1, jax.lax.select(is_moving_up, -1, 0))

    move_down_achievement = LEVEL_ACHIEVEMENT_MAP[state.player_level + delta_floor]

    new_achievements = state.achievements.at[:, move_down_achievement].set(
        jnp.logical_or(
            (state.player_level + delta_floor) != 0,
            state.achievements[:, move_down_achievement],
        )
    )

    new_floor = jnp.logical_and(
        (state.player_level + delta_floor) != 0,
        jnp.logical_not(state.achievements[:, move_down_achievement]),
    )

    # Clear pending handovers on level change — handovers are level-local
    # and their stored positions would be invalid on a different level.
    # Refund materials for any active construction handovers being cleared.
    is_changing_level = delta_floor != 0
    pending = state.pending_handovers
    is_active = pending[:, 0] == 1
    active_initiator = pending[:, 4]
    active_build_type = pending[:, 5]  # 0=mining, 1=shelter, 2=forge, 3=beacon
    should_refund = is_changing_level & is_active
    is_refund_shelter = should_refund & (active_build_type == 1)
    is_refund_forge = should_refund & (active_build_type == 2)
    is_refund_beacon = should_refund & (active_build_type == 3)

    wood_refund = jnp.zeros(static_params.player_count, dtype=state.inventory.wood.dtype)
    wood_refund = wood_refund.at[active_initiator].add(SHELTER_COST_WOOD * is_refund_shelter)
    stone_refund = jnp.zeros(static_params.player_count, dtype=state.inventory.stone.dtype)
    stone_refund = stone_refund.at[active_initiator].add(
        SHELTER_COST_STONE * is_refund_shelter + FORGE_COST_STONE * is_refund_forge
    )
    iron_refund = jnp.zeros(static_params.player_count, dtype=state.inventory.iron.dtype)
    iron_refund = iron_refund.at[active_initiator].add(
        FORGE_COST_IRON * is_refund_forge + BEACON_COST_IRON * is_refund_beacon
    )
    coal_refund = jnp.zeros(static_params.player_count, dtype=state.inventory.coal.dtype)
    coal_refund = coal_refund.at[active_initiator].add(
        FORGE_COST_COAL * is_refund_forge + BEACON_COST_COAL * is_refund_beacon
    )

    cleared_handovers = jnp.where(
        is_changing_level,
        jnp.zeros_like(state.pending_handovers),
        state.pending_handovers,
    )

    new_player_level = state.player_level + delta_floor
    state = state.replace(
        player_level=new_player_level,
        max_player_level=jnp.maximum(state.max_player_level, new_player_level),
        player_position=position,
        achievements=new_achievements,
        player_xp=state.player_xp + 1 * new_floor,
        pending_handovers=cleared_handovers,
        inventory=state.inventory.replace(
            wood=state.inventory.wood + wood_refund,
            stone=state.inventory.stone + stone_refund,
            iron=state.inventory.iron + iron_refund,
            coal=state.inventory.coal + coal_refund,
        ),
    )

    return state


def shoot_projectile(state: EnvState, action: int, static_params: StaticEnvParams):
    """Spawn arrows for players performing the shoot action.

    Args:
        state: Current environment state.
        action: Per-player action identifiers.
        static_params: Static projectile limits and player count.

    Returns:
        State with projectiles, arrow inventory, and achievements updated.
    """

    # Arrow
    def _spawn_player_projectiles(projectile_info, player_index):
        player_projectiles, player_projectile_directions, player_projectile_owners = projectile_info

        is_shooting_arrow = jnp.logical_and(
            action[player_index] == Action.SHOOT_ARROW.value,
            jnp.logical_and(
                state.inventory.bow[player_index] >= 1,
                jnp.logical_and(
                    state.inventory.arrows[player_index] >= 1,
                    player_projectiles.mask[state.player_level].sum()
                    < (static_params.max_player_projectiles * static_params.player_count),
                ),
            ),
        )

        new_player_projectiles, new_player_projectile_directions, new_player_projectile_owners = (
            spawn_projectile(
                state,
                static_params,
                player_projectiles,
                player_projectile_directions,
                player_projectile_owners,
                state.player_position[player_index],
                is_shooting_arrow,
                player_index,
                DIRECTIONS[state.player_direction[player_index]],
                ProjectileType.ARROW2.value,
            )
        )

        return (
            new_player_projectiles,
            new_player_projectile_directions,
            new_player_projectile_owners,
        ), is_shooting_arrow

    (
        (new_player_projectiles, new_player_projectile_directions, new_player_projectile_owners),
        is_shooting_arrow,
    ) = jax.lax.scan(
        _spawn_player_projectiles,
        (
            state.player_projectiles,
            state.player_projectile_directions,
            state.player_projectile_owners,
        ),
        jnp.arange(static_params.player_count),
    )

    new_achievements = state.achievements.at[:, Achievement.FIRE_BOW.value].set(
        jnp.logical_or(state.achievements[:, Achievement.FIRE_BOW.value], is_shooting_arrow)
    )

    return state.replace(
        player_projectiles=new_player_projectiles,
        player_projectile_directions=new_player_projectile_directions,
        player_projectile_owners=new_player_projectile_owners,
        inventory=state.inventory.replace(arrows=state.inventory.arrows - 1 * is_shooting_arrow),
        achievements=new_achievements,
    )


def cast_spell(state, action, static_params):
    """Cast learned spells and spawn their projectile or healing effects.

    Args:
        state: Current environment state.
        action: Per-player action identifiers.
        static_params: Static projectile limits and player count.

    Returns:
        State with spell effects, mana, and achievements updated.
    """
    is_miner = state.player_specialization == Specialization.MINER.value
    is_warrior = state.player_specialization == Specialization.WARRIOR.value
    is_forager = state.player_specialization == Specialization.FORAGER.value

    spell_mana_cost = jnp.array([2, 6])  # fireball costs 2, healing costs 6

    def _cast_player_spell(player_info, player_index):
        (
            player_projectiles,
            player_projectile_directions,
            player_projectile_owners,
            player_health,
        ) = player_info

        is_casting_spell = jnp.logical_and(
            action[player_index] == Action.CAST_SPELL.value, state.learned_spells[player_index]
        )

        # Warriors/Miners -> Cast Fireball
        is_casting_fireball = jnp.logical_and(
            is_casting_spell, state.player_mana[player_index] >= spell_mana_cost[0]
        )
        is_casting_fireball = jnp.logical_and(
            is_casting_fireball,
            jnp.logical_and(
                jnp.logical_or(is_miner[player_index], is_warrior[player_index]),
                player_projectiles.mask[state.player_level].sum()
                < (static_params.max_player_projectiles * static_params.player_count),
            ),
        )
        new_player_projectiles, new_player_projectile_directions, new_player_projectile_owners = (
            spawn_projectile(
                state,
                static_params,
                player_projectiles,
                player_projectile_directions,
                player_projectile_owners,
                state.player_position[player_index],
                is_casting_fireball,
                player_index,
                DIRECTIONS[state.player_direction[player_index]],
                ProjectileType.FIREBALL.value,
            )
        )

        # Foragers -> Healing
        is_casting_healing = jnp.logical_and(is_casting_spell, is_forager[player_index])
        is_casting_healing = jnp.logical_and(
            is_casting_healing, state.player_mana[player_index] >= spell_mana_cost[1]
        )
        health_increase = 2
        new_player_health = jnp.minimum(
            player_health + state.player_alive * (health_increase * is_casting_healing),
            get_max_health(state),
        )

        spell_cast = jnp.array([is_casting_fireball, is_casting_healing])

        return (
            new_player_projectiles,
            new_player_projectile_directions,
            new_player_projectile_owners,
            new_player_health,
        ), spell_cast

    (
        (
            new_player_projectiles,
            new_player_projectile_directions,
            new_player_projectile_owners,
            new_player_health,
        ),
        spell_cast,
    ) = jax.lax.scan(
        _cast_player_spell,
        (
            state.player_projectiles,
            state.player_projectile_directions,
            state.player_projectile_owners,
            state.player_health,
        ),
        jnp.arange(static_params.player_count),
    )
    did_cast_spell = spell_cast.any(axis=-1)
    new_achievements = state.achievements.at[:, Achievement.CAST_SPELL.value].set(
        jnp.logical_or(state.achievements[:, Achievement.CAST_SPELL.value], did_cast_spell)
    )

    return state.replace(
        player_projectiles=new_player_projectiles,
        player_projectile_directions=new_player_projectile_directions,
        player_projectile_owners=new_player_projectile_owners,
        player_health=new_player_health,
        player_mana=state.player_mana - jnp.dot(spell_cast, spell_mana_cost),
        achievements=new_achievements,
    )


def drink_potion(state, action):
    """Consume selected potions and apply their intrinsic effects.

    Args:
        state: Current environment state.
        action: Per-player action identifiers.

    Returns:
        State with potion inventory, attributes, and achievements updated.
    """
    drinking_potion_index = -1
    is_drinking_potion = False

    # Red
    is_drinking_red_potion = jnp.logical_and(
        action == Action.DRINK_POTION_RED.value, state.inventory.potions[:, 0] > 0
    )
    drinking_potion_index = (
        is_drinking_red_potion * 0 + (1 - is_drinking_red_potion) * drinking_potion_index
    )
    is_drinking_potion = jnp.logical_or(is_drinking_potion, is_drinking_red_potion)

    # Green
    is_drinking_green_potion = jnp.logical_and(
        action == Action.DRINK_POTION_GREEN.value, state.inventory.potions[:, 1] > 0
    )
    drinking_potion_index = (
        is_drinking_green_potion * 1 + (1 - is_drinking_green_potion) * drinking_potion_index
    )
    is_drinking_potion = jnp.logical_or(is_drinking_potion, is_drinking_green_potion)

    # Blue
    is_drinking_blue_potion = jnp.logical_and(
        action == Action.DRINK_POTION_BLUE.value, state.inventory.potions[:, 2] > 0
    )
    drinking_potion_index = (
        is_drinking_blue_potion * 2 + (1 - is_drinking_blue_potion) * drinking_potion_index
    )
    is_drinking_potion = jnp.logical_or(is_drinking_potion, is_drinking_blue_potion)

    # Pink
    is_drinking_pink_potion = jnp.logical_and(
        action == Action.DRINK_POTION_PINK.value, state.inventory.potions[:, 3] > 0
    )
    drinking_potion_index = (
        is_drinking_pink_potion * 3 + (1 - is_drinking_pink_potion) * drinking_potion_index
    )
    is_drinking_potion = jnp.logical_or(is_drinking_potion, is_drinking_pink_potion)

    # Cyan
    is_drinking_cyan_potion = jnp.logical_and(
        action == Action.DRINK_POTION_CYAN.value, state.inventory.potions[:, 4] > 0
    )
    drinking_potion_index = (
        is_drinking_cyan_potion * 4 + (1 - is_drinking_cyan_potion) * drinking_potion_index
    )
    is_drinking_potion = jnp.logical_or(is_drinking_potion, is_drinking_cyan_potion)

    # Yellow
    is_drinking_yellow_potion = jnp.logical_and(
        action == Action.DRINK_POTION_YELLOW.value, state.inventory.potions[:, 5] > 0
    )
    drinking_potion_index = (
        is_drinking_yellow_potion * 5 + (1 - is_drinking_yellow_potion) * drinking_potion_index
    )
    is_drinking_potion = jnp.logical_or(is_drinking_potion, is_drinking_yellow_potion)

    # Potion mapping
    potion_effect_index = state.potion_mapping[drinking_potion_index]

    # Potion effect
    delta_health = 0
    delta_health += is_drinking_potion * (potion_effect_index == 0) * 8
    delta_health += is_drinking_potion * (potion_effect_index == 1) * (-3)

    delta_mana = 0
    delta_mana += is_drinking_potion * (potion_effect_index == 2) * 8
    delta_mana += is_drinking_potion * (potion_effect_index == 3) * (-3)

    delta_energy = 0
    delta_energy += is_drinking_potion * (potion_effect_index == 4) * 8
    delta_energy += is_drinking_potion * (potion_effect_index == 5) * (-3)

    new_achievements = state.achievements.at[:, Achievement.DRINK_POTION.value].set(
        jnp.logical_or(state.achievements[:, Achievement.DRINK_POTION.value], is_drinking_potion)
    )

    return state.replace(
        inventory=state.inventory.replace(
            potions=state.inventory.potions.at[
                jnp.arange(state.inventory.potions.shape[0]), drinking_potion_index
            ].set(
                state.inventory.potions[
                    jnp.arange(state.inventory.potions.shape[0]), drinking_potion_index
                ]
                - 1 * is_drinking_potion
            )
        ),
        player_health=state.player_health + delta_health,
        player_mana=state.player_mana + delta_mana,
        player_energy=state.player_energy + delta_energy,
        achievements=new_achievements,
    )


def read_book(state, action):
    """Consume books to learn spells for eligible players.

    Args:
        state: Current environment state.
        action: Per-player action identifiers.

    Returns:
        State with book inventory, learned spells, and achievements updated.
    """
    is_reading_book = jnp.logical_and(action == Action.READ_BOOK.value, state.inventory.books > 0)
    new_spells = jnp.logical_or(state.learned_spells, is_reading_book)
    new_achievements = state.achievements.at[:, Achievement.LEARN_SPELL.value].set(
        jnp.logical_or(state.achievements[:, Achievement.LEARN_SPELL.value], is_reading_book)
    )

    return state.replace(
        inventory=state.inventory.replace(books=state.inventory.books - 1 * is_reading_book),
        learned_spells=new_spells,
        achievements=new_achievements,
    )


def enchant(rng, state: EnvState, action, env_params, static_params: StaticEnvParams):
    """Resolve weapon, bow, and armour enchantment actions.

    Args:
        rng: JAX random key used by specialization gates.
        state: Current environment state.
        action: Per-player action identifiers.
        env_params: Gameplay parameters controlling enchantment gates.
        static_params: Static parameters used for nearby-block checks.

    Returns:
        State with enchantments, materials, mana, and achievements updated.
    """
    is_warrior = state.player_specialization == Specialization.WARRIOR.value
    # Apply soft gating to specialization
    is_warrior, rng = soft_gate_specialization(rng, is_warrior, env_params)

    target_block_position = state.player_position + DIRECTIONS[state.player_direction]
    target_block = state.map[
        state.player_level, target_block_position[:, 0], target_block_position[:, 1]
    ]
    target_block_is_enchantment_table = jnp.logical_or(
        target_block == BlockType.ENCHANTMENT_TABLE_FIRE.value,
        target_block == BlockType.ENCHANTMENT_TABLE_ICE.value,
    )

    enchantment_type = jnp.where(target_block == BlockType.ENCHANTMENT_TABLE_FIRE.value, 1, 2)

    num_gems = jnp.where(
        target_block == BlockType.ENCHANTMENT_TABLE_FIRE.value,
        state.inventory.ruby,
        state.inventory.sapphire,
    )

    could_enchant = jnp.logical_and(
        state.player_mana >= 9,
        jnp.logical_and(target_block_is_enchantment_table, num_gems >= 1),
    )

    could_enchant_warrior = jnp.logical_and(is_warrior, could_enchant)

    is_enchanting_bow = jnp.logical_and(
        could_enchant_warrior,
        jnp.logical_and(action == Action.ENCHANT_BOW.value, state.inventory.bow > 0),
    )

    is_enchanting_sword = jnp.logical_and(
        could_enchant_warrior,
        jnp.logical_and(action == Action.ENCHANT_SWORD.value, state.inventory.sword > 0),
    )

    is_enchanting_armour = jnp.logical_and(
        could_enchant,
        jnp.logical_and(
            action == Action.ENCHANT_ARMOUR.value, state.inventory.armour.sum(axis=1) > 0
        ),
    )

    rng, _rng = jax.random.split(rng)
    unenchanted_armour = state.armour_enchantments == 0
    opposite_enchanted_armour = jnp.logical_and(
        state.armour_enchantments != 0, state.armour_enchantments != enchantment_type[:, None]
    )

    armour_targets = (
        unenchanted_armour
        + (unenchanted_armour.sum(axis=1) == 0)[:, None] * opposite_enchanted_armour
    )

    _rngs = jax.random.split(rng, static_params.player_count + 1)
    rng, _rng = _rngs[0], _rngs[1:]
    armour_target = jax.vmap(jax.random.choice, in_axes=(0, None, None, None, 0))(
        _rng, jnp.arange(4), (), True, armour_targets
    )

    is_enchanting = jnp.logical_or(
        is_enchanting_sword, jnp.logical_or(is_enchanting_bow, is_enchanting_armour)
    )

    new_sword_enchantment = (
        is_enchanting_sword * enchantment_type + (1 - is_enchanting_sword) * state.sword_enchantment
    )
    new_bow_enchantment = (
        is_enchanting_bow * enchantment_type + (1 - is_enchanting_bow) * state.bow_enchantment
    )

    new_armour_enchantments = state.armour_enchantments.at[
        jnp.arange(static_params.player_count), armour_target
    ].set(
        is_enchanting_armour * enchantment_type
        + (1 - is_enchanting_armour)
        * state.armour_enchantments[jnp.arange(static_params.player_count), armour_target]
    )

    new_sapphire = state.inventory.sapphire - 1 * is_enchanting * (enchantment_type == 2)
    new_ruby = state.inventory.ruby - 1 * is_enchanting * (enchantment_type == 1)
    new_mana = state.player_mana - 9 * is_enchanting

    new_achievements = state.achievements.at[:, Achievement.ENCHANT_SWORD.value].set(
        jnp.logical_or(state.achievements[:, Achievement.ENCHANT_SWORD.value], is_enchanting_sword)
    )

    new_achievements = new_achievements.at[:, Achievement.ENCHANT_ARMOUR.value].set(
        jnp.logical_or(new_achievements[:, Achievement.ENCHANT_ARMOUR.value], is_enchanting_armour)
    )

    return state.replace(
        sword_enchantment=new_sword_enchantment,
        bow_enchantment=new_bow_enchantment,
        armour_enchantments=new_armour_enchantments,
        inventory=state.inventory.replace(
            sapphire=new_sapphire,
            ruby=new_ruby,
        ),
        player_mana=new_mana,
        achievements=new_achievements,
    )


def boss_logic(state, static_params):
    """Advance boss spawn timers and award boss damage achievements.

    Args:
        state: Current environment state.
        static_params: Static parameters used to identify boss combat.

    Returns:
        State with boss timers and achievements updated.
    """
    new_achievements = state.achievements.at[:, Achievement.DEFEAT_NECROMANCER.value].set(
        jnp.logical_or(
            state.achievements[:, Achievement.DEFEAT_NECROMANCER.value],
            has_beaten_boss(state, static_params),
        )
    )

    return state.replace(
        boss_timesteps_to_spawn_this_round=state.boss_timesteps_to_spawn_this_round
        - 1 * is_fighting_boss(state, static_params),
        achievements=new_achievements,
    )


def calculate_inventory_achievements(state):
    """Derive collection and equipment achievements from inventory state.

    Args:
        state: Current environment state.

    Returns:
        State with inventory-dependent achievements updated.
    """
    # Some achievements (e.g. make_diamond_pickaxe) can be achieved in multiple ways (finding in chest or crafting)
    # Rather than duplicating achievement code, we simply look in the inventory for these types of achievements
    # at the end of each timestep
    # Wood
    achievements = state.achievements.at[:, Achievement.COLLECT_WOOD.value].set(
        jnp.logical_or(
            state.achievements[:, Achievement.COLLECT_WOOD.value], state.inventory.wood > 0
        )
    )
    # Stone
    achievements = achievements.at[:, Achievement.COLLECT_STONE.value].set(
        jnp.logical_or(achievements[:, Achievement.COLLECT_STONE.value], state.inventory.stone > 0)
    )
    # Coal
    achievements = achievements.at[:, Achievement.COLLECT_COAL.value].set(
        jnp.logical_or(achievements[:, Achievement.COLLECT_COAL.value], state.inventory.coal > 0)
    )
    # Iron
    achievements = achievements.at[:, Achievement.COLLECT_IRON.value].set(
        jnp.logical_or(achievements[:, Achievement.COLLECT_IRON.value], state.inventory.iron > 0)
    )
    # Diamond
    achievements = achievements.at[:, Achievement.COLLECT_DIAMOND.value].set(
        jnp.logical_or(
            achievements[:, Achievement.COLLECT_DIAMOND.value], state.inventory.diamond > 0
        )
    )
    # Ruby
    achievements = achievements.at[:, Achievement.COLLECT_RUBY.value].set(
        jnp.logical_or(achievements[:, Achievement.COLLECT_RUBY.value], state.inventory.ruby > 0)
    )
    # Sapphire
    achievements = achievements.at[:, Achievement.COLLECT_SAPPHIRE.value].set(
        jnp.logical_or(
            achievements[:, Achievement.COLLECT_SAPPHIRE.value],
            state.inventory.sapphire > 0,
        )
    )
    # Sapling
    achievements = achievements.at[:, Achievement.COLLECT_SAPLING.value].set(
        jnp.logical_or(
            achievements[:, Achievement.COLLECT_SAPLING.value], state.inventory.sapling > 0
        )
    )
    # Bow
    achievements = achievements.at[:, Achievement.FIND_BOW.value].set(
        jnp.logical_or(achievements[:, Achievement.FIND_BOW.value], state.inventory.bow > 0)
    )
    # Arrow
    achievements = achievements.at[:, Achievement.MAKE_ARROW.value].set(
        jnp.logical_or(achievements[:, Achievement.MAKE_ARROW.value], state.inventory.arrows > 0)
    )
    # Torch
    achievements = achievements.at[:, Achievement.MAKE_TORCH.value].set(
        jnp.logical_or(achievements[:, Achievement.MAKE_TORCH.value], state.inventory.torches > 0)
    )

    # Pickaxe
    achievements = achievements.at[:, Achievement.MAKE_WOOD_PICKAXE.value].set(
        jnp.logical_or(
            achievements[:, Achievement.MAKE_WOOD_PICKAXE.value],
            state.inventory.pickaxe >= 1,
        )
    )
    achievements = achievements.at[:, Achievement.MAKE_STONE_PICKAXE.value].set(
        jnp.logical_or(
            achievements[:, Achievement.MAKE_STONE_PICKAXE.value],
            state.inventory.pickaxe >= 2,
        )
    )
    achievements = achievements.at[:, Achievement.MAKE_IRON_PICKAXE.value].set(
        jnp.logical_or(
            achievements[:, Achievement.MAKE_IRON_PICKAXE.value],
            state.inventory.pickaxe >= 3,
        )
    )
    achievements = achievements.at[:, Achievement.MAKE_DIAMOND_PICKAXE.value].set(
        jnp.logical_or(
            achievements[:, Achievement.MAKE_DIAMOND_PICKAXE.value],
            state.inventory.pickaxe >= 4,
        )
    )

    # Sword
    achievements = achievements.at[:, Achievement.MAKE_WOOD_SWORD.value].set(
        jnp.logical_or(
            achievements[:, Achievement.MAKE_WOOD_SWORD.value], state.inventory.sword >= 1
        )
    )
    achievements = achievements.at[:, Achievement.MAKE_STONE_SWORD.value].set(
        jnp.logical_or(
            achievements[:, Achievement.MAKE_STONE_SWORD.value], state.inventory.sword >= 2
        )
    )
    achievements = achievements.at[:, Achievement.MAKE_IRON_SWORD.value].set(
        jnp.logical_or(
            achievements[:, Achievement.MAKE_IRON_SWORD.value], state.inventory.sword >= 3
        )
    )
    achievements = achievements.at[:, Achievement.MAKE_DIAMOND_SWORD.value].set(
        jnp.logical_or(
            achievements[:, Achievement.MAKE_DIAMOND_SWORD.value],
            state.inventory.sword >= 4,
        )
    )

    return state.replace(achievements=achievements)


def trade_materials(state, action, static_params):
    """Transfer requested resources between adjacent players.

    Args:
        state: Current environment state.
        action: Per-player action identifiers.
        static_params: Static player count and communication parameters.

    Returns:
        State with inventories and trade metrics updated.
    """
    new_achievements = state.achievements
    new_trade_count = state.trade_count
    new_food_trade_count = state.food_trade_count
    new_drink_trade_count = state.drink_trade_count
    new_give_attempt_count = state.give_attempt_count

    player_trading_to = action - Action.GIVE.value
    player_trading_to += 1 * (player_trading_to >= jnp.arange(static_params.player_count))
    # Clamp to valid range so indexing is safe when player_count=1 (is_giving is False in that case)
    player_trading_to = jnp.clip(player_trading_to, 0, static_params.player_count - 1)
    is_giving = jnp.logical_and(
        action >= Action.GIVE.value, action < (Action.GIVE.value + static_params.player_count - 1)
    )
    new_give_attempt_count += is_giving.astype(new_give_attempt_count.dtype)
    other_player_is_requesting = jnp.logical_and(
        state.request_duration[player_trading_to] > 0, state.player_alive[player_trading_to]
    )

    def _new_material_value(
        material_type,
        current_material_stock,
        material_max_value,
        old_trade_count,
        old_request_received_count,
    ):
        material_max_arr = jnp.broadcast_to(
            jnp.asarray(material_max_value), current_material_stock.shape
        )
        other_player_is_requesting_material = jnp.logical_and(
            other_player_is_requesting, state.request_type[player_trading_to] == material_type
        )
        is_giving_material = jnp.logical_and(
            jnp.logical_and(  # Checks that other player is requesting and can take materials
                other_player_is_requesting_material,
                current_material_stock[player_trading_to] < material_max_arr[player_trading_to],
            ),
            jnp.logical_and(  # Checks that player has materials and is giving
                is_giving, current_material_stock > 0
            ),
        )
        new_material = current_material_stock - 1 * is_giving_material
        new_material = new_material.at[player_trading_to].add(is_giving_material)
        # Attribute trade to the giving agent (shape: (player_count,)).
        received_increments = jnp.zeros_like(old_request_received_count)
        received_increments = received_increments.at[player_trading_to].add(
            is_giving_material.astype(old_request_received_count.dtype)
        )
        return (
            new_material,
            old_trade_count + is_giving_material.astype(old_trade_count.dtype),
            old_request_received_count + received_increments,
        )

    # Food
    food_trade_count = jnp.zeros((static_params.player_count,), dtype=jnp.int32)
    new_request_received_count = state.request_received_count
    new_food, food_trade_count, new_request_received_count = _new_material_value(
        Action.REQUEST_FOOD.value,
        state.player_food,
        get_max_food(state),
        food_trade_count,
        new_request_received_count,
    )
    new_hunger = jnp.where(new_food > state.player_food, 0.0, state.player_hunger)
    new_achievements = new_achievements.at[:, Achievement.COLLECT_FOOD.value].set(
        jnp.logical_or(
            new_achievements[:, Achievement.COLLECT_FOOD.value], new_food > state.player_food
        )
    )
    new_food_trade_count += food_trade_count
    new_trade_count += food_trade_count

    # Drink
    drink_trade_count = jnp.zeros((static_params.player_count,), dtype=jnp.int32)
    new_drink, drink_trade_count, new_request_received_count = _new_material_value(
        Action.REQUEST_DRINK.value,
        state.player_drink,
        get_max_drink(state),
        drink_trade_count,
        new_request_received_count,
    )
    new_thirst = jnp.where(new_drink > state.player_drink, 0.0, state.player_thirst)
    new_achievements = new_achievements.at[:, Achievement.COLLECT_DRINK.value].set(
        jnp.logical_or(
            new_achievements[:, Achievement.COLLECT_DRINK.value], new_drink > state.player_drink
        )
    )
    new_drink_trade_count += drink_trade_count
    new_trade_count += drink_trade_count

    # Inventory Materials
    new_wood, new_trade_count, new_request_received_count = _new_material_value(
        Action.REQUEST_WOOD.value,
        state.inventory.wood,
        99,
        new_trade_count,
        new_request_received_count,
    )
    new_stone, new_trade_count, new_request_received_count = _new_material_value(
        Action.REQUEST_STONE.value,
        state.inventory.stone,
        99,
        new_trade_count,
        new_request_received_count,
    )
    new_iron, new_trade_count, new_request_received_count = _new_material_value(
        Action.REQUEST_IRON.value,
        state.inventory.iron,
        99,
        new_trade_count,
        new_request_received_count,
    )
    new_coal, new_trade_count, new_request_received_count = _new_material_value(
        Action.REQUEST_COAL.value,
        state.inventory.coal,
        99,
        new_trade_count,
        new_request_received_count,
    )
    new_diamond, new_trade_count, new_request_received_count = _new_material_value(
        Action.REQUEST_DIAMOND.value,
        state.inventory.diamond,
        99,
        new_trade_count,
        new_request_received_count,
    )
    new_ruby, new_trade_count, new_request_received_count = _new_material_value(
        Action.REQUEST_RUBY.value,
        state.inventory.ruby,
        99,
        new_trade_count,
        new_request_received_count,
    )
    new_sapphire, new_trade_count, new_request_received_count = _new_material_value(
        Action.REQUEST_SAPPHIRE.value,
        state.inventory.sapphire,
        99,
        new_trade_count,
        new_request_received_count,
    )

    # Update State
    state = state.replace(
        player_food=new_food,
        player_drink=new_drink,
        player_hunger=new_hunger,
        player_thirst=new_thirst,
        inventory=state.inventory.replace(
            wood=new_wood,
            stone=new_stone,
            iron=new_iron,
            coal=new_coal,
            diamond=new_diamond,
            ruby=new_ruby,
            sapphire=new_sapphire,
        ),
        achievements=new_achievements,
        trade_count=new_trade_count,
        food_trade_count=new_food_trade_count,
        drink_trade_count=new_drink_trade_count,
        give_attempt_count=new_give_attempt_count,
        request_received_count=new_request_received_count,
    )
    return state


def make_request(state, action):
    """Record material requests initiated by player actions.

    Args:
        state: Current environment state.
        action: Per-player action identifiers.

    Returns:
        State with request types, timers, and request metrics updated.
    """
    # Requests persist for a short window. Count expiries before a new request can
    # refresh the slot so timeout metrics describe genuinely unresolved requests.
    request_duration_after_decay = jnp.maximum(0, state.request_duration - 1)

    # Initialize New Request
    is_making_request = jnp.logical_and(
        action >= Action.REQUEST_FOOD.value, action <= Action.REQUEST_SAPPHIRE.value
    )
    request_expired = (
        (state.request_duration > 0) & (request_duration_after_decay == 0) & ~is_making_request
    )
    new_request_type = jnp.where(is_making_request, action, state.request_type)
    state = state.replace(
        request_duration=jnp.maximum(
            request_duration_after_decay,
            is_making_request.astype(state.request_duration.dtype) * REQUEST_MAX_DURATION,
        ),
        request_type=new_request_type,
        request_count=state.request_count + is_making_request.astype(state.request_count.dtype),
        request_expiry_count=state.request_expiry_count
        + request_expired.astype(state.request_expiry_count.dtype),
    )
    return state


def process_communication(state, actions, static_params):
    """Process MPE-style discrete communication actions.

    Each agent can send a one-hot message by taking a comm action.
    Messages reset every step; a comm action sets comm[agent, i] = 1.0.

    Args:
        state: Current environment state.
        actions: Per-player action identifiers.
        static_params: Static communication-channel and player counts.

    Returns:
        State with one-step messages and communication counts updated.
    """
    nc = static_params.num_comm_channels
    pc = static_params.player_count

    # Reset messages to zeros each step
    new_messages = jnp.zeros((pc, nc), dtype=jnp.float32)

    # Comm action base = len(Action) + (player_count - 2)
    # offset for give actions
    comm_base = len(Action) + (pc - 2)

    # For each agent, check if action is a comm action
    comm_action_offset = actions - comm_base  # which channel (0..nc-1)
    is_comm = (comm_action_offset >= 0) & (comm_action_offset < nc)

    # Set one-hot: new_messages[agent, offset] = 1.0 where is_comm
    one_hot = jax.nn.one_hot(comm_action_offset, num_classes=nc)  # (pc, nc)
    new_messages = jnp.where(is_comm[:, None], one_hot, new_messages)

    # Increment per-agent communication count
    new_comm_count = state.comm_count + is_comm.astype(jnp.int32)

    return state.replace(comm_messages=new_messages, comm_count=new_comm_count)


def level_up_attributes(state: EnvState, action: jnp.array, params: EnvParams) -> EnvState:
    """Spend experience to increase selected player attributes.

    Args:
        state: Current environment state.
        action: Per-player action identifiers.
        params: Gameplay parameters controlling level-up behavior.

    Returns:
        State with attributes and experience updated.
    """
    can_level_up = state.player_xp >= 1

    # Levelling up attributes
    is_levelling_up_dex = jnp.logical_and(
        can_level_up,
        jnp.logical_and(
            action == Action.LEVEL_UP_DEXTERITY.value,
            state.player_dexterity < params.max_attribute,
        ),
    )
    is_levelling_up_str = jnp.logical_and(
        can_level_up,
        jnp.logical_and(
            action == Action.LEVEL_UP_STRENGTH.value,
            state.player_strength < params.max_attribute,
        ),
    )
    is_levelling_up_int = jnp.logical_and(
        can_level_up,
        jnp.logical_and(
            action == Action.LEVEL_UP_INTELLIGENCE.value,
            state.player_intelligence < params.max_attribute,
        ),
    )
    is_levelling_up = jnp.logical_or(
        is_levelling_up_dex, jnp.logical_or(is_levelling_up_str, is_levelling_up_int)
    )

    return state.replace(
        player_dexterity=state.player_dexterity + 1 * is_levelling_up_dex,
        player_strength=state.player_strength + 1 * is_levelling_up_str,
        player_intelligence=state.player_intelligence + 1 * is_levelling_up_int,
        player_xp=state.player_xp - 1 * is_levelling_up,
    )


def alem_step(
    rng: chex.PRNGKey,
    state: EnvState,
    actions: Int[Array, "n_agents"],
    params: EnvParams,
    static_params: StaticEnvParams,
) -> tuple[EnvState, Float[Array, "n_agents"]]:
    """Execute one complete simultaneous environment transition.

    Args:
        rng: JAX random key used by all stochastic transition stages.
        state: Current environment state.
        actions: Per-player action identifiers.
        params: Dynamic gameplay and reward parameters.
        static_params: Static shapes, limits, and map parameters.

    Returns:
        Next environment state and per-player reward vector.
    """
    init_achievements = state.achievements
    init_health = state.player_health
    init_alive = state.player_alive

    # Interrupt action if dead, sleeping or resting
    cant_do_action = jnp.logical_or(
        jnp.logical_not(state.player_alive),
        jnp.logical_or(state.is_sleeping, state.is_resting),
    )
    # Exposure denominators for behavior rates.
    # Alive agent-steps count everyone still in the episode, while actionable
    # agent-steps exclude dead/sleeping/resting agents that cannot choose actions.
    state = state.replace(
        alive_agent_steps=state.alive_agent_steps + state.player_alive.sum(),
        actionable_agent_steps=state.actionable_agent_steps + (~cant_do_action).sum(),
    )
    actions = jnp.where(cant_do_action, Action.NOOP.value, actions)

    # Change floor
    state = change_floor(state, actions, params, static_params)

    # Crafting
    rng, _rng = jax.random.split(rng)
    state = do_crafting(_rng, state, actions, params, static_params)

    # Interact (mining, melee attacking, eating plants, drinking water, reviving)
    rng, _rng = jax.random.split(rng)
    state = do_action(_rng, state, actions, params, static_params)
    health_after_action = state.player_health  # snapshot after friendly-fire window

    # Placing
    rng, _rng = jax.random.split(rng)
    state = place_block(_rng, state, actions, params, static_params)

    # Construction (building at construction sites)
    rng, _rng = jax.random.split(rng)
    state = do_construction(_rng, state, actions, params, static_params)

    # Shooting
    state = shoot_projectile(state, actions, static_params)

    # Casting
    state = cast_spell(state, actions, static_params)

    # Potions
    state = drink_potion(state, actions)

    # Read
    state = read_book(state, actions)

    # Enchant
    rng, _rng = jax.random.split(rng)
    state = enchant(_rng, state, actions, params, static_params)

    # Boss
    state = boss_logic(state, static_params)

    # Attributes
    state = level_up_attributes(state, actions, params)

    # Trade
    state = trade_materials(state, actions, static_params)

    # Request Materials
    state = make_request(state, actions)

    # Communication (MPE-style discrete messages)
    state = process_communication(state, actions, static_params)

    # Movement
    state = move_player(state, actions, params, static_params)

    # Mobs
    rng, _rng = jax.random.split(rng)
    state = update_mobs(_rng, state, params, params, static_params)

    rng, _rng = jax.random.split(rng)
    state = spawn_mobs(state, _rng, params, static_params)
    health_after_mobs = state.player_health  # snapshot after mob damage window

    # Plants
    state = update_plants(state, static_params)

    # Intrinsics
    state = update_player_intrinsics(state, actions, static_params)

    # Cap inv
    state = clip_inventory_and_intrinsics(state, params)

    # Inventory achievements
    state = calculate_inventory_achievements(state)

    # Reward
    achievement_coefficients = ACHIEVEMENT_REWARD_MAP
    achievement_reward = (
        (state.achievements.astype(int) - init_achievements.astype(int)) * achievement_coefficients
    ).sum(axis=1)

    # Gain reward if player gained health
    health_reward = (state.player_health - init_health) * 0.1

    individual_reward = achievement_reward + health_reward

    shared_reward = individual_reward.sum().repeat(static_params.player_count)

    reward = jax.lax.select(params.shared_reward, shared_reward, individual_reward)

    player_alive = state.player_health > 0.0

    # Death cause tracking
    died_this_step = init_alive & ~player_alive

    ff_damage = jnp.maximum(init_health - health_after_action, 0.0)
    combat_damage = jnp.maximum(health_after_action - health_after_mobs, 0.0)

    died_from_ff = died_this_step & (ff_damage > 0)
    died_from_combat = died_this_step & ~died_from_ff & (combat_damage > 0)
    died_from_intrinsic = died_this_step & ~died_from_ff & ~died_from_combat

    died_from_starvation = died_from_intrinsic & (state.player_food <= 0)
    died_from_dehydration = died_from_intrinsic & ~died_from_starvation & (state.player_drink <= 0)
    # remaining intrinsic deaths are attributed to exhaustion

    new_death_cause = jnp.where(
        ~died_this_step,
        state.player_death_cause,
        jnp.where(
            died_from_ff,
            DeathCause.FRIENDLY_FIRE,
            jnp.where(
                died_from_combat,
                DeathCause.MOB_COMBAT,
                jnp.where(
                    died_from_starvation,
                    DeathCause.STARVATION,
                    jnp.where(died_from_dehydration, DeathCause.DEHYDRATION, DeathCause.EXHAUSTION),
                ),
            ),
        ),
    )

    new_level_at_death = jnp.where(died_this_step, state.player_level, state.player_level_at_death)

    rng, _rng = jax.random.split(rng)

    state = state.replace(
        player_alive=player_alive,
        player_death_cause=new_death_cause,
        player_level_at_death=new_level_at_death,
        timestep=state.timestep + 1,
        light_level=calculate_light_level(state.timestep + 1, params),
        state_rng=_rng,
    )

    return state, reward
