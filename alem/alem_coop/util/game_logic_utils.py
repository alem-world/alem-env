from __future__ import annotations

from typing import TYPE_CHECKING

import chex
import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from jaxtyping import Array, Bool, Float, Int

from ..alem_state import EnvState, StaticEnvParams
from ..constants import (
    BOSS_FIGHT_EXTRA_DAMAGE,
    CLOSE_BLOCKS,
    MOB_ACHIEVEMENT_MAP,
    MOB_TYPE_DEFENSE_MAPPING,
    SOLID_BLOCK_MAPPING,
    Achievement,
    BlockType,
    Specialization,
)

# For utility functions - functions called more than once in meaningfully different parts of the codebase


def is_fighting_boss(state, static_params):
    """Identify players currently on a real boss level.

    Args:
        state: Current environment state.
        static_params: Static parameters containing the level count.

    Returns:
        Boolean mask indicating players on the final multi-level floor.
    """
    # Must be on last level AND have more than 1 level (to avoid treating single-level envs as boss fights)
    return jnp.logical_and(
        state.player_level == (static_params.num_levels - 1), static_params.num_levels > 1
    )


def is_boss_spawn_wave(state, static_params):
    """Identify players waiting for a boss minion wave to spawn.

    Args:
        state: Current environment state.
        static_params: Static parameters containing the level count.

    Returns:
        Boolean mask for active boss-wave spawn periods.
    """
    return jnp.logical_and(
        is_fighting_boss(state, static_params),
        state.boss_timesteps_to_spawn_this_round >= 1,
    )


def is_boss_vulnerable(state):
    """Check whether all current boss-wave mobs have been cleared.

    Args:
        state: Current environment state.

    Returns:
        Scalar boolean indicating whether the boss can take damage.
    """
    return jnp.logical_and(
        state.melee_mobs.mask[state.player_level].sum() == 0,
        jnp.logical_and(
            state.ranged_mobs.mask[state.player_level].sum() == 0,
            state.boss_timesteps_to_spawn_this_round <= 0,
        ),
    )


def has_beaten_boss(state, static_params):
    """Check whether progression has passed the final configured boss.

    Args:
        state: Current environment state.
        static_params: Static parameters containing the level count.

    Returns:
        Scalar boolean indicating campaign completion.
    """
    return state.boss_progress >= static_params.num_levels - 1


def attack_mob_class(
    state: EnvState,
    doing_attack: Bool[Array, "n_agents"],
    mobs,
    position: Int[Array, "n_agents 2"],
    damage_vector: Float[Array, "n_agents 3"],
    can_get_achievement: Bool[Array, "n_agents"],
    mob_class_index: int,
):
    """Apply one attack batch to a single class of mobs.

    Args:
        state: Current environment state.
        doing_attack: Per-agent mask of active attacks.
        mobs: Mob collection targeted by the attack class.
        position: Per-agent target coordinates.
        damage_vector: Physical, fire, and ice damage per agent.
        can_get_achievement: Agents eligible for the class kill achievement.
        mob_class_index: Index into class-specific defense and achievement tables.

    Returns:
        Updated mobs, kill/attack masks, kill count, and achievements.
    """

    def is_attacking_mob_at_index(mob_index):
        in_mob = (mobs.position[state.player_level, mob_index] == position).all(axis=1)
        return jnp.logical_and(in_mob, mobs.mask[state.player_level, mob_index])

    is_attacking_mob_array = jax.vmap(is_attacking_mob_at_index)(jnp.arange(mobs.mask.shape[1]))
    is_attacking_mob = jnp.logical_and(
        is_attacking_mob_array.sum(axis=0) > 0,
        doing_attack,
    )
    target_mob_index = jnp.argmax(is_attacking_mob_array, axis=0)

    damage = get_damage(
        damage_vector,
        MOB_TYPE_DEFENSE_MAPPING[
            mobs.type_id[state.player_level, target_mob_index], mob_class_index
        ],
    )

    new_mob_health = mobs.health.at[state.player_level, target_mob_index].add(
        -damage * is_attacking_mob
    )
    mobs = mobs.replace(health=new_mob_health)

    old_mask = mobs.mask[state.player_level]
    mobs = mobs.replace(mask=jnp.logical_and(mobs.health > 0, mobs.mask))
    did_kill_mob = jnp.logical_and(
        jnp.logical_and(
            old_mask[target_mob_index],
            jnp.logical_not(mobs.mask[state.player_level, target_mob_index]),
        ),
        is_attacking_mob,
    )

    mobs_killed = jnp.sum(
        jnp.logical_and(
            old_mask,
            jnp.logical_not(mobs.mask[state.player_level]),
        )
    )

    achievement_for_kill = MOB_ACHIEVEMENT_MAP[
        mob_class_index, mobs.type_id[state.player_level, target_mob_index]
    ]

    new_achievements = state.achievements.at[
        jnp.arange(len(state.achievements)), achievement_for_kill
    ].set(
        jnp.logical_or(
            state.achievements[jnp.arange(len(state.achievements)), achievement_for_kill],
            jnp.logical_and(did_kill_mob, can_get_achievement),
        )
    )

    return mobs, did_kill_mob, is_attacking_mob, mobs_killed, new_achievements


def attack_mob_class_with_elite_coordination(
    state,
    doing_attack,
    mobs,
    position,
    damage_vector,
    can_get_achievement,
    mob_class_index,
    mob_coordination,
    mob_agents_required,
    equal_block_placement,
    static_params,
):
    """Extended attack_mob_class that handles per-mob elite coordination.

    Args:
        mob_coordination: Per-mob coordination encoding (0=normal, 1=elite soft, 2=elite hard)
        equal_block_placement: Matrix of agents attacking same position
        static_params: Static environment parameters

    Returns:
        Updated mobs, combat masks and counts, achievements, multiplier, and
        elite coordination attempt/success metrics.
    """

    def is_attacking_mob_at_index(mob_index):
        in_mob = (mobs.position[state.player_level, mob_index] == position).all(axis=1)
        return jnp.logical_and(in_mob, mobs.mask[state.player_level, mob_index])

    is_attacking_mob_array = jax.vmap(is_attacking_mob_at_index)(jnp.arange(mobs.mask.shape[1]))
    is_attacking_mob = jnp.logical_and(
        is_attacking_mob_array.sum(axis=0) > 0,
        doing_attack,
    )
    target_mob_index = jnp.argmax(is_attacking_mob_array, axis=0)

    # Get coordination requirement for the target mob (per-agent)
    mob_coord_req = mob_coordination[state.player_level, target_mob_index]
    is_elite = mob_coord_req > 0
    mob_hard = mob_coord_req == 2

    # Count agents attacking the same mob position
    agents_at_same_pos = (equal_block_placement & is_attacking_mob[:, None]).sum(axis=0)

    # Check if coordination requirement is met (per-mob agents_required)
    agents_req = mob_agents_required[state.player_level, target_mob_index]
    agents_req = jnp.maximum(agents_req, 2)  # fallback for non-elite
    coord_met = agents_at_same_pos >= agents_req

    # Hard coordination: attack fails if not met
    # Soft coordination: attack works but reduced effect
    hard_coord_fails = is_elite & mob_hard & ~coord_met
    attack_succeeds = is_attacking_mob & ~hard_coord_fails

    # Calculate damage with coordination bonuses
    damage = get_damage(
        damage_vector,
        MOB_TYPE_DEFENSE_MAPPING[
            mobs.type_id[state.player_level, target_mob_index], mob_class_index
        ],
    )

    # Coordination multiplier (bonus for coordinating)
    coord_multiplier = jnp.where(
        is_elite & ~mob_hard & coord_met,
        2.0,  # 2x damage/drops if coordinated on soft elite
        jnp.where(
            is_elite & mob_hard & coord_met,
            3.0,  # 3x drops if coordinated on hard elite
            1.0,
        ),
    )
    # For soft elites attacking solo: reduced damage
    soft_solo_penalty = jnp.where(
        is_elite & ~mob_hard & ~coord_met,
        0.5,  # Half damage if solo on soft elite
        1.0,
    )
    effective_damage = damage * coord_multiplier * soft_solo_penalty

    new_mob_health = mobs.health.at[state.player_level, target_mob_index].add(
        -effective_damage * attack_succeeds
    )
    mobs = mobs.replace(health=new_mob_health)

    old_mask = mobs.mask[state.player_level]
    mobs = mobs.replace(mask=jnp.logical_and(mobs.health > 0, mobs.mask))
    did_kill_mob = jnp.logical_and(
        jnp.logical_and(
            old_mask[target_mob_index],
            jnp.logical_not(mobs.mask[state.player_level, target_mob_index]),
        ),
        attack_succeeds,
    )

    mobs_killed = jnp.sum(
        jnp.logical_and(
            old_mask,
            jnp.logical_not(mobs.mask[state.player_level]),
        )
    )

    achievement_for_kill = MOB_ACHIEVEMENT_MAP[
        mob_class_index, mobs.type_id[state.player_level, target_mob_index]
    ]

    new_achievements = state.achievements.at[
        jnp.arange(len(state.achievements)), achievement_for_kill
    ].set(
        jnp.logical_or(
            state.achievements[jnp.arange(len(state.achievements)), achievement_for_kill],
            jnp.logical_and(did_kill_mob, can_get_achievement),
        )
    )

    # Track elite kills with coordination (any agent that killed an elite with coord met)
    elite_killed_with_coord = did_kill_mob & is_elite & coord_met

    # Track elite coordination attempts and successes (for coordination_success_rate)
    elite_coord_attempt = is_attacking_mob & is_elite  # agent attacked an elite mob
    elite_coord_success = elite_coord_attempt & coord_met  # agent met coord requirement

    # Coordination multiplier applies to food drops for passive mobs
    return (
        mobs,
        did_kill_mob,
        is_attacking_mob,
        mobs_killed,
        new_achievements,
        coord_multiplier,
        elite_killed_with_coord,
        elite_coord_attempt,
        elite_coord_success,
    )


def attack_mob(
    state,
    doing_attack,
    position,
    damage_vector,
    can_eat,
    equal_block_placement=None,
    env_params=None,
    static_params=None,
):
    """Resolve player attacks against passive, melee, and ranged mobs.

    Args:
        state: Current environment state.
        doing_attack: Per-agent mask of active attacks.
        position: Per-agent target coordinates.
        damage_vector: Physical, fire, and ice damage per agent.
        can_eat: Per-agent eligibility for passive-mob food rewards.
        equal_block_placement: Optional matrix identifying shared targets.
        env_params: Optional gameplay parameters for elite coordination.
        static_params: Optional static parameters for elite coordination.

    Returns:
        Updated state plus attack, kill, and elite-coordination metrics.
    """
    monsters_killed = state.monsters_killed

    # Use elite coordination if we have the mob coordination arrays
    use_elite_coord = (
        static_params is not None
        and equal_block_placement is not None
        and hasattr(state, "melee_mob_coordination")
    )

    _zero_bool = jnp.zeros(static_params.player_count if static_params else 1, dtype=jnp.bool_)

    if use_elite_coord:
        # Melee with elite coordination
        (
            new_melee_mobs,
            did_kill_melee_mob,
            is_attacking_melee_mob,
            melee_mobs_killed,
            new_achievements,
            _,
            elite_melee_killed_with_coord,
            melee_elite_attempt,
            melee_elite_success,
        ) = attack_mob_class_with_elite_coordination(
            state,
            doing_attack,
            state.melee_mobs,
            position,
            damage_vector,
            True,
            1,
            state.melee_mob_coordination,
            state.melee_mob_agents_required,
            equal_block_placement,
            static_params,
        )
    else:
        # Fallback to original attack_mob_class
        (
            new_melee_mobs,
            did_kill_melee_mob,
            is_attacking_melee_mob,
            melee_mobs_killed,
            new_achievements,
        ) = attack_mob_class(
            state,
            doing_attack,
            state.melee_mobs,
            position,
            damage_vector,
            True,
            1,
        )
        elite_melee_killed_with_coord = _zero_bool
        melee_elite_attempt = _zero_bool
        melee_elite_success = _zero_bool

    monsters_killed = monsters_killed.at[state.player_level].add(melee_mobs_killed)

    state = state.replace(
        melee_mobs=new_melee_mobs,
        achievements=new_achievements,
    )

    if use_elite_coord:
        # Passive mobs with elite coordination (Large Cow/Buffalo)
        (
            new_passive_mobs,
            did_kill_passive_mob,
            is_attacking_passive_mob,
            passive_mobs_killed,
            new_achievements,
            food_coord_multiplier,
            large_passive_killed_with_coord,
            passive_elite_attempt,
            passive_elite_success,
        ) = attack_mob_class_with_elite_coordination(
            state,
            doing_attack,
            state.passive_mobs,
            position,
            damage_vector,
            can_eat,
            0,
            state.passive_mob_coordination,
            state.passive_mob_agents_required,
            equal_block_placement,
            static_params,
        )
        # Food yield scales with coordination for large passives
        base_food = 6
        food_yield = (base_food * food_coord_multiplier).astype(jnp.int32)
    else:
        (
            new_passive_mobs,
            did_kill_passive_mob,
            is_attacking_passive_mob,
            passive_mobs_killed,
            new_achievements,
        ) = attack_mob_class(
            state,
            doing_attack,
            state.passive_mobs,
            position,
            damage_vector,
            can_eat,
            0,
        )
        food_yield = 6
        large_passive_killed_with_coord = _zero_bool
        passive_elite_attempt = _zero_bool
        passive_elite_success = _zero_bool

    new_food = jnp.where(
        jnp.logical_and(did_kill_passive_mob, can_eat),
        jnp.minimum(get_max_food(state), state.player_food + food_yield),
        state.player_food,
    )
    new_hunger = jnp.where(
        jnp.logical_and(did_kill_passive_mob, can_eat),
        0.0,
        state.player_hunger,
    )
    new_achievements = new_achievements.at[:, Achievement.COLLECT_FOOD.value].set(
        jnp.logical_or(
            state.achievements[:, Achievement.COLLECT_FOOD.value],
            jnp.logical_and(did_kill_passive_mob, can_eat),
        )
    )

    state = state.replace(
        passive_mobs=new_passive_mobs,
        player_food=new_food,
        player_hunger=new_hunger,
        achievements=new_achievements,
    )

    if use_elite_coord:
        # Ranged with elite coordination
        (
            new_ranged_mobs,
            did_kill_ranged_mob,
            is_attacking_ranged_mob,
            ranged_mobs_killed,
            new_achievements,
            _,
            elite_ranged_killed_with_coord,
            ranged_elite_attempt,
            ranged_elite_success,
        ) = attack_mob_class_with_elite_coordination(
            state,
            doing_attack,
            state.ranged_mobs,
            position,
            damage_vector,
            True,
            2,
            state.ranged_mob_coordination,
            state.ranged_mob_agents_required,
            equal_block_placement,
            static_params,
        )
    else:
        (
            new_ranged_mobs,
            did_kill_ranged_mob,
            is_attacking_ranged_mob,
            ranged_mobs_killed,
            new_achievements,
        ) = attack_mob_class(
            state,
            doing_attack,
            state.ranged_mobs,
            position,
            damage_vector,
            True,
            2,
        )
        elite_ranged_killed_with_coord = _zero_bool
        ranged_elite_attempt = _zero_bool
        ranged_elite_success = _zero_bool

    monsters_killed = monsters_killed.at[state.player_level].add(ranged_mobs_killed)

    state = state.replace(
        ranged_mobs=new_ranged_mobs,
        achievements=new_achievements,
    )

    # Update mob map on kill
    did_attack_mob = jnp.logical_or(
        jnp.logical_or(is_attacking_melee_mob, is_attacking_passive_mob),
        is_attacking_ranged_mob,
    )

    did_kill_monster = jnp.logical_or(did_kill_melee_mob, did_kill_ranged_mob)
    did_kill_mob = jnp.logical_or(did_kill_monster, did_kill_passive_mob)

    state = state.replace(
        mob_map=state.mob_map.at[state.player_level, position[:, 0], position[:, 1]].min(
            jnp.logical_not(did_kill_mob)
        ),
        monsters_killed=monsters_killed,
    )

    # Per-agent elite kill booleans and aggregated counts for metrics
    elite_melee_kills = elite_melee_killed_with_coord.any().astype(jnp.int32)
    elite_ranged_kills = elite_ranged_killed_with_coord.any().astype(jnp.int32)
    large_passive_kills = large_passive_killed_with_coord.any().astype(jnp.int32)

    # Aggregate elite coordination attempts/successes across all mob classes
    total_elite_attempt = (
        (melee_elite_attempt | passive_elite_attempt | ranged_elite_attempt).any().astype(jnp.int32)
    )
    total_elite_success = (
        (melee_elite_success | passive_elite_success | ranged_elite_success).any().astype(jnp.int32)
    )

    return (
        state,
        did_attack_mob,
        did_kill_mob,
        elite_melee_kills,
        elite_ranged_kills,
        large_passive_kills,
        elite_melee_killed_with_coord,
        elite_ranged_killed_with_coord,
        large_passive_killed_with_coord,
        total_elite_attempt,
        total_elite_success,
    )


def spawn_projectile(
    state,
    static_params,
    projectiles,
    projectile_directions,
    projectile_owners,
    new_projectile_position,
    is_spawning_projectile,
    owner,
    direction,
    projectile_type,
):
    """Insert a projectile into the first free slot on the current level.

    Args:
        state: Current environment state used to select the level.
        static_params: Static entity limits retained by the common call contract.
        projectiles: Projectile entity collection to update.
        projectile_directions: Direction vector for every projectile slot.
        projectile_owners: Owner index for every projectile slot.
        new_projectile_position: Spawn coordinate.
        is_spawning_projectile: Whether this invocation should populate a slot.
        owner: Owner index for the new projectile.
        direction: Travel direction for the new projectile.
        projectile_type: Type identifier for the new projectile.

    Returns:
        Updated projectile entities, directions, and owners.
    """
    new_projectile_index = jnp.argmax(jnp.logical_not(projectiles.mask[state.player_level]))
    new_projectile_position = jax.lax.select(
        is_spawning_projectile,
        new_projectile_position,
        projectiles.position[state.player_level, new_projectile_index],
    )
    new_projectile_mask = jax.lax.select(
        is_spawning_projectile,
        True,
        projectiles.mask[state.player_level, new_projectile_index],
    )
    new_projectile_direction = jax.lax.select(
        is_spawning_projectile,
        direction,
        projectile_directions[state.player_level, new_projectile_index],
    )
    new_projectile_owner = jax.lax.select(
        is_spawning_projectile,
        owner,
        projectile_owners[state.player_level, new_projectile_index],
    )
    new_projectile_type = jax.lax.select(
        is_spawning_projectile,
        projectile_type,
        projectiles.type_id[state.player_level, new_projectile_index],
    )

    new_projectiles = projectiles.replace(
        position=projectiles.position.at[state.player_level, new_projectile_index].set(
            new_projectile_position
        ),
        mask=projectiles.mask.at[state.player_level, new_projectile_index].set(new_projectile_mask),
        type_id=projectiles.type_id.at[state.player_level, new_projectile_index].set(
            new_projectile_type
        ),
    )

    new_projectile_directions = projectile_directions.at[
        state.player_level, new_projectile_index
    ].set(new_projectile_direction)

    new_projectile_owners = projectile_owners.at[state.player_level, new_projectile_index].set(
        new_projectile_owner
    )

    return new_projectiles, new_projectile_directions, new_projectile_owners


def get_damage_done_to_player(state, static_params, damage_vector):
    """Apply boss scaling and player defense to incoming damage.

    Args:
        state: Current environment state.
        static_params: Static parameters used to identify boss combat.
        damage_vector: Incoming physical, fire, and ice damage.

    Returns:
        Effective damage received by each player.
    """
    damage_vector *= 1 + is_fighting_boss(state, static_params) * BOSS_FIGHT_EXTRA_DAMAGE
    defense_vector = get_player_defense_vector(state)
    return get_damage(damage_vector, defense_vector)


def get_damage_between_players(state, other_player_index):
    """Calculate friendly-fire damage against selected players.

    Args:
        state: Current environment state.
        other_player_index: Target player index for each attacker.

    Returns:
        Effective damage dealt to the selected players.
    """
    # Damage player inflicts on other player
    damage_vector = get_player_damage_vector(state) * (
        1 + 2.5 * state.is_sleeping[other_player_index, None]
    )

    # Defense of damaged player
    defense_vector = get_player_defense_vector(state)[other_player_index]

    return get_damage(damage_vector, defense_vector)


def get_player_damage_vector(state):
    """Build each player's physical, fire, and ice damage vector.

    Args:
        state: Current equipment, attributes, and specialization state.

    Returns:
        Damage array with one three-channel vector per player.
    """
    physical_damages = jnp.array(
        [1, 2, 3, 5, 8],
        dtype=jnp.int32,
    )
    physical_damage = physical_damages[state.inventory.sword] * (
        1 + (state.player_specialization == Specialization.WARRIOR.value)
    )  # warrior has 2x base damage
    fire_damage = physical_damage * (state.sword_enchantment == 1) * 0.5
    ice_damage = physical_damage * (state.sword_enchantment == 2) * 0.5

    physical_damage *= 1 + 0.25 * (state.player_strength - 1)  # Strength=5 does double damage
    fire_damage *= 1 + 0.05 * (state.player_intelligence - 1)  # Int=5 does 25% more enchant damage
    ice_damage *= 1 + 0.05 * (state.player_intelligence - 1)  # Int=5 does 25% more enchant damage

    return jnp.stack([physical_damage, fire_damage, ice_damage], axis=1)


def get_player_defense_vector(state):
    """Build each player's physical, fire, and ice defense vector.

    Args:
        state: Current armour and enchantment state.

    Returns:
        Defense array with one three-channel vector per player.
    """
    scaled_defenses = jnp.stack(
        [
            state.inventory.armour * 0.1,
            (state.armour_enchantments == 1) * 0.2,
            (state.armour_enchantments == 2) * 0.2,
        ],
        axis=1,
    )
    defense_vector = scaled_defenses.sum(axis=2)
    return defense_vector


def get_damage(
    damage_vector: Float[Array, "... 3"],
    defense_vector: Float[Array, "... 3"],
) -> Float[Array, ...]:
    """Reduce typed damage channels after applying matching defenses.

    Args:
        damage_vector: Incoming physical, fire, and ice damage.
        defense_vector: Fractional defense for the same channels.

    Returns:
        Total effective damage across all channels.
    """
    damages = (1.0 - defense_vector) * damage_vector
    return damages.sum(axis=-1)


def in_bounds(position: Int[Array, "n 2"], static_params: StaticEnvParams) -> Bool[Array, "n"]:
    """Check a coordinate batch against the configured map bounds.

    Args:
        position: Row/column coordinate pairs.
        static_params: Static parameters containing the map dimensions.

    Returns:
        Boolean validity mask for the coordinate batch.
    """
    in_bounds_x = jnp.logical_and(0 <= position[:, 0], position[:, 0] < static_params.map_size[0])
    in_bounds_y = jnp.logical_and(0 <= position[:, 1], position[:, 1] < static_params.map_size[1])
    return jnp.logical_and(in_bounds_x, in_bounds_y)


def is_in_solid_block(level_map, position):
    """Check whether positions contain collision-blocking tiles.

    Args:
        level_map: Block identifiers for one world level.
        position: Row/column coordinate pairs to inspect.

    Returns:
        Boolean solid-block mask for the coordinates.
    """
    return SOLID_BLOCK_MAPPING[level_map[position[:, 0], position[:, 1]]]


def is_position_not_colliding_other_player(state, position):
    """Check proposed player positions for current or mutual collisions.

    Args:
        state: Current environment state.
        position: Proposed position for each player.

    Returns:
        Boolean mask of collision-free proposals.
    """
    # Verify that next step isn't in another player's next position
    next_pos_clash = jnp.fill_diagonal(
        (jnp.expand_dims(position, axis=1) == jnp.expand_dims(position, axis=0)).all(axis=2),
        False,
        inplace=False,
    )
    next_pos_clash = next_pos_clash.any(axis=1)

    # Verify that next step isn't in another player's current position
    curr_pos_clash = is_in_other_player(state, position)

    return jnp.logical_not(jnp.logical_or(next_pos_clash, curr_pos_clash))


def is_position_in_bounds_not_in_mob_not_colliding(state, position, collision_map, static_params):
    """Validate proposed movement against bounds, mobs, terrain, and abilities.

    Args:
        state: Current environment state.
        position: Proposed position for each player.
        collision_map: Per-player permissions for ground, water, and lava.
        static_params: Static parameters containing map dimensions.

    Returns:
        Boolean mask of valid movement proposals.
    """
    pos_in_bounds = in_bounds(position, static_params)
    in_solid_block = is_in_solid_block(state.map[state.player_level], position)
    in_mob = is_in_mob(state, position)
    in_lava = state.map[state.player_level][position[:, 0], position[:, 1]] == BlockType.LAVA.value
    in_water = (
        state.map[state.player_level][position[:, 0], position[:, 1]] == BlockType.WATER.value
    )
    on_ground_block = jnp.logical_and(
        jnp.logical_not(in_solid_block),
        jnp.logical_and(jnp.logical_not(in_water), jnp.logical_not(in_lava)),
    )

    valid_move = jnp.logical_and(
        pos_in_bounds,
        jnp.logical_and(jnp.logical_not(in_mob), jnp.logical_not(in_solid_block)),
    )

    # Ground blocks
    valid_move = jnp.logical_and(
        valid_move,
        jnp.logical_or(jnp.logical_not(collision_map[0]), jnp.logical_not(on_ground_block)),
    )

    # Water
    valid_move = jnp.logical_and(
        valid_move,
        jnp.logical_or(jnp.logical_not(collision_map[1]), jnp.logical_not(in_water)),
    )

    # Lava
    valid_move = jnp.logical_and(
        valid_move,
        jnp.logical_or(jnp.logical_not(collision_map[2]), jnp.logical_not(in_lava)),
    )

    return valid_move


def is_near_block(state, block_type, static_params):
    """Check whether each player is adjacent to a block type.

    Args:
        state: Current environment state.
        block_type: Integer block identifier to search for.
        static_params: Static parameters containing map dimensions.

    Returns:
        Boolean adjacency mask for all players.
    """
    close_blocks = jax.vmap(jnp.add, in_axes=(0, None))(state.player_position, CLOSE_BLOCKS)
    in_bound_blocks = jax.vmap(in_bounds, in_axes=(0, None))(close_blocks, static_params)
    correct_blocks = (
        state.map[state.player_level, close_blocks[:, :, 0], close_blocks[:, :, 1]] == block_type
    )
    return (jnp.logical_and(in_bound_blocks, correct_blocks)).any(axis=1)


def get_nearest_block_pos(state, block_type, static_params):
    """For each agent, returns the position of an adjacent block of the given type.
    Returns (-1, -1) for agents with no adjacent matching block.

    Args:
        state: Current environment state.
        block_type: Integer block identifier to search for.
        static_params: Static parameters containing map dimensions.

    Returns:
        has_adj_block: (num_agents,) bool — agent has at least one adjacent matching block
        adj_block_pos: (num_agents, 2) int — position of an adjacent matching block, or (-1,-1)
    """
    has_adj_block = is_near_block(state, block_type, static_params)  # (num_agents,)

    close_blocks = jax.vmap(jnp.add, in_axes=(0, None))(
        state.player_position, CLOSE_BLOCKS
    )  # (num_agents, num_close_blocks, 2)
    in_bound_blocks = jax.vmap(in_bounds, in_axes=(0, None))(close_blocks, static_params)
    is_adj_block = (
        state.map[state.player_level, close_blocks[:, :, 0], close_blocks[:, :, 1]] == block_type
    ) & in_bound_blocks  # (num_agents, num_close_blocks)
    block_idx = jnp.argmax(is_adj_block, axis=1)  # (num_agents,)
    adj_pos = close_blocks[jnp.arange(static_params.player_count), block_idx]
    # Use sentinel (-1, -1) for agents with no adjacent matching block
    adj_block_pos = jnp.where(
        has_adj_block[:, None],
        adj_pos,
        jnp.full((2,), -1, dtype=jnp.int32),
    )  # (num_agents, 2)
    return has_adj_block, adj_block_pos


def calculate_light_level(timestep, params):
    """Calculate ambient daylight from the current day-cycle position.

    Args:
        timestep: Current episode timestep.
        params: Environment parameters containing day length.

    Returns:
        Ambient light intensity in the unit interval.
    """
    progress = (timestep / params.day_length) % 1 + 0.3
    return 1 - jnp.abs(jnp.cos(jnp.pi * progress)) ** 3


def is_in_other_player(state: EnvState, position: chex.Array):
    """Check whether coordinates overlap any current player position.

    Args:
        state: Current environment state.
        position: Coordinate batch to inspect.

    Returns:
        Boolean overlap mask for the coordinates.
    """
    is_pos_in_other_player = (
        jnp.expand_dims(state.player_position, axis=1) == jnp.expand_dims(position, axis=0)
    ).all(axis=2)
    is_pos_in_other_player = is_pos_in_other_player.any(axis=0)
    return is_pos_in_other_player


def is_in_mob(state: EnvState, position: chex.Array):
    """Read the current level's mob occupancy at a coordinate batch.

    Args:
        state: Current environment state.
        position: Coordinate batch to inspect.

    Returns:
        Mob occupancy values at the coordinates.
    """
    return state.mob_map[state.player_level, position[:, 0], position[:, 1]]


def get_max_health(state):
    """Return each player's strength-scaled health capacity.

    Args:
        state: Current player attributes.

    Returns:
        Maximum health for every player.
    """
    return 8 + state.player_strength


def get_max_food(state):
    """Return each player's dexterity- and role-scaled food capacity.

    Args:
        state: Current player attributes and specializations.

    Returns:
        Maximum food for every player.
    """
    return (7 + 2 * state.player_dexterity) * (
        1 + (state.player_specialization == Specialization.FORAGER.value) * 2
    )


def get_max_drink(state):
    """Return each player's dexterity- and role-scaled drink capacity.

    Args:
        state: Current player attributes and specializations.

    Returns:
        Maximum drink for every player.
    """
    return (7 + 2 * state.player_dexterity) * (
        1 + (state.player_specialization == Specialization.FORAGER.value) * 2
    )


def get_max_energy(state):
    """Return each player's dexterity-scaled energy capacity.

    Args:
        state: Current player attributes.

    Returns:
        Maximum energy for every player.
    """
    return 7 + 2 * state.player_dexterity


def get_max_mana(state):
    """Return each player's intelligence-scaled mana capacity.

    Args:
        state: Current player attributes.

    Returns:
        Maximum mana for every player.
    """
    return 6 + 3 * state.player_intelligence


def clip_inventory_and_intrinsics(state, params):
    """Clamp inventory and intrinsic values to their legal ranges.

    Args:
        state: Environment state to normalize.
        params: Gameplay parameters including god-mode behavior.

    Returns:
        State with bounded inventories and player resources.
    """
    capped_inv = jax.tree_util.tree_map(lambda x: jnp.minimum(x, 99), state.inventory)

    min_health = jax.lax.select(params.god_mode, 9, 0)

    state = state.replace(
        inventory=capped_inv,
        player_health=jnp.minimum(
            jnp.maximum(state.player_health, min_health), get_max_health(state)
        ),
        player_food=jnp.minimum(jnp.maximum(state.player_food, 0), get_max_food(state)),
        player_drink=jnp.minimum(jnp.maximum(state.player_drink, 0), get_max_drink(state)),
        player_energy=jnp.minimum(jnp.maximum(state.player_energy, 0), get_max_energy(state)),
        player_mana=jnp.minimum(jnp.maximum(state.player_mana, 0), get_max_mana(state)),
    )

    return state


def find_valid_ladder_areas(valid_ladder_map, player_count):
    """Find horizontal map spans that fit all player ladder positions.

    Args:
        valid_ladder_map: Boolean map of individually valid ladder tiles.
        player_count: Number of players that must fit at two-tile spacing.

    Returns:
        Boolean map marking valid leftmost ladder positions.
    """
    d = player_count * 2 - 1
    s = jnp.ones((d,))

    valid_areas = jax.vmap(jnp.convolve, in_axes=(0, None, None))(valid_ladder_map, s, "valid")
    valid_areas = valid_areas == d
    valid_areas = jnp.pad(
        valid_areas, ((0, 0), (0, valid_ladder_map.shape[1] - valid_areas.shape[1]))
    )
    return valid_areas


def get_ladder_positions(rng, static_params, config, map):
    """Sample a valid row of evenly spaced player ladders.

    Args:
        rng: JAX random key used to select the ladder span.
        static_params: Static map dimensions and player count.
        config: Level configuration defining valid ladder terrain.
        map: Block identifiers for the generated level.

    Returns:
        One ladder coordinate per player.
    """
    valid_ladder_down = (map == config.valid_ladder).astype(jnp.float32)
    valid_ladder_down = find_valid_ladder_areas(
        valid_ladder_down, static_params.player_count
    ).flatten()
    ladder_index = jax.random.choice(
        rng,
        jnp.arange(static_params.map_size[0] * static_params.map_size[1]),
        p=valid_ladder_down / valid_ladder_down.sum(),
    )
    ladder_positions_corner = jnp.array(
        [
            ladder_index // static_params.map_size[0],
            ladder_index % static_params.map_size[0],
        ]
    )
    ladder_positions = jnp.array(
        [
            ladder_positions_corner[0].repeat(static_params.player_count),
            ladder_positions_corner[1] + jnp.arange(static_params.player_count) * 2,
        ]
    ).T
    return ladder_positions


def get_player_icon_positions(player_count):
    """Lay out player portrait icons across two dashboard columns.

    Args:
        player_count: Number of player icons to position.

    Returns:
        Row/column dashboard coordinates for each icon.
    """
    col1 = jnp.arange((player_count + 1) // 2)

    col2_values = jnp.array([0, 6])

    col2 = jnp.tile(col2_values, (player_count + 1) // len(col2_values))
    col1 = jnp.repeat(col1, len(col2_values))[:player_count]

    result = jnp.stack((col1, col2[:player_count]), axis=-1)

    return result
