# Debug environment with only the overworld (single level)
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import chex
import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    pass

from ..alem_state import EnvParams, EnvState, StaticEnvParams
from ..constants import (
    BOSS_FIGHT_SPAWN_TURNS,
    NUM_ACHIEVEMENTS,
    Action,
    BlockType,
    Specialization,
)
from ..util.game_logic_utils import calculate_light_level
from ..world_gen.world_gen import (
    create_projectiles,
    generate_coordination_map,
    generate_empty_mobs,
    generate_smoothworld,
    get_new_empty_inventory,
    get_new_full_inventory,
    place_construction_sites,
)
from ..world_gen.world_gen_configs import OVERWORLD_CONFIG
from .alem_symbolic_env import AlemCoopSymbolicEnv


def generate_world_single_level(
    rng: chex.PRNGKey, params: EnvParams, static_params: StaticEnvParams
) -> EnvState:
    """Generate a debug world containing only the overworld.

    Args:
        rng: JAX random key used for procedural generation.
        params: Dynamic gameplay and generation parameters.
        static_params: Static map, player, and entity parameters.

    Returns:
        Fully initialized single-level environment state.
    """

    # Start players in the middle of the map
    def get_player_spawn(idx):
        width = jnp.ceil(jnp.sqrt(static_params.player_count)).astype(jnp.int32)
        return jnp.array(
            [
                (static_params.map_size[0] // 2) + (idx // width),
                (static_params.map_size[1] // 2) + (idx % width),
            ]
        )

    player_position = jax.vmap(get_player_spawn)(jnp.arange(0, static_params.player_count))

    # Fix player specializations
    player_specialization_order = jnp.array(
        [Specialization.WARRIOR.value, Specialization.FORAGER.value, Specialization.MINER.value]
    )
    player_specializations = player_specialization_order[jnp.arange(static_params.player_count) % 3]

    # Generate only the overworld
    rng, _rng = jax.random.split(rng)
    map, item_map, light_map, ladders_down, ladders_up = generate_smoothworld(
        _rng, static_params, player_position, OVERWORLD_CONFIG
    )

    # Add level dimension (shape: [1, ...])
    map = jnp.expand_dims(map, axis=0)
    item_map = jnp.expand_dims(item_map, axis=0)
    light_map = jnp.expand_dims(light_map, axis=0)
    ladders_down = jnp.expand_dims(ladders_down, axis=0)
    ladders_up = jnp.expand_dims(ladders_up, axis=0)

    melee_mobs = generate_empty_mobs(
        static_params.max_melee_mobs * static_params.player_count, static_params.num_levels
    )
    ranged_mobs = generate_empty_mobs(
        static_params.max_ranged_mobs * static_params.player_count, static_params.num_levels
    )
    passive_mobs = generate_empty_mobs(
        static_params.max_passive_mobs * static_params.player_count, static_params.num_levels
    )

    mob_projectiles, mob_projectile_directions, mob_projectile_owners = create_projectiles(
        static_params.max_mob_projectiles * static_params.player_count, static_params.num_levels
    )
    player_projectiles, player_projectile_directions, player_projectile_owners = create_projectiles(
        static_params.max_player_projectiles * static_params.player_count, static_params.num_levels
    )

    # Plants
    growing_plants_positions = jnp.zeros(
        (static_params.max_growing_plants * static_params.player_count, 2), dtype=jnp.int32
    )
    growing_plants_age = jnp.zeros(
        static_params.max_growing_plants * static_params.player_count, dtype=jnp.int32
    )
    growing_plants_mask = jnp.zeros(
        static_params.max_growing_plants * static_params.player_count, dtype=bool
    )

    # Potion mapping for episode
    rng, _rng = jax.random.split(rng)
    potion_mapping = jax.random.permutation(_rng, jnp.arange(6))

    # Generate coordination map for the single level
    rng, coord_rng = jax.random.split(rng)
    coord_map_level, soft_mask_level = generate_coordination_map(
        coord_rng, map[0], params, static_params
    )
    coordination_map = jnp.expand_dims(coord_map_level, axis=0)  # Add level dimension
    soft_coordination_mask = jnp.expand_dims(soft_mask_level, axis=0)

    # Place construction sites
    rng, construction_rng = jax.random.split(rng)
    construction_map, site_positions, site_coord, site_soft = place_construction_sites(
        construction_rng, map[0], params.num_construction_sites, params, static_params
    )
    map = jnp.where(params.construction_enabled, map.at[0].set(construction_map), map)
    construction_site_positions = (
        jnp.zeros(
            (static_params.num_levels, static_params.max_construction_sites, 2), dtype=jnp.int32
        )
        .at[0]
        .set(jnp.where(params.construction_enabled, site_positions, jnp.zeros_like(site_positions)))
    )
    construction_site_coord = (
        jnp.zeros((static_params.num_levels, static_params.max_construction_sites), dtype=jnp.int32)
        .at[0]
        .set(jnp.where(params.construction_enabled, site_coord, jnp.zeros_like(site_coord)))
    )
    construction_site_soft = (
        jnp.zeros((static_params.num_levels, static_params.max_construction_sites), dtype=jnp.bool_)
        .at[0]
        .set(jnp.where(params.construction_enabled, site_soft, jnp.zeros_like(site_soft)))
    )

    # Apply construction site coordination values and soft mask to maps (vectorized)
    all_pos = construction_site_positions[0]  # (max_construction_sites, 2)
    all_coord_val = construction_site_coord[0]  # (max_construction_sites,)
    all_soft_val = construction_site_soft[0]  # (max_construction_sites,)
    block_at_pos = map[0, all_pos[:, 0], all_pos[:, 1]]
    is_valid = (all_coord_val != 0) & (block_at_pos == BlockType.CONSTRUCTION_SITE.value)

    existing_coord = coordination_map[0, all_pos[:, 0], all_pos[:, 1]]
    coordination_map = coordination_map.at[0, all_pos[:, 0], all_pos[:, 1]].set(
        jnp.where(is_valid, all_coord_val, existing_coord)
    )
    existing_soft = soft_coordination_mask[0, all_pos[:, 0], all_pos[:, 1]]
    soft_coordination_mask = soft_coordination_mask.at[0, all_pos[:, 0], all_pos[:, 1]].set(
        jnp.where(is_valid, all_soft_val, existing_soft)
    )

    # Initialize pending handovers
    pending_handovers = jnp.zeros((static_params.max_pending_handovers, 6), dtype=jnp.int32)

    # Inventory
    inventory = jax.tree_util.tree_map(
        lambda x, y: jax.lax.select(params.god_mode, x, y),
        get_new_full_inventory(static_params.player_count),
        get_new_empty_inventory(static_params.player_count),
    )

    rng, _rng = jax.random.split(rng)

    state = EnvState(
        map=map,
        item_map=item_map,
        mob_map=jnp.zeros((static_params.num_levels, *static_params.map_size), dtype=bool),
        light_map=light_map,
        down_ladders=ladders_down,
        up_ladders=ladders_up,
        chests_opened=jnp.zeros((static_params.num_levels, static_params.player_count), dtype=bool),
        monsters_killed=jnp.zeros(static_params.num_levels, dtype=jnp.int32)
        .at[0]
        .set(10),  # First ladder starts open
        player_position=player_position,
        player_direction=jnp.full((static_params.player_count,), Action.UP.value, dtype=jnp.int32),
        player_level=jnp.asarray(0, dtype=jnp.int32),
        player_health=jnp.full(
            (static_params.player_count,), 9.0, dtype=jnp.float32
        ),  # health rests back — scaling it has no lasting effect
        player_alive=jnp.full((static_params.player_count,), True, dtype=bool),
        player_death_cause=jnp.zeros((static_params.player_count,), dtype=jnp.int32),
        player_level_at_death=jnp.full((static_params.player_count,), -1, dtype=jnp.int32),
        player_food=jnp.full(
            (static_params.player_count,),
            jnp.round(9 * params.starting_resource_multiplier).astype(jnp.int32),
            dtype=jnp.int32,
        ),
        player_drink=jnp.full(
            (static_params.player_count,),
            jnp.round(9 * params.starting_resource_multiplier).astype(jnp.int32),
            dtype=jnp.int32,
        ),
        player_energy=jnp.full(
            (static_params.player_count,),
            jnp.round(9 * params.starting_resource_multiplier).astype(jnp.int32),
            dtype=jnp.int32,
        ),
        player_mana=jnp.full(
            (static_params.player_count,), 9, dtype=jnp.int32
        ),  # mana unaffected — not a survival resource
        player_recover=jnp.full((static_params.player_count,), 0.0, dtype=jnp.float32),
        player_hunger=jnp.full((static_params.player_count,), 0.0, dtype=jnp.float32),
        player_thirst=jnp.full((static_params.player_count,), 0.0, dtype=jnp.float32),
        player_fatigue=jnp.full((static_params.player_count,), 0.0, dtype=jnp.float32),
        player_recover_mana=jnp.full((static_params.player_count,), 0.0, dtype=jnp.float32),
        is_sleeping=jnp.full((static_params.player_count,), False, dtype=jnp.bool_),
        is_resting=jnp.full((static_params.player_count,), False, dtype=jnp.bool_),
        player_xp=jnp.full((static_params.player_count,), 0, dtype=jnp.int32),
        player_dexterity=jnp.full((static_params.player_count,), 1, dtype=jnp.int32),
        player_strength=jnp.full((static_params.player_count,), 1, dtype=jnp.int32),
        player_intelligence=jnp.full((static_params.player_count,), 1, dtype=jnp.int32),
        player_specialization=player_specializations,
        request_duration=jnp.full((static_params.player_count,), 0, dtype=jnp.int32),
        request_type=jnp.full((static_params.player_count,), 0, dtype=jnp.int32),
        inventory=inventory,
        sword_enchantment=jnp.full((static_params.player_count,), 0, dtype=jnp.int32),
        bow_enchantment=jnp.full((static_params.player_count,), 0, dtype=jnp.int32),
        armour_enchantments=jnp.full((static_params.player_count, 4), 0, dtype=jnp.int32),
        melee_mobs=melee_mobs,
        ranged_mobs=ranged_mobs,
        passive_mobs=passive_mobs,
        mob_projectiles=mob_projectiles,
        mob_projectile_directions=mob_projectile_directions,
        mob_projectile_owners=mob_projectile_owners,
        player_projectiles=player_projectiles,
        player_projectile_directions=player_projectile_directions,
        player_projectile_owners=player_projectile_owners,
        growing_plants_positions=growing_plants_positions,
        growing_plants_age=growing_plants_age,
        growing_plants_mask=growing_plants_mask,
        potion_mapping=potion_mapping,
        learned_spells=jnp.full((static_params.player_count,), False, dtype=jnp.bool_),
        boss_progress=jnp.asarray(0, dtype=jnp.int32),
        boss_timesteps_to_spawn_this_round=jnp.asarray(BOSS_FIGHT_SPAWN_TURNS, dtype=jnp.int32),
        achievements=jnp.zeros((static_params.player_count, NUM_ACHIEVEMENTS), dtype=bool),
        light_level=jnp.asarray(calculate_light_level(0, params), dtype=jnp.float32),
        trade_count=jnp.zeros((static_params.player_count,), dtype=jnp.int32),
        food_trade_count=jnp.zeros((static_params.player_count,), dtype=jnp.int32),
        drink_trade_count=jnp.zeros((static_params.player_count,), dtype=jnp.int32),
        give_attempt_count=jnp.zeros((static_params.player_count,), dtype=jnp.int32),
        request_count=jnp.zeros((static_params.player_count,), dtype=jnp.int32),
        request_expiry_count=jnp.zeros((static_params.player_count,), dtype=jnp.int32),
        request_received_count=jnp.zeros((static_params.player_count,), dtype=jnp.int32),
        revives=jnp.asarray(0, dtype=jnp.int32),
        ff_damage_dealt=jnp.zeros((static_params.player_count,), dtype=jnp.float32),
        alive_agent_steps=jnp.asarray(0, dtype=jnp.int32),
        actionable_agent_steps=jnp.asarray(0, dtype=jnp.int32),
        max_player_level=jnp.asarray(0, dtype=jnp.int32),
        all_necessities_frac=jnp.ones((static_params.player_count,), dtype=jnp.float32),
        coordination_map=coordination_map,
        soft_coordination_mask=soft_coordination_mask,
        pending_handovers=pending_handovers,
        handover_successes=jnp.asarray(0, dtype=jnp.int32),
        handover_setups=jnp.asarray(0, dtype=jnp.int32),
        handover_expiries=jnp.asarray(0, dtype=jnp.int32),
        sync_coord_by_agents=jnp.zeros(2, dtype=jnp.int32),
        # Domain-specific coordination metrics
        coord_mine_sync_soft_count=jnp.asarray(0, dtype=jnp.int32),
        coord_mine_sync_hard_count=jnp.asarray(0, dtype=jnp.int32),
        coord_mine_handover_count=jnp.asarray(0, dtype=jnp.int32),
        coord_mine_handover_expiries=jnp.asarray(0, dtype=jnp.int32),
        coord_sync_attempts=jnp.asarray(0, dtype=jnp.int32),
        coord_sync_successes=jnp.asarray(0, dtype=jnp.int32),
        soft_sync_events=jnp.asarray(0, dtype=jnp.int32),
        soft_sync_bonus_events=jnp.asarray(0, dtype=jnp.int32),
        coord_solo_soft_attempts=jnp.asarray(0, dtype=jnp.int32),
        coord_solo_soft_successes=jnp.asarray(0, dtype=jnp.int32),
        coord_construction_attempts=jnp.asarray(0, dtype=jnp.int32),
        coord_construction_successes=jnp.asarray(0, dtype=jnp.int32),
        coord_construction_handover_count=jnp.asarray(0, dtype=jnp.int32),
        coord_construction_handover_setups=jnp.asarray(0, dtype=jnp.int32),
        coord_construction_handover_expiries=jnp.asarray(0, dtype=jnp.int32),
        construction_build_at_site_unfunded=jnp.asarray(0, dtype=jnp.int32),
        coord_build_shelter_count=jnp.asarray(0, dtype=jnp.int32),
        coord_build_forge_count=jnp.asarray(0, dtype=jnp.int32),
        coord_build_beacon_count=jnp.asarray(0, dtype=jnp.int32),
        coord_elite_attempts=jnp.asarray(0, dtype=jnp.int32),
        coord_elite_successes=jnp.asarray(0, dtype=jnp.int32),
        coord_elite_melee_kills=jnp.asarray(0, dtype=jnp.int32),
        coord_elite_ranged_kills=jnp.asarray(0, dtype=jnp.int32),
        coord_large_passive_kills=jnp.asarray(0, dtype=jnp.int32),
        coord_craft_attempts=jnp.asarray(0, dtype=jnp.int32),
        coord_craft_successes=jnp.asarray(0, dtype=jnp.int32),
        coord_diamond_pickaxe_count=jnp.asarray(0, dtype=jnp.int32),
        coord_diamond_sword_count=jnp.asarray(0, dtype=jnp.int32),
        coord_diamond_armour_count=jnp.asarray(0, dtype=jnp.int32),
        # Construction sites
        construction_sites_built=jnp.zeros(
            (static_params.num_levels, static_params.max_construction_sites), dtype=jnp.int32
        ),
        construction_site_positions=construction_site_positions,
        construction_handover_deadline=jnp.zeros(
            (static_params.num_levels, static_params.max_construction_sites), dtype=jnp.int32
        ),
        # Mob coordination
        melee_mob_coordination=jnp.zeros(
            (static_params.num_levels, static_params.max_melee_mobs * static_params.player_count),
            dtype=jnp.int32,
        ),
        ranged_mob_coordination=jnp.zeros(
            (static_params.num_levels, static_params.max_ranged_mobs * static_params.player_count),
            dtype=jnp.int32,
        ),
        passive_mob_coordination=jnp.zeros(
            (static_params.num_levels, static_params.max_passive_mobs * static_params.player_count),
            dtype=jnp.int32,
        ),
        # Per-mob agents_required
        melee_mob_agents_required=jnp.zeros(
            (static_params.num_levels, static_params.max_melee_mobs * static_params.player_count),
            dtype=jnp.int32,
        ),
        ranged_mob_agents_required=jnp.zeros(
            (static_params.num_levels, static_params.max_ranged_mobs * static_params.player_count),
            dtype=jnp.int32,
        ),
        passive_mob_agents_required=jnp.zeros(
            (static_params.num_levels, static_params.max_passive_mobs * static_params.player_count),
            dtype=jnp.int32,
        ),
        # Communication messages
        comm_messages=jnp.zeros(
            (static_params.player_count, static_params.num_comm_channels), dtype=jnp.float32
        ),
        comm_count=jnp.zeros(static_params.player_count, dtype=jnp.int32),
        sampled_alpha=jnp.asarray(0.0, dtype=jnp.float32),
        state_rng=_rng,
        timestep=jnp.asarray(0, dtype=jnp.int32),
    )

    return state


class AlemCoopSymbolicEnvDebug(AlemCoopSymbolicEnv):
    """Debug environment with only the overworld (single level)."""

    def __init__(
        self,
        num_agents: int = 3,
        env_params: EnvParams = None,
        static_env_params: StaticEnvParams = None,
        compute_full_info: bool = True,
    ):
        """Initialize the single-level debug environment.

        Args:
            num_agents: Number of players used when static parameters are omitted.
            env_params: Optional episode and gameplay parameters.
            static_env_params: Optional static parameters for the debug world.
            compute_full_info: Whether steps should calculate all score metrics.
        """
        if static_env_params is not None:
            self.static_env_params = static_env_params
        elif num_agents is not None:
            self.static_env_params = StaticEnvParams(player_count=num_agents, num_levels=1)
        else:
            self.static_env_params = AlemCoopSymbolicEnvDebug.default_static_params()
        self.num_agents = self.static_env_params.player_count
        self._env_params = env_params  # Store custom params if provided
        self.compute_full_info = compute_full_info

        self.agents = [f"agent_{i}" for i in range(self.static_env_params.player_count)]
        self.action_spaces = {name: self.action_shape() for name in self.agents}
        self.observation_spaces = {name: self.observation_shape() for name in self.agents}

    @staticmethod
    def default_static_params() -> StaticEnvParams:
        """Return static parameters configured for one world level.

        Returns:
            Default static parameters with ``num_levels`` set to one.
        """
        return StaticEnvParams(num_levels=1)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey, _=None) -> tuple[dict[str, chex.Array], EnvState]:
        """Generate a new single-level world and symbolic observations.

        Args:
            key: JAX random key used for world generation.
            _: Ignored compatibility parameter.

        Returns:
            Per-agent observations and the initial debug state.
        """
        state = generate_world_single_level(key, self.default_params, self.static_env_params)
        return self.get_obs(state), state

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check terminal conditions without a boss-completion test.

        Args:
            state: Current environment state.
            params: Dynamic parameters containing the time limit.

        Returns:
            Scalar boolean indicating time expiry or total player death.
        """
        done_steps = state.timestep >= params.max_timesteps
        is_dead = jnp.logical_not(state.player_alive).all()
        return jnp.logical_or(is_dead, done_steps)
