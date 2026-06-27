import jax
import jax.numpy as jnp

from ..alem_state import EnvState, StaticEnvParams
from ..constants import (
    MAX_OBS_DIM,
    MONSTERS_KILLED_TO_CLEAR_LEVEL,
    OBS_DIM,
    Action,
    BlockType,
    ItemType,
    Specialization,
)
from ..util.game_logic_utils import is_boss_vulnerable


def render_alem_symbolic(state: EnvState, static_params: StaticEnvParams):
    """Render batched flat symbolic observations.

    Args:
        state: Environment state to encode.
        static_params: Static map, player, and entity parameters.

    Returns:
        Symbolic observation vector for every player.
    """
    map = state.map[state.player_level]

    obs_dim_array = jnp.array([OBS_DIM[0], OBS_DIM[1]], dtype=jnp.int32)

    # Map
    padded_grid = jnp.pad(
        map,
        (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
        constant_values=BlockType.OUT_OF_BOUNDS.value,
    )

    tl_corner = state.player_position - obs_dim_array // 2 + MAX_OBS_DIM + 2

    map_view = jax.vmap(jax.lax.dynamic_slice, in_axes=(None, 0, None))(
        padded_grid, tl_corner, OBS_DIM
    )
    map_view_one_hot = jax.nn.one_hot(map_view, num_classes=len(BlockType))

    # Items
    padded_items_map = jnp.pad(
        state.item_map[state.player_level],
        (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
        constant_values=ItemType.NONE.value,
    )

    # Create item map view for each player
    item_map_view = jax.vmap(jax.lax.dynamic_slice, in_axes=(None, 0, None))(
        padded_items_map, tl_corner, OBS_DIM
    )
    item_map_view_one_hot = jax.nn.one_hot(item_map_view, num_classes=len(ItemType))

    # Mobs + Mob Coordination Markers (merged to avoid redundant scans)
    mob_types_per_class = 8
    mob_map = jnp.zeros(
        (static_params.player_count, *OBS_DIM, 5 * mob_types_per_class), dtype=jnp.int32
    )  # 5 classes * 8 types
    # Mob coordination markers: channel 0 = requires_coord, channel 1 = is_hard_coord
    mob_coord_map = jnp.zeros((static_params.player_count, *OBS_DIM, 2), dtype=jnp.float32)

    def _add_mob_to_map(carry, mob_index):
        mob_map, mob_coord_map, mobs, mob_class_index, coord_array = carry

        local_position = -1 * state.player_position + mobs.position[mob_index] + obs_dim_array // 2

        on_screen = jnp.logical_and(local_position >= 0, local_position < obs_dim_array).all(
            axis=-1
        )
        on_screen *= mobs.mask[mob_index]

        mob_identifier = mob_class_index * mob_types_per_class + mobs.type_id[mob_index]

        def _set_mobs_on_map(mob_map, local_position, on_screen):
            return mob_map.at[local_position[0], local_position[1], mob_identifier].set(
                on_screen.astype(jnp.int32)
            )

        mob_map = jax.vmap(_set_mobs_on_map, in_axes=(0, 0, 0))(mob_map, local_position, on_screen)

        # Coordination markers (reuses local_position/on_screen computed above)
        requires_coord = coord_array[mob_index] > 0
        coord_marker = (on_screen & requires_coord).astype(jnp.float32)
        is_hard = coord_array[mob_index] == 2
        hard_marker = (on_screen & is_hard).astype(jnp.float32)

        def _set_coord(mob_coord_map, local_pos, coord_val, hard_val):
            mob_coord_map = mob_coord_map.at[local_pos[0], local_pos[1], 0].set(
                jnp.maximum(mob_coord_map[local_pos[0], local_pos[1], 0], coord_val)
            )
            mob_coord_map = mob_coord_map.at[local_pos[0], local_pos[1], 1].set(
                jnp.maximum(mob_coord_map[local_pos[0], local_pos[1], 1], hard_val)
            )
            return mob_coord_map

        mob_coord_map = jax.vmap(_set_coord, in_axes=(0, 0, 0, 0))(
            mob_coord_map, local_position, coord_marker, hard_marker
        )

        return (mob_map, mob_coord_map, mobs, mob_class_index, coord_array), None

    # Melee mobs (with coordination)
    (mob_map, mob_coord_map, _, _, _), _ = jax.lax.scan(
        _add_mob_to_map,
        (
            mob_map,
            mob_coord_map,
            jax.tree_util.tree_map(lambda x: x[state.player_level], state.melee_mobs),
            0,
            state.melee_mob_coordination[state.player_level],
        ),
        jnp.arange(state.melee_mobs.mask.shape[1]),
    )
    # Passive mobs (with coordination)
    (mob_map, mob_coord_map, _, _, _), _ = jax.lax.scan(
        _add_mob_to_map,
        (
            mob_map,
            mob_coord_map,
            jax.tree_util.tree_map(lambda x: x[state.player_level], state.passive_mobs),
            1,
            state.passive_mob_coordination[state.player_level],
        ),
        jnp.arange(state.passive_mobs.mask.shape[1]),
    )
    # Ranged mobs (with coordination)
    (mob_map, mob_coord_map, _, _, _), _ = jax.lax.scan(
        _add_mob_to_map,
        (
            mob_map,
            mob_coord_map,
            jax.tree_util.tree_map(lambda x: x[state.player_level], state.ranged_mobs),
            2,
            state.ranged_mob_coordination[state.player_level],
        ),
        jnp.arange(state.ranged_mobs.mask.shape[1]),
    )
    # Projectiles (no coordination — pass zeros)
    num_mob_proj = state.mob_projectiles.mask.shape[1]
    (mob_map, mob_coord_map, _, _, _), _ = jax.lax.scan(
        _add_mob_to_map,
        (
            mob_map,
            mob_coord_map,
            jax.tree_util.tree_map(lambda x: x[state.player_level], state.mob_projectiles),
            3,
            jnp.zeros(num_mob_proj, dtype=jnp.int32),
        ),
        jnp.arange(num_mob_proj),
    )
    num_player_proj = state.player_projectiles.mask.shape[1]
    (mob_map, mob_coord_map, _, _, _), _ = jax.lax.scan(
        _add_mob_to_map,
        (
            mob_map,
            mob_coord_map,
            jax.tree_util.tree_map(lambda x: x[state.player_level], state.player_projectiles),
            4,
            jnp.zeros(num_player_proj, dtype=jnp.int32),
        ),
        jnp.arange(num_player_proj),
    )

    # COORDINATION OBSERVATION CHANNELS (5 total spatial channels)

    # All channels below are naturally zero when coordination is disabled
    # (coordination_map, mob coordination arrays, and pending_handovers
    #  are initialized as zeros during world gen).
    #
    # Channel breakdown:
    #   coord_obs       (2): coordination type+magnitude, soft/hard flag
    #   mob_coord_map   (2): requires_coord flag (any elite/large mob), is_hard_coord flag
    #   handover_obs    (1): normalized time remaining (0=inactive)

    # ----- Coordination Requirements (2 channels) -----
    # Shows which blocks require multi-agent coordination to mine/use:
    # - Channel 1: Coordination type and magnitude (sign indicates type, value indicates requirement)
    # - Channel 2: Soft (1) vs Hard (0) coordination flag

    # Extract local view of coordination map for each agent
    padded_coord_map = jnp.pad(
        state.coordination_map[state.player_level].astype(jnp.float32),
        (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
        constant_values=0.0,
    )
    coord_map_view = jax.vmap(jax.lax.dynamic_slice, in_axes=(None, 0, None))(
        padded_coord_map, tl_corner, OBS_DIM
    )

    # Normalize coordination values to approximately [-1, 1]:
    # Sync (positive values): agents required / total players  (e.g., 2/3 = 0.67)
    # Handover (negative values): window size / ceiling        (e.g., -15/-20 = -0.75)
    HANDOVER_WINDOW_CEILING = 20.0  # Conservative upper bound for all difficulty configs

    is_sync_coordination = coord_map_view > 0
    sync_normalized = coord_map_view / static_params.player_count
    handover_normalized = coord_map_view / HANDOVER_WINDOW_CEILING
    coord_value_view = jnp.where(is_sync_coordination, sync_normalized, handover_normalized)

    # Extract soft/hard coordination flag (soft=always works but scales reward, hard=requires N agents)
    padded_soft_mask = jnp.pad(
        state.soft_coordination_mask[state.player_level].astype(jnp.float32),
        (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
        constant_values=0.0,
    )
    soft_mask_view = jax.vmap(jax.lax.dynamic_slice, in_axes=(None, 0, None))(
        padded_soft_mask, tl_corner, OBS_DIM
    )

    # Stack into 2-channel coordination observation
    coord_obs = jnp.stack([coord_value_view, soft_mask_view], axis=-1)

    # ----- Active Handover Tasks (1 channel) -----
    # Shows locations where handover coordination is in-progress (Agent A started, Agent B must complete).
    # Single channel: normalized time remaining [0, 1]. 0.0 = inactive, >0.0 = active with time left.
    # (handover_active is redundant since time_remaining > 0 already implies active)

    pending = (
        state.pending_handovers
    )  # Shape: (max_pending, 6) = [active, x, y, deadline, initiator, build_type]

    # Filter to active handovers that haven't expired
    is_active_handover = (pending[:, 0] == 1) & (pending[:, 3] > state.timestep)
    handover_pos = pending[:, 1:3]  # Extract (x, y) positions

    # Normalize time remaining: 1.0 at start, 0.0 at deadline (assume max window ~15 steps)
    time_remaining = jnp.clip((pending[:, 3] - state.timestep) / 15.0, 0.0, 1.0)

    # Project handovers onto spatial map (use max to handle multiple handovers at same position)
    handover_time_map = (
        jnp.zeros(map.shape, dtype=jnp.float32)
        .at[handover_pos[:, 0], handover_pos[:, 1]]
        .max(jnp.where(is_active_handover, time_remaining, 0.0))
    )

    # Extract local view for each agent
    padded_handover_time = jnp.pad(
        handover_time_map, (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2), constant_values=0.0
    )
    handover_time_view = jax.vmap(jax.lax.dynamic_slice, in_axes=(None, 0, None))(
        padded_handover_time, tl_corner, OBS_DIM
    )

    # Single-channel handover observation
    handover_obs = handover_time_view[..., None]

    def reorder_teammate_info(teammate_info, player_index):
        i1 = (jnp.arange(static_params.player_count) == 0) * player_index
        i2 = jnp.logical_and(
            jnp.arange(static_params.player_count) > 0,
            jnp.arange(static_params.player_count) <= (player_index),
        ) * (jnp.arange(static_params.player_count) - 1)
        i3 = (jnp.arange(static_params.player_count) > player_index) * (
            jnp.arange(static_params.player_count)
        )
        indices = i1 + i2 + i3
        return teammate_info[indices].flatten()

    # Teammate map (One-hot encoding of teammate + bit for dead/alive)
    def _add_teammate(player_index):
        """Creates teammate map for each player"""
        teammate_map = jnp.zeros((*OBS_DIM, static_params.player_count + 1), dtype=jnp.int32)
        local_position = (
            -1 * state.player_position[player_index] + state.player_position + obs_dim_array // 2
        )
        on_screen = jnp.logical_and(local_position >= 0, local_position < obs_dim_array).all(
            axis=-1
        )

        # Add teammate encoding
        teammate_map = teammate_map.at[
            local_position[:, 0],
            local_position[:, 1],
            (
                (jnp.arange(static_params.player_count) < player_index)
                * (jnp.arange(static_params.player_count) + 1)
                + (jnp.arange(static_params.player_count) == player_index) * 0
                + (jnp.arange(static_params.player_count) > player_index)
                * (jnp.arange(static_params.player_count))
            ),
        ].max(on_screen)

        # Add dead/alive bit
        teammate_map = teammate_map.at[local_position[:, 0], local_position[:, 1], -1].set(
            jnp.logical_and(on_screen, state.player_alive)
        )

        """
        Find direction to teammates
        """
        direction_index_2d = jnp.where(
            local_position < 0, 1, jnp.where(local_position >= obs_dim_array, 2, 0)
        )
        direction_index = direction_index_2d[:, 0] * 3 + direction_index_2d[:, 1] - 1
        teammate_directions = jax.nn.one_hot(direction_index, num_classes=8)
        teammate_directions = reorder_teammate_info(teammate_directions, player_index)
        return teammate_map, teammate_directions

    teammate_map, teammate_directions = jax.vmap(_add_teammate, in_axes=0)(
        jnp.arange(static_params.player_count)
    )
    # return teammate_map
    # teammate_map  = jax.vmap(_add_teammate, in_axes=0)(jnp.arange(static_params.player_count))

    # Concat all spatial maps
    # Base:        map_view_one_hot, item_map_view_one_hot, mob_map, teammate_map
    # Coordination: coord_obs(2), mob_coord_map(2), handover_obs(1)
    all_map = jnp.concatenate(
        [
            map_view_one_hot,
            item_map_view_one_hot,
            mob_map,
            teammate_map,
            coord_obs,
            mob_coord_map,
            handover_obs,
        ],
        axis=-1,
    )

    # Light map
    padded_light_map = jnp.pad(
        state.light_map[state.player_level],
        (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
        constant_values=0.0,
    )

    # create light map for each player
    light_map_view = jax.vmap(jax.lax.dynamic_slice, in_axes=(None, 0, None))(
        padded_light_map, tl_corner, OBS_DIM
    )
    light_map_view = light_map_view > 0.05

    # Mask out tiles and mobs in darkness
    all_map = all_map * light_map_view[:, :, :, None]
    all_map = jnp.concatenate((all_map, jnp.expand_dims(light_map_view, axis=-1)), axis=-1)

    # Inventory
    inventory = jnp.stack(
        (
            jnp.sqrt(state.inventory.wood) / 10.0,
            jnp.sqrt(state.inventory.stone) / 10.0,
            jnp.sqrt(state.inventory.coal) / 10.0,
            jnp.sqrt(state.inventory.iron) / 10.0,
            jnp.sqrt(state.inventory.diamond) / 10.0,
            jnp.sqrt(state.inventory.sapphire) / 10.0,
            jnp.sqrt(state.inventory.ruby) / 10.0,
            jnp.sqrt(state.inventory.sapling) / 10.0,
            jnp.sqrt(state.inventory.torches) / 10.0,
            jnp.sqrt(state.inventory.arrows) / 10.0,
            state.inventory.books,
            state.inventory.pickaxe / 4.0,
            state.inventory.sword / 4.0,
            state.sword_enchantment,
            state.bow_enchantment,
            state.inventory.bow,
        ),
        axis=1,
        dtype=jnp.float32,
    )

    potions = jnp.sqrt(state.inventory.potions) / 10.0
    armour = state.inventory.armour / 2.0
    armour_enchantments = state.armour_enchantments

    intrinsics = jnp.stack(
        (
            # state.player_health / 10.0, # -- Removed and placed as part of the teammate dashboard
            state.player_food / 10.0,
            state.player_drink / 10.0,
            state.player_energy / 10.0,
            state.player_mana / 10.0,
            state.player_xp / 10.0,
            state.player_dexterity / 10.0,
            state.player_strength / 10.0,
            state.player_intelligence / 10.0,
        ),
        axis=1,
        dtype=jnp.float32,
    )

    direction = jax.nn.one_hot(state.player_direction - 1, num_classes=4)

    special_values_per_player = jnp.stack(
        (
            state.is_sleeping,
            state.is_resting,
            state.learned_spells,
        ),
        axis=1,
    )
    special_values_level = jnp.array(
        [
            state.light_level,
            state.player_level / jnp.maximum(static_params.num_levels - 1, 1),
            state.monsters_killed[state.player_level] >= MONSTERS_KILLED_TO_CLEAR_LEVEL,
            is_boss_vulnerable(state),
        ]
    )

    """
    Teammate Dashboard
        Includes:
            - Player Health
            - Player Dead or Alive
            - Specialization
            - Requested Material
    Teammate Dashboard appears the same for all players
    """
    players_health = state.player_health / 10.0
    players_alive = state.player_alive
    # Map FORAGER(1)->0, WARRIOR(2)->1, MINER(3)->2; UNASSIGNED(0) clamps to all-zeros
    spec_index = jnp.clip(state.player_specialization - Specialization.FORAGER.value, 0, 2)
    players_specialization = (
        jax.nn.one_hot(spec_index, num_classes=3) * (state.player_specialization > 0)[:, None]
    )
    requested_material = (
        jax.nn.one_hot(
            state.request_type - Action.REQUEST_FOOD.value,
            num_classes=(Action.REQUEST_SAPPHIRE.value - Action.REQUEST_FOOD.value + 1),
        )
        * (state.request_duration > 0)[:, None]
    )
    player_data_parts = [
        players_health[:, None],
        players_alive[:, None],
        players_specialization,
        requested_material,
    ]
    # Append raw communication messages when comm channels are enabled
    if static_params.num_comm_channels > 0:
        player_data_parts.append(state.comm_messages)
    player_data = jnp.concatenate(player_data_parts, axis=-1)
    teammate_dashboard = jax.vmap(lambda i: reorder_teammate_info(player_data, i))(
        jnp.arange(static_params.player_count)
    )

    all_flattened = jnp.concatenate(
        [
            all_map.reshape(all_map.shape[0], -1),
            teammate_dashboard,
            teammate_directions,
            inventory,
            potions,
            intrinsics,
            direction,
            armour,
            armour_enchantments,
            special_values_per_player,
            special_values_level[None, :].repeat(static_params.player_count, axis=0),
        ],
        axis=1,
    )

    return all_flattened
