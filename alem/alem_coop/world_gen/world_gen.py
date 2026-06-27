from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from ..alem_state import EnvState, Inventory, Mobs, _jax_difficulty_params_from_alpha
from ..constants import (
    BOSS_FIGHT_SPAWN_TURNS,
    MAX_ROOM_SIZE,
    MIN_ROOM_SIZE,
    NUM_ACHIEVEMENTS,
    NUM_ROOMS,
    TORCH_LIGHT_MAP,
    Action,
    BlockType,
    ItemType,
    Specialization,
)
from ..util.game_logic_utils import calculate_light_level, get_ladder_positions
from ..util.maths_utils import get_all_players_distance_map
from ..util.noise import generate_fractal_noise_2d
from .world_gen_configs import (
    ALL_DUNGEON_CONFIGS,
    ALL_SMOOTHGEN_CONFIGS,
)

if TYPE_CHECKING:
    import chex
    from jaxtyping import Array, Bool, Float, Int

    from ..alem_state import EnvParams, StaticEnvParams


def generate_coordination_map(
    rng: chex.PRNGKey,
    block_map: Int[Array, "map_height map_width"],
    params: EnvParams,
    static_params: StaticEnvParams,
) -> tuple[Int[Array, "map_height map_width"], Bool[Array, "map_height map_width"]]:
    """Generate coordination requirements for a single level.

    Args:
        rng: JAX random key used to sample coordination requirements.
        block_map: Generated block map whose eligible resources may require coordination.
        params: Gameplay parameters controlling coordination frequency and difficulty.
        static_params: Static parameters including player count and map shape.

    Returns:
        coordination_map: 0=none, positive=sync (N agents), negative=handover (window)
        soft_mask: True=soft (scales reward), False=hard (requires N agents)
    """
    if not params.coordination_enabled:
        zeros = jnp.zeros_like(block_map, dtype=jnp.int32)
        return zeros, jnp.zeros_like(block_map, dtype=jnp.bool_)

    # Eligible blocks for coordination - mining blocks only (no placement)
    mining_eligible = jnp.isin(
        block_map,
        jnp.array(
            [
                BlockType.TREE.value,
                BlockType.FIRE_TREE.value,
                BlockType.ICE_SHRUB.value,
                BlockType.STONE.value,
                BlockType.COAL.value,
                BlockType.IRON.value,
                BlockType.DIAMOND.value,
                BlockType.SAPPHIRE.value,
                BlockType.RUBY.value,
            ]
        ),
    )
    # REMOVED: placement_eligible - replaced by construction sites
    eligible = mining_eligible

    # Sample ~coordination_probability % of eligible blocks for coordination
    rng, k1, k2, k3, k4, k5 = jax.random.split(rng, 6)
    coord_mask = (
        jax.random.uniform(k1, block_map.shape) < params.coordination_probability
    ) & eligible

    # Split: synchronous vs handover
    is_handover = jax.random.uniform(k2, block_map.shape) < params.handover_ratio

    # Split: soft vs hard (only for sync - handover is always hard)
    is_soft = jax.random.uniform(k5, block_map.shape) < params.soft_coordination_ratio

    # Synchronous: agents required via p_max_agents binary sampling
    max_agents_rolls = jax.random.uniform(k3, block_map.shape)
    sync_agents = jnp.where(max_agents_rolls < params.p_max_agents, static_params.player_count, 2)

    # Handover: window size (negative value)
    handover_window = -jax.random.randint(
        k4, block_map.shape, params.handover_window_min, params.handover_window_max + 1
    )

    # Combine: positive=sync, negative=handover
    coord_values = jnp.where(is_handover, handover_window, sync_agents)
    coordination_map = jnp.where(coord_mask, coord_values, 0).astype(jnp.int32)

    # Soft mask: only for sync blocks (handover is always hard)
    soft_mask = coord_mask & is_soft & ~is_handover

    return coordination_map, soft_mask


def get_new_empty_inventory(player_count: int) -> Inventory:
    """Create zero-filled inventory arrays for a player batch.

    Args:
        player_count: Number of inventory rows to allocate.

    Returns:
        An empty inventory for every player.
    """
    return Inventory(
        wood=jnp.full((player_count,), 0, dtype=jnp.int32),
        stone=jnp.full((player_count,), 0, dtype=jnp.int32),
        coal=jnp.full((player_count,), 0, dtype=jnp.int32),
        iron=jnp.full((player_count,), 0, dtype=jnp.int32),
        diamond=jnp.full((player_count,), 0, dtype=jnp.int32),
        sapling=jnp.full((player_count,), 0, dtype=jnp.int32),
        pickaxe=jnp.full((player_count,), 0, dtype=jnp.int32),
        sword=jnp.full((player_count,), 0, dtype=jnp.int32),
        bow=jnp.full((player_count,), 0, dtype=jnp.int32),
        arrows=jnp.full((player_count,), 0, dtype=jnp.int32),
        torches=jnp.full((player_count,), 0, dtype=jnp.int32),
        ruby=jnp.full((player_count,), 0, dtype=jnp.int32),
        sapphire=jnp.full((player_count,), 0, dtype=jnp.int32),
        books=jnp.full((player_count,), 0, dtype=jnp.int32),
        potions=jnp.full((player_count, 6), 0, dtype=jnp.int32),
        armour=jnp.full((player_count, 4), 0, dtype=jnp.int32),
    )


def get_new_full_inventory(player_count: int) -> Inventory:
    """Create debug inventories filled to each item's useful maximum.

    Args:
        player_count: Number of inventory rows to allocate.

    Returns:
        A fully stocked inventory for every player.
    """
    return Inventory(
        wood=jnp.full((player_count,), 99, dtype=jnp.int32),
        stone=jnp.full((player_count,), 99, dtype=jnp.int32),
        coal=jnp.full((player_count,), 99, dtype=jnp.int32),
        iron=jnp.full((player_count,), 99, dtype=jnp.int32),
        diamond=jnp.full((player_count,), 99, dtype=jnp.int32),
        sapling=jnp.full((player_count,), 99, dtype=jnp.int32),
        pickaxe=jnp.full((player_count,), 4, dtype=jnp.int32),
        sword=jnp.full((player_count,), 4, dtype=jnp.int32),
        bow=jnp.full((player_count,), 1, dtype=jnp.int32),
        arrows=jnp.full((player_count,), 99, dtype=jnp.int32),
        torches=jnp.full((player_count,), 99, dtype=jnp.int32),
        ruby=jnp.full((player_count,), 99, dtype=jnp.int32),
        sapphire=jnp.full((player_count,), 99, dtype=jnp.int32),
        books=jnp.full((player_count,), 99, dtype=jnp.int32),
        potions=jnp.full((player_count, 6), 99, dtype=jnp.int32),
        armour=jnp.full((player_count, 4), 2, dtype=jnp.int32),
    )


def generate_dungeon(
    rng: chex.PRNGKey, static_params: StaticEnvParams, config
) -> tuple[
    Int[Array, "map_height map_width"],
    Int[Array, "map_height map_width"],
    Float[Array, "map_height map_width"],
    Int[Array, "num_ladders 2"],
    Int[Array, "num_ladders 2"],
]:
    """Generate a room-and-corridor dungeon level.

    Args:
        rng: JAX random key used for procedural generation.
        static_params: Static map dimensions and entity limits.
        config: Dungeon-specific block, item, and lighting configuration.

    Returns:
        Block map, item map, light map, down-ladder positions, and up-ladder positions.
    """
    chunk_size = 16
    world_chunk_width = static_params.map_size[0] // chunk_size
    world_chunk_height = static_params.map_size[1] // chunk_size
    room_occupancy_chunks = jnp.ones(world_chunk_width * world_chunk_height)

    rng, _rng, __rng = jax.random.split(rng, 3)
    room_sizes = jax.random.randint(
        __rng, shape=(NUM_ROOMS, 2), minval=MIN_ROOM_SIZE, maxval=MAX_ROOM_SIZE
    )

    map = jnp.ones(static_params.map_size, dtype=jnp.int32) * BlockType.WALL.value
    padded_map = jnp.pad(map, MAX_ROOM_SIZE, constant_values=0)

    item_map = jnp.zeros(static_params.map_size, dtype=jnp.int32)
    padded_item_map = jnp.pad(item_map, MAX_ROOM_SIZE, constant_values=0)

    def _add_room(carry, room_index):
        block_map, item_map, room_occupancy_chunks, rng = carry

        rng, _rng = jax.random.split(rng)
        room_chunk = jax.random.choice(
            _rng,
            jnp.arange(world_chunk_width * world_chunk_height),
            p=room_occupancy_chunks,
        )
        room_occupancy_chunks = room_occupancy_chunks.at[room_chunk].set(0)

        room_position = jnp.array(
            [
                (room_chunk % world_chunk_height) * chunk_size,
                (room_chunk // world_chunk_height) * chunk_size,
            ]
        ) + jnp.array([MAX_ROOM_SIZE, MAX_ROOM_SIZE])
        rng, _rng = jax.random.split(rng)
        room_position += jax.random.randint(_rng, (2,), minval=0, maxval=chunk_size - MIN_ROOM_SIZE)

        slice = jax.lax.dynamic_slice(block_map, room_position, (MAX_ROOM_SIZE, MAX_ROOM_SIZE))
        xs = jnp.expand_dims(jnp.arange(MAX_ROOM_SIZE), axis=-1).repeat(MAX_ROOM_SIZE, axis=-1)
        ys = jnp.expand_dims(jnp.arange(MAX_ROOM_SIZE), axis=0).repeat(MAX_ROOM_SIZE, axis=0)

        room_mask = jnp.logical_and(xs < room_sizes[room_index, 0], ys < room_sizes[room_index, 1])

        slice = room_mask * BlockType.PATH.value + (1 - room_mask) * slice

        block_map = jax.lax.dynamic_update_slice(
            block_map,
            slice,
            room_position,
        )

        # Torches in corner
        item_map = item_map.at[room_position[0], room_position[1]].set(ItemType.TORCH.value)
        item_map = item_map.at[
            room_position[0] + room_sizes[room_index, 0] - 1, room_position[1]
        ].set(ItemType.TORCH.value)
        item_map = item_map.at[
            room_position[0], room_position[1] + room_sizes[room_index, 1] - 1
        ].set(ItemType.TORCH.value)
        item_map = item_map.at[
            room_position[0] + room_sizes[room_index, 0] - 1,
            room_position[1] + room_sizes[room_index, 1] - 1,
        ].set(ItemType.TORCH.value)

        # Chest
        rng, _rng = jax.random.split(rng)
        chest_position = jax.random.randint(
            _rng,
            shape=(static_params.player_count, 2),
            minval=jnp.ones(2),
            maxval=room_sizes[room_index] - jnp.ones(2),
        )
        block_map = block_map.at[
            room_position[0] + chest_position[:, 0], room_position[1] + chest_position[:, 1]
        ].set(BlockType.CHEST.value)

        # Fountain
        rng, _rng, __rng = jax.random.split(rng, 3)
        fountain_position = jax.random.randint(
            _rng,
            shape=(2,),
            minval=jnp.ones(2),
            maxval=room_sizes[room_index] - jnp.ones(2),
        )
        room_has_fountain = jax.random.uniform(__rng) > 0.5
        fountain_block = (
            room_has_fountain * config.fountain_block
            + (1 - room_has_fountain)
            * block_map[
                room_position[0] + fountain_position[0],
                room_position[1] + fountain_position[1],
            ]
        )
        block_map = block_map.at[
            room_position[0] + fountain_position[0],
            room_position[1] + fountain_position[1],
        ].set(fountain_block)

        return (block_map, item_map, room_occupancy_chunks, rng), room_position

    rng, _rng = jax.random.split(rng)
    (padded_map, padded_item_map, _, _), room_positions = jax.lax.scan(
        _add_room,
        (padded_map, padded_item_map, room_occupancy_chunks, _rng),
        jnp.arange(NUM_ROOMS),
    )

    def _add_path(carry, path_index):
        cmap, included_rooms_mask, rng = carry

        path_source = room_positions[path_index]

        rng, _rng = jax.random.split(rng)
        sink_index = jax.random.choice(_rng, jnp.arange(NUM_ROOMS), p=included_rooms_mask)
        path_sink = room_positions[sink_index]

        # Horizontal component
        entire_row = cmap[path_source[0]]
        path_indexes = jnp.arange(static_params.map_size[0] + 2 * MAX_ROOM_SIZE)
        path_indexes = path_indexes - path_source[1]
        horizontal_distance = path_sink[1] - path_source[1]
        path_indexes = path_indexes * jnp.sign(horizontal_distance)

        horizontal_mask = jnp.logical_and(
            path_indexes >= 0, path_indexes <= jnp.abs(horizontal_distance)
        )
        horizontal_mask = jnp.logical_and(horizontal_mask, jnp.sign(horizontal_distance))
        horizontal_mask = jnp.logical_and(horizontal_mask, entire_row == BlockType.WALL.value)

        new_row = horizontal_mask * BlockType.PATH.value + (1 - horizontal_mask) * entire_row

        cmap = jax.lax.dynamic_update_slice(
            cmap,
            jnp.expand_dims(new_row, axis=0),
            path_source,
        )

        # Vertical component
        entire_col = cmap[:, path_sink[1]]
        path_indexes = jnp.arange(static_params.map_size[1] + 2 * MAX_ROOM_SIZE)
        path_indexes = path_indexes - path_source[0]
        vertical_distance = path_sink[0] - path_source[0]
        path_indexes = path_indexes * jnp.sign(vertical_distance)

        vertical_mask = jnp.logical_and(
            path_indexes >= 0, path_indexes <= jnp.abs(vertical_distance)
        )
        vertical_mask = jnp.logical_and(vertical_mask, jnp.sign(vertical_distance))

        vertical_mask = jnp.logical_and(vertical_mask, entire_col == BlockType.WALL.value)

        new_col = vertical_mask * BlockType.PATH.value + (1 - vertical_mask) * entire_col

        cmap = jax.lax.dynamic_update_slice(
            cmap,
            jnp.expand_dims(new_col, axis=-1),
            path_sink,
        )

        rng, _rng = jax.random.split(rng)
        included_rooms_mask = included_rooms_mask.at[path_index].set(True)
        return (cmap, included_rooms_mask, _rng), None

    rng, _rng = jax.random.split(rng)
    included_rooms_mask = jnp.zeros(NUM_ROOMS, dtype=bool).at[-1].set(True)
    (
        (padded_map, _, _),
        _,
    ) = jax.lax.scan(_add_path, (padded_map, included_rooms_mask, _rng), jnp.arange(0, NUM_ROOMS))

    # Place special block in a random room
    special_block_position = room_positions[0] + jnp.array([2, 2])
    padded_map = padded_map.at[special_block_position[0], special_block_position[1]].set(
        config.special_block
    )

    map = padded_map[MAX_ROOM_SIZE:-MAX_ROOM_SIZE, MAX_ROOM_SIZE:-MAX_ROOM_SIZE]
    item_map = padded_item_map[MAX_ROOM_SIZE:-MAX_ROOM_SIZE, MAX_ROOM_SIZE:-MAX_ROOM_SIZE]

    # Visual stuff
    c_path_map = map != BlockType.WALL.value
    z = jnp.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    adj_path_map = jsp.signal.convolve(c_path_map, z, mode="same")
    adj_path_map = adj_path_map > 0.5

    rng, _rng = jax.random.split(rng)
    rare_map = jax.random.choice(
        _rng,
        jnp.array([False, True]),
        static_params.map_size,
        p=jnp.array([0.9, 0.1]),
    )

    wall_map = rare_map * BlockType.WALL_MOSS.value + (1 - rare_map) * BlockType.WALL.value

    rare_map = jnp.logical_and(rare_map, map == BlockType.PATH.value)
    rare_map = jnp.logical_and(rare_map, item_map == ItemType.NONE.value)
    path_map = rare_map * config.rare_path_replacement_block + (1 - rare_map) * map

    is_wall_map = jnp.logical_and(map == BlockType.WALL.value, adj_path_map)
    is_darkness_map = jnp.logical_not(adj_path_map)
    is_path_map = jnp.logical_not(jnp.logical_or(is_wall_map, is_darkness_map))

    map = (
        is_path_map * path_map + is_wall_map * wall_map + is_darkness_map * BlockType.DARKNESS.value
    )

    light_map = jnp.ones(static_params.map_size, dtype=jnp.float32)

    # Ladders
    rng, _rng = jax.random.split(rng)
    ladders_down = get_ladder_positions(_rng, static_params, config, map)
    item_map = item_map.at[ladders_down[:, 0], ladders_down[:, 1]].set(ItemType.LADDER_DOWN.value)

    rng, _rng = jax.random.split(rng)
    ladders_up = get_ladder_positions(_rng, static_params, config, map)
    item_map = item_map.at[ladders_up[:, 0], ladders_up[:, 1]].set(ItemType.LADDER_UP.value)

    return map, item_map, light_map, ladders_down, ladders_up


def generate_smoothworld(
    rng: chex.PRNGKey,
    static_params: StaticEnvParams,
    player_position: Int[Array, "player_count 2"],
    config,
    params: EnvParams = None,
) -> tuple[
    Int[Array, "map_height map_width"],
    Int[Array, "map_height map_width"],
    Float[Array, "map_height map_width"],
    Int[Array, "num_ladders 2"],
    Int[Array, "num_ladders 2"],
]:
    """Generate a noise-based outdoor or cavern level.

    Args:
        rng: JAX random key used for procedural generation.
        static_params: Static map dimensions and entity limits.
        player_position: Initial player coordinates kept clear during generation.
        config: Biome-specific terrain, resource, and lighting configuration.
        params: Optional environment parameters containing fixed noise angles.

    Returns:
        Block map, item map, light map, down-ladder positions, and up-ladder positions.
    """
    if params is not None:
        fractal_noise_angles = params.fractal_noise_angles
    else:
        fractal_noise_angles = (None, None, None, None, None)

    player_proximity_map = get_all_players_distance_map(
        player_position, jnp.full(static_params.player_count, True), static_params
    )
    player_proximity_map_water = player_proximity_map / config.player_proximity_map_water_strength
    player_proximity_map_water = jnp.clip(
        player_proximity_map_water, 0.0, config.player_proximity_map_water_max
    )

    player_proximity_map_mountain = (
        player_proximity_map / config.player_proximity_map_mountain_strength
    )
    player_proximity_map_mountain = jnp.clip(
        player_proximity_map_mountain,
        0.0,
        config.player_proximity_map_mountain_max,
    )

    larger_res = (static_params.map_size[0] // 4, static_params.map_size[1] // 4)
    small_res = (static_params.map_size[0] // 16, static_params.map_size[1] // 16)
    x_res = (static_params.map_size[0] // 8, static_params.map_size[1] // 2)

    rng, _rng = jax.random.split(rng)
    water = generate_fractal_noise_2d(
        _rng,
        static_params.map_size,
        small_res,
        octaves=1,
        override_angles=fractal_noise_angles[0],
    )
    water = water + player_proximity_map_water - 1.0

    # Water
    rng, _rng = jax.random.split(rng)
    map = jnp.where(water > config.water_threshold, config.sea_block, config.default_block)

    sand_map = jnp.logical_and(
        water > config.sand_threshold,
        map != config.sea_block,
    )

    map = jnp.where(sand_map, config.coast_block, map)

    # Mountain vs grass
    mountain_threshold = 0.7

    rng, _rng = jax.random.split(rng)
    mountain = (
        generate_fractal_noise_2d(
            _rng,
            static_params.map_size,
            small_res,
            octaves=1,
            override_angles=fractal_noise_angles[1],
        )
        + 0.05
    )
    mountain = mountain + player_proximity_map_mountain - 1.0
    map = jnp.where(mountain > mountain_threshold, config.mountain_block, map)

    # Paths
    rng, _rng = jax.random.split(rng)
    path_x = generate_fractal_noise_2d(
        _rng,
        static_params.map_size,
        x_res,
        octaves=1,
        override_angles=fractal_noise_angles[2],
    )
    path = jnp.logical_and(mountain > mountain_threshold, path_x > 0.8)
    map = jnp.where(path > 0.5, config.path_block, map)

    path_y = path_x.T
    path = jnp.logical_and(mountain > mountain_threshold, path_y > 0.8)
    map = jnp.where(path > 0.5, config.path_block, map)

    # Caves
    rng, _rng = jax.random.split(rng)
    caves = jnp.logical_and(mountain > 0.85, water > 0.4)
    map = jnp.where(caves > 0.5, config.inner_mountain_block, map)

    # Trees
    rng, _rng = jax.random.split(rng)
    tree_noise = generate_fractal_noise_2d(
        _rng,
        static_params.map_size,
        larger_res,
        octaves=1,
        override_angles=fractal_noise_angles[3],
    )
    tree = (tree_noise > config.tree_threshold_perlin) * jax.random.uniform(
        rng, shape=static_params.map_size
    ) > config.tree_threshold_uniform
    tree = jnp.logical_and(tree, map == config.tree_requirement_block)
    map = jnp.where(tree, config.tree, map)

    # Ores
    def _add_ore(carry, index):
        rng, map = carry
        rng, _rng = jax.random.split(rng)
        ore_map = jnp.logical_and(
            map == config.ore_requirement_blocks[index],
            jax.random.uniform(_rng, static_params.map_size) < config.ore_chances[index],
        )
        map = jnp.where(ore_map, config.ores[index], map)

        return (rng, map), None

    rng, _rng = jax.random.split(rng)
    (_, map), _ = jax.lax.scan(_add_ore, (_rng, map), jnp.arange(5))

    # Lava
    lava_map = jnp.logical_and(
        mountain > 0.85,
        tree_noise > 0.7,
    )
    map = jnp.where(lava_map, config.lava, map)

    # Light map
    light_map = jnp.ones(static_params.map_size, dtype=jnp.float32) * config.default_light

    # Make sure player spawns on grass
    map = map.at[player_position[:, 0], player_position[:, 1]].set(config.player_spawn)

    item_map = jnp.zeros(static_params.map_size, dtype=jnp.int32)

    rng, _rng = jax.random.split(rng)
    ladders_down = get_ladder_positions(_rng, static_params, config, map)

    item_map = item_map.at[ladders_down[:, 0], ladders_down[:, 1]].set(
        ItemType.LADDER_DOWN.value * config.ladder_down
        + map[ladders_down[:, 0], ladders_down[:, 1]] * (1 - config.ladder_down)
    )

    rng, _rng = jax.random.split(rng)
    ladders_up = get_ladder_positions(_rng, static_params, config, map)

    LIGHT_MAP_AROUND_LADDER = TORCH_LIGHT_MAP * (
        1 - config.default_light
    ) + config.default_light * jnp.ones((9, 9))

    def _set_ladder_light(light_map, ladder_position):
        out = jax.lax.dynamic_update_slice(
            light_map, LIGHT_MAP_AROUND_LADDER, ladder_position - jnp.array([4, 4])
        )
        return out, None

    light_map, _ = jax.lax.scan(_set_ladder_light, light_map, ladders_up)
    light_map, _ = jax.lax.scan(_set_ladder_light, light_map, ladders_down)

    z = jnp.array([[0.2, 0.7, 0.2], [0.7, 1, 0.7], [0.2, 0.7, 0.2]]) * (
        config.lava == BlockType.LAVA.value
    )
    light_map += jsp.signal.convolve(lava_map, z, mode="same")
    light_map = jnp.clip(light_map, 0.0, 1.0)

    item_map = item_map.at[ladders_up[:, 0], ladders_up[:, 1]].set(
        ItemType.LADDER_UP.value * config.ladder_up
        + map[ladders_up[:, 0], ladders_up[:, 1]] * (1 - config.ladder_up)
    )

    return map, item_map, light_map, ladders_down, ladders_up


def place_construction_sites(
    rng: chex.PRNGKey,
    block_map: Int[Array, "map_height map_width"],
    num_sites,
    params: EnvParams,
    static_params: StaticEnvParams,
) -> tuple[
    Int[Array, "map_height map_width"],
    Int[Array, "max_construction_sites 2"],
    Int[Array, "max_construction_sites"],
    Bool[Array, "max_construction_sites"],
]:
    """Place construction sites at valid locations spread across map.

    Construction sites support both sync and handover coordination:
    - Sync: 2 agents build simultaneously
    - Handover: Agent A starts foundation, Agent B completes within time window

    Args:
        rng: Random key
        block_map: Current level map
        num_sites: Number of sites to place (can be a traced JAX value)
        params: Environment parameters
        static_params: Static environment parameters

    Returns:
        block_map: Updated map with CONSTRUCTION_SITE blocks
        site_positions: Positions of sites, shape: (max_sites, 2)
        site_coordination: Coordination requirements for each site
        site_soft_mask: Whether each site uses soft coordination (bool array)
    """
    if not params.construction_enabled:
        # Return unchanged map and empty site arrays
        empty_positions = jnp.zeros((static_params.max_construction_sites, 2), dtype=jnp.int32)
        empty_coord = jnp.zeros(static_params.max_construction_sites, dtype=jnp.int32)
        empty_soft = jnp.zeros(static_params.max_construction_sites, dtype=jnp.bool_)
        return block_map, empty_positions, empty_coord, empty_soft

    # Valid placement: grass or path, away from edges
    valid = jnp.isin(block_map, jnp.array([BlockType.GRASS.value, BlockType.PATH.value]))

    # Exclude edges (5 blocks from border)
    edge_buffer = 5
    valid = valid.at[:edge_buffer, :].set(False)
    valid = valid.at[-edge_buffer:, :].set(False)
    valid = valid.at[:, :edge_buffer].set(False)
    valid = valid.at[:, -edge_buffer:].set(False)

    # Flatten for sampling
    flat_valid = valid.flatten()
    num_positions = flat_valid.shape[0]

    # Always scan over max_construction_sites; num_sites gates which iterations
    # actually place a site (supports traced num_sites for per-level variation)
    rng, *site_rngs = jax.random.split(rng, static_params.max_construction_sites + 1)

    def _sample_site(carry, inputs):
        block_map, placed_mask = carry
        site_rng, site_idx = inputs

        is_active = site_idx < num_sites

        # Valid positions not yet used
        available = flat_valid & ~placed_mask

        # Sample a position (weighted by availability)
        probs = available.astype(jnp.float32)
        probs = probs / jnp.maximum(probs.sum(), 1.0)

        flat_idx = jax.random.choice(site_rng, jnp.arange(num_positions), p=probs)

        # Convert back to 2D
        pos_x = flat_idx // static_params.map_size[1]
        pos_y = flat_idx % static_params.map_size[1]

        # Mark as placed (only if active)
        placed_mask = placed_mask.at[flat_idx].set(placed_mask[flat_idx] | is_active)

        # Place construction site (only if active and available)
        should_place = available.any() & is_active
        new_block = jnp.where(
            should_place, BlockType.CONSTRUCTION_SITE.value, block_map[pos_x, pos_y]
        )
        block_map = block_map.at[pos_x, pos_y].set(new_block)

        position = jnp.where(
            is_active & available.any(), jnp.array([pos_x, pos_y]), jnp.zeros(2, dtype=jnp.int32)
        )
        return (block_map, placed_mask), position

    site_rngs = jnp.stack(site_rngs)
    placed_mask = jnp.zeros(num_positions, dtype=jnp.bool_)
    (block_map, _), site_positions = jax.lax.scan(
        _sample_site,
        (block_map, placed_mask),
        (site_rngs, jnp.arange(static_params.max_construction_sites)),
    )

    # Generate coordination requirements for each site
    rng, coord_rng = jax.random.split(rng)
    is_handover = (
        jax.random.uniform(coord_rng, (static_params.max_construction_sites,))
        < params.handover_ratio
    )

    rng, window_rng = jax.random.split(rng)
    handover_window = jax.random.randint(
        window_rng,
        (static_params.max_construction_sites,),
        params.handover_window_min,
        params.handover_window_max + 1,
    )

    # Sample agents_required per site via p_max_agents
    rng, agents_rng = jax.random.split(rng)
    agents_rolls = jax.random.uniform(agents_rng, (static_params.max_construction_sites,))
    sync_agents_req = jnp.where(agents_rolls < params.p_max_agents, static_params.player_count, 2)

    # Soft/hard mask using construction-specific ratio
    # (only applies to sync sites — handover is always hard)
    rng, soft_rng = jax.random.split(rng)
    site_soft_mask = (
        jax.random.uniform(soft_rng, (static_params.max_construction_sites,))
        < params.soft_construction_ratio
    )
    site_soft_mask = site_soft_mask & ~is_handover  # handover sites are always hard
    # Zero out for inactive slots
    site_soft_mask = site_soft_mask & (jnp.arange(static_params.max_construction_sites) < num_sites)

    # Sync uses positive agents_required, handover uses negative window size
    # Zero out coordination for inactive site slots
    site_coordination = jnp.where(is_handover, -handover_window, sync_agents_req).astype(jnp.int32)
    site_coordination = jnp.where(
        jnp.arange(static_params.max_construction_sites) < num_sites, site_coordination, 0
    )

    return block_map, site_positions, site_coordination, site_soft_mask


# Mobs
def generate_empty_mobs(max_mobs: int, num_levels: int) -> Mobs:
    """Allocate an inactive mob collection for all levels.

    Args:
        max_mobs: Mob slots allocated per level.
        num_levels: Number of world levels.

    Returns:
        A ``Mobs`` structure with every slot inactive.
    """
    return Mobs(
        position=jnp.zeros((num_levels, max_mobs, 2), dtype=jnp.int32),
        health=jnp.ones((num_levels, max_mobs), dtype=jnp.float32),
        mask=jnp.zeros((num_levels, max_mobs), dtype=bool),
        attack_cooldown=jnp.zeros((num_levels, max_mobs), dtype=jnp.int32),
        type_id=jnp.zeros((num_levels, max_mobs), dtype=jnp.int32),
    )


# Projectiles
def create_projectiles(
    max_num: int, num_levels: int
) -> tuple[Mobs, Int[Array, "num_levels max_num 2"], Int[Array, "num_levels max_num"]]:
    """Allocate inactive projectile state and its direction/owner arrays.

    Args:
        max_num: Projectile slots allocated per level.
        num_levels: Number of world levels.

    Returns:
        Projectile entities, direction vectors, and owner indices.
    """
    projectiles = generate_empty_mobs(max_num, num_levels)
    projectile_directions = jnp.ones((num_levels, max_num, 2), dtype=jnp.int32)

    projectile_owners = jnp.zeros((num_levels, max_num), dtype=jnp.int32)

    return projectiles, projectile_directions, projectile_owners


def generate_world(
    rng: chex.PRNGKey, params: EnvParams, static_params: StaticEnvParams
) -> EnvState:
    """Generate every level and initialize a complete environment state.

    Args:
        rng: JAX random key used for procedural generation and difficulty sampling.
        params: Gameplay and generation parameters.
        static_params: Static shapes, player count, and entity limits.

    Returns:
        Fully initialized environment state ready for the first step.
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

    # Sample α for randomized difficulty (domain randomization)
    if params.randomize_alpha:
        rng, alpha_rng = jax.random.split(rng)
        sampled_alpha = jax.random.uniform(
            alpha_rng, minval=params.alpha_min, maxval=params.alpha_max
        )
        dp = _jax_difficulty_params_from_alpha(
            sampled_alpha, scale_base=params.scale_base_difficulty
        )
        params = params.replace(**dp)
    else:
        sampled_alpha = jnp.float32(0.0)

    # Fix player specializations
    player_specialization_order = jnp.array(
        [Specialization.WARRIOR.value, Specialization.FORAGER.value, Specialization.MINER.value]
    )
    player_specializations = player_specialization_order[jnp.arange(static_params.player_count) % 3]

    # Generate smoothgens (overworld, caves, elemental levels, boss level)
    rngs = jax.random.split(rng, 7)
    rng, _rng = rngs[0], rngs[1:]
    smoothgens = jax.vmap(generate_smoothworld, in_axes=(0, None, None, 0))(
        _rng, static_params, player_position, ALL_SMOOTHGEN_CONFIGS
    )

    # Generate dungeons
    rngs = jax.random.split(rng, 4)
    rng, _rng = rngs[0], rngs[1:]
    dungeons = jax.vmap(generate_dungeon, in_axes=(0, None, 0))(
        _rng, static_params, ALL_DUNGEON_CONFIGS
    )

    # Returns stacked versions of the map, item_map, light_map and ladders
    # 9 elements in each of these stacks representing each of the levels.
    # Splice smoothgens and dungeons in order of levels
    map, item_map, light_map, ladders_down, ladders_up = jax.tree_util.tree_map(
        lambda x, y: jnp.stack((x[0], y[0], x[1], y[1], y[2], x[2], x[3], x[4], x[5]), axis=0),
        smoothgens,
        dungeons,
    )

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

    # Generate coordination maps for all levels
    rng, *coord_rngs = jax.random.split(rng, static_params.num_levels + 1)
    coord_rngs = jnp.stack(coord_rngs)

    def _generate_coord_map_for_level(rng_and_level_map):
        level_rng, level_map = rng_and_level_map
        return generate_coordination_map(level_rng, level_map, params, static_params)

    coordination_map, soft_coordination_mask = jax.vmap(_generate_coord_map_for_level)(
        (coord_rngs, map)
    )

    # Place construction sites on each level (vmapped across levels):
    # - Level 0 (Overworld): num_construction_sites general construction sites
    # - Levels 2, 5, 7 (diamond-bearing mining levels): num_mining_construction_sites
    #   for forge construction to enable diamond equipment crafting
    MINING_LEVELS = jnp.array([2, 5, 7])  # Gnomish Mines, Troll Mines, Ice Level

    rng, *construction_rngs = jax.random.split(rng, static_params.num_levels + 1)
    construction_rngs = jnp.stack(construction_rngs)

    # Pre-compute num_sites per level: overworld=num_construction_sites,
    # mining levels=num_mining_construction_sites, others=0
    level_indices = jnp.arange(static_params.num_levels)
    is_overworld = level_indices == 0
    is_mining = jnp.isin(level_indices, MINING_LEVELS)
    num_sites_per_level = jnp.where(
        is_overworld,
        params.num_construction_sites,
        jnp.where(is_mining, params.num_mining_construction_sites, 0),
    )

    # vmap place_construction_sites across all levels simultaneously
    vmapped_place = jax.vmap(place_construction_sites, in_axes=(0, 0, 0, None, None))
    new_maps, all_positions, all_coord, all_soft = vmapped_place(
        construction_rngs, map, num_sites_per_level, params, static_params
    )

    # Only use placed results for levels that actually have sites
    should_place = (num_sites_per_level > 0)[:, None, None]  # (num_levels, 1, 1)
    map = jnp.where(should_place, new_maps, map)
    construction_site_positions = all_positions
    construction_site_coord = all_coord
    construction_site_soft = all_soft

    # Apply construction site coordination values and soft mask to the maps
    # using vectorized scatter instead of sequential scan.
    # Build flat index arrays for all (num_levels * max_construction_sites) sites
    all_level_idx = jnp.repeat(
        jnp.arange(static_params.num_levels), static_params.max_construction_sites
    )
    all_site_idx = jnp.tile(
        jnp.arange(static_params.max_construction_sites), static_params.num_levels
    )
    # Gather positions, coord values, and soft values for all sites
    all_pos = construction_site_positions[all_level_idx, all_site_idx]  # (N, 2)
    all_coord_val = construction_site_coord[all_level_idx, all_site_idx]  # (N,)
    all_soft_val = construction_site_soft[all_level_idx, all_site_idx]  # (N,)

    # Check validity: coord_val != 0 and block at position is CONSTRUCTION_SITE
    block_at_pos = map[all_level_idx, all_pos[:, 0], all_pos[:, 1]]
    is_valid = (all_coord_val != 0) & (block_at_pos == BlockType.CONSTRUCTION_SITE.value)

    # Apply valid coord values and soft mask via single scatter operations
    existing_coord = coordination_map[all_level_idx, all_pos[:, 0], all_pos[:, 1]]
    new_coord_val = jnp.where(is_valid, all_coord_val, existing_coord)
    coordination_map = coordination_map.at[all_level_idx, all_pos[:, 0], all_pos[:, 1]].set(
        new_coord_val
    )

    existing_soft = soft_coordination_mask[all_level_idx, all_pos[:, 0], all_pos[:, 1]]
    new_soft_val = jnp.where(is_valid, all_soft_val, existing_soft)
    soft_coordination_mask = soft_coordination_mask.at[
        all_level_idx, all_pos[:, 0], all_pos[:, 1]
    ].set(new_soft_val)

    # Initialize construction state
    construction_sites_built = jnp.zeros(
        (static_params.num_levels, static_params.max_construction_sites), dtype=jnp.int32
    )
    construction_handover_deadline = jnp.zeros(
        (static_params.num_levels, static_params.max_construction_sites), dtype=jnp.int32
    )

    # Pre-generate elite/large mob status for all mob slots
    # This is determined once at world gen, not at spawn time
    num_melee_slots = static_params.max_melee_mobs * static_params.player_count
    num_ranged_slots = static_params.max_ranged_mobs * static_params.player_count
    num_passive_slots = static_params.max_passive_mobs * static_params.player_count

    rng, melee_rng, ranged_rng, passive_rng, hard_melee_rng, hard_ranged_rng, hard_passive_rng = (
        jax.random.split(rng, 7)
    )
    rng, melee_ar_rng, ranged_ar_rng, passive_ar_rng = jax.random.split(rng, 4)

    # Elite probability increases with floor depth: +5% per floor
    floor_bonuses = jnp.arange(static_params.num_levels) * 0.05
    elite_probs = params.elite_mob_probability + floor_bonuses[:, None]  # (num_levels, 1)

    # Melee mobs: 0=normal, 1=elite soft, 2=elite hard
    melee_rolls = jax.random.uniform(melee_rng, (static_params.num_levels, num_melee_slots))
    melee_is_elite = melee_rolls < elite_probs
    melee_hard_rolls = jax.random.uniform(
        hard_melee_rng, (static_params.num_levels, num_melee_slots)
    )
    melee_is_hard = melee_is_elite & (melee_hard_rolls < params.hard_mob_probability)
    melee_mob_coordination = jnp.where(melee_is_elite, jnp.where(melee_is_hard, 2, 1), 0).astype(
        jnp.int32
    )

    # Per-mob agents_required via p_max_agents sampling
    melee_ar_rolls = jax.random.uniform(melee_ar_rng, (static_params.num_levels, num_melee_slots))
    melee_mob_agents_required = jnp.where(
        melee_is_elite,
        jnp.where(melee_ar_rolls < params.p_max_agents, static_params.player_count, 2),
        0,
    ).astype(jnp.int32)

    # Ranged mobs: 0=normal, 1=elite soft, 2=elite hard
    ranged_rolls = jax.random.uniform(ranged_rng, (static_params.num_levels, num_ranged_slots))
    ranged_is_elite = ranged_rolls < elite_probs
    ranged_hard_rolls = jax.random.uniform(
        hard_ranged_rng, (static_params.num_levels, num_ranged_slots)
    )
    ranged_is_hard = ranged_is_elite & (ranged_hard_rolls < params.hard_mob_probability)
    ranged_mob_coordination = jnp.where(ranged_is_elite, jnp.where(ranged_is_hard, 2, 1), 0).astype(
        jnp.int32
    )

    ranged_ar_rolls = jax.random.uniform(
        ranged_ar_rng, (static_params.num_levels, num_ranged_slots)
    )
    ranged_mob_agents_required = jnp.where(
        ranged_is_elite,
        jnp.where(ranged_ar_rolls < params.p_max_agents, static_params.player_count, 2),
        0,
    ).astype(jnp.int32)

    # Passive mobs: 0=normal, 1=large soft, 2=large hard
    passive_rolls = jax.random.uniform(passive_rng, (static_params.num_levels, num_passive_slots))
    passive_is_large = passive_rolls < params.large_passive_probability
    passive_hard_rolls = jax.random.uniform(
        hard_passive_rng, (static_params.num_levels, num_passive_slots)
    )
    passive_is_hard = passive_is_large & (passive_hard_rolls < params.hard_mob_probability)
    passive_mob_coordination = jnp.where(
        passive_is_large, jnp.where(passive_is_hard, 2, 1), 0
    ).astype(jnp.int32)

    passive_ar_rolls = jax.random.uniform(
        passive_ar_rng, (static_params.num_levels, num_passive_slots)
    )
    passive_mob_agents_required = jnp.where(
        passive_is_large,
        jnp.where(passive_ar_rolls < params.p_max_agents, static_params.player_count, 2),
        0,
    ).astype(jnp.int32)

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
        sync_coord_by_agents=jnp.zeros(2, dtype=jnp.int32),  # [2-agent, 3+-agent]
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
        # Construction system
        construction_sites_built=construction_sites_built,
        construction_site_positions=construction_site_positions,
        construction_handover_deadline=construction_handover_deadline,
        # Elite/Large mob coordination
        melee_mob_coordination=melee_mob_coordination,
        ranged_mob_coordination=ranged_mob_coordination,
        passive_mob_coordination=passive_mob_coordination,
        # Per-mob agents_required for elite/large coordination
        melee_mob_agents_required=melee_mob_agents_required,
        ranged_mob_agents_required=ranged_mob_agents_required,
        passive_mob_agents_required=passive_mob_agents_required,
        # Communication messages
        comm_messages=jnp.zeros(
            (static_params.player_count, static_params.num_comm_channels), dtype=jnp.float32
        ),
        comm_count=jnp.zeros(static_params.player_count, dtype=jnp.int32),
        sampled_alpha=jnp.asarray(sampled_alpha, dtype=jnp.float32),
        state_rng=_rng,
        timestep=jnp.asarray(0, dtype=jnp.int32),
    )

    return state
