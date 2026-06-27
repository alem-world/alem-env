from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from jaxtyping import Array, Bool, Float, Int


def get_distance_map(
    position: Int[Array, 2],
    map_size: tuple[int, int],
) -> Float[Array, "H W"]:
    """Compute Euclidean distance from one coordinate to every map cell.

    Args:
        position: Source row/column coordinate.
        map_size: Height and width of the output map.

    Returns:
        Two-dimensional Euclidean distance map.
    """
    dist_x = jnp.abs(jnp.arange(0, map_size[0]) - position[0])
    dist_x = jnp.expand_dims(dist_x, axis=1)
    dist_x = jnp.tile(dist_x, (1, map_size[1]))

    dist_y = jnp.abs(jnp.arange(0, map_size[1]) - position[1])
    dist_y = jnp.expand_dims(dist_y, axis=0)
    dist_y = jnp.tile(dist_y, (map_size[0], 1))

    coords = jnp.stack([dist_x, dist_y], axis=-1)

    def _euclid_distance(x):
        return jnp.sqrt(x[0] ** 2 + x[1] ** 2)

    dist = jax.vmap(jax.vmap(_euclid_distance))(coords)

    return dist


def get_all_players_distance_map(
    position: Int[Array, "n_agents 2"],
    mask: Bool[Array, "n_agents"],
    static_params,
) -> Float[Array, "H W"]:
    """Compute distance to the nearest active player for every map cell.

    Args:
        position: Coordinate of each player.
        mask: Boolean mask selecting players included in the minimum.
        static_params: Static map dimensions and player count.

    Returns:
        Distance map reduced across active players.
    """
    player_proximity_map = jax.vmap(get_distance_map, in_axes=(0, None))(
        position, static_params.map_size
    )
    max_dist = jnp.sqrt(static_params.map_size[0] ** 2 + static_params.map_size[1] ** 2)

    # If player is dead, remove from distance consideration
    player_proximity_map_masked = jnp.where(
        mask[:, None, None],
        player_proximity_map,
        jnp.full(
            (static_params.player_count, static_params.map_size[0], static_params.map_size[1]),
            max_dist,
        ),
    )

    all_players_proximity_map = jnp.min(player_proximity_map_masked, axis=0).astype(jnp.float32)
    return all_players_proximity_map
