from functools import partial

import jax
import jax.numpy as jnp

from ..constants import (
    INVENTORY_OBS_HEIGHT,
    MAX_OBS_DIM,
    MONSTERS_KILLED_TO_CLEAR_LEVEL,
    OBS_DIM,
    TEXTURES,
    Action,
    BlockType,
    ItemType,
    Specialization,
)
from ..util.game_logic_utils import get_player_icon_positions, is_boss_vulnerable

# Coordination rendering styles
COORD_RENDER_NONE = 0  # No coordination visualization
COORD_RENDER_BORDER = 1  # V1: Icon-based with borders, player icons, number badges
COORD_RENDER_TINT = 2  # V2: Minecraft-style texture color shifts


def render_coordination_borders(
    map_pixels, coord_map_view, soft_mask_view, map_view, block_pixel_size, textures
):
    """V1: Rich coordination rendering with icons, borders, and badges.

    Visual distinction:
    - Hard sync: Solid CYAN border (must coordinate or fail)
    - Soft sync: Solid GREEN border (scales reward)
    - Handover: Dashed ORANGE border (always hard)
    - Number badges show required agents (2/3/4+)

    Only applies to blocks that are still minable resources.

    Args:
        map_pixels: Batched rendered map pixels to decorate.
        coord_map_view: Per-player local coordination requirement maps.
        soft_mask_view: Per-player local masks distinguishing soft requirements.
        map_view: Per-player local block identifiers.
        block_pixel_size: Width and height of each rendered tile.
        textures: Texture dictionary containing coordination icons and digits.

    Returns:
        Map pixels with coordination borders, icons, and badges applied.
    """
    px = jnp.arange(block_pixel_size)
    py = jnp.arange(block_pixel_size)
    pxx, pyy = jnp.meshgrid(px, py)

    # Define which block types are minable resources that should show coordination
    RESOURCE_BLOCKS = jnp.array(
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
            BlockType.CONSTRUCTION_SITE.value,
            BlockType.CONSTRUCTION_IN_PROGRESS.value,
        ]
    )

    # Expand map_view to pixel space to check block types
    map_pixels_blocks = jnp.repeat(
        jnp.repeat(map_view, repeats=block_pixel_size, axis=1),
        repeats=block_pixel_size,
        axis=2,
    )

    # Check if each pixel's block is a resource
    is_resource = jnp.isin(map_pixels_blocks, RESOURCE_BLOCKS)

    # Expand coordination map to pixel space
    coord_pixels = jnp.repeat(
        jnp.repeat(coord_map_view, repeats=block_pixel_size, axis=1),
        repeats=block_pixel_size,
        axis=2,
    )

    # Expand soft mask to pixel space
    soft_pixels = jnp.repeat(
        jnp.repeat(soft_mask_view, repeats=block_pixel_size, axis=1),
        repeats=block_pixel_size,
        axis=2,
    )

    # Identify coordination types (only for resource blocks)
    is_sync = (
        coord_pixels > 0
    ) & is_resource  # Positive = sync coordination (N agents simultaneously)
    is_handover = (coord_pixels < 0) & is_resource  # Negative = handover (sequential within window)
    has_any_coord = is_sync | is_handover

    # Split sync into hard/soft
    is_hard_sync = is_sync & ~soft_pixels
    is_soft_sync = is_sync & soft_pixels

    # Track required agents for sync
    is_coord_2 = coord_pixels == 2
    is_coord_3 = coord_pixels == 3
    is_coord_4 = coord_pixels >= 4

    # =========================================================================
    # BORDERS: Hard sync=cyan, Soft sync=green, Handover=orange dashed
    # =========================================================================
    border_width = 2
    is_border = (
        (pxx < border_width)
        | (pxx >= block_pixel_size - border_width)
        | (pyy < border_width)
        | (pyy >= block_pixel_size - border_width)
    )
    is_dash = ((pxx + pyy) // 6) % 2 == 0
    is_dashed_border = is_border & is_dash

    border_tiled = jnp.tile(is_border, (OBS_DIM[0], OBS_DIM[1]))
    dashed_border_tiled = jnp.tile(is_dashed_border, (OBS_DIM[0], OBS_DIM[1]))

    hard_sync_color = jnp.array([120.0, 200.0, 255.0])  # Cyan (hard - required)
    soft_sync_color = jnp.array([100.0, 230.0, 100.0])  # Green (soft - bonus)
    handover_color = jnp.array([250.0, 150.0, 60.0])  # Orange (handover)

    # Apply hard sync borders (solid cyan - required)
    map_pixels = jnp.where(
        is_hard_sync[:, :, :, None] & border_tiled[None, :, :, None],
        hard_sync_color[None, None, None, :],
        map_pixels,
    )

    # Apply soft sync borders (dashed green - optional bonus)
    map_pixels = jnp.where(
        is_soft_sync[:, :, :, None] & dashed_border_tiled[None, :, :, None],
        soft_sync_color[None, None, None, :],
        map_pixels,
    )

    # Apply handover borders (dashed orange)
    map_pixels = jnp.where(
        is_handover[:, :, :, None] & dashed_border_tiled[None, :, :, None],
        handover_color[None, None, None, :],
        map_pixels,
    )

    # Placement helpers
    def build_region(origin_x, origin_y, size):
        region = (
            (pxx >= origin_x)
            & (pxx < origin_x + size)
            & (pyy >= origin_y)
            & (pyy < origin_y + size)
        )
        region_tiled = jnp.tile(region, (OBS_DIM[0], OBS_DIM[1]))
        local_x = jnp.clip(pxx - origin_x, 0, size - 1).astype(jnp.int32)
        local_y = jnp.clip(pyy - origin_y, 0, size - 1).astype(jnp.int32)
        local_x_tiled = jnp.tile(local_x, (OBS_DIM[0], OBS_DIM[1]))
        local_y_tiled = jnp.tile(local_y, (OBS_DIM[0], OBS_DIM[1]))
        return region_tiled, local_x_tiled, local_y_tiled

    # =========================================================================
    # PLAYER ICONS: Two overlapping player silhouettes (top-left)
    # =========================================================================
    icon_size = textures["coord_icon_size"]
    player_icon = textures["coord_player_texture"].astype(jnp.float32)
    player_icon_alpha = textures["coord_player_texture_alpha"].astype(jnp.float32)
    player_icon_alpha = player_icon_alpha[:, :, None]

    # Two overlapping player icons
    icon_pad = 2
    overlap = max(1, icon_size // 3)
    icon1_region, icon1_lx, icon1_ly = build_region(icon_pad, icon_pad, icon_size)
    icon2_region, icon2_lx, icon2_ly = build_region(
        icon_pad + overlap, icon_pad + overlap // 2, icon_size
    )

    # Color tints: cyan and gold to distinguish players
    icon1_rgb = player_icon * jnp.array([0.45, 0.9, 1.0])  # Cyan tint
    icon2_rgb = player_icon * jnp.array([1.0, 0.82, 0.45])  # Gold tint

    icon1_sample = icon1_rgb[icon1_ly, icon1_lx]
    icon2_sample = icon2_rgb[icon2_ly, icon2_lx]
    icon1_alpha = player_icon_alpha[icon1_ly, icon1_lx]
    icon2_alpha = player_icon_alpha[icon2_ly, icon2_lx]

    icon1_mask = has_any_coord[:, :, :, None] & icon1_region[None, :, :, None]
    icon2_mask = has_any_coord[:, :, :, None] & icon2_region[None, :, :, None]

    map_pixels = jnp.where(
        icon1_mask,
        map_pixels * (1 - icon1_alpha[None, :, :, :])
        + icon1_sample[None, :, :, :] * icon1_alpha[None, :, :, :],
        map_pixels,
    )
    map_pixels = jnp.where(
        icon2_mask,
        map_pixels * (1 - icon2_alpha[None, :, :, :])
        + icon2_sample[None, :, :, :] * icon2_alpha[None, :, :, :],
        map_pixels,
    )

    # =========================================================================
    # NUMBER BADGES: Bottom-right showing 2/3/4+ agents required
    # =========================================================================
    number_textures = textures["coord_number_textures"].astype(jnp.float32)
    number_alphas = textures["coord_number_textures_alpha"].astype(jnp.float32)
    badge_size = number_textures.shape[1]
    badge_x = block_pixel_size - badge_size - 3
    badge_y = block_pixel_size - badge_size - 3
    badge_region, badge_lx, badge_ly = build_region(badge_x, badge_y, badge_size)

    badge_tex2 = number_textures[2][badge_ly, badge_lx]
    badge_tex3 = number_textures[3][badge_ly, badge_lx]
    badge_tex4 = number_textures[4][badge_ly, badge_lx]

    badge_a2 = number_alphas[2][badge_ly, badge_lx]
    badge_a3 = number_alphas[3][badge_ly, badge_lx]
    badge_a4 = number_alphas[4][badge_ly, badge_lx]

    badge_mask = badge_region[None, :, :, None]
    show_badge2 = badge_mask & is_coord_2[:, :, :, None]
    show_badge3 = badge_mask & is_coord_3[:, :, :, None]
    show_badge4 = badge_mask & is_coord_4[:, :, :, None]

    map_pixels = jnp.where(
        show_badge2,
        map_pixels * (1 - badge_a2[None, :, :, :])
        + badge_tex2[None, :, :, :] * badge_a2[None, :, :, :],
        map_pixels,
    )
    map_pixels = jnp.where(
        show_badge3,
        map_pixels * (1 - badge_a3[None, :, :, :])
        + badge_tex3[None, :, :, :] * badge_a3[None, :, :, :],
        map_pixels,
    )
    map_pixels = jnp.where(
        show_badge4,
        map_pixels * (1 - badge_a4[None, :, :, :])
        + badge_tex4[None, :, :, :] * badge_a4[None, :, :, :],
        map_pixels,
    )

    # =========================================================================
    # CLOCK ICON: Bottom-left for handover blocks (temporal coordination)
    # =========================================================================
    clock_texture = textures["coord_clock_texture"].astype(jnp.float32)
    clock_alpha = textures["coord_clock_texture_alpha"].astype(jnp.float32)
    clock_size = clock_texture.shape[0]
    clock_x = 3
    clock_y = block_pixel_size - clock_size - 3
    clock_region, clock_lx, clock_ly = build_region(clock_x, clock_y, clock_size)
    clock_mask = is_handover[:, :, :, None] & clock_region[None, :, :, None]
    clock_sample = clock_texture[clock_ly, clock_lx]
    clock_a = clock_alpha[clock_ly, clock_lx]

    map_pixels = jnp.where(
        clock_mask,
        map_pixels * (1 - clock_a[None, :, :, :])
        + clock_sample[None, :, :, :] * clock_a[None, :, :, :],
        map_pixels,
    )

    return map_pixels


def render_coordination_tint(
    map_pixels, coord_map_view, soft_mask_view, map_view, block_pixel_size, textures
):
    """Route tint requests to the cleaner border-based coordination rendering.

    Args:
        map_pixels: Batched rendered map pixels to decorate.
        coord_map_view: Per-player local coordination requirement maps.
        soft_mask_view: Per-player local soft-coordination masks.
        map_view: Per-player local block identifiers.
        block_pixel_size: Width and height of each rendered tile.
        textures: Texture dictionary containing coordination overlays.

    Returns:
        Map pixels decorated by ``render_coordination_borders``.
    """
    return render_coordination_borders(
        map_pixels, coord_map_view, soft_mask_view, map_view, block_pixel_size, textures
    )


def render_handover_countdown(
    map_pixels,
    pending_handovers,
    timestep,
    player_position,
    obs_dim_array,
    block_pixel_size,
    textures,
):
    """Render countdown bars for active handovers (like Overcooked).

    Shows a shrinking yellow->red timer tray above blocks with pending handovers.
    The tray includes a clock icon so the temporal handover mechanic reads
    immediately in static figures.

    Args:
        map_pixels: Current rendered pixels (batch, height, width, 3)
        pending_handovers: Active handovers (max_pending, 6) - [active, pos_x, pos_y, deadline, initiator_id, build_type]
        timestep: Current game timestep
        player_position: Position of each player (batch, 2)
        obs_dim_array: Observation dimensions
        block_pixel_size: Size of each block in pixels

    Returns:
        map_pixels with countdown bars rendered
    """
    h, w = map_pixels.shape[1], map_pixels.shape[2]
    clock_texture = textures["coord_clock_texture"].astype(jnp.float32)
    clock_alpha = textures["coord_clock_texture_alpha"].astype(jnp.float32)
    clock_size = clock_texture.shape[0]

    # For each pending handover, render a countdown bar
    def render_single_handover(pixels, handover_idx):
        handover = pending_handovers[handover_idx]
        is_active = handover[0] == 1
        pos_x, pos_y = handover[1], handover[2]
        deadline = handover[3]

        # Calculate time remaining (as fraction, max window ~15 steps)
        time_remaining = jnp.maximum(0, deadline - timestep)
        progress = jnp.clip(time_remaining / 15.0, 0.0, 1.0)

        # Bar color: yellow (progress=1) -> orange -> red (progress=0)
        bar_r = 255.0
        bar_g = 50.0 + 200.0 * progress  # 250 at full, 50 at empty
        bar_b = 50.0

        # Process each player's view
        def render_for_player(player_pixels, player_idx):
            player_pos = player_position[player_idx]

            # Calculate local position in view
            local_x = pos_x - player_pos[0] + obs_dim_array[0] // 2
            local_y = pos_y - player_pos[1] + obs_dim_array[1] // 2

            # Check if in view bounds
            in_view = (
                is_active
                & (local_x >= 0)
                & (local_x < obs_dim_array[0])
                & (local_y >= 0)
                & (local_y < obs_dim_array[1])
            )

            tile_row = local_x * block_pixel_size
            tile_col = local_y * block_pixel_size
            hover_gap = max(3, block_pixel_size // 10)
            tray_pad = max(2, block_pixel_size // 16)
            tray_h = max(clock_size + 4, block_pixel_size // 4)
            tray_w = block_pixel_size
            tray_row = jnp.clip(
                tile_row - tray_h - hover_gap,
                0,
                max(0, h - tray_h),
            )
            tray_col = jnp.clip(
                tile_col,
                0,
                max(0, w - tray_w),
            )

            clock_row = tray_row + (tray_h - clock_size) // 2
            clock_col = tray_col + tray_pad
            bar_height = max(4, block_pixel_size // 16)
            bar_row = tray_row + (tray_h - bar_height) // 2
            bar_col = clock_col + clock_size + tray_pad
            max_bar_width = jnp.maximum(6, tray_w - (bar_col - tray_col) - tray_pad)
            bar_width = (progress * max_bar_width).astype(jnp.int32)

            # Create coordinate grids
            y_coords = jnp.arange(h)[:, None]
            x_coords = jnp.arange(w)[None, :]

            tray_mask = (
                in_view
                & (y_coords >= tray_row)
                & (y_coords < tray_row + tray_h)
                & (x_coords >= tray_col)
                & (x_coords < tray_col + tray_w)
            )
            outline_mask = in_view & (
                ((y_coords == tray_row) | (y_coords == tray_row + tray_h - 1))
                & (x_coords >= tray_col)
                & (x_coords < tray_col + tray_w)
            )
            outline_mask = outline_mask | (
                in_view
                & (
                    ((x_coords == tray_col) | (x_coords == tray_col + tray_w - 1))
                    & (y_coords >= tray_row)
                    & (y_coords < tray_row + tray_h)
                )
            )
            bar_bg_mask = (
                in_view
                & (y_coords >= bar_row)
                & (y_coords < bar_row + bar_height)
                & (x_coords >= bar_col)
                & (x_coords < bar_col + max_bar_width)
            )
            fill_mask = (
                in_view
                & (y_coords >= bar_row)
                & (y_coords < bar_row + bar_height)
                & (x_coords >= bar_col)
                & (x_coords < bar_col + bar_width)
            )

            clock_mask = (
                in_view
                & (y_coords >= clock_row)
                & (y_coords < clock_row + clock_size)
                & (x_coords >= clock_col)
                & (x_coords < clock_col + clock_size)
            )
            clock_ly = jnp.clip(y_coords - clock_row, 0, clock_size - 1).astype(jnp.int32)
            clock_lx = jnp.clip(x_coords - clock_col, 0, clock_size - 1).astype(jnp.int32)
            clock_sample = clock_texture[clock_ly, clock_lx]
            clock_a = clock_alpha[clock_ly, clock_lx]

            tray_color = jnp.array([248.0, 239.0, 221.0])
            outline_color = jnp.array([165.0, 103.0, 54.0])
            bar_bg_color = jnp.array([226.0, 206.0, 170.0])
            bar_color = jnp.array([bar_r, bar_g, bar_b])

            new_pixels = jnp.where(
                tray_mask[:, :, None], player_pixels * 0.15 + tray_color * 0.85, player_pixels
            )
            new_pixels = jnp.where(outline_mask[:, :, None], outline_color, new_pixels)
            new_pixels = jnp.where(
                bar_bg_mask[:, :, None], new_pixels * 0.18 + bar_bg_color * 0.82, new_pixels
            )
            new_pixels = jnp.where(fill_mask[:, :, None], bar_color, new_pixels)
            new_pixels = jnp.where(
                clock_mask[:, :, None],
                new_pixels * (1.0 - clock_a) + clock_sample * clock_a,
                new_pixels,
            )

            return new_pixels, None

        # Apply to all player views using vmap
        pixels, _ = jax.lax.scan(
            lambda carry, idx: (carry.at[idx].set(render_for_player(carry[idx], idx)[0]), None),
            pixels,
            jnp.arange(player_position.shape[0]),
        )

        return pixels, None

    # Process all pending handovers
    max_pending = pending_handovers.shape[0]
    map_pixels, _ = jax.lax.scan(render_single_handover, map_pixels, jnp.arange(max_pending))

    return map_pixels


def apply_elite_tint(texture, is_elite):
    """Apply golden/menacing tint to elite mobs.

    Args:
        texture: Mob texture array (H, W, 3)
        is_elite: Boolean scalar or array indicating if mob is elite

    Returns:
        Tinted texture with golden glow for elite mobs
    """
    warm_gold = jnp.array([224.0, 184.0, 92.0])
    elite_texture = texture * jnp.array([1.04, 1.01, 0.90])
    elite_texture = elite_texture * 0.84 + warm_gold * 0.16
    elite_texture = jnp.clip((elite_texture - 128.0) * 1.04 + 128.0, 0.0, 255.0)

    # Return elite texture if elite, otherwise original
    return jnp.where(is_elite, elite_texture, texture)


def apply_elite_border(texture, texture_alpha, agents_required):
    """Draw a dark red silhouette border around elite mob sprites.

    Any coordinated mob receives the same publication-friendly border treatment.
    The border follows the sprite silhouette rather than the bounding box.

    Args:
        texture: (H, W, 3) mob texture
        texture_alpha: (H, W, 3) alpha mask (>0 where sprite is opaque)
        agents_required: int scalar — 0 for normal mobs, 2+ for elites

    Returns:
        Texture with a silhouette border when coordination is required.
    """
    has_alpha = texture_alpha.max(axis=-1) > 0  # (H, W)
    padded = jnp.pad(has_alpha, ((2, 2), (2, 2)), constant_values=False)
    up1 = padded[1:-3, 2:-2]
    down1 = padded[3:-1, 2:-2]
    left1 = padded[2:-2, 1:-3]
    right1 = padded[2:-2, 3:-1]
    up2 = padded[0:-4, 2:-2]
    down2 = padded[4:, 2:-2]
    left2 = padded[2:-2, 0:-4]
    right2 = padded[2:-2, 4:]
    core_interior = has_alpha & up1 & down1 & left1 & right1
    deep_interior = core_interior & up2 & down2 & left2 & right2
    border_mask = has_alpha & ~deep_interior

    border_color = jnp.array([132.0, 24.0, 34.0])
    blend = 0.78
    bordered = jnp.where(
        border_mask[..., None],
        texture * (1.0 - blend) + border_color * blend,
        texture,
    )
    return jnp.where(agents_required > 0, bordered, texture)


@partial(
    jax.jit,
    static_argnums=(
        1,
        2,
        4,
        5,
    ),
)
def render_alem_pixels(
    state,
    block_pixel_size,
    static_params,
    player_specific_textures,
    do_night_noise=True,
    coordination_render_style=COORD_RENDER_BORDER,
):
    """Render batched player-centric RGB observations.

    Args:
        state: Environment state to render.
        block_pixel_size: Pixel width and height of each map tile.
        static_params: Static map, player, and entity parameters.
        player_specific_textures: Colorized textures for each player.
        do_night_noise: Whether darkness should add visual noise.
        coordination_render_style: Border or tint treatment for coordination tiles.

    Returns:
        RGB observation image for every player.
    """
    textures = TEXTURES[block_pixel_size]
    obs_dim_array = jnp.array([OBS_DIM[0], OBS_DIM[1]], dtype=jnp.int32)

    # RENDER MAP
    # Get view of map
    map = state.map[state.player_level]
    padded_grid = jnp.pad(
        map,
        (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
        constant_values=BlockType.OUT_OF_BOUNDS.value,
    )

    tl_corner = state.player_position - obs_dim_array // 2 + MAX_OBS_DIM + 2

    map_view = jax.vmap(jax.lax.dynamic_slice, in_axes=(None, 0, None))(
        padded_grid, tl_corner, OBS_DIM
    )

    # Boss
    boss_block = jax.lax.select(
        is_boss_vulnerable(state),
        BlockType.NECROMANCER_VULNERABLE.value,
        BlockType.NECROMANCER.value,
    )

    map_view_boss = map_view == BlockType.NECROMANCER.value
    map_view = map_view_boss * boss_block + (1 - map_view_boss) * map_view

    # Render map tiles
    map_pixels_indexes = jnp.repeat(
        jnp.repeat(map_view, repeats=block_pixel_size, axis=1),
        repeats=block_pixel_size,
        axis=2,
    )
    map_pixels_indexes = jnp.expand_dims(map_pixels_indexes, axis=-1)
    map_pixels_indexes = jnp.repeat(map_pixels_indexes, repeats=3, axis=-1)

    map_pixels = jnp.zeros_like(map_pixels_indexes, dtype=jnp.float32)

    def _add_block_type_to_pixels(pixels, block_index):
        return (
            pixels
            + textures["full_map_block_textures"][block_index]
            * (map_pixels_indexes == block_index),
            None,
        )

    map_pixels, _ = jax.lax.scan(_add_block_type_to_pixels, map_pixels, jnp.arange(len(BlockType)))

    if "transparent_block_textures" in textures and "transparent_block_texture_alphas" in textures:
        transparent_structure_blocks = jnp.array(
            [
                BlockType.CONSTRUCTION_SITE.value,
                BlockType.CONSTRUCTION_IN_PROGRESS.value,
                BlockType.EPIC_SHELTER.value,
                BlockType.EPIC_FORGE.value,
                BlockType.EPIC_BEACON.value,
            ],
            dtype=jnp.int32,
        )
        grass_full_texture = textures["full_map_block_textures"][BlockType.GRASS.value].astype(
            jnp.float32
        )

        def _overlay_transparent_structure(pixels, block_index):
            block_mask = (map_pixels_indexes[:, :, :, 0] == block_index).astype(jnp.float32)[
                :, :, :, None
            ]
            texture = jnp.tile(textures["transparent_block_textures"][block_index], (*OBS_DIM, 1))
            texture_alpha = jnp.tile(
                textures["transparent_block_texture_alphas"][block_index], (*OBS_DIM, 1)
            )
            structure_pixels = grass_full_texture * (1.0 - texture_alpha) + texture * texture_alpha
            pixels = pixels * (1.0 - block_mask) + structure_pixels * block_mask
            return pixels, None

        map_pixels, _ = jax.lax.scan(
            _overlay_transparent_structure, map_pixels, transparent_structure_blocks
        )

    # Render coordination indicators (if enabled)
    if coordination_render_style != COORD_RENDER_NONE:
        # Get coordination map view
        padded_coord_map = jnp.pad(
            state.coordination_map[state.player_level],
            (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
            constant_values=0,
        )
        coord_map_view = jax.vmap(jax.lax.dynamic_slice, in_axes=(None, 0, None))(
            padded_coord_map, tl_corner, OBS_DIM
        )

        # Get soft coordination mask view
        padded_soft_mask = jnp.pad(
            state.soft_coordination_mask[state.player_level],
            (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
            constant_values=False,
        )
        soft_mask_view = jax.vmap(jax.lax.dynamic_slice, in_axes=(None, 0, None))(
            padded_soft_mask, tl_corner, OBS_DIM
        )

        if coordination_render_style == COORD_RENDER_BORDER:
            map_pixels = render_coordination_borders(
                map_pixels, coord_map_view, soft_mask_view, map_view, block_pixel_size, textures
            )
        elif coordination_render_style == COORD_RENDER_TINT:
            map_pixels = render_coordination_tint(
                map_pixels, coord_map_view, soft_mask_view, map_view, block_pixel_size, textures
            )

        # Render handover countdown bars (like Overcooked timers)
        map_pixels = render_handover_countdown(
            map_pixels,
            state.pending_handovers,
            state.timestep,
            state.player_position,
            obs_dim_array,
            block_pixel_size,
            textures,
        )

    # # Render Colored Chests
    # def _add_player_chests(chest_map_view, player_index):
    #     """Adds players chest position to other players"""
    #     local_position = (
    #         state.chest_positions[state.player_level, player_index]
    #         - state.player_position[:, None]
    #         + jnp.ones((2,), dtype=jnp.int32) * (obs_dim_array // 2)
    #     )
    #     def _single_batch_index(data_slice, row_idx, col_idx):
    #         return data_slice.at[row_idx, col_idx].set(player_index)
    #     chest_map_view = jax.vmap(_single_batch_index, in_axes=(0,0,0))(
    #         chest_map_view, local_position[..., 0], local_position[..., 1]
    #     )
    #     return chest_map_view, None

    # chest_map_view = jnp.full_like(map_view, fill_value=-1)
    # chest_map_view, _ = jax.lax.scan(
    #     _add_player_chests,
    #     chest_map_view,
    #     jnp.arange(static_params.player_count),
    # )
    # chest_map_pixels_indexes = jnp.repeat(
    #     jnp.repeat(chest_map_view, repeats=block_pixel_size, axis=1),
    #     repeats=block_pixel_size,
    #     axis=2,
    # )
    # chest_map_pixels_indexes = jnp.expand_dims(chest_map_pixels_indexes, axis=-1)
    # chest_map_pixels_indexes = jnp.repeat(chest_map_pixels_indexes, repeats=3, axis=-1)

    # def _add_player_chest_to_pixels(pixels, player_index):
    #     return (
    #         pixels
    #         + (player_specific_textures.chest_textures[player_index] - pixels)
    #         * (chest_map_pixels_indexes == player_index)
    #         * (map_pixels_indexes == BlockType.CHEST.value),
    #         None,
    #     )

    # map_pixels, _ = jax.lax.scan(
    #     _add_player_chest_to_pixels, map_pixels, jnp.arange(static_params.player_count)
    # )

    # Items
    padded_item_map = jnp.pad(
        state.item_map[state.player_level],
        (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
        constant_values=ItemType.NONE.value,
    )

    item_map_view = jax.vmap(jax.lax.dynamic_slice, in_axes=(None, 0, None))(
        padded_item_map, tl_corner, OBS_DIM
    )

    # Insert blocked ladders
    is_ladder_down_open = (
        state.monsters_killed[state.player_level] >= MONSTERS_KILLED_TO_CLEAR_LEVEL
    )
    ladder_down_item = jax.lax.select(
        is_ladder_down_open,
        ItemType.LADDER_DOWN.value,
        ItemType.LADDER_DOWN_BLOCKED.value,
    )

    item_map_view_is_ladder_down = item_map_view == ItemType.LADDER_DOWN.value
    item_map_view = (
        item_map_view_is_ladder_down * ladder_down_item
        + (1 - item_map_view_is_ladder_down) * item_map_view
    )

    map_pixels_item_indexes = jnp.repeat(
        jnp.repeat(item_map_view, repeats=block_pixel_size, axis=1),
        repeats=block_pixel_size,
        axis=2,
    )
    map_pixels_item_indexes = jnp.expand_dims(map_pixels_item_indexes, axis=-1)
    map_pixels_item_indexes = jnp.repeat(map_pixels_item_indexes, repeats=3, axis=-1)

    def _add_item_type_to_pixels(pixels, item_index):
        full_map_texture = textures["full_map_item_textures"][item_index]
        mask = map_pixels_item_indexes == item_index

        pixels = pixels * (1 - full_map_texture[:, :, 3] * mask[:, :, :, 0])[:, :, :, None]
        pixels = pixels + full_map_texture[:, :, :3] * mask * full_map_texture[:, :, 3][:, :, None]

        return pixels, None

    map_pixels, _ = jax.lax.scan(_add_item_type_to_pixels, map_pixels, jnp.arange(1, len(ItemType)))

    # Render player
    # Helper functions to display and update slice
    def _slice_pixel_map(player_pixels, local_position):
        return jax.lax.dynamic_slice(
            player_pixels,
            (
                local_position[0] * block_pixel_size,
                local_position[1] * block_pixel_size,
                0,
            ),
            (block_pixel_size, block_pixel_size, 3),
        )

    def _update_slice_pixel_map(player_pixels, texture_with_background, local_position):
        return jax.lax.dynamic_update_slice(
            player_pixels,
            texture_with_background,
            (
                local_position[0] * block_pixel_size,
                local_position[1] * block_pixel_size,
                0,
            ),
        )

    def _composite_texture(player_pixels, texture, texture_alpha, local_position):
        background = _slice_pixel_map(player_pixels, local_position)
        shadow_alpha = jnp.pad(
            texture_alpha * 0.22,
            ((2, 0), (1, 0), (0, 0)),
            constant_values=0.0,
        )[:block_pixel_size, :block_pixel_size, :]
        background = background * (1.0 - shadow_alpha)
        return background * (1.0 - texture_alpha) + texture * texture_alpha

    # Render each player on the map of other players
    def _render_friends(pixels, player_index):
        local_position = (
            state.player_position[player_index]
            - state.player_position
            + jnp.ones((2,), dtype=jnp.int32) * (obs_dim_array // 2)
        )
        on_screen = jnp.logical_and(local_position >= 0, local_position < obs_dim_array).all(
            axis=-1
        )

        player_texture_index = jax.lax.select(
            state.is_sleeping[player_index], 4, state.player_direction[player_index] - 1
        )
        player_texture_index = jax.lax.select(
            state.player_alive[player_index], player_texture_index, 5
        )
        player_texture = player_specific_textures.player_textures[
            player_index, player_texture_index
        ]
        player_texture, player_texture_alpha = (
            player_texture[:, :, :3],
            player_texture[:, :, 3:],
        )

        player_texture_alpha = jax.vmap(jnp.multiply, in_axes=(None, 0))(
            player_texture_alpha, on_screen
        )
        player_texture = jax.vmap(jnp.multiply, in_axes=(None, 0))(player_texture, on_screen)
        player_texture_with_background = jax.vmap(_composite_texture, in_axes=(0, 0, 0, 0))(
            pixels, player_texture, player_texture_alpha, local_position
        )

        pixels = jax.vmap(_update_slice_pixel_map, in_axes=(0, 0, 0))(
            pixels, player_texture_with_background, local_position
        )

        return pixels, None

    map_pixels, _ = jax.lax.scan(
        _render_friends, map_pixels, jnp.arange(static_params.player_count)
    )

    # Render mobs with elite tinting and agents_required border
    def _add_mob_to_pixels_with_elite(carry, mob_index):
        pixels, mobs, texture_name, alpha_texture_name, mob_coordination, mob_agents_req = carry
        local_position = (
            mobs.position[state.player_level, mob_index]
            - state.player_position
            + jnp.ones((2,), dtype=jnp.int32) * (obs_dim_array // 2)
        )
        on_screen = jnp.logical_and(local_position >= 0, local_position < obs_dim_array).all(
            axis=-1
        )
        on_screen *= mobs.mask[state.player_level, mob_index]

        mob_texture = texture_name[mobs.type_id[state.player_level, mob_index]]
        mob_texture_alpha = alpha_texture_name[mobs.type_id[state.player_level, mob_index]]

        # Apply elite tint if this mob has coordination requirement > 0
        is_elite = mob_coordination[state.player_level, mob_index] > 0
        mob_texture = apply_elite_tint(mob_texture, is_elite)

        # Apply colored border encoding agents_required (orange=2, red=3+)
        agents_req = mob_agents_req[state.player_level, mob_index]
        mob_texture = apply_elite_border(mob_texture, mob_texture_alpha, agents_req)

        mob_texture_alpha = jax.vmap(jnp.multiply, in_axes=(None, 0))(mob_texture_alpha, on_screen)
        mob_texture = jax.vmap(jnp.multiply, in_axes=(None, 0))(mob_texture, on_screen)
        mob_texture_with_background = jax.vmap(_composite_texture, in_axes=(0, 0, 0, 0))(
            pixels, mob_texture, mob_texture_alpha, local_position
        )

        pixels = jax.vmap(_update_slice_pixel_map, in_axes=(0, 0, 0))(
            pixels, mob_texture_with_background, local_position
        )

        return (
            pixels,
            mobs,
            texture_name,
            alpha_texture_name,
            mob_coordination,
            mob_agents_req,
        ), None

    # Melee mobs with elite tint
    (map_pixels, _, _, _, _, _), _ = jax.lax.scan(
        _add_mob_to_pixels_with_elite,
        (
            map_pixels,
            state.melee_mobs,
            textures["melee_mob_textures"],
            textures["melee_mob_texture_alphas"],
            state.melee_mob_coordination,
            state.melee_mob_agents_required,
        ),
        jnp.arange(state.melee_mobs.mask.shape[1]),
    )

    # Passive mobs with elite tint and variant textures (buffalo/large cow)
    def _add_passive_mob_to_pixels(carry, mob_index):
        pixels, mobs, texture_name, alpha_texture_name, mob_coordination, mob_agents_req = carry
        local_position = (
            mobs.position[state.player_level, mob_index]
            - state.player_position
            + jnp.ones((2,), dtype=jnp.int32) * (obs_dim_array // 2)
        )
        on_screen = jnp.logical_and(local_position >= 0, local_position < obs_dim_array).all(
            axis=-1
        )
        on_screen *= mobs.mask[state.player_level, mob_index]

        base_type_id = mobs.type_id[state.player_level, mob_index]
        is_elite = mob_coordination[state.player_level, mob_index] > 0

        # For elite cow (type_id=0), use buffalo (index 3) or large_cow (index 4)
        # based on coordination value (2=hard, 1=soft)
        is_cow = base_type_id == 0
        is_hard = mob_coordination[state.player_level, mob_index] == 2
        texture_id = jnp.where(
            is_elite & is_cow,
            jnp.where(is_hard, 3, 4),  # buffalo=3, large_cow=4
            base_type_id,  # regular texture
        )

        mob_texture = texture_name[texture_id]
        mob_texture_alpha = alpha_texture_name[texture_id]

        # Apply elite tint for non-cow elites (bat, snail stay with tint)
        # Cow variants already have distinct textures, so skip tint
        should_tint = is_elite & ~is_cow
        mob_texture = apply_elite_tint(mob_texture, should_tint)

        # Apply colored border encoding agents_required (orange=2, red=3+)
        agents_req = mob_agents_req[state.player_level, mob_index]
        mob_texture = apply_elite_border(mob_texture, mob_texture_alpha, agents_req)

        mob_texture_alpha = jax.vmap(jnp.multiply, in_axes=(None, 0))(mob_texture_alpha, on_screen)
        mob_texture = jax.vmap(jnp.multiply, in_axes=(None, 0))(mob_texture, on_screen)
        mob_texture_with_background = jax.vmap(_composite_texture, in_axes=(0, 0, 0, 0))(
            pixels, mob_texture, mob_texture_alpha, local_position
        )

        pixels = jax.vmap(_update_slice_pixel_map, in_axes=(0, 0, 0))(
            pixels, mob_texture_with_background, local_position
        )

        return (
            pixels,
            mobs,
            texture_name,
            alpha_texture_name,
            mob_coordination,
            mob_agents_req,
        ), None

    (map_pixels, _, _, _, _, _), _ = jax.lax.scan(
        _add_passive_mob_to_pixels,
        (
            map_pixels,
            state.passive_mobs,
            textures["passive_mob_textures"],
            textures["passive_mob_texture_alphas"],
            state.passive_mob_coordination,
            state.passive_mob_agents_required,
        ),
        jnp.arange(state.passive_mobs.mask.shape[1]),
    )

    # Ranged mobs with elite tint
    (map_pixels, _, _, _, _, _), _ = jax.lax.scan(
        _add_mob_to_pixels_with_elite,
        (
            map_pixels,
            state.ranged_mobs,
            textures["ranged_mob_textures"],
            textures["ranged_mob_texture_alphas"],
            state.ranged_mob_coordination,
            state.ranged_mob_agents_required,
        ),
        jnp.arange(state.ranged_mobs.mask.shape[1]),
    )

    def _add_projectile_to_pixels(carry, projectile_index):
        pixels, projectiles, projectile_directions = carry
        local_position = (
            projectiles.position[state.player_level, projectile_index]
            - state.player_position
            + jnp.ones((2,), dtype=jnp.int32) * (obs_dim_array // 2)
        )
        on_screen = jnp.logical_and(local_position >= 0, local_position < obs_dim_array).all(
            axis=-1
        )
        on_screen *= projectiles.mask[state.player_level, projectile_index]

        projectile_texture = textures["projectile_textures"][
            projectiles.type_id[state.player_level, projectile_index]
        ]
        projectile_texture_alpha = textures["projectile_texture_alphas"][
            projectiles.type_id[state.player_level, projectile_index]
        ]

        flipped_projectile_texture = jnp.flip(projectile_texture, axis=0)
        flipped_projectile_texture_alpha = jnp.flip(projectile_texture_alpha, axis=0)
        flip_projectile = jnp.logical_or(
            projectile_directions[state.player_level, projectile_index, 0] > 0,
            projectile_directions[state.player_level, projectile_index, 1] > 0,
        )

        projectile_texture = jax.lax.select(
            flip_projectile,
            flipped_projectile_texture,
            projectile_texture,
        )
        projectile_texture_alpha = jax.lax.select(
            flip_projectile,
            flipped_projectile_texture_alpha,
            projectile_texture_alpha,
        )

        transposed_projectile_texture = jnp.transpose(projectile_texture, (1, 0, 2))
        transposed_projectile_texture_alpha = jnp.transpose(projectile_texture_alpha, (1, 0, 2))

        projectile_texture = jax.lax.select(
            projectile_directions[state.player_level, projectile_index, 1] != 0,
            transposed_projectile_texture,
            projectile_texture,
        )
        projectile_texture_alpha = jax.lax.select(
            projectile_directions[state.player_level, projectile_index, 1] != 0,
            transposed_projectile_texture_alpha,
            projectile_texture_alpha,
        )

        projectile_texture = jax.vmap(jnp.multiply, in_axes=(None, 0))(
            projectile_texture, on_screen
        )
        projectile_texture_with_background = 1 - jax.vmap(jnp.multiply, in_axes=(None, 0))(
            projectile_texture_alpha, on_screen
        )

        projectile_texture_with_background = projectile_texture_with_background * jax.vmap(
            _slice_pixel_map, in_axes=(0, 0)
        )(pixels, local_position)

        projectile_texture_with_background = (
            projectile_texture_with_background + projectile_texture * projectile_texture_alpha
        )

        pixels = jax.vmap(_update_slice_pixel_map, in_axes=(0, 0, 0))(
            pixels, projectile_texture_with_background, local_position
        )

        return (pixels, projectiles, projectile_directions), None

    (map_pixels, _, _), _ = jax.lax.scan(
        _add_projectile_to_pixels,
        (map_pixels, state.mob_projectiles, state.mob_projectile_directions),
        jnp.arange(state.mob_projectiles.mask.shape[1]),
    )

    (map_pixels, _, _), _ = jax.lax.scan(
        _add_projectile_to_pixels,
        (map_pixels, state.player_projectiles, state.player_projectile_directions),
        jnp.arange(state.player_projectiles.mask.shape[1]),
    )

    # Apply darkness (underground)
    light_map = state.light_map[state.player_level]
    padded_light_map = jnp.pad(
        light_map,
        (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
        constant_values=False,
    )

    light_map_view = jax.vmap(jax.lax.dynamic_slice, in_axes=(None, 0, None))(
        padded_light_map, tl_corner, OBS_DIM
    )
    light_map_pixels = light_map_view.repeat(block_pixel_size, axis=1).repeat(
        block_pixel_size, axis=2
    )

    map_pixels = (light_map_pixels)[:, :, :, None] * map_pixels

    # Apply night
    night_pixels = textures["night_texture"]
    daylight = state.light_level
    daylight = jax.lax.select(state.player_level == 0, daylight, 1.0)

    if do_night_noise:
        night_noise = jax.random.uniform(state.state_rng, night_pixels.shape[:2]) * 95 + 32
        night_noise = jnp.expand_dims(night_noise, axis=-1).repeat(3, axis=-1)

        night_intensity = 2 * (0.5 - daylight)
        night_intensity = jnp.maximum(night_intensity, 0.0)
        night_mask = textures["night_noise_intensity_texture"] * night_intensity
        night = (1.0 - night_mask) * map_pixels + night_mask * night_noise

        night = night_pixels * 0.5 + 0.5 * night
        map_pixels = daylight * map_pixels + (1 - daylight) * night
    else:
        night_noise = jnp.full(night_pixels.shape, 64)

        night_intensity = 2 * (0.5 - daylight)
        night_intensity = jnp.maximum(night_intensity, 0.0)
        night_mask = (
            jnp.ones_like(textures["night_noise_intensity_texture"]) * night_intensity * 0.5
        )
        night = (1.0 - night_mask) * map_pixels + night_mask * night_noise

        night = night_pixels * 0.5 + 0.5 * night
        map_pixels = daylight * map_pixels + (1 - daylight) * night
        # map_pixels = daylight * map_pixels
        # night_noise = jnp.ones(night_pixels.shape[:2]) * 64

    # Apply sleep
    sleep_level = 1.0 - state.is_sleeping * 0.5
    map_pixels = jax.vmap(jnp.multiply, in_axes=(0, 0))(sleep_level, map_pixels)

    # RENDER INVENTORY
    inv_pixel_left_space = (block_pixel_size - int(0.8 * block_pixel_size)) // 2
    inv_pixel_right_space = block_pixel_size - int(0.8 * block_pixel_size) - inv_pixel_left_space

    inv_pixels = jnp.zeros(
        (
            map_pixels.shape[0],
            INVENTORY_OBS_HEIGHT * block_pixel_size,
            OBS_DIM[1] * block_pixel_size,
            3,
        ),
        dtype=jnp.float32,
    )

    number_size = int(block_pixel_size * 0.4)
    number_offset = block_pixel_size - number_size
    number_double_offset = block_pixel_size - 2 * number_size

    def _render_digit(pixels, number, x, y):
        pixels = pixels.at[
            y * block_pixel_size + number_offset : (y + 1) * block_pixel_size,
            x * block_pixel_size + number_offset : (x + 1) * block_pixel_size,
        ].mul(1 - textures["number_textures_alpha"][number])

        pixels = pixels.at[
            y * block_pixel_size + number_offset : (y + 1) * block_pixel_size,
            x * block_pixel_size + number_offset : (x + 1) * block_pixel_size,
        ].add(textures["number_textures"][number])

        return pixels

    def _render_two_digit_number(pixels, number, x, y):
        tens = number // 10
        ones = number % 10

        ones_textures = jax.lax.select(
            number == 0,
            textures["number_textures"],
            textures["number_textures_with_zero"],
        )

        ones_textures_alpha = jax.lax.select(
            number == 0,
            textures["number_textures_alpha"],
            textures["number_textures_alpha_with_zero"],
        )

        pixels = pixels.at[
            y * block_pixel_size + number_offset : (y + 1) * block_pixel_size,
            x * block_pixel_size + number_offset : (x + 1) * block_pixel_size,
        ].mul(1 - ones_textures_alpha[ones])

        pixels = pixels.at[
            y * block_pixel_size + number_offset : (y + 1) * block_pixel_size,
            x * block_pixel_size + number_offset : (x + 1) * block_pixel_size,
        ].add(ones_textures[ones])

        pixels = pixels.at[
            y * block_pixel_size + number_offset : (y + 1) * block_pixel_size,
            x * block_pixel_size + number_double_offset : x * block_pixel_size + number_offset,
        ].mul(1 - textures["number_textures_alpha"][tens])

        pixels = pixels.at[
            y * block_pixel_size + number_offset : (y + 1) * block_pixel_size,
            x * block_pixel_size + number_double_offset : x * block_pixel_size + number_offset,
        ].add(textures["number_textures"][tens])

        return pixels

    def _render_icon(pixels, texture, x, y):
        return pixels.at[
            block_pixel_size * y + inv_pixel_left_space : block_pixel_size * (y + 1)
            - inv_pixel_right_space,
            block_pixel_size * x + inv_pixel_left_space : block_pixel_size * (x + 1)
            - inv_pixel_right_space,
        ].set(texture)

    def _render_icon_with_alpha(pixels, texture, x, y):
        existing_slice = pixels[
            block_pixel_size * y + inv_pixel_left_space : block_pixel_size * (y + 1)
            - inv_pixel_right_space,
            block_pixel_size * x + inv_pixel_left_space : block_pixel_size * (x + 1)
            - inv_pixel_right_space,
        ]

        new_slice = (
            existing_slice * (1 - texture[:, :, 3][:, :, None])
            + texture[:, :, :3] * texture[:, :, 3][:, :, None]
        )

        return pixels.at[
            block_pixel_size * y + inv_pixel_left_space : block_pixel_size * (y + 1)
            - inv_pixel_right_space,
            block_pixel_size * x + inv_pixel_left_space : block_pixel_size * (x + 1)
            - inv_pixel_right_space,
        ].set(new_slice)

    def _render_icons(pixels, textures, locs):
        def _render_single_icon(carry, idx):
            pixels, textures, locs = carry
            icon_slice = textures[idx]
            pixels = jax.lax.dynamic_update_slice(
                pixels,
                icon_slice,
                (
                    block_pixel_size * locs[idx, 0] + inv_pixel_left_space,
                    block_pixel_size * locs[idx, 1] + inv_pixel_left_space,
                    0,
                ),
            )
            return (pixels, textures, locs), None

        (pixels, _, _), _ = jax.lax.scan(
            _render_single_icon, (pixels, textures, locs), jnp.arange(locs.shape[0])
        )
        return pixels

    def _render_icons_with_alpha(pixels, textures_rgba, locs):
        icon_h = textures_rgba.shape[1]
        icon_w = textures_rgba.shape[2]

        def _render_single_icon(carry, idx):
            pixels, textures_rgba, locs = carry
            icon_slice = textures_rgba[idx]
            start = (
                block_pixel_size * locs[idx, 0] + inv_pixel_left_space,
                block_pixel_size * locs[idx, 1] + inv_pixel_left_space,
                0,
            )
            existing_slice = jax.lax.dynamic_slice(
                pixels,
                start,
                (icon_h, icon_w, 3),
            )
            alpha = icon_slice[:, :, 3:]
            new_slice = existing_slice * (1.0 - alpha) + icon_slice[:, :, :3] * alpha
            pixels = jax.lax.dynamic_update_slice(pixels, new_slice, start)
            return (pixels, textures_rgba, locs), None

        (pixels, _, _), _ = jax.lax.scan(
            _render_single_icon, (pixels, textures_rgba, locs), jnp.arange(locs.shape[0])
        )
        return pixels

    def _render_two_digit_numbers(pixels, numbers, locs):
        tens = numbers // 10
        ones = numbers % 10

        ones_textures = jnp.where(
            (numbers == 0)[:, None, None, None, None],
            textures["number_textures"],
            textures["number_textures_with_zero"],
        )

        ones_textures_alpha = jnp.where(
            (numbers == 0)[:, None, None, None, None],
            textures["number_textures_alpha"],
            textures["number_textures_alpha_with_zero"],
        )

        def _render_single_two_digit_number(pixels, idx):
            ones_texture = ones_textures[idx, ones[idx]]
            ones_texture_alpha = ones_textures_alpha[idx, ones[idx]]
            tens_texture = textures["number_textures"][tens[idx]]
            tens_texture_alpha = textures["number_textures_alpha"][tens[idx]]

            # Render Ones
            original_ones_slice = jax.lax.dynamic_slice(
                pixels,
                (
                    block_pixel_size * locs[idx, 0] + number_offset,
                    block_pixel_size * locs[idx, 1] + number_offset,
                    0,
                ),
                (number_size, number_size, 3),
            )
            updated_ones_slice = original_ones_slice * (1 - ones_texture_alpha) + ones_texture
            pixels = jax.lax.dynamic_update_slice(
                pixels,
                updated_ones_slice,
                (
                    block_pixel_size * locs[idx, 0] + number_offset,
                    block_pixel_size * locs[idx, 1] + number_offset,
                    0,
                ),
            )

            # Render Tens
            original_tens_slice = jax.lax.dynamic_slice(
                pixels,
                (
                    block_pixel_size * locs[idx, 0] + number_offset,
                    block_pixel_size * locs[idx, 1] + number_double_offset,
                    0,
                ),
                (number_size, number_size, 3),
            )
            updated_tens_slice = original_tens_slice * (1 - tens_texture_alpha) + tens_texture
            pixels = jax.lax.dynamic_update_slice(
                pixels,
                updated_tens_slice,
                (
                    block_pixel_size * locs[idx, 0] + number_offset,
                    block_pixel_size * locs[idx, 1] + number_double_offset,
                    0,
                ),
            )

            return pixels, None

        pixels, _ = jax.lax.scan(
            _render_single_two_digit_number, pixels, jnp.arange(static_params.player_count)
        )

        return pixels

    def _render_dashboard(inv_pixels, player_index):
        # Render player stats
        player_health = jnp.maximum(jnp.floor(state.player_health[player_index]), 0).astype(int)
        health_texture = jax.lax.select(
            player_health > 0,
            textures["health_texture"],
            textures["smaller_empty_texture"],
        )
        inv_pixels = _render_icon(inv_pixels, health_texture, 0, 0)
        inv_pixels = _render_two_digit_number(inv_pixels, player_health, 0, 0)

        hunger_texture = jax.lax.select(
            state.player_food[player_index] > 0,
            textures["hunger_texture"],
            textures["smaller_empty_texture"],
        )
        inv_pixels = _render_icon(inv_pixels, hunger_texture, 1, 0)
        inv_pixels = _render_two_digit_number(inv_pixels, state.player_food[player_index], 1, 0)

        thirst_texture = jax.lax.select(
            state.player_drink[player_index] > 0,
            textures["thirst_texture"],
            textures["smaller_empty_texture"],
        )
        inv_pixels = _render_icon(inv_pixels, thirst_texture, 2, 0)
        inv_pixels = _render_two_digit_number(inv_pixels, state.player_drink[player_index], 2, 0)

        energy_texture = jax.lax.select(
            state.player_energy[player_index] > 0,
            textures["energy_texture"],
            textures["smaller_empty_texture"],
        )
        inv_pixels = _render_icon(inv_pixels, energy_texture, 3, 0)
        inv_pixels = _render_two_digit_number(inv_pixels, state.player_energy[player_index], 3, 0)

        mana_texture = jax.lax.select(
            state.player_mana[player_index] > 0,
            textures["mana_texture"],
            textures["smaller_empty_texture"],
        )
        inv_pixels = _render_icon(inv_pixels, mana_texture, 4, 0)
        inv_pixels = _render_two_digit_number(inv_pixels, state.player_mana[player_index], 4, 0)

        # Render inventory

        inv_wood_texture = jax.lax.select(
            state.inventory.wood[player_index] > 0,
            textures["smaller_block_textures"][BlockType.WOOD.value],
            textures["smaller_empty_texture"],
        )
        inv_pixels = _render_icon(inv_pixels, inv_wood_texture, 0, 2)
        inv_pixels = _render_two_digit_number(inv_pixels, state.inventory.wood[player_index], 0, 2)

        inv_stone_texture = jax.lax.select(
            state.inventory.stone[player_index] > 0,
            textures["smaller_block_textures"][BlockType.STONE.value],
            textures["smaller_empty_texture"],
        )
        inv_pixels = _render_icon(inv_pixels, inv_stone_texture, 1, 2)
        inv_pixels = _render_two_digit_number(inv_pixels, state.inventory.stone[player_index], 1, 2)

        inv_coal_texture = jax.lax.select(
            state.inventory.coal[player_index] > 0,
            textures["smaller_block_textures"][BlockType.COAL.value],
            textures["smaller_empty_texture"],
        )
        inv_pixels = _render_icon(inv_pixels, inv_coal_texture, 0, 1)
        inv_pixels = _render_two_digit_number(inv_pixels, state.inventory.coal[player_index], 0, 1)

        inv_iron_texture = jax.lax.select(
            state.inventory.iron[player_index] > 0,
            textures["smaller_block_textures"][BlockType.IRON.value],
            textures["smaller_empty_texture"],
        )
        inv_pixels = _render_icon(inv_pixels, inv_iron_texture, 1, 1)
        inv_pixels = _render_two_digit_number(inv_pixels, state.inventory.iron[player_index], 1, 1)

        inv_diamond_texture = jax.lax.select(
            state.inventory.diamond[player_index] > 0,
            textures["smaller_block_textures"][BlockType.DIAMOND.value],
            textures["smaller_empty_texture"],
        )
        inv_pixels = _render_icon(inv_pixels, inv_diamond_texture, 2, 1)
        inv_pixels = _render_two_digit_number(
            inv_pixels, state.inventory.diamond[player_index], 2, 1
        )

        inv_sapphire_texture = jax.lax.select(
            state.inventory.sapphire[player_index] > 0,
            textures["smaller_block_textures"][BlockType.SAPPHIRE.value],
            textures["smaller_empty_texture"],
        )
        inv_pixels = _render_icon(inv_pixels, inv_sapphire_texture, 3, 1)
        inv_pixels = _render_two_digit_number(
            inv_pixels, state.inventory.sapphire[player_index], 3, 1
        )

        inv_ruby_texture = jax.lax.select(
            state.inventory.ruby[player_index] > 0,
            textures["smaller_block_textures"][BlockType.RUBY.value],
            textures["smaller_empty_texture"],
        )
        inv_pixels = _render_icon(inv_pixels, inv_ruby_texture, 4, 1)
        inv_pixels = _render_two_digit_number(inv_pixels, state.inventory.ruby[player_index], 4, 1)

        inv_sapling_texture = jax.lax.select(
            state.inventory.sapling[player_index] > 0,
            textures["sapling_texture"],
            textures["smaller_empty_texture"],
        )
        inv_pixels = _render_icon(inv_pixels, inv_sapling_texture, 5, 1)
        inv_pixels = _render_two_digit_number(
            inv_pixels, state.inventory.sapling[player_index], 5, 1
        )

        # Render tools
        # Pickaxe
        pickaxe_texture = textures["pickaxe_textures"][state.inventory.pickaxe[player_index]]
        inv_pixels = _render_icon(inv_pixels, pickaxe_texture, 8, 2)

        # Sword
        sword_texture = textures["sword_textures"][state.inventory.sword[player_index]]
        inv_pixels = _render_icon(inv_pixels, sword_texture, 8, 1)

        # Bow and arrows
        bow_texture = textures["bow_textures"][state.inventory.bow[player_index]]
        inv_pixels = _render_icon(inv_pixels, bow_texture, 6, 1)

        arrow_texture = jax.lax.select(
            state.inventory.arrows[player_index] > 0,
            textures["player_projectile_textures"][0],
            textures["smaller_empty_texture"],
        )
        inv_pixels = _render_icon(inv_pixels, arrow_texture, 6, 2)
        inv_pixels = _render_two_digit_number(
            inv_pixels, state.inventory.arrows[player_index], 6, 2
        )

        # Armour
        for i in range(4):
            armour_texture = textures["armour_textures"][state.inventory.armour[player_index][i], i]
            inv_pixels = _render_icon(inv_pixels, armour_texture, 7, i)

        # Torch
        torch_texture = jax.lax.select(
            state.inventory.torches[player_index] > 0,
            textures["torch_inv_texture"],
            textures["smaller_empty_texture"],
        )
        inv_pixels = _render_icon(inv_pixels, torch_texture, 2, 2)
        inv_pixels = _render_two_digit_number(
            inv_pixels, state.inventory.torches[player_index], 2, 2
        )

        # Potions
        potion_names = ["red", "green", "blue", "pink", "cyan", "yellow"]
        for potion_index, potion_name in enumerate(potion_names):
            potion_texture = jax.lax.select(
                state.inventory.potions[player_index][potion_index] > 0,
                textures["potion_textures"][potion_index],
                textures["smaller_empty_texture"],
            )
            inv_pixels = _render_icon(inv_pixels, potion_texture, potion_index, 3)
            inv_pixels = _render_two_digit_number(
                inv_pixels,
                state.inventory.potions[player_index][potion_index],
                potion_index,
                3,
            )

        # Books
        book_texture = jax.lax.select(
            state.inventory.books[player_index] > 0,
            textures["book_texture"],
            textures["smaller_empty_texture"],
        )
        inv_pixels = _render_icon(inv_pixels, book_texture, 3, 2)
        inv_pixels = _render_two_digit_number(inv_pixels, state.inventory.books[player_index], 3, 2)

        # Learned spells
        spell_texture = jax.lax.select(
            state.player_specialization[player_index] == Specialization.FORAGER.value,
            textures["heal_inv_texture"],
            textures["fireball_inv_texture"],
        )
        spell_texture = jax.lax.select(
            jnp.logical_and(
                state.player_specialization[player_index] != Specialization.UNASSIGNED.value,
                state.learned_spells[player_index],
            ),
            spell_texture,
            textures["smaller_empty_texture"],
        )
        inv_pixels = _render_icon(inv_pixels, spell_texture, 4, 2)

        # Enchantments
        sword_enchantment_texture = textures["sword_enchantment_textures"][
            state.sword_enchantment[player_index]
        ]
        inv_pixels = _render_icon_with_alpha(inv_pixels, sword_enchantment_texture, 8, 1)

        arrow_enchantment_level = state.bow_enchantment[player_index] * (
            state.inventory.arrows[player_index] > 0
        )
        arrow_enchantment_texture = textures["arrow_enchantment_textures"][arrow_enchantment_level]
        inv_pixels = _render_icon_with_alpha(inv_pixels, arrow_enchantment_texture, 6, 2)

        for i in range(4):
            armour_enchantment_texture = textures["armour_enchantment_textures"][
                state.armour_enchantments[player_index][i], i
            ]
            inv_pixels = _render_icon_with_alpha(inv_pixels, armour_enchantment_texture, 7, i)

        # Dungeon level
        inv_pixels = _render_digit(inv_pixels, state.player_level, 6, 0)

        # Attributes
        xp_texture = jax.lax.select(
            state.player_xp[player_index] > 0,
            textures["xp_texture"],
            textures["smaller_empty_texture"],
        )
        inv_pixels = _render_icon(inv_pixels, xp_texture, 9, 0)
        inv_pixels = _render_digit(inv_pixels, state.player_xp[player_index], 9, 0)

        inv_pixels = _render_icon(inv_pixels, textures["dex_texture"], 9, 1)
        inv_pixels = _render_digit(inv_pixels, state.player_dexterity[player_index], 9, 1)

        inv_pixels = _render_icon(inv_pixels, textures["str_texture"], 9, 2)
        inv_pixels = _render_digit(inv_pixels, state.player_strength[player_index], 9, 2)

        inv_pixels = _render_icon(inv_pixels, textures["int_texture"], 9, 3)
        inv_pixels = _render_digit(inv_pixels, state.player_intelligence[player_index], 9, 3)

        # Specializations
        picked_specialization_texture = (
            (state.player_specialization[player_index] == Specialization.FORAGER.value)
            * textures["forager_texture"]
            + (state.player_specialization[player_index] == Specialization.WARRIOR.value)
            * textures["warrior_texture"]
            + (state.player_specialization[player_index] == Specialization.MINER.value)
            * textures["miner_texture"]
        )
        spec_texture = jax.lax.select(
            state.player_specialization[player_index] == Specialization.UNASSIGNED.value,
            textures["smaller_empty_texture"],
            picked_specialization_texture,
        )
        inv_pixels = _render_icon(inv_pixels, spec_texture, 8, 0)

        return inv_pixels

    inv_pixels = jax.vmap(_render_dashboard, in_axes=(0, 0))(
        inv_pixels, jnp.arange(static_params.player_count)
    )

    def _render_teammate_info(player_index):
        info_pixels = jnp.zeros(
            (
                (static_params.player_count + 1) // 2 * block_pixel_size,
                OBS_DIM[1] * block_pixel_size,
                3,
            ),
            dtype=jnp.float32,
        )

        # quick icon location calc
        # player_icon_locations = get_player_icon_positions(static_params.player_count)
        player_icon_locations = get_player_icon_positions(static_params.player_count)

        # Render players icons
        player_icon_to_render = jnp.where(
            state.player_alive[:, None, None, None],
            player_specific_textures.player_icon_textures[:, 0],
            player_specific_textures.player_icon_textures[:, 1],
        )

        info_pixels = _render_icons(info_pixels, player_icon_to_render, player_icon_locations)

        if static_params.num_comm_channels > 0:
            _COMM_COLORS = jnp.array(
                [
                    [220, 50, 50],  # 0: red
                    [220, 220, 50],  # 1: yellow
                    [50, 220, 220],  # 2: cyan
                    [220, 50, 220],  # 3: magenta
                    [220, 150, 50],  # 4: orange
                    [50, 220, 50],  # 5: green
                    [50, 100, 220],  # 6: blue
                    [200, 200, 200],  # 7: white
                ],
                dtype=jnp.float32,
            )
            active_ch = jnp.argmax(state.comm_messages, axis=-1)
            has_comm = state.comm_messages.any(axis=-1)
            comm_colors = _COMM_COLORS[active_ch % 8]
            base_badge = textures["comm_badge_texture"].astype(jnp.float32)
            badge_rgb = base_badge[:, :, :3]
            badge_alpha = base_badge[:, :, 3:]
            tinted_badges = jnp.zeros(
                (static_params.player_count, *base_badge.shape), dtype=jnp.float32
            )
            tinted_badges = tinted_badges.at[:, :, :, :3].set(
                jnp.clip(
                    badge_rgb[None, :, :, :] * 0.82 + comm_colors[:, None, None, :] * 0.18,
                    0.0,
                    255.0,
                )
            )
            tinted_badges = tinted_badges.at[:, :, :, 3:].set(
                badge_alpha[None, :, :, :] * has_comm[:, None, None, None]
            )

            info_pixels = _render_icons_with_alpha(
                info_pixels, tinted_badges, player_icon_locations
            )

            badge_start_x = int(textures["comm_badge_start_x"])
            badge_start_y = int(textures["comm_badge_start_y"])
            badge_size = int(textures["comm_badge_size"])
            digit_size = textures["comm_number_textures_with_zero"].shape[1]
            digit_y = badge_start_y + (badge_size - digit_size) // 2 - 1
            digit_x = badge_start_x + (badge_size - digit_size) // 2 - 1

            def _place_comm_channel_number(carry, idx):
                pixels = carry

                def _render_number(current_pixels):
                    channel_digit = (active_ch[idx] + 1) % 10
                    digit_alpha = textures["comm_number_textures_alpha_with_zero"][
                        channel_digit
                    ].astype(jnp.float32)
                    row, col = player_icon_locations[idx]
                    digit_py = block_pixel_size * row + inv_pixel_left_space + digit_y
                    digit_px = block_pixel_size * col + inv_pixel_left_space + digit_x

                    digit_background = jax.lax.dynamic_slice(
                        current_pixels,
                        (digit_py, digit_px, 0),
                        (digit_size, digit_size, 3),
                    )
                    digit_bg_fill = jnp.clip(comm_colors[idx] * 0.32 + 18.0, 0.0, 255.0)
                    digit_background = (
                        digit_background * (1.0 - digit_alpha)
                        + digit_bg_fill[None, None, :] * digit_alpha
                    )
                    current_pixels = jax.lax.dynamic_update_slice(
                        current_pixels,
                        digit_background,
                        (digit_py, digit_px, 0),
                    )

                    digit_foreground = jax.lax.dynamic_slice(
                        current_pixels,
                        (digit_py, digit_px, 0),
                        (digit_size, digit_size, 3),
                    )
                    digit_color = jnp.full((digit_size, digit_size, 3), 12.0, dtype=jnp.float32)
                    digit_overlay = (
                        digit_foreground * (1.0 - digit_alpha) + digit_color * digit_alpha
                    )
                    current_pixels = jax.lax.dynamic_update_slice(
                        current_pixels,
                        digit_overlay,
                        (digit_py, digit_px, 0),
                    )
                    return current_pixels

                pixels = jax.lax.cond(has_comm[idx], _render_number, lambda x: x, pixels)
                return pixels, None

            info_pixels, _ = jax.lax.scan(
                _place_comm_channel_number,
                info_pixels,
                jnp.arange(static_params.player_count),
            )

        # Render teammate healths
        health_icon_locations = player_icon_locations + jnp.array([0, 1])
        # teammate_health = jnp.maximum(
        #     jnp.floor(state.player_health), 1
        # ).astype(int)
        teammate_health = jnp.where(
            state.player_alive,
            jnp.maximum(jnp.floor(state.player_health), 0),
            0,
        ).astype(int)
        health_texture = jnp.where(
            (teammate_health > 0)[:, None, None, None],
            textures["health_texture"],
            textures["smaller_empty_texture"],
        ).astype(float)
        info_pixels = _render_icons(info_pixels, health_texture, health_icon_locations)
        info_pixels = _render_two_digit_numbers(info_pixels, teammate_health, health_icon_locations)

        # Render teammate directions
        direction_icon_locations = player_icon_locations + jnp.array([0, 2])
        local_position = (
            state.player_position
            - state.player_position[player_index]
            + jnp.ones((2,), dtype=jnp.int32) * (obs_dim_array // 2)
        )
        on_screen = jnp.logical_and(local_position >= 0, local_position < obs_dim_array).all(
            axis=-1
        )
        render_direction = jnp.logical_not(on_screen)

        direction_index_2d = jnp.where(
            local_position < 0, 0, jnp.where(local_position >= obs_dim_array, 2, 1)
        )
        direction_texture = textures["direction_textures"][
            direction_index_2d[:, 0], direction_index_2d[:, 1]
        ][:, :, :, :3]
        direction_texture = jax.vmap(jnp.multiply, in_axes=(0, 0))(
            direction_texture, render_direction
        ).astype(float)
        info_pixels = _render_icons(info_pixels, direction_texture, direction_icon_locations)

        # Render Teammate Specializations
        spec_icon_locations = player_icon_locations + jnp.array([0, 3])
        spec_texture = (
            (state.player_specialization == Specialization.FORAGER.value)[:, None, None, None]
            * textures["forager_texture"]
            + (state.player_specialization == Specialization.WARRIOR.value)[:, None, None, None]
            * textures["warrior_texture"]
            + (state.player_specialization == Specialization.MINER.value)[:, None, None, None]
            * textures["miner_texture"]
            + (state.player_specialization == Specialization.UNASSIGNED.value)[:, None, None, None]
            * textures["smaller_empty_texture"]
        ).astype(float)
        info_pixels = _render_icons(info_pixels, spec_texture, spec_icon_locations)

        # Render Teammate Messages
        message_icon_locations = player_icon_locations + jnp.array([0, 4])
        max_request_idx = textures["request_message_textures"].shape[0] - 1
        message_texture_index = jnp.clip(
            state.request_type - Action.REQUEST_FOOD.value,
            0,
            max_request_idx,
        )
        has_request = state.request_duration > 0
        message_texture = jnp.where(
            has_request[:, None, None, None],
            textures["request_message_textures"][message_texture_index][:, :, :, :3],
            textures["smaller_empty_texture"][None, :],
        ).astype(float)
        info_pixels = _render_icons(info_pixels, message_texture, message_icon_locations)

        return info_pixels

    teammate_info_pixels = jax.vmap(_render_teammate_info)(jnp.arange(static_params.player_count))

    # Combine map and inventory
    pixels = jnp.concatenate([teammate_info_pixels, map_pixels, inv_pixels], axis=1)

    # # Downscale by 2
    # pixels = pixels[::downscale, ::downscale]

    return pixels
