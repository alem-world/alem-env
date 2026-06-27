"""ASCII map renderer for Alem-Coop LLM observations.

Provides render_ascii_map(), a drop-in replacement for AlemLanguageWrapper.describe_env()
that returns a 9×11 character grid instead of a "You see:" text list.

Rendering priority (highest overwrites lower):
  1. Agent overlay    — @=self  0/1/2=teammates
  2. Mob/projectile   — Z=melee  K=ranged  A=animal  !=projectile
  3. Item layer       — o=torch  v=ladder↓  ^=ladder↑  x=blocked
  4. Block layer      — see _ASCII_BLOCK
  5. Darkness         — ?=unlit cell (light_map ≤ 0.05)
"""

import numpy as np

from alem.alem_coop.constants import MAX_OBS_DIM, OBS_DIM, BlockType, ItemType

# ---------------------------------------------------------------------------
# Symbol tables
# ---------------------------------------------------------------------------

# One character per BlockType value.  All 42 values must be present.
ASCII_BLOCK = {
    BlockType.INVALID.value: "?",
    BlockType.OUT_OF_BOUNDS.value: " ",
    BlockType.GRASS.value: ".",
    BlockType.WATER.value: "~",
    BlockType.STONE.value: "s",
    BlockType.TREE.value: "T",
    BlockType.WOOD.value: "W",
    BlockType.PATH.value: ".",
    BlockType.COAL.value: "c",
    BlockType.IRON.value: "i",
    BlockType.DIAMOND.value: "d",
    BlockType.CRAFTING_TABLE.value: "=",
    BlockType.FURNACE.value: "f",
    BlockType.SAND.value: ".",
    BlockType.LAVA.value: "L",
    BlockType.PLANT.value: "p",
    BlockType.RIPE_PLANT.value: "P",
    BlockType.WALL.value: "#",
    BlockType.DARKNESS.value: ".",  # dungeon floor — visible when lit
    BlockType.WALL_MOSS.value: "#",
    BlockType.STALAGMITE.value: "*",
    BlockType.SAPPHIRE.value: "q",
    BlockType.RUBY.value: "r",
    BlockType.CHEST.value: "X",
    BlockType.FOUNTAIN.value: "U",
    BlockType.FIRE_GRASS.value: ".",
    BlockType.ICE_GRASS.value: ".",
    BlockType.GRAVEL.value: ".",
    BlockType.FIRE_TREE.value: "t",
    BlockType.ICE_SHRUB.value: "y",
    BlockType.ENCHANTMENT_TABLE_FIRE.value: "E",
    BlockType.ENCHANTMENT_TABLE_ICE.value: "e",
    BlockType.NECROMANCER.value: "N",
    BlockType.GRAVE.value: "G",
    BlockType.GRAVE2.value: "G",
    BlockType.GRAVE3.value: "G",
    BlockType.NECROMANCER_VULNERABLE.value: "V",
    BlockType.CONSTRUCTION_SITE.value: "C",
    BlockType.EPIC_SHELTER.value: "H",
    BlockType.EPIC_FORGE.value: "F",
    BlockType.EPIC_BEACON.value: "B",
    BlockType.CONSTRUCTION_IN_PROGRESS.value: "+",
}

ASCII_ITEM = {
    ItemType.TORCH.value: "o",
    ItemType.LADDER_DOWN.value: "v",
    ItemType.LADDER_UP.value: "^",
    ItemType.LADDER_DOWN_BLOCKED.value: "x",
}

# Human-readable label for each symbol (used to build the per-view legend).
# Agent digit chars '0'-'3' are generated dynamically in render_ascii_map.
ASCII_LEGEND = {
    ".": "ground",
    "~": "water",
    "#": "wall",
    "*": "stalagmite",
    "L": "lava",
    "?": "dark",
    " ": "void",
    "s": "stone",
    "c": "coal",
    "i": "iron",
    "d": "diamond",
    "r": "ruby",
    "q": "sapphire",
    "T": "tree",
    "t": "fire_tree",
    "y": "ice_shrub",
    "W": "wood",
    "p": "plant",
    "P": "ripe_plant",
    "=": "table",
    "f": "furnace",
    "X": "chest",
    "U": "fountain",
    "E": "ench_fire",
    "e": "ench_ice",
    "G": "grave",
    "N": "necromancer",
    "V": "necro(vuln)",
    "C": "constr_site",
    "+": "constr_wip",
    "H": "shelter",
    "F": "forge",
    "B": "beacon",
    "o": "torch",
    "v": "ladder_down",
    "^": "ladder_up",
    "x": "blocked_down",
    "@": "you",
    "Z": "melee_mob",
    "K": "ranged_mob",
    "A": "animal",
    "!": "projectile",
}

# Legend display order: most important / most commonly seen first.
ASCII_LEGEND_ORDER = [
    "@",
    "0",
    "1",
    "2",
    "3",  # agents
    "s",
    "c",
    "i",
    "d",
    "r",
    "q",  # resources
    "T",
    "t",
    "y",
    "W",
    "p",
    "P",  # plants / trees
    "C",
    "+",
    "H",
    "F",
    "B",  # construction
    "Z",
    "K",
    "A",
    "!",  # enemies / projectiles
    "=",
    "f",
    "X",
    "U",
    "E",
    "e",  # crafting / interactables
    "N",
    "V",
    "G",  # boss / graves
    "o",
    "v",
    "^",
    "x",  # items (torch / ladders)
    "~",
    "L",
    "#",
    "*",  # terrain hazards
    "?",  # darkness
]

# Facing direction mappings (Action enum values: 0=noop 1=LEFT 2=RIGHT 3=UP 4=DOWN)
_FACING_ARROW = {0: " ", 1: "←", 2: "→", 3: "↑", 4: "↓"}
_FACING_VEC = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}  # (row_delta, col_delta)


# ---------------------------------------------------------------------------
# Public rendering function
# ---------------------------------------------------------------------------


def render_ascii_map(
    state,
    player_idx,
    num_agents,
    direction_names,
    block_id_to_name,
    egocentric,
    entity_at_fn,
    light_mask_fn,
):
    """Render the local OBS_DIM view as a compact ASCII grid string.

    Parameters match what AlemLanguageWrapper already has at hand — pass
    them in so this function has no direct dependency on the wrapper instance.

    Args:
        state:            EnvState (JAX pytree, converted to numpy internally).
        player_idx:       Index of the agent whose viewpoint to render.
        num_agents:       Total number of agents.
        direction_names:  Dict mapping direction int → name string.
        block_id_to_name: Dict mapping block int id → lowercase name string.
        egocentric:       Unused here (facing line uses cardinal directions).
        entity_at_fn:     Callable(state, player_idx, world_pos, level) → str|None.
        light_mask_fn:    Callable(state, player_idx) → bool array [view_h, view_w].

    Returns:
        Multi-line string: N-compass header + 9×11 grid + inline legend + facing line.
    """
    view_h, view_w = OBS_DIM  # (9, 11)
    half_h, half_w = view_h // 2, view_w // 2  # (4, 5)

    current_level = int(state.player_level)
    player_pos = np.array(state.player_position[player_idx])
    pr, pc = int(player_pos[0]), int(player_pos[1])

    # --- Local map and item slices (identical padding to describe_env) ---
    pad = MAX_OBS_DIM + 2
    map_arr = np.array(state.map[current_level])
    item_arr = np.array(state.item_map[current_level])
    padded_map = np.pad(map_arr, pad, constant_values=BlockType.OUT_OF_BOUNDS.value)
    padded_item = np.pad(item_arr, pad, constant_values=ItemType.NONE.value)
    tl_r = pr - half_h + pad
    tl_c = pc - half_w + pad
    local_map = padded_map[tl_r : tl_r + view_h, tl_c : tl_c + view_w]
    local_item = padded_item[tl_r : tl_r + view_h, tl_c : tl_c + view_w]
    light_mask = light_mask_fn(state, player_idx)  # bool [view_h, view_w]

    # --- Layer 1: block + item ---
    grid = [["." for _ in range(view_w)] for _ in range(view_h)]
    for r in range(view_h):
        for c in range(view_w):
            if not light_mask[r, c]:
                grid[r][c] = "?"
                continue
            iid = int(local_item[r, c])
            if iid != ItemType.NONE.value and iid in ASCII_ITEM:
                grid[r][c] = ASCII_ITEM[iid]
            else:
                grid[r][c] = ASCII_BLOCK.get(int(local_map[r, c]), "?")

    # --- Layer 2: mob overlay ---
    for mob_state, mob_char in [
        (state.melee_mobs, "Z"),
        (state.ranged_mobs, "K"),
        (state.passive_mobs, "A"),
    ]:
        mob_mask = np.array(mob_state.mask[current_level])
        mob_pos = np.array(mob_state.position[current_level])
        for mi in range(len(mob_mask)):
            if int(mob_mask[mi]) < 1:
                continue
            mr, mc = int(mob_pos[mi, 0]), int(mob_pos[mi, 1])
            lr, lc = mr - pr + half_h, mc - pc + half_w
            if 0 <= lr < view_h and 0 <= lc < view_w and light_mask[lr, lc]:
                grid[lr][lc] = mob_char

    # --- Layer 3: projectile overlay ---
    for proj_state in (state.mob_projectiles, state.player_projectiles):
        proj_mask = np.array(proj_state.mask[current_level])
        proj_pos = np.array(proj_state.position[current_level])
        for pi in range(len(proj_mask)):
            if int(proj_mask[pi]) < 1:
                continue
            mr, mc = int(proj_pos[pi, 0]), int(proj_pos[pi, 1])
            lr, lc = mr - pr + half_h, mc - pc + half_w
            if 0 <= lr < view_h and 0 <= lc < view_w and light_mask[lr, lc]:
                grid[lr][lc] = "!"

    # --- Layer 4: agent overlay (highest priority) ---
    for i in range(num_agents):
        ar = int(state.player_position[i][0]) - pr + half_h
        ac = int(state.player_position[i][1]) - pc + half_w
        if not (0 <= ar < view_h and 0 <= ac < view_w):
            continue
        if i == player_idx:
            grid[ar][ac] = "@"
        elif bool(state.player_alive[i]):
            grid[ar][ac] = str(i)

    # --- Facing line (matches describe_env behaviour exactly) ---
    direction = int(state.player_direction[player_idx])
    facing_name = direction_names.get(direction, "none")
    facing_arrow = _FACING_ARROW.get(direction, " ")
    facing_target = None  # None → "Facing: none"

    if direction != 0:
        fv = _FACING_VEC[direction]  # (row_delta, col_delta)
        tlr, tlc = half_h + fv[0], half_w + fv[1]
        if 0 <= tlr < view_h and 0 <= tlc < view_w and light_mask[tlr, tlc]:
            tile_name = block_id_to_name.get(int(local_map[tlr, tlc]), "unknown")
            entity = entity_at_fn(state, player_idx, player_pos + np.array(fv), current_level)
            facing_target = f"{entity} (on {tile_name})" if entity else tile_name
        else:
            facing_target = "darkness"

    # --- Legend: only symbols actually present in the grid ---
    used = set()
    for row in grid:
        used.update(row)
    used.discard(" ")  # out-of-bounds — obvious
    used.discard(".")  # ground — obvious

    legend_entries = []
    seen = set()
    for ch in ASCII_LEGEND_ORDER:
        if ch in used and ch in ASCII_LEGEND and ch not in seen:
            legend_entries.append(f"{ch}={ASCII_LEGEND[ch]}")
            seen.add(ch)
    # Agent digit chars are generated dynamically
    for i in range(num_agents):
        ch = str(i)
        if ch in used and ch not in seen:
            legend_entries.append(f"{ch}=Agent{i}")
            seen.add(ch)

    # --- Render ---
    indent = "  "
    n_offset = " " * (half_w * 2)  # N above centre column
    lines = [f"{indent}{n_offset}N"]

    # 3 legend entries per grid row, inline to the right
    chunks = ["  ".join(legend_entries[k : k + 3]) for k in range(0, len(legend_entries), 3)]

    for r_idx in range(view_h):
        row_str = " ".join(grid[r_idx])
        leg = f"   {chunks[r_idx]}" if r_idx < len(chunks) else ""
        lines.append(f"{indent}{row_str}{leg}")

    # Overflow legend rows (more than 9 * 3 = 27 entries — rare)
    for k in range(view_h, len(chunks)):
        lines.append(f"{indent}{' ' * (view_w * 2 - 1)}   {chunks[k]}")

    if facing_target is not None:
        lines.append(
            f"\n{indent}Facing: {facing_name} {facing_arrow} \n  (Do target: {facing_target})"
        )
    else:
        lines.append(f"\n{indent}Facing: none")

    return "\n".join(lines)
