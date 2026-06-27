"""
Language Wrapper for Alem-Coop Environment (alem)
Converts symbolic observations to text descriptions for LLM agents.
"""

import logging
import re

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from scipy import ndimage

from alem.alem_coop.action_masking import compute_action_mask
from alem.alem_coop.alem_state import (
    EnvParams,
    StaticEnvParams,
    get_coordination_params,
)
from alem.alem_coop.constants import (
    BEACON_COST_COAL,
    BEACON_COST_IRON,
    BLOCK_PIXEL_SIZE_AGENT,
    FLOOR_MOB_MAPPING,
    FORGE_COST_COAL,
    FORGE_COST_IRON,
    FORGE_COST_STONE,
    MAX_OBS_DIM,
    MONSTERS_KILLED_TO_CLEAR_LEVEL,
    OBS_DIM,
    SHELTER_COST_STONE,
    SHELTER_COST_WOOD,
    TEXTURES,
    Achievement,
    Action,
    BlockType,
    ItemType,
    Specialization,
    load_player_specific_textures,
)
from alem.alem_coop.renderer.renderer_pixels import render_alem_pixels
from alem.alem_coop.util.game_logic_utils import (
    is_boss_vulnerable,
)
from alem.alem_env import make_alem_env_from_name
from alem.llm.action_parser import extract_action_multistrategy
from alem.llm.ascii_map import render_ascii_map as _render_ascii_map

logger = logging.getLogger(__name__)


def make_alem_env(config):
    """Create an alem env by name with configuration.

    All EnvParams are set explicitly so that text, symbolic, and pixel
    interfaces evaluate the same underlying environment configuration.

    Args:
        config: Configuration dict with env params

    Returns:
        Alem env instance
    """
    env_kwargs = {}

    env_kwargs["max_timesteps"] = config.get("max_timesteps", 10000)
    env_kwargs["god_mode"] = config.get("god_mode", False)

    # Coordination settings
    coordination_difficulty = config.get("coordination_difficulty", "none")
    if coordination_difficulty != "none":
        coord_params = get_coordination_params(coordination_difficulty)
        env_kwargs.update(coord_params)

    # Specialization & reward — must match RL eval for comparability
    env_kwargs["soft_specialization"] = config.get("soft_specialization", True)
    env_kwargs["shared_reward"] = config.get("shared_reward", False)
    env_kwargs["specialist_efficiency"] = config.get("specialist_efficiency", 1.0)
    env_kwargs["non_specialist_efficiency"] = config.get("non_specialist_efficiency", 0.2)
    env_kwargs["randomize_alpha"] = config.get("randomize_alpha", False)

    env_params = EnvParams(**env_kwargs)

    num_agents = config.get("num_agents", None)
    static_env_params = StaticEnvParams(player_count=num_agents) if num_agents is not None else None

    env_name = config.get("ENV_NAME", "Alem-Coop-Symbolic")
    if env_name == "Alem-Coop-Symbolic":
        env_name = "Alem-Coop-Symbolic"
    elif env_name == "Alem-Coop-Symbolic-Debug":
        env_name = "Alem-Coop-Symbolic-Debug"
    elif env_name == "Alem-Coop-Pixels":
        env_name = "Alem-Coop-Pixels"
    env = make_alem_env_from_name(
        env_name, env_params=env_params, static_env_params=static_env_params
    )
    return env


# ============================================================================
# Action Space (55 canonical labels; targeted Give actions map to extra slots)
# ============================================================================

ACTIONS = [
    "Noop",  # 0: NOOP
    "Move West",  # 1: LEFT
    "Move East",  # 2: RIGHT
    "Move North",  # 3: UP
    "Move South",  # 4: DOWN
    "Do",  # 5: DO
    "Sleep",  # 6: SLEEP
    "Place Stone",  # 7: PLACE_STONE
    "Place Table",  # 8: PLACE_TABLE
    "Place Furnace",  # 9: PLACE_FURNACE
    "Place Plant",  # 10: PLACE_PLANT
    "Make Wood Pickaxe",  # 11
    "Make Stone Pickaxe",  # 12
    "Make Iron Pickaxe",  # 13
    "Make Wood Sword",  # 14
    "Make Stone Sword",  # 15
    "Make Iron Sword",  # 16
    "Rest",  # 17
    "Descend",  # 18
    "Ascend",  # 19
    "Make Diamond Pickaxe",  # 20
    "Make Diamond Sword",  # 21
    "Make Iron Armour",  # 22
    "Make Diamond Armour",  # 23
    "Shoot Arrow",  # 24
    "Make Arrow",  # 25
    "Cast Spell",  # 26
    "Place Torch",  # 27
    "Drink Potion Red",  # 28
    "Drink Potion Green",  # 29
    "Drink Potion Blue",  # 30
    "Drink Potion Pink",  # 31
    "Drink Potion Cyan",  # 32
    "Drink Potion Yellow",  # 33
    "Read Book",  # 34
    "Enchant Sword",  # 35
    "Enchant Armour",  # 36
    "Make Torch",  # 37
    "Level Up Dexterity",  # 38
    "Level Up Strength",  # 39
    "Level Up Intelligence",  # 40
    "Enchant Bow",  # 41
    "Request Food",  # 42
    "Request Drink",  # 43
    "Request Wood",  # 44
    "Request Stone",  # 45
    "Request Iron",  # 46
    "Request Coal",  # 47
    "Request Diamond",  # 48
    "Request Ruby",  # 49
    "Request Sapphire",  # 50
    "Build Shelter",  # 51
    "Build Forge",  # 52
    "Build Beacon",  # 53
    "Give",  # 54
]

# Level names match the "Enter X" achievements
LEVEL_NAMES = {
    0: "Overworld",
    1: "Gnomish Mines",
    2: "Dungeon",
    3: "Sewers",
    4: "Vault",
    5: "Troll Mines",
    6: "Fire Realm",
    7: "Ice Realm",
    8: "Graveyard",
}

# Build mapping from Action enum value to ACTIONS index
_ACTION_ENUM_VALUES = [a.value for a in Action]
_ACTION_NAME_TO_ENUM_VALUE = {}
for i, name in enumerate(ACTIONS):
    if i < len(_ACTION_ENUM_VALUES):
        _ACTION_NAME_TO_ENUM_VALUE[name] = _ACTION_ENUM_VALUES[i]

# Actions that are always valid — omitted from affordance lists to save tokens
_ALWAYS_VALID_ACTIONS = frozenset(
    {
        "Noop",
        "Move West",
        "Move East",
        "Move North",
        "Move South",
        "Do",
        "Sleep",
        "Rest",
        "Request Food",
        "Request Drink",
        "Request Wood",
        "Request Stone",
        "Request Iron",
        "Request Coal",
        "Request Diamond",
        "Request Ruby",
        "Request Sapphire",
    }
)

ACTION_DICT = {
    "Noop": "do nothing",
    "Move West": "move west",
    "Move East": "move east",
    "Move North": "move north",
    "Move South": "move south",
    "Do": "interact with the tile you are facing — chop trees, mine resources (requires a matching pickaxe tier: stone/coal needs wood pickaxe, iron needs stone pickaxe, diamond needs iron pickaxe, ruby/sapphire need diamond pickaxe), attack creatures, drink from water tiles, open chests, or work on a construction/coordination target. If the faced tile contains a downed teammate, Do revives them. If the faced tile contains a living teammate, Do targets them instead and can cause friendly fire. For sync-style coordination, all required agents must stand on tiles adjacent to the shared target, each facing it, and choose Do on the same turn.",
    "Sleep": "restore energy at 2x the normal rate (use when energy is low). Sleep ends automatically once energy is full",
    "Place Stone": "place a stone block on the tile you face (costs 1 stone)",
    "Place Table": "place a crafting table on the tile you face (costs 2 wood)",
    "Place Furnace": "place a furnace on the tile you face (costs 1 stone)",
    "Place Plant": "place a sapling on the tile you face",
    "Make Wood Pickaxe": "craft a wood pickaxe (need: adjacent table, 1 wood)",
    "Make Stone Pickaxe": "craft a stone pickaxe (need: adjacent table, 1 wood + 1 stone)",
    "Make Iron Pickaxe": "craft an iron pickaxe (need: adjacent table + furnace, 1 wood + 1 stone + 1 iron + 1 coal)",
    "Make Wood Sword": "craft a wood sword (need: adjacent table, 1 wood)",
    "Make Stone Sword": "craft a stone sword (need: adjacent table, 1 wood + 1 stone)",
    "Make Iron Sword": "craft an iron sword (need: adjacent table + furnace, 1 wood + 1 stone + 1 iron + 1 coal)",
    "Rest": "recover health gradually (requires food, drink, and energy > 0 to heal; ends when health is full or food/drink runs out)",
    "Descend": "go down to the next dungeon level (must be on a ladder down)",
    "Ascend": "go up to the previous level (must be on a ladder up)",
    "Make Diamond Pickaxe": "craft a diamond pickaxe (need: adjacent epic forge, 1 wood + 3 diamond)",
    "Make Diamond Sword": "craft a diamond sword (need: adjacent epic forge, 1 wood + 2 diamond)",
    "Make Iron Armour": "craft iron armour (need: adjacent table + furnace, 3 iron + 3 coal)",
    "Make Diamond Armour": "craft diamond armour (need: adjacent epic forge, 3 diamond)",
    "Shoot Arrow": "shoot an arrow at the creature you face (need: bow + arrows)",
    "Make Arrow": "craft arrows (need: adjacent table, 1 wood + 1 stone; yields 2 arrows)",
    "Cast Spell": "cast a learned spell at the tile you face (costs mana)",
    "Place Torch": "place a torch on the tile you face to light dark areas",
    "Drink Potion Red": "drink a red potion",
    "Drink Potion Green": "drink a green potion",
    "Drink Potion Blue": "drink a blue potion",
    "Drink Potion Pink": "drink a pink potion",
    "Drink Potion Cyan": "drink a cyan potion",
    "Drink Potion Yellow": "drink a yellow potion",
    "Read Book": "read a book to learn a spell",
    "Enchant Sword": "enchant your sword (need: adjacent enchantment table + 1 gem + 9 mana)",
    "Enchant Armour": "enchant your armour (need: adjacent enchantment table + 1 gem + 9 mana)",
    "Make Torch": "craft torches (need: adjacent table, 1 wood + 1 coal; yields 4 torches)",
    "Level Up Dexterity": "spend XP to increase dexterity",
    "Level Up Strength": "spend XP to increase strength",
    "Level Up Intelligence": "spend XP to increase intelligence",
    "Enchant Bow": "enchant your bow (need: adjacent enchantment table + 1 gem + 9 mana)",
    "Request Food": "request food from teammates",
    "Request Drink": "request drink from teammates",
    "Request Wood": "request wood from teammates",
    "Request Stone": "request stone from teammates",
    "Request Iron": "request iron from teammates",
    "Request Coal": "request coal from teammates",
    "Request Diamond": "request diamond from teammates",
    "Request Ruby": "request ruby from teammates",
    "Request Sapphire": "request sapphire from teammates",
    "Give": "give requested resources to a specific teammate — no adjacency required, works at any distance (format: Give to Agent X). Only appears in your available actions when a teammate has an active Request.",
    "Build Shelter": f"build an epic shelter at a construction site (needs {SHELTER_COST_WOOD} wood + {SHELTER_COST_STONE} stone; effect: +50% energy regeneration while resting for all agents)",
    "Build Forge": f"build an epic forge at a construction site (needs {FORGE_COST_STONE} stone + {FORGE_COST_IRON} iron + {FORGE_COST_COAL} coal; effect: enables diamond gear crafting at this location)",
    "Build Beacon": f"build an epic beacon at a construction site (needs {BEACON_COST_IRON} iron + {BEACON_COST_COAL} coal; effect: expands the lit area on this level)",
}

# Mob names indexed by type_id (matching FLOOR_MOB_MAPPING indices)
MELEE_MOB_NAMES = [
    "zombie",
    "gnome_warrior",
    "orc_soldier",
    "lizard",
    "knight",
    "troll",
    "pigman",
    "frost_troll",
]
RANGED_MOB_NAMES = [
    "skeleton",
    "gnome_archer",
    "orc_mage",
    "kobold",
    "knight_archer",
    "deep_thing",
    "fire_elemental",
    "ice_elemental",
]
PASSIVE_MOB_NAMES = ["cow", "bat", "snail", "buffalo", "large_cow"]
PROJECTILE_NAMES = [
    "arrow",
    "dagger",
    "fireball",
    "iceball",
    "arrow2",
    "slimeball",
    "fireball2",
    "iceball2",
]

# Direction names matching DIRECTIONS: 0=noop, 1=LEFT(west), 2=RIGHT(east), 3=UP(north), 4=DOWN(south)
DIRECTION_NAMES = {0: "none", 1: "west", 2: "east", 3: "north", 4: "south"}

# Item names for item_map (ItemType enum)
ITEM_NAMES = {
    ItemType.TORCH.value: "placed torch",
    ItemType.LADDER_DOWN.value: "ladder down",
    ItemType.LADDER_UP.value: "ladder up",
    ItemType.LADDER_DOWN_BLOCKED.value: "ladder down (blocked — clear more monsters)",
}


# ============================================================================
# Achievement List
# ============================================================================


def _build_achievement_list():
    """Build achievement list from the Achievement enum (no index numbers)."""
    return [ach.name.replace("_", " ").title() for ach in Achievement]


ALL_ACHIEVEMENTS = _build_achievement_list()

# Minimum dungeon level at which each achievement is shown in the prompt.
# Mirrors _ON_OVERWORLD in alem_coop/envs/common.py — keep in sync if new
# achievements are added. Uses enum names (not .value) so renumbering is safe.

# Level 0: achievable on the overworld — mirrors _ON_OVERWORLD exactly.
# Also includes ENTER_GNOMISH_MINES so agents know to descend (shown before they get there).
_OVERWORLD_ACHIEVEMENTS = frozenset(
    {
        # Resources (all except sapphire/ruby which are dungeon-only)
        Achievement.COLLECT_WOOD,
        Achievement.COLLECT_STONE,
        Achievement.COLLECT_COAL,
        Achievement.COLLECT_IRON,
        Achievement.COLLECT_DIAMOND,
        Achievement.COLLECT_SAPLING,
        Achievement.COLLECT_DRINK,
        Achievement.COLLECT_FOOD,
        # Crafting — diamond gear via Epic Forge is buildable on the overworld
        Achievement.MAKE_WOOD_PICKAXE,
        Achievement.MAKE_WOOD_SWORD,
        Achievement.MAKE_STONE_PICKAXE,
        Achievement.MAKE_STONE_SWORD,
        Achievement.MAKE_IRON_PICKAXE,
        Achievement.MAKE_IRON_SWORD,
        Achievement.MAKE_DIAMOND_PICKAXE,
        Achievement.MAKE_DIAMOND_SWORD,
        Achievement.MAKE_IRON_ARMOUR,
        Achievement.MAKE_DIAMOND_ARMOUR,
        Achievement.MAKE_ARROW,
        Achievement.MAKE_TORCH,
        # Placement
        Achievement.PLACE_TABLE,
        Achievement.PLACE_STONE,
        Achievement.PLACE_FURNACE,
        Achievement.PLACE_PLANT,
        Achievement.PLACE_TORCH,
        # Survival
        Achievement.EAT_COW,
        Achievement.EAT_PLANT,
        Achievement.WAKE_UP,
        # Combat — zombies and skeletons spawn on overworld
        Achievement.DEFEAT_ZOMBIE,
        Achievement.DEFEAT_SKELETON,
        # Enter next level — shown one level early so agents know to aim for it
        Achievement.ENTER_GNOMISH_MINES,
        # Coordination — all of these are achievable on the overworld
        Achievement.COORD_2_AGENTS_SOFT,
        Achievement.COORD_2_AGENTS_HARD,
        Achievement.COORD_3_AGENTS_SOFT,
        Achievement.COORD_3_AGENTS_HARD,
        Achievement.HANDOVER_COMPLETE,
        Achievement.COORD_MINE_HANDOVER,
        Achievement.COORD_BUILD_SHELTER,
        Achievement.COORD_BUILD_FORGE,
        Achievement.COORD_BUILD_BEACON,
        Achievement.COORD_ELITE_MELEE_KILL,
        Achievement.COORD_ELITE_RANGED_KILL,
        Achievement.COORD_LARGE_PASSIVE_KILL,
        Achievement.COORD_MINE_STONE_SOFT,
        Achievement.COORD_MINE_STONE_HARD,
        Achievement.COORD_MINE_COAL_SOFT,
        Achievement.COORD_MINE_COAL_HARD,
        Achievement.COORD_MINE_IRON_SOFT,
        Achievement.COORD_MINE_IRON_HARD,
        Achievement.COORD_MINE_DIAMOND_SOFT,
        Achievement.COORD_MINE_DIAMOND_HARD,
        Achievement.COORD_DIAMOND_PICKAXE,
        Achievement.COORD_DIAMOND_SWORD,
        Achievement.COORD_DIAMOND_ARMOUR,
    }
)

# Level 1 (Gnomish Mines): gems, gnome/orc mobs, chests, bows, potions.
# Also includes ENTER_DUNGEON and ENTER_SEWERS so agents know to keep descending.
_LEVEL1_ACHIEVEMENTS = frozenset(
    {
        Achievement.COLLECT_SAPPHIRE,
        Achievement.COLLECT_RUBY,
        Achievement.COORD_MINE_SAPPHIRE_SOFT,
        Achievement.COORD_MINE_SAPPHIRE_HARD,
        Achievement.COORD_MINE_RUBY_SOFT,
        Achievement.COORD_MINE_RUBY_HARD,
        Achievement.ENTER_DUNGEON,
        Achievement.ENTER_SEWERS,  # shown one level early
        Achievement.DEFEAT_GNOME_WARRIOR,
        Achievement.DEFEAT_GNOME_ARCHER,
        Achievement.DEFEAT_ORC_SOLIDER,
        Achievement.DEFEAT_ORC_MAGE,
        Achievement.EAT_BAT,
        Achievement.EAT_SNAIL,
        Achievement.FIND_BOW,
        Achievement.FIRE_BOW,
        Achievement.OPEN_CHEST,
        Achievement.DRINK_POTION,
    }
)

# Level 3 (Sewers): lizards, kobolds, knights, spells, enchantments.
# ENTER_VAULT and ENTER_TROLL_MINES shown here so agents aim deeper.
_LEVEL3_ACHIEVEMENTS = frozenset(
    {
        Achievement.ENTER_VAULT,
        Achievement.ENTER_TROLL_MINES,  # shown one level early
        Achievement.DEFEAT_LIZARD,
        Achievement.DEFEAT_KOBOLD,
        Achievement.DEFEAT_KNIGHT,
        Achievement.DEFEAT_ARCHER,
        Achievement.DEFEAT_TROLL,
        Achievement.DEFEAT_DEEP_THING,
        Achievement.LEARN_SPELL,
        Achievement.CAST_SPELL,
        Achievement.ENCHANT_SWORD,
        Achievement.ENCHANT_ARMOUR,
    }
)

# Level 5 (Troll Mines / Realms): pigmen, elementals, necromancer.
_LEVEL5_ACHIEVEMENTS = frozenset(
    {
        Achievement.ENTER_FIRE_REALM,
        Achievement.ENTER_ICE_REALM,
        Achievement.ENTER_GRAVEYARD,
        Achievement.DEFEAT_PIGMAN,
        Achievement.DEFEAT_FIRE_ELEMENTAL,
        Achievement.DEFEAT_FROST_TROLL,
        Achievement.DEFEAT_ICE_ELEMENTAL,
        Achievement.DAMAGE_NECROMANCER,
        Achievement.DEFEAT_NECROMANCER,
    }
)


def _achievement_min_level(ach):
    """Return the minimum dungeon level at which this achievement is disclosed."""
    if ach in _OVERWORLD_ACHIEVEMENTS:
        return 0
    elif ach in _LEVEL1_ACHIEVEMENTS:
        return 1
    elif ach in _LEVEL3_ACHIEVEMENTS:
        return 3
    elif ach in _LEVEL5_ACHIEVEMENTS:
        return 5
    return 99  # Unknown achievement — don't show


# Separate core vs coordination achievements (no index numbers — enum values can change)
def _ach_name(ach):
    """Convert an Achievement enum member to a display name (e.g. COLLECT_WOOD → 'Collect Wood')."""
    return ach.name.replace("_", " ").title()


CORE_ACHIEVEMENTS = [
    _ach_name(ach)
    for ach in Achievement
    if not (ach.name.startswith("COORD_") or ach.name == "HANDOVER_COMPLETE")
]
COORD_ACHIEVEMENTS = [
    _ach_name(ach)
    for ach in Achievement
    if ach.name.startswith("COORD_") or ach.name == "HANDOVER_COMPLETE"
]


def _achievements_for_level(current_level, coordination_enabled):
    """Return achievement strings filtered to those disclosed at current_level."""
    entries = []
    for ach in Achievement:
        if _achievement_min_level(ach) > current_level:
            continue
        is_coord = ach.name.startswith("COORD_") or ach.name == "HANDOVER_COMPLETE"
        if is_coord and not coordination_enabled:
            continue
        entries.append(_ach_name(ach))
    return entries


# ============================================================================
# Helper Functions
# ============================================================================


def _absolute_direction(dr, dc):
    """Convert row/col delta to absolute cardinal direction parts."""
    parts = []
    if dr < 0:
        parts.append("north")
    elif dr > 0:
        parts.append("south")
    if dc < 0:
        parts.append("west")
    elif dc > 0:
        parts.append("east")
    return parts


# Egocentric rotation: map absolute (dr, dc) to agent-relative (dr', dc')
# based on facing direction. Agent's facing becomes "forward" (+row direction).
#
# Facing direction vectors: 1=west(0,-1), 2=east(0,+1), 3=north(-1,0), 4=south(+1,0)
# For each facing, define the rotation from absolute -> egocentric:
#   forward = facing vector, right = 90° clockwise from forward
_EGOCENTRIC_LABELS = {
    # (dr, dc) -> label, for each relative direction
    (-1, 0): "ahead",
    (1, 0): "behind",
    (0, -1): "to your left",
    (0, 1): "to your right",
    (-1, -1): "ahead-left",
    (-1, 1): "ahead-right",
    (1, -1): "behind-left",
    (1, 1): "behind-right",
}


def _rotate_to_egocentric(dr, dc, facing):
    """Rotate absolute (dr, dc) into agent-relative frame based on facing direction.

    Returns (ego_dr, ego_dc) where -row = ahead, +row = behind, -col = left, +col = right.
    """
    # Facing vectors: 1=west(0,-1), 2=east(0,+1), 3=north(-1,0), 4=south(+1,0)
    if facing == 3:  # north: identity (north=ahead, east=right)
        return dr, dc
    elif facing == 4:  # south: rotate 180°
        return -dr, -dc
    elif facing == 2:  # east: (north->left, east->ahead, south->right, west->behind)
        return -dc, dr
    elif facing == 1:  # west: (north->right, west->ahead, south->left, east->behind)
        return dc, -dr
    else:  # noop/unknown: fall back to identity
        return dr, dc


def _egocentric_direction(dr, dc, facing):
    """Convert absolute (dr, dc) to egocentric direction parts."""
    ego_dr, ego_dc = _rotate_to_egocentric(dr, dc, facing)
    parts = []
    if ego_dr < 0:
        parts.append("ahead")
    elif ego_dr > 0:
        parts.append("behind")
    if ego_dc < 0:
        parts.append("left")
    elif ego_dc > 0:
        parts.append("right")
    return parts


def describe_loc_precise(ref, P, facing=None):
    """Describe the location of P relative to ref with precise distances.

    Args:
        ref: Reference position [row, col] in array coordinates.
        P: Target position [row, col] in array coordinates.
        facing: If provided (1-4), use egocentric directions (ahead/behind/left/right).
                If None, use absolute cardinal directions (north/south/east/west).

    Returns:
        Human-readable per-axis direction and distance.
    """
    # Array coordinates: row increases southward, col increases eastward
    dr = int(P[0] - ref[0])  # row delta: positive = south
    dc = int(P[1] - ref[1])  # col delta: positive = east

    if dr == 0 and dc == 0:
        return "at your location"

    if facing is not None:
        parts = _egocentric_direction(dr, dc, facing)
    else:
        parts = _absolute_direction(dr, dc)

    desc = []
    # Report per-axis distances
    if dr != 0:
        label = parts[0] if parts else "away"
        desc.append(f"{abs(dr)} step{'s' if abs(dr) > 1 else ''} {label}")
    if dc != 0:
        label = parts[-1] if len(parts) > (1 if dr != 0 else 0) else (parts[0] if parts else "away")
        desc.append(f"{abs(dc)} step{'s' if abs(dc) > 1 else ''} {label}")

    return " and ".join(desc) if desc else "at your location"


def describe_loc_old(ref, P, facing=None):
    """Describe relative location using direction and total distance.

    For diagonal positions (dr != 0 and dc != 0) this always uses per-axis
    format ("N steps X and M steps Y") regardless of precise_location, because
    a combined label like "5 steps ahead-right" is ambiguous about the split.
    Pure axis-aligned positions use the compact "D steps direction" form.

    Args:
        ref: Reference position [row, col] in array coordinates.
        P: Target position [row, col] in array coordinates.
        facing: If provided (1-4), use egocentric directions.
                If None, use absolute cardinal directions.

    Returns:
        Human-readable relative direction and Manhattan distance.
    """
    # Array coordinates: row increases southward, col increases eastward
    dr = int(P[0] - ref[0])  # row delta: positive = south
    dc = int(P[1] - ref[1])  # col delta: positive = east

    if dr == 0 and dc == 0:
        return "at your location"

    if facing is not None:
        parts = _egocentric_direction(dr, dc, facing)
    else:
        parts = _absolute_direction(dr, dc)

    # Egocentric diagonal: split per axis — "2 steps ahead and 3 steps right"
    # is unambiguous; "5 steps ahead-right" hides the split.
    # Cardinal diagonal: keep compact — "5 steps north-east" is clear enough.
    if facing is not None and dr != 0 and dc != 0:
        label_r = parts[0] if parts else "away"
        label_c = parts[-1] if len(parts) > 1 else (parts[0] if parts else "away")
        desc_r = f"{abs(dr)} step{'s' if abs(dr) > 1 else ''} {label_r}"
        desc_c = f"{abs(dc)} step{'s' if abs(dc) > 1 else ''} {label_c}"
        return f"{desc_r} and {desc_c}"

    distance = abs(dr) + abs(dc)
    direction = "-".join(parts) if parts else "away"
    return f"{distance} step{'s' if distance > 1 else ''} {direction}"


def get_edge_items(semantic, item_idx):
    """Get an edge mask for a specific semantic item type.

    Args:
        semantic: Semantic item-identifier map.
        item_idx: Item identifier whose region edge should be detected.

    Returns:
        Boolean mask containing only boundary pixels of the item region.
    """
    item_mask = semantic == item_idx
    not_item_mask = semantic != item_idx
    item_edge = ndimage.binary_dilation(not_item_mask) & item_mask
    return item_edge


# ============================================================================
# Instruction Prompt
# ============================================================================


def get_instruction_prompt(
    llm_mode=None,
    coordination_enabled=False,
    num_agents=3,
    agent_id=None,
    role=None,
    include_all_actions=True,
    progressive_disclosure=False,
    current_level=0,
    prompt_mode=None,
):
    """Build the system-prompt instruction block for an LLM agent.

    Args:
        llm_mode: Deprecated. Accepted for backward compatibility but ignored.
            Use prompt_mode instead.
        coordination_enabled: Whether coordination mechanics are active. Controls
            whether coordination hints appear in action descriptions and the
            Coordination section is included.
        num_agents: Number of agents in the game.
        agent_id: Index of this agent (used for role assignment in the intro).
        role: Role name string (e.g. "warrior"). Used in the intro if agent_id
            is also provided.
        include_all_actions: If True, include the full action list with
            per-action descriptions (<all_actions> block). If False, omit it —
            use this when show_affordances=True so agents see available actions
            per-turn instead. Note: has no effect in prompt_mode="general",
            which always includes the action list.
        progressive_disclosure: If True, gate late-game sections (chests,
            potions, enchantments, attributes, extra progression steps) and
            achievements by current_level to save tokens. If False, show
            everything upfront.
        current_level: Current dungeon level (0 = overworld). Only used when
            progressive_disclosure=True.
        prompt_mode: Controls how much structured game knowledge is included.
            - "general": Minimal baseline. Intro + full action list with
              descriptions + achievements. No game rules, no coordination info.
              include_all_actions is ignored (action list always shown).
            - "specific": Full structured game rules (How to play, survival
              stats, roles, resource chain, crafting, progression) + optional
              action list (respects include_all_actions) + achievements +
              late-game sections. No coordination info of any kind.
            - "specific_collaborative": Same as "specific" plus coordination
              section (Sync, Handover, Construction, Elite mobs, Revive, Epic
              forge) and coordination hints appended to relevant action
              descriptions (diamond crafting, Build actions). Also enables
              per-turn coordination cues in observations.
            Defaults to "specific_collaborative" if not provided.

    Returns:
        Complete system-prompt text for the requested mode and level.
    """

    valid_prompt_modes = {"general", "specific", "specific_collaborative"}
    if prompt_mode is None:
        prompt_mode = "specific_collaborative"
    if prompt_mode not in valid_prompt_modes:
        raise ValueError(
            f"Invalid prompt_mode '{prompt_mode}'. Expected one of: {sorted(valid_prompt_modes)}"
        )

    # --- Level thresholds for progressive disclosure ---
    # When progressive_disclosure is False, all sections are shown (level=99).
    lvl = current_level if progressive_disclosure else 99

    # Achievements are always included; filtered by level when disclosure is on.
    if progressive_disclosure:
        achievement_str = "\n".join(_achievements_for_level(current_level, coordination_enabled))
    elif coordination_enabled:
        achievement_str = "\n".join(CORE_ACHIEVEMENTS + COORD_ACHIEVEMENTS)
    else:
        achievement_str = "\n".join(CORE_ACHIEVEMENTS)

    # Append coordination hints to relevant action descriptions
    COORDINATION_HINTS = {
        "Make Diamond Pickaxe": "; also requires enough agents crafting this same item at the same epic forge on the same turn",
        "Make Diamond Sword": "; also requires enough agents crafting this same item at the same epic forge on the same turn",
        "Make Diamond Armour": "; also requires enough agents crafting this same item at the same epic forge on the same turn",
        "Build Shelter": ", requires coordination",
        "Build Forge": ", requires coordination",
        "Build Beacon": ", requires coordination",
    }

    def _action_desc(action):
        desc = ACTION_DICT[action]
        if (
            prompt_mode == "specific_collaborative"
            and coordination_enabled
            and action in COORDINATION_HINTS
        ):
            if desc.endswith(")"):
                desc = desc[:-1] + COORDINATION_HINTS[action] + ")"
            else:
                desc = desc + COORDINATION_HINTS[action]
        return desc

    action_strings = "\n".join(f"{action}: {_action_desc(action)}" for action in ACTIONS)

    # we add more details -- specifically you can coordinate
    if prompt_mode == "specific_collaborative":
        extra_coord_intro = ", while coordinating with teammates"
    else:
        extra_coord_intro = ""

    if agent_id is not None and role:
        intro = f"You are Agent {agent_id} ({role}) in a {num_agents}-agent cooperative survival game. Your goal is to gather resources, craft gear, fight monsters, and descend through 9 dungeon levels{extra_coord_intro}. You must survive — if your health reaches zero, you die, and if all agents die the game ends. Maximize the number of achievements while staying alive."

    else:
        intro = f"You are an agent in a {num_agents}-agent cooperative survival game. Your goal is to gather resources, craft gear, fight monsters, and descend through 9 dungeon levels{extra_coord_intro}. You must survive — if your health reaches zero, you die, and if all agents die the game ends. Maximize the number of achievements while staying alive."

    if prompt_mode == "general":
        # Minimal baseline: action descriptions are the only game knowledge.
        # include_all_actions is intentionally ignored — without game_rules,
        # the action list is the agent's sole source of mechanics information.
        return f"""{intro}

## Actions
Each turn, choose exactly one action. Your observation will list which actions are currently available.

{action_strings}

## Achievements
{achievement_str}

You must survive — if your health reaches zero you die, and if all agents die the game ends. Choose actions to maximize achievements while staying alive. Your observations show what you see, your inventory, teammates, and available actions."""

    _do_coord_note = (
        " For synchronous-style coordination, all required agents must stand next to the same target tile, face it, and act together."
        if prompt_mode == "specific_collaborative" and coordination_enabled
        else ""
    )
    _elite_coord_note = (
        " coordinating with teammates (multiple agents attacking together) makes them much easier to defeat."
        if prompt_mode == "specific_collaborative" and coordination_enabled
        else "."
    )
    core_mechanics = f"""
## How to play
- Each turn, choose exactly one action.
- **Movement** uses absolute directions: north, south, east, and west. Any move attempt changes your facing to that direction, even if the move is blocked and you stay in place. A move is blocked if the target tile is solid, including trees, stone, ore veins, walls, crafting stations, chests, and plants, or if it contains water, lava, a mob, or another player. If repeated move attempts do not change your position, that direction is blocked. You can also use a blocked move to turn in place, for example to face an adjacent tree.
- **Facing**: your facing direction is set by your last movement action and persists until you move again. **Do** always targets the tile in your current facing direction.
- **Do** is your main interaction: face a tile and use the **Do** action on exactly that tile to chop trees, mine ore, attack creatures, drink water, open chests, or revive a downed teammate. If the faced tile contains a downed teammate, Do revives them. If the faced tile contains a living teammate, Do targets that teammate instead and can cause friendly fire.{_do_coord_note}
- **Crafting**: stand next to (including diagonally) the required station and use the craft action; you do NOT need to face it. Diamond items always require an adjacent epic forge, not a table.
- **Placing**: face the target tile, then use the place action. Tables and furnaces need an empty non-solid tile that is not water or lava; stone can also be placed into water (costs 1 stone). Place Plant puts a sapling on the faced tile. Place Torch lights dark areas.
- **Ranged combat**: use Shoot Arrow while facing a creature (requires a bow + arrows). Bows are found in dungeon chests.
- **Elite mobs** are tougher and deal more damage;{_elite_coord_note}
- **Request/Give**: use Request [Resource] to broadcast a resource request to teammates for 10 turns; teammates can use Give to Agent X to transfer one unit of the requested resource directly — no adjacency required, works at any distance. Give only appears as an available action when a teammate has an active Request.
"""

    survival_stats = """
## Survival stats
Food, drink, and energy deplete gradually over time — roughly every 20-30 steps you lose 1 point of each (dexterity slows this rate). When food or drink reaches 0, your health starts dropping. When energy reaches 0, you automatically fall asleep and cannot act until energy is full. While sleeping, you take 2.5x damage from all sources. Mana does NOT decay — it is only spent by casting spells or enchanting. Mana slowly regenerates over time (faster while sleeping).
- **Sleep**: choose this voluntarily to recover energy at 2x the passive rate. Ends automatically when energy is full.
- **Rest**: choose this to recover health gradually. Requires food, drink, and energy all > 0; ends when health is full or a stat runs out."""

    resource_chain = """
## Resource chain
Trees → wood (no tool required) → Stone/Coal (needs wood pickaxe) → Iron (needs stone pickaxe) → Diamond (iron pickaxe) → Ruby/Sapphire (diamond pickaxe)"""

    # Diamond gear is craftable on the overworld via epic forge — always show.
    _diamond_coord_note = (
        " + enough agents crafting the same item there on the same turn"
        if prompt_mode == "specific_collaborative" and coordination_enabled
        else ""
    )
    crafting_recipes = f"""
## Crafting recipes
All recipes consume the listed materials.
Stations: Table (2 wood), Furnace (1 stone)
- Wood pickaxe/sword: table + 1 wood
- Stone pickaxe/sword: table + 1 wood + 1 stone
- Iron pickaxe/sword: table + furnace + 1 wood + 1 stone + 1 iron + 1 coal
- Iron armour: table + furnace + 3 iron + 3 coal
- Diamond pickaxe: epic forge + 1 wood + 3 diamond{_diamond_coord_note}
- Diamond sword: epic forge + 1 wood + 2 diamond{_diamond_coord_note}
- Diamond armour: epic forge + 3 diamond{_diamond_coord_note}
- Arrows: table + 1 wood + 1 stone (yields 2)
- Torch: table + 1 wood + 1 coal (yields 4)

Construction (at a construction site, face it and use Build action):
- Build Shelter: needs {SHELTER_COST_WOOD} wood + {SHELTER_COST_STONE} stone. Shelters result in +50% energy regeneration while resting (all agents).
- Build Forge: needs {FORGE_COST_STONE} stone + {FORGE_COST_IRON} iron + {FORGE_COST_COAL} coal. Creates an epic forge, which enables diamond gear crafting.
- Build Beacon: needs {BEACON_COST_IRON} iron + {BEACON_COST_COAL} coal. Expands the lit area on this level."""

    # --- Chests (level >= 1) ---
    chests = ""
    if lvl >= 1:
        chests = """
## Chests
Open chests with Do while facing them. Loot is role-biased and partially random:
- Any role (50% chance): 1-2 potions of a random color
- Miner (60% chance): 1-5 wood, 4-7 torches, or 1-3 ores (coal/iron/diamond/sapphire/ruby)
- Miner (20% chance): pickaxe upgrade to a random higher tier
- Warrior (50% chance): 4-8 arrows
- Warrior only, first chest on dungeon level 1: bow
- Any role, first chest on level 3 or 4 only: spell book"""

    # --- Spells (level >= 3: books only drop on levels 3-4) ---
    spells = ""
    if lvl >= 3:
        spells = """
## Spells
Learn spells by using Read Book when you have a book in your inventory. Each role learns a different spell:
- Fireball (miner, warrior): costs 2 mana; fires a projectile at the tile you are facing.
- Heal (forager): costs 6 mana; restores +2 health to yourself.
Use Cast Spell while facing the desired target tile."""

    # --- Potions (level >= 1: come from chests) ---
    potions = ""
    if lvl >= 1:
        potions = """
## Potions
Each potion action drinks one potion of that color, if you have one. The color-to-effect mapping is randomized at the start of each episode and fixed for its duration — the same color always has the same effect within one game.
- Possible effects: +8 health, -3 health, +8 mana, -3 mana, +8 energy, or -3 energy.
- Potion colors: red, green, blue, pink, cyan, yellow.
- The mapping is not shown to you directly. Observe the changes after each drink to learn which colors are safe."""

    # --- Boss fight (level >= 5: Necromancer on final level) ---
    boss = ""
    if lvl >= 5:
        boss = """
## Boss
The Necromancer on the final level is only vulnerable during specific windows (shown in your observation). Attack during vulnerable phases and survive the spawn waves between them."""

    # --- Enchantments (level >= 5: need gems + enchantment tables) ---
    enchantments = ""
    if lvl >= 5:
        enchantments = """
## Enchantments
Enchant Sword and Enchant Bow are warrior-restricted (not guaranteed for non-warriors). Enchant Armour works for all roles. Fire tables use ruby, ice tables use sapphire, and every enchantment costs 1 matching gem + 9 mana.
- Sword enchant: enchants your sword with the table's element (fire or ice).
- Bow enchant: enchants your bow with the table's element (fire or ice).
- Armour enchant: enchants one armour piece with the table's element; it targets an unenchanted piece first, otherwise it can replace a piece with the opposite element."""

    # --- Attributes (level >= 1: XP gained on descent) ---
    # Attributes shown from level 0: XP/Level Up actions are available as soon
    # as the agent descends, so agents need to know this from the start.
    attributes = """
## Attributes
Gain 1 XP each time you descend to a new floor. Spend XP with Level Up actions.
- **Strength**: max health = 8 + strength
- **Dexterity**: max food = 7 + 2*dexterity (+2 extra for foragers); max drink = same; max energy = 7 + 2*dexterity
- **Intelligence**: max mana = 6 + 3*intelligence; enchantment damage +5% per point above 1."""

    # --- Progression: gate later steps by level ---
    progression_steps = [
        "1. Gather wood → place a table → craft a wood pickaxe; craft a wood sword early if combat is likely.",
        "2. Mine stone and coal → place a furnace → craft iron tools and iron armour.",
        "3. To descend: stand on the `ladder_down` tile (visible in your observation when close) and use the Descend action. The ladder only becomes usable after enough monsters on that level have been killed. Only one agent needs to use Descend/Ascend — all teammates are teleported with them.",
    ]
    if lvl >= 1:
        progression_steps.append(
            "4. In dungeons, open chests for loot such as bows, potions, and spell books, and mine gems as your tools improve."
        )
    if lvl >= 3:
        progression_steps.append(
            "5. Enchant weapons, bows, or armour at enchantment tables by spending 9 mana and 1 gem; ruby gives fire, sapphire gives ice."
        )
    if lvl >= 5:
        _forge_note = (
            ", coordinating with teammates when required"
            if prompt_mode == "specific_collaborative"
            else ""
        )
        progression_steps.append(f"6. Craft diamond gear at an epic forge{_forge_note}.")
        progression_steps.append("7. Repeat across all dungeon levels until the final boss.")

    progression = "\n## Progression\n" + "\n".join(progression_steps)

    roles = """
## Roles
Role-restricted actions succeed with reduced probability for non-specialists. Depending on the difficulty configuration, non-specialist success rates are 10%, 40%, or 70%. Specialist success rate is always 100%.

- **Forager**: collecting water, saplings, eating passive mobs (e.g. cows/bats/snails). Also has 3x base food and drink capacity.
- **Miner**: crafting pickaxes/torches, placing stone.
- **Warrior**: crafting swords and arrows. Also deals 2x melee damage, and specializes in enchanting swords and bows.
- No role restriction: Place Table, Place Furnace, Wood Sword, Iron Armour, Diamond Armour"""

    coordination = ""
    if prompt_mode == "specific_collaborative" and coordination_enabled:
        coordination = """
## Coordination
Some tasks, tiles, creatures, and structures require multiple agents. When collaboration is possible the observation will include a short coordination hint. Follow the rules below to coordinate safely and efficiently.

- **Sync**: N agents must each stand on a different tile adjacent to the shared target, each facing it, and all choose the **Do** action on the same turn. Approach from different sides so every agent targets the shared tile directly. If a teammate is between you and the target, your **Do** will hit the teammate instead and can cause friendly fire.
- **Handover**: one agent starts the task with **Do** (pays the resources required), and another agent finishes it with **Do** within a small time window shown in the observation. If no agent completes it in time, the site resets and materials are refunded to the initiator. Follow the exact handover timing when specified.
- **Construction**: Construction sites (shelters, forges, beacons) may require either sync or handover. Always follow the coordination rule shown in the observation for that site.
- **Elite mobs**: stronger enemies may benefit from or require coordinated attacks. Attack from different sides, avoid standing between a teammate and the mob, and avoid blocking another agent's attack.
- **Revive**: Face a downed teammate and use **Do** to revive them.
- **Epic forge / Diamond crafting**: Diamond-tier items require multiple agents to craft simultaneously at an adjacent epic forge. All required agents must choose the crafting action for the same item on the same turn while adjacent to the forge."""

    actions_section = ""
    if include_all_actions:
        actions_section = f"""
<all_actions>
## Actions
Each turn, choose exactly one action. Your observation will list which actions are currently available.

{action_strings}
</all_actions>"""

    achievements_section = f"""
<achievements>
## Achievements
{achievement_str}
</achievements>"""

    # Assemble late-game reference from the level-gated sections
    late_game_parts = [s for s in [chests, spells, potions, enchantments, boss] if s]
    late_game_section = ""
    if late_game_parts:
        late_game_section = "\n<late_game>" + "".join(late_game_parts) + "\n</late_game>"

    instruction_prompt = f"""{intro}
<game_rules>
{core_mechanics}
{survival_stats}
{roles}
{coordination}
{resource_chain}
{crafting_recipes}
{attributes}
{progression}
</game_rules>
{actions_section}
{achievements_section}
{late_game_section}"""

    return instruction_prompt


# ============================================================================
# Language Wrapper
# ============================================================================


class AlemLanguageWrapper:
    """Language wrapper for multi-agent Alem-Coop environment.

    Converts symbolic observations to text descriptions for LLM control.

    Args:
        env: Alem environment instance.
        env_params: EnvParams for the environment.
        llm_mode: Deprecated. Kept for backward compatibility; has no effect
            on the prompt. Use prompt_mode to control prompt content.
        prompt_mode: Controls how much game knowledge is in the system prompt
            and which coordination info appears in per-turn observations.
            - "general": action list + achievements only (no game rules).
            - "specific": full game rules, no coordination info.
            - "specific_collaborative": full game rules + coordination section
              and per-turn coordination cues in observations.
        show_affordances: If True, append the list of legal action names to
            each per-turn observation (the <all_actions> block in the system
            prompt becomes optional via include_all_actions).
        exact_coordinates: If True, replace relative location text with
            absolute coordinates "(x=C, y=R)" for all reported locations.
        use_ascii: If True, render the local view as an ASCII grid instead of
            the "You see:" text list.
    """

    def __init__(
        self,
        env,
        env_params,
        llm_mode="easy",
        prompt_mode="specific_collaborative",
        max_episode_steps=10000,
        unique_items=True,
        precise_location=False,
        exact_coordinates=False,
        egocentric=False,
        skip_items=None,
        edge_only_items=None,
        render_pixel_size=None,
        render_downscale=1,
        debug=False,
        show_affordances=False,
        use_ascii=False,
    ):
        """Configure symbolic-to-language conversion and rendering options.

        Args:
            env: ALEM environment instance to wrap.
            env_params: Dynamic parameters used by the environment.
            llm_mode: Deprecated compatibility setting.
            prompt_mode: Prompt detail and coordination-disclosure mode.
            max_episode_steps: Step count displayed in language observations.
            unique_items: Whether repeated visible items should be collapsed.
            precise_location: Whether descriptions include axis distances.
            exact_coordinates: Whether descriptions append absolute coordinates.
            egocentric: Whether directions are relative to player facing.
            skip_items: Block names omitted from visible-object descriptions.
            edge_only_items: Block names reported only at region edges.
            render_pixel_size: Tile size used by optional debug rendering.
            render_downscale: Integer scale reduction for debug frames.
            debug: Whether to retain textures and verbose render data.
            show_affordances: Whether observations list currently legal actions.
            use_ascii: Whether local views use the ASCII renderer.
        """
        self.env = env
        self.env_params = env_params
        self.static_env_params = env.static_env_params
        self.max_steps = max_episode_steps
        self.language_action_space = ACTIONS
        self.llm_mode = llm_mode
        self.prompt_mode = prompt_mode
        self.show_affordances = show_affordances
        self.debug = debug
        self.use_ascii = use_ascii

        # Get number of agents from the environment
        self.num_agents = env.static_env_params.player_count

        # Track stats per agent
        self.score_trackers = [0.0] * self.num_agents
        self.achievements = [None] * self.num_agents
        self.failed_candidates = [[] for _ in range(self.num_agents)]

        # Description settings
        self.unique_items = unique_items
        self.precise_location = precise_location
        self.exact_coordinates = exact_coordinates
        self.egocentric = egocentric  # True: ahead/behind/left/right, False: north/south/east/west
        self.skip_items = skip_items if skip_items is not None else ["grass", "sand", "path"]
        self.edge_only_items = edge_only_items if edge_only_items is not None else []

        # Create block type name mapping
        self.block_id_to_name = {block.value: block.name.lower() for block in BlockType}

        # Specialization names
        self.spec_names = {s.value: s.name.lower() for s in Specialization}

        # Player-specific textures for pixel rendering (only loaded when debug=True)
        self.render_pixel_size = (
            render_pixel_size if render_pixel_size is not None else BLOCK_PIXEL_SIZE_AGENT
        )
        self.render_downscale = render_downscale
        if self.debug:
            texture_set = TEXTURES[self.render_pixel_size]
            self.player_specific_textures = load_player_specific_textures(
                texture_set, self.num_agents
            )
        else:
            self.player_specific_textures = None

    def reset(self, rng):
        """Reset the environment and describe every initial observation.

        Args:
            rng: JAX random key used to reset the environment.

        Returns:
            Text observations, initial state, and advanced random key.
        """
        rng, _rng = jax.random.split(rng)
        obs, state = self.env.reset(_rng)

        # Reset stats for all agents
        self.score_trackers = [0.0] * self.num_agents
        self.achievements = [None] * self.num_agents
        self.failed_candidates = [[] for _ in range(self.num_agents)]

        # Get observations for all agents
        text_obs_list = [
            self.process_obs(obs, state, agent_idx) for agent_idx in range(self.num_agents)
        ]

        return text_obs_list, state, rng

    def get_action_index(self, action_name, agent_idx=0):
        """Convert action name to index, tracking failed attempts.

        Uses the same multi-strategy parser as robust agents to avoid drift
        between agent-side and env-side action matching.

        Args:
            action_name: Candidate action text or tagged model output.
            agent_idx: Index used for target-specific GIVE slots and failures.

        Returns:
            Integer action index, defaulting to NOOP for invalid text.
        """
        if not action_name or not action_name.strip():
            self.failed_candidates[agent_idx].append(action_name)
            return 0

        # Clean: strip whitespace, quotes, trailing punctuation
        text = action_name.strip().strip("\"'`").strip()
        text = re.sub(r"[.!?,;:]+$", "", text).strip()

        parsed = extract_action_multistrategy(text, ACTIONS)
        if parsed:
            # Targeted GIVE form (e.g. "Give to Agent 2") maps to agent-specific slots.
            m = re.search(r"^\s*give\s+(?:to\s+)?agent\s+(\d+)\s*$", parsed, re.IGNORECASE)
            if m:
                target_idx = int(m.group(1))
                if 0 <= target_idx < self.num_agents and target_idx != agent_idx:
                    # Slot mapping: target = k if k < giver else k+1  =>  k = target if target < giver else target-1
                    give_slot = target_idx if target_idx < agent_idx else target_idx - 1
                    return Action.GIVE.value + give_slot
                self.failed_candidates[agent_idx].append(action_name)
                return 0

            if parsed in ACTIONS:
                return ACTIONS.index(parsed)

        # Track failed action for this agent
        self.failed_candidates[agent_idx].append(action_name)
        return 0  # Default to Noop

    def step(self, state, actions, rng):
        """Take a step in the environment with actions for all agents.

        Args:
            state: Current environment state
            actions: List of action strings/indices for all agents
            rng: JAX random key

        Returns:
            text_obs_list, new_state, rewards, dones, info, rng
        """
        # Convert actions to dict format for JaxMARL MultiAgentEnv
        actions_dict = {}
        canonical_actions = []
        action_indices = []
        for agent_idx, action in enumerate(actions):
            if isinstance(action, str):
                action_idx = self.get_action_index(action, agent_idx)
                canonical_actions.append(action if action_idx is not None else "Noop")
            else:
                action_idx = int(action)
                canonical_actions.append(
                    ACTIONS[action_idx] if 0 <= action_idx < len(ACTIONS) else "Noop"
                )
            action_indices.append(action_idx)
            actions_dict[f"agent_{agent_idx}"] = jnp.int32(action_idx)

        # Check action mask BEFORE stepping to detect actions that will have no effect
        action_mask = compute_action_mask(state, self.env_params, self.static_env_params)
        action_failed = []
        for agent_idx in range(self.num_agents):
            player_mask = np.array(action_mask[agent_idx])
            idx = action_indices[agent_idx]
            failed = idx < len(player_mask) and not player_mask[idx]
            action_failed.append(failed)

        # Take step in environment
        rng, _rng = jax.random.split(rng)
        obs, new_state, rewards_dict, dones_dict, info = self.env.step_env(
            _rng, state, actions_dict
        )

        # Convert rewards and dones from dicts to lists
        rewards_list = [float(rewards_dict[f"agent_{i}"]) for i in range(self.num_agents)]
        dones_list = [bool(dones_dict[f"agent_{i}"]) for i in range(self.num_agents)]

        # Capture achievements before update to detect newly earned ones
        prev_achievements = [
            dict(self.achievements[i]) if self.achievements[i] else {}
            for i in range(self.num_agents)
        ]

        # Update progress for all agents
        for agent_idx in range(self.num_agents):
            self.update_progress(new_state, agent_idx)

        # Compute newly earned achievements per agent
        new_achievements_list = []
        for agent_idx in range(self.num_agents):
            prev = prev_achievements[agent_idx]
            curr = self.achievements[agent_idx] or {}
            newly_earned = [
                name for name, val in curr.items() if val > 0 and prev.get(name, 0) == 0
            ]
            new_achievements_list.append(newly_earned)

        # Get observations for all agents
        text_obs_list = [
            self.process_obs(
                obs,
                new_state,
                agent_idx,
                reward=rewards_list[agent_idx],
                new_achievements=new_achievements_list[agent_idx],
                last_action=canonical_actions[agent_idx],
                action_failed=action_failed[agent_idx],
            )
            for agent_idx in range(self.num_agents)
        ]

        return text_obs_list, new_state, rewards_list, dones_list, info, rng

    def process_obs(
        self,
        obs,
        state,
        player_idx,
        reward=None,
        new_achievements=None,
        last_action=None,
        action_failed=False,
    ):
        """Convert one player's observation and feedback into model input.

        Args:
            obs: Raw per-agent observation mapping.
            state: Current environment state.
            player_idx: Player whose observation should be described.
            reward: Optional reward from the preceding transition.
            new_achievements: Newly earned achievement names.
            last_action: Canonical action attempted on the preceding transition.
            action_failed: Whether the attempted action had no effect.

        Returns:
            Text contexts, optional image, raw observation, and inactivity flag.
        """
        long_term_context, short_term_context = self.describe_frame(
            state,
            player_idx,
            reward=reward,
            new_achievements=new_achievements,
            last_action=last_action,
            action_failed=action_failed,
        )

        if self.show_affordances:
            affordances = self.get_affordances(state, player_idx)
            if affordances:
                short_term_context = short_term_context + "\n\n" + affordances

        img = None
        if self.debug:
            pixels = render_alem_pixels(
                state,
                block_pixel_size=self.render_pixel_size,
                static_params=self.env.static_env_params,
                player_specific_textures=self.player_specific_textures,
            )
            pixels_np = np.array(pixels[player_idx])
            if self.render_downscale > 1:
                pixels_np = pixels_np[:: self.render_downscale, :: self.render_downscale, :]
            pixels_np = pixels_np.astype(np.uint8)
            img = Image.fromarray(pixels_np)

        # Mark dead, sleeping, and resting agents as inactive — all three are
        # converted to NOOP by alem_step, so repeated observations add no
        # decision value. Suppressing scratchpad storage preserves the last
        # pre-inactive plan so it's ready on revival/wake-up.
        # Resting is short-lived (a few turns), sleeping can last 64+ steps.
        is_inactive = bool(
            state.player_health[player_idx] <= 0
            or state.is_sleeping[player_idx]
            or state.is_resting[player_idx]
        )

        return {
            "text": {
                "long_term_context": long_term_context,
                "short_term_context": short_term_context,
            },
            "image": img,
            "obs": obs,
            "is_inactive": is_inactive,
        }

    def describe_step_feedback(
        self, reward, new_achievements, last_action=None, action_failed=False
    ):
        """Describe action outcome, reward, and newly earned achievements.

        Args:
            reward: Optional scalar reward from the preceding transition.
            new_achievements: Newly earned achievement names.
            last_action: Canonical action attempted on the preceding transition.
            action_failed: Whether the attempted action had no effect.

        Returns:
            Feedback text, or an empty string when no feedback exists.
        """
        if reward is None and not new_achievements and last_action is None:
            return ""
        parts = []
        if last_action is not None:
            if action_failed:
                parts.append(
                    f"Last action: {last_action} (had no effect — not currently available)"
                )
            else:
                parts.append(f"Last action: {last_action}")
        if reward is not None:
            parts.append(f"Reward: {reward:+.3f}")
        if new_achievements:
            achievement_strs = ", ".join(a.replace("_", " ").title() for a in new_achievements)
            parts.append(f"Achievements unlocked: {achievement_strs}")
        return "\n".join(parts)

    def describe_frame(
        self,
        state,
        player_idx,
        reward=None,
        new_achievements=None,
        last_action=None,
        action_failed=False,
    ):
        """Build long- and short-term language context for one player.

        Args:
            state: Current environment state.
            player_idx: Player whose perspective should be described.
            reward: Optional reward from the preceding transition.
            new_achievements: Newly earned achievement names.
            last_action: Canonical action attempted on the preceding transition.
            action_failed: Whether the attempted action had no effect.

        Returns:
            Long-term world context and short-term inventory context.
        """
        try:
            is_dead = bool(state.player_health[player_idx] <= 0)
            is_sleeping = bool(state.is_sleeping[player_idx])
            is_resting = bool(state.is_resting[player_idx])

            # Step feedback (reward + new achievements + action failure)
            feedback = self.describe_step_feedback(
                reward, new_achievements, last_action=last_action, action_failed=action_failed
            )

            # Inventory (needed for all cases)
            inventory_desc = self.describe_inventory(state, player_idx)

            if is_dead or is_sleeping or is_resting:
                # COMPRESSED OBSERVATION FOR INACTIVE STATES
                # Strips away map, mobs, and coordination cues to save tokens.
                result = ""
                if feedback:
                    result += feedback + "\n\n"

                # Status header
                timestep = int(state.timestep)
                max_timesteps = int(self.env_params.max_timesteps)
                result += (
                    f"Step: {timestep}/{max_timesteps} ({max_timesteps - timestep} remaining)\n"
                )
                if self.exact_coordinates:
                    result += f"Position: {self._xy_coord_str(state.player_position[player_idx])}\n"

                if is_dead:
                    result += "Status: dead (you cannot act; you can only use communication to coordinate with teammates)\n\n"
                elif is_sleeping:
                    energy = int(state.player_energy[player_idx])
                    dex = int(state.player_dexterity[player_idx])
                    max_energy = 7 + 2 * dex
                    result += f"Status: sleeping (energy {energy}/{max_energy} - cannot act until energy is full)\n\n"
                elif is_resting:
                    result += "Status: resting (recovering health and mana)\n\n"

                # Teammate info is still critical for coordination
                result += self.describe_teammates(state, player_idx)

                return result.strip(), inventory_desc

            # --- Full Observation for Active Agents ---
            result = ""

            if feedback:
                result += feedback + "\n\n"

            # Status (sleeping/dead/active)
            result += self.describe_status(state, player_idx)
            if result:
                result += "\n\n"

            # Level info (light, monsters killed, boss)
            level_info = self.describe_level_info(state)
            if level_info:
                result += level_info + "\n\n"

            # Environment description (text list or ASCII grid)
            if self.use_ascii:
                result += self.render_ascii_map(state, player_idx)
            else:
                result += self.describe_env(state, player_idx)

            # Mobs and projectiles
            mobs_desc = self.describe_mobs(state, player_idx)
            if mobs_desc:
                result += "\n\n" + mobs_desc

            # Coordination cues are shown only in collaborative prompt mode.
            if (
                self.env_params.coordination_enabled
                and self.prompt_mode == "specific_collaborative"
            ):
                coord_desc = self.describe_coordination_cues(state, player_idx)
                if coord_desc:
                    result += "\n\n" + coord_desc

            # Construction hints are shown only in collaborative prompt mode.
            if (
                self.env_params.construction_enabled
                and self.prompt_mode == "specific_collaborative"
            ):
                constr_desc = self.describe_construction(state, player_idx)
                if constr_desc:
                    result += "\n\n" + constr_desc

            result += "\n\n"

            # Teammate info
            result += self.describe_teammates(state, player_idx)

            return result.strip(), inventory_desc
        except Exception as e:
            logger.error(f"describe_frame failed for agent {player_idx}: {e}", exc_info=True)
            raise
        # except Exception as e:
        #     return f"Error describing state: {e}", self.describe_inventory(state, player_idx)

    def describe_status(self, state, player_idx):
        """Describe one player's time, condition, role, level, and progress.

        Args:
            state: Current environment state.
            player_idx: Player whose status should be described.

        Returns:
            Multi-line status text.
        """
        lines = []

        # Step count
        timestep = int(state.timestep)
        max_timesteps = int(self.env_params.max_timesteps)
        lines.append(
            f"Step: {timestep}/{max_timesteps} ({max_timesteps - timestep} remaining, ends early if all agents die)"
        )
        if self.exact_coordinates:
            lines.append(f"Position: {self._xy_coord_str(state.player_position[player_idx])}")

        # Condition
        if state.is_sleeping[player_idx]:
            lines.append("Status: sleeping (cannot act until energy is full)")
        elif state.player_health[player_idx] <= 0:
            lines.append("Status: dead")
        elif state.is_resting[player_idx]:
            lines.append("Status: resting (recovering health and mana)")

        # Specialization
        spec = int(state.player_specialization[player_idx])
        if spec > 0:
            spec_name = self.spec_names.get(spec, "unknown")
            lines.append(f"Role: {spec_name}")

        # Current level
        level = int(state.player_level)
        level_name = LEVEL_NAMES.get(level)
        if level == 0:
            lines.append(f"Location: {level_name} (surface)")
        elif level_name:
            lines.append(f"Location: dungeon level {level} — {level_name}")
        else:
            lines.append(f"Location: dungeon level {level}")

        # Achievement count — total shown matches what's listed in the system prompt
        achievements_arr = np.array(state.achievements[player_idx])
        num_done = int(achievements_arr.sum())
        num_total = len(achievements_arr)
        visible_count = len(_achievements_for_level(level, self.env_params.coordination_enabled))
        lines.append(
            f"Achievements: {num_done}/{num_total} ({num_total - visible_count} unlock later)"
        )

        return "\n".join(lines)

    def _get_local_light_mask(self, state, player_idx):
        """Build a boolean light mask for the local view, matching RL symbolic obs (light_map > 0.05)."""
        current_level, player_pos, view_h, view_w, half_h, half_w = self._get_local_view_params(
            state, player_idx
        )
        light_map = np.array(state.light_map[current_level])
        r, c = int(player_pos[0]), int(player_pos[1])

        # Pad light map the same way the symbolic renderer does
        pad = MAX_OBS_DIM + 2
        padded = np.pad(light_map, pad, constant_values=0.0)
        tl_r = r - half_h + pad
        tl_c = c - half_w + pad
        local_light = padded[tl_r : tl_r + view_h, tl_c : tl_c + view_w]
        return local_light > 0.05

    def _scan_visible_objects(self, state, player_idx, skip_items):
        """Scan the local view and return (name, location_str) pairs for visible objects.

        Parameterised on skip_items so it can be reused by describe_env() (which uses
        self.skip_items) and by hybrid mode (which uses _HYBRID_SKIP_ITEMS).
        Items (torches, ladders) are always included regardless of skip_items.
        """
        current_level = int(state.player_level)
        map_state = np.array(state.map[current_level])
        item_map_state = np.array(state.item_map[current_level])
        player_pos = np.array(state.player_position[player_idx])

        view_h, view_w = OBS_DIM
        half_h, half_w = view_h // 2, view_w // 2
        r, c = int(player_pos[0]), int(player_pos[1])

        pad = MAX_OBS_DIM + 2
        padded_map = np.pad(map_state, pad, constant_values=BlockType.OUT_OF_BOUNDS.value)
        padded_items = np.pad(item_map_state, pad, constant_values=ItemType.NONE.value)
        tl_r = r - half_h + pad
        tl_c = c - half_w + pad
        local_view = padded_map[tl_r : tl_r + view_h, tl_c : tl_c + view_w]
        local_items = padded_items[tl_r : tl_r + view_h, tl_c : tl_c + view_w]
        light_mask = self._get_local_light_mask(state, player_idx)
        center = np.array([half_h, half_w])

        direction = int(state.player_direction[player_idx])
        facing_for_loc = direction if self.egocentric else None
        describe_loc = describe_loc_precise if self.precise_location else describe_loc_old

        # Edge detection for large terrain features (e.g. water — show only edges)
        edge_masks = {}
        for item_name in self.edge_only_items:
            for block_id, bname in self.block_id_to_name.items():
                if bname == item_name:
                    edge_masks[block_id] = get_edge_items(local_view, block_id)
                    break

        # Scan the environment for interesting blocks and items
        obj_info_list = []
        for i in range(view_h):
            for j in range(view_w):
                if i == center[0] and j == center[1]:
                    continue

                # Light masking — skip cells in darkness
                if not light_mask[i, j]:
                    continue

                # Block layer
                block_id = int(local_view[i, j])
                block_name = self.block_id_to_name.get(block_id, "unknown")

                if block_id in edge_masks and not edge_masks[block_id][i, j]:
                    pass  # edge-only: skip interior cell but still check item layer
                elif block_name not in skip_items:
                    rel_pos = np.array([i, j]) - center
                    dist = int(abs(rel_pos[0]) + abs(rel_pos[1]))
                    loc = describe_loc(np.array([0, 0]), rel_pos, facing=facing_for_loc)
                    map_pos = np.array([r - half_h + i, c - half_w + j])
                    loc = self._append_exact_coordinate(loc, map_pos)
                    obj_info_list.append((block_name, loc, dist))

                # Item layer (torches, ladders)
                item_id = int(local_items[i, j])
                if item_id != ItemType.NONE.value and item_id in ITEM_NAMES:
                    rel_pos = np.array([i, j]) - center
                    dist = int(abs(rel_pos[0]) + abs(rel_pos[1]))
                    loc = describe_loc(np.array([0, 0]), rel_pos, facing=facing_for_loc)
                    map_pos = np.array([r - half_h + i, c - half_w + j])
                    loc = self._append_exact_coordinate(loc, map_pos)
                    obj_info_list.append((ITEM_NAMES[item_id], loc, dist))

        # Filter to unique items (closest of each type)
        if self.unique_items:
            obj_info_list = self._filter_closest_items(obj_info_list)

        # Strip distance before returning; downstream consumers expect (name, loc) pairs.
        return [(name, loc) for name, loc, _ in obj_info_list]

    def describe_env(self, state, player_idx):
        """Describe the environment around the player.

        Applies light masking to match RL symbolic obs — blocks/items in
        darkness (light_map <= 0.05) are hidden.

        Args:
            state: Current environment state.
            player_idx: Player whose local view should be described.

        Returns:
            Text description of visible nearby blocks and items.
        """
        # Get the map for the current level
        current_level = int(state.player_level)
        map_state = np.array(state.map[current_level])
        player_pos = np.array(state.player_position[player_idx])

        # View dimensions from OBS_DIM
        view_h, view_w = OBS_DIM
        half_h, half_w = view_h // 2, view_w // 2
        r, c = int(player_pos[0]), int(player_pos[1])

        # Extract local map using padding (matching symbolic renderer)
        pad = MAX_OBS_DIM + 2
        padded_map = np.pad(map_state, pad, constant_values=BlockType.OUT_OF_BOUNDS.value)
        tl_r = r - half_h + pad
        tl_c = c - half_w + pad
        local_view = padded_map[tl_r : tl_r + view_h, tl_c : tl_c + view_w]

        # Light mask — hide tiles in darkness
        light_mask = self._get_local_light_mask(state, player_idx)
        center = np.array([half_h, half_w])

        # Facing direction
        # Index: 0=noop, 1=LEFT/west, 2=RIGHT/east, 3=UP/north, 4=DOWN/south
        direction = int(state.player_direction[player_idx])
        direction_vectors = [
            np.array([0, 0]),  # 0: noop
            np.array([0, -1]),  # 1: LEFT (west)
            np.array([0, 1]),  # 2: RIGHT (east)
            np.array([-1, 0]),  # 3: UP (north)
            np.array([1, 0]),  # 4: DOWN (south)
        ]
        facing_name = DIRECTION_NAMES.get(direction, "none")
        facing_vec = direction_vectors[direction]
        target_pos = center + facing_vec

        if facing_name != "none" and 0 <= target_pos[0] < view_h and 0 <= target_pos[1] < view_w:
            if light_mask[target_pos[0], target_pos[1]]:
                target_id = int(local_view[target_pos[0], target_pos[1]])
                # Don't skip the facing cell even if block type is in skip_items (e.g. grass/path)
                block_name = self.block_id_to_name.get(target_id, "unknown")

                # Check if a teammate or mob occupies the facing cell
                entity = self._entity_at(state, player_idx, player_pos + facing_vec, current_level)
                if entity:
                    # Show entity AND underlying block — block matters for Do action effects
                    # (e.g. forager doing Do on grass can yield a sapling)
                    target_name = f"{entity} (on {block_name})"
                else:
                    # Always show the actual block at the facing cell (Do acts here),
                    # even if the block is in skip_items (e.g. grass)
                    target_name = block_name
            else:
                target_name = "darkness"
            if self.exact_coordinates:
                target_map_pos = player_pos + facing_vec
                target_name = (
                    f"{target_name} {self._append_exact_coordinate('', target_map_pos)}".strip()
                )
            obs = f"Facing: {facing_name}.\n Do target: {target_name}."
        else:
            obs = "Facing: none."

        # Object list (reuses shared scan logic, see _scan_visible_objects)
        obj_info_list = self._scan_visible_objects(state, player_idx, self.skip_items)

        # Format the object list
        if obj_info_list:
            status_str = "You see:\n" + "\n".join(f"- {name} {loc}" for name, loc in obj_info_list)
        else:
            status_str = "You see nothing away from you."

        return (status_str + "\n\n" + obs).strip()

    def _filter_closest_items(self, obj_info_list):
        """Keep only the nearest instance of each item type by true player-relative distance.

        Expects (name, loc, dist) tuples. Previously this parsed numbers out of `loc`,
        which breaks when exact_coordinates=True (loc becomes "(x=C, y=R)" — summing
        those picks the item nearest map origin, not nearest to the player).
        """
        closest = {}  # name -> (loc, dist)
        for item_name, loc, dist in obj_info_list:
            if item_name not in closest or dist < closest[item_name][1]:
                closest[item_name] = (loc, dist)
        return [(name, loc, dist) for name, (loc, dist) in closest.items()]

    def _get_local_view_params(self, state, player_idx):
        """Return common view parameters: current_level, player_pos, view dims, half dims."""
        current_level = int(state.player_level)
        player_pos = np.array(state.player_position[player_idx])
        view_h, view_w = OBS_DIM
        half_h, half_w = view_h // 2, view_w // 2
        return current_level, player_pos, view_h, view_w, half_h, half_w

    def _entity_at(self, state, player_idx, pos, current_level):
        """Return name of teammate or mob at pos, or None."""
        for i in range(self.num_agents):
            if (
                i != player_idx
                and bool(state.player_alive[i])
                and np.array_equal(state.player_position[i], pos)
            ):
                return f"Agent {i}"
        for mobs, names in [
            (state.melee_mobs, MELEE_MOB_NAMES),
            (state.ranged_mobs, RANGED_MOB_NAMES),
            (state.passive_mobs, PASSIVE_MOB_NAMES),
        ]:
            mask = np.array(mobs.mask[current_level])
            positions = np.array(mobs.position[current_level])
            type_ids = np.array(mobs.type_id[current_level])
            hit = np.where((mask >= 1) & np.all(positions == pos, axis=1))[0]
            if len(hit):
                tid = int(type_ids[hit[0]])
                return names[tid] if tid < len(names) else "creature"
        return None

    def _xy_coord_str(self, abs_pos):
        """Format an absolute map position as Cartesian-style (x, y)."""
        r, c = int(abs_pos[0]), int(abs_pos[1])
        return f"(x={c}, y={r})"

    def _append_exact_coordinate(self, loc_str, abs_pos):
        """Append absolute (x=C, y=R) coords to a relative-direction string.

        Previously this replaced `loc_str` entirely, which silently discarded any
        `precise_location` / egocentric direction string — i.e. agents lost the
        relative "3 steps north" hint whenever exact_coordinates=True.
        Now returns "{loc_str} (x=C, y=R)" when both are available; falls back to
        coord-only when loc_str is empty (matches the target-name caller that
        passes an empty first arg).
        """
        if not self.exact_coordinates:
            return loc_str
        if abs_pos is None:
            return loc_str
        coord = self._xy_coord_str(abs_pos)
        if loc_str:
            return f"{loc_str} {coord}"
        return coord

    def render_ascii_map(self, state, player_idx):
        """Render the local 9x11 view as a compact ASCII grid.

        Args:
            state: Current environment state.
            player_idx: Player whose local view should be rendered.

        Returns:
            Multi-line ASCII map with a legend.
        """
        return _render_ascii_map(
            state=state,
            player_idx=player_idx,
            num_agents=self.num_agents,
            direction_names=DIRECTION_NAMES,
            block_id_to_name=self.block_id_to_name,
            egocentric=self.egocentric,
            entity_at_fn=self._entity_at,
            light_mask_fn=self._get_local_light_mask,
        )

    def _describe_agent_side(self, dr, dc, player_direction):
        """Describe the agent's position relative to a coordination target.

        Explicitly states which side of the target the agent is on, preventing
        the common LLM inversion error (e.g. "tree 1 step north" → incorrectly
        concluding "I'm north of the tree" when the agent is actually south).

        Args:
            dr, dc: Delta from player to target (target_row - player_row, etc.)
            player_direction: Player's current facing direction (1-4).

        Returns:
            Position hint string, e.g. "You are north of it (adjacent, face south to Do it)."
        """
        # dr > 0 means target is south of player → player is north of target
        # dc > 0 means target is east of player → player is west of target
        side_parts = []
        if dr > 0:
            side_parts.append("north")
        elif dr < 0:
            side_parts.append("south")
        if dc > 0:
            side_parts.append("west")
        elif dc < 0:
            side_parts.append("east")
        side = "-".join(side_parts) if side_parts else "on"

        dist = abs(dr) + abs(dc)
        if dist == 1:
            needed_facing = {
                (1, 0): "south",
                (-1, 0): "north",
                (0, 1): "east",
                (0, -1): "west",
            }.get((dr, dc), "")
            facing_map = {1: "west", 2: "east", 3: "north", 4: "south"}
            currently_facing = facing_map.get(player_direction, "")
            facing_str = (
                "facing target"
                if currently_facing == needed_facing
                else f"facing {currently_facing}"
            )
            return f"You are on the {side} side, adjacent ({facing_str})."
        elif dist == 0:
            return "You are on it."
        else:
            return f"You are on the {side} side, {dist} steps away."

    def _relative_direction_str(self, dr, dc, facing=None, abs_pos=None):
        """Convert row/col delta to a human-readable direction string.

        Respects both precise_location and egocentric flags, consistent with
        describe_items_in_view, describe_teammates, and describe_coordination_cues.

        Args:
            facing: Player facing direction (1-4). Used when self.egocentric is True.
        """
        describe_loc = describe_loc_precise if self.precise_location else describe_loc_old
        loc = describe_loc(np.array([0, 0]), np.array([dr, dc]), facing=facing)
        return self._append_exact_coordinate(loc, abs_pos)

    def describe_mobs(self, state, player_idx):
        """Describe visible mobs and projectiles near a player.

        Args:
            state: Current environment state.
            player_idx: Player whose local view should be described.

        Returns:
            Creature and projectile sections, or an empty string.
        """
        current_level, player_pos, view_h, view_w, half_h, half_w = self._get_local_view_params(
            state, player_idx
        )
        pr, pc = int(player_pos[0]), int(player_pos[1])

        # Get light map for visibility check
        light_map = np.array(state.light_map[current_level])

        # Facing direction for egocentric descriptions
        facing = int(state.player_direction[player_idx]) if self.egocentric else None

        # Floor mob type mapping
        floor_types = np.array(
            FLOOR_MOB_MAPPING[current_level]
        )  # [passive_type, melee_type, ranged_type]

        mob_lines = []
        projectile_lines = []

        # Helper to check if a position is in view and lit
        def _in_view_and_lit(mob_r, mob_c):
            lr = mob_r - pr + half_h
            lc = mob_c - pc + half_w
            if not (0 <= lr < view_h and 0 <= lc < view_w):
                return False, 0, 0
            # Check light at actual map position
            if 0 <= mob_r < light_map.shape[0] and 0 <= mob_c < light_map.shape[1]:
                if light_map[mob_r, mob_c] <= 0.05:
                    return False, 0, 0
            return True, mob_r - pr, mob_c - pc

        # --- Melee mobs ---
        melee_mobs = state.melee_mobs
        melee_mask = np.array(melee_mobs.mask[current_level])
        melee_pos = np.array(melee_mobs.position[current_level])
        melee_type_ids = np.array(melee_mobs.type_id[current_level])
        melee_coord = np.array(state.melee_mob_coordination[current_level])

        for i in range(len(melee_mask)):
            if melee_mask[i] < 1:
                continue
            mr, mc = int(melee_pos[i, 0]), int(melee_pos[i, 1])
            visible, dr, dc = _in_view_and_lit(mr, mc)
            if not visible:
                continue
            type_id = int(melee_type_ids[i]) if melee_type_ids.ndim > 0 else int(floor_types[1])
            name = (
                MELEE_MOB_NAMES[type_id]
                if type_id < len(MELEE_MOB_NAMES)
                else f"melee_mob_{type_id}"
            )
            loc = self._relative_direction_str(dr, dc, facing, abs_pos=np.array([mr, mc]))
            coord_tag = ""
            if self.prompt_mode == "specific_collaborative":
                coord_val = int(melee_coord[i])
                if coord_val == 1:
                    coord_tag = " [elite — fight alongside teammates for bonus]"
                elif coord_val == 2:
                    coord_tag = " [elite — must fight alongside teammates]"
            mob_lines.append(f"- {name} {loc}{coord_tag}")

        # --- Ranged mobs ---
        ranged_mobs = state.ranged_mobs
        ranged_mask = np.array(ranged_mobs.mask[current_level])
        ranged_pos = np.array(ranged_mobs.position[current_level])
        ranged_type_ids = np.array(ranged_mobs.type_id[current_level])
        ranged_coord = np.array(state.ranged_mob_coordination[current_level])

        for i in range(len(ranged_mask)):
            if ranged_mask[i] < 1:
                continue
            mr, mc = int(ranged_pos[i, 0]), int(ranged_pos[i, 1])
            visible, dr, dc = _in_view_and_lit(mr, mc)
            if not visible:
                continue
            type_id = int(ranged_type_ids[i]) if ranged_type_ids.ndim > 0 else int(floor_types[2])
            name = (
                RANGED_MOB_NAMES[type_id]
                if type_id < len(RANGED_MOB_NAMES)
                else f"ranged_mob_{type_id}"
            )
            loc = self._relative_direction_str(dr, dc, facing, abs_pos=np.array([mr, mc]))
            coord_tag = " [ranged]"
            if self.prompt_mode == "specific_collaborative":
                coord_val = int(ranged_coord[i])
                if coord_val == 1:
                    coord_tag = " [ranged, elite — fight alongside teammates for bonus]"
                elif coord_val == 2:
                    coord_tag = " [ranged, elite — must fight alongside teammates]"
            mob_lines.append(f"- {name} {loc}{coord_tag}")

        # --- Passive mobs ---
        passive_mobs = state.passive_mobs
        passive_mask = np.array(passive_mobs.mask[current_level])
        passive_pos = np.array(passive_mobs.position[current_level])
        passive_type_ids = np.array(passive_mobs.type_id[current_level])
        passive_coord = np.array(state.passive_mob_coordination[current_level])

        for i in range(len(passive_mask)):
            if passive_mask[i] < 1:
                continue
            mr, mc = int(passive_pos[i, 0]), int(passive_pos[i, 1])
            visible, dr, dc = _in_view_and_lit(mr, mc)
            if not visible:
                continue
            type_id = int(passive_type_ids[i]) if passive_type_ids.ndim > 0 else int(floor_types[0])
            name = (
                PASSIVE_MOB_NAMES[type_id]
                if type_id < len(PASSIVE_MOB_NAMES)
                else f"passive_mob_{type_id}"
            )
            loc = self._relative_direction_str(dr, dc, facing, abs_pos=np.array([mr, mc]))
            coord_tag = ""
            if self.prompt_mode == "specific_collaborative":
                coord_val = int(passive_coord[i])
                if coord_val == 1:
                    coord_tag = " [large — hunt alongside teammates for bonus]"
                elif coord_val == 2:
                    coord_tag = " [large — must hunt alongside teammates]"
            mob_lines.append(f"- {name} {loc}{coord_tag}")

        # --- Mob projectiles ---
        proj = state.mob_projectiles
        proj_mask = np.array(proj.mask[current_level])
        proj_pos = np.array(proj.position[current_level])
        proj_type_ids = np.array(proj.type_id[current_level])
        for i in range(len(proj_mask)):
            if proj_mask[i] < 1:
                continue
            mr, mc = int(proj_pos[i, 0]), int(proj_pos[i, 1])
            visible, dr, dc = _in_view_and_lit(mr, mc)
            if not visible:
                continue
            type_id = int(proj_type_ids[i]) if proj_type_ids.ndim > 0 else 0
            name = (
                PROJECTILE_NAMES[type_id]
                if type_id < len(PROJECTILE_NAMES)
                else f"projectile_{type_id}"
            )
            loc = self._relative_direction_str(dr, dc, facing, abs_pos=np.array([mr, mc]))
            projectile_lines.append(f"- {name} {loc}")

        # --- Player projectiles (arrows/spells fired by players) ---
        player_proj = state.player_projectiles
        pp_mask = np.array(player_proj.mask[current_level])
        pp_pos = np.array(player_proj.position[current_level])
        pp_type_ids = np.array(player_proj.type_id[current_level])
        player_proj_lines = []
        for i in range(len(pp_mask)):
            if pp_mask[i] < 1:
                continue
            mr, mc = int(pp_pos[i, 0]), int(pp_pos[i, 1])
            visible, dr, dc = _in_view_and_lit(mr, mc)
            if not visible:
                continue
            type_id = int(pp_type_ids[i]) if pp_type_ids.ndim > 0 else 0
            name = (
                PROJECTILE_NAMES[type_id]
                if type_id < len(PROJECTILE_NAMES)
                else f"projectile_{type_id}"
            )
            loc = self._relative_direction_str(dr, dc, facing, abs_pos=np.array([mr, mc]))
            player_proj_lines.append(f"- {name} (yours) {loc}")

        result = ""
        if mob_lines:
            result += "Nearby creatures:\n" + "\n".join(mob_lines)
        if projectile_lines:
            if result:
                result += "\n\n"
            result += "Incoming projectiles:\n" + "\n".join(projectile_lines)
        if player_proj_lines:
            if result:
                result += "\n\n"
            result += "Your projectiles:\n" + "\n".join(player_proj_lines)
        return result

    def describe_level_info(self, state):
        """Describe level-wide information matching the RL symbolic observation.

        Only includes information available in the symbolic obs vector:
        - light_level (float)
        - level cleared (bool: monsters_killed >= threshold)
        - boss vulnerable (bool)

        Args:
            state: Current environment state.

        Returns:
            Multi-line level-wide status text.
        """
        current_level = int(state.player_level)
        lines = []

        # Light level (matches symbolic obs)
        light = float(state.light_level)
        if light > 0.7:
            light_desc = "bright"
        elif light > 0.3:
            light_desc = "dim"
        else:
            light_desc = "dark"
        lines.append(f"- Light: {light_desc} ({light:.2f})")

        # Level cleared boolean (matches symbolic obs: monsters_killed >= threshold)
        monsters_killed = int(state.monsters_killed[current_level])
        level_cleared = bool(monsters_killed >= MONSTERS_KILLED_TO_CLEAR_LEVEL)
        if level_cleared:
            lines.append(
                "- Level: cleared — you can find the ladder down tile and use Descend to go deeper."
            )
        else:
            lines.append("- Level: not yet cleared (kill more monsters to unlock ladder)")

        # Boss vulnerable boolean (matches symbolic obs)
        if is_boss_vulnerable(state):
            lines.append("- Boss: vulnerable — attack now!")

        return "Level info:\n" + "\n".join(lines)

    def describe_coordination_cues(self, state, player_idx):
        """Describe coordination requirements visible in the local view.

        Matches RL symbolic obs: coordination channels are masked by light_map > 0.05,
        so cells in darkness are not reported.

        Args:
            state: Current environment state.
            player_idx: Player whose local coordination cues should be described.

        Returns:
            Coordination section, or an empty string when no cues are visible.
        """
        current_level, player_pos, view_h, view_w, half_h, half_w = self._get_local_view_params(
            state, player_idx
        )
        pr, pc = int(player_pos[0]), int(player_pos[1])
        player_direction = int(state.player_direction[player_idx])
        facing = player_direction if self.egocentric else None

        coord_map = np.array(state.coordination_map[current_level])
        soft_mask = np.array(state.soft_coordination_mask[current_level])
        map_state = np.array(state.map[current_level])

        # Light mask — coordination cues in darkness are hidden (matches RL obs)
        light_mask = self._get_local_light_mask(state, player_idx)

        lines = []
        seen_types = {}  # key: (coord_val, is_soft) -> closest distance

        # Scan local view for coordination cues
        for lr in range(view_h):
            for lc in range(view_w):
                # Skip cells in darkness
                if not light_mask[lr, lc]:
                    continue
                mr = pr - half_h + lr
                mc = pc - half_w + lc
                if mr < 0 or mc < 0 or mr >= coord_map.shape[0] or mc >= coord_map.shape[1]:
                    continue
                val = int(coord_map[mr, mc])
                if val == 0:
                    continue

                dr, dc = mr - pr, mc - pc
                dist = abs(dr) + abs(dc)
                is_soft = bool(soft_mask[mr, mc])
                key = (val, is_soft)

                if key in seen_types and seen_types[key][0] <= dist:
                    continue
                seen_types[key] = (dist, dr, dc, mr, mc)

        for (val, is_soft), (dist, dr, dc, mr, mc) in sorted(
            seen_types.items(), key=lambda x: x[1][0]
        ):
            block_id = int(map_state[mr, mc])
            block_name = self.block_id_to_name.get(block_id, "unknown")
            loc = self._relative_direction_str(dr, dc, facing, abs_pos=np.array([mr, mc]))

            # Explicitly state which side of the target the agent is on — prevents
            # the LLM from inverting relative directions (e.g. "tree 1 step north"
            # → "I'm north of the tree"). Derivable from the RL coord_map_view offset.
            position_hint = self._describe_agent_side(dr, dc, player_direction)

            is_construction = block_id in (
                BlockType.CONSTRUCTION_SITE.value,
                BlockType.CONSTRUCTION_IN_PROGRESS.value,
            )
            action_verb = "use a Build action" if is_construction else "select Do"

            if val > 0:
                if is_soft:
                    lines.append(
                        f"- {block_name} {loc}: works solo but grants a bonus when {val} agents {action_verb} simultaneously. {position_hint}"
                    )
                else:
                    lines.append(
                        f"- {block_name} {loc}: requires {val} agents to {action_verb} simultaneously (fails alone). {position_hint}"
                    )
            else:
                window = abs(val)
                lines.append(
                    f"- {block_name} {loc}: one agent begins with a Build action, then another completes it within {window} steps. {position_hint}"
                    if is_construction
                    else f"- {block_name} {loc}: one agent begins the task, then another completes it within {window} steps. {position_hint}"
                )

        # Check pending handovers in view (matches RL handover_obs channel)
        # Only show active handovers that haven't expired and are within the lit local view
        pending = np.array(state.pending_handovers)
        for i in range(pending.shape[0]):
            if int(pending[i, 0]) < 1:
                continue
            hx, hy = int(pending[i, 1]), int(pending[i, 2])
            dr, dc = hx - pr, hy - pc
            # Check within view bounds
            lr, lc = dr + half_h, dc + half_w
            if not (0 <= lr < view_h and 0 <= lc < view_w):
                continue
            # Check light
            if not light_mask[lr, lc]:
                continue
            deadline = int(pending[i, 3])
            remaining = deadline - int(state.timestep)
            if remaining <= 0:
                continue
            loc = self._relative_direction_str(dr, dc, facing, abs_pos=np.array([hx, hy]))
            lines.append(
                f"- Active handover {loc}: a teammate started this — go complete it within {remaining} steps!"
            )

        if lines:
            return "Coordination:\n" + "\n".join(lines)
        return ""

    def describe_construction(self, state, player_idx):
        """Construction sites appear in describe_env() by block name.

        The coordination type (sync/handover, agents required) is shown in
        describe_coordination_cues(), which now correctly says 'use a Build action'
        instead of 'select Do'. Available Build actions appear in the affordances
        list. No separate section needed.

        Args:
            state: Current environment state, unused by this compatibility hook.
            player_idx: Player index, unused by this compatibility hook.

        Returns:
            An empty string because construction is described elsewhere.
        """
        return ""

    def describe_teammates(self, state, player_idx):
        """Describe the positions and status of teammates.

        Matches RL symbolic obs: on-screen teammates get spatial position,
        off-screen teammates get a direction indicator (matching teammate_directions
        in the symbolic renderer).

        Args:
            state: Current environment state.
            player_idx: Player whose teammates should be described.

        Returns:
            Teammate section, or an empty string for a solo team.
        """
        parts = []
        player_pos = np.array(state.player_position[player_idx])
        view_h, view_w = OBS_DIM
        half_h, half_w = view_h // 2, view_w // 2
        facing = int(state.player_direction[player_idx]) if self.egocentric else None
        facing_for_loc = facing  # pass to describe_loc helpers
        describe_loc = describe_loc_precise if self.precise_location else describe_loc_old

        for i in range(self.num_agents):
            if i == player_idx:
                continue

            teammate_pos = np.array(state.player_position[i])
            alive = bool(state.player_alive[i])

            if not alive:
                parts.append(f"Agent {i} is dead.")
                continue

            # Check if teammate is on screen (matching symbolic renderer logic)
            local_pos = teammate_pos - player_pos + np.array([half_h, half_w])
            on_screen = (
                local_pos[0] >= 0
                and local_pos[0] < view_h
                and local_pos[1] >= 0
                and local_pos[1] < view_w
            )

            spec = int(state.player_specialization[i])
            spec_name = self.spec_names.get(spec, "")
            spec_str = f" ({spec_name})" if spec > 0 else ""
            health = int(state.player_health[i])

            if on_screen:
                # On-screen: give precise relative position
                rel = teammate_pos - player_pos
                dist = abs(int(rel[0])) + abs(int(rel[1]))
                if dist == 0:
                    loc_str = "at your location"
                else:
                    loc_str = describe_loc(np.array([0, 0]), rel, facing=facing_for_loc)
                loc_str = self._append_exact_coordinate(loc_str, teammate_pos)
                parts.append(f"Agent {i}{spec_str}: {loc_str}, health={health}")
            else:
                # Off-screen: give direction indicator (matching RL teammate_directions)
                # Compute absolute direction first
                abs_parts = []
                if local_pos[0] < 0:
                    abs_parts.append("north")
                elif local_pos[0] >= view_h:
                    abs_parts.append("south")
                if local_pos[1] < 0:
                    abs_parts.append("west")
                elif local_pos[1] >= view_w:
                    abs_parts.append("east")

                if self.egocentric and facing is not None:
                    # Convert off-screen direction to egocentric
                    off_dr = -1 if local_pos[0] < 0 else (1 if local_pos[0] >= view_h else 0)
                    off_dc = -1 if local_pos[1] < 0 else (1 if local_pos[1] >= view_w else 0)
                    ego_parts = _egocentric_direction(off_dr, off_dc, facing)
                    dir_str = "-".join(ego_parts) if ego_parts else "nearby"
                else:
                    dir_str = "-".join(abs_parts) if abs_parts else "nearby"
                if self.exact_coordinates:
                    dir_str = self._append_exact_coordinate(dir_str, teammate_pos)
                parts.append(f"Agent {i}{spec_str}: off-screen {dir_str}, health={health}")

            # Request info (RL obs encodes request type one-hot masked by duration > 0, no duration value)
            req_type = int(state.request_type[i])
            req_duration = int(state.request_duration[i])
            if req_type > 0 and req_duration > 0:
                req_names = [
                    "food",
                    "drink",
                    "wood",
                    "stone",
                    "iron",
                    "coal",
                    "diamond",
                    "ruby",
                    "sapphire",
                ]
                req_idx = req_type - Action.REQUEST_FOOD.value
                if 0 <= req_idx < len(req_names):
                    parts.append(f"  -> Requesting {req_names[req_idx]}")

        if parts:
            return "Teammates:\n" + "\n".join(parts)
        return ""

    def describe_inventory(self, state, player_idx):
        """Describe one player's inventory, equipment, attributes, and vitals.

        Args:
            state: Current environment state.
            player_idx: Player whose inventory should be described.

        Returns:
            Multi-line inventory and status text.
        """
        result = []

        # Player vitals — show raw values (RL obs normalizes by constant /10, not dynamic max)
        result.append("Your status:")
        result.append(f"- health: {int(state.player_health[player_idx])}")
        result.append(f"- food: {int(state.player_food[player_idx])}")
        result.append(f"- drink: {int(state.player_drink[player_idx])}")
        result.append(f"- energy: {int(state.player_energy[player_idx])}")
        result.append(f"- mana: {int(state.player_mana[player_idx])}")
        result.append(f"- xp: {int(state.player_xp[player_idx])}")
        result.append("")

        # Attributes
        dex = int(state.player_dexterity[player_idx])
        strength = int(state.player_strength[player_idx])
        intelligence = int(state.player_intelligence[player_idx])
        if dex > 0 or strength > 0 or intelligence > 0:
            result.append("Attributes:")
            if dex > 0:
                result.append(f"- dexterity: {dex}")
            if strength > 0:
                result.append(f"- strength: {strength}")
            if intelligence > 0:
                result.append(f"- intelligence: {intelligence}")
            result.append("")

        # Inventory items
        inv = state.inventory
        inventory_items = []

        items = {
            "wood": int(inv.wood[player_idx]),
            "stone": int(inv.stone[player_idx]),
            "coal": int(inv.coal[player_idx]),
            "iron": int(inv.iron[player_idx]),
            "diamond": int(inv.diamond[player_idx]),
            "ruby": int(inv.ruby[player_idx]),
            "sapphire": int(inv.sapphire[player_idx]),
            "sapling": int(inv.sapling[player_idx]),
            "arrows": int(inv.arrows[player_idx]),
            "torches": int(inv.torches[player_idx]),
        }

        for item_name, count in items.items():
            if count > 0:
                inventory_items.append(f"- {item_name}: {count}")

        # Tools (pickaxe level: 0=none, 1=wood, 2=stone, 3=iron, 4=diamond)
        pickaxe_level = int(inv.pickaxe[player_idx])
        if pickaxe_level > 0:
            pickaxe_names = {1: "wood", 2: "stone", 3: "iron", 4: "diamond"}
            inventory_items.append(f"- pickaxe: {pickaxe_names.get(pickaxe_level, 'unknown')}")

        sword_level = int(inv.sword[player_idx])
        if sword_level > 0:
            sword_names = {1: "wood", 2: "stone", 3: "iron", 4: "diamond"}
            inventory_items.append(f"- sword: {sword_names.get(sword_level, 'unknown')}")

        bow_level = int(inv.bow[player_idx])
        if bow_level > 0:
            inventory_items.append("- bow: yes")

        # Armour: per-slot tier in inv.armour (shape: (player_count, 4) — 4 slots).
        # Report a per-tier breakdown so heterogeneous loadouts (e.g. iron+diamond)
        # are visible; previous "max + total count" wording implied all slots
        # matched the strongest tier.
        armour_names = {1: "iron", 2: "diamond"}
        armour_arr = np.array(inv.armour[player_idx])
        if armour_arr.ndim > 0:
            tier_counts = {}
            for level in armour_arr.tolist():
                level = int(level)
                if level > 0:
                    tier_counts[level] = tier_counts.get(level, 0) + 1
            if tier_counts:
                parts = [
                    f"{count}x {armour_names.get(level, f'level {level}')}"
                    for level, count in sorted(tier_counts.items(), reverse=True)
                ]
                inventory_items.append(f"- armour: {', '.join(parts)}")
        else:
            armour_level = int(armour_arr)
            if armour_level > 0:
                inventory_items.append(f"- armour: {armour_names.get(armour_level, 'unknown')}")

        # Potions
        potions = np.array(inv.potions[player_idx])
        potion_names = ["red", "green", "blue", "pink", "cyan", "yellow"]
        for i, name in enumerate(potion_names):
            if i < len(potions) and int(potions[i]) > 0:
                inventory_items.append(f"- potion_{name}: {int(potions[i])}")

        # Books
        books = int(inv.books[player_idx])
        if books > 0:
            inventory_items.append(f"- books: {books}")

        # Enchantments
        ench_names = {1: "fire", 2: "ice"}
        sword_ench = int(state.sword_enchantment[player_idx])
        if sword_ench > 0:
            inventory_items.append(f"- sword enchantment: {ench_names.get(sword_ench, sword_ench)}")
        bow_ench = int(state.bow_enchantment[player_idx])
        if bow_ench > 0:
            inventory_items.append(f"- bow enchantment: {ench_names.get(bow_ench, bow_ench)}")
        armour_enchs = np.array(state.armour_enchantments[player_idx])
        total_armour_ench = int(armour_enchs.sum())
        if total_armour_ench > 0:
            inventory_items.append(f"- armour_enchantments: {total_armour_ench}")

        # Learned spells: learned_spells is 1D (player_count,) — one bool per player.
        # Spell identity follows role (hard-gated in game_logic.py even with soft_specialization):
        # Forager → heal, Warrior/Miner → fireball.
        if bool(state.learned_spells[player_idx]):
            spec = int(state.player_specialization[player_idx])
            spell_name = "heal" if spec == Specialization.FORAGER.value else "fireball"
            inventory_items.append(f"- spells known: {spell_name}")

        if inventory_items:
            result.append("Your inventory:")
            result.extend(inventory_items)
        else:
            result.append("You have nothing in your inventory.")

        return "\n".join(result)

    def update_progress(self, state, player_idx):
        """Update achievement tracking for a specific player.

        Args:
            state: Current environment state.
            player_idx: Player whose achievements should be recorded.

        Returns:
            Updated scalar achievement score.
        """
        achievements_array = np.array(state.achievements[player_idx])
        self.score_trackers[player_idx] = float(np.sum(achievements_array))

        achievement_names = [a.name for a in Achievement]
        self.achievements[player_idx] = {
            name: int(achievements_array[i]) for i, name in enumerate(achievement_names)
        }
        return self.score_trackers[player_idx]

    def get_stats(self, player_idx=None):
        """Get achievement statistics for one or all players.

        Args:
            player_idx: Optional index selecting one player.

        Returns:
            One statistics dictionary, or a list for every player.
        """
        num_achievements = len(Achievement)
        if player_idx is not None:
            return {
                "score": self.score_trackers[player_idx],
                "progression": float(self.score_trackers[player_idx]) / num_achievements,
                "achievements": self.achievements[player_idx]
                if self.achievements[player_idx]
                else {},
            }
        else:
            return [
                {
                    "score": self.score_trackers[i],
                    "progression": float(self.score_trackers[i]) / num_achievements,
                    "achievements": self.achievements[i] if self.achievements[i] else {},
                }
                for i in range(self.num_agents)
            ]

    def get_instruction_prompt(
        self, agent_idx=None, instructions=None, progressive_disclosure=False, current_level=0
    ):
        """Get the configured instruction prompt for an LLM agent.

        Args:
            agent_idx: Optional agent index used to assign a role.
            instructions: Unused compatibility argument.
            progressive_disclosure: Whether late-game rules are level-gated.
            current_level: Current level used for progressive disclosure.

        Returns:
            Complete system-prompt text.
        """
        role = None
        if agent_idx is not None:
            # Role assignment matches world_gen: [WARRIOR, FORAGER, MINER] cycling
            spec_order = [Specialization.WARRIOR, Specialization.FORAGER, Specialization.MINER]
            spec = spec_order[agent_idx % 3]
            role = spec.name.lower()
        return get_instruction_prompt(
            coordination_enabled=self.env_params.coordination_enabled,
            num_agents=self.num_agents,
            agent_id=agent_idx,
            role=role,
            progressive_disclosure=progressive_disclosure,
            current_level=current_level,
            prompt_mode=self.prompt_mode,
        )

    def get_affordances(self, state, player_idx):
        """Return a string listing the non-trivial legal actions for this player.

        Computes the action mask from the current game state,
        and returns the remaining available actions.

        Args:
            state: Current environment state.
            player_idx: Player whose legal actions should be listed.

        Returns:
            Available-actions section, or an empty string.
        """
        mask = compute_action_mask(state, self.env_params, self.static_env_params)
        player_mask = np.array(mask[player_idx])

        available = []
        for i, action_name in enumerate(ACTIONS):
            if i >= len(player_mask):
                break
            if i == Action.GIVE.value:
                continue
            if player_mask[i]:
                available.append(action_name)

        # Add all GIVE slots as target-specific affordances for this player
        for k in range(self.static_env_params.player_count - 1):
            idx = Action.GIVE.value + k
            if idx < len(player_mask) and player_mask[idx]:
                target_idx = k if k < player_idx else k + 1
                available.append(f"Give to Agent {target_idx}")

        if not available:
            return ""
        return "Available actions:\n - " + "\n - ".join(available)

    def check_action_validity(self, action):
        """Canonicalize an action or return the default action.

        Args:
            action: Candidate action text.

        Returns:
            Exact or partial canonical match, falling back to ``Noop``.
        """
        if action in self.language_action_space:
            return action
        else:
            for valid_action in self.language_action_space:
                if action.lower() in valid_action.lower() or valid_action.lower() in action.lower():
                    return valid_action
            return "Noop"


def main():
    """Test the language wrapper."""
    print("Creating Alem-Coop environment...")

    config = {
        "max_timesteps": 10000,
        "god_mode": False,
        "coordination_difficulty": "easy",
        "soft_specialization": True,
        "shared_reward": False,
        "seed": 42,
        "num_agents": 3,
    }

    env = make_alem_env(config=config)
    env_params = env.default_params

    wrapped_env = AlemLanguageWrapper(
        env,
        env_params,
        unique_items=True,
        precise_location=False,
    )

    print("\nResetting environment...")
    rng = jax.random.PRNGKey(config["seed"])
    text_obs_list, state, rng = wrapped_env.reset(rng)

    instruction_prompt = wrapped_env.get_instruction_prompt()

    print("=" * 80)
    print("ENVIRONMENT CREATED SUCCESSFULLY")
    print(f"Number of agents: {wrapped_env.num_agents}")
    print("=" * 80)

    print("\n=== Instruction Prompt ===")
    print(instruction_prompt)
    print("\n" + "=" * 80)

    print("\n=== Initial Observations ===")
    for agent_idx in range(wrapped_env.num_agents):
        print(f"\n--- Agent {agent_idx} ---")
        print("\nLong-term context:")
        print(text_obs_list[agent_idx]["text"]["long_term_context"])
        print("\nShort-term context:")
        print(text_obs_list[agent_idx]["text"]["short_term_context"])

    # Run a few steps as a test
    print("\n" + "=" * 80)
    print("RUNNING TEST EPISODE (10 steps with random actions)")
    print("=" * 80)

    for step in range(10):
        actions = []
        action_indices = []
        for agent_idx in range(wrapped_env.num_agents):
            action_idx = np.random.randint(0, len(ACTIONS))
            action_indices.append(action_idx)
            action_str = ACTIONS[action_idx]
            actions.append(action_str)

        print(f"\n{'=' * 80}")
        print(f"Step {step + 1}:")
        print(f"Action indices: {action_indices}")
        print(f"Action strings: {actions}")
        print("=" * 80)

        text_obs_list, state, rewards, dones, info, rng = wrapped_env.step(state, actions, rng)

        print(f"\nRewards: {rewards}")
        print(f"Dones: {dones}")

        for agent_idx in range(wrapped_env.num_agents):
            print(f"\n--- Agent {agent_idx} ---")
            print("Long-term context:")
            print(text_obs_list[agent_idx]["text"]["long_term_context"])
            print("\nShort-term context:")
            print(text_obs_list[agent_idx]["text"]["short_term_context"])

            stats = wrapped_env.get_stats(agent_idx)
            print(f"Score: {stats['score']}, Progression: {stats['progression']:.2%}")
            if stats["achievements"]:
                achieved = [k for k, v in stats["achievements"].items() if v > 0]
                if achieved:
                    print(f"Achievements: {', '.join(achieved)}")

        if all(dones):
            print("\n" + "=" * 80)
            print("ALL AGENTS FINISHED!")
            print("=" * 80)
            break

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
