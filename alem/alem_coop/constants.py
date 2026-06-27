import os
import pathlib
from enum import Enum

import imageio.v3 as iio
import jax.numpy as jnp
import numpy as np
from flax import struct
from PIL import Image
from seaborn import husl_palette

from ..environment_base.util import load_compressed_pickle, save_compressed_pickle
from .util.maths_utils import get_distance_map

# GAME CONSTANTS
OBS_DIM = (9, 11)
assert OBS_DIM[0] % 2 == 1 and OBS_DIM[1] % 2 == 1
MAX_OBS_DIM = max(OBS_DIM)
BLOCK_PIXEL_SIZE_HUMAN = 64
BLOCK_PIXEL_SIZE_IMG = 16
BLOCK_PIXEL_SIZE_AGENT = 10
INVENTORY_OBS_HEIGHT = 4
TEXTURE_CACHE_FILE = os.path.join(
    pathlib.Path(__file__).parent.resolve(), "assets", "texture_cache.pbz2"
)

REQUEST_MAX_DURATION = 10

# DUNGEON ROOM CONSTANTS
NUM_ROOMS = 8
MIN_ROOM_SIZE = 5
MAX_ROOM_SIZE = 10


# ENUMS
class BlockType(Enum):
    INVALID = 0
    OUT_OF_BOUNDS = 1
    GRASS = 2
    WATER = 3
    STONE = 4
    TREE = 5
    WOOD = 6
    PATH = 7
    COAL = 8
    IRON = 9
    DIAMOND = 10
    CRAFTING_TABLE = 11
    FURNACE = 12
    SAND = 13
    LAVA = 14
    PLANT = 15
    RIPE_PLANT = 16
    WALL = 17
    DARKNESS = 18
    WALL_MOSS = 19
    STALAGMITE = 20
    SAPPHIRE = 21
    RUBY = 22
    CHEST = 23
    FOUNTAIN = 24
    FIRE_GRASS = 25
    ICE_GRASS = 26
    GRAVEL = 27
    FIRE_TREE = 28
    ICE_SHRUB = 29
    ENCHANTMENT_TABLE_FIRE = 30
    ENCHANTMENT_TABLE_ICE = 31
    NECROMANCER = 32
    GRAVE = 33
    GRAVE2 = 34
    GRAVE3 = 35
    NECROMANCER_VULNERABLE = 36
    # Construction and Epic Structures
    CONSTRUCTION_SITE = 37  # Pre-placed buildable location
    EPIC_SHELTER = 38  # Built structure - faster rest regen
    EPIC_FORGE = 39  # Built structure - enables diamond crafting
    EPIC_BEACON = 40  # Built structure - permanent light radius
    CONSTRUCTION_IN_PROGRESS = 41  # For handover - partially built


class ItemType(Enum):
    NONE = 0
    TORCH = 1
    LADDER_DOWN = 2
    LADDER_UP = 3
    LADDER_DOWN_BLOCKED = 4


class Action(Enum):
    NOOP = 0  #
    LEFT = 1  # a
    RIGHT = 2  # d
    UP = 3  # w
    DOWN = 4  # s
    DO = 5  # space
    SLEEP = 6  # tab
    PLACE_STONE = 7  # r
    PLACE_TABLE = 8  # t
    PLACE_FURNACE = 9  # f
    PLACE_PLANT = 10  # p
    MAKE_WOOD_PICKAXE = 11  # 1
    MAKE_STONE_PICKAXE = 12  # 2
    MAKE_IRON_PICKAXE = 13  # 3
    MAKE_WOOD_SWORD = 14  # 5
    MAKE_STONE_SWORD = 15  # 6
    MAKE_IRON_SWORD = 16  # 7
    REST = 17  # e
    DESCEND = 18  # >
    ASCEND = 19  # <
    MAKE_DIAMOND_PICKAXE = 20  # 4
    MAKE_DIAMOND_SWORD = 21  # 8
    MAKE_IRON_ARMOUR = 22  # y
    MAKE_DIAMOND_ARMOUR = 23  # u
    SHOOT_ARROW = 24  # i
    MAKE_ARROW = 25  # o
    CAST_SPELL = 26  # g
    PLACE_TORCH = 27  # j
    DRINK_POTION_RED = 28  # z
    DRINK_POTION_GREEN = 29  # x
    DRINK_POTION_BLUE = 30  # c
    DRINK_POTION_PINK = 31  # v
    DRINK_POTION_CYAN = 32  # b
    DRINK_POTION_YELLOW = 33  # n
    READ_BOOK = 34  # m
    ENCHANT_SWORD = 35  # k
    ENCHANT_ARMOUR = 36  # l
    MAKE_TORCH = 37  # [
    LEVEL_UP_DEXTERITY = 38  # ]
    LEVEL_UP_STRENGTH = 39  # -
    LEVEL_UP_INTELLIGENCE = 40  # =
    ENCHANT_BOW = 41  # ;
    REQUEST_FOOD = 42  # Backspace
    REQUEST_DRINK = 43  # Back slash
    REQUEST_WOOD = 44  # Return
    REQUEST_STONE = 45  # Right Shift
    REQUEST_IRON = 46  # Up Arrow
    REQUEST_COAL = 47  # Down Arrow
    REQUEST_DIAMOND = 48  # Left Arrow
    REQUEST_RUBY = 49  # Left Arrow
    REQUEST_SAPPHIRE = 50  # Left Arrow
    # Construction Actions
    BUILD_SHELTER = 51
    BUILD_FORGE = 52
    BUILD_BEACON = 53
    GIVE = 54  # Needs to be last action since we generate more actions per agent!
    # Give is not just one action: there is one Give slot per OTHER teammate.
    # Slot 0 is action GIVE; slot 1+ uses subsequent action indices.
    # Example for slot 0:
    # agent_0 gives to agent_1
    # agent_1 gives to agent_0
    # agent_2 gives to agent_0


class MobType(Enum):
    PASSIVE = 0
    MELEE = 1
    RANGED = 2
    PROJECTILE = 3


class ProjectileType(Enum):
    ARROW = 0
    DAGGER = 1
    FIREBALL = 2
    ICEBALL = 3
    ARROW2 = 4
    SLIMEBALL = 5
    FIREBALL2 = 6
    ICEBALL2 = 7


class Specialization(Enum):
    UNASSIGNED = 0
    FORAGER = 1
    WARRIOR = 2
    MINER = 3


class DeathCause(int, Enum):
    """Cause of a player's death. Stored per-player on EnvState."""

    ALIVE = 0  # Player has not died
    STARVATION = 1  # Food reached 0, recovery ticked to -15
    DEHYDRATION = 2  # Drink reached 0, recovery ticked to -15
    EXHAUSTION = 3  # Energy reached 0, recovery ticked to -15
    MOB_COMBAT = 4  # Killed by a mob (melee or projectile)
    FRIENDLY_FIRE = 5  # Killed by another player (friendly fire enabled)


# FLOOR MECHANICS

FLOOR_MOB_MAPPING = jnp.array(
    [
        # (passive, melee, ranged)
        jnp.array([0, 0, 0]),  # Floor 0 (overworld)
        jnp.array([2, 2, 2]),  # Floor 1 (dungeon)
        jnp.array([1, 1, 1]),  # Floor 2 (gnomish mines)
        jnp.array([2, 3, 3]),  # Floor 3 (sewers)
        jnp.array([2, 4, 4]),  # Floor 4 (vaults)
        jnp.array([1, 5, 5]),  # Floor 5 (troll mines)
        jnp.array([1, 6, 6]),  # Floor 6 (fire)
        jnp.array([1, 7, 7]),  # Floor 7 (ice)
        jnp.array([0, 0, 0]),  # Floor 8 (boss)
    ],
    dtype=jnp.int32,
)


FLOOR_MOB_SPAWN_CHANCE = jnp.array(
    [
        # (passive, melee, ranged, melee-night)
        jnp.array([0.1, 0.02, 0.05, 0.1]),  # Floor 0 (overworld)
        jnp.array([0.1, 0.06, 0.05, 0.0]),  # Floor 1 (gnomish mines)
        jnp.array([0.1, 0.06, 0.05, 0.0]),  # Floor 2 (dungeon)
        jnp.array([0.1, 0.06, 0.05, 0.0]),  # Floor 3 (sewers)
        jnp.array([0.1, 0.06, 0.05, 0.0]),  # Floor 4 (vaults)
        jnp.array([0.1, 0.06, 0.05, 0.0]),  # Floor 5 (troll mines)
        jnp.array([0.1, 0.06, 0.05, 0.0]),  # Floor 6 (fire)
        jnp.array([0.0, 0.06, 0.05, 0.0]),  # Floor 7 (ice)
        jnp.array([0.1, 0.06, 0.05, 0.0]),  # Floor 8 (boss)
    ],
    dtype=jnp.float32,
)

# Path blocks, water, lava  (everything collides with solid blocks)
COLLISION_LAND_CREATURE = [False, True, True]
COLLISION_FLYING = [False, False, False]
COLLISION_AQUATIC = [True, False, True]
COLLISION_AMPHIBIAN = [False, False, True]


MOB_TYPE_COLLISION_MAPPING = jnp.array(
    [
        # (passive, melee, ranged, projectile)
        jnp.array(
            [
                COLLISION_LAND_CREATURE,
                COLLISION_LAND_CREATURE,
                COLLISION_LAND_CREATURE,
                COLLISION_FLYING,
            ]
        ),  # Floor 0 (overworld)
        jnp.array(
            [
                COLLISION_FLYING,
                COLLISION_LAND_CREATURE,
                COLLISION_LAND_CREATURE,
                COLLISION_FLYING,
            ]
        ),  # Floor 1 (gnomish mines)
        jnp.array(
            [
                COLLISION_LAND_CREATURE,
                COLLISION_LAND_CREATURE,
                COLLISION_LAND_CREATURE,
                COLLISION_FLYING,
            ]
        ),  # Floor 2 (dungeon)
        jnp.array(
            [
                COLLISION_LAND_CREATURE,
                COLLISION_AMPHIBIAN,
                COLLISION_LAND_CREATURE,
                COLLISION_FLYING,
            ]
        ),  # Floor 3 (sewers)
        jnp.array(
            [
                COLLISION_LAND_CREATURE,
                COLLISION_LAND_CREATURE,
                COLLISION_LAND_CREATURE,
                COLLISION_FLYING,
            ]
        ),  # Floor 4 (vaults)
        jnp.array(
            [
                COLLISION_LAND_CREATURE,
                COLLISION_LAND_CREATURE,
                COLLISION_AQUATIC,
                COLLISION_FLYING,
            ]
        ),  # Floor 5 (troll mines)
        jnp.array(
            [
                COLLISION_LAND_CREATURE,
                COLLISION_LAND_CREATURE,
                COLLISION_FLYING,
                COLLISION_FLYING,
            ]
        ),  # Floor 6 (fire)
        jnp.array(
            [
                COLLISION_LAND_CREATURE,
                COLLISION_LAND_CREATURE,
                COLLISION_FLYING,
                COLLISION_FLYING,
            ]
        ),  # Floor 7 (ice)
        jnp.array(
            [
                COLLISION_LAND_CREATURE,
                COLLISION_LAND_CREATURE,
                COLLISION_LAND_CREATURE,
                COLLISION_FLYING,
            ]
        ),  # Floor 8 (boss)
    ],
    dtype=jnp.int32,
)

NO_DAMAGE = jnp.array([0, 0, 0])
MOB_TYPE_DAMAGE_MAPPING = jnp.array(
    [
        # (-, melee, -, projectile)
        [NO_DAMAGE, [2, 0, 0], NO_DAMAGE, [2, 0, 0]],  # zombie, arrow
        [NO_DAMAGE, [4, 0, 0], NO_DAMAGE, [4, 0, 0]],  # gnome, dagger
        [NO_DAMAGE, [3, 0, 0], NO_DAMAGE, [0, 3, 0]],  # orc, fireball
        [NO_DAMAGE, [5, 0, 0], NO_DAMAGE, [0, 0, 3]],  # lizard, iceball
        [NO_DAMAGE, [6, 0, 0], NO_DAMAGE, [5, 0, 0]],  # knight, arrow2
        [NO_DAMAGE, [6, 1, 1], NO_DAMAGE, [4, 3, 3]],  # troll, slimeball
        [NO_DAMAGE, [3, 5, 0], NO_DAMAGE, [3, 5, 0]],  # pigman, fireball2
        [NO_DAMAGE, [4, 0, 5], NO_DAMAGE, [4, 0, 5]],  # ice troll, iceball2
    ],
    dtype=jnp.float32,
)

MOB_TYPE_HEALTH_MAPPING = jnp.array(
    [
        # (passive, melee, ranged, -)
        jnp.array([3, 5, 3, 0]),  # Floor 0 (overworld)
        jnp.array([4, 7, 5, 0]),  # Floor 1 (gnomish mines)
        jnp.array([6, 9, 6, 0]),  # Floor 2 (dungeon)
        jnp.array([8, 11, 8, 0]),  # Floor 3 (sewers)
        jnp.array([0, 12, 12, 0]),  # Floor 4 (vaults)
        jnp.array([0, 20, 4, 0]),  # Floor 5 (troll mines)
        jnp.array([0, 20, 14, 0]),  # Floor 6 (fire)
        jnp.array([0, 24, 16, 0]),  # Floor 7 (ice)
        jnp.array([0, 0, 0, 0]),  # Floor 8 (boss)
    ],
    dtype=jnp.float32,
)

NO_DEFENSE = [0, 0, 0]
MOB_TYPE_DEFENSE_MAPPING = jnp.array(
    [
        # (passive, melee, ranged, -)
        jnp.array([NO_DEFENSE, NO_DEFENSE, NO_DEFENSE, NO_DEFENSE]),  # Floor 0 (overworld)
        jnp.array([NO_DEFENSE, NO_DEFENSE, NO_DEFENSE, NO_DEFENSE]),  # Floor 1 (gnomish mines)
        jnp.array([NO_DEFENSE, NO_DEFENSE, NO_DEFENSE, NO_DEFENSE]),  # Floor 2 (dungeon)
        jnp.array([NO_DEFENSE, NO_DEFENSE, NO_DEFENSE, NO_DEFENSE]),  # Floor 3 (sewers)
        jnp.array([NO_DEFENSE, [0.5, 0, 0], [0.5, 0, 0], NO_DEFENSE]),  # Floor 4 (vaults)
        jnp.array([NO_DEFENSE, [0.2, 0, 0], [0.0, 0.0, 0.0], NO_DEFENSE]),  # Floor 5 (troll mines)
        jnp.array([NO_DEFENSE, [0.9, 1.0, 0.0], [0.9, 1.0, 0.0], NO_DEFENSE]),  # Floor 6 (fire)
        jnp.array([NO_DEFENSE, [0.9, 0.0, 1.0], [0.9, 0.0, 1.0], NO_DEFENSE]),  # Floor 7 (ice)
        jnp.array([NO_DEFENSE, NO_DEFENSE, NO_DEFENSE, NO_DEFENSE]),  # Floor 8 (boss)
    ],
    dtype=jnp.float32,
)

RANGED_MOB_TYPE_TO_PROJECTILE_TYPE_MAPPING = jnp.array(
    [
        0,  # Skeleton --> Arrow
        0,  # Gnome archer --> Arrow
        2,  # Orc mage --> Fireball
        1,  # Kobold --> Dagger
        4,  # Knight archer --> Arrow2
        5,  # Deep thing --> Slime ball
        6,  # Fire elemental --> Fireball2
        7,  # Ice elemental --> Iceball2
    ]
)


# GAME MECHANICS
MONSTERS_KILLED_TO_CLEAR_LEVEL = 8
BOSS_FIGHT_EXTRA_DAMAGE = 0.5
BOSS_FIGHT_SPAWN_TURNS = 7

DIRECTIONS = jnp.concatenate(
    (
        jnp.array([[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]], dtype=jnp.int32),
        jnp.zeros((11, 2), dtype=jnp.int32),
    ),
    axis=0,
)

CLOSE_BLOCKS = jnp.array(
    [
        [0, -1],
        [0, 1],
        [-1, 0],
        [1, 0],
        [-1, -1],
        [-1, 1],
        [1, -1],
        [1, 1],
    ],
    dtype=jnp.int32,
)

# Can't walk through these
SOLID_BLOCKS = [
    BlockType.STONE.value,
    BlockType.TREE.value,
    BlockType.COAL.value,
    BlockType.IRON.value,
    BlockType.DIAMOND.value,
    BlockType.CRAFTING_TABLE.value,
    BlockType.FURNACE.value,
    BlockType.PLANT.value,
    BlockType.RIPE_PLANT.value,
    BlockType.WALL.value,
    BlockType.WALL_MOSS.value,
    BlockType.STALAGMITE.value,
    BlockType.RUBY.value,
    BlockType.SAPPHIRE.value,
    BlockType.CHEST.value,
    BlockType.FOUNTAIN.value,
    BlockType.FIRE_TREE.value,
    BlockType.ICE_SHRUB.value,
    BlockType.ENCHANTMENT_TABLE_FIRE.value,
    BlockType.ENCHANTMENT_TABLE_ICE.value,
    BlockType.GRAVE.value,
    BlockType.GRAVE2.value,
    BlockType.GRAVE3.value,
    BlockType.NECROMANCER.value,
    BlockType.NECROMANCER_VULNERABLE.value,
    BlockType.CONSTRUCTION_SITE.value,
    BlockType.EPIC_SHELTER.value,
    BlockType.EPIC_FORGE.value,
    BlockType.EPIC_BEACON.value,
    BlockType.CONSTRUCTION_IN_PROGRESS.value,
]

SOLID_BLOCK_MAPPING = jnp.array([(block.value in SOLID_BLOCKS) for block in BlockType], dtype=bool)

CAN_PLACE_ITEM_BLOCKS = [
    BlockType.GRASS.value,
    BlockType.SAND.value,
    BlockType.PATH.value,
    BlockType.FIRE_GRASS.value,
    BlockType.ICE_GRASS.value,
]

CAN_PLACE_ITEM_MAPPING = jnp.array(
    [(block.value in CAN_PLACE_ITEM_BLOCKS) for block in BlockType], dtype=bool
)


# ACHIEVEMENTS
class Achievement(Enum):
    # ── Basic (reward=1) ── matches original Craftax 0-25
    COLLECT_WOOD = 0
    PLACE_TABLE = 1
    EAT_COW = 2
    COLLECT_SAPLING = 3
    COLLECT_DRINK = 4
    COLLECT_FOOD = 5
    MAKE_WOOD_PICKAXE = 6
    MAKE_WOOD_SWORD = 7
    PLACE_PLANT = 8
    DEFEAT_ZOMBIE = 9
    COLLECT_STONE = 10
    PLACE_STONE = 11
    EAT_PLANT = 12
    DEFEAT_SKELETON = 13
    MAKE_STONE_PICKAXE = 14
    MAKE_STONE_SWORD = 15
    WAKE_UP = 16
    PLACE_FURNACE = 17
    COLLECT_COAL = 18
    COLLECT_IRON = 19
    COLLECT_DIAMOND = 20
    MAKE_IRON_PICKAXE = 21
    MAKE_IRON_SWORD = 22
    MAKE_ARROW = 23
    MAKE_TORCH = 24
    PLACE_TORCH = 25

    # ── Intermediate (reward=3)
    COLLECT_SAPPHIRE = 26
    COLLECT_RUBY = 27
    MAKE_DIAMOND_PICKAXE = 28
    MAKE_DIAMOND_SWORD = 29
    MAKE_IRON_ARMOUR = 30
    MAKE_DIAMOND_ARMOUR = 31
    ENTER_GNOMISH_MINES = 32
    ENTER_DUNGEON = 33
    DEFEAT_GNOME_WARRIOR = 34
    DEFEAT_GNOME_ARCHER = 35
    DEFEAT_ORC_SOLIDER = 36
    DEFEAT_ORC_MAGE = 37
    EAT_BAT = 38
    EAT_SNAIL = 39
    FIND_BOW = 40
    FIRE_BOW = 41
    OPEN_CHEST = 42
    DRINK_POTION = 43

    # ── Advanced (reward=5)
    ENTER_SEWERS = 44
    ENTER_VAULT = 45
    ENTER_TROLL_MINES = 46
    DEFEAT_LIZARD = 47
    DEFEAT_KOBOLD = 48
    DEFEAT_KNIGHT = 49
    DEFEAT_ARCHER = 50
    DEFEAT_TROLL = 51
    DEFEAT_DEEP_THING = 52
    LEARN_SPELL = 53
    CAST_SPELL = 54
    ENCHANT_SWORD = 55
    ENCHANT_ARMOUR = 56

    # ── Very Advanced (reward=8)
    ENTER_FIRE_REALM = 57
    ENTER_ICE_REALM = 58
    ENTER_GRAVEYARD = 59
    DEFEAT_PIGMAN = 60
    DEFEAT_FIRE_ELEMENTAL = 61
    DEFEAT_FROST_TROLL = 62
    DEFEAT_ICE_ELEMENTAL = 63
    DAMAGE_NECROMANCER = 64
    DEFEAT_NECROMANCER = 65

    # ── Coordination achievements ──
    # Intermediate (reward=3)
    COORD_2_AGENTS_SOFT = 66
    COORD_LARGE_PASSIVE_KILL = 67
    COORD_MINE_STONE_SOFT = 72
    COORD_MINE_STONE_HARD = 73
    COORD_MINE_COAL_SOFT = 74
    COORD_MINE_COAL_HARD = 75

    # Advanced (reward=5)
    COORD_2_AGENTS_HARD = 68
    COORD_3_AGENTS_SOFT = 69  # 3+ agents
    HANDOVER_COMPLETE = 70
    COORD_MINE_HANDOVER = 71
    COORD_MINE_IRON_SOFT = 76
    COORD_MINE_IRON_HARD = 77
    COORD_MINE_DIAMOND_SOFT = 78
    COORD_MINE_DIAMOND_HARD = 79
    COORD_BUILD_SHELTER = 84

    # Very Advanced (reward=8)
    COORD_MINE_SAPPHIRE_SOFT = 80
    COORD_MINE_SAPPHIRE_HARD = 81
    COORD_MINE_RUBY_SOFT = 82
    COORD_MINE_RUBY_HARD = 83
    COORD_3_AGENTS_HARD = 85  # 3+ agents
    COORD_BUILD_FORGE = 86
    COORD_BUILD_BEACON = 87
    COORD_DIAMOND_PICKAXE = 88
    COORD_DIAMOND_SWORD = 89
    COORD_DIAMOND_ARMOUR = 90
    COORD_ELITE_MELEE_KILL = 91
    COORD_ELITE_RANGED_KILL = 92


NUM_ACHIEVEMENTS = len(Achievement)


BASIC_ACHIEVEMENTS = {  # reward=1, match original Craftax 0-25
    Achievement.COLLECT_WOOD.value,
    Achievement.PLACE_TABLE.value,
    Achievement.EAT_COW.value,
    Achievement.COLLECT_SAPLING.value,
    Achievement.COLLECT_DRINK.value,
    Achievement.COLLECT_FOOD.value,
    Achievement.MAKE_WOOD_PICKAXE.value,
    Achievement.MAKE_WOOD_SWORD.value,
    Achievement.PLACE_PLANT.value,
    Achievement.DEFEAT_ZOMBIE.value,
    Achievement.COLLECT_STONE.value,
    Achievement.PLACE_STONE.value,
    Achievement.EAT_PLANT.value,
    Achievement.DEFEAT_SKELETON.value,
    Achievement.MAKE_STONE_PICKAXE.value,
    Achievement.MAKE_STONE_SWORD.value,
    Achievement.WAKE_UP.value,
    Achievement.PLACE_FURNACE.value,
    Achievement.COLLECT_COAL.value,
    Achievement.COLLECT_IRON.value,
    Achievement.COLLECT_DIAMOND.value,
    Achievement.MAKE_IRON_PICKAXE.value,
    Achievement.MAKE_IRON_SWORD.value,
    Achievement.MAKE_ARROW.value,
    Achievement.MAKE_TORCH.value,
    Achievement.PLACE_TORCH.value,
}

INTERMEDIATE_ACHIEVEMENTS = {  # reward=3
    Achievement.COLLECT_SAPPHIRE.value,
    Achievement.COLLECT_RUBY.value,
    Achievement.MAKE_DIAMOND_PICKAXE.value,
    Achievement.MAKE_DIAMOND_SWORD.value,
    Achievement.MAKE_IRON_ARMOUR.value,
    Achievement.MAKE_DIAMOND_ARMOUR.value,
    Achievement.ENTER_GNOMISH_MINES.value,
    Achievement.ENTER_DUNGEON.value,
    Achievement.DEFEAT_GNOME_WARRIOR.value,
    Achievement.DEFEAT_GNOME_ARCHER.value,
    Achievement.DEFEAT_ORC_SOLIDER.value,
    Achievement.DEFEAT_ORC_MAGE.value,
    Achievement.EAT_BAT.value,
    Achievement.EAT_SNAIL.value,
    Achievement.FIND_BOW.value,
    Achievement.FIRE_BOW.value,
    Achievement.OPEN_CHEST.value,
    Achievement.DRINK_POTION.value,
    Achievement.COORD_2_AGENTS_SOFT.value,
    Achievement.COORD_LARGE_PASSIVE_KILL.value,
    Achievement.COORD_MINE_STONE_SOFT.value,
    Achievement.COORD_MINE_STONE_HARD.value,
    Achievement.COORD_MINE_COAL_SOFT.value,
    Achievement.COORD_MINE_COAL_HARD.value,
}

ADVANCED_ACHIEVEMENTS = {  # reward=5
    Achievement.ENTER_SEWERS.value,
    Achievement.ENTER_VAULT.value,
    Achievement.ENTER_TROLL_MINES.value,
    Achievement.DEFEAT_LIZARD.value,
    Achievement.DEFEAT_KOBOLD.value,
    Achievement.DEFEAT_KNIGHT.value,
    Achievement.DEFEAT_ARCHER.value,
    Achievement.DEFEAT_TROLL.value,
    Achievement.DEFEAT_DEEP_THING.value,
    Achievement.LEARN_SPELL.value,
    Achievement.CAST_SPELL.value,
    Achievement.ENCHANT_SWORD.value,
    Achievement.ENCHANT_ARMOUR.value,
    Achievement.COORD_2_AGENTS_HARD.value,
    Achievement.COORD_3_AGENTS_SOFT.value,
    Achievement.HANDOVER_COMPLETE.value,
    Achievement.COORD_MINE_HANDOVER.value,
    Achievement.COORD_MINE_IRON_SOFT.value,
    Achievement.COORD_MINE_IRON_HARD.value,
    Achievement.COORD_MINE_DIAMOND_SOFT.value,
    Achievement.COORD_MINE_DIAMOND_HARD.value,
    Achievement.COORD_BUILD_SHELTER.value,
}

VERY_ADVANCED_ACHIEVEMENTS = {  # reward=8
    Achievement.ENTER_FIRE_REALM.value,
    Achievement.ENTER_ICE_REALM.value,
    Achievement.ENTER_GRAVEYARD.value,
    Achievement.DEFEAT_PIGMAN.value,
    Achievement.DEFEAT_FIRE_ELEMENTAL.value,
    Achievement.DEFEAT_FROST_TROLL.value,
    Achievement.DEFEAT_ICE_ELEMENTAL.value,
    Achievement.DAMAGE_NECROMANCER.value,
    Achievement.DEFEAT_NECROMANCER.value,
    Achievement.COORD_3_AGENTS_HARD.value,
    Achievement.COORD_BUILD_FORGE.value,
    Achievement.COORD_BUILD_BEACON.value,
    Achievement.COORD_DIAMOND_PICKAXE.value,
    Achievement.COORD_DIAMOND_SWORD.value,
    Achievement.COORD_DIAMOND_ARMOUR.value,
    Achievement.COORD_ELITE_MELEE_KILL.value,
    Achievement.COORD_ELITE_RANGED_KILL.value,
    Achievement.COORD_MINE_SAPPHIRE_SOFT.value,
    Achievement.COORD_MINE_SAPPHIRE_HARD.value,
    Achievement.COORD_MINE_RUBY_SOFT.value,
    Achievement.COORD_MINE_RUBY_HARD.value,
}


def achievement_mapping(achievement_value):
    """Map an achievement identifier to its scalar reward.

    Args:
        achievement_value: Integer value of an ``Achievement`` member.

    Returns:
        Reward assigned to the achievement's difficulty tier.
    """
    if achievement_value in VERY_ADVANCED_ACHIEVEMENTS:
        return 8
    elif achievement_value in ADVANCED_ACHIEVEMENTS:
        return 5
    elif achievement_value in INTERMEDIATE_ACHIEVEMENTS:
        return 3
    else:
        return 1


ACHIEVEMENT_REWARD_MAP = jnp.array([achievement_mapping(i) for i in range(NUM_ACHIEVEMENTS)])


LEVEL_ACHIEVEMENT_MAP = jnp.array(
    [
        0,
        Achievement.ENTER_DUNGEON.value,
        Achievement.ENTER_GNOMISH_MINES.value,
        Achievement.ENTER_SEWERS.value,
        Achievement.ENTER_VAULT.value,
        Achievement.ENTER_TROLL_MINES.value,
        Achievement.ENTER_FIRE_REALM.value,
        Achievement.ENTER_ICE_REALM.value,
        Achievement.ENTER_GRAVEYARD.value,
    ]
)

MOB_ACHIEVEMENT_MAP = jnp.array(
    [
        # Passive
        [
            Achievement.EAT_COW.value,
            Achievement.EAT_BAT.value,
            Achievement.EAT_SNAIL.value,
            0,
            0,
            0,
            0,
            0,
        ],
        # Melee
        [
            Achievement.DEFEAT_ZOMBIE.value,
            Achievement.DEFEAT_GNOME_WARRIOR.value,
            Achievement.DEFEAT_ORC_SOLIDER.value,
            Achievement.DEFEAT_LIZARD.value,
            Achievement.DEFEAT_KNIGHT.value,
            Achievement.DEFEAT_TROLL.value,
            Achievement.DEFEAT_PIGMAN.value,
            Achievement.DEFEAT_FROST_TROLL.value,
        ],
        # Ranged
        [
            Achievement.DEFEAT_SKELETON.value,
            Achievement.DEFEAT_GNOME_ARCHER.value,
            Achievement.DEFEAT_ORC_MAGE.value,
            Achievement.DEFEAT_KOBOLD.value,
            Achievement.DEFEAT_ARCHER.value,
            Achievement.DEFEAT_DEEP_THING.value,
            Achievement.DEFEAT_FIRE_ELEMENTAL.value,
            Achievement.DEFEAT_ICE_ELEMENTAL.value,
        ],
    ]
)

# CONSTRUCTION MATERIAL COSTS
SHELTER_COST_WOOD = 10
SHELTER_COST_STONE = 5

FORGE_COST_STONE = 10
FORGE_COST_IRON = 3
FORGE_COST_COAL = 2

BEACON_COST_IRON = 3
BEACON_COST_COAL = 2

# PRE-COMPUTATION
TORCH_LIGHT_MAP = get_distance_map(jnp.array([4, 4]), (9, 9))
TORCH_LIGHT_MAP /= 5.0
TORCH_LIGHT_MAP = jnp.clip(1 - TORCH_LIGHT_MAP, 0.0, 1.0)

BEACON_LIGHT_MAP = get_distance_map(jnp.array([4, 4]), (9, 9))
BEACON_LIGHT_MAP = jnp.clip(1.0 - BEACON_LIGHT_MAP / 4.0, 0.0, 1.0)


# TEXTURES
@struct.dataclass
class PlayerSpecificTextures:
    player_textures: jnp.ndarray
    player_icon_textures: jnp.ndarray
    chest_textures: jnp.ndarray


def load_texture(filename, block_pixel_size):
    """Load an asset and resize it to the requested pixel grid.

    Args:
        filename: Asset filename relative to the package's asset directory.
        block_pixel_size: Width and height of the returned square texture.

    Returns:
        Integer RGB or RGBA texture array.
    """
    filename = os.path.join(pathlib.Path(__file__).parent.resolve(), "assets", filename)
    img = iio.imread(filename)
    jnp_img = jnp.array(img).astype(int)
    assert jnp_img.shape[:2] == (16, 16)

    if jnp_img.shape[2] == 4:
        jnp_img = jnp_img.at[:, :, 3].set(jnp_img[:, :, 3] // 255)

    if block_pixel_size != 16:
        img = np.array(jnp_img, dtype=np.uint8)
        image = Image.fromarray(img)
        image = image.resize((block_pixel_size, block_pixel_size), resample=Image.NEAREST)
        jnp_img = jnp.array(image, dtype=jnp.int32)

    return jnp_img


def apply_alpha(texture):
    """Premultiply an RGBA texture's RGB channels by its alpha channel.

    Args:
        texture: RGBA texture array.

    Returns:
        RGB texture with transparent pixels suppressed.
    """
    return texture[:, :, :3] * jnp.repeat(jnp.expand_dims(texture[:, :, 3], axis=-1), 3, axis=-1)


def load_player_specific_textures(texture_set, player_count) -> PlayerSpecificTextures:
    """Colorize player, portrait, and chest textures for each player.

    Args:
        texture_set: Base texture dictionary for one pixel size.
        player_count: Number of distinct player palettes to generate.

    Returns:
        Batched textures keyed implicitly by player index.
    """
    color_palette = (jnp.array(husl_palette(player_count, h=0.5, l=0.5)) * 255).astype(jnp.uint32)
    return PlayerSpecificTextures(
        player_textures=load_multiplayer_textures(
            texture_set["player_textures"], color_palette, player_count
        ),
        player_icon_textures=load_multiplayer_textures(
            texture_set["player_icon_textures"], color_palette, player_count
        )[:, :, :, :, :3],
        chest_textures=load_colored_block_textures(
            texture_set["full_map_block_textures"][BlockType.CHEST.value],
            color_palette,
            player_count,
        ),
    )


def load_multiplayer_textures(base_textures, color_palette, player_count):
    """Replace marker pixels in a texture batch with player colors.

    Args:
        base_textures: Base RGBA textures containing color marker pixels.
        color_palette: One RGB color per player.
        player_count: Number of player-specific batches to produce.

    Returns:
        RGBA textures with a leading player dimension.
    """
    color_palette = jnp.concatenate([color_palette, jnp.ones((player_count, 1))], axis=-1)
    colors_broadcasted = color_palette[:, None, None, None, :]
    multiplayer_textures = base_textures[None, :].repeat(player_count, 0)
    mask = (multiplayer_textures == jnp.array([0, 0, 0, 1])).all(axis=-1)[..., None]
    multiplayer_textures_colored = jnp.where(mask, colors_broadcasted, multiplayer_textures)
    return multiplayer_textures_colored


def load_colored_block_textures(base_textures, color_palette, player_count):
    """Create player-colored variants of an RGB block texture.

    Args:
        base_textures: RGB block texture containing black marker pixels.
        color_palette: One RGB color per player.
        player_count: Number of variants to produce.

    Returns:
        RGB block textures with a leading player dimension.
    """
    colors_broadcasted = color_palette[:, None, None, :]
    multiplayer_textures = base_textures[None, :].repeat(player_count, 0)
    mask = (multiplayer_textures == jnp.array([0, 0, 0])).all(axis=-1)[..., None]
    multiplayer_textures_colored = jnp.where(mask, colors_broadcasted, multiplayer_textures)
    return multiplayer_textures_colored


def load_mob_texture_set(filenames, block_pixel_size):
    """Load mob textures and their broadcast RGB alpha masks.

    Args:
        filenames: Asset filenames for the mob variants.
        block_pixel_size: Width and height of each returned texture.

    Returns:
        A pair containing RGB textures and matching alpha masks.
    """
    textures = np.zeros((len(filenames), block_pixel_size, block_pixel_size, 3))
    texture_alphas = np.zeros((len(filenames), block_pixel_size, block_pixel_size, 3))

    for file_index, filename in enumerate(filenames):
        rgba_img = jnp.array(load_texture(filename, block_pixel_size))
        texture = apply_alpha(rgba_img)
        texture_alpha = np.repeat(np.expand_dims(rgba_img[:, :, 3], axis=-1), repeats=3, axis=2)

        textures[file_index] = texture
        texture_alphas[file_index] = texture_alpha

    return jnp.array(textures), jnp.array(texture_alphas)


def load_request_message_textures(block_pixel_size):
    """Build speech-bubble textures for every requestable resource.

    Args:
        block_pixel_size: Width and height of each composed texture.

    Returns:
        A texture batch ordered by request action resource.
    """
    icon_pixel_size = int(block_pixel_size * 0.6)
    start_loc_x = (block_pixel_size - icon_pixel_size) // 2
    start_loc_y = (block_pixel_size - icon_pixel_size) // 3
    message_bubble_texture = load_texture("message_bubble.png", block_pixel_size)

    def _overlay_item(icon_texture):
        combined_message_texture = message_bubble_texture

        # Only for areas where the icon is not transparent overlay the icon
        if icon_texture.shape[-1] == 4:
            original_slice = combined_message_texture[
                start_loc_y : start_loc_y + icon_pixel_size,
                start_loc_x : start_loc_x + icon_pixel_size,
                :3,
            ]
            updated_slice = jnp.where(
                (icon_texture[:, :, 3] == 1)[:, :, None], icon_texture[:, :, :3], original_slice
            )
        else:
            updated_slice = icon_texture

        combined_message_texture = combined_message_texture.at[
            start_loc_y : start_loc_y + icon_pixel_size,
            start_loc_x : start_loc_x + icon_pixel_size,
            :3,
        ].set(updated_slice)
        return combined_message_texture

    item_name_list = [
        "food.png",
        "drink.png",
        "wood.png",
        "stone.png",
        "iron.png",
        "coal.png",
        "diamond.png",
        "ruby.png",
        "sapphire.png",
    ]
    return jnp.array([_overlay_item(load_texture(f, icon_pixel_size)) for f in item_name_list])


def load_comm_badge_texture(block_pixel_size):
    """Build the speech-bubble overlay used for communication cues.

    Args:
        block_pixel_size: Width and height of the destination portrait.

    Returns:
        The overlay canvas, its x/y offsets, and the badge size.
    """
    badge_pixel_size = max(int(block_pixel_size * 0.48), 6)
    badge_texture = load_texture("message_bubble.png", badge_pixel_size)
    badge_canvas = jnp.zeros((block_pixel_size, block_pixel_size, 4), dtype=jnp.int32)
    start_y = 1
    start_x = max(0, block_pixel_size - badge_pixel_size - 1)
    badge_canvas = badge_canvas.at[
        start_y : start_y + badge_pixel_size, start_x : start_x + badge_pixel_size, :
    ].set(badge_texture)
    return badge_canvas, start_x, start_y, badge_pixel_size


def load_all_textures(block_pixel_size):
    """Load and compose every renderer texture for one pixel size.

    Args:
        block_pixel_size: Width and height of a rendered map tile.

    Returns:
        Texture dictionary consumed by the pixel renderers.
    """
    small_block_pixel_size = int(block_pixel_size * 0.8)

    # Blocks
    block_texture_names = [
        "debug_tile.png",
        "debug_tile.png",
        "grass.png",
        "water.png",
        "stone.png",
        "tree.png",
        "wood.png",
        "path.png",
        "coal.png",
        "iron.png",
        "diamond.png",
        "table.png",
        "furnace.png",
        "sand.png",
        "lava.png",
        "plant_on_grass.png",
        "ripe_plant_on_grass.png",
        "wall2.png",
        "debug_tile.png",
        "wall_moss.png",
        "stalagmite.png",
        "sapphire.png",
        "ruby.png",
        "chest.png",
        "fountain.png",
        "fire_grass.png",
        "ice_grass.png",
        "gravel.png",
        "fire_tree.png",
        "ice_shrub.png",
        "enchantment_table_fire.png",
        "enchantment_table_ice.png",
        "necromancer.png",
        "grave.png",
        "grave2.png",
        "grave3.png",
        "necromancer_vulnerable.png",
        # Construction and Epic Structures (indices 37-41)
        "construction_site.png",  # CONSTRUCTION_SITE = 37
        "epic_shelter.png",  # EPIC_SHELTER = 38
        "epic_forge.png",  # EPIC_FORGE = 39
        "epic_beacon.png",  # EPIC_BEACON = 40
        "construction_in_progress.png",  # CONSTRUCTION_IN_PROGRESS = 41
    ]

    # Some structure sprites contain transparency and are intended to be rendered
    # over terrain. Blend those over grass at load time to avoid dark matte boxes.
    alpha_blended_over_grass = {
        BlockType.CONSTRUCTION_SITE.value,
        BlockType.EPIC_SHELTER.value,
        BlockType.EPIC_FORGE.value,
        BlockType.EPIC_BEACON.value,
        BlockType.CONSTRUCTION_IN_PROGRESS.value,
    }

    def _load_block_rgb(fname, size, idx, grass_rgb):
        tex = load_texture(fname, size)
        rgb = tex[:, :, :3].astype(jnp.float32)
        if tex.shape[-1] == 4 and idx in alpha_blended_over_grass:
            alpha = tex[:, :, 3:4].astype(jnp.float32)
            rgb = rgb * alpha + grass_rgb * (1.0 - alpha)
            alpha_mask = alpha[:, :, 0] > 0
            padded = jnp.pad(alpha_mask, ((1, 1), (1, 1)), constant_values=False)
            edge_mask = alpha_mask & (
                (~padded[:-2, 1:-1])
                | (~padded[2:, 1:-1])
                | (~padded[1:-1, :-2])
                | (~padded[1:-1, 2:])
            )
            brightness = rgb.mean(axis=-1)
            edge_soften = jnp.clip((110.0 - brightness) / 110.0, 0.0, 1.0) * edge_mask.astype(
                jnp.float32
            )
            edge_soften = edge_soften[:, :, None] * 0.78
            rgb = rgb * (1.0 - edge_soften) + grass_rgb * edge_soften
        return rgb.astype(jnp.int32)

    grass_texture = load_texture(block_texture_names[BlockType.GRASS.value], block_pixel_size)
    grass_rgb = grass_texture[:, :, :3].astype(jnp.float32)
    block_textures = jnp.array(
        [
            _load_block_rgb(fname, block_pixel_size, i, grass_rgb)
            for i, fname in enumerate(block_texture_names)
        ]
    )
    transparent_block_textures = []
    transparent_block_texture_alphas = []
    for i, fname in enumerate(block_texture_names):
        tex = load_texture(fname, block_pixel_size)
        if tex.shape[-1] == 4 and i in alpha_blended_over_grass:
            transparent_block_textures.append(tex[:, :, :3].astype(jnp.float32))
            transparent_block_texture_alphas.append(
                jnp.repeat(tex[:, :, 3:4].astype(jnp.float32), 3, axis=-1)
            )
        else:
            transparent_block_textures.append(
                jnp.zeros((block_pixel_size, block_pixel_size, 3), dtype=jnp.float32)
            )
            transparent_block_texture_alphas.append(
                jnp.zeros((block_pixel_size, block_pixel_size, 3), dtype=jnp.float32)
            )
    transparent_block_textures = jnp.array(transparent_block_textures)
    transparent_block_texture_alphas = jnp.array(transparent_block_texture_alphas)

    # Manually set some textures
    block_textures = block_textures.at[BlockType.OUT_OF_BOUNDS.value].set(
        jnp.ones((block_pixel_size, block_pixel_size, 3), dtype=jnp.int32) * 128
    )
    block_textures = block_textures.at[BlockType.DARKNESS.value].set(
        jnp.zeros((block_pixel_size, block_pixel_size, 3), dtype=jnp.int32)
    )

    small_grass_texture = load_texture(
        block_texture_names[BlockType.GRASS.value], small_block_pixel_size
    )
    small_grass_rgb = small_grass_texture[:, :, :3].astype(jnp.float32)
    smaller_block_textures = jnp.array(
        [
            _load_block_rgb(fname, small_block_pixel_size, i, small_grass_rgb)
            for i, fname in enumerate(block_texture_names)
        ]
    )

    full_map_block_textures = jnp.array(
        [jnp.tile(block_textures[block.value], (*OBS_DIM, 1)) for block in BlockType]
    )

    # Items (torches, ladders)
    item_texture_names = [
        "debug.png",
        "torch_in_inventory.png",
        "ladder_down.png",
        "ladder_up.png",
        "ladder_down_blocked.png",
    ]

    item_textures = jnp.array(
        [load_texture(fname, block_pixel_size) for fname in item_texture_names]
    )
    full_map_item_textures = jnp.array(
        [jnp.tile(item_textures[item.value], (*OBS_DIM, 1)) for item in ItemType]
    )

    # Player
    pad_pixels = (
        (OBS_DIM[0] // 2) * block_pixel_size,
        (OBS_DIM[1] // 2) * block_pixel_size,
    )

    player_textures = jnp.array(
        [
            load_texture("player-left.png", block_pixel_size),
            load_texture("player-right.png", block_pixel_size),
            load_texture("player-up.png", block_pixel_size),
            load_texture("player-down.png", block_pixel_size),
            load_texture("player-sleep.png", block_pixel_size),
            load_texture("player-dead.png", block_pixel_size),
        ]
    )

    player_icon_textures = jnp.array(
        [
            load_texture("player.png", small_block_pixel_size),
            load_texture("player-dead.png", small_block_pixel_size),
        ]
    )

    full_map_player_textures_rgba = [
        jnp.pad(
            player_texture,
            ((pad_pixels[0], pad_pixels[0]), (pad_pixels[1], pad_pixels[1]), (0, 0)),
        )
        for player_texture in player_textures
    ]

    full_map_player_textures = jnp.array(
        [player_texture[:, :, :3] for player_texture in full_map_player_textures_rgba]
    )

    full_map_player_textures_alpha = jnp.array(
        [
            jnp.repeat(jnp.expand_dims(player_texture[:, :, 3], axis=-1), repeats=3, axis=2)
            for player_texture in full_map_player_textures_rgba
        ]
    )

    # Teammate directions
    def _generate_all_direction_textures(horizontal_texture_base, diagonal_texture_base):
        right = horizontal_texture_base
        up = jnp.rot90(right, k=1)
        left = jnp.rot90(right, k=2)
        down = jnp.rot90(right, k=3)
        top_right = diagonal_texture_base
        top_left = jnp.rot90(top_right, k=1)
        bottom_left = jnp.rot90(top_right, k=2)
        bottom_right = jnp.rot90(top_right, k=3)
        return jnp.array(
            [
                [top_left, up, top_right],
                [left, left, right],
                [bottom_left, down, bottom_right],
            ]
        )

    direction_texture_base = load_texture("pointer-right.png", small_block_pixel_size)
    direction_diagonal_texture_base = load_texture("pointer-top-right.png", small_block_pixel_size)
    direction_textures = _generate_all_direction_textures(
        direction_texture_base, direction_diagonal_texture_base
    )

    # inventory

    empty_texture = jnp.zeros((block_pixel_size, block_pixel_size, 3), dtype=jnp.int32)
    smaller_empty_texture = jnp.zeros(
        (small_block_pixel_size, small_block_pixel_size, 3), dtype=jnp.int32
    )

    ones_texture = jnp.ones((block_pixel_size, block_pixel_size, 3), dtype=jnp.int32)

    number_size = int(block_pixel_size * 0.4)

    number_textures_rgba = [
        jnp.zeros((number_size, number_size, 3), dtype=jnp.int32),
        load_texture("1.png", number_size),
        load_texture("2.png", number_size),
        load_texture("3.png", number_size),
        load_texture("4.png", number_size),
        load_texture("5.png", number_size),
        load_texture("6.png", number_size),
        load_texture("7.png", number_size),
        load_texture("8.png", number_size),
        load_texture("9.png", number_size),
    ]

    number_textures = jnp.array(
        [
            number_texture[:, :, :3]
            * jnp.repeat(jnp.expand_dims(number_texture[:, :, 3], axis=-1), 3, axis=-1)
            for number_texture in number_textures_rgba
        ]
    )

    number_textures_alpha = jnp.array(
        [
            jnp.repeat(jnp.expand_dims(number_texture[:, :, 3], axis=-1), repeats=3, axis=2)
            for number_texture in number_textures_rgba
        ]
    )

    number_textures_with_zero_rgba = [
        load_texture("0.png", number_size),
        load_texture("1.png", number_size),
        load_texture("2.png", number_size),
        load_texture("3.png", number_size),
        load_texture("4.png", number_size),
        load_texture("5.png", number_size),
        load_texture("6.png", number_size),
        load_texture("7.png", number_size),
        load_texture("8.png", number_size),
        load_texture("9.png", number_size),
    ]

    number_textures_with_zero = jnp.array(
        [
            number_texture[:, :, :3]
            * jnp.repeat(jnp.expand_dims(number_texture[:, :, 3], axis=-1), 3, axis=-1)
            for number_texture in number_textures_with_zero_rgba
        ]
    )

    number_textures_with_zero_alpha = jnp.array(
        [
            jnp.repeat(jnp.expand_dims(number_texture[:, :, 3], axis=-1), repeats=3, axis=2)
            for number_texture in number_textures_with_zero_rgba
        ]
    )

    comm_number_size = max(int(small_block_pixel_size * 0.22), 6)
    comm_number_textures_rgba = [
        jnp.zeros((comm_number_size, comm_number_size, 4), dtype=jnp.int32),
        load_texture("1.png", comm_number_size),
        load_texture("2.png", comm_number_size),
        load_texture("3.png", comm_number_size),
        load_texture("4.png", comm_number_size),
        load_texture("5.png", comm_number_size),
        load_texture("6.png", comm_number_size),
        load_texture("7.png", comm_number_size),
        load_texture("8.png", comm_number_size),
        load_texture("9.png", comm_number_size),
    ]

    comm_number_textures_with_zero = jnp.array(
        [
            number_texture[:, :, :3]
            * jnp.repeat(jnp.expand_dims(number_texture[:, :, 3], axis=-1), 3, axis=-1)
            for number_texture in comm_number_textures_rgba
        ]
    )

    comm_number_textures_with_zero_alpha = jnp.array(
        [
            jnp.repeat(jnp.expand_dims(number_texture[:, :, 3], axis=-1), repeats=3, axis=2)
            for number_texture in comm_number_textures_rgba
        ]
    )

    # COORDINATION TEXTURES - Icons for V1 border rendering style
    # small coordination icons for showing number of agents required for coordination, handover, and player icons in coordination overlay
    # used in pixel oberservations - 22% of a block's pixel size
    coord_icon_size = int(block_pixel_size * 0.22)

    # Number textures (2, 3, 4) for showing required agents
    # bottom right of coordination overlay
    coord_number_textures_rgba = [
        jnp.zeros((coord_icon_size, coord_icon_size, 4), dtype=jnp.int32),  # 0 - empty
        jnp.zeros((coord_icon_size, coord_icon_size, 4), dtype=jnp.int32),  # 1 - not used
        load_texture("2.png", coord_icon_size),
        load_texture("3.png", coord_icon_size),
        load_texture("4.png", coord_icon_size),
    ]

    coord_number_textures = jnp.array(
        [
            tex[:, :, :3] * jnp.repeat(jnp.expand_dims(tex[:, :, 3], axis=-1), 3, axis=-1)
            if tex.shape[-1] == 4
            else tex[:, :, :3]
            for tex in coord_number_textures_rgba
        ]
    )

    coord_number_textures_alpha = jnp.array(
        [
            jnp.repeat(jnp.expand_dims(tex[:, :, 3], axis=-1), repeats=3, axis=2)
            if tex.shape[-1] == 4
            else jnp.zeros((coord_icon_size, coord_icon_size, 3), dtype=jnp.int32)
            for tex in coord_number_textures_rgba
        ]
    )

    # Clock icon for handover/temporal coordination
    coord_clock_texture_rgba = load_texture("clock.png", coord_icon_size)
    coord_clock_texture = coord_clock_texture_rgba[:, :, :3] * jnp.repeat(
        jnp.expand_dims(coord_clock_texture_rgba[:, :, 3], axis=-1), 3, axis=-1
    )
    coord_clock_texture_alpha = jnp.repeat(
        jnp.expand_dims(coord_clock_texture_rgba[:, :, 3], axis=-1), repeats=3, axis=2
    )

    # Player icon for coordination overlay
    # top left of coordination overlay
    coord_player_texture_rgba = load_texture("player.png", coord_icon_size)
    coord_player_texture = coord_player_texture_rgba[:, :, :3]
    coord_player_texture_alpha = (
        coord_player_texture_rgba[:, :, 3]
        if coord_player_texture_rgba.shape[-1] == 4
        else jnp.ones((coord_icon_size, coord_icon_size))
    )

    health_texture = jnp.array(load_texture("health.png", small_block_pixel_size)[:, :, :3])
    hunger_texture = jnp.array(load_texture("food.png", small_block_pixel_size)[:, :, :3])
    thirst_texture = jnp.array(load_texture("drink.png", small_block_pixel_size)[:, :, :3])
    energy_texture = jnp.array(load_texture("energy.png", small_block_pixel_size)[:, :, :3])
    mana_texture = jnp.array(load_texture("mana.png", small_block_pixel_size)[:, :, :3])

    pickaxe_textures = jnp.array(
        [
            apply_alpha(load_texture(filename, small_block_pixel_size))
            for filename in [
                "debug.png",
                "wood_pickaxe.png",
                "stone_pickaxe.png",
                "iron_pickaxe.png",
                "diamond_pickaxe.png",
            ]
        ]
    )
    pickaxe_textures = pickaxe_textures.at[0].set(smaller_empty_texture)

    sword_textures = jnp.array(
        [
            apply_alpha(load_texture(filename, small_block_pixel_size))
            for filename in [
                "debug.png",
                "wood_sword.png",
                "stone_sword.png",
                "iron_sword.png",
                "diamond_sword.png",
            ]
        ]
    )
    sword_textures = sword_textures.at[0].set(smaller_empty_texture)

    iron_armour_textures = jnp.array(
        [
            apply_alpha(load_texture(filename, small_block_pixel_size))
            for filename in [
                "iron_helmet.png",
                "iron_chestplate.png",
                "iron_pants.png",
                "iron_boots.png",
            ]
        ]
    )
    diamond_armour_textures = jnp.array(
        [
            apply_alpha(load_texture(filename, small_block_pixel_size))
            for filename in [
                "diamond_helmet.png",
                "diamond_chestplate.png",
                "diamond_pants.png",
                "diamond_boots.png",
            ]
        ]
    )
    empty_armour_textures = jnp.stack(
        [
            smaller_empty_texture,
            smaller_empty_texture,
            smaller_empty_texture,
            smaller_empty_texture,
        ],
        axis=0,
    )

    armour_textures = jnp.stack(
        [empty_armour_textures, iron_armour_textures, diamond_armour_textures], axis=0
    )

    bow_texture = load_texture("bow.png", small_block_pixel_size)[:, :, :3]
    bow_textures = jnp.stack([smaller_empty_texture, bow_texture], axis=0)
    player_projectile_textures = jnp.array(
        [
            apply_alpha(load_texture(filename, small_block_pixel_size))
            for filename in ["arrow-up.png", "debug.png", "fireball.png", "iceball.png"]
        ]
    )

    sapling_texture = jnp.array(load_texture("sapling.png", small_block_pixel_size)[:, :, :3])

    torch_inv_texture = jnp.array(
        load_texture("torch_in_inventory.png", small_block_pixel_size)[:, :, :3]
    )

    # entities
    melee_mob_textures, melee_mob_texture_alphas = load_mob_texture_set(
        [
            "zombie.png",
            "gnome_warrior.png",
            "orc_soldier.png",
            "lizard.png",
            "knight.png",
            "troll.png",
            "pigman.png",
            "frost_troll.png",
        ],
        block_pixel_size,
    )
    passive_mob_textures, passive_mob_texture_alphas = load_mob_texture_set(
        ["cow.png", "bat.png", "snail.png", "buffalo.png", "large_cow.png"], block_pixel_size
    )
    ranged_mob_textures, ranged_mob_texture_alphas = load_mob_texture_set(
        [
            "skeleton.png",
            "gnome_archer.png",
            "orc_mage.png",
            "kobold.png",
            "knight_archer.png",
            "deep_thing.png",
            "fire_elemental.png",
            "ice_elemental.png",
        ],
        block_pixel_size,
    )
    projectile_textures, projectile_texture_alphas = load_mob_texture_set(
        [
            "arrow-up.png",
            "dagger.png",
            "fireball.png",
            "iceball.png",
            "arrow-up.png",
            "slimeball.png",
            "fireball.png",
            "iceball.png",
        ],
        block_pixel_size,
    )

    night_texture = (
        jnp.array([[[0, 16, 64]]])
        .repeat(OBS_DIM[0] * block_pixel_size, axis=0)
        .repeat(OBS_DIM[1] * block_pixel_size, axis=1)
    )

    night_noise_intensity_texture = jnp.array(
        [
            [
                jnp.sqrt(
                    (x - (OBS_DIM[0] * block_pixel_size // 2)) ** 2
                    + (y - (OBS_DIM[1] * block_pixel_size // 2)) ** 2
                )
                for y in range(OBS_DIM[1] * block_pixel_size)
            ]
            for x in range(OBS_DIM[0] * block_pixel_size)
        ]
    )
    night_noise_intensity_texture = (
        night_noise_intensity_texture / night_noise_intensity_texture.max()
    )

    night_noise_intensity_texture = jnp.expand_dims(night_noise_intensity_texture, axis=-1).repeat(
        3, axis=-1
    )

    potion_textures = jnp.array(
        [
            load_texture("potion_red.png", small_block_pixel_size)[:, :, :3],
            load_texture("potion_green.png", small_block_pixel_size)[:, :, :3],
            load_texture("potion_blue.png", small_block_pixel_size)[:, :, :3],
            load_texture("potion_pink.png", small_block_pixel_size)[:, :, :3],
            load_texture("potion_cyan.png", small_block_pixel_size)[:, :, :3],
            load_texture("potion_yellow.png", small_block_pixel_size)[:, :, :3],
        ]
    )

    book_texture = load_texture("book.png", small_block_pixel_size)[:, :, :3]

    fireball_inv_texture = load_texture("fireball.png", small_block_pixel_size)[:, :, :3]
    iceball_inv_texture = load_texture("iceball.png", small_block_pixel_size)[:, :, :3]
    heal_inv_texture = load_texture("heal_cross.png", small_block_pixel_size)[:, :, :3]

    # Attributes
    xp_texture = load_texture("xp.png", small_block_pixel_size)[:, :, :3]
    dex_texture = load_texture("dexterity.png", small_block_pixel_size)[:, :, :3]
    str_texture = load_texture("strength.png", small_block_pixel_size)[:, :, :3]
    int_texture = load_texture("intelligence.png", small_block_pixel_size)[:, :, :3]

    # Specializations
    forager_texture = load_texture("forager.png", small_block_pixel_size)[:, :, :3]
    warrior_texture = load_texture("warrior.png", small_block_pixel_size)[:, :, :3]
    miner_texture = load_texture("miner.png", small_block_pixel_size)[:, :, :3]

    armour_enchantment_textures = jnp.array(
        [
            [
                jnp.zeros((small_block_pixel_size, small_block_pixel_size, 4)),
                jnp.zeros((small_block_pixel_size, small_block_pixel_size, 4)),
                jnp.zeros((small_block_pixel_size, small_block_pixel_size, 4)),
                jnp.zeros((small_block_pixel_size, small_block_pixel_size, 4)),
            ],
            [
                load_texture("helmet_fire_enchantment.png", small_block_pixel_size),
                load_texture("chestplate_fire_enchantment.png", small_block_pixel_size),
                load_texture("pants_fire_enchantment.png", small_block_pixel_size),
                load_texture("boots_fire_enchantment.png", small_block_pixel_size),
            ],
            [
                load_texture("helmet_ice_enchantment.png", small_block_pixel_size),
                load_texture("chestplate_ice_enchantment.png", small_block_pixel_size),
                load_texture("pants_ice_enchantment.png", small_block_pixel_size),
                load_texture("boots_ice_enchantment.png", small_block_pixel_size),
            ],
        ]
    )

    sword_enchantment_textures = jnp.array(
        [
            jnp.zeros((small_block_pixel_size, small_block_pixel_size, 4)),
            load_texture("sword_fire_enchantment.png", small_block_pixel_size),
            load_texture("sword_ice_enchantment.png", small_block_pixel_size),
        ]
    )

    arrow_enchantment_textures = jnp.array(
        [
            jnp.zeros((small_block_pixel_size, small_block_pixel_size, 4)),
            load_texture("arrow_fire_enchantment.png", small_block_pixel_size),
            load_texture("arrow_ice_enchantment.png", small_block_pixel_size),
        ]
    )

    request_message_textures = load_request_message_textures(small_block_pixel_size)
    comm_badge_texture, comm_badge_start_x, comm_badge_start_y, comm_badge_size = (
        load_comm_badge_texture(small_block_pixel_size)
    )

    return {
        "block_textures": block_textures,
        "smaller_block_textures": smaller_block_textures,
        "full_map_block_textures": full_map_block_textures,
        "transparent_block_textures": transparent_block_textures,
        "transparent_block_texture_alphas": transparent_block_texture_alphas,
        "full_map_item_textures": full_map_item_textures,
        "player_textures": player_textures,
        "full_map_player_textures": full_map_player_textures,
        "full_map_player_textures_alpha": full_map_player_textures_alpha,
        "player_icon_textures": player_icon_textures,
        "empty_texture": empty_texture,
        "smaller_empty_texture": smaller_empty_texture,
        "ones_texture": ones_texture,
        "number_textures": number_textures,
        "number_textures_alpha": number_textures_alpha,
        "number_textures_with_zero": number_textures_with_zero,
        "number_textures_alpha_with_zero": number_textures_with_zero_alpha,
        "comm_number_textures_with_zero": comm_number_textures_with_zero,
        "comm_number_textures_alpha_with_zero": comm_number_textures_with_zero_alpha,
        "coord_number_textures": coord_number_textures,
        "coord_number_textures_alpha": coord_number_textures_alpha,
        "coord_clock_texture": coord_clock_texture,
        "coord_clock_texture_alpha": coord_clock_texture_alpha,
        "coord_player_texture": coord_player_texture,
        "coord_player_texture_alpha": coord_player_texture_alpha,
        "coord_icon_size": coord_icon_size,
        "health_texture": health_texture,
        "hunger_texture": hunger_texture,
        "thirst_texture": thirst_texture,
        "energy_texture": energy_texture,
        "mana_texture": mana_texture,
        "pickaxe_textures": pickaxe_textures,
        "sword_textures": sword_textures,
        "sapling_texture": sapling_texture,
        "night_texture": night_texture,
        "night_noise_intensity_texture": night_noise_intensity_texture,
        "melee_mob_textures": melee_mob_textures,
        "melee_mob_texture_alphas": melee_mob_texture_alphas,
        "passive_mob_textures": passive_mob_textures,
        "passive_mob_texture_alphas": passive_mob_texture_alphas,
        "direction_textures": direction_textures,
        "ranged_mob_textures": ranged_mob_textures,
        "ranged_mob_texture_alphas": ranged_mob_texture_alphas,
        "projectile_textures": projectile_textures,
        "projectile_texture_alphas": projectile_texture_alphas,
        "armour_textures": armour_textures,
        "bow_textures": bow_textures,
        "player_projectile_textures": player_projectile_textures,
        "torch_inv_texture": torch_inv_texture,
        "potion_textures": potion_textures,
        "book_texture": book_texture,
        "fireball_inv_texture": fireball_inv_texture,
        "iceball_inv_texture": iceball_inv_texture,
        "heal_inv_texture": heal_inv_texture,
        "armour_enchantment_textures": armour_enchantment_textures,
        "sword_enchantment_textures": sword_enchantment_textures,
        "arrow_enchantment_textures": arrow_enchantment_textures,
        "xp_texture": xp_texture,
        "dex_texture": dex_texture,
        "str_texture": str_texture,
        "int_texture": int_texture,
        "forager_texture": forager_texture,
        "warrior_texture": warrior_texture,
        "miner_texture": miner_texture,
        "comm_badge_texture": comm_badge_texture,
        "comm_badge_start_x": comm_badge_start_x,
        "comm_badge_start_y": comm_badge_start_y,
        "comm_badge_size": comm_badge_size,
        "request_message_textures": request_message_textures,
    }


_REQUIRED_TEXTURE_KEYS = {
    "comm_badge_texture",
    "comm_badge_start_x",
    "comm_badge_start_y",
    "comm_badge_size",
    "comm_number_textures_with_zero",
    "comm_number_textures_alpha_with_zero",
    "coord_icon_size",
    "coord_clock_texture",
    "request_message_textures",
}


def _cache_is_valid(textures):
    for pixel_size_dict in textures.values():
        if not _REQUIRED_TEXTURE_KEYS.issubset(pixel_size_dict.keys()):
            return False
    return True


if os.path.exists(TEXTURE_CACHE_FILE) and not os.environ.get("CRAFTAX_RELOAD_TEXTURES", False):
    TEXTURES = load_compressed_pickle(TEXTURE_CACHE_FILE)
    if not _cache_is_valid(TEXTURES):
        print("Texture cache is stale (missing keys) — regenerating...")
        TEXTURES = None
    else:
        print("Loading textures from cache")
else:
    TEXTURES = None

if TEXTURES is None:
    print("Processing textures")
    TEXTURES = {
        BLOCK_PIXEL_SIZE_AGENT: load_all_textures(BLOCK_PIXEL_SIZE_AGENT),
        BLOCK_PIXEL_SIZE_IMG: load_all_textures(BLOCK_PIXEL_SIZE_IMG),
        BLOCK_PIXEL_SIZE_HUMAN: load_all_textures(BLOCK_PIXEL_SIZE_HUMAN),
    }
    save_compressed_pickle(TEXTURE_CACHE_FILE, TEXTURES)
    print("Textures saved to cache")
