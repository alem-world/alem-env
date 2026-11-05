"""
Play Alem - interactive Alem-Coop game.

Each player takes turns providing an action, then the environment steps.

Controls:
    Movement: WASD          Interact: SPACE
    Sleep: TAB              Rest: E              No-op: Q
    Descend/Ascend: . / ,

    Placement: R=Stone  T=Table  F=Furnace  P=Plant  J=Torch
    Pickaxes:  1=Wood  2=Stone  3=Iron  4=Diamond
    Swords:    5=Wood  6=Stone  7=Iron  8=Diamond
    Armour:    Y=Iron  U=Diamond
    Other:     O=Arrow  [=Torch

    Combat:    I=Shoot Arrow   G=Cast Spell
    Potions:   Z=Red  X=Green  C=Blue  V=Pink  B=Cyan  N=Yellow
    Magic:     M=Read Book  K=Enchant Sword  L=Enchant Armour  ;=Enchant Bow
    Level Up:  ]=Dex  -=Str  ==Int

    Requests:  F1=Food F2=Drink F3=Wood F4=Stone F5=Iron F6=Coal F7=Diamond F8=Ruby F9=Sapphire
    Give:      Backspace=give to the lower-numbered other player
               Delete=give to the higher-numbered other player
    Build:     9=Shelter  0=Forge  `=Beacon

The window has three parts: the map (framed in the colour of the player whose turn
it is), a status strip under it (your stats, inventory, and what you are facing), and
a sidebar listing the whole team's requests plus every key, drawn white when that
action would work right now and faded when it would not. See PLAY_WITH_HUMAN_INTERFACE.md for a walkthrough.

Usage:
    python examples/play_alem.py                     # Default 3 players
    python examples/play_alem.py --players 2 --god   # 2 players, god mode
    python examples/play_alem.py --seed 42           # Reproducible seed
    python examples/play_alem.py --coord easy        # Coordination difficulty
    python examples/play_alem.py --size 64           # Smaller window (default: fit the screen)
"""

import argparse
import sys
from collections import deque

import jax
import jax.numpy as jnp
import numpy as np
import pygame

from alem.alem_coop.action_masking import compute_action_mask
from alem.alem_coop.alem_state import EnvParams, StaticEnvParams, get_coordination_params
from alem.alem_coop.constants import (
    BLOCK_PIXEL_SIZE_HUMAN,
    DIRECTIONS,
    MONSTERS_KILLED_TO_CLEAR_LEVEL,
    OBS_DIM,
    TEXTURES,
    Achievement,
    Action,
    BlockType,
    ItemType,
    Specialization,
    load_player_specific_textures,
)
from alem.alem_coop.envs.alem_pixels_env import AlemCoopPixelsEnv
from alem.alem_coop.renderer.renderer_pixels import render_alem_pixels
from alem.alem_coop.util.game_logic_utils import (
    get_max_drink,
    get_max_energy,
    get_max_food,
    get_max_health,
    get_max_mana,
)

KEY_MAPPING = {
    pygame.K_q: Action.NOOP,
    # Movement
    pygame.K_w: Action.UP,
    pygame.K_a: Action.LEFT,
    pygame.K_s: Action.DOWN,
    pygame.K_d: Action.RIGHT,
    # Core actions
    pygame.K_SPACE: Action.DO,
    pygame.K_TAB: Action.SLEEP,
    pygame.K_e: Action.REST,
    pygame.K_PERIOD: Action.DESCEND,
    pygame.K_COMMA: Action.ASCEND,
    # Placement
    pygame.K_r: Action.PLACE_STONE,
    pygame.K_t: Action.PLACE_TABLE,
    pygame.K_f: Action.PLACE_FURNACE,
    pygame.K_p: Action.PLACE_PLANT,
    pygame.K_j: Action.PLACE_TORCH,
    # Pickaxes
    pygame.K_1: Action.MAKE_WOOD_PICKAXE,
    pygame.K_2: Action.MAKE_STONE_PICKAXE,
    pygame.K_3: Action.MAKE_IRON_PICKAXE,
    pygame.K_4: Action.MAKE_DIAMOND_PICKAXE,
    # Swords
    pygame.K_5: Action.MAKE_WOOD_SWORD,
    pygame.K_6: Action.MAKE_STONE_SWORD,
    pygame.K_7: Action.MAKE_IRON_SWORD,
    pygame.K_8: Action.MAKE_DIAMOND_SWORD,
    # Other crafting
    pygame.K_y: Action.MAKE_IRON_ARMOUR,
    pygame.K_u: Action.MAKE_DIAMOND_ARMOUR,
    pygame.K_o: Action.MAKE_ARROW,
    pygame.K_LEFTBRACKET: Action.MAKE_TORCH,
    # Combat
    pygame.K_i: Action.SHOOT_ARROW,
    pygame.K_g: Action.CAST_SPELL,
    # Potions
    pygame.K_z: Action.DRINK_POTION_RED,
    pygame.K_x: Action.DRINK_POTION_GREEN,
    pygame.K_c: Action.DRINK_POTION_BLUE,
    pygame.K_v: Action.DRINK_POTION_PINK,
    pygame.K_b: Action.DRINK_POTION_CYAN,
    pygame.K_n: Action.DRINK_POTION_YELLOW,
    # Enchanting & books
    pygame.K_m: Action.READ_BOOK,
    pygame.K_k: Action.ENCHANT_SWORD,
    pygame.K_l: Action.ENCHANT_ARMOUR,
    pygame.K_SEMICOLON: Action.ENCHANT_BOW,
    # Level up
    pygame.K_RIGHTBRACKET: Action.LEVEL_UP_DEXTERITY,
    pygame.K_MINUS: Action.LEVEL_UP_STRENGTH,
    pygame.K_EQUALS: Action.LEVEL_UP_INTELLIGENCE,
    # Requests
    pygame.K_F1: Action.REQUEST_FOOD,
    pygame.K_F2: Action.REQUEST_DRINK,
    pygame.K_F3: Action.REQUEST_WOOD,
    pygame.K_F4: Action.REQUEST_STONE,
    pygame.K_F5: Action.REQUEST_IRON,
    pygame.K_F6: Action.REQUEST_COAL,
    pygame.K_F7: Action.REQUEST_DIAMOND,
    pygame.K_F8: Action.REQUEST_RUBY,
    pygame.K_F9: Action.REQUEST_SAPPHIRE,
    pygame.K_BACKSPACE: Action.GIVE,  # give slot 0: the lower-numbered other player
    pygame.K_DELETE: Action.GIVE.value + 1,  # give slot 1: the higher-numbered other player
    # Construction
    pygame.K_9: Action.BUILD_SHELTER,
    pygame.K_0: Action.BUILD_FORGE,
    pygame.K_BACKQUOTE: Action.BUILD_BEACON,
}

# Give is one action per *other* teammate ("slots"); only two keys are bound, which
# covers the 3-player protocol. Slot k of giver i targets player k if k < i else k + 1.
GIVE_KEY_LABELS = ["Bksp", "Del"]


def give_target(giver, slot, num_players):
    """Index of the teammate that give slot ``slot`` reaches for ``giver`` (or None)."""
    if slot >= num_players - 1:
        return None
    return slot if slot < giver else slot + 1


# (key label, action, display name, icon spec) in F1..F9 order == Action enum order.
REQUESTS = [
    ("F1", Action.REQUEST_FOOD, "Food", ("hunger_texture", None)),
    ("F2", Action.REQUEST_DRINK, "Drink", ("thirst_texture", None)),
    ("F3", Action.REQUEST_WOOD, "Wood", ("smaller_block_textures", BlockType.WOOD.value)),
    ("F4", Action.REQUEST_STONE, "Stone", ("smaller_block_textures", BlockType.STONE.value)),
    ("F5", Action.REQUEST_IRON, "Iron", ("smaller_block_textures", BlockType.IRON.value)),
    ("F6", Action.REQUEST_COAL, "Coal", ("smaller_block_textures", BlockType.COAL.value)),
    ("F7", Action.REQUEST_DIAMOND, "Diamond", ("smaller_block_textures", BlockType.DIAMOND.value)),
    ("F8", Action.REQUEST_RUBY, "Ruby", ("smaller_block_textures", BlockType.RUBY.value)),
    ("F9", Action.REQUEST_SAPPHIRE, "Sapphire", ("smaller_block_textures", BlockType.SAPPHIRE.value)),
]
REQUEST_NAMES = [r[2] for r in REQUESTS]
# Snapshot field holding each requestable resource (food/drink are stat bars).
REQUEST_FIELDS = ["food", "drink", "wood", "stone", "iron", "coal", "diamond", "ruby", "sapphire"]

LEVEL_NAMES = [
    "Overworld", "Dungeon", "Gnomish Mines", "Sewers", "Vaults",
    "Troll Mines", "Fire Realm", "Ice Realm", "Boss",
]
TIER_NAMES = ["none", "wood", "stone", "iron", "diamond"]
ARMOUR_PIECES = ["Helmet", "Chestplate", "Leggings", "Boots"]
ARMOUR_TIERS = ["none", "iron", "diamond"]
POTION_NAMES = ["Red", "Green", "Blue", "Pink", "Cyan", "Yellow"]
# Same order as alem.llm.alem_language_wrapper (copied so this file stays light).
MELEE_MOB_NAMES = ["zombie", "gnome_warrior", "orc_soldier", "lizard", "knight", "troll", "pigman", "frost_troll"]
RANGED_MOB_NAMES = ["skeleton", "gnome_archer", "orc_mage", "kobold", "knight_archer", "deep_thing", "fire_elemental", "ice_elemental"]
PASSIVE_MOB_NAMES = ["cow", "bat", "snail", "buffalo", "large_cow"]
# Pickaxe tier needed to mine a block (index into TIER_NAMES).
PICKAXE_NEEDED = {
    BlockType.STONE.value: 1, BlockType.COAL.value: 1, BlockType.IRON.value: 2,
    BlockType.DIAMOND.value: 3, BlockType.SAPPHIRE.value: 4, BlockType.RUBY.value: 4,
}
DIRECTION_NAMES = {1: "west", 2: "east", 3: "north", 4: "south"}
_DIRS = np.asarray(DIRECTIONS)


def action_index(a):
    """KEY_MAPPING holds Action enums plus raw ints for the extra Give slots."""
    return a if isinstance(a, int) else a.value


def action_name(a):
    """Name for logging; Give slots beyond the first are not enum members."""
    a = int(a)
    if a in Action._value2member_map_:
        return Action(a).name
    return f"GIVE_SLOT{a - Action.GIVE.value}"


def _humanize(name):
    return name.replace("_", " ").capitalize()


# Fallback accent colours (used only if the real sprite palette can't be built).
_HUD_COLORS = [
    (100, 210, 255),  # cyan
    (130, 230, 120),  # green
    (255, 200, 80),  # amber
    (210, 140, 255),  # purple
    (255, 130, 130),  # pink
]
_HUD_SPEC_NAMES = {1: "Forager", 2: "Warrior", 3: "Miner"}

# Palette: warm paper, charcoal text, one terracotta accent, hairline separators.
# Colour carries meaning only where the map already uses it (coordination borders).
C_BG = (245, 243, 236)
C_LINE = (225, 222, 212)
C_TILE = (38, 37, 34)  # inventory tiles keep the sprites on their dark ground
C_TEXT = (28, 27, 24)
C_DIM = (122, 119, 109)
C_FAINT = (178, 175, 165)
C_HEAD = (138, 135, 125)
C_ACCENT = (201, 104, 63)
C_OK = (76, 154, 91)
C_WARN = (208, 138, 62)
C_BAD = (201, 83, 63)
C_SOFT = (76, 154, 91)  # soft coordination (green border on the map)
C_HARD = (46, 143, 163)  # hard coordination (cyan border on the map)
C_HANDOVER = (208, 138, 62)  # handover (orange dashed border on the map)
C_ELITE = (201, 83, 63)  # elite mobs (red silhouette on the map)
KEY_STYLE = {
    # name: (cap background, cap text, border)
    "on": ((255, 255, 255), C_TEXT, (196, 193, 183)),  # would work now: a white key
    "off": (C_BG, C_FAINT, C_LINE),  # fades into the paper
    "neutral": ((255, 255, 255), C_TEXT, (196, 193, 183)),
    "active": ((251, 236, 227), C_ACCENT, C_ACCENT),
}
BAR_COLORS = {
    "hp": (214, 104, 88), "food": (214, 160, 92), "drink": (96, 142, 206),
    "energy": (112, 172, 122), "mana": (150, 124, 206),
}

# Layout in "design pixels" (the 64 px-per-block base); everything scales with --size.
TOP_H, STATUS_H, LOG_H, PANEL_W, PAD = 44, 198, 84, 470, 10
SIZE_CHOICES = (64, 80, 96, 112, 128, 160, 192)


def window_size_for(block_px, num_players=3):
    """Window (w, h) in pixels for a given on-screen block size."""
    s = block_px / BLOCK_PIXEL_SIZE_HUMAN
    w = OBS_DIM[1] * block_px + int(PANEL_W * s)
    h = int(TOP_H * s) + OBS_DIM[0] * block_px + int(STATUS_H * s) + int(LOG_H * s)
    return w, h


def auto_block_size(num_players=3):
    """Largest block size whose window fits the current desktop (64 if unknown)."""
    try:
        pygame.display.init()
        info = pygame.display.Info()
        dw, dh = int(info.current_w), int(info.current_h)
    except Exception:
        return 64
    if dw <= 0 or dh <= 0:
        return 64
    best = 64
    for size in SIZE_CHOICES:
        w, h = window_size_for(size, num_players)
        if w <= dw * 0.96 and h <= dh * 0.90:
            best = size
    return best


def resolve_block_size(size_arg, num_players=3):
    """``--size`` value (None = fit the screen) → on-screen block size in pixels."""
    if size_arg is None or int(size_arg) <= 0:
        return auto_block_size(num_players)
    return max(32, int(size_arg))


# A clean UI font, in preference order across platforms. SysFont("Arial") falls
# back to a rough bitmap-ish default on Linux; matching a real family fixes the
# "pixelated" look. First match wins.
_PREFERRED_FONTS = [
    "Inter", "Helvetica Neue", "Segoe UI", "Cantarell", "Roboto", "Noto Sans",
    "Liberation Sans", "DejaVu Sans", "Arial", "Helvetica",
]
_FONT_PATH_CACHE = {}


def _font_path(bold):
    if bold not in _FONT_PATH_CACHE:
        chosen = None
        for family in _PREFERRED_FONTS:
            chosen = pygame.font.match_font(family, bold=bold)
            if chosen:
                break
        _FONT_PATH_CACHE[bold] = chosen
    return _FONT_PATH_CACHE[bold]


def load_font(size, bold=False):
    """Load a crisp UI font at the given pixel size (falls back gracefully)."""
    try:
        path = _font_path(bold)
        if path:
            return pygame.font.Font(path, size)
    except Exception:
        pass
    return pygame.font.SysFont("Arial", size, bold=bold)


def player_palette(num_players):
    """Return each player's on-screen colour.

    Matches the exact per-player sprite colour used by the renderer
    (``load_player_specific_textures`` calls the same husl_palette), so a HUD
    swatch or accent line is the same colour as that player's character in the
    world — the basis for showing who controls whom.
    """
    try:
        from seaborn import husl_palette

        return [
            tuple(int(c * 255) for c in rgb) for rgb in husl_palette(num_players, h=0.5, l=0.5)
        ]
    except Exception:
        return [_HUD_COLORS[i % len(_HUD_COLORS)] for i in range(num_players)]


_HUD_HINTS = ["WASD move · Space act · Q skip · F1–F9 ask · Esc menu"]


def _theme_color(color):
    """Map the saturated colours callers pass to push_notification onto the palette."""
    r, g, b = color[:3]
    if r + g + b > 600:
        return C_TEXT  # white prompts
    if r > 200 and g > 180 and b < 120:
        return C_ACCENT  # gold: achievements
    if g > r and g > b:
        return C_OK  # green: rewards, saves
    if r > g and g >= 120:
        return C_WARN  # orange: penalties
    return C_BAD  # red: damage


def _build_panel_groups():
    """Sidebar key reference, two columns of (group title, [(key, label, Action|int|None, icon)]).

    The Action, when present, is looked up in the legal-action mask so the row
    lights up green when it would work this turn. None = always available.
    ``icon`` is an optional (texture key, index) drawn in front of the label.
    """
    a = Action
    pot = [("Z", 0), ("X", 1), ("C", 2), ("V", 3), ("B", 4), ("N", 5)]
    potion_actions = [
        a.DRINK_POTION_RED, a.DRINK_POTION_GREEN, a.DRINK_POTION_BLUE,
        a.DRINK_POTION_PINK, a.DRINK_POTION_CYAN, a.DRINK_POTION_YELLOW,
    ]
    left = [
        ("MOVE", [("W A S D", "Move / turn to face", None, None)]),
        ("ACT", [
            ("Space", "Do / interact", a.DO, None),
            ("E", "Rest (heal)", a.REST, None),
            ("Tab", "Sleep (energy)", a.SLEEP, None),
            ("Q", "Skip turn", a.NOOP, None),
            (".  ,", "Descend / ascend ladder", a.DESCEND, None),
        ]),
        ("FIGHT", [
            ("I", "Shoot arrow", a.SHOOT_ARROW, ("bow_textures", 1)),
            ("G", "Cast spell", a.CAST_SPELL, ("fireball_inv_texture", None)),
        ]),
        ("PLACE", [
            ("R", "Stone", a.PLACE_STONE, ("smaller_block_textures", BlockType.STONE.value)),
            ("T", "Table (2 wood)", a.PLACE_TABLE, ("smaller_block_textures", BlockType.CRAFTING_TABLE.value)),
            ("F", "Furnace (1 stone)", a.PLACE_FURNACE, ("smaller_block_textures", BlockType.FURNACE.value)),
            ("P", "Plant sapling", a.PLACE_PLANT, ("sapling_texture", None)),
            ("J", "Torch", a.PLACE_TORCH, ("torch_inv_texture", None)),
        ]),
        ("PICKAXE  (at table)", [
            ("1", "Wood", a.MAKE_WOOD_PICKAXE, ("pickaxe_textures", 1)),
            ("2", "Stone", a.MAKE_STONE_PICKAXE, ("pickaxe_textures", 2)),
            ("3", "Iron (+furnace)", a.MAKE_IRON_PICKAXE, ("pickaxe_textures", 3)),
            ("4", "Diamond (forge)", a.MAKE_DIAMOND_PICKAXE, ("pickaxe_textures", 4)),
        ]),
        ("SWORD  (at table)", [
            ("5", "Wood", a.MAKE_WOOD_SWORD, ("sword_textures", 1)),
            ("6", "Stone", a.MAKE_STONE_SWORD, ("sword_textures", 2)),
            ("7", "Iron (+furnace)", a.MAKE_IRON_SWORD, ("sword_textures", 3)),
            ("8", "Diamond (forge)", a.MAKE_DIAMOND_SWORD, ("sword_textures", 4)),
        ]),
    ]
    right = [
        ("CRAFT", [
            ("Y", "Iron armour", a.MAKE_IRON_ARMOUR, ("armour_textures", (1, 1))),
            ("U", "Diamond armour", a.MAKE_DIAMOND_ARMOUR, ("armour_textures", (2, 1))),
            ("O", "Arrows ×2", a.MAKE_ARROW, ("player_projectile_textures", 0)),
            ("[", "Torches ×4", a.MAKE_TORCH, ("torch_inv_texture", None)),
        ]),
        ("BUILD  (at a site)", [
            ("9", "Shelter", a.BUILD_SHELTER, ("smaller_block_textures", BlockType.EPIC_SHELTER.value)),
            ("0", "Forge", a.BUILD_FORGE, ("smaller_block_textures", BlockType.EPIC_FORGE.value)),
            ("`", "Beacon", a.BUILD_BEACON, ("smaller_block_textures", BlockType.EPIC_BEACON.value)),
        ]),
        ("POTIONS  (effects differ per world)", [
            (k, f"{POTION_NAMES[i]} potion", potion_actions[i], ("potion_textures", i))
            for k, i in pot
        ]),
        ("MAGIC", [
            ("M", "Read book", a.READ_BOOK, ("book_texture", None)),
            ("K", "Enchant sword", a.ENCHANT_SWORD, None),
            ("L", "Enchant armour", a.ENCHANT_ARMOUR, None),
            (";", "Enchant bow", a.ENCHANT_BOW, None),
        ]),
        ("LEVEL UP  (spends 1 XP)", [
            ("]", "Dexterity", a.LEVEL_UP_DEXTERITY, ("dex_texture", None)),
            ("-", "Strength", a.LEVEL_UP_STRENGTH, ("str_texture", None)),
            ("=", "Intelligence", a.LEVEL_UP_INTELLIGENCE, ("int_texture", None)),
        ]),
    ]
    return left, right


class AlemRenderer:
    """Pygame window for hotseat play.

    Layout (left column over right sidebar):

    * top bar — one chip per player (their sprite colour), the current one boxed,
      plus the floor, step count and monsters-killed progress for the ladder;
    * map — the 9×11 view centred on the current player, framed in their colour;
    * status strip — the current player's stat bars, labelled inventory, and a
      one-line description of the tile they are facing (coordination rules,
      pickaxe needed, elite-mob rules, ladder state);
    * event log — recent rewards / damage / achievements plus key hints;
    * sidebar — the team (health, requests, which Give key reaches whom), the
      nine request keys, and every other key grouped, drawn white when usable now
      and faded when not.

    ``pixel_render_size`` is the on-screen scale relative to the 64 px base block
    (2 → 128 px blocks). It may be fractional; see ``resolve_block_size``.
    """

    def __init__(
        self, env, env_params, static_params, player_textures, pixel_render_size=1, key_repeat=True
    ):
        self.env = env
        self.env_params = env_params
        self.static_params = static_params
        self.num_players = static_params.player_count
        self.pixel_render_size = float(pixel_render_size)
        self.pygame_events = []

        s = self.pixel_render_size
        self._s = s
        px = self.px
        self.block_px = max(8, int(round(BLOCK_PIXEL_SIZE_HUMAN * s)))
        self.map_w = OBS_DIM[1] * self.block_px
        self.map_h = OBS_DIM[0] * self.block_px
        # Kept for callers that used the old names.
        self.game_w, self.game_h = self.map_w, self.map_h
        self._pad = px(PAD)
        self._top_h = px(TOP_H)
        self._status_h = px(STATUS_H)
        self._log_h = px(LOG_H)
        self._panel_w = px(PANEL_W)
        self._game_y = self._top_h
        self._status_y = self._top_h + self.map_h
        self._log_y = self._status_y + self._status_h
        self.screen_size = (self.map_w + self._panel_w, self._log_y + self._log_h)

        # Real per-player sprite colours, so chips/frames match the characters.
        self._player_colors = player_palette(self.num_players)

        pygame.init()
        if key_repeat:
            pygame.key.set_repeat(250, 75)
        else:
            pygame.key.set_repeat()  # a held key never spills into the next player's turn
        self.screen_surface = pygame.display.set_mode(self.screen_size)
        pygame.display.set_caption("Alem - Alem-Coop")

        self._font_title = load_font(px(15), bold=True)
        self._font_status = load_font(px(14), bold=True)
        self._font_event = load_font(px(14), bold=True)
        self._font_label = load_font(px(12))
        self._font_small = load_font(px(10))
        self._font_hints = load_font(px(11))
        self._font_legend = load_font(px(13), bold=True)
        self._font_head = load_font(px(10), bold=True)
        self._font_key = load_font(px(11), bold=True)
        self._font_banner = load_font(px(22), bold=True)
        # Fonts for the full-screen text pages (rules / briefings / menus).
        self._font_page_title = load_font(px(26), bold=True)
        self._font_page_head = load_font(px(18), bold=True)
        self._font_page_body = load_font(px(15))
        # Older callers referenced these names.
        self._font_panel_title = self._font_title
        self._font_panel_head = self._font_head
        self._font_panel_key = self._font_key
        self._font_panel_label = self._font_label

        # Per-player legal-action mask for the sidebar (recomputed only when the
        # state object changes, so idle re-renders don't re-run it).
        self._action_mask_fn = jax.jit(
            lambda st: compute_action_mask(st, self.env_params, self.static_params)
        )
        self._mask_cache_state = None
        self._mask_cache = None
        self._snap_state = None
        self._snap = None
        self._map_cache_key = None
        self._map_cache = None
        self._panel_groups = _build_panel_groups()

        # JIT compile the render function (block_pixel_size and static_params are static)
        self._render = jax.jit(render_alem_pixels, static_argnums=(1, 2))
        self._player_textures = player_textures
        self._tex = TEXTURES[BLOCK_PIXEL_SIZE_HUMAN]
        self._icon_cache = {}

        self._notification = None  # sticky banner: (expires_at_ms, text, color)
        self._toast = None  # short flash over the map: (expires_at_ms, text, color)
        self._events = deque(maxlen=4)  # recent short notifications: (text, color)

    # ── small drawing helpers ───────────────────────────────────────────────

    def px(self, v):
        """Design pixels → screen pixels."""
        return max(1, int(round(v * self._s)))

    def _icon(self, spec, size):
        """Pygame surface for a texture-dict icon, scaled to ``size`` px and cached."""
        if spec is None:
            return None
        key, idx = spec
        cache_key = (key, idx, size)
        surf = self._icon_cache.get(cache_key)
        if surf is not None:
            return surf
        try:
            arr = self._tex[key]
            if idx is not None:
                arr = arr[idx] if not isinstance(idx, tuple) else arr[idx]
            surf = self._surface_from_texture(np.asarray(arr), size)
        except Exception:
            surf = None
        self._icon_cache[cache_key] = surf
        return surf

    def _player_icon(self, p, size):
        cache_key = ("player_icon", p, size)
        surf = self._icon_cache.get(cache_key)
        if surf is None:
            arr = np.asarray(self._player_textures.player_icon_textures[p, 0])
            surf = self._surface_from_texture(arr, size)
            self._icon_cache[cache_key] = surf
        return surf

    @staticmethod
    def _surface_from_texture(arr, size):
        """(H, W, 3|4) texture array (0–255 colour, 0/1 or 0–255 alpha) → Surface."""
        arr = np.asarray(arr)
        h, w = arr.shape[:2]
        rgb = np.clip(arr[..., :3], 0, 255).astype(np.uint8)
        if arr.shape[-1] == 4:
            alpha = arr[..., 3].astype(np.float32)
            if alpha.max() <= 1.0:
                alpha = alpha * 255.0
            rgba = np.dstack([rgb, np.clip(alpha, 0, 255).astype(np.uint8)])
            surf = pygame.image.frombuffer(np.ascontiguousarray(rgba).tobytes(), (w, h), "RGBA")
            surf = surf.convert_alpha()
        else:
            surf = pygame.image.frombuffer(np.ascontiguousarray(rgb).tobytes(), (w, h), "RGB")
            surf = surf.convert()
            surf.set_colorkey((0, 0, 0))  # icon backgrounds are pure black
        if size != w:
            surf = pygame.transform.smoothscale(surf, (size, size))
        return surf

    def _text(self, font, text, color):
        return font.render(str(text), True, color)

    def _blit_text(self, font, text, color, x, y, anchor="topleft"):
        surf = self._text(font, text, color)
        rect = surf.get_rect(**{anchor: (x, y)})
        self.screen_surface.blit(surf, rect)
        return rect

    def _keycap(self, x, y, w, label, style="neutral", h=None):
        """Rounded key cap with centred label. Returns the cap rect."""
        h = h or (self._font_key.get_height() + self.px(4))
        bg, fg, border = KEY_STYLE[style]
        cap = pygame.Rect(x, y, w, h)
        pygame.draw.rect(self.screen_surface, bg, cap, border_radius=self.px(4))
        if border:
            pygame.draw.rect(
                self.screen_surface, border, cap, width=self.px(1), border_radius=self.px(4)
            )
        self._blit_text(self._font_key, label, fg, cap.centerx, cap.centery, anchor="center")
        return cap

    def _bar(self, x, y, w, h, frac, color):
        pygame.draw.rect(self.screen_surface, C_LINE, (x, y, w, h), border_radius=self.px(3))
        fill = int(round(w * max(0.0, min(1.0, frac))))
        if fill > 0:
            pygame.draw.rect(self.screen_surface, color, (x, y, fill, h), border_radius=self.px(3))

    def _slot(self, x, y, w, icon, count, name, dim=False, count_text=None):
        """Inventory slot: a dark tile with the sprite and its count, name underneath."""
        box_h = self.px(38)
        box = pygame.Rect(x, y, w - self.px(6), box_h)
        tile = C_TILE if not dim else (214, 211, 202)
        pygame.draw.rect(self.screen_surface, tile, box, border_radius=self.px(6))
        isz = self.px(28)
        if icon is not None:
            icon_surf = pygame.transform.smoothscale(icon, (isz, isz)) if icon.get_width() != isz else icon
            if dim:
                icon_surf = icon_surf.copy()
                icon_surf.set_alpha(60)
            self.screen_surface.blit(icon_surf, (box.x + self.px(4), box.y + (box_h - isz) // 2))
        label = count_text if count_text is not None else (str(count) if count is not None else "")
        if label:
            self._blit_text(
                self._font_status, label, (255, 255, 255) if not dim else C_DIM,
                box.right - self.px(4), box.bottom - self.px(2), anchor="bottomright",
            )
        self._blit_text(
            self._font_small, name, C_FAINT if dim else C_DIM,
            box.centerx, box.bottom + self.px(2), anchor="midtop",
        )

    # ── notifications ───────────────────────────────────────────────────────

    def push_notification(self, text, color=(255, 255, 255), duration_ms=2500):
        """Show a message. Long durations (≥10 s) are sticky prompts drawn as a
        banner over the map; short ones flash briefly and go into the event log."""
        now = pygame.time.get_ticks()
        if duration_ms >= 10_000:
            self._notification = (now + duration_ms, text, color)
        else:
            self._toast = (now + duration_ms, text, color)
            self._events.append((text, color))

    def update(self):
        self.pygame_events = list(pygame.event.get())
        pygame.display.flip()

    # ── state access (cached per state object) ─────────────────────────────

    def _current_mask(self, state):
        """Legal-action mask for all players, recomputed only when state changes."""
        if state is not self._mask_cache_state:
            try:
                self._mask_cache = np.asarray(self._action_mask_fn(state))
            except Exception:
                self._mask_cache = None
            self._mask_cache_state = state
        return self._mask_cache

    def _snapshot(self, state):
        """NumPy copies of everything the panels read, fetched once per state."""
        if state is self._snap_state:
            return self._snap
        lvl = int(state.player_level)
        inv = state.inventory
        fields = {
            "pos": state.player_position, "dir": state.player_direction,
            "hp": state.player_health, "food": state.player_food, "drink": state.player_drink,
            "energy": state.player_energy, "mana": state.player_mana,
            "max_hp": get_max_health(state), "max_food": get_max_food(state),
            "max_drink": get_max_drink(state), "max_energy": get_max_energy(state),
            "max_mana": get_max_mana(state),
            "spec": state.player_specialization, "alive": state.player_alive,
            "sleeping": state.is_sleeping, "resting": state.is_resting,
            "req_type": state.request_type, "req_dur": state.request_duration,
            "xp": state.player_xp, "dex": state.player_dexterity, "str": state.player_strength,
            "int": state.player_intelligence, "spells": state.learned_spells,
            "sword_ench": state.sword_enchantment, "bow_ench": state.bow_enchantment,
            "wood": inv.wood, "stone": inv.stone, "coal": inv.coal, "iron": inv.iron,
            "diamond": inv.diamond, "sapphire": inv.sapphire, "ruby": inv.ruby,
            "sapling": inv.sapling, "torches": inv.torches, "arrows": inv.arrows,
            "pickaxe": inv.pickaxe, "sword": inv.sword, "bow": inv.bow, "armour": inv.armour,
            "potions": inv.potions, "books": inv.books,
            "map": state.map[lvl], "items": state.item_map[lvl], "light": state.light_map[lvl],
            "coord": state.coordination_map[lvl], "soft": state.soft_coordination_mask[lvl],
            "pending": state.pending_handovers, "kills": state.monsters_killed[lvl],
            "t": state.timestep,
        }
        for kind in ("melee", "ranged", "passive"):
            mobs = getattr(state, f"{kind}_mobs")
            fields[f"{kind}_pos"] = mobs.position[lvl]
            fields[f"{kind}_mask"] = mobs.mask[lvl]
            fields[f"{kind}_type"] = mobs.type_id[lvl]
            coord = getattr(state, f"{kind}_mob_coordination", None)
            req = getattr(state, f"{kind}_mob_agents_required", None)
            fields[f"{kind}_coord"] = coord[lvl] if coord is not None else jnp.zeros_like(mobs.mask[lvl], dtype=jnp.int32)
            fields[f"{kind}_req"] = req[lvl] if req is not None else jnp.zeros_like(mobs.mask[lvl], dtype=jnp.int32)
        snap = {k: np.asarray(v) for k, v in jax.device_get(fields).items()}
        snap["level"] = lvl
        self._snap_state, self._snap = state, snap
        return snap

    def _map_surface(self, state, player):
        """The 9×11 map for ``player`` as a scaled Surface (cached per state/player)."""
        key = (id(state), player)
        if self._map_cache_key == key and self._map_cache_state is state:
            return self._map_cache
        all_pixels = self._render(
            state, BLOCK_PIXEL_SIZE_HUMAN, self.static_params, self._player_textures
        )
        pixels = np.asarray(all_pixels[player])
        if pixels.dtype != np.uint8:
            pixels = np.clip(pixels, 0, 255).astype(np.uint8)
        # The pixel obs stacks [teammate strip | map | inventory]; keep the map rows only
        # (the strip and inventory are redrawn with labels by this renderer).
        strip_rows = (self.num_players + 1) // 2 * BLOCK_PIXEL_SIZE_HUMAN
        pixels = pixels[strip_rows : strip_rows + OBS_DIM[0] * BLOCK_PIXEL_SIZE_HUMAN]
        surf = pygame.surfarray.make_surface(pixels.transpose((1, 0, 2)))
        if (self.map_w, self.map_h) != surf.get_size():
            scaler = pygame.transform.scale if self.block_px % BLOCK_PIXEL_SIZE_HUMAN == 0 else pygame.transform.smoothscale
            surf = scaler(surf, (self.map_w, self.map_h))
        self._map_cache_key, self._map_cache_state, self._map_cache = key, state, surf
        return surf

    # ── frame ───────────────────────────────────────────────────────────────

    def render(self, env_state, player=0, score=0.0):
        self.screen_surface.fill(C_BG)
        snap = self._snapshot(env_state)
        mask = self._current_mask(env_state)
        self.screen_surface.blit(self._map_surface(env_state, player), (0, self._game_y))
        self._draw_top_bar(snap, player)
        self._draw_map_overlay(snap, player)
        self._draw_status(snap, player, score)
        self._draw_log()
        self._draw_sidebar(snap, player, mask)

    def _player_label(self, snap, p):
        role = _HUD_SPEC_NAMES.get(int(snap["spec"][p]), "")
        return f"P{p + 1}" + (f" · {role}" if role else "")

    def _dot(self, x, cy, p, size=None):
        """A player's own sprite (the character they see on the map) as a small badge."""
        size = size or self.px(20)
        icon = self._player_icon(p, size)
        if icon is None:
            r = size // 2
            pygame.draw.circle(self.screen_surface, self._player_colors[p % len(self._player_colors)], (x + r, cy), r)
        else:
            self.screen_surface.blit(icon, (x, cy - size // 2))
        return size

    def _pcol(self, p):
        return self._player_colors[p % len(self._player_colors)]

    def _draw_top_bar(self, snap, current_player):
        """Top strip: the players (current one named as 'your move'), floor and step."""
        sw, strip_h = self.map_w, self._top_h
        cy = strip_h // 2
        x = self._pad
        gap = self._pad * 2
        for p in range(self.num_players):
            x += self._dot(x, cy, p, self.px(24)) + self.px(6)
            current = p == current_player
            font = self._font_legend if current else self._font_label
            r = self._blit_text(font, self._player_label(snap, p), self._pcol(p) if current else C_DIM, x, cy, anchor="midleft")
            x = r.right
            if current:
                x = self._blit_text(self._font_label, "your move", C_ACCENT, x + self.px(8), cy, anchor="midleft").right
            x += gap

        lvl = snap["level"]
        name = LEVEL_NAMES[lvl] if lvl < len(LEVEL_NAMES) else f"Level {lvl}"
        kills = int(snap["kills"])
        if lvl >= len(LEVEL_NAMES) - 1:
            ladder = ""
        elif kills >= MONSTERS_KILLED_TO_CLEAR_LEVEL:
            ladder = "Ladder down open"
        else:
            ladder = f"Kills {kills}/{MONSTERS_KILLED_TO_CLEAR_LEVEL} to open the ladder"
        # Right-aligned floor/step info, trimmed until it clears the player names.
        candidates = [
            f"{name}   ·   Step {int(snap['t'])}" + (f"   ·   {ladder}" if ladder else ""),
            f"{name}   ·   Step {int(snap['t'])}",
            f"Step {int(snap['t'])}",
        ]
        for info in candidates:
            surf = self._font_label.render(info, True, C_DIM)
            if sw - self._pad - surf.get_width() >= x:
                self.screen_surface.blit(surf, surf.get_rect(midright=(sw - self._pad, cy)))
                break
        pygame.draw.line(self.screen_surface, C_LINE, (0, strip_h - 1), (sw, strip_h - 1), 1)

    def _draw_map_overlay(self, snap, player):
        """Frame the map in the current player's colour and draw toasts/banners."""
        col = self._player_colors[player % len(self._player_colors)]
        rect = pygame.Rect(0, self._game_y, self.map_w, self.map_h)
        pygame.draw.rect(self.screen_surface, col, rect, width=self.px(4))

        # Who-is-this chip in the corner of the map.
        tag = self._player_label(snap, player)
        if not snap["alive"][player]:
            tag += " · downed"
        elif snap["sleeping"][player]:
            tag += " · asleep, turn skipped"
        elif snap["resting"][player]:
            tag += " · resting"
        self._pill(tag, rect.x + self.px(10), rect.y + self.px(10), font=self._font_status, border=col)

        now = pygame.time.get_ticks()
        # Short flash (reward / damage / achievement) at the top of the map.
        if self._toast is not None:
            exp, text, color = self._toast
            if now < exp:
                w = self._font_event.size(text)[0] + self.px(24)
                self._pill(text, rect.centerx - w // 2, rect.y + self.px(10), font=self._font_event, color=_theme_color(color))
            else:
                self._toast = None
        # Sticky banner (e.g. "press ENTER") in the upper third of the map.
        if self._notification is not None:
            exp, text, color = self._notification
            if now < exp:
                w = self._font_banner.size(text)[0] + self.px(40)
                self._pill(text, rect.centerx - w // 2, rect.y + rect.height // 3, font=self._font_banner, pad=self.px(14))
            else:
                self._notification = None

    def _pill(self, text, x, y, font=None, color=C_TEXT, border=None, pad=None):
        """White rounded label drawn over the map."""
        font = font or self._font_label
        pad = pad or self.px(7)
        surf = font.render(text, True, color)
        box = pygame.Rect(x, y, surf.get_width() + pad * 2, surf.get_height() + pad)
        pygame.draw.rect(self.screen_surface, (255, 255, 255), box, border_radius=box.height // 2)
        pygame.draw.rect(self.screen_surface, border or C_LINE, box, width=self.px(1), border_radius=box.height // 2)
        self.screen_surface.blit(surf, (box.x + pad, box.y + pad // 2))
        return box

    def _draw_status(self, snap, p, score):
        """Strip under the map: identity + attributes, stat bars, inventory, facing."""
        px, pad = self.px, self._pad
        x0, y0, w = 0, self._status_y, self.map_w
        left, right = x0 + pad, x0 + w - pad

        # Row A: who you are, score, attributes.
        ya = y0 + px(5)
        cy = ya + self._font_status.get_height() // 2
        x = left + self._dot(left, cy, p, px(22)) + px(6)
        r = self._blit_text(self._font_status, self._player_label(snap, p), self._pcol(p), x, ya)
        r = self._blit_text(self._font_label, "you", C_ACCENT, r.right + px(6), cy, anchor="midleft")
        attrs = (
            f"Score {score:+.2f}   ·   XP {int(snap['xp'][p])}   ·   "
            f"Dex {int(snap['dex'][p])}   Str {int(snap['str'][p])}   Int {int(snap['int'][p])}"
        )
        attrs_rect = self._blit_text(self._font_label, attrs, C_DIM, right, cy, anchor="midright")
        # Status tag in the gap, long form if it fits, short form otherwise. (An open
        # request is not repeated here — the TEAM panel shows it with its countdown.)
        tag = None
        if not snap["alive"][p]:
            tag = ("Downed — a teammate must face you and press Space", "Downed", C_BAD)
        elif snap["sleeping"][p]:
            tag = ("Asleep — turns are skipped until energy is full", "Asleep", C_WARN)
        elif snap["resting"][p]:
            tag = ("Resting — heals while food, drink and energy are above 0", "Resting", C_OK)
        if tag is not None:
            avail = attrs_rect.left - pad - (r.right + pad)
            for text in tag[:2]:
                if self._font_label.size(text)[0] <= avail:
                    self._blit_text(self._font_label, text, tag[2], r.right + pad, cy, anchor="midleft")
                    break

        # Row B: stat bars.
        yb = y0 + px(30)
        stats = [
            ("hp", "HP", "health_texture"), ("food", "Food", "hunger_texture"),
            ("drink", "Drink", "thirst_texture"), ("energy", "Energy", "energy_texture"),
            ("mana", "Mana", "mana_texture"),
        ]
        pitch = (w - 2 * pad) // len(stats)
        isz = px(18)
        for i, (key, label, tex) in enumerate(stats):
            x = left + i * pitch
            cur, mx = int(snap[key][p]), max(1, int(snap["max_" + key][p]))
            icon = self._icon((tex, None), isz)
            if icon is not None:
                self.screen_surface.blit(icon, (x, yb))
            bar_x = x + isz + px(4)
            bar_w = pitch - isz - px(52)
            self._bar(bar_x, yb + px(3), bar_w, px(12), cur / mx, BAR_COLORS[key])
            color = C_BAD if (key == "hp" and cur <= 3) or (key != "hp" and cur <= 2) else C_TEXT
            self._blit_text(self._font_label, f"{cur}/{mx}", color, bar_x + bar_w + px(4), yb + px(9), anchor="midleft")

        # Row C1: materials (always shown so the layout is stable).
        yc = y0 + px(54)
        materials = [
            ("Wood", "wood", ("smaller_block_textures", BlockType.WOOD.value)),
            ("Stone", "stone", ("smaller_block_textures", BlockType.STONE.value)),
            ("Coal", "coal", ("smaller_block_textures", BlockType.COAL.value)),
            ("Iron", "iron", ("smaller_block_textures", BlockType.IRON.value)),
            ("Diamond", "diamond", ("smaller_block_textures", BlockType.DIAMOND.value)),
            ("Sapphire", "sapphire", ("smaller_block_textures", BlockType.SAPPHIRE.value)),
            ("Ruby", "ruby", ("smaller_block_textures", BlockType.RUBY.value)),
            ("Sapling", "sapling", ("sapling_texture", None)),
            ("Torches", "torches", ("torch_inv_texture", None)),
            ("Arrows", "arrows", ("player_projectile_textures", 0)),
        ]
        pitch = min(px(66), (w - 2 * pad) // len(materials))
        for i, (name, key, spec) in enumerate(materials):
            n = int(snap[key][p])
            self._slot(left + i * pitch, yc, pitch, self._icon(spec, px(28)), n, name, dim=(n == 0))

        # Row C2: gear and consumables (only the pieces you own take a slot).
        yd = yc + px(56)
        gear = []
        tier = int(snap["pickaxe"][p])
        gear.append((f"{TIER_NAMES[tier].capitalize()} pickaxe" if tier else "No pickaxe", ("pickaxe_textures", tier or 1), None, tier == 0, None))
        tier = int(snap["sword"][p])
        gear.append((f"{TIER_NAMES[tier].capitalize()} sword" if tier else "No sword", ("sword_textures", tier or 1), None, tier == 0, None))
        has_bow = int(snap["bow"][p]) > 0
        gear.append(("Bow" if has_bow else "No bow", ("bow_textures", 1), None, not has_bow, None))
        armour = [int(a) for a in snap["armour"][p]]
        if any(armour):
            for i, t in enumerate(armour):
                if t > 0:
                    gear.append((f"{ARMOUR_TIERS[t].capitalize()} {ARMOUR_PIECES[i].lower()}", ("armour_textures", (t, i)), None, False, None))
        else:
            gear.append(("No armour", ("armour_textures", (1, 1)), None, True, None))
        books = int(snap["books"][p])
        gear.append(("Book", ("book_texture", None), books, books == 0, None))
        if bool(snap["spells"][p]):
            heal = int(snap["spec"][p]) == Specialization.FORAGER.value
            gear.append(("Heal spell" if heal else "Fireball", ("heal_inv_texture" if heal else "fireball_inv_texture", None), None, False, "G"))
        potions = [int(c) for c in snap["potions"][p]]
        if any(potions):
            for i, c in enumerate(potions):
                if c > 0:
                    gear.append((f"{POTION_NAMES[i]} potion", ("potion_textures", i), c, False, None))
        else:
            gear.append(("Potions", ("potion_textures", 0), 0, True, None))
        pitch = min(px(66), (w - 2 * pad) // max(1, len(gear)))
        for i, (name, spec, count, dim, count_text) in enumerate(gear):
            self._slot(left + i * pitch, yd, pitch, self._icon(spec, px(28)), count, name, dim=dim, count_text=count_text)

        # Row D: what you are facing. The tile name stays charcoal; the rule after
        # the dash takes the colour of the matching border on the map.
        text, color = self._describe_facing(snap, p)
        head, sep, rule = text.partition(" — ")
        r = self._blit_text(self._font_label, head, C_TEXT, left, y0 + self._status_h - px(8), anchor="bottomleft")
        if rule:
            self._blit_text(self._font_label, f" — {rule}", color, r.right, r.bottom, anchor="bottomleft")
        pygame.draw.line(self.screen_surface, C_LINE, (x0, y0 + self._status_h - 1), (x0 + w, y0 + self._status_h - 1), 1)

    def _describe_facing(self, snap, p):
        """One line about the tile in front of player ``p`` — the same facts the text
        observation gives agents (coordination rule, elite rule, ladder state)."""
        n = self.num_players
        pos = snap["pos"][p]
        d = int(snap["dir"][p])
        dr, dc = int(_DIRS[d][0]), int(_DIRS[d][1])
        if dr == 0 and dc == 0:
            return ("Facing: nothing yet — press a move key to turn", C_DIM)
        r, c = int(pos[0]) + dr, int(pos[1]) + dc
        where = f"Facing {DIRECTION_NAMES.get(d, '')}: "
        h, w = snap["map"].shape
        if not (0 <= r < h and 0 <= c < w):
            return (where + "the edge of the map", C_DIM)
        if float(snap["light"][r, c]) <= 0.05:
            return (where + "darkness — place a torch (J) to see", C_DIM)

        for q in range(n):
            if q != p and int(snap["pos"][q][0]) == r and int(snap["pos"][q][1]) == c:
                who = self._player_label(snap, q)
                if not snap["alive"][q]:
                    return (where + f"{who} is DOWNED — press Space to revive them", C_OK)
                return (where + f"{who} — Space would hit them (friendly fire)", C_WARN)

        for kind, names in (("melee", MELEE_MOB_NAMES), ("ranged", RANGED_MOB_NAMES), ("passive", PASSIVE_MOB_NAMES)):
            mask, mpos = snap[f"{kind}_mask"], snap[f"{kind}_pos"]
            for i in range(mask.shape[0]):
                if mask[i] and int(mpos[i][0]) == r and int(mpos[i][1]) == c:
                    t = int(snap[f"{kind}_type"][i])
                    name = _humanize(names[t]) if t < len(names) else "Creature"
                    coord = int(snap[f"{kind}_coord"][i])
                    req = max(2, int(snap[f"{kind}_req"][i]))
                    food = " (food)" if kind == "passive" else ""
                    if coord == 2:
                        return (where + f"{name}{food} — ELITE, hard: {req} of you must attack it in the same step; hits alone do nothing", C_HARD)
                    if coord == 1:
                        return (where + f"{name}{food} — ELITE, soft: {req} attacking together = 2× damage; alone = half damage", C_SOFT)
                    verb = "Space to attack" + (" / eat" if kind == "passive" else "")
                    return (where + f"{name}{food} — {verb}", C_TEXT)

        block = int(snap["map"][r, c])
        item = int(snap["items"][r, c])
        kills = int(snap["kills"])
        if item == ItemType.LADDER_DOWN.value:
            if kills >= MONSTERS_KILLED_TO_CLEAR_LEVEL:
                return (where + "LADDER DOWN — open: stand on it and press . (the whole team goes)", C_OK)
            return (where + f"LADDER DOWN — locked until {MONSTERS_KILLED_TO_CLEAR_LEVEL} monsters are killed on this floor ({kills}/{MONSTERS_KILLED_TO_CLEAR_LEVEL})", C_WARN)
        if item == ItemType.LADDER_UP.value:
            return (where + "LADDER UP — stand on it and press , to go back up", C_TEXT)

        try:
            name = _humanize(BlockType(block).name)
        except ValueError:
            name = "Unknown"
        is_site = block in (BlockType.CONSTRUCTION_SITE.value, BlockType.CONSTRUCTION_IN_PROGRESS.value)
        val = int(snap["coord"][r, c])
        soft = bool(snap["soft"][r, c])
        verb = "press a Build key (9 / 0 / `)" if is_site else "press Space"
        color, extra = C_TEXT, ""

        # An in-flight handover at this tile beats everything else: finish it.
        pending = snap["pending"]
        for i in range(pending.shape[0]):
            if int(pending[i, 0]) >= 1 and int(pending[i, 1]) == r and int(pending[i, 2]) == c:
                remaining = int(pending[i, 3]) - int(snap["t"])
                if remaining > 0:
                    starter = int(pending[i, 4])
                    return (where + f"{name} — handover started by P{starter + 1}: {verb} within {remaining} steps to complete it!", C_HANDOVER)

        if val > 0 and soft:
            color, extra = C_SOFT, f" — SOFT coordination: works alone, bonus if {val} {verb} together"
        elif val > 0:
            color, extra = C_HARD, f" — HARD coordination: {val} of you must {verb} in the same step (fails alone)"
        elif val < 0:
            color, extra = C_HANDOVER, f" — HANDOVER: one of you starts ({verb}), another finishes within {abs(val)} steps"

        need = PICKAXE_NEEDED.get(block)
        if need is not None:
            have = int(snap["pickaxe"][p])
            if have < need:
                extra += f" · needs a {TIER_NAMES[need]} pickaxe (you: {TIER_NAMES[have]})"
                color = C_WARN if color == C_TEXT else color
            else:
                extra += " · Space to mine"
        elif block == BlockType.TREE.value:
            extra += " · Space to chop wood"
        elif block == BlockType.WATER.value:
            extra += " · Space to drink (forager task)"
        elif block == BlockType.GRASS.value:
            extra += " · Space may give a sapling (forager task)"
        elif block == BlockType.RIPE_PLANT.value:
            extra += " · Space to eat (+4 food)"
        elif block == BlockType.PLANT.value:
            extra += " · still growing"
        elif block == BlockType.CHEST.value:
            extra += " · Space to open"
        elif block in (BlockType.CRAFTING_TABLE.value, BlockType.FURNACE.value):
            extra += " · craft while standing next to it"
        elif block == BlockType.CONSTRUCTION_IN_PROGRESS.value:
            extra += " · a build is under way here"
        return (where + name + extra, color)

    def _draw_log(self):
        """Bottom strip: recent events (newest last) and the key hints."""
        px, pad = self.px, self._pad
        y0, w = self._log_y, self.map_w
        self._header("Recent", pad, y0 + px(6))
        line_h = self._font_label.get_height() + px(2)
        y = y0 + px(6) + self._font_head.get_height() + px(2)
        events = list(self._events)
        for i, (text, color) in enumerate(events):
            col = _theme_color(color) if i == len(events) - 1 else C_DIM
            self._blit_text(self._font_label, text, col, pad, y)
            y += line_h
        self._blit_text(self._font_hints, _HUD_HINTS[0], C_FAINT, w - pad, y0 + px(8), anchor="topright")

    # ── sidebar ─────────────────────────────────────────────────────────────

    def _draw_sidebar(self, snap, me, mask):
        px, pad = self.px, self._pad
        x0, pw, sh = self.map_w, self._panel_w, self.screen_size[1]
        pygame.draw.line(self.screen_surface, C_LINE, (x0, 0), (x0, sh), 1)
        left, right = x0 + pad * 2, x0 + pw - pad
        alive = bool(snap["alive"][me])

        def usable(action):
            if action is None or mask is None:
                return "neutral"
            return "on" if alive and bool(mask[me, action_index(action)]) else "off"

        y = pad
        y = self._draw_team(snap, me, mask, left, right, y)
        y = self._draw_requests(snap, me, left, right, y + px(6))
        y = self._draw_give(snap, me, mask, left, right, y + px(6))
        self._draw_actions(snap, me, usable, left, right, y + px(6))

    def _header(self, text, x, y, note=None):
        r = self._blit_text(self._font_head, text.upper(), C_HEAD, x, y)
        if note:
            self._blit_text(self._font_small, note, C_FAINT, r.right + self.px(8), r.centery, anchor="midleft")
        return r.bottom + self.px(4)

    def _draw_team(self, snap, me, mask, left, right, y):
        px = self.px
        y = self._header("Team", left, y)
        n = self.num_players
        line_h = self._font_label.get_height() + px(2)
        for p in range(n):
            cy = y + self._font_status.get_height() // 2
            sw = self._dot(left, cy, p, px(22))
            label = self._player_label(snap, p)
            r = self._blit_text(self._font_status, label, self._pcol(p), left + sw + px(6), y)
            x = r.right + px(8)
            if p == me:
                x = self._blit_text(self._font_label, "you", C_ACCENT, x, cy, anchor="midleft").right + px(10)
            hp, mx = int(snap["hp"][p]), max(1, int(snap["max_hp"][p]))
            hp_col = C_BAD if hp <= 3 else C_DIM
            x = self._blit_text(self._font_label, f"HP {hp}/{mx}", hp_col, x, cy, anchor="midleft").right + px(10)
            if not snap["alive"][p]:
                status = ("downed", C_BAD)
            elif snap["sleeping"][p]:
                status = ("asleep", C_WARN)
            elif snap["resting"][p]:
                status = ("resting", C_OK)
            else:
                status = None
            if status:
                x = self._blit_text(self._font_label, status[0], status[1], x, cy, anchor="midleft").right + px(8)
            if p != me:
                dr = int(snap["pos"][p][0]) - int(snap["pos"][me][0])
                dc = int(snap["pos"][p][1]) - int(snap["pos"][me][1])
                if dr == 0 and dc == 0:
                    rel = "same tile"
                else:
                    parts = []
                    if dr:
                        parts.append(f"{abs(dr)} {'N' if dr < 0 else 'S'}")
                    if dc:
                        parts.append(f"{abs(dc)} {'W' if dc < 0 else 'E'}")
                    rel = " ".join(parts)
                    if abs(dr) > OBS_DIM[0] // 2 or abs(dc) > OBS_DIM[1] // 2:
                        rel += " (off-screen)"
                self._blit_text(self._font_small, rel, C_FAINT, right, cy, anchor="midright")
            y += line_h
            # Request line.
            x = left + sw + px(6)
            if snap["req_dur"][p] > 0:
                idx = int(snap["req_type"][p]) - Action.REQUEST_FOOD.value
                name = REQUEST_NAMES[idx] if 0 <= idx < len(REQUEST_NAMES) else "?"
                who = "You asked for" if p == me else "Wants"
                r = self._blit_text(self._font_label, f"{who} {name.lower()} · {int(snap['req_dur'][p])} steps left", C_ACCENT, x, y)
                if p != me:
                    slot = next((k for k in range(n - 1) if give_target(me, k, n) == p), None)
                    if slot is not None and slot < len(GIVE_KEY_LABELS):
                        legal, note = self._give_status(snap, mask, me, p, slot)
                        cap = self._keycap(r.right + px(10), y - px(1), px(44), GIVE_KEY_LABELS[slot], "on" if legal else "off")
                        self._blit_text(self._font_small, note, C_TEXT if legal else C_FAINT, cap.right + px(6), cap.centery, anchor="midleft")
            else:
                self._blit_text(self._font_small, "No request", C_FAINT, x, y + px(1))
            y += line_h + px(4)
        return y

    def _draw_requests(self, snap, me, left, right, y):
        px = self.px
        y = self._header("Ask the team", left, y, "teammates see it for 10 steps")
        cols = 3
        cell_w = (right - left) // cols
        cell_h = self._font_key.get_height() + px(10)
        my_req = int(snap["req_type"][me]) - Action.REQUEST_FOOD.value if snap["req_dur"][me] > 0 else -1
        for i, (key, action, name, icon_spec) in enumerate(REQUESTS):
            cx = left + (i % cols) * cell_w
            cy = y + (i // cols) * cell_h
            style = "active" if i == my_req else "neutral"
            cap = self._keycap(cx, cy, px(30), key, style)
            icon = self._icon(icon_spec, px(16))
            x = cap.right + px(6)
            if icon is not None:
                self.screen_surface.blit(icon, (x, cap.centery - px(8)))
                x += px(20)
            self._blit_text(self._font_label, name, C_ACCENT if i == my_req else C_TEXT, x, cap.centery, anchor="midleft")
        return y + 3 * cell_h

    def _draw_give(self, snap, me, mask, left, right, y):
        px = self.px
        n = self.num_players
        y = self._header("Give", left, y, "one unit per press, while they have a request")
        row_h = self._font_key.get_height() + px(10)
        for slot in range(n - 1):
            target = give_target(me, slot, n)
            if target is None:
                continue
            key = GIVE_KEY_LABELS[slot] if slot < len(GIVE_KEY_LABELS) else "—"
            legal, note = self._give_status(snap, mask, me, target, slot)
            style = "on" if legal else "off"
            cap = self._keycap(left, y, px(44), key, style)
            who = self._player_label(snap, target)
            r = self._blit_text(self._font_label, f"to {who}", C_TEXT if legal else C_DIM, cap.right + px(8), cap.centery, anchor="midleft")
            color = C_ACCENT if legal else C_FAINT
            self._blit_text(self._font_small, note, color, r.right + px(10), cap.centery, anchor="midleft")
            y += row_h
        return y

    def _give_status(self, snap, mask, me, target, slot):
        """(legal, note) for giving from ``me`` to ``target`` via give slot ``slot``."""
        legal = (
            mask is not None
            and bool(mask[me, Action.GIVE.value + slot])
            and bool(snap["alive"][me])
        )
        if snap["req_dur"][target] <= 0:
            return False, f"P{target + 1} has no open request"
        idx = int(snap["req_type"][target]) - Action.REQUEST_FOOD.value
        if not 0 <= idx < len(REQUESTS):
            return legal, ""
        name = REQUEST_NAMES[idx]
        if legal:
            return True, f"give 1 {name} now"
        if not snap["alive"][me]:
            return False, "you are downed"
        if snap["sleeping"][me]:
            return False, "you are asleep"
        if not snap["alive"][target]:
            return False, f"P{target + 1} is downed"
        field = REQUEST_FIELDS[idx]
        if int(snap[field][me]) <= 0:
            return False, f"you have no {name.lower()}"
        cap = int(snap["max_" + field][target]) if field in ("food", "drink") else 99
        if int(snap[field][target]) >= cap:
            return False, f"P{target + 1} is already full"
        return False, "not possible right now"

    def _draw_actions(self, snap, me, usable, left, right, y):
        px = self.px
        y = self._header("Keys", left, y, "white keys would work right now")
        col_w = (right - left) // 2
        key_w = px(50)
        key_h = self._font_key.get_height() + px(4)
        row_h = key_h + px(3)
        isz = px(14)
        potions = [int(c) for c in snap["potions"][me]]
        for ci, groups in enumerate(self._panel_groups):
            x = left + ci * col_w
            yy = y
            for title, rows in groups:
                yy = self._blit_text(self._font_small, title, C_HEAD, x, yy).bottom + px(2)
                for key_label, label, action, icon_spec in rows:
                    style = usable(action)
                    cap = self._keycap(x, yy, key_w, key_label, style, h=key_h)
                    fg = C_TEXT if style != "off" else C_FAINT
                    tx = cap.right + px(6)
                    icon = self._icon(icon_spec, isz)
                    if icon is not None:
                        if style == "off":
                            icon = icon.copy()
                            icon.set_alpha(70)
                        self.screen_surface.blit(icon, (tx, cap.centery - isz // 2))
                        tx += isz + px(4)
                    if icon_spec and icon_spec[0] == "potion_textures":
                        label = f"{label} ×{potions[icon_spec[1]]}"
                    self._blit_text(self._font_label, label, fg, tx, cap.centery, anchor="midleft")
                    yy += row_h
                yy += px(4)

    # ── input ───────────────────────────────────────────────────────────────

    def is_quit_requested(self):
        for event in self.pygame_events:
            if event.type == pygame.QUIT:
                return True
        return False

    def get_action_from_keypress(self, state, player=0):
        if state.is_sleeping[player] or not state.player_alive[player]:
            return Action.NOOP.value

        for event in self.pygame_events:
            if event.type == pygame.KEYDOWN:
                if event.key in KEY_MAPPING:
                    return action_index(KEY_MAPPING[event.key])
        return None

    # ── full-screen pages ───────────────────────────────────────────────────

    def _wrap_page(self, body, width):
        """Turn markdown-ish text into (font, colour, text) lines wrapped to width."""
        head_col, body_col = C_ACCENT, C_TEXT
        out = []

        def wrap(text, font, color):
            cur = ""
            for word in text.split(" "):
                trial = (cur + " " + word).strip()
                if not cur or font.size(trial)[0] <= width:
                    cur = trial
                else:
                    out.append((font, color, cur))
                    cur = word
            out.append((font, color, cur))

        for raw in body.split("\n"):
            line = raw.replace("**", "").replace("`", "").rstrip()
            if not line.strip():
                out.append((self._font_page_body, body_col, ""))  # blank spacer
            elif line.startswith("## "):
                out.append((self._font_page_body, body_col, ""))
                wrap(line[3:].strip(), self._font_page_head, head_col)
            elif line.startswith("# "):
                wrap(line[2:].strip(), self._font_page_head, head_col)
            elif line.startswith("- "):
                wrap("•  " + line[2:].strip(), self._font_page_body, body_col)
            else:
                wrap(line, self._font_page_body, body_col)
        return out

    def show_text_screen(self, title, body, footer="ENTER to continue"):
        """Show a scrollable full-window text page. Blocks until the user proceeds.

        Returns True when the user presses ENTER, False if they close the window.
        Scroll with Up/Down, PageUp/PageDown/Space, or the mouse wheel.
        """
        sw, sh = self.screen_size
        margin = self._pad * 2
        lines = self._wrap_page(body, sw - margin * 2)
        line_h = self._font_page_body.get_height() + 4
        title_surf = self._font_page_title.render(title, True, C_TEXT)
        footer_surf = self._font_hints.render(footer, True, C_DIM)
        top = margin + title_surf.get_height() + self._pad
        bottom = sh - self._pad * 2 - footer_surf.get_height()
        view_h = bottom - top
        max_scroll = max(0, len(lines) * line_h - view_h)
        scroll = 0
        clock = pygame.time.Clock()

        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    return False
                if e.type == pygame.KEYDOWN:
                    if e.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        return True
                    if e.key == pygame.K_ESCAPE:
                        return False
                    if e.key == pygame.K_DOWN:
                        scroll = min(max_scroll, scroll + line_h * 2)
                    elif e.key == pygame.K_UP:
                        scroll = max(0, scroll - line_h * 2)
                    elif e.key in (pygame.K_PAGEDOWN, pygame.K_SPACE):
                        scroll = min(max_scroll, scroll + view_h)
                    elif e.key == pygame.K_PAGEUP:
                        scroll = max(0, scroll - view_h)
                elif e.type == pygame.MOUSEWHEEL:
                    scroll = min(max_scroll, max(0, scroll - e.y * line_h * 3))

            self.screen_surface.fill(C_BG)
            self.screen_surface.blit(title_surf, (margin, self._pad))

            clip = self.screen_surface.get_clip()
            self.screen_surface.set_clip(pygame.Rect(margin, top, sw - margin * 2, view_h))
            y = top - scroll
            for font, color, text in lines:
                if text and y + line_h > top and y < bottom:
                    self.screen_surface.blit(font.render(text, True, color), (margin, y))
                y += line_h
            self.screen_surface.set_clip(clip)

            if max_scroll > 0:  # scrollbar
                bar_h = max(24, int(view_h * view_h / (len(lines) * line_h)))
                bar_y = top + int((view_h - bar_h) * scroll / max_scroll)
                pygame.draw.rect(
                    self.screen_surface, C_LINE, (sw - self._pad, bar_y, 5, bar_h),
                    border_radius=2,
                )
            self.screen_surface.blit(
                footer_surf, ((sw - footer_surf.get_width()) // 2, sh - self._pad - footer_surf.get_height())
            )
            pygame.display.flip()
            clock.tick(60)

    def show_menu(self, title, options, subtitle=None):
        """Show a full-window keyboard-navigable menu. Blocks until a choice is made.

        ``options`` is a list of ``label`` strings or ``(label, description)``
        tuples. Navigate with Up/Down or W/S, choose with ENTER/SPACE, cancel
        with ESCAPE.

        Returns the selected index, or None if cancelled / window closed.
        """
        sw, sh = self.screen_size
        margin = self._pad * 2
        norm = [(o, "") if isinstance(o, str) else (o[0], o[1] if len(o) > 1 else "") for o in options]

        title_surf = self._font_page_title.render(title, True, C_TEXT)
        sub_surf = self._font_hints.render(subtitle, True, C_DIM) if subtitle else None
        footer_surf = self._font_hints.render(
            "↑ / ↓ select   ·   ENTER choose   ·   ESC cancel", True, C_DIM
        )

        selected = 0
        clock = pygame.time.Clock()

        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    return None
                if e.type == pygame.KEYDOWN:
                    if e.key in (pygame.K_UP, pygame.K_w):
                        selected = (selected - 1) % len(norm)
                    elif e.key in (pygame.K_DOWN, pygame.K_s):
                        selected = (selected + 1) % len(norm)
                    elif e.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                        return selected
                    elif e.key == pygame.K_ESCAPE:
                        return None

            self.screen_surface.fill(C_BG)
            self.screen_surface.blit(title_surf, (margin, self._pad))
            y = self._pad + title_surf.get_height() + self._pad // 2
            if sub_surf is not None:
                self.screen_surface.blit(sub_surf, (margin, y))
                y += sub_surf.get_height() + self._pad
            y += self._pad // 2

            for i, (label, desc) in enumerate(norm):
                is_sel = i == selected
                row_bg = (255, 255, 255) if is_sel else C_BG
                label_col = C_TEXT
                desc_col = C_DIM if is_sel else C_FAINT
                label_surf = self._font_page_head.render(label, True, label_col)
                desc_surf = self._font_hints.render(desc, True, desc_col) if desc else None
                row_h = (
                    label_surf.get_height()
                    + (desc_surf.get_height() + 2 if desc_surf else 0)
                    + self._pad
                )
                rect = pygame.Rect(margin, y, sw - margin * 2, row_h)
                pygame.draw.rect(self.screen_surface, row_bg, rect, border_radius=8)
                pygame.draw.rect(
                    self.screen_surface, C_ACCENT if is_sel else C_LINE, rect, width=self.px(1), border_radius=8
                )
                ty = y + self._pad // 2
                self.screen_surface.blit(label_surf, (margin + self._pad, ty))
                if desc_surf is not None:
                    self.screen_surface.blit(
                        desc_surf, (margin + self._pad, ty + label_surf.get_height() + 2)
                    )
                y += row_h + max(4, self._pad // 3)

            self.screen_surface.blit(
                footer_surf,
                ((sw - footer_surf.get_width()) // 2, sh - self._pad - footer_surf.get_height()),
            )
            pygame.display.flip()
            clock.tick(60)


def print_new_achievements(old_achievements, new_achievements, num_players):
    for player in range(num_players):
        for i in range(old_achievements.shape[1]):
            if old_achievements[player, i] == 0 and new_achievements[player, i] == 1:
                total = new_achievements.shape[1]
                unlocked = int(new_achievements[player].sum())
                print(f"  Player {player + 1} achieved {Achievement(i).name} ({unlocked}/{total})")


def main(args):
    env_params = EnvParams()
    if args.god:
        env_params = env_params.replace(god_mode=True)
    if args.coord != "none":
        coord_overrides = get_coordination_params(args.coord)
        env_params = env_params.replace(**coord_overrides)

    env = AlemCoopPixelsEnv(num_agents=args.players, env_params=env_params)
    static_params = env.static_env_params
    num_players = static_params.player_count

    block_px = BLOCK_PIXEL_SIZE_HUMAN
    player_textures = load_player_specific_textures(TEXTURES[block_px], num_players)
    size = resolve_block_size(args.size, num_players)
    print(f"Window: {window_size_for(size, num_players)} px ({size} px per block; override with --size)")

    print("Controls:")
    for k, v in KEY_MAPPING.items():
        print(f"  {pygame.key.name(k):12s}: {action_name(action_index(v)).lower().replace('_', ' ')}")
    print()

    rng = jax.random.PRNGKey(args.seed if args.seed is not None else np.random.randint(2**31))
    rng, _rng = jax.random.split(rng)
    _, env_state = env.reset(_rng)

    renderer = AlemRenderer(
        env, env_params, static_params, player_textures, pixel_render_size=size / block_px
    )
    scores = [0.0] * num_players
    renderer.render(env_state, 0, scores[0])

    current_player = 0
    actions = jnp.zeros(num_players, dtype=jnp.int32)

    step_fn = jax.jit(env.step_env)

    # Pre-compile all JIT-traced functions so the first keypress is instant.
    print("Warming up JAX kernels (first launch only, ~30-60 s)...", end="", flush=True)
    _warmup_actions = {f"agent_{i}": jnp.int32(0) for i in range(num_players)}
    rng, _warm_rng = jax.random.split(rng)
    _, _warmup_state, _, _, _ = step_fn(_warm_rng, env_state, _warmup_actions)
    jax.block_until_ready(_warmup_state)
    print(" done.")

    clock = pygame.time.Clock()
    print(f"Waiting for Player 1/{num_players} input...")

    while not renderer.is_quit_requested():
        action = renderer.get_action_from_keypress(env_state, current_player)

        if action is not None:
            actions = actions.at[current_player].set(action)
            print(f"  Player {current_player + 1}: {action_name(action)}")
            current_player += 1

            # All players have acted — step the environment
            if current_player == num_players:
                rng, _rng = jax.random.split(rng)
                old_achievements = env_state.achievements
                old_health = env_state.player_health

                actions_dict = {f"agent_{i}": actions[i] for i in range(num_players)}
                _, env_state, rewards, dones, info = step_fn(_rng, env_state, actions_dict)

                print_new_achievements(old_achievements, env_state.achievements, num_players)

                # HUD notifications — push in order of priority (last push wins display)
                for p in range(num_players):
                    delta_hp = float(old_health[p]) - float(env_state.player_health[p])
                    if delta_hp >= 1:
                        renderer.push_notification(
                            f"P{p + 1}: −{delta_hp:.0f} HP", color=(255, 80, 80), duration_ms=2000
                        )

                for p in range(num_players):
                    reward = float(rewards[f"agent_{p}"])
                    scores[p] += reward
                    if abs(reward) > 0.01:
                        print(f"  Player {p + 1} reward: {reward:+.2f}")
                    if reward > 0.01:
                        renderer.push_notification(
                            f"P{p + 1}: +{reward:.2f}", color=(100, 230, 100), duration_ms=2000
                        )
                    elif reward < -0.01:
                        renderer.push_notification(
                            f"P{p + 1}: {reward:.2f}", color=(255, 160, 60), duration_ms=2000
                        )

                for p in range(num_players):
                    for i in range(old_achievements.shape[1]):
                        if old_achievements[p, i] == 0 and env_state.achievements[p, i] == 1:
                            name = Achievement(i).name.replace("_", " ").title()
                            renderer.push_notification(
                                f"P{p + 1}: {name}", color=(255, 215, 0), duration_ms=3500
                            )

                if dones.get("__all__", False):
                    print("\n=== EPISODE ENDED ===")
                    if "user_info" in info:
                        for k, v in info["user_info"].items():
                            print(f"  {k}: {v}")
                    print("Restarting...\n")
                    rng, _rng = jax.random.split(rng)
                    _, env_state = env.reset(_rng)
                    scores = [0.0] * num_players

                actions = jnp.zeros(num_players, dtype=jnp.int32)
                current_player = 0
                print(f"Step {int(env_state.timestep)} — Waiting for Player 1/{num_players}...")

        renderer.render(env_state, current_player, scores[current_player])
        renderer.update()
        clock.tick(args.fps)

    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play Alem (Alem-Coop) interactively")
    parser.add_argument("--debug", action="store_true", help="Disable JIT for debugging")
    parser.add_argument("--fps", type=int, default=60, help="Target FPS (default: 60)")
    parser.add_argument("--god", action="store_true", help="God mode (invincible)")
    parser.add_argument("--players", type=int, default=3, help="Number of players (default: 3)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--coord",
        type=str,
        default="easy",
        choices=["none", "easy", "medium", "hard"],
        help="Coordination difficulty (default: easy)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help="Display block size in pixels (default: the largest that fits your screen; e.g. 64 or 128)",
    )

    args, rest_args = parser.parse_known_args(sys.argv[1:])
    if rest_args:
        raise ValueError(f"Unknown args: {rest_args}")

    if args.debug:
        print("JIT disabled (debug mode)")
        with jax.disable_jit():
            main(args)
    else:
        main(args)
