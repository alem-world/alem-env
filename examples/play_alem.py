"""
Play Alem - interactive Alem-Coop game.

Each player takes turns providing an action, then the environment steps.

Controls:
    Movement: WASD          Interact: SPACE
    Sleep: TAB              Rest: E
    Descend/Ascend: . / ,

    Placement: R=Stone  T=Table  F=Furnace  P=Plant  J=Torch
    Pickaxes:  1=Wood  2=Stone  3=Iron  4=Diamond
    Swords:    5=Wood  6=Stone  7=Iron  8=Diamond
    Armour:    Y=Iron  U=Diamond
    Other:     O=Arrow  [=Torch

    Combat:    I=Shoot Arrow   G=Cast Spell
    Potions:   Z=Red  X=Green  C=Blue  V=Pink  B=Cyan  N=Yellow
    Enchant:   K=Sword  L=Armour  ;=Bow   M=Read Book
    Level Up:  ]=Dex  -=Str  ==Int

    Requests:  F1-F9 (Food..Sapphire)   Backspace=Give
    Build:     9=Shelter  0=Forge  `=Beacon
    No-op:     Q

Usage:
    python examples/play_alem.py                     # Default 3 players
    python examples/play_alem.py --players 2 --god   # 2 players, god mode
    python examples/play_alem.py --seed 42           # Reproducible seed
    python examples/play_alem.py --coord easy        # Coordination difficulty
"""

import argparse
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pygame

from alem.alem_coop.action_masking import compute_action_mask
from alem.alem_coop.alem_state import EnvParams, StaticEnvParams, get_coordination_params
from alem.alem_coop.constants import (
    BLOCK_PIXEL_SIZE_HUMAN,
    INVENTORY_OBS_HEIGHT,
    OBS_DIM,
    TEXTURES,
    Achievement,
    Action,
    load_player_specific_textures,
)
from alem.alem_coop.envs.alem_pixels_env import AlemCoopPixelsEnv
from alem.alem_coop.renderer.renderer_pixels import render_alem_pixels

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
    pygame.K_BACKSPACE: Action.GIVE,
    # Construction
    pygame.K_9: Action.BUILD_SHELTER,
    pygame.K_0: Action.BUILD_FORGE,
    pygame.K_BACKQUOTE: Action.BUILD_BEACON,
}


# Fallback accent colours (used only if the real sprite palette can't be built).
_HUD_COLORS = [
    (100, 210, 255),  # cyan
    (130, 230, 120),  # green
    (255, 200, 80),  # amber
    (210, 140, 255),  # purple
    (255, 130, 130),  # pink
]
_HUD_SPEC_NAMES = {1: "Forager", 2: "Warrior", 3: "Miner"}


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

# 5 most-used controls, shown in the HUD strip
_HUD_HINTS = "WASD: Move  ·  Space: Do  ·  E: Rest  ·  Tab: Sleep  ·  Q: Skip"


def _build_panel_groups():
    """Sidebar layout: (group title, [(key label, action label, Action|None)]).

    The Action enum, when present, is used to look up whether that action is
    legal for the current player this turn (so the row lights up when usable).
    None means "always available" (movement) — shown neutral, never dimmed.
    """
    a = Action
    return [
        ("MOVE", [("W A S D", "Move / turn", None)]),
        ("ACT", [
            ("Space", "Do / interact", a.DO),
            ("E", "Rest", a.REST),
            ("Tab", "Sleep", a.SLEEP),
            ("Q", "Skip turn", a.NOOP),
            (".  ,", "Descend / Ascend", a.DESCEND),
        ]),
        ("PLACE", [
            ("R", "Stone", a.PLACE_STONE),
            ("T", "Table", a.PLACE_TABLE),
            ("F", "Furnace", a.PLACE_FURNACE),
            ("P", "Plant", a.PLACE_PLANT),
            ("J", "Torch", a.PLACE_TORCH),
        ]),
        ("PICKAXE", [
            ("1", "Wood", a.MAKE_WOOD_PICKAXE),
            ("2", "Stone", a.MAKE_STONE_PICKAXE),
            ("3", "Iron", a.MAKE_IRON_PICKAXE),
            ("4", "Diamond", a.MAKE_DIAMOND_PICKAXE),
        ]),
        ("SWORD", [
            ("5", "Wood", a.MAKE_WOOD_SWORD),
            ("6", "Stone", a.MAKE_STONE_SWORD),
            ("7", "Iron", a.MAKE_IRON_SWORD),
            ("8", "Diamond", a.MAKE_DIAMOND_SWORD),
        ]),
        ("CRAFT", [
            ("Y", "Iron armour", a.MAKE_IRON_ARMOUR),
            ("U", "Diamond armour", a.MAKE_DIAMOND_ARMOUR),
            ("O", "Arrow", a.MAKE_ARROW),
            ("[", "Torch", a.MAKE_TORCH),
        ]),
        ("FIGHT", [
            ("I", "Shoot arrow", a.SHOOT_ARROW),
            ("G", "Cast spell", a.CAST_SPELL),
        ]),
        ("TEAM", [
            ("F1-F9", "Request item", a.REQUEST_FOOD),
            ("Bksp", "Give item", a.GIVE),
        ]),
        ("BUILD", [
            ("9", "Shelter", a.BUILD_SHELTER),
            ("0", "Forge", a.BUILD_FORGE),
            ("`", "Beacon", a.BUILD_BEACON),
        ]),
    ]


class AlemRenderer:
    def __init__(self, env, env_params, static_params, player_textures, pixel_render_size=1):
        self.env = env
        self.env_params = env_params
        self.static_params = static_params
        self.num_players = static_params.player_count
        self.pixel_render_size = pixel_render_size
        self.pygame_events = []

        # Real per-player sprite colours, so the HUD/legend match the characters.
        self._player_colors = player_palette(self.num_players)

        dashboard_h = (self.num_players + 1) // 2
        self.game_w = OBS_DIM[1] * BLOCK_PIXEL_SIZE_HUMAN * pixel_render_size
        game_h = (
            (OBS_DIM[0] + INVENTORY_OBS_HEIGHT + dashboard_h)
            * BLOCK_PIXEL_SIZE_HUMAN
            * pixel_render_size
        )
        self.game_h = game_h
        # Top strip: a per-player legend that highlights whose turn it is.
        self._legend_h = 34 * pixel_render_size
        self._game_y = self._legend_h  # game view is blitted below the legend
        # Two-row HUD: a status row (who / event / score) + a controls row.
        self._hud_h = 50 * pixel_render_size
        self._pad = 12 * pixel_render_size
        # Right sidebar: an always-visible action/key reference for the current player.
        self._panel_w = 240 * pixel_render_size
        self.screen_size = (
            self.game_w + self._panel_w,
            self._legend_h + game_h + self._hud_h,
        )

        pygame.init()
        pygame.key.set_repeat(250, 75)
        self.screen_surface = pygame.display.set_mode(self.screen_size)
        pygame.display.set_caption("Alem - Alem-Coop")

        s = pixel_render_size
        self._font_status = load_font(15 * s, bold=True)
        self._font_event = load_font(15 * s, bold=True)
        self._font_hints = load_font(13 * s)
        self._font_legend = load_font(14 * s, bold=True)
        # Right-panel fonts.
        self._font_panel_title = load_font(16 * s, bold=True)
        self._font_panel_head = load_font(12 * s, bold=True)
        self._font_panel_key = load_font(12 * s, bold=True)
        self._font_panel_label = load_font(13 * s)
        # Fonts for the full-screen text pages (rules / briefings).
        self._font_page_title = load_font(26 * s, bold=True)
        self._font_page_head = load_font(18 * s, bold=True)
        self._font_page_body = load_font(15 * s)

        # Per-player legal-action mask for the sidebar (recomputed only when the
        # state object changes, so idle re-renders don't re-run it).
        self._action_mask_fn = jax.jit(
            lambda st: compute_action_mask(st, self.env_params, self.static_params)
        )
        self._mask_cache_state = None
        self._mask_cache = None
        self._panel_groups = _build_panel_groups()

        # JIT compile the render function (block_pixel_size and static_params are static)
        self._render = jax.jit(render_alem_pixels, static_argnums=(1, 2))
        self._player_textures = player_textures
        self._notification = None  # (expires_at_ms, text, color)

    def push_notification(self, text, color=(255, 255, 255), duration_ms=2500):
        self._notification = (pygame.time.get_ticks() + duration_ms, text, color)

    def update(self):
        self.pygame_events = list(pygame.event.get())
        pygame.display.flip()

    def render(self, env_state, player=0, score=0.0):
        self.screen_surface.fill((0, 0, 0))

        all_pixels = self._render(
            env_state, BLOCK_PIXEL_SIZE_HUMAN, self.static_params, self._player_textures
        )
        pixels = all_pixels[player]
        pixels = jnp.repeat(pixels, repeats=self.pixel_render_size, axis=0)
        pixels = jnp.repeat(pixels, repeats=self.pixel_render_size, axis=1)

        surface = pygame.surfarray.make_surface(np.array(pixels).transpose((1, 0, 2)))
        self.screen_surface.blit(surface, (0, self._game_y))
        self._draw_legend(env_state, player)
        self._draw_actions_panel(env_state, player)
        self._draw_hud(env_state, player, score)

    def _draw_legend(self, env_state, current_player):
        """Top strip: one colour-coded chip per player, current player highlighted.

        Answers "who controls whom" at a glance — each chip's swatch is the same
        colour as that player's character on the map, and the player whose turn it
        is (the one centred on screen) is boxed and marked YOU.
        """
        sw = self.game_w  # legend spans the game column; the sidebar owns the right
        strip_h = self._legend_h
        font = self._font_legend
        pygame.draw.rect(self.screen_surface, (16, 16, 20), (0, 0, sw, strip_h))

        swatch = int(font.get_height() * 0.85)
        inner_gap = max(4, self._pad // 3)  # swatch → label
        chip_gap = self._pad * 2  # chip → chip

        chips = []  # (player, label_surface, chip_width)
        for p in range(self.num_players):
            role = _HUD_SPEC_NAMES.get(int(env_state.player_specialization[p]), "")
            label = f"P{p + 1}" + (f" {role}" if role else "")
            if p == current_player:
                label += "  ◀ YOU"
            txt_col = (255, 255, 255) if p == current_player else (185, 185, 185)
            surf = font.render(label, True, txt_col)
            chips.append((p, surf, swatch + inner_gap + surf.get_width()))

        total = sum(w for *_, w in chips) + chip_gap * (len(chips) - 1)
        x = max(self._pad, (sw - total) // 2)
        cy = strip_h // 2
        for p, surf, w in chips:
            col = self._player_colors[p % len(self._player_colors)]
            if p == current_player:
                pygame.draw.rect(
                    self.screen_surface,
                    col,
                    (x - inner_gap, cy - swatch // 2 - 3, w + inner_gap * 2, swatch + 6),
                    width=2,
                    border_radius=4,
                )
            pygame.draw.rect(
                self.screen_surface, col, (x, cy - swatch // 2, swatch, swatch), border_radius=3
            )
            self.screen_surface.blit(surf, (x + swatch + inner_gap, cy - surf.get_height() // 2))
            x += w + chip_gap

    def _current_mask(self, state):
        """Legal-action mask for all players, recomputed only when state changes."""
        if state is not self._mask_cache_state:
            try:
                self._mask_cache = np.asarray(self._action_mask_fn(state))
            except Exception:
                self._mask_cache = None
            self._mask_cache_state = state
        return self._mask_cache

    def _draw_actions_panel(self, state, current_player):
        """Right sidebar: key → action reference for the current player.

        A row lights up green when that action is legal for the player right now
        (e.g. Make Wood Pickaxe brightens next to a table with wood), and dims
        when it isn't — so players always see both the key and whether it works.
        """
        s = self.pixel_render_size
        x0 = self.game_w
        pw = self._panel_w
        sw, sh = self.screen_size
        color = self._player_colors[current_player % len(self._player_colors)]
        pygame.draw.rect(self.screen_surface, (24, 24, 30), (x0, 0, pw, sh))
        pygame.draw.line(self.screen_surface, color, (x0, 0), (x0, sh), max(1, s))

        mask = self._current_mask(state)
        alive = bool(state.player_alive[current_player])
        pad = self._pad
        left = x0 + pad
        y = pad

        title = self._font_panel_title.render("ACTIONS", True, (240, 240, 240))
        self.screen_surface.blit(title, (left, y))
        y += title.get_height() + 2
        sub = self._font_hints.render("green = usable now", True, (150, 150, 155))
        self.screen_surface.blit(sub, (left, y))
        y += sub.get_height() + max(4, 4 * s)

        key_h = self._font_panel_key.get_height()
        row_h = key_h + 5 * s
        keycap_w = int(pw * 0.30)  # fixed-width key column keeps labels aligned

        for group_title, rows in self._panel_groups:
            head = self._font_panel_head.render(group_title, True, (150, 150, 160))
            self.screen_surface.blit(head, (left, y))
            y += head.get_height() + 2 * s
            for key_label, action_label, action in rows:
                if action is None or mask is None:
                    on = None  # neutral (always shown, e.g. movement)
                else:
                    on = alive and bool(mask[current_player, action.value])
                # keycap
                cap_bg = (
                    (46, 46, 54) if on is None else (44, 92, 56) if on else (34, 34, 40)
                )
                cap_fg = (
                    (225, 225, 230) if on is None else (210, 245, 215) if on else (110, 110, 120)
                )
                lbl_fg = cap_fg
                cap = pygame.Rect(left, y, keycap_w, key_h + 2 * s)
                pygame.draw.rect(self.screen_surface, cap_bg, cap, border_radius=4)
                if on:
                    pygame.draw.rect(
                        self.screen_surface, (90, 200, 120), cap, width=max(1, s), border_radius=4
                    )
                key_surf = self._font_panel_key.render(key_label, True, cap_fg)
                self.screen_surface.blit(
                    key_surf,
                    (left + (keycap_w - key_surf.get_width()) // 2, y + s),
                )
                lbl_surf = self._font_panel_label.render(action_label, True, lbl_fg)
                self.screen_surface.blit(
                    lbl_surf,
                    (left + keycap_w + pad // 2, y + (cap.height - lbl_surf.get_height()) // 2),
                )
                y += row_h
            y += 3 * s

    def _draw_hud(self, env_state, player, score):
        sw = self.game_w  # HUD spans the game column; the sidebar owns the right
        y0 = self._game_y + self.game_h
        pad = self._pad
        color = self._player_colors[player % len(self._player_colors)]

        # Two rows: status (top) and controls (bottom)
        row_h = self._hud_h // 2
        row1 = y0 + row_h // 2  # vertical centre of status row
        row2 = y0 + row_h + row_h // 2  # vertical centre of controls row

        def blit_left(surf, cy, x=pad):
            self.screen_surface.blit(surf, (x, cy - surf.get_height() // 2))

        def blit_right(surf, cy, x=None):
            x = sw - pad if x is None else x
            self.screen_surface.blit(surf, (x - surf.get_width(), cy - surf.get_height() // 2))

        def blit_centre(surf, cy):
            self.screen_surface.blit(
                surf, ((sw - surf.get_width()) // 2, cy - surf.get_height() // 2)
            )

        # Accent line separating game from HUD, tinted to the active player
        pygame.draw.line(self.screen_surface, color, (0, y0), (sw, y0), self.pixel_render_size)

        # ── Status row ──────────────────────────────────────────────────────
        # Left: which player you control + role
        spec = _HUD_SPEC_NAMES.get(int(env_state.player_specialization[player]), "")
        label = f"Player {player + 1}/{self.num_players}"
        if spec:
            label += f"  ·  {spec}"
        blit_left(self._font_status.render(label, True, color), row1)

        # Right: health and cumulative score (always visible)
        hp = int(env_state.player_health[player])
        score_col = (
            (120, 230, 120) if score > 0 else (220, 90, 90) if score < 0 else (210, 210, 210)
        )
        score_surf = self._font_status.render(f"Score {score:+.2f}", True, score_col)
        blit_right(score_surf, row1)
        hp_surf = self._font_status.render(f"HP {hp}", True, (235, 235, 235))
        blit_right(hp_surf, row1, x=sw - pad - score_surf.get_width() - pad * 2)

        # Centre: transient event flash (achievement / damage / reward)
        if self._notification is not None:
            exp, ntext, ncolor = self._notification
            if pygame.time.get_ticks() < exp:
                blit_centre(self._font_event.render(ntext, True, ncolor), row1)
            else:
                self._notification = None

        # ── Controls row (always visible, never blocked) ────────────────────
        blit_centre(self._font_hints.render(_HUD_HINTS, True, (165, 165, 165)), row2)

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
                    return KEY_MAPPING[event.key].value
        return None

    def _wrap_page(self, body, width):
        """Turn markdown-ish text into (font, colour, text) lines wrapped to width."""
        head_col, body_col = (255, 210, 120), (220, 220, 220)
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
        title_surf = self._font_page_title.render(title, True, (255, 255, 255))
        footer_surf = self._font_hints.render(footer, True, (190, 190, 190))
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

            self.screen_surface.fill((12, 12, 16))
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
                    self.screen_surface, (90, 90, 100), (sw - self._pad, bar_y, 5, bar_h),
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

        title_surf = self._font_page_title.render(title, True, (255, 255, 255))
        sub_surf = (
            self._font_hints.render(subtitle, True, (190, 190, 190)) if subtitle else None
        )
        footer_surf = self._font_hints.render(
            "↑ / ↓ select   ·   ENTER choose   ·   ESC cancel", True, (190, 190, 190)
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

            self.screen_surface.fill((12, 12, 16))
            self.screen_surface.blit(title_surf, (margin, self._pad))
            y = self._pad + title_surf.get_height() + self._pad // 2
            if sub_surf is not None:
                self.screen_surface.blit(sub_surf, (margin, y))
                y += sub_surf.get_height() + self._pad
            y += self._pad // 2

            for i, (label, desc) in enumerate(norm):
                is_sel = i == selected
                row_bg = (46, 90, 60) if is_sel else (28, 28, 34)
                label_col = (235, 255, 235) if is_sel else (215, 215, 215)
                desc_col = (190, 230, 200) if is_sel else (150, 150, 155)
                label_surf = self._font_page_head.render(label, True, label_col)
                desc_surf = self._font_hints.render(desc, True, desc_col) if desc else None
                row_h = (
                    label_surf.get_height()
                    + (desc_surf.get_height() + 2 if desc_surf else 0)
                    + self._pad
                )
                rect = pygame.Rect(margin, y, sw - margin * 2, row_h)
                pygame.draw.rect(self.screen_surface, row_bg, rect, border_radius=6)
                if is_sel:
                    pygame.draw.rect(
                        self.screen_surface, (110, 220, 140), rect, width=2, border_radius=6
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
    pixel_render_size = args.size // block_px

    print("Controls:")
    for k, v in KEY_MAPPING.items():
        print(f"  {pygame.key.name(k):12s}: {v.name.lower().replace('_', ' ')}")
    print()

    rng = jax.random.PRNGKey(args.seed if args.seed is not None else np.random.randint(2**31))
    rng, _rng = jax.random.split(rng)
    _, env_state = env.reset(_rng)

    renderer = AlemRenderer(
        env, env_params, static_params, player_textures, pixel_render_size=pixel_render_size
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
            print(f"  Player {current_player + 1}: {Action(action).name}")
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
        default=128,
        help="Display block size in pixels (default: 128; use 64 for smaller window)",
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
