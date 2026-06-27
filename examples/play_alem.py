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


# Per-player accent colours
_HUD_COLORS = [
    (100, 210, 255),  # player 0 — cyan
    (130, 230, 120),  # player 1 — green
    (255, 200, 80),  # player 2 — amber
    (210, 140, 255),  # player 3 — purple
    (255, 130, 130),  # player 4 — pink
]
_HUD_SPEC_NAMES = {1: "Forager", 2: "Warrior", 3: "Miner"}

# 5 most-used controls, shown in the HUD strip
_HUD_HINTS = "WASD: Move  ·  Space: Do  ·  E: Rest  ·  Tab: Sleep  ·  Q: Skip"


class AlemRenderer:
    def __init__(self, env, env_params, static_params, player_textures, pixel_render_size=1):
        self.env = env
        self.env_params = env_params
        self.static_params = static_params
        self.num_players = static_params.player_count
        self.pixel_render_size = pixel_render_size
        self.pygame_events = []

        dashboard_h = (self.num_players + 1) // 2
        game_w = OBS_DIM[1] * BLOCK_PIXEL_SIZE_HUMAN * pixel_render_size
        game_h = (
            (OBS_DIM[0] + INVENTORY_OBS_HEIGHT + dashboard_h)
            * BLOCK_PIXEL_SIZE_HUMAN
            * pixel_render_size
        )
        self.game_h = game_h
        # Two-row HUD: a status row (who / event / score) + a controls row.
        self._hud_h = 50 * pixel_render_size
        self._pad = 12 * pixel_render_size
        self.screen_size = (game_w, game_h + self._hud_h)

        pygame.init()
        pygame.key.set_repeat(250, 75)
        self.screen_surface = pygame.display.set_mode(self.screen_size)
        pygame.display.set_caption("Alem - Alem-Coop")

        self._font_status = pygame.font.SysFont("Arial", 15 * pixel_render_size, bold=True)
        self._font_event = pygame.font.SysFont("Arial", 15 * pixel_render_size, bold=True)
        self._font_hints = pygame.font.SysFont("Arial", 13 * pixel_render_size)

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
        self.screen_surface.blit(surface, (0, 0))
        self._draw_hud(env_state, player, score)

    def _draw_hud(self, env_state, player, score):
        sw = self.screen_size[0]
        y0 = self.game_h
        pad = self._pad
        color = _HUD_COLORS[player % len(_HUD_COLORS)]

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
