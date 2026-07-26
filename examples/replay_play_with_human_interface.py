"""
Replay a saved Alem human-interface play episode from its recorded state snapshots.

``examples/play_with_human_interface.py`` saves one ``episode_XX_states.pkl.gz`` per episode
(via ``alem.trajectory_io.save_state_bundle`` — the same format the LLM/RL eval
harness uses), a gzip-pickled bundle of every pre-step ``EnvState`` plus the
``StaticEnvParams`` used to render them. This script loads that bundle and
replays it visually — for reviewing a session, sanity-checking that save/reload
during play didn't corrupt anything, or producing a demo GIF.

Usage:
    # Interactively view episode 0 of a session (arrow keys / space to control):
    python examples/replay_play_with_human_interface.py outputs/play_with_human_interface/T01_easy_20260725-153000 --episode 0

    # Or point directly at the states file:
    python examples/replay_play_with_human_interface.py outputs/play_with_human_interface/T01_easy_.../episode_00_states.pkl.gz

    # Export a GIF instead (works headless, no display needed):
    python examples/replay_play_with_human_interface.py <session_dir> --episode 0 --headless --save-gif out.gif

Interactive controls:
    SPACE       play / pause
    LEFT/RIGHT  step back / forward one env step
    UP/DOWN     speed up / slow down playback
    P           cycle which player's viewpoint is rendered
    Q / ESC     quit
"""

import argparse
import json
import os
import sys
from pathlib import Path


def _resolve_states_path(path: Path, episode: str) -> Path:
    if path.is_file():
        return path
    tag = "practice" if episode == "practice" else f"{int(episode):02d}"
    candidate = path / f"episode_{tag}_states.pkl.gz"
    if not candidate.exists():
        raise FileNotFoundError(
            f"No states file at {candidate}. Pass --episode N to pick a different one, "
            f"or point directly at a *_states.pkl.gz file."
        )
    return candidate


def _env_params_from_session(session_dir: Path):
    """Reconstruct the EnvParams the episode was actually played with, if session.json
    is available alongside the states file. Falls back to protocol defaults otherwise
    (only affects the sidebar's "legal action" highlighting, not the replayed states).
    """
    from alem.alem_coop.alem_state import EnvParams, get_coordination_params

    session_path = session_dir / "session.json"
    if not session_path.exists():
        return EnvParams()
    meta = json.loads(session_path.read_text())
    protocol = meta.get("protocol", {})
    coord = protocol.get("coordination_difficulty", "easy")
    env_kwargs = dict(
        max_timesteps=protocol.get("max_steps", 10000),
        god_mode=protocol.get("god_mode", False),
        soft_specialization=True,
        shared_reward=False,
        specialist_efficiency=1.0,
        non_specialist_efficiency=0.2,
        randomize_alpha=False,
    )
    if coord != "none":
        env_kwargs.update(get_coordination_params(coord))
    return EnvParams(**env_kwargs)


def _print_episode_context(session_dir: Path, states_path: Path):
    ep_path = session_dir / "episodes.jsonl"
    if not ep_path.exists():
        return
    tag = states_path.name.removeprefix("episode_").removesuffix("_states.pkl.gz")
    for line in ep_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        rec_tag = "practice" if rec.get("practice") else f"{rec['episode']:02d}"
        if rec_tag == tag:
            pct = 100 * rec["metrics"].get("Team/achievement_pct", 0.0)
            print(
                f"Participant {rec['participant']} · {rec['difficulty']} · "
                f"status={rec['status']} · {rec['env_steps']} steps · team_ach {pct:.1f}%"
            )
            return


def _export_gif(states, renderer, pygame, player, save_gif, gif_fps):
    import imageio

    frames = []
    for i, s in enumerate(states):
        renderer.render(s, player, 0.0)
        renderer.update()
        arr = pygame.surfarray.array3d(renderer.screen_surface).transpose((1, 0, 2))
        frames.append(arr)
        if (i + 1) % 200 == 0:
            print(f"  Rendered {i + 1}/{len(states)} frames...")
    out = Path(save_gif)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, frames, fps=gif_fps)
    print(f"Saved {len(frames)} frame(s) to {out}")


def _interactive_viewer(states, renderer, pygame, clock, num_players, args):
    print(f"Loaded {len(states)} step(s).")
    print("SPACE play/pause · LEFT/RIGHT step · UP/DOWN speed · P switch player · Q/ESC quit")

    idx = 0
    playing = not args.paused
    speed = max(1, args.fps)
    player = max(0, min(args.player, num_players - 1))

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif e.key == pygame.K_SPACE:
                    playing = not playing
                elif e.key == pygame.K_RIGHT:
                    playing = False
                    idx = min(len(states) - 1, idx + 1)
                elif e.key == pygame.K_LEFT:
                    playing = False
                    idx = max(0, idx - 1)
                elif e.key == pygame.K_UP:
                    speed = min(60, speed + 5)
                elif e.key == pygame.K_DOWN:
                    speed = max(1, speed - 5)
                elif e.key == pygame.K_p:
                    player = (player + 1) % num_players

        if playing:
            if idx < len(states) - 1:
                idx += 1
            else:
                playing = False  # stop at the end

        renderer.render(states[idx], player, 0.0)
        renderer.update()
        clock.tick(speed)


def main(args):
    if args.headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    import pygame

    from play_alem import AlemRenderer

    from alem.alem_coop.constants import (
        BLOCK_PIXEL_SIZE_HUMAN,
        TEXTURES,
        load_player_specific_textures,
    )
    from alem.alem_coop.envs.alem_pixels_env import AlemCoopPixelsEnv
    from alem.trajectory_io import load_state_bundle

    path = Path(args.path)
    states_path = _resolve_states_path(path, args.episode)
    session_dir = states_path.parent

    states, static_params = load_state_bundle(states_path)
    if not states:
        print(f"{states_path} has no recorded steps.")
        return

    _print_episode_context(session_dir, states_path)

    n = static_params.player_count
    env_params = _env_params_from_session(session_dir)
    env = AlemCoopPixelsEnv(num_agents=n, env_params=env_params, static_env_params=static_params)

    block_px = BLOCK_PIXEL_SIZE_HUMAN
    player_textures = load_player_specific_textures(TEXTURES[block_px], n)
    renderer = AlemRenderer(
        env, env_params, static_params, player_textures, pixel_render_size=args.size // block_px
    )
    clock = pygame.time.Clock()

    try:
        if args.save_gif:
            _export_gif(states, renderer, pygame, args.player, args.save_gif, args.gif_fps)
        else:
            _interactive_viewer(states, renderer, pygame, clock, n, args)
    finally:
        pygame.quit()


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Replay a saved Alem human-interface play episode")
    parser.add_argument(
        "path",
        type=str,
        help="A session directory (use --episode to pick) or a direct path to an "
        "episode_XX_states.pkl.gz file",
    )
    parser.add_argument(
        "--episode",
        type=str,
        default="0",
        help="Episode number (e.g. 0) or 'practice' — used when path is a session "
        "directory (default: 0)",
    )
    parser.add_argument("--player", type=int, default=0, help="Viewpoint to render (default: 0)")
    parser.add_argument("--size", type=int, default=128, help="Display block size px (default: 128)")
    parser.add_argument("--fps", type=int, default=20, help="Initial playback speed (default: 20)")
    parser.add_argument("--paused", action="store_true", help="Start paused instead of auto-playing")
    parser.add_argument(
        "--headless", action="store_true", help="Dummy SDL video driver (needed for --save-gif with no display)"
    )
    parser.add_argument(
        "--save-gif",
        type=str,
        default=None,
        help="Render every step to a GIF at this path instead of an interactive viewer",
    )
    parser.add_argument("--gif-fps", type=int, default=10, help="Playback speed for --save-gif (default: 10)")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
