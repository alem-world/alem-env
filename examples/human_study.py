"""
Human-study runner for Alem — the pygame interface from play_alem.py plus metrics logging.

Runs the standard evaluation protocol (EVALUATION.md) with humans at the keyboard and
writes everything needed to report a human baseline:

- Worlds match the agent evaluation: episode ``i`` uses world seed ``--seed-base + i``
  (default 9999 = ``EVAL_SEED``) with the same key-split pattern as the LLM/RL harness,
  and the same ``EnvParams`` (``max_timesteps=10000``, ``soft_specialization=True``,
  ``shared_reward=False``). Humans therefore play the exact worlds agents were scored on.
- ``steps.jsonl``   one record per env step: per-player action, decision latency (ms),
  rewards, health/food/drink/energy, positions, achievement unlocks. With the world seed
  and this action log an episode is fully replayable.
- ``episodes.jsonl`` one record per episode: status, duration, and the canonical
  ``compute_score()`` metrics (``Team/achievement_pct`` etc. — the leaderboard metrics).
- ``session.json``   participant id, full config, git commit, library versions.

Files land in ``outputs/human_study/<session_id>/`` and every line is flushed as it is
written, so a crash or Ctrl+C loses nothing.

Controls are identical to play_alem.py (WASD move, Space do, E rest, ... see its
docstring for the full list), plus:

    ENTER   start the next episode (between episodes)
    ESC     abort the current episode (still logged, marked "aborted")
    Close window / Ctrl+C   end the session (partial data kept)

Hotseat flow is unchanged: players act in turn on one keyboard and the world advances
once all players have acted. For a coordination study seat one human per player
(talking allowed — the human analogue of the communication channel); use anonymous
participant/team ids.

Usage:
    # A 3-human team (or one human controlling all three), 2 episodes on easy:
    python examples/human_study.py --participant T01 --coord easy --episodes 2

    # Pilot yourself with a shorter episode cap:
    python examples/human_study.py --participant pilot --coord medium --max-steps 2000

    # Verify the whole pipeline with a random bot, no display needed:
    python examples/human_study.py --bot --headless --episodes 2 --max-steps 50

    # Summarise every session collected so far (mean ± std per difficulty):
    python examples/human_study.py --aggregate
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HEADLINE_METRICS = [
    "Team/achievement_pct",
    "Team/coordination_achievement_pct",
    "Team/normal_achievement_pct",
    "Team/reward_pct_of_max",
]

# Keyboard reference (mirrors examples/play_alem.py KEY_MAPPING) — the human
# analogue of the LLM action list, appended to the rules so the screen is
# self-contained.
CONTROLS_TEXT = """
## Controls (keyboard)
Players take turns on one keyboard: the highlighted player at the top of the screen acts, then the next. The world advances once every player has chosen. You control the player centred on screen.
- Move: W A S D  (north / west / south / east). Moving into a wall just turns you to face it.
- Do / interact: Space  — acts on the tile you face (chop, mine, attack, drink, open chest, revive)
- Sleep: Tab      Rest: E      Skip turn / no-op: Q
- Descend / Ascend: .  /  ,
- Place: R Stone · T Table · F Furnace · P Plant · J Torch
- Craft pickaxe: 1 Wood · 2 Stone · 3 Iron · 4 Diamond
- Craft sword: 5 Wood · 6 Stone · 7 Iron · 8 Diamond
- Craft armour: Y Iron · U Diamond      Craft: O Arrow · [ Torch
- Combat: I Shoot Arrow · G Cast Spell
- Request a resource from the team: F1–F9 (Food … Sapphire)
- Give held resource to a teammate: Backspace
- Build: 9 Shelter · 0 Forge · ` Beacon

## During the study
- ENTER starts each episode · ESC aborts the current episode (still recorded) · close the window to end the session.
- New achievements pop up on screen as you unlock them — unlocking as many as possible (while staying alive) is the goal.
"""


def build_rules_text(num_agents, coordination_enabled):
    """Game rules a human sees before playing: the agents' system-prompt rules + controls.

    Pulls the same ``<game_rules>`` block the LLM agents receive (via
    get_instruction_prompt) so humans and agents are briefed identically, then
    appends the keyboard controls.
    """
    from alem.llm.alem_language_wrapper import get_instruction_prompt

    prompt = get_instruction_prompt(
        coordination_enabled=coordination_enabled,
        num_agents=num_agents,
        include_all_actions=False,
        prompt_mode="specific_collaborative",
    )
    intro = prompt.split("<game_rules>")[0].strip()
    rules = prompt.split("<game_rules>")[1].split("</game_rules>")[0].strip()
    return f"{intro}\n\n{rules}\n{CONTROLS_TEXT}"


def _py(x):
    """Convert JAX/NumPy values (nested in dicts/lists) to plain Python for json.dumps."""
    if isinstance(x, dict):
        return {k: _py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_py(v) for v in x]
    a = np.asarray(x)
    if a.ndim == 0:
        v = a.item()
        return round(v, 6) if isinstance(v, float) else v
    return [_py(v) for v in a.tolist()]


def _normalize_metrics(metrics):
    """Collapse compute_score's scalar->(num_agents,) broadcast back to scalars.

    compute_score broadcasts every scalar metric to (num_agents,) for batching
    (see common.py); only ``Achievements/<name>`` entries are genuinely per-player.
    """
    return {
        k: v[0] if isinstance(v, list) and not k.startswith("Achievements/") else v
        for k, v in metrics.items()
    }


def _now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git_commit():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


class StudyLogger:
    """Append-only JSONL logging for one study session (crash-safe: flushed per line)."""

    def __init__(self, out_dir: Path, meta: dict):
        self.dir = out_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta = dict(meta)
        self._write_meta()
        self._steps = open(self.dir / "steps.jsonl", "a", buffering=1)
        self._episodes = open(self.dir / "episodes.jsonl", "a", buffering=1)

    def _write_meta(self):
        (self.dir / "session.json").write_text(json.dumps(self.meta, indent=2) + "\n")

    def log_step(self, record: dict):
        self._steps.write(json.dumps(record) + "\n")

    def log_episode(self, record: dict):
        self._episodes.write(json.dumps(record) + "\n")

    def close(self, **updates):
        self.meta.update(updates)
        self._write_meta()
        self._steps.close()
        self._episodes.close()


# ── Study session (heavy imports kept inside so --aggregate stays instant) ──────────


def run_session(args):
    import jax
    import jax.numpy as jnp
    import pygame

    if args.headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    from play_alem import AlemRenderer

    from alem.alem_coop.alem_state import EnvParams, get_coordination_params
    from alem.alem_coop.constants import (
        BLOCK_PIXEL_SIZE_HUMAN,
        TEXTURES,
        Achievement,
        Action,
        load_player_specific_textures,
    )
    from alem.alem_coop.envs.alem_pixels_env import AlemCoopPixelsEnv
    from alem.alem_coop.envs.common import compute_score

    # EnvParams built exactly like the eval harness (alem/llm/alem_language_wrapper.py
    # make_alem_env) so human episodes are comparable to leaderboard runs. Note this
    # differs from play_alem.py, which uses EnvParams() defaults (shared_reward=True,
    # soft_specialization=False).
    env_kwargs = dict(
        max_timesteps=args.max_steps,
        god_mode=args.god,
        soft_specialization=True,
        shared_reward=False,
        specialist_efficiency=1.0,
        non_specialist_efficiency=0.2,
        randomize_alpha=False,
    )
    if args.coord != "none":
        env_kwargs.update(get_coordination_params(args.coord))
    env_params = EnvParams(**env_kwargs)

    env = AlemCoopPixelsEnv(num_agents=args.players, env_params=env_params)
    static_params = env.static_env_params
    n = static_params.player_count

    participant = args.participant or ("bot" if args.bot else None)
    session_id = "{}_{}_{}".format(
        re.sub(r"[^\w.-]", "_", participant),
        args.coord,
        datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    out_dir = Path(args.out) / session_id
    meta = {
        "session_id": session_id,
        "participant": participant,
        "created_at": _now_iso(),
        "argv": sys.argv[1:],
        "git_commit": _git_commit(),
        "versions": {"jax": jax.__version__, "pygame": pygame.version.ver},
        "protocol": {
            "env": "Alem-Coop-Pixels",
            "players": n,
            "coordination_difficulty": args.coord,
            "seed_base": args.seed_base,
            "episodes_planned": args.episodes,
            "max_steps": args.max_steps,
            "god_mode": args.god,
            "bot": args.bot,
            "eval_matched_env_params": True,  # see EVALUATION.md
        },
    }
    logger = StudyLogger(out_dir, meta)
    print(f"Logging to {out_dir}")

    block_px = BLOCK_PIXEL_SIZE_HUMAN
    player_textures = load_player_specific_textures(TEXTURES[block_px], n)
    renderer = AlemRenderer(
        env, env_params, static_params, player_textures, pixel_render_size=args.size // block_px
    )
    clock = pygame.time.Clock()

    step_fn = jax.jit(env.step_env)
    score_fn = jax.jit(lambda s, d: compute_score(s, d, static_params))

    print("Warming up JAX kernels (first launch only, ~30-60 s)...", end="", flush=True)
    _, warm_state = env.reset(jax.random.PRNGKey(0))
    warm_actions = {f"agent_{i}": jnp.int32(0) for i in range(n)}
    _, warm_state, _, _, _ = step_fn(jax.random.PRNGKey(0), warm_state, warm_actions)
    jax.block_until_ready(score_fn(warm_state, jnp.bool_(True)))
    renderer.render(warm_state, 0, 0.0)  # compile the render kernel too
    print(" done.")

    num_achievements = int(warm_state.achievements.shape[1])

    # Briefing: the same game rules the agents receive, shown before play and
    # saved with the session for the record.
    rules_text = build_rules_text(n, coordination_enabled=(args.coord != "none"))
    (out_dir / "rules.txt").write_text(rules_text + "\n")
    print("\n" + rules_text + "\n")
    if not args.bot and not args.skip_rules:
        if not renderer.show_text_screen(
            "Alem — How to Play", rules_text, "↑ / ↓ scroll   ·   ENTER to begin"
        ):
            logger.close(ended_at=_now_iso(), episodes_completed=0, episode_statuses=[])
            pygame.quit()
            print("Closed before starting. Rules saved to rules.txt.")
            return

    def wait_for_enter(state, message):
        """Idle-render until ENTER (True) or window close (False)."""
        renderer.push_notification(message, color=(255, 255, 255), duration_ms=10**9)
        while True:
            if renderer.is_quit_requested():
                return False
            for e in renderer.pygame_events:
                if e.type == pygame.KEYDOWN and e.key == pygame.K_RETURN:
                    renderer._notification = None
                    return True
            renderer.render(state, 0, 0.0)
            renderer.update()
            clock.tick(args.fps)

    def run_episode(ep_idx):
        world_seed = args.seed_base + ep_idx
        rng = jax.random.PRNGKey(world_seed)
        rng, _rng = jax.random.split(rng)  # same split pattern as the eval harness
        _, state = env.reset(_rng)
        bot_rng = np.random.default_rng(world_seed)

        print(f"Episode {ep_idx + 1}/{args.episodes} (world seed {world_seed}) — playing...")
        if not args.bot:
            if not wait_for_enter(state, f"Episode {ep_idx + 1}/{args.episodes} — press ENTER"):
                return "quit", None
        # Drop keys pressed before the episode started (warmup / ENTER screen) so
        # they can't become the first action or an instant ESC abort.
        pygame.event.clear()
        renderer.pygame_events = []

        scores = [0.0] * n
        pending = []  # per-player action records for the current env step
        current = 0
        status = "completed"
        env_steps = 0
        ep_t0 = time.monotonic()
        decision_t0 = ep_t0

        try:
            while True:
                if renderer.is_quit_requested():
                    status = "quit"
                    break
                if any(
                    e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE
                    for e in renderer.pygame_events
                ):
                    status = "aborted"
                    break

                auto = bool(state.is_sleeping[current]) or not bool(state.player_alive[current])
                if args.bot:
                    action = (
                        Action.NOOP.value if auto else int(bot_rng.integers(0, len(Action)))
                    )
                else:
                    action = renderer.get_action_from_keypress(state, current)

                if action is not None:
                    now = time.monotonic()
                    pending.append(
                        {
                            "player": current,
                            "id": int(action),
                            "name": Action(action).name,
                            "decision_ms": int((now - decision_t0) * 1000),
                            "auto": auto,
                        }
                    )
                    decision_t0 = now
                    current += 1

                if current == n:  # all players acted — step the world
                    step_idx = int(state.timestep)
                    old_ach = np.asarray(state.achievements)
                    old_hp = np.asarray(state.player_health)

                    rng, _rng = jax.random.split(rng)
                    acts = {f"agent_{i}": jnp.int32(pending[i]["id"]) for i in range(n)}
                    _, state, rewards, dones, _ = step_fn(_rng, state, acts)
                    env_steps = int(state.timestep)

                    new_ach = np.asarray(state.achievements)
                    unlocks = [
                        {"player": p, "name": Achievement(i).name}
                        for p in range(n)
                        for i in range(num_achievements)
                        if old_ach[p, i] == 0 and new_ach[p, i] == 1
                    ]
                    rew = [float(rewards[f"agent_{i}"]) for i in range(n)]
                    for i in range(n):
                        scores[i] += rew[i]
                    done = bool(dones["__all__"])

                    logger.log_step(
                        {
                            "episode": ep_idx,
                            "world_seed": world_seed,
                            "step": step_idx,
                            "t": _now_iso(),
                            "level": int(state.player_level),
                            "actions": pending,
                            "rewards": _py(rew),
                            "scores": _py(scores),
                            "health": _py(state.player_health),
                            "food": _py(state.player_food),
                            "drink": _py(state.player_drink),
                            "energy": _py(state.player_energy),
                            "alive": _py(state.player_alive),
                            "pos": _py(state.player_position),
                            "new_achievements": unlocks,
                            "done": done,
                        }
                    )

                    # HUD notifications (same behaviour as play_alem.py)
                    for p in range(n):
                        dhp = float(old_hp[p]) - float(np.asarray(state.player_health)[p])
                        if dhp >= 1:
                            renderer.push_notification(
                                f"P{p + 1}: -{dhp:.0f} HP", color=(255, 80, 80), duration_ms=2000
                            )
                    for p in range(n):
                        if rew[p] > 0.01:
                            renderer.push_notification(
                                f"P{p + 1}: +{rew[p]:.2f}", color=(100, 230, 100), duration_ms=2000
                            )
                        elif rew[p] < -0.01:
                            renderer.push_notification(
                                f"P{p + 1}: {rew[p]:.2f}", color=(255, 160, 60), duration_ms=2000
                            )
                    for u in unlocks:
                        pretty = u["name"].replace("_", " ").title()
                        print(f"  Player {u['player'] + 1} achieved {pretty}")
                        renderer.push_notification(
                            f"P{u['player'] + 1}: {pretty}", color=(255, 215, 0), duration_ms=3500
                        )

                    pending = []
                    current = 0
                    decision_t0 = time.monotonic()
                    if done:
                        break

                renderer.render(state, current, scores[current])
                renderer.update()
                clock.tick(args.fps)
        except KeyboardInterrupt:
            status = "interrupted"

        # Canonical metrics, recomputed with done=True so aborted episodes score too
        # (same trick as alem/llm/alem_env.py).
        metrics = _normalize_metrics(_py(score_fn(state, jnp.bool_(True))))
        team_ach = np.asarray(state.achievements).any(axis=0)
        record = {
            "session_id": session_id,
            "participant": participant,
            "difficulty": args.coord,
            "players": n,
            "episode": ep_idx,
            "world_seed": world_seed,
            "status": status,
            "env_steps": env_steps,
            "duration_s": round(time.monotonic() - ep_t0, 1),
            "scores": _py(scores),
            "team_achievement_names": [
                Achievement(i).name for i in range(num_achievements) if team_ach[i]
            ],
            "metrics": metrics,
        }
        logger.log_episode(record)

        print(
            f"Episode {ep_idx + 1} {status}: {env_steps} env steps, "
            f"{record['duration_s'] / 60:.1f} min"
        )
        _print_metrics_line(metrics)
        return status, record

    episodes = []
    try:
        for ep_idx in range(args.episodes):
            status, record = run_episode(ep_idx)
            if record is not None:
                episodes.append(record)
            if status in ("quit", "interrupted"):
                break
    finally:
        logger.close(
            ended_at=_now_iso(),
            episodes_completed=len(episodes),
            episode_statuses=[r["status"] for r in episodes],
        )
        pygame.quit()

    if episodes:
        print(f"\nSession {session_id}: {len(episodes)} episode(s)")
        _print_summary(episodes)
    print(f"\nData written to {out_dir}")
    print("Aggregate all sessions with: python examples/human_study.py --aggregate")


def _print_metrics_line(metrics):
    parts = []
    for key in HEADLINE_METRICS:
        if key in metrics:
            parts.append(f"{key.split('/')[1]} {100 * metrics[key]:.1f}%")
    print("  " + "  |  ".join(parts))


def _print_summary(records):
    for key in HEADLINE_METRICS:
        vals = [
            r["metrics"][key]
            for r in records
            if isinstance(r["metrics"].get(key), (int, float))
        ]
        if not vals:
            continue
        mean = 100 * statistics.mean(vals)
        std = 100 * statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"  {key:<40s} {mean:6.1f}% ± {std:.1f}")
    steps = [r["env_steps"] for r in records]
    mins = [r["duration_s"] / 60 for r in records]
    print(
        f"  {'episode length':<40s} {statistics.mean(steps):6.0f} steps, "
        f"{statistics.mean(mins):.1f} min"
    )


def aggregate(root: Path):
    """Summarise every session under ``root`` grouped by coordination difficulty."""
    rows = []
    for f in sorted(root.glob("*/episodes.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        print(f"No episodes found under {root}")
        return

    sessions = {r["session_id"] for r in rows}
    print(f"{len(rows)} episode(s) from {len(sessions)} session(s) under {root}\n")

    for diff in ["easy", "medium", "hard", "none"]:
        recs = [r for r in rows if r["difficulty"] == diff]
        if not recs:
            continue
        participants = {r["participant"] for r in recs}
        print(f"── {diff} — {len(recs)} episode(s), {len(participants)} participant(s)/team(s)")
        _print_summary(recs)
        for r in recs:
            pct = 100 * r["metrics"].get("Team/achievement_pct", 0.0)
            print(
                f"    {r['participant']:<12s} ep{r['episode']} seed {r['world_seed']} "
                f"[{r['status']:<11s}] {r['env_steps']:>5d} steps  team_ach {pct:5.1f}%"
            )
        print()


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Run or aggregate an Alem human study")
    parser.add_argument("--participant", type=str, default=None, help="Anonymous participant/team id")
    parser.add_argument(
        "--coord",
        type=str,
        default="easy",
        choices=["none", "easy", "medium", "hard"],
        help="Coordination difficulty (default: easy)",
    )
    parser.add_argument("--episodes", type=int, default=1, help="Episodes this session (default: 1)")
    parser.add_argument("--players", type=int, default=3, help="Players (default: 3, the protocol)")
    parser.add_argument(
        "--seed-base",
        type=int,
        default=9999,
        help="Episode i uses world seed seed-base+i (default: 9999 = EVAL_SEED, the agent-eval worlds)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10000,
        help="Env step cap per episode (default: 10000, the protocol; lower it for pilots)",
    )
    parser.add_argument("--god", action="store_true", help="God mode (breaks protocol; pilots only)")
    parser.add_argument("--size", type=int, default=128, help="Display block size px (default: 128)")
    parser.add_argument("--fps", type=int, default=60, help="Target FPS (default: 60)")
    parser.add_argument(
        "--out", type=str, default="outputs/human_study", help="Results root directory"
    )
    parser.add_argument("--bot", action="store_true", help="Random-action smoke test (no human)")
    parser.add_argument("--headless", action="store_true", help="Dummy SDL video driver (for --bot)")
    parser.add_argument("--skip-rules", action="store_true", help="Skip the on-screen rules page")
    parser.add_argument("--aggregate", action="store_true", help="Summarise sessions under --out")
    args = parser.parse_args(argv)

    if not args.aggregate and not args.bot and not args.participant:
        parser.error("--participant is required (or use --bot for a smoke test)")
    return args


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    if args.aggregate:
        aggregate(Path(args.out))
    else:
        run_session(args)
