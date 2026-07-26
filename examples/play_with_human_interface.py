"""
Human-interface play runner for Alem — the pygame interface from play_alem.py plus metrics logging.

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
- ``session.json``   participant id, consent/demographics, full config, git commit,
  library versions.
- ``episode_XX_trajectory.npz`` / ``episode_XX_states.pkl.gz`` — per-episode symbolic
  observations, actions, rewards and full env-state snapshots, saved with
  ``alem.trajectory_io`` — the exact same save format the LLM/RL eval harness uses
  (see ``baselines/llm/eval_utils/evaluator.py``). Replay with ``examples/replay_play_with_human_interface.py``.

Files land in ``outputs/play_with_human_interface/<session_id>/`` and every line is flushed as it is
written, so a crash or Ctrl+C loses nothing. See PLAY_WITH_HUMAN_INTERFACE.md for the full protocol
(consent, practice runs, breaks, backing up and analysing data).

Controls are identical to play_alem.py (WASD move, Space do, E rest, ... see its
docstring for the full list), plus:

    ENTER   start the next episode (between episodes)
    ESC     pause menu — resume, save a checkpoint, reload the last checkpoint,
            save & quit, abort the episode, or quit without saving
    Close window / Ctrl+C   end the session (partial data kept)

Hotseat flow is unchanged: players act in turn on one keyboard and the world advances
once all players have acted. For a coordination session seat one human per player
(talking allowed — the human analogue of the communication channel); use anonymous
participant/team ids.

Usage:
    # A 3-human team (or one human controlling all three), 2 episodes on easy:
    python examples/play_with_human_interface.py --participant T01 --coord easy --episodes 2

    # A short practice round first, so the team learns the controls before it counts:
    python examples/play_with_human_interface.py --participant T01 --coord easy --episodes 2 --practice

    # Pilot yourself with a shorter episode cap:
    python examples/play_with_human_interface.py --participant pilot --coord medium --max-steps 2000

    # Resume a session after a break, crash, or a checkpoint saved from the pause menu:
    python examples/play_with_human_interface.py --resume outputs/play_with_human_interface/T01_easy_20260725-153000

    # Verify the whole pipeline with a random bot, no display needed:
    python examples/play_with_human_interface.py --bot --headless --episodes 2 --max-steps 50

    # Summarise every session collected so far (mean ± std per difficulty):
    python examples/play_with_human_interface.py --aggregate

    # Export every session's episodes to a flat CSV for stats software:
    python examples/play_with_human_interface.py --export-csv outputs/play_with_human_interface/episodes.csv
"""

import argparse
import csv
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

from alem.trajectory_io import safe_json_dumps, save_state_bundle, save_trajectory_npz

HEADLINE_METRICS = [
    "Team/achievement_pct",
    "Team/coordination_achievement_pct",
    "Team/normal_achievement_pct",
    "Team/reward_pct_of_max",
]

# Sentinel episode index for the (unscored) practice round.
PRACTICE_EP_IDX = -1
# Fixed world seed for the practice round, distinct from the default eval-seed
# range (seed_base=9999+), so it never collides with a real episode's world.
PRACTICE_WORLD_SEED = 1

# ESC pause-menu options: (label, description). Order matches on-screen order.
PAUSE_MENU_OPTIONS = [
    ("Resume", "Continue playing"),
    ("Save Checkpoint", "Save progress right now, without stopping"),
    ("Reload Last Checkpoint", "Undo back to your last save and continue from there"),
    ("Save & Quit", "Save progress and end the session here"),
    ("Abort Episode", "End this episode now, unsaved (marked aborted)"),
    ("Quit Without Saving", "Close the session immediately"),
]

# Placeholder consent script — NOT an approved IRB/ethics form. Replace with
# your institution's actual approved text before running a real session; see
# PLAY_WITH_HUMAN_INTERFACE.md.
_CONSENT_SCRIPT = """
You are about to take part in a short session of human play in Alem, a multi-agent
survival/coordination game. If several people are hotseating on this same keyboard,
this consent covers everyone taking part. Your keystrokes, in-game actions, and
performance metrics will be recorded to disk for research analysis. Use an anonymous
participant/team id — do not enter any real names.

(This is a placeholder consent script, not an approved IRB/ethics form. Replace it
with your institution's actual approved text — see PLAY_WITH_HUMAN_INTERFACE.md.)
"""

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

## During the session
- ENTER starts each episode · ESC opens the pause menu (resume / save / reload / quit) · close the window to end the session.
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


def _episode_tag(ep_idx):
    return "practice" if ep_idx == PRACTICE_EP_IDX else f"{ep_idx:02d}"


def _load_step_records(steps_path: Path, episode_idx) -> dict:
    """Read one episode's step records from steps.jsonl, keyed by step index.

    The log is append-only, so a resumed or reloaded episode can leave a stale
    entry behind for a step index that was later redone; later lines win,
    which reconstructs the one authoritative record per step.
    """
    by_step = {}
    if not steps_path.exists():
        return by_step
    for line in steps_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("episode") == episode_idx:
            by_step[rec["step"]] = rec
    return by_step


def _next_episode_index(out_dir: Path, total_planned: int) -> int:
    """Index of the first non-practice episode that hasn't reached a final state.

    "completed" and "aborted" are final; "quit"/"interrupted" are not — those
    episodes get replayed from their checkpoint (or from scratch) on --resume.
    """
    finished = set()
    path = out_dir / "episodes.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("practice"):
                continue
            if rec.get("status") in ("completed", "aborted"):
                finished.add(rec["episode"])
    for i in range(total_planned):
        if i not in finished:
            return i
    return total_planned


def _checkpoint_path(out_dir: Path, ep_idx) -> Path:
    return out_dir / f"episode_{_episode_tag(ep_idx)}_checkpoint.json"


def save_checkpoint(out_dir, ep_idx, world_seed, completed_env_steps, current_player, pending):
    """Save a small resumable bookmark: which step this episode is at, plus any
    partial round in progress (some but not all players have acted this round).

    Deliberately NOT a full state snapshot — Alem is a pure function of
    (world_seed, action sequence), so ``_replay_episode_prefix`` can always
    reconstruct the exact state by replaying the logged actions in steps.jsonl.
    That keeps checkpoints tiny enough to save after every single step.
    """
    payload = {
        "episode": ep_idx,
        "world_seed": world_seed,
        "completed_env_steps": completed_env_steps,
        "current_player": current_player,
        "pending": pending,
        "saved_at": _now_iso(),
    }
    _checkpoint_path(out_dir, ep_idx).write_text(safe_json_dumps(payload) + "\n")


def load_checkpoint(out_dir, ep_idx):
    path = _checkpoint_path(out_dir, ep_idx)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _delete_checkpoint(out_dir, ep_idx):
    _checkpoint_path(out_dir, ep_idx).unlink(missing_ok=True)


def collect_participant_info(args):
    """Console-based consent + optional per-player demographics, saved into session.json.

    Returns a dict to attach to session meta, or None if consent was declined —
    the caller must then abort without creating any output files. Skipped
    entirely for --bot smoke tests and --skip-consent runs. Players typically
    hotseat on one keyboard/PC, so one consent covers the whole group but
    demographics are asked per player.
    """
    if args.bot or args.skip_consent:
        return {"consent_skipped": True}

    print(_CONSENT_SCRIPT)
    reply = input(
        f"Does everyone taking part ({args.players} player(s), same keyboard) consent "
        "to join and have this session logged? [y/N]: "
    ).strip().lower()
    if reply not in ("y", "yes"):
        return None

    print("\nA few optional questions per player (press Enter to skip any of them).")
    players = []
    for p in range(args.players):
        print(f"-- Player {p + 1} of {args.players} --")
        age_range = input("  Age range (e.g. 18-24, 25-34, ...): ").strip()
        gaming_experience = input("  Gaming experience, 1 (none) - 5 (expert): ").strip()
        familiarity = input("  Familiar with Minecraft/Crafter-like survival games? [y/n]: ").strip().lower()
        players.append(
            {
                "age_range": age_range or None,
                "gaming_experience": gaming_experience or None,
                "familiar_with_survival_games": familiarity or None,
            }
        )

    return {
        "consent_skipped": False,
        "consented_at": _now_iso(),
        "num_players": args.players,
        "players": players,
    }


class SessionLogger:
    """Append-only JSONL logging for one play session (crash-safe: flushed per line)."""

    def __init__(self, out_dir: Path, meta: dict):
        self.dir = out_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta = dict(meta)
        self._write_meta()
        self._steps = open(self.dir / "steps.jsonl", "a", buffering=1)
        self._episodes = open(self.dir / "episodes.jsonl", "a", buffering=1)

    def _write_meta(self):
        (self.dir / "session.json").write_text(safe_json_dumps(self.meta) + "\n")

    def log_step(self, record: dict):
        self._steps.write(safe_json_dumps(record) + "\n")

    def log_episode(self, record: dict):
        self._episodes.write(safe_json_dumps(record) + "\n")

    def close(self, **updates):
        self.meta.update(updates)
        self._write_meta()
        self._steps.close()
        self._episodes.close()


# ── Play session (heavy imports kept inside so --aggregate stays instant) ──────────


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
    from alem.alem_coop.renderer.renderer_symbolic import render_alem_symbolic

    # --resume loads the original session's protocol from disk and overrides
    # the CLI args with it, so the world/env are reconstructed identically —
    # only display flags (--size/--fps) are taken from the current invocation.
    resumed_meta = None
    if args.resume:
        resume_dir = Path(args.resume)
        session_path = resume_dir / "session.json"
        if not session_path.exists():
            print(f"No session.json found under {resume_dir} — nothing to resume.")
            return
        resumed_meta = json.loads(session_path.read_text())
        protocol = resumed_meta["protocol"]
        args.participant = resumed_meta["participant"]
        args.coord = protocol["coordination_difficulty"]
        args.players = protocol["players"]
        args.seed_base = protocol["seed_base"]
        args.episodes = protocol["episodes_planned"]
        args.max_steps = protocol["max_steps"]
        args.god = protocol["god_mode"]
        args.bot = protocol["bot"]
        print(f"Resuming session {resumed_meta['session_id']} from {resume_dir}")

    participant = args.participant or ("bot" if args.bot else None)

    # Consent/demographics happen before any heavy JAX warmup or file writes —
    # a decline should cost the participant nothing and leave no trace.
    # Skipped when resuming: consent was already collected for this session.
    participant_info = None
    if resumed_meta is None:
        participant_info = collect_participant_info(args)
        if participant_info is None:
            print("Consent declined — exiting without recording any data.")
            return

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

    if resumed_meta is not None:
        session_id = resumed_meta["session_id"]
        out_dir = Path(args.resume)
        meta = resumed_meta
    else:
        session_id = "{}_{}_{}".format(
            re.sub(r"[^\w.-]", "_", participant),
            args.coord,
            datetime.now().strftime("%Y%m%d-%H%M%S"),
        )
        out_dir = Path(args.out) / session_id
        meta = {
            "session_id": session_id,
            "participant": participant,
            "participant_info": participant_info,
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
    logger = SessionLogger(out_dir, meta)
    print(f"Logging to {out_dir}")

    block_px = BLOCK_PIXEL_SIZE_HUMAN
    player_textures = load_player_specific_textures(TEXTURES[block_px], n)
    renderer = AlemRenderer(
        env, env_params, static_params, player_textures, pixel_render_size=args.size // block_px
    )
    clock = pygame.time.Clock()

    step_fn = jax.jit(env.step_env)
    score_fn = jax.jit(lambda s, d: compute_score(s, d, static_params))
    render_symbolic_fn = jax.jit(lambda s: render_alem_symbolic(s, static_params))

    print("Warming up JAX kernels (first launch only, ~30-60 s)...", end="", flush=True)
    _, warm_state = env.reset(jax.random.PRNGKey(0))
    warm_actions = {f"agent_{i}": jnp.int32(0) for i in range(n)}
    _, warm_state, _, _, _ = step_fn(jax.random.PRNGKey(0), warm_state, warm_actions)
    jax.block_until_ready(score_fn(warm_state, jnp.bool_(True)))
    jax.block_until_ready(render_symbolic_fn(warm_state))
    renderer.render(warm_state, 0, 0.0)  # compile the render kernel too
    print(" done.")

    num_achievements = int(warm_state.achievements.shape[1])

    # Briefing: the same game rules the agents receive, shown before play and
    # saved with the session for the record. Skipped on --resume — already seen.
    if resumed_meta is None:
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

    def _replay_episode_prefix(world_seed, records_by_step, up_to_step):
        """Deterministically replay steps [0, up_to_step) from logged actions.

        Alem is a pure function of (world seed, action sequence), so replaying
        the exact actions already recorded in steps.jsonl reproduces the exact
        same states. This is what lets --resume and "Reload Last Checkpoint"
        rebuild a complete, gap-free trajectory even when the live in-memory
        arrays were lost (process restart) or are being rolled back (reload).
        """
        rng = jax.random.PRNGKey(world_seed)
        rng, _rng = jax.random.split(rng)
        _, state = env.reset(_rng)
        scores = [0.0] * n
        t_obs, t_actions, t_rewards, t_dones, t_states, t_text_actions = [], [], [], [], [], []
        for step_idx in range(up_to_step):
            rec = records_by_step.get(step_idx)
            if rec is None:
                print(f"  Warning: missing logged step {step_idx} — replay stopped early.")
                break
            t_obs.append(np.asarray(render_symbolic_fn(state)))
            t_states.append(jax.device_get(state))
            t_text_actions.append([a["name"] for a in rec["actions"]])
            act_ids = [a["id"] for a in rec["actions"]]
            t_actions.append(np.array(act_ids, dtype=np.int32))

            rng, _rng = jax.random.split(rng)
            acts = {f"agent_{i}": jnp.int32(act_ids[i]) for i in range(n)}
            _, state, rewards, dones, _ = step_fn(_rng, state, acts)

            rew = [float(rewards[f"agent_{i}"]) for i in range(n)]
            for i in range(n):
                scores[i] += rew[i]
            t_rewards.append(np.array(rew, dtype=np.float32))
            t_dones.append(bool(dones["__all__"]))
            if (step_idx + 1) % 500 == 0:
                print(f"  Replayed {step_idx + 1}/{up_to_step} steps...")
        return state, rng, scores, t_obs, t_actions, t_rewards, t_dones, t_states, t_text_actions

    def run_episode(ep_idx, practice=False, resume=None):
        world_seed = PRACTICE_WORLD_SEED if practice else args.seed_base + ep_idx
        ep_label = "Practice round" if practice else f"Episode {ep_idx + 1}/{args.episodes}"

        if resume is not None:
            print(
                f"{ep_label} (world seed {world_seed}) — resuming from checkpoint "
                f"at step {resume['completed_env_steps']}..."
            )
            records = _load_step_records(logger.dir / "steps.jsonl", ep_idx)
            (
                state, rng, scores,
                traj_obs, traj_actions, traj_rewards, traj_dones, traj_states, traj_text_actions,
            ) = _replay_episode_prefix(world_seed, records, resume["completed_env_steps"])
            current = resume["current_player"]
            pending = list(resume["pending"])
            env_steps = resume["completed_env_steps"]
            bot_rng = np.random.default_rng(world_seed)
            status = "completed"
            ep_t0 = time.monotonic()
            decision_t0 = ep_t0
            pygame.event.clear()
            renderer.pygame_events = []
        else:
            rng = jax.random.PRNGKey(world_seed)
            rng, _rng = jax.random.split(rng)  # same split pattern as the eval harness
            _, state = env.reset(_rng)
            bot_rng = np.random.default_rng(world_seed)

            print(f"{ep_label} (world seed {world_seed}) — playing...")
            if not args.bot:
                if not wait_for_enter(state, f"{ep_label} — press ENTER"):
                    return "quit", None
            # Drop keys pressed before the episode started (warmup / ENTER screen) so
            # they can't become the first action or an instant pause-menu trigger.
            pygame.event.clear()
            renderer.pygame_events = []

            scores = [0.0] * n
            pending = []  # per-player action records for the current env step
            current = 0
            status = "completed"
            env_steps = 0
            ep_t0 = time.monotonic()
            decision_t0 = ep_t0
            traj_obs, traj_actions, traj_rewards, traj_dones = [], [], [], []
            traj_states, traj_text_actions = [], []

        def _pause_menu():
            choice = renderer.show_menu("Paused", PAUSE_MENU_OPTIONS, subtitle=ep_label)
            if choice is None or choice == 0:
                return "resume"
            label = PAUSE_MENU_OPTIONS[choice][0]
            if label == "Save Checkpoint":
                save_checkpoint(out_dir, ep_idx, world_seed, int(state.timestep), current, pending)
                renderer.push_notification("Checkpoint saved", color=(120, 230, 120), duration_ms=2000)
                return "resume"
            if label == "Reload Last Checkpoint":
                if load_checkpoint(out_dir, ep_idx) is None:
                    renderer.push_notification(
                        "No checkpoint to reload", color=(255, 160, 60), duration_ms=2500
                    )
                    return "resume"
                return "reload"
            if label == "Save & Quit":
                save_checkpoint(out_dir, ep_idx, world_seed, int(state.timestep), current, pending)
                return "quit"
            return "abort" if label == "Abort Episode" else "quit"  # "Quit Without Saving"

        def _reload_from_checkpoint():
            ckpt = load_checkpoint(out_dir, ep_idx)
            renderer.screen_surface.fill((12, 12, 16))
            msg = renderer._font_page_title.render("Reloading checkpoint...", True, (255, 255, 255))
            renderer.screen_surface.blit(
                msg,
                (
                    (renderer.screen_size[0] - msg.get_width()) // 2,
                    (renderer.screen_size[1] - msg.get_height()) // 2,
                ),
            )
            pygame.display.flip()
            records = _load_step_records(logger.dir / "steps.jsonl", ep_idx)
            (
                new_state, new_rng, new_scores,
                t_obs, t_actions, t_rewards, t_dones, t_states, t_text_actions,
            ) = _replay_episode_prefix(world_seed, records, ckpt["completed_env_steps"])
            return (
                new_state, new_rng, new_scores, ckpt["current_player"], list(ckpt["pending"]),
                t_obs, t_actions, t_rewards, t_dones, t_states, t_text_actions,
            )

        try:
            while True:
                if renderer.is_quit_requested():
                    status = "quit"
                    break
                if not args.bot and any(
                    e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE
                    for e in renderer.pygame_events
                ):
                    outcome = _pause_menu()
                    pygame.event.clear()
                    renderer.pygame_events = []
                    if outcome == "abort":
                        status = "aborted"
                        break
                    if outcome == "quit":
                        status = "quit"
                        break
                    if outcome == "reload":
                        (
                            state, rng, scores, current, pending,
                            traj_obs, traj_actions, traj_rewards, traj_dones,
                            traj_states, traj_text_actions,
                        ) = _reload_from_checkpoint()
                        decision_t0 = time.monotonic()
                    continue

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

                    # Trajectory capture (pre-step) — same format alem.trajectory_io
                    # uses for LLM/RL eval episodes, so all three tracks are
                    # comparable and replayable with the same tooling.
                    traj_obs.append(np.asarray(render_symbolic_fn(state)))
                    traj_states.append(jax.device_get(state))
                    traj_text_actions.append([pending[i]["name"] for i in range(n)])

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

                    traj_actions.append(
                        np.array([pending[i]["id"] for i in range(n)], dtype=np.int32)
                    )
                    traj_rewards.append(np.array(rew, dtype=np.float32))
                    traj_dones.append(done)

                    # Cheap (tiny JSON) auto-checkpoint after every completed round,
                    # so --resume can pick up mid-episode even without an explicit
                    # pause-menu save.
                    save_checkpoint(out_dir, ep_idx, world_seed, env_steps, 0, [])

                    logger.log_step(
                        {
                            "episode": ep_idx,
                            "practice": practice,
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

        traj_path = save_trajectory_npz(
            out_dir / f"episode_{_episode_tag(ep_idx)}_trajectory.npz",
            obs=traj_obs,
            actions=traj_actions,
            rewards=traj_rewards,
            dones=traj_dones,
            text_actions=traj_text_actions,
        )
        states_path = save_state_bundle(
            out_dir / f"episode_{_episode_tag(ep_idx)}_states.pkl.gz", traj_states, static_params
        )

        record = {
            "session_id": session_id,
            "participant": participant,
            "practice": practice,
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
            "trajectory_path": traj_path,
            "states_path": states_path,
            "metrics": metrics,
        }
        logger.log_episode(record)

        # A finished episode has nothing left to resume into.
        if status in ("completed", "aborted"):
            _delete_checkpoint(out_dir, ep_idx)

        print(f"{ep_label} {status}: {env_steps} env steps, {record['duration_s'] / 60:.1f} min")
        _print_metrics_line(metrics)
        return status, record

    if args.practice and resumed_meta is None:
        print("\n=== Practice round (not scored) ===")
        run_episode(PRACTICE_EP_IDX, practice=True)
        print()

    start_ep_idx = 0
    resume_checkpoint = None
    if args.resume:
        start_ep_idx = _next_episode_index(out_dir, args.episodes)
        if start_ep_idx >= args.episodes:
            print(f"Session already has all {args.episodes} planned episode(s) — nothing to resume.")
        else:
            resume_checkpoint = load_checkpoint(out_dir, start_ep_idx)
            if resume_checkpoint is not None:
                print(
                    f"Resuming episode {start_ep_idx + 1}/{args.episodes} from checkpoint "
                    f"(saved {resume_checkpoint['saved_at']}, "
                    f"{resume_checkpoint['completed_env_steps']} steps in)."
                )
            else:
                print(
                    f"Resuming session at episode {start_ep_idx + 1}/{args.episodes} "
                    "(no in-episode checkpoint found; starting it fresh)."
                )

    episodes = []
    try:
        for ep_idx in range(start_ep_idx, args.episodes):
            status, record = run_episode(
                ep_idx, resume=(resume_checkpoint if ep_idx == start_ep_idx else None)
            )
            if record is not None:
                episodes.append(record)
            if status in ("quit", "interrupted"):
                break
    finally:
        logger.close(
            ended_at=_now_iso(),
            episodes_completed=_next_episode_index(out_dir, args.episodes),
            episode_statuses=[r["status"] for r in episodes],
        )
        pygame.quit()

    if episodes:
        print(f"\nSession {session_id}: {len(episodes)} episode(s)")
        _print_summary(episodes)
    print(f"\nData written to {out_dir}")
    print("Aggregate all sessions with: python examples/play_with_human_interface.py --aggregate")


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


def _load_all_episodes(root: Path):
    """Read every episodes.jsonl under root, excluding practice rounds."""
    rows = []
    for f in sorted(root.glob("*/episodes.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return [r for r in rows if not r.get("practice")]


def aggregate(root: Path):
    """Summarise every session under ``root`` grouped by coordination difficulty."""
    rows = _load_all_episodes(root)
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


def export_csv(root: Path, out_csv: Path):
    """Export every non-practice episode under ``root`` to a flat CSV for stats software."""
    rows = _load_all_episodes(root)
    if not rows:
        print(f"No episodes found under {root}")
        return

    fieldnames = [
        "session_id", "participant", "difficulty", "players", "episode", "world_seed",
        "status", "env_steps", "duration_s", "team_score",
    ] + HEADLINE_METRICS
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            row = {
                "session_id": r["session_id"],
                "participant": r["participant"],
                "difficulty": r["difficulty"],
                "players": r["players"],
                "episode": r["episode"],
                "world_seed": r["world_seed"],
                "status": r["status"],
                "env_steps": r["env_steps"],
                "duration_s": r["duration_s"],
                "team_score": sum(r.get("scores", [])),
            }
            for key in HEADLINE_METRICS:
                row[key] = r["metrics"].get(key)
            writer.writerow(row)
    print(f"Wrote {len(rows)} episode(s) to {out_csv}")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Run or aggregate Alem human-interface play sessions")
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
        "--out", type=str, default="outputs/play_with_human_interface", help="Results root directory"
    )
    parser.add_argument("--bot", action="store_true", help="Random-action smoke test (no human)")
    parser.add_argument("--headless", action="store_true", help="Dummy SDL video driver (for --bot)")
    parser.add_argument("--skip-rules", action="store_true", help="Skip the on-screen rules page")
    parser.add_argument(
        "--practice", action="store_true", help="Play one short, unscored practice round first"
    )
    parser.add_argument(
        "--skip-consent", action="store_true", help="Skip the console consent/demographics prompts"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to a session directory to resume"
    )
    parser.add_argument("--aggregate", action="store_true", help="Summarise sessions under --out")
    parser.add_argument(
        "--export-csv", type=str, default=None, help="Write all episodes under --out to this CSV path"
    )
    args = parser.parse_args(argv)

    if (
        not args.aggregate
        and not args.export_csv
        and not args.bot
        and not args.participant
        and not args.resume
    ):
        parser.error("--participant is required (or use --bot for a smoke test, or --resume)")
    return args


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    if args.export_csv:
        export_csv(Path(args.out), Path(args.export_csv))
    if args.aggregate:
        aggregate(Path(args.out))
    if not args.aggregate and not args.export_csv:
        run_session(args)
