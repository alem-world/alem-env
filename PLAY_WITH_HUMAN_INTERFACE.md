# Alem Play-with-Human-Interface Guide

`examples/play_with_human_interface.py` runs the standard [evaluation protocol](EVALUATION.md)
with humans at the keyboard instead of an agent, so a human baseline is directly
comparable to the RL/LLM leaderboard numbers. It reuses the same trajectory/state
saving code as the LLM/RL eval harness (`alem.trajectory_io`), so all three
tracks' episodes can be loaded and replayed with the same tooling.

## Standard settings

Matches [`EVALUATION.md`](EVALUATION.md) so human runs are comparable to agent
runs on the leaderboard — don't change these for a result you intend to compare.

| Setting | Value | Flag |
| --- | --- | --- |
| Agents | 3 | `--players` |
| Soft specialisation | on | (baked in) |
| Shared reward | off | (baked in) |
| Max steps / episode | 10000 | `--max-steps` |
| Eval seed base | 9999 | `--seed-base` |
| Coordination difficulty | `easy`, `medium`, or `hard` | `--coord` |

Episode `i` in a session uses world seed `seed-base + i` — the identical worlds
agents were scored on. Run one session per participant/team per difficulty you
want to report.

## Before you start

1. **Consent script.** `play_with_human_interface.py` prints a placeholder consent notice and
   records a single console y/N reply that covers everyone hotseating on the
   keyboard — it is **not** an approved IRB/ethics form. Get your institution's
   actual approved consent text and either read it aloud before running the
   tool, or edit `_CONSENT_SCRIPT` in `play_with_human_interface.py` to show it
   on-screen. Consent is collected before any file is written; declining exits
   with nothing recorded. After consent, the optional demographics questions
   (age range, gaming experience, survival-game familiarity) are asked once
   per player (`--players`), one after another, so each hotseat player answers
   for themself.
2. **Anonymous ids.** Use an anonymous participant/team id (e.g. `T01`), never a
   real name — `--participant` becomes part of the output directory name and
   every logged record.
3. **A short pilot run.** Try `--max-steps 500` or so on yourself first to check
   controls, framerate, and that files land where you expect, before running a
   real participant at the full 10000-step cap.
4. **Smoke-test the pipeline**, not a real session, with `--bot --headless` (see
   below) — it drives random actions with no display, useful for CI or after
   changing this file. On a machine with a single GPU that also drives the
   desktop, force `JAX_PLATFORMS=cpu` for headless smoke tests to avoid the
   video driver fighting JAX for VRAM.

## Running a session

```bash
# A 3-human team (or one human hot-seating all three), 2 episodes on easy:
python examples/play_with_human_interface.py --participant T01 --coord easy --episodes 2

# A short practice round first, so the team learns the controls before it counts:
python examples/play_with_human_interface.py --participant T01 --coord easy --episodes 2 --practice

# Pilot yourself with a shorter episode cap:
python examples/play_with_human_interface.py --participant pilot --coord medium --max-steps 2000

# Verify the whole pipeline with a random bot, no display needed:
JAX_PLATFORMS=cpu python examples/play_with_human_interface.py --bot --headless --episodes 2 --max-steps 50
```

Hotseat flow: players take turns on one keyboard; the world advances once every
player has acted. The highlighted player at the top of the screen is the one
who currently acts, and the view is centred on them. For a genuine coordination
session, seat one human per player at the same keyboard and allow them to talk —
that's the human analogue of the game's communication channel.

Before the first episode, participants see the same `<game_rules>` briefing the
LLM agents receive (also saved to `rules.txt` in the session directory), plus a
keyboard reference. Skip the on-screen page with `--skip-rules` if you brief
participants yourself instead.

### Controls

| Action | Key(s) |
| --- | --- |
| Move (N/W/S/E) | `W` `A` `S` `D` |
| Do / interact | `Space` |
| Sleep / Rest / Skip turn | `Tab` / `E` / `Q` |
| Descend / Ascend | `.` / `,` |
| Place stone/table/furnace/plant/torch | `R` `T` `F` `P` `J` |
| Craft pickaxe (wood/stone/iron/diamond) | `1` `2` `3` `4` |
| Craft sword (wood/stone/iron/diamond) | `5` `6` `7` `8` |
| Craft armour (iron/diamond) | `Y` `U` |
| Craft arrow / torch | `O` / `[` |
| Shoot arrow / cast spell | `I` / `G` |
| Request a resource from the team | `F1`–`F9` |
| Give held resource to a teammate | `Backspace` |
| Build shelter/forge/beacon | `9` `0` `` ` `` |
| Start next episode | `Enter` |
| Pause menu | `Esc` |

## Pausing, saving, and resuming

Press **Esc** at any point during an episode to open the pause menu:

- **Resume** — close the menu, keep playing.
- **Save Checkpoint** — write a checkpoint without stopping (also happens
  automatically after every completed round of play, so accidental closes lose
  at most one round).
- **Reload Last Checkpoint** — discard everything since the last checkpoint and
  continue from there. Useful if the team wants to retry a section.
- **Save & Quit** — checkpoint, then end the process. Resume later.
- **Abort Episode** — end this episode now (recorded `status=aborted`, scored
  as-is); the team moves on to the next episode or ends the session.
- **Quit Without Saving** — close immediately.

To resume a session that was saved-and-quit, interrupted (Ctrl+C), or crashed,
point `--resume` at its output directory:

```bash
python examples/play_with_human_interface.py --resume outputs/play_with_human_interface/T01_easy_20260725-153000
```

`--resume` restores the original protocol (participant, difficulty, seeds,
episode count) from `session.json` — you don't repeat any flags — and picks up
at the first episode that never reached a final (`completed`/`aborted`) state,
replaying its logged actions to reconstruct the in-progress episode exactly
before handing control back to the player. Consent and the rules screen are not
shown again on resume.

Checkpoints are tiny (a JSON bookmark, not a full state snapshot) because Alem
is a deterministic function of `(world seed, action sequence)` — resuming
always works by replaying the actions already logged in `steps.jsonl`, never by
loading a saved simulator state. One known limitation: an episode's reported
`duration_s` only covers wall-clock time within a single process, so it resets
across a `--resume` (env step counts and all other data are unaffected).

## Output files

Every session writes to `outputs/play_with_human_interface/<participant>_<coord>_<timestamp>/`:

| File | Contents |
| --- | --- |
| `session.json` | Participant id, consent/demographics, full protocol config, git commit, library versions |
| `rules.txt` | The exact briefing text shown to the participant |
| `steps.jsonl` | One record per env step: per-player action + decision latency (ms), rewards, health/food/drink/energy, positions, achievement unlocks |
| `episodes.jsonl` | One record per episode: status, duration, env steps, and the canonical `compute_score()` metrics (leaderboard-comparable) |
| `episode_XX_trajectory.npz` | Symbolic obs/actions/rewards/dones for episode `XX`, same format `alem.trajectory_io` uses for LLM/RL eval episodes |
| `episode_XX_states.pkl.gz` | Full env-state snapshot per step, for exact replay |
| `episode_XX_checkpoint.json` | Present only while an episode is mid-flight and unfinished; deleted once it completes or is aborted |

Practice episodes use the tag `practice` instead of a zero-padded index (e.g.
`episode_practice_trajectory.npz`) and are excluded from `--aggregate` and
`--export-csv`.

Everything is flushed to disk as it's written (line-buffered JSONL), so a crash
or `Ctrl+C` loses at most the current in-progress step.

## After collecting data

```bash
# Mean ± std per coordination difficulty, across every session under --out:
python examples/play_with_human_interface.py --aggregate

# Flat CSV of every episode, for R/pandas/etc.:
python examples/play_with_human_interface.py --export-csv outputs/play_with_human_interface/episodes.csv
```

Both read every `episodes.jsonl` under `--out` (default `outputs/play_with_human_interface/`)
and need no JAX/pygame imports, so they're fast even over many sessions.
`Team/achievement_pct` (and its coordination/normal/reward breakdowns) is the
same headline metric reported for RL/LLM agents — see
[`EVALUATION.md`](EVALUATION.md#headline-metric).

Back up the whole `outputs/play_with_human_interface/` directory (or each session directory)
after every session visit — it's the only copy of the raw data.

## Reviewing a session

`examples/replay_play_with_human_interface.py` plays back a saved episode from its
`*_states.pkl.gz` file — for spot-checking that nothing was corrupted by a
save/reload, or producing a clip for a talk:

```bash
# Interactively view episode 0 of a session (space=play/pause, arrows=step/speed, P=player, Q/Esc=quit):
python examples/replay_play_with_human_interface.py outputs/play_with_human_interface/T01_easy_20260725-153000 --episode 0

# Export a GIF instead (works headless, no display needed):
python examples/replay_play_with_human_interface.py <session_dir> --episode 0 --headless --save-gif out.gif
```

## Checklist for a play session visit

1. Approved consent text ready (replace the placeholder — see above).
2. Confirm `--coord`, `--episodes`, and `--players` match your protocol before
   the first participant, not after.
3. Run `--practice` first so the team learns the controls before anything is
   scored.
4. Mention Esc is the pause menu, not an abort — the old play-testing habit of
   mashing Esc to quit will otherwise save/reload data unexpectedly.
5. After the session, spot check with `replay_play_with_human_interface.py`, then back up the
   output directory.
6. Once all visits are done: `--aggregate` for a quick look, `--export-csv` for
   stats software.
