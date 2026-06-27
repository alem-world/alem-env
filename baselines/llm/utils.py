"""
Utility functions for Alem LLM evaluation.
"""

import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import wandb
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


_REWARD_PCT_USER_INFO_KEYS = (
    "Team/reward_pct_of_max",
    "Team/normal_reward_pct_of_max",
    "Team/coord_reward_pct_of_max",
)


def setup_environment(original_cwd=""):
    """Setup environment variables and paths."""
    if original_cwd:
        os.chdir(original_cwd)
    random.seed(42)
    np.random.seed(42)
    logger.info("Environment setup complete")


def get_unique_seed(process_num=None, episode_idx=0):
    """Generate a unique seed based on process number and episode index."""
    base = int(time.time() * 1000) % 1000000
    if process_num is not None:
        if isinstance(process_num, str):
            process_hash = hash(process_num) % 1000
        else:
            process_hash = int(process_num) % 1000
        base += process_hash * 1000
    seed = base + episode_idx
    return seed % (2**31)


def collect_and_summarize_results(output_dir):
    """Collect results from all evaluation episodes."""
    summary = defaultdict(
        lambda: {
            "episodes": [],
            "failed_episodes": [],  # episodes excluded from stats (e.g. API errors, repeated incomplete responses)
            "total_reward": 0.0,
            "total_steps": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "incomplete_response_count": 0,
            "incomplete_response_reasons": defaultdict(int),
            "stop_reason_counts": defaultdict(int),
            "success_rate": 0.0,
            "avg_reward": 0.0,
            "avg_steps": 0.0,
            "avg_score": 0.0,
            "avg_progression": 0.0,
            "achievement_counts": defaultdict(int),
            "all_progressions": [],
            "all_achievements": set(),
            "per_agent_rewards": defaultdict(list),
            "per_agent_achievements": defaultdict(list),
            "per_agent_achievement_pcts": defaultdict(list),
            "user_info_accum": defaultdict(list),
            "all_episode_steps": [],
        }
    )

    output_path = Path(output_dir)

    for json_file in output_path.rglob("*.json"):
        if (
            "_run_" in json_file.name and json_file.name.endswith(".json")
        ) or json_file.name == "episode_log.json":
            try:
                with open(json_file) as f:
                    episode_log = json.load(f)

                relative_path = json_file.relative_to(output_path)
                parts = relative_path.parts

                if len(parts) >= 2:
                    env_name = parts[0]
                    task = parts[1] if len(parts) > 2 else "default"
                    key = f"{env_name}/{task}"

                    episode_crashed = "error" in episode_log
                    episode_invalid = (
                        episode_crashed
                        or episode_log.get("early_stop_reason")
                        == "consecutive_length_incomplete_responses"
                    )
                    summary[key]["episodes"].append(episode_log)
                    if episode_invalid:
                        summary[key]["failed_episodes"].append(episode_log)
                        # Skip invalid episodes from all stat accumulators below
                        continue
                    summary[key]["total_reward"] += episode_log.get("episode_return", 0.0)
                    ep_steps = episode_log.get("num_steps", 0)
                    summary[key]["total_steps"] += ep_steps
                    summary[key]["all_episode_steps"].append(ep_steps)
                    summary[key]["input_tokens"] += episode_log.get("input_tokens", 0)
                    summary[key]["output_tokens"] += episode_log.get("output_tokens", 0)
                    summary[key]["reasoning_tokens"] += episode_log.get("reasoning_tokens", 0)
                    summary[key]["incomplete_response_count"] += episode_log.get(
                        "incomplete_response_count", 0
                    )
                    for reason, count in episode_log.get("incomplete_response_reasons", {}).items():
                        summary[key]["incomplete_response_reasons"][reason] += count
                    for reason, count in episode_log.get("stop_reason_counts", {}).items():
                        summary[key]["stop_reason_counts"][reason] += count

                    agent_stats = []
                    for i in range(10):
                        agent_key = f"agent_{i}"
                        if agent_key in episode_log:
                            agent_stats.append((i, episode_log[agent_key]))
                            if f"agent_{i}_return" in episode_log:
                                summary[key]["per_agent_rewards"][i].append(
                                    episode_log[f"agent_{i}_return"]
                                )

                    if not agent_stats:
                        if "score" in episode_log:
                            agent_stats = [
                                (
                                    0,
                                    {
                                        "score": episode_log["score"],
                                        "progression": episode_log.get("progression", 0.0),
                                        "achievements": episode_log.get("achievements", {}),
                                    },
                                )
                            ]
                            if "episode_return" in episode_log:
                                summary[key]["per_agent_rewards"][0].append(
                                    episode_log["episode_return"]
                                )

                    for agent_id, agent_stat in agent_stats:
                        if "score" in agent_stat:
                            summary[key]["total_score"] = (
                                summary[key].get("total_score", 0.0) + agent_stat["score"]
                            )
                        if "progression" in agent_stat:
                            progression = agent_stat["progression"]
                            summary[key]["total_progression"] = (
                                summary[key].get("total_progression", 0.0) + progression
                            )
                            summary[key]["all_progressions"].append(progression)

                        if "achievements" in agent_stat and agent_stat["achievements"]:
                            for achievement_name in agent_stat["achievements"].keys():
                                summary[key]["all_achievements"].add(achievement_name)

                            num_achievements = len(agent_stat["achievements"])
                            achieved_count = sum(
                                1 for v in agent_stat["achievements"].values() if v == 1
                            )

                            summary[key]["per_agent_achievements"][agent_id].append(achieved_count)
                            if num_achievements > 0:
                                summary[key]["per_agent_achievement_pcts"][agent_id].append(
                                    (achieved_count / num_achievements) * 100.0
                                )

                            for achievement_name, achieved in agent_stat["achievements"].items():
                                if achieved == 1:
                                    summary[key]["achievement_counts"][achievement_name] += 1

                    # Accumulate user_info metrics from compute_score
                    if "user_info" in episode_log:
                        for metric_key, value in episode_log["user_info"].items():
                            summary[key]["user_info_accum"][metric_key].append(value)

                    # Collect per-agent debriefs for the aggregated debrief file
                    if "debriefs" in episode_log:
                        ep_idx = episode_log.get("seed", len(summary[key].get("debriefs", [])))
                        if "debriefs" not in summary[key]:
                            summary[key]["debriefs"] = []
                        summary[key]["debriefs"].append(
                            {
                                "episode": ep_idx,
                                "agents": episode_log["debriefs"],
                            }
                        )

            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")

    # Calculate averages and achievement percentages (over valid episodes only)
    for key, data in summary.items():
        num_episodes = len(data["episodes"]) - len(data["failed_episodes"])
        if num_episodes > 0:
            data["avg_reward"] = data["total_reward"] / num_episodes
            data["avg_steps"] = data["total_steps"] / num_episodes
            steps_arr = np.array(data["all_episode_steps"])
            data["sem_steps"] = (
                float(np.std(steps_arr) / np.sqrt(num_episodes)) if num_episodes > 1 else 0.0
            )
            data["success_rate"] = (
                sum(1 for ep in data["episodes"] if ep.get("done", False)) / num_episodes
            )

            if "total_score" in data:
                num_agent_measurements = (
                    len(data["all_progressions"]) if data["all_progressions"] else num_episodes
                )
                data["avg_score"] = data["total_score"] / num_agent_measurements
            if "total_progression" in data:
                num_agent_measurements = (
                    len(data["all_progressions"]) if data["all_progressions"] else num_episodes
                )
                data["avg_progression"] = data["total_progression"] / num_agent_measurements
                if len(data["all_progressions"]) > 1:
                    data["std_progression"] = np.std(data["all_progressions"])
                else:
                    data["std_progression"] = 0.0

            num_agent_measurements = (
                len(data["all_progressions"]) if data["all_progressions"] else num_episodes
            )
            data["achievement_percentages"] = {}
            total_achievements_achieved = 0
            total_possible_achievements = (
                len(data["all_achievements"]) * num_agent_measurements
                if data["all_achievements"]
                else 0
            )

            for achievement in sorted(data["all_achievements"]):
                count = data["achievement_counts"].get(achievement, 0)
                data["achievement_percentages"][achievement] = (
                    count / num_agent_measurements
                ) * 100.0
                total_achievements_achieved += count

            if total_possible_achievements > 0:
                data["overall_achievement_percentage"] = (
                    total_achievements_achieved / total_possible_achievements
                ) * 100.0
            else:
                data["overall_achievement_percentage"] = 0.0

            if num_agent_measurements > 0:
                data["avg_achievements_per_agent"] = (
                    total_achievements_achieved / num_agent_measurements
                )
            else:
                data["avg_achievements_per_agent"] = 0.0

            # Aggregate user_info metrics (from compute_score)
            if data["user_info_accum"]:
                data["user_info_means"] = {
                    k: float(np.mean(v)) for k, v in data["user_info_accum"].items()
                }
                data["user_info_stds"] = {
                    k: float(np.std(v)) for k, v in data["user_info_accum"].items()
                }
                data["user_info_sem"] = {
                    k: float(np.std(v) / np.sqrt(len(v))) if len(v) > 1 else 0.0
                    for k, v in data["user_info_accum"].items()
                }
                data["user_info_n"] = {k: len(v) for k, v in data["user_info_accum"].items()}
            del data["user_info_accum"]  # Don't serialize the raw lists

    return dict(summary)


def print_summary_table(summary):
    """Print a formatted summary table of results."""
    if not summary:
        print("\nNo results found to summarize")
        return

    print("\n" + "=" * 120)
    print("EVALUATION SUMMARY")
    print("=" * 120)

    print(
        f"\n{'Environment/Task':<40} {'Episodes':>10} {'Failed':>8} {'Mean Return':>12} {'Avg Steps':>10} {'Team Ach%':>10} {'Team Ach#':>10} {'Player Lvl':>12}"
    )
    print("-" * 130)

    for key, data in sorted(summary.items()):
        num_episodes = len(data["episodes"])
        num_failed = len(data.get("failed_episodes", []))
        avg_steps = data["avg_steps"]
        ui = data.get("user_info_means", {})
        team_ach_pct = ui.get("Team/achievement_pct", 0.0) * 100
        team_ach_total = ui.get("Team/total_achievements", 0.0)
        player_level = ui.get("Progression/player_level", 0.0)
        # Mean episode return matching wandb: mean of per-agent rewards
        per_agent = data.get("per_agent_rewards", {})
        all_r = [r for rewards in per_agent.values() for r in rewards]
        mean_return = np.mean(all_r) if all_r else 0.0
        print(
            f"{key:<40} {num_episodes:>10} {num_failed:>8} {mean_return:>12.2f} {avg_steps:>10.1f} {team_ach_pct:>9.2f}% {team_ach_total:>10.2f} {player_level:>12.2f}"
        )

    print("=" * 130)

    total_episodes = sum(len(data["episodes"]) for data in summary.values())
    total_failed = sum(len(data.get("failed_episodes", [])) for data in summary.values())
    total_valid = total_episodes - total_failed
    total_steps = sum(data["total_steps"] for data in summary.values())

    # Collect per-agent rewards (valid episodes only — failed ones were never added)
    all_per_agent = {}
    for data in summary.values():
        for agent_id, rewards in data.get("per_agent_rewards", {}).items():
            all_per_agent.setdefault(agent_id, []).extend(rewards)
    all_rewards_flat = [r for rewards in all_per_agent.values() for r in rewards]

    # Collect user_info_means across tasks (same source as wandb)
    all_user_info = defaultdict(list)
    for data in summary.values():
        if "user_info_means" in data:
            for k, v in data["user_info_means"].items():
                all_user_info[k].append(v)

    # Collect user_info_sem across tasks
    all_user_info_sem = defaultdict(list)
    for data in summary.values():
        if "user_info_sem" in data:
            for k, v in data["user_info_sem"].items():
                all_user_info_sem[k].append(v)

    # Episode length SE across valid episodes only
    all_ep_lengths = []
    for data in summary.values():
        all_ep_lengths.extend(data.get("all_episode_steps", []))

    print("\nOVERALL STATISTICS")
    print(
        f"   Total Episodes: {total_valid} valid, {total_failed} failed (API error), {total_episodes} total"
    )
    print(f"   Total Steps: {total_steps} (valid episodes only)")
    if total_valid > 0:
        mean_len = total_steps / total_valid
        se_len = (
            float(np.std(all_ep_lengths) / np.sqrt(len(all_ep_lengths)))
            if len(all_ep_lengths) > 1
            else 0.0
        )
        print(f"   Mean Episode Length: {mean_len:.1f} ± {se_len:.1f}  (valid only)")

        # Mean episode return: mean of per-agent rewards (valid episodes only)
        if all_rewards_flat:
            mean_ret = np.mean(all_rewards_flat)
            se_ret = (
                np.std(all_rewards_flat) / np.sqrt(len(all_rewards_flat))
                if len(all_rewards_flat) > 1
                else 0.0
            )
            print(f"   Mean Episode Return: {mean_ret:.3f} ± {se_ret:.3f}  (valid only)")
            for agent_id in sorted(all_per_agent):
                rewards = all_per_agent[agent_id]
                print(f"   Agent {agent_id} Mean Reward: {np.mean(rewards):.3f}")

        # Key paper metrics with mean ± SE
        _PAPER_METRICS = [
            ("Team/reward_pct_of_max", "Team Reward % of Max", 100.0),
            ("Team/normal_reward_pct_of_max", "Team Normal Reward % of Max", 100.0),
            ("Team/coord_reward_pct_of_max", "Team Coord Reward % of Max", 100.0),
            ("Team/normal_achievement_pct", "Team Normal Achievement %", 100.0),
            ("Team/coordination_achievement_pct", "Team Coord Achievement %", 100.0),
            ("Team/achievement_pct", "Team Achievement %", 100.0),
            ("Progression/player_level", "Mean Player Level", 1.0),
        ]

        print("\n   KEY PAPER METRICS (mean ± SE):")
        for metric_key, label, scale in _PAPER_METRICS:
            if metric_key in all_user_info:
                vals = all_user_info[metric_key]
                mean_val = np.mean(vals) * scale
                # Pooled SE: combine per-task SEMs in quadrature (tasks are independent)
                sems = all_user_info_sem.get(metric_key, [])
                se_val = float(np.sqrt(np.mean(np.square(sems)))) * scale if sems else 0.0
                print(f"   {label:<35} {mean_val:>8.3f} ± {se_val:.3f}")

    print("\nACHIEVEMENT STATISTICS")
    for key, data in sorted(summary.items()):
        if data.get("achievement_percentages"):
            print(f"\n   {key}:")
            overall_pct = data.get("overall_achievement_percentage", 0.0)
            avg_per_agent = data.get("avg_achievements_per_agent", 0.0)
            total_achievements_in_game = len(data["all_achievements"])
            print(f"      Achievement Rate: {overall_pct:.1f}% ({avg_per_agent:.2f} avg per agent)")
            print(
                f"      Unique Achievements Reached: {sum(1 for p in data['achievement_percentages'].values() if p > 0)}/{total_achievements_in_game}"
            )
            print()
            sorted_achievements = sorted(
                data["achievement_percentages"].items(), key=lambda x: x[1], reverse=True
            )
            for achievement, percentage in sorted_achievements:
                print(f"      {achievement:<30} {percentage:>6.1f}%")

    print("=" * 120 + "\n")


def save_config(config, output_dir):
    """Save configuration to output directory."""
    config_path = Path(output_dir) / "config.yaml"
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(config))
    logger.info(f"Saved configuration to {config_path}")


def save_summary_stats(summary, output_dir):
    """Save detailed summary statistics to JSON file."""
    summary_path = Path(output_dir) / "summary_stats.json"

    summary_clean = {}
    for key, data in summary.items():
        clean_data = {
            "num_episodes": len(data["episodes"]),
            "avg_reward": float(data["avg_reward"]),
            "avg_steps": float(data["avg_steps"]),
            "success_rate": float(data["success_rate"]),
            "avg_score": float(data.get("avg_score", 0.0)),
            "avg_progression": float(data.get("avg_progression", 0.0)),
            "std_progression": float(data.get("std_progression", 0.0)),
            "overall_achievement_percentage": float(
                data.get("overall_achievement_percentage", 0.0)
            ),
            "avg_achievements_per_agent": float(data.get("avg_achievements_per_agent", 0.0)),
            "unique_achievements_reached": sum(
                1 for p in data.get("achievement_percentages", {}).values() if p > 0
            ),
            "total_achievements_available": len(data.get("all_achievements", [])),
            "achievement_percentages": {
                k: float(v) for k, v in data.get("achievement_percentages", {}).items()
            },
            "achievement_counts": dict(data.get("achievement_counts", {})),
        }
        num_valid_episodes = max(len(data["episodes"]) - len(data.get("failed_episodes", [])), 1)
        clean_data["input_tokens"] = float(data.get("input_tokens", 0))
        clean_data["output_tokens"] = float(data.get("output_tokens", 0))
        clean_data["reasoning_tokens"] = float(data.get("reasoning_tokens", 0))
        clean_data["avg_input_tokens"] = float(data.get("input_tokens", 0) / num_valid_episodes)
        clean_data["avg_output_tokens"] = float(data.get("output_tokens", 0) / num_valid_episodes)
        clean_data["avg_reasoning_tokens"] = float(
            data.get("reasoning_tokens", 0) / num_valid_episodes
        )
        clean_data["incomplete_response_count"] = int(data.get("incomplete_response_count", 0))
        clean_data["avg_incomplete_responses"] = float(
            data.get("incomplete_response_count", 0) / num_valid_episodes
        )
        clean_data["incomplete_response_reasons"] = {
            k: int(v) for k, v in data.get("incomplete_response_reasons", {}).items()
        }
        clean_data["stop_reason_counts"] = {
            k: int(v) for k, v in data.get("stop_reason_counts", {}).items()
        }

        per_agent_stats = {}
        if "per_agent_rewards" in data:
            for agent_id, rewards in data["per_agent_rewards"].items():
                if rewards:
                    rewards_array = np.array(rewards)
                    if agent_id not in per_agent_stats:
                        per_agent_stats[agent_id] = {}
                    per_agent_stats[agent_id]["mean_reward"] = float(rewards_array.mean())
                    per_agent_stats[agent_id]["std_reward"] = float(rewards_array.std())

        if "per_agent_achievements" in data:
            for agent_id, achievements in data["per_agent_achievements"].items():
                if achievements:
                    ach_array = np.array(achievements)
                    if agent_id not in per_agent_stats:
                        per_agent_stats[agent_id] = {}
                    per_agent_stats[agent_id]["mean_achievements"] = float(ach_array.mean())
                    per_agent_stats[agent_id]["std_achievements"] = float(ach_array.std())

        if "per_agent_achievement_pcts" in data:
            for agent_id, pcts in data["per_agent_achievement_pcts"].items():
                if pcts:
                    pct_array = np.array(pcts)
                    if agent_id not in per_agent_stats:
                        per_agent_stats[agent_id] = {}
                    per_agent_stats[agent_id]["mean_achievement_pct"] = float(pct_array.mean())
                    per_agent_stats[agent_id]["std_achievement_pct"] = float(pct_array.std())

        if per_agent_stats:
            clean_data["per_agent_stats"] = {str(k): v for k, v in per_agent_stats.items()}

        if "user_info_means" in data:
            clean_data["user_info_means"] = dict(data["user_info_means"])
            for metric_key in _REWARD_PCT_USER_INFO_KEYS:
                clean_data["user_info_means"].setdefault(metric_key, 0.0)
        if "user_info_stds" in data:
            clean_data["user_info_stds"] = dict(data["user_info_stds"])
            for metric_key in _REWARD_PCT_USER_INFO_KEYS:
                clean_data["user_info_stds"].setdefault(metric_key, 0.0)
        if "user_info_sem" in data:
            clean_data["user_info_sem"] = dict(data["user_info_sem"])
            for metric_key in _REWARD_PCT_USER_INFO_KEYS:
                clean_data["user_info_sem"].setdefault(metric_key, 0.0)
        if "user_info_n" in data:
            clean_data["user_info_n"] = dict(data["user_info_n"])
            for metric_key in _REWARD_PCT_USER_INFO_KEYS:
                clean_data["user_info_n"].setdefault(metric_key, 0)

        clean_data["reward_pct_of_max_metrics"] = {
            metric_key: clean_data.get("user_info_means", {}).get(metric_key, 0.0)
            for metric_key in _REWARD_PCT_USER_INFO_KEYS
        }

        summary_clean[key] = clean_data

    with open(summary_path, "w") as f:
        json.dump(summary_clean, f, indent=2)

    logger.info(f"Saved summary statistics to {summary_path}")
    print(f"\nSummary statistics saved to: {summary_path}")

    # Write aggregated debrief file if any episodes had debriefs
    all_debriefs = []
    for key, data in summary.items():
        for entry in data.get("debriefs", []):
            all_debriefs.append((key, entry))

    if all_debriefs:
        debrief_path = Path(output_dir) / "debriefs.md"
        with open(debrief_path, "w", encoding="utf-8") as f:
            f.write("# Post-Episode Debriefs\n\n")
            f.write(f"*{len(all_debriefs)} episode(s) with debriefs across all tasks.*\n\n")
            f.write("---\n\n")
            for task_key, entry in all_debriefs:
                ep = entry["episode"]
                f.write(f"## Episode {ep} — {task_key}\n\n")
                for agent_key, text in sorted(entry["agents"].items()):
                    f.write(f"### {agent_key}\n\n")
                    f.write((text or "(no debrief)").strip())
                    f.write("\n\n")
                f.write("---\n\n")
        logger.info(f"Saved aggregated debriefs to {debrief_path}")
        print(f"Aggregated debriefs saved to: {debrief_path}")


@contextmanager
def redirect_to_file(filepath):
    """Redirect stdout to file while also printing to console."""
    original = sys.stdout
    with open(filepath, "w") as file:
        sys.stdout = file
        try:
            yield
        finally:
            sys.stdout = original


def log_results_to_wandb(summary, config, output_dir=None):
    """Log evaluation results to W&B.

    Logs:
    - Scalar metrics (rewards, achievements, progression, etc.)
    - Summary table
    - Debug artifacts: GIFs (as wandb.Video), HTML viewers, debug JSONL,
      episode JSONs, and CSVs uploaded as a wandb.Artifact.
    """
    import numpy as np

    num_agents = config.alem.num_agents
    # Difficulty prefix to match RL eval key format: eval/{difficulty}/{metric}
    difficulty = getattr(config.alem, "coordination_difficulty", None) or "default"
    ep = f"eval/{difficulty}"  # e.g. eval/easy, eval/medium, eval/hard

    total_episodes = sum(len(data.get("episodes", [])) for data in summary.values())
    total_failed = sum(len(data.get("failed_episodes", [])) for data in summary.values())
    total_valid = total_episodes - total_failed
    total_steps = sum(data.get("total_steps", 0) for data in summary.values())

    for task_key, data in summary.items():
        env_name = task_key.replace("/", "_")
        log_dict = {"task": env_name}

        # ── Episode-level stats (not in user_info) ──
        log_dict[f"{ep}/num_episodes"] = len(data.get("episodes", []))
        log_dict[f"{ep}/total_valid_episodes"] = int(total_valid)
        log_dict[f"{ep}/total_steps"] = int(total_steps)
        log_dict[f"{ep}/mean_episode_length"] = float(data.get("avg_steps", 0.0))
        log_dict[f"{ep}/mean_episode_length_se"] = float(data.get("sem_steps", 0.0))
        if "success_rate" in data:
            log_dict[f"{ep}/success_rate"] = float(data["success_rate"])

        # ── Rewards (not in user_info) ──
        all_rewards = []
        for agent_id in range(num_agents):
            if agent_id in data.get("per_agent_rewards", {}):
                rewards = np.array(data["per_agent_rewards"][agent_id])
                if len(rewards) > 0:
                    log_dict[f"{ep}/Agent{agent_id}/mean_reward"] = float(rewards.mean())
                    log_dict[f"{ep}/Agent{agent_id}/std_reward"] = float(rewards.std())
                    all_rewards.extend(rewards)
        if all_rewards:
            rewards_array = np.array(all_rewards)
            log_dict[f"{ep}/mean_episode_return"] = float(rewards_array.mean())
            log_dict[f"{ep}/mean_episode_return_se"] = (
                float(rewards_array.std() / np.sqrt(len(rewards_array)))
                if len(rewards_array) > 1
                else 0.0
            )

        # ── compute_score metrics: matches RL eval keys exactly ──
        # Keys: eval/{diff}/Team/achievement_pct, eval/{diff}/Agent{N}/achievement_pct,
        #       eval/{diff}/Achievements/{name}, eval/{diff}/Cooperation/*, etc.
        if "user_info_means" in data:
            for metric_key, value in data["user_info_means"].items():
                log_dict[f"{ep}/{metric_key}"] = float(value)
        if "user_info_sem" in data:
            for metric_key, se_value in data["user_info_sem"].items():
                log_dict[f"{ep}/{metric_key}_se"] = float(se_value)

        # --- Upload GIFs as wandb.Video and HTML viewers ---
        if output_dir:
            output_path = Path(output_dir)

            # Log GIFs as wandb.Video (combined + per-agent)
            all_gifs = sorted(output_path.rglob("*.gif"))
            for gif_path in all_gifs:
                episode_name = gif_path.stem  # e.g. "default_run_00" or "default_run_00_agent0"
                if "_llm" in gif_path.name:
                    key_prefix = "episodes/llm_view"
                elif "_agent" in gif_path.name:
                    key_prefix = "episodes/agent_view"
                else:
                    key_prefix = "episodes/combined"
                try:
                    log_dict[f"{key_prefix}/{episode_name}"] = wandb.Video(
                        str(gif_path), fps=10, format="gif"
                    )
                except Exception as e:
                    logger.warning(f"Failed to log GIF {gif_path}: {e}")

            # Log LLM MP4 videos
            all_mp4s = sorted(output_path.rglob("*_llm.mp4"))
            for mp4_path in all_mp4s:
                episode_name = mp4_path.stem
                try:
                    log_dict[f"episodes/llm_video/{episode_name}"] = wandb.Video(
                        str(mp4_path), fps=3, format="mp4"
                    )
                except Exception as e:
                    logger.warning(f"Failed to log MP4 {mp4_path}: {e}")

            # Log debug HTML viewers
            html_files = sorted(output_path.rglob("*_debug.html"))
            for html_path in html_files:
                episode_name = html_path.stem.replace("_debug", "")
                try:
                    with open(html_path, encoding="utf-8") as f:
                        html_content = f.read()
                    log_dict[f"debug/{episode_name}"] = wandb.Html(html_content)
                except Exception as e:
                    logger.warning(f"Failed to log HTML {html_path}: {e}")

            # Log LLM HTML viewers
            llm_html_files = sorted(output_path.rglob("*_llm.html"))
            for html_path in llm_html_files:
                episode_name = html_path.stem
                try:
                    with open(html_path, encoding="utf-8") as f:
                        html_content = f.read()
                    log_dict[f"episodes/llm_html/{episode_name}"] = wandb.Html(html_content)
                except Exception as e:
                    logger.warning(f"Failed to log LLM HTML {html_path}: {e}")

        # ── Windowed ICL metrics (reward & parse rate over episode steps) ──
        all_episodes = data.get("episodes", [])

        # ── Raw per-episode metrics as individual scalars — retrieve via run.history(). ──
        # Logs every scalar in the episode JSON: returns, lengths, achievements,
        # parse rates, token counts, per-agent stats. Use run.history() to pull.
        _EP_SKIP = {
            "task",
            "action_frequency",
            "agent",
            "client",
            "windowed_metrics",
            "windowed_window_size",
            "error",
            "failed_candidates",
            "user_info",
        }
        _EP_SKIP_SUFFIX = ("_action_frequency", "_failed_candidates")
        for ep_data in all_episodes:
            ep_log = {}
            # All top-level scalars (returns, lengths, parse rates, tokens, seed…)
            for k, v in ep_data.items():
                if k in _EP_SKIP or k.endswith(_EP_SKIP_SUFFIX):
                    continue
                if isinstance(v, (dict, list, tuple)):
                    continue
                try:
                    ep_log[f"eval_ep/{difficulty}/{k}"] = float(v)
                except (TypeError, ValueError):
                    pass
            # Per-agent stat dicts from env.get_stats() (score, progression, achievements)
            for k, v in ep_data.items():
                if k.startswith("agent_") and isinstance(v, dict):
                    for stat_k, stat_v in v.items():
                        if stat_k == "achievements" and isinstance(stat_v, dict):
                            for ach_k, ach_v in stat_v.items():
                                try:
                                    ep_log[f"eval_ep/{difficulty}/{k}/achievements/{ach_k}"] = (
                                        float(ach_v)
                                    )
                                except (TypeError, ValueError):
                                    pass
                        elif not isinstance(stat_v, (dict, list)):
                            try:
                                ep_log[f"eval_ep/{difficulty}/{k}/{stat_k}"] = float(stat_v)
                            except (TypeError, ValueError):
                                pass
            # compute_score metrics (Team/*, Agent*/, Achievements/*, Cooperation/*, etc.)
            for k, v in ep_data.get("user_info", {}).items():
                try:
                    ep_log[f"eval_ep/{difficulty}/{k}"] = float(v)
                except (TypeError, ValueError):
                    pass
            if ep_log:
                wandb.log(ep_log)
        windowed_data = defaultdict(
            lambda: {"rewards": [], "parse_rates": [], "cumulative_rewards": []}
        )
        for ep_data in all_episodes:
            for wm in ep_data.get("windowed_metrics", []):
                w_idx = wm["window_start"]
                windowed_data[w_idx]["rewards"].append(wm["mean_reward"])
                windowed_data[w_idx]["parse_rates"].append(wm["parse_rate"])
                windowed_data[w_idx]["cumulative_rewards"].append(wm["cumulative_reward"])

        if windowed_data:
            window_size = all_episodes[0].get("windowed_window_size", 100) if all_episodes else 100
            table_rows = []
            for w_start in sorted(windowed_data.keys()):
                wd = windowed_data[w_start]
                table_rows.append(
                    [
                        w_start + window_size // 2,  # midpoint of window
                        float(np.mean(wd["rewards"])),
                        float(np.mean(wd["parse_rates"])),
                        float(np.mean(wd["cumulative_rewards"])),
                        len(wd["rewards"]),
                    ]
                )

            icl_table = wandb.Table(
                columns=["step", "mean_reward", "parse_rate", "cumulative_reward", "num_episodes"],
                data=table_rows,
            )
            log_dict[f"{ep}/icl_reward_curve"] = wandb.plot.line(
                icl_table,
                "step",
                "mean_reward",
                title=f"ICL Reward Curve [{difficulty}] (window={window_size})",
            )
            log_dict[f"{ep}/icl_parse_rate_curve"] = wandb.plot.line(
                icl_table,
                "step",
                "parse_rate",
                title=f"ICL Parse Rate Curve [{difficulty}] (window={window_size})",
            )
            log_dict[f"{ep}/icl_cumulative_reward"] = wandb.plot.line(
                icl_table,
                "step",
                "cumulative_reward",
                title=f"ICL Cumulative Reward [{difficulty}]",
            )

        wandb.log(log_dict)

        _WANDB_PAPER_METRICS = [
            (f"{ep}/mean_episode_return", "Mean Episode Return", 1.0),
            (f"{ep}/mean_episode_length", "Mean Episode Length", 1.0),
            (f"{ep}/Team/reward_pct_of_max", "Team Reward % of Max", 100.0),
            (f"{ep}/Team/normal_reward_pct_of_max", "Team Normal Reward % of Max", 100.0),
            (f"{ep}/Team/coord_reward_pct_of_max", "Team Coord Reward % of Max", 100.0),
            (f"{ep}/Team/normal_achievement_pct", "Team Normal Achievement %", 100.0),
            (f"{ep}/Team/coordination_achievement_pct", "Team Coord Achievement %", 100.0),
            (f"{ep}/Team/achievement_pct", "Team Achievement %", 100.0),
            (f"{ep}/Progression/player_level", "Mean Player Level", 1.0),
        ]
        summary_rows = [
            ["Num Episodes", str(log_dict.get(f"{ep}/num_episodes", 0))],
            ["Total Valid Episodes", str(log_dict.get(f"{ep}/total_valid_episodes", 0))],
            ["Total Steps", str(log_dict.get(f"{ep}/total_steps", 0))],
        ]
        for log_key, label, scale in _WANDB_PAPER_METRICS:
            if log_key in log_dict:
                mean_val = log_dict[log_key] * scale
                se_key = log_key + "_se"
                if se_key in log_dict:
                    summary_rows.append([label, f"{mean_val:.3f} ± {log_dict[se_key] * scale:.3f}"])
                else:
                    summary_rows.append([label, f"{mean_val:.3f}"])

        summary_table = wandb.Table(columns=["Metric", "Value"], data=summary_rows)
        wandb.log({f"{ep}/summary": summary_table})

        print(f"\nW&B Logging Summary for {env_name} [{difficulty}]:")
        print(f"   Total Valid Episodes: {log_dict[f'{ep}/total_valid_episodes']}")
        print(f"   Total Steps: {log_dict[f'{ep}/total_steps']}")
        for log_key, label, scale in _WANDB_PAPER_METRICS:
            if log_key in log_dict:
                mean_val = log_dict[log_key] * scale
                se_key = log_key + "_se"
                if se_key in log_dict:
                    print(f"   {label}: {mean_val:.3f} ± {log_dict[se_key] * scale:.3f}")
                else:
                    print(f"   {label}: {mean_val:.3f}")

    # --- Upload all debug files as a wandb Artifact ---
    if output_dir:
        output_path = Path(output_dir)
        artifact = wandb.Artifact(
            name=f"eval-debug-{wandb.run.id}",
            type="eval_debug",
            description="Evaluation debug files: episode JSONs, CSVs, debug JSONL, GIFs, HTML viewers",
        )
        # Add all relevant file types
        file_extensions = [
            "*.json",
            "*.csv",
            "*.jsonl",
            "*.gif",
            "*.mp4",
            "*.html",
            "*.log",
            "*.yaml",
            "*.npz",
        ]
        files_added = 0
        for ext in file_extensions:
            for fpath in output_path.rglob(ext):
                # Use relative path within the artifact
                rel = fpath.relative_to(output_path)
                artifact.add_file(str(fpath), name=str(rel))
                files_added += 1

        if files_added > 0:
            wandb.log_artifact(artifact)
            print(
                f"\n   Uploaded {files_added} debug files as W&B artifact 'eval-debug-{wandb.run.id}'"
            )
