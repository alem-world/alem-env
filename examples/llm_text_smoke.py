"""Smoke-test the 3-agent Alem text interface without calling an LLM."""

import argparse
import os
import textwrap

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax.numpy as jnp

from alem.alem_coop.constants import Achievement
from alem.alem_coop.envs.common import compute_score


def _excerpt(text, width=100, lines=8):
    wrapped = textwrap.wrap(" ".join(text.split()), width=width)
    return "\n".join(wrapped[:lines])


def team_achievement_names(state):
    team_achievements = jnp.asarray(state.achievements).any(axis=0)
    return [
        achievement.name.lower()
        for achievement in Achievement
        if bool(team_achievements[achievement.value])
    ]


def final_score_metrics(state, static_env_params):
    score = compute_score(state, jnp.array(True), static_env_params)
    return {
        "Total%": jnp.mean(score["Team/reward_pct_of_max"]) * 100.0,
        "Base%": jnp.mean(score["Team/normal_reward_pct_of_max"]) * 100.0,
        "Coord.%": jnp.mean(score.get("Team/coord_reward_pct_of_max", jnp.array(0.0))) * 100.0,
    }


def make_log_state():
    return {
        "last_step": 0,
        "last_return": 0.0,
        "total_return": 0.0,
        "last_team_achievements": 0,
        "last_agent_achievements": 0,
    }


def log_rollout_step(log_state, step, team_reward, state, log_every, force=False):
    log_state["total_return"] += team_reward
    if not log_every:
        return
    if step != 1 and step % log_every != 0 and not force:
        return

    team_achievements = int(jnp.asarray(state.achievements).any(axis=0).sum())
    agent_achievements = int(jnp.asarray(state.achievements).sum())
    interval_return = log_state["total_return"] - log_state["last_return"]
    print(
        f"step={step:04d} "
        f"step_reward={team_reward:7.2f} "
        f"interval_return={interval_return:7.2f} "
        f"total_return={log_state['total_return']:7.2f} "
        f"team_achievements={team_achievements:02d} "
        f"new_achievements={team_achievements - log_state['last_team_achievements']:02d} "
        f"agent_achievements={agent_achievements:02d} "
        f"new_agent_achievements={agent_achievements - log_state['last_agent_achievements']:02d} "
        f"alive={[int(v) for v in state.player_alive]}"
    )
    log_state["last_step"] = step
    log_state["last_return"] = log_state["total_return"]
    log_state["last_team_achievements"] = team_achievements
    log_state["last_agent_achievements"] = agent_achievements


def main():
    parser = argparse.ArgumentParser(description="Preview Alem's 3-agent LLM text interface")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--coord", choices=["none", "easy", "medium", "hard"], default="easy")
    parser.add_argument(
        "--steps",
        type=int,
        default=1,
        help="Run this many no-model Noop steps after previewing observations",
    )
    parser.add_argument(
        "--ascii", action="store_true", help="Render the local view as an ASCII map"
    )
    parser.add_argument(
        "--show-affordances", action="store_true", help="Append legal actions to each observation"
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Print rollout metrics every N steps; use 0 to disable",
    )
    args = parser.parse_args()

    import jax

    from alem.llm.alem_language_wrapper import (
        ACTIONS,
        AlemLanguageWrapper,
        get_instruction_prompt,
        make_alem_env,
    )

    env = make_alem_env(
        {
            "ENV_NAME": "Alem-Coop-Symbolic",
            "num_agents": 3,
            "coordination_difficulty": args.coord,
            "soft_specialization": True,
            "shared_reward": False,
            "specialist_efficiency": 1.0,
            "non_specialist_efficiency": 0.2,
            "randomize_alpha": False,
            "max_timesteps": 10000,
            "god_mode": False,
        }
    )
    wrapper = AlemLanguageWrapper(
        env,
        env.default_params,
        prompt_mode="specific_collaborative",
        show_affordances=args.show_affordances,
        use_ascii=args.ascii,
        debug=False,
    )

    prompt = get_instruction_prompt(
        coordination_enabled=args.coord != "none",
        num_agents=3,
        agent_id=0,
        role="warrior",
        include_all_actions=False,
        progressive_disclosure=True,
        current_level=0,
        prompt_mode="specific_collaborative",
    )
    obs_list, state, rng = wrapper.reset(jax.random.PRNGKey(args.seed))

    print(f"Alem 3-agent text smoke | coord={args.coord} | actions={len(ACTIONS)}")
    print("\nSystem prompt excerpt:")
    print(_excerpt(prompt))

    for agent_idx, obs in enumerate(obs_list):
        text = obs["text"]
        print(f"\nAgent {agent_idx} long-term observation:")
        print(_excerpt(text["long_term_context"], lines=5))
        print(f"\nAgent {agent_idx} short-term observation:")
        print(_excerpt(text["short_term_context"], lines=5))

    print("\nAction parser examples:")
    for action in ["<action>Move North</action>", "ACTION: Do", "Give to Agent 2"]:
        print(f"{action!r} -> {wrapper.get_action_index(action, agent_idx=0)}")

    log_state = make_log_state()
    agent_returns = [0.0] * wrapper.num_agents
    completed_steps = 0

    if args.steps:
        print(f"\nNo-model rollout | policy=Noop | steps={args.steps}")

    for step in range(args.steps):
        actions = ["Noop"] * wrapper.num_agents
        obs_list, state, rewards, dones, info, rng = wrapper.step(state, actions, rng)
        completed_steps = step + 1
        shared_reward = bool(wrapper.env.default_params.shared_reward)
        team_reward = float(rewards[0] if shared_reward else sum(rewards))
        agent_returns = [total + float(reward) for total, reward in zip(agent_returns, rewards)]
        episode_done = all(bool(done) for done in dones)

        log_rollout_step(
            log_state,
            completed_steps,
            team_reward,
            state,
            args.log_every,
            force=episode_done or completed_steps == args.steps,
        )

        if episode_done:
            print("Episode ended.")
            break

    score_metrics = final_score_metrics(state, wrapper.static_env_params)

    print("\nFinal:")
    print(f"  steps: {completed_steps}")
    print(f"  team_return: {log_state['total_return']:.2f}")
    print(f"  agent_returns: {[round(v, 2) for v in agent_returns]}")
    print(f"  Total% (Team/reward_pct_of_max): {float(score_metrics['Total%']):.2f}")
    print(f"  Base% (Team/normal_reward_pct_of_max): {float(score_metrics['Base%']):.2f}")
    print(f"  Coord.% (Team/coord_reward_pct_of_max): {float(score_metrics['Coord.%']):.2f}")
    print(f"  team_achievements: {int(jnp.asarray(state.achievements).any(axis=0).sum())}")
    print(f"  agent_achievements: {[int(v) for v in jnp.asarray(state.achievements).sum(axis=1)]}")
    print(f"  team_achievement_names: {team_achievement_names(state)}")
    print(f"  alive: {[int(v) for v in state.player_alive]}")


if __name__ == "__main__":
    main()
