"""Run a tiny 3-agent Alem LLM smoke test with an OpenAI-compatible API."""

import argparse
import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax.numpy as jnp

from alem.alem_coop.constants import Achievement
from alem.alem_coop.envs.common import compute_score

ROLES = ("warrior", "forager", "miner")


def make_wrapper(coord):
    from alem.llm.alem_language_wrapper import AlemLanguageWrapper, make_alem_env

    env = make_alem_env(
        {
            "ENV_NAME": "Alem-Coop-Symbolic",
            "num_agents": 3,
            "coordination_difficulty": coord,
            "soft_specialization": True,
            "shared_reward": False,
            "specialist_efficiency": 1.0,
            "non_specialist_efficiency": 0.2,
            "randomize_alpha": False,
            "max_timesteps": 10000,
            "god_mode": False,
        }
    )
    return AlemLanguageWrapper(
        env,
        env.default_params,
        prompt_mode="specific_collaborative",
        show_affordances=True,
        debug=False,
    )


def build_messages(obs, coord, agent_idx, current_level):
    from alem.llm.alem_language_wrapper import get_instruction_prompt

    prompt = get_instruction_prompt(
        coordination_enabled=coord != "none",
        num_agents=3,
        agent_id=agent_idx,
        role=ROLES[agent_idx % len(ROLES)],
        include_all_actions=False,
        progressive_disclosure=True,
        current_level=current_level,
        prompt_mode="specific_collaborative",
    )
    text = obs["text"]
    observation = (
        f"{text['long_term_context']}\n\n"
        f"{text['short_term_context']}\n\n"
        "Choose exactly one available action. "
        "Return only: <action>ACTION_NAME</action>"
    )
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": observation},
    ]


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


def parse_args():
    parser = argparse.ArgumentParser(description="Run one tiny Alem 3-agent LLM smoke test")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument(
        "--base-url", default=None, help="OpenAI-compatible base URL, e.g. vLLM /v1"
    )
    parser.add_argument(
        "--api-key", default=None, help="Defaults to OPENAI_API_KEY, or EMPTY for --base-url"
    )
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--coord", choices=["none", "easy", "medium", "hard"], default="easy")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Print rollout metrics every N steps; use 0 to disable",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Install the OpenAI client first: pip install openai") from exc

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if args.base_url and not api_key:
        api_key = "EMPTY"
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY or pass --api-key.")

    import jax

    from alem.llm.action_parser import extract_action_multistrategy
    from alem.llm.alem_language_wrapper import ACTIONS

    client = OpenAI(api_key=api_key, base_url=args.base_url)
    wrapper = make_wrapper(args.coord)
    obs_list, state, rng = wrapper.reset(jax.random.PRNGKey(args.seed))
    log_state = make_log_state()
    agent_returns = [0.0] * wrapper.num_agents

    print(f"Alem 3-agent LLM smoke | model={args.model} | coord={args.coord} | steps={args.steps}")

    completed_steps = 0
    for step in range(args.steps):
        current_level = int(state.player_level)
        completions = []
        parsed_actions = []

        for agent_idx, obs in enumerate(obs_list):
            response = client.chat.completions.create(
                model=args.model,
                messages=build_messages(obs, args.coord, agent_idx, current_level),
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            completion = response.choices[0].message.content or ""
            parsed = extract_action_multistrategy(completion, ACTIONS) or "Noop"
            completions.append(completion)
            parsed_actions.append(parsed)

        obs_list, state, rewards, dones, info, rng = wrapper.step(state, completions, rng)
        completed_steps = step + 1
        shared_reward = bool(wrapper.env.default_params.shared_reward)
        team_reward = float(rewards[0] if shared_reward else sum(rewards))
        agent_returns = [total + float(reward) for total, reward in zip(agent_returns, rewards)]

        print(f"\nStep {step + 1}")
        for agent_idx, (raw, parsed, reward) in enumerate(
            zip(completions, parsed_actions, rewards)
        ):
            raw_one_line = " ".join(raw.split())
            print(
                f"  agent_{agent_idx}: {parsed} | reward={float(reward):+.3f} | raw={raw_one_line[:160]}"
            )

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
