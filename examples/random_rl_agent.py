"""Run a random RL-style policy in the symbolic Alem environment."""

# ===========================
# Imports and Configuration
# ===========================
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp

from alem.alem_coop.alem_state import EnvParams, StaticEnvParams, get_coordination_params
from alem.alem_coop.constants import Achievement
from alem.alem_coop.envs.common import compute_score
from alem.alem_env import make_alem_env_from_name


# ===========================
# Data Structures and Utilities
# ===========================
def stack_agents(x: dict, agent_list):
    return jnp.stack([x[a] for a in agent_list])


def unstack_actions(actions: jnp.ndarray, agent_list):
    return {agent: actions[i] for i, agent in enumerate(agent_list)}


def sample_random_actions(rng, avail_actions, action_dim, action_masking):
    action_rngs = jax.random.split(rng, avail_actions.shape[0])

    if action_masking:
        avail_actions = avail_actions.astype(bool)
        noop_action = jnp.arange(action_dim) == 0
        avail_actions = jnp.where(
            avail_actions.any(axis=-1, keepdims=True),
            avail_actions,
            noop_action,
        )
        logits = jnp.where(avail_actions, 0.0, -1.0e10)
        return jax.vmap(
            lambda action_rng, action_logits: jax.random.categorical(
                action_rng, action_logits
            ).astype(jnp.int32)
        )(action_rngs, logits)

    return jax.vmap(
        lambda action_rng: jax.random.randint(
            action_rng, shape=(), minval=0, maxval=action_dim, dtype=jnp.int32
        )
    )(action_rngs)


def completed_steps(done_flags, num_steps):
    done_flags = jnp.asarray(done_flags)
    if not bool(done_flags.any()):
        return num_steps
    return int(jnp.argmax(done_flags)) + 1


def log_rollout(metrics, num_steps, log_every):
    if not log_every:
        return

    previous_logged_step = 0
    previous_logged_achievements = 0
    previous_logged_agent_achievements = 0
    for step_idx in range(num_steps):
        completed = step_idx + 1
        if completed == 1 or completed % log_every == 0 or completed == num_steps:
            team_achievements = int(metrics["team_achievements"][step_idx])
            agent_achievements = int(metrics["agent_achievement_total"][step_idx])
            print(
                f"step={completed:04d} "
                f"step_reward={float(metrics['team_reward'][step_idx]):7.2f} "
                f"interval_return={float(metrics['team_reward'][previous_logged_step:completed].sum()):7.2f} "
                f"total_return={float(metrics['team_reward'][:completed].sum()):7.2f} "
                f"team_achievements={team_achievements:02d} "
                f"new_achievements={team_achievements - previous_logged_achievements:02d} "
                f"agent_achievements={agent_achievements:02d} "
                f"new_agent_achievements={agent_achievements - previous_logged_agent_achievements:02d} "
                f"alive={[int(v) for v in metrics['alive'][step_idx]]}"
            )
            previous_logged_step = completed
            previous_logged_achievements = team_achievements
            previous_logged_agent_achievements = agent_achievements


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


# ===========================
# Training Function
# ===========================
def make_train(config, env):
    agent_list = tuple(env.agents)
    action_dim = env.action_space(agent_list[0]).n

    def train(rng):
        rng, reset_rng = jax.random.split(rng)
        _, env_state = env.reset(reset_rng)

        def _env_step(runner_state, unused):
            env_state, episode_done, rng = runner_state
            rng, action_rng, step_rng = jax.random.split(rng, 3)

            avail_actions = stack_agents(env.get_avail_actions(env_state), agent_list)
            actions = unstack_actions(
                sample_random_actions(
                    action_rng,
                    avail_actions,
                    action_dim,
                    config["ACTION_MASKING"],
                ),
                agent_list,
            )

            def step_active(_):
                _, next_state, reward, done, _ = env.step_env(step_rng, env_state, actions)
                return next_state, stack_agents(reward, agent_list), done["__all__"]

            def step_finished(_):
                return env_state, jnp.zeros((env.num_agents,), dtype=jnp.float32), jnp.array(True)

            next_state, reward, step_done = jax.lax.cond(
                episode_done,
                step_finished,
                step_active,
                operand=None,
            )
            next_done = jnp.logical_or(episode_done, step_done)

            metrics = {
                "team_reward": reward[0] if config["SHARED_REWARD"] else reward.sum(),
                "agent_reward": reward,
                "done": next_done,
                "team_achievements": next_state.achievements.any(axis=0).sum(),
                "agent_achievement_total": next_state.achievements.sum(),
                "alive": next_state.player_alive,
            }
            return (next_state, next_done, rng), metrics

        runner_state, metrics = jax.lax.scan(
            _env_step,
            (env_state, jnp.array(False), rng),
            None,
            config["NUM_STEPS"],
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train


# ===========================
# Main Run Function
# ===========================
def build_env(config):
    env_params = EnvParams().replace(
        max_timesteps=config["MAX_TIMESTEPS"],
        shared_reward=config["SHARED_REWARD"],
        **get_coordination_params(config["TRAINING_COORDINATION_DIFFICULTY"]),
    )
    static_env_params = StaticEnvParams(player_count=config["NUM_AGENTS"])

    return make_alem_env_from_name(
        config["ENV_NAME"],
        env_params=env_params,
        static_env_params=static_env_params,
        compute_full_info=config["COMPUTE_FULL_INFO"],
    )


def single_run(config):
    env = build_env(config)
    rng = jax.random.PRNGKey(config["SEED"])

    print(
        f"Alem random RL smoke | players={env.num_agents} | "
        f"coord={config['TRAINING_COORDINATION_DIFFICULTY']} | "
        f"steps={config['NUM_STEPS']} | masked={config['ACTION_MASKING']} | "
        f"shared_reward={config['SHARED_REWARD']} | "
        "jitted_scan=True"
    )

    train = jax.jit(make_train(config, env))
    outs = jax.block_until_ready(train(rng))

    final_state = outs["runner_state"][0]
    metrics = outs["metrics"]
    num_steps = completed_steps(metrics["done"], config["NUM_STEPS"])
    team_return = float(metrics["team_reward"][:num_steps].sum())
    agent_returns = [float(v) for v in metrics["agent_reward"][:num_steps].sum(axis=0)]
    score_metrics = final_score_metrics(final_state, env.static_env_params)

    log_rollout(metrics, num_steps, config["LOG_EVERY"])

    print("\nFinal:")
    print(f"  steps: {num_steps}")
    print(f"  team_return: {team_return:.2f}")
    print(f"  agent_returns: {[round(v, 2) for v in agent_returns]}")
    print(f"  Total% (Team/reward_pct_of_max): {float(score_metrics['Total%']):.2f}")
    print(f"  Base% (Team/normal_reward_pct_of_max): {float(score_metrics['Base%']):.2f}")
    print(f"  Coord.% (Team/coord_reward_pct_of_max): {float(score_metrics['Coord.%']):.2f}")
    print(f"  team_achievements: {int(final_state.achievements.any(axis=0).sum())}")
    print(f"  agent_achievements: {[int(v) for v in final_state.achievements.sum(axis=1)]}")
    print(f"  team_achievement_names: {team_achievement_names(final_state)}")
    print(f"  alive: {[int(v) for v in final_state.player_alive]}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run a random RL-style policy in Alem")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--players", type=int, default=3)
    parser.add_argument("--coord", choices=["none", "easy", "medium", "hard"], default="easy")
    parser.add_argument("--max-timesteps", type=int, default=10000)
    parser.add_argument(
        "--unmasked",
        action="store_true",
        help="Sample from the full action space instead of legal action masks",
    )
    parser.add_argument(
        "--shared-reward",
        action="store_true",
        help="Use the environment's shared team reward instead of individual rewards",
    )
    parser.add_argument(
        "--full-info",
        action="store_true",
        help="Compute the full info metrics dict on each step",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="Print progress every N steps; use 0 to disable",
    )
    args = parser.parse_args()

    return {
        "ALG_NAME": "random",
        "ENV_NAME": "Alem-Coop-Symbolic",
        "SEED": args.seed,
        "NUM_STEPS": args.steps,
        "NUM_AGENTS": args.players,
        "TRAINING_COORDINATION_DIFFICULTY": args.coord,
        "MAX_TIMESTEPS": args.max_timesteps,
        "ACTION_MASKING": not args.unmasked,
        "SHARED_REWARD": args.shared_reward,
        "COMPUTE_FULL_INFO": args.full_info,
        "LOG_EVERY": args.log_every,
    }


def main():
    single_run(parse_args())


if __name__ == "__main__":
    main()
