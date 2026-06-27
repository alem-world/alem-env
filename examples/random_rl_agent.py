"""Run a masked-random multi-agent policy in the symbolic Alem environment.

This is a minimal RL-facing smoke test: it uses the public environment factory,
samples legal actions from the symbolic action masks, and runs the rollout as a
JAX-compiled ``lax.scan``. It intentionally does not depend on any training
framework.
"""

import argparse

import jax
import jax.numpy as jnp


def _stack_agent_dict(values, agents):
    return jnp.stack([values[agent] for agent in agents])


def _vector_to_agent_dict(values, agents):
    return {agent: values[i] for i, agent in enumerate(agents)}


def _sample_masked_actions(key, masks):
    masks = masks.astype(bool)
    noop_only = jnp.arange(masks.shape[-1]) == 0
    masks = jnp.where(masks.any(axis=-1, keepdims=True), masks, noop_only)
    logits = jnp.where(masks, 0.0, -1.0e9)
    keys = jax.random.split(key, masks.shape[0])
    return jax.vmap(lambda k, l: jax.random.categorical(k, l).astype(jnp.int32))(keys, logits)


def _sample_unmasked_actions(key, num_agents, num_actions):
    keys = jax.random.split(key, num_agents)
    return jax.vmap(
        lambda k: jax.random.randint(k, shape=(), minval=0, maxval=num_actions, dtype=jnp.int32)
    )(keys)


def _make_rollout(env, use_masks=True):
    agents = tuple(env.agents)
    num_agents = env.num_agents
    num_actions = env.action_space(agents[0]).n

    def sample_actions(key, state):
        if use_masks:
            masks = _stack_agent_dict(env.get_avail_actions(state), agents)
            action_vec = _sample_masked_actions(key, masks)
        else:
            action_vec = _sample_unmasked_actions(key, num_agents, num_actions)
        return _vector_to_agent_dict(action_vec, agents)

    def rollout(rng, state, steps):
        def scan_step(carry, _):
            rng, state, episode_done = carry
            rng, action_key, step_key = jax.random.split(rng, 3)
            actions = sample_actions(action_key, state)

            def step_active(_):
                _, next_state, rewards, dones, _ = env.step_env(step_key, state, actions)
                reward_vec = _stack_agent_dict(rewards, agents)
                return next_state, reward_vec, dones["__all__"]

            def step_done(_):
                return state, jnp.zeros((num_agents,), dtype=jnp.float32), jnp.array(True)

            next_state, reward_vec, step_done_flag = jax.lax.cond(
                episode_done,
                step_done,
                step_active,
                operand=None,
            )
            next_done = jnp.logical_or(episode_done, step_done_flag)

            metrics = {
                "reward": reward_vec.sum(),
                "done": next_done,
                "team_achievements": next_state.achievements.any(axis=0).sum(),
                "agent_achievements": next_state.achievements.sum(axis=1),
                "alive": next_state.player_alive,
            }
            return (rng, next_state, next_done), metrics

        return jax.lax.scan(scan_step, (rng, state, jnp.array(False)), None, length=steps)

    return jax.jit(rollout, static_argnames=("steps",))


def _team_achievements(state):
    return int(jnp.asarray(state.achievements).any(axis=0).sum())


def _agent_achievements(state):
    return [int(v) for v in jnp.asarray(state.achievements).sum(axis=1)]


def _completed_steps(done_flags, requested_steps):
    done_flags = jnp.asarray(done_flags)
    if not bool(done_flags.any()):
        return requested_steps
    return int(jnp.argmax(done_flags)) + 1


def main():
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
        "--full-info", action="store_true", help="Compute the full info metrics dict on each step"
    )
    parser.add_argument(
        "--log-every", type=int, default=25, help="Print progress every N steps; use 0 to disable"
    )
    args = parser.parse_args()

    from alem.alem_coop.alem_state import EnvParams, StaticEnvParams, get_coordination_params
    from alem.alem_env import make_alem_env_from_name

    env_overrides = {"max_timesteps": args.max_timesteps}
    if args.coord != "none":
        env_overrides.update(get_coordination_params(args.coord))
    env_params = EnvParams().replace(**env_overrides)
    static_env_params = StaticEnvParams(player_count=args.players)

    env = make_alem_env_from_name(
        "Alem-Coop-Symbolic",
        env_params=env_params,
        static_env_params=static_env_params,
        compute_full_info=args.full_info,
    )

    rng = jax.random.PRNGKey(args.seed)
    rng, reset_key = jax.random.split(rng)
    _, state = env.reset(reset_key)

    use_masks = not args.unmasked
    print(
        f"Alem random RL smoke | players={env.num_agents} | coord={args.coord} | "
        f"steps={args.steps} | masked={use_masks} | jitted_scan=True"
    )

    rollout = _make_rollout(env, use_masks=use_masks)
    (_, state, _), metrics = rollout(rng, state, steps=args.steps)

    completed_steps = _completed_steps(metrics["done"], args.steps)
    total_reward = float(metrics["reward"][:completed_steps].sum())

    if args.log_every:
        for step_idx in range(completed_steps):
            completed = step_idx + 1
            if completed == 1 or completed % args.log_every == 0 or completed == completed_steps:
                print(
                    f"step={completed:04d} "
                    f"reward={float(metrics['reward'][step_idx]):7.2f} "
                    f"team_achievements={int(metrics['team_achievements'][step_idx]):02d} "
                    f"alive={[int(v) for v in metrics['alive'][step_idx]]}"
                )

    print("\nFinal:")
    print(f"  steps: {completed_steps}")
    print(f"  total_reward: {total_reward:.2f}")
    print(f"  team_achievements: {_team_achievements(state)}")
    print(f"  agent_achievements: {_agent_achievements(state)}")
    print(f"  alive: {[int(v) for v in jnp.asarray(state.player_alive)]}")


if __name__ == "__main__":
    main()
