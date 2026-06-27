import os
from functools import partial
from types import SimpleNamespace

import imageio
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import wandb
from jaxmarl.wrappers.baselines import JaxMARLWrapper


def save_checkpoint(train_state, config):
    """Save train_state with orbax and upload to wandb as artifact."""
    ckpt_dir = os.path.join(wandb.run.dir, "checkpoint")
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(ckpt_dir, train_state)

    artifact_name = f"{wandb.run.name}-checkpoint".replace(" ", "")
    artifact = wandb.Artifact(
        name=artifact_name,
        type="checkpoint",
    )
    artifact.add_dir(ckpt_dir)
    wandb.log_artifact(artifact)
    print(f"Checkpoint saved and uploaded to wandb: {wandb.run.name}-checkpoint")


def load_checkpoint(train_state, checkpoint_path=None, wandb_artifact=None):
    """Restore train_state from a local path or wandb artifact.

    Args:
        train_state: Template train_state (provides structure, apply_fn, tx).
        checkpoint_path: Local directory containing the orbax checkpoint.
        wandb_artifact: Wandb artifact reference, e.g. "entity/project/name:v0".

    Returns:
        Restored train_state with params/opt_state from checkpoint.
    """
    if wandb_artifact is not None:
        artifact = wandb.use_artifact(wandb_artifact)
        checkpoint_path = artifact.download()
    if checkpoint_path is None:
        raise ValueError("Provide checkpoint_path or wandb_artifact")

    checkpointer = ocp.PyTreeCheckpointer()
    return checkpointer.restore(checkpoint_path, item=train_state)


def _resolve_checkpoint_dir(path):
    """Resolve a user-supplied path to the actual orbax checkpoint directory.

    Accepts either the orbax directory itself, or a parent directory (e.g. the run
    folder from the wandb export / HuggingFace download, which contains
    ``artifacts/<run-name>-checkpoint_v<N>/``). When several checkpoint versions are
    present, the highest version is used.
    """
    import glob
    import re

    path = os.path.realpath(path)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")

    # Already an orbax checkpoint directory?
    if os.path.exists(os.path.join(path, "_CHECKPOINT_METADATA")) or os.path.exists(
        os.path.join(path, "_METADATA")
    ):
        return path

    # Otherwise search for a saved checkpoint dir underneath it.
    candidates = [
        c for c in glob.glob(os.path.join(path, "**", "*checkpoint*"), recursive=True)
        if os.path.isdir(c)
        and (
            os.path.exists(os.path.join(c, "_CHECKPOINT_METADATA"))
            or os.path.exists(os.path.join(c, "_METADATA"))
        )
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No orbax checkpoint found in {path}. Point LOAD_CHECKPOINT at the "
            "checkpoint directory (or a run folder containing artifacts/*-checkpoint_v*)."
        )

    def _version(c):
        m = re.search(r"_v(\d+)$", os.path.basename(c))
        return int(m.group(1)) if m else -1

    return sorted(candidates, key=_version)[-1]


def restore_baseline_checkpoint(checkpoint_path):
    """Restore a saved baseline checkpoint to a host pytree, no template required.

    Works for any baseline checkpoint regardless of how it was sharded at save time
    (restores every leaf to a host ``jnp`` array). The returned pytree matches what
    the trainer saved:

      - IPPO / IPPO-NOPS / HyperMARL: ``{"params": ..., "opt_state": ..., "step": ...}``
      - MAPPO:                        ``{"actor": {...}, "critic": {...}}``
      - PQN-VDN:                      ``{"params": ..., "batch_stats": ..., ...}``

    Args:
        checkpoint_path: Orbax checkpoint dir, or a run folder containing one (e.g. a
            HuggingFace download of the released policies).

    Returns:
        The restored pytree with ``jnp`` array leaves.
    """
    path = _resolve_checkpoint_dir(checkpoint_path)
    checkpointer = ocp.PyTreeCheckpointer()
    meta = checkpointer.metadata(path)
    restore_args = jax.tree.map(lambda _: ocp.RestoreArgs(restore_type=np.ndarray), meta)
    restored = checkpointer.restore(path, restore_args=restore_args)
    print(f"Restored checkpoint from {path}")
    return jax.tree.map(jnp.asarray, restored)


class AppendAgentIDWrapper(JaxMARLWrapper):
    def __init__(self, env, obs_with_agent_id=True):
        super().__init__(env)
        self.obs_with_agent_id = obs_with_agent_id
        self._obs_dim = self._env.observation_space(self._env.agents[0]).shape[0]
        if self.obs_with_agent_id:
            self._obs_dim = self._obs_dim + self._env.num_agents
        self.observation_spaces = {
            agent: SimpleNamespace(shape=(self._obs_dim,)) for agent in self._env.agents
        }

    @partial(jax.jit, static_argnums=0)
    def reset(self, key):
        obs, env_state = self._env.reset(key)
        obs = self._append_ids(obs)
        return obs, env_state

    @partial(jax.jit, static_argnums=0)
    def step(self, key, state, action):
        obs, env_state, reward, done, info = self._env.step(key, state, action)
        obs = self._append_ids(obs)
        return obs, env_state, reward, done, info

    @partial(jax.jit, static_argnums=0)
    def step_env(self, key, state, action):
        obs, env_state, reward, done, info = self._env.step_env(key, state, action)
        obs = self._append_ids(obs)
        return obs, env_state, reward, done, info

    def _append_ids(self, obs):
        if self.obs_with_agent_id:
            agent_ids = jnp.eye(self._env.num_agents)
            for i, agent_key in enumerate(self._env.agents):
                obs[agent_key] = jnp.concatenate([obs[agent_key], agent_ids[i]])
        return obs

    def observation_space(self, agent):
        return self.observation_spaces[agent]


def _flatten_user_info_prefix(prefix, tree, out):
    """Recursively flatten tree with prefix, writing scalars into out."""
    if isinstance(tree, dict):
        for k, v in tree.items():
            _flatten_user_info_prefix(f"{prefix}/{k}" if prefix else k, v, out)
    else:
        val = float(jnp.array(tree).mean()) if hasattr(tree, "__len__") else float(tree)
        if val == val and abs(val) != float("inf"):
            out[prefix] = val


def _run_eval_sequential(
    env, policy_fn, init_hstate, rng, num_episodes, max_steps, config=None, collect_data=False
):
    """Run N sequential evaluation episodes — mirrors LLM evaluator.run_episode() structure.

    Each episode:
      1. reset env with a unique RNG seed
      2. step until terminal or max_steps (Python while loop, JIT-compiled step)
      3. compute_score(state, done=True) at episode end

    Returns:
        aggregated: dict of mean metrics across episodes
        per_episode: list of per-episode metric dicts
        trajectories: list of per-episode trajectory dicts (only when collect_data=True, else None)
    """
    from alem.alem_coop.envs.common import compute_score

    action_masking = config is not None and config.get("ACTION_MASKING", False)
    if action_masking:
        from alem.alem_coop.action_masking import compute_action_mask

        _inner = getattr(env, "_env", env)
        _mask_fn = getattr(_inner, "action_mask_fn", compute_action_mask)

    # JIT-compile one step: policy inference + env step_env (no auto-reset)
    @jax.jit
    def _step(state, obs, hstate, done_batch, rng):
        obs_batch = jnp.stack([obs[a] for a in env.agents])  # (num_agents, obs_dim)
        rng, act_rng, step_rng = jax.random.split(rng, 3)
        if action_masking:
            avail_actions = _mask_fn(state, env.default_params, env.static_env_params)
        else:
            avail_actions = jnp.ones(
                (env.num_agents, env.action_space(env.agents[0]).n), dtype=jnp.bool_
            )
        new_hstate, actions = policy_fn(hstate, obs_batch, done_batch, act_rng, avail_actions)
        env_actions = {a: actions[i] for i, a in enumerate(env.agents)}
        new_obs, new_state, reward_dict, done_dict, _ = env.step_env(step_rng, state, env_actions)
        per_agent_rewards = jnp.stack(list(reward_dict.values()))  # (num_agents,)
        reward_mean = per_agent_rewards.mean()
        done = done_dict["__all__"]
        new_done_batch = jnp.stack([done_dict[a] for a in env.agents])
        return (
            new_state,
            new_obs,
            new_hstate,
            new_done_batch,
            done,
            reward_mean,
            rng,
            obs_batch,
            actions,
            per_agent_rewards,
        )

    eval_seed = config.get("EVAL_SEED", 9999) if config else 9999

    eval_seed = config.get("EVAL_SEED", 9999) if config else 9999

    per_episode = []
    trajectories = [] if collect_data else None

    for ep_idx in range(num_episodes):
        reset_rng = jax.random.PRNGKey(eval_seed + ep_idx)
        obs, state = env.reset(reset_rng)
        hstate = init_hstate
        done_batch = jnp.zeros(env.num_agents, dtype=bool)

        episode_return = 0.0
        if collect_data:
            ep_obs, ep_actions, ep_rewards, ep_dones = [], [], [], []

        for step in range(max_steps):
            result = _step(state, obs, hstate, done_batch, rng)
            state, obs, hstate, done_batch, done_jax, reward_mean, rng = result[:7]
            step_obs, step_actions, step_rewards = result[7], result[8], result[9]

            episode_return += float(reward_mean)
            if collect_data:
                ep_obs.append(np.array(step_obs))
                ep_actions.append(np.array(step_actions))
                ep_rewards.append(np.array(step_rewards))
                ep_dones.append(bool(done_jax))
            if bool(done_jax):
                break

        if collect_data:
            trajectories.append(
                {
                    "obs": np.stack(ep_obs),  # (T, num_agents, obs_dim)
                    "actions": np.stack(ep_actions),  # (T, num_agents)
                    "rewards": np.stack(ep_rewards),  # (T, num_agents)
                    "dones": np.array(ep_dones),  # (T,)
                }
            )

        # Compute score at episode end with done=True (mirrors LLM eval alem_env.py)
        score = compute_score(state, jnp.array(True), env.static_env_params)

        done = bool(done_jax)
        ep_metrics = {
            "mean_episode_return": episode_return,
            "mean_episode_length": step,
            "done": float(done),
            "seed": ep_idx,
        }
        _flatten_user_info_prefix("", score, ep_metrics)
        per_episode.append(ep_metrics)

    # Aggregate: mean + SE over episodes (mirrors collect_and_summarize_results in LLM eval)
    all_keys = set(k for ep in per_episode for k in ep)
    aggregated = {}
    for k in all_keys:
        if k == "seed":
            continue
        vals = np.array([ep[k] for ep in per_episode if k in ep], dtype=float)
        aggregated[k] = float(vals.mean())
        if len(vals) > 1:
            aggregated[k + "_se"] = float(vals.std() / np.sqrt(len(vals)))
    aggregated["num_episodes"] = num_episodes
    aggregated["num_completed_episodes"] = num_episodes  # backward compat
    aggregated["success_rate"] = float(np.mean([ep["done"] for ep in per_episode]))

    return aggregated, per_episode, trajectories


# TODO: explore adding parallel eval for LLM agents (currently LLM eval is sequential)
def _run_eval_parallel(env, policy_fn, init_hstate, rng, num_envs, num_steps, config=None):
    """Run parallel evaluation using vmap + lax.scan. Faster than sequential but
    reports mean over however many episodes complete in num_steps (not exactly N episodes).
    """
    from jaxmarl.wrappers.baselines import LogWrapper

    action_masking = config is not None and config.get("ACTION_MASKING", False)
    if action_masking:
        from alem.alem_coop.action_masking import compute_action_mask

        _inner = getattr(env, "_env", env)
        _mask_fn = getattr(_inner, "action_mask_fn", compute_action_mask)

    num_agents = env.num_agents
    wrapped_env = LogWrapper(env)
    v_reset = jax.vmap(wrapped_env.reset, in_axes=(0,))
    v_step = jax.vmap(wrapped_env.step, in_axes=(0, 0, 0))

    @jax.jit
    def _eval_loop(rng):
        rng, reset_rng = jax.random.split(rng)
        reset_rngs = jax.random.split(reset_rng, num_envs)
        obs, env_states = v_reset(reset_rngs)
        hstates = jnp.tile(init_hstate[None, :, :], (num_envs, 1, 1))
        dones = jnp.zeros((num_envs, num_agents), dtype=bool)

        def _eval_step(carry, _):
            env_states, obs, hstates, dones, rng = carry
            obs_batch = jnp.stack([obs[a] for a in env.agents], axis=1)
            rng, act_rng = jax.random.split(rng)
            act_rngs = jax.random.split(act_rng, num_envs)
            if action_masking:
                avail_actions = jax.vmap(
                    lambda s: _mask_fn(s, env.default_params, env.static_env_params)
                )(env_states.env_state)  # (num_envs, num_agents, num_actions)
            else:
                avail_actions = jnp.ones(
                    (num_envs, num_agents, env.action_space(env.agents[0]).n), dtype=jnp.bool_
                )
            new_hstates, actions = jax.vmap(policy_fn)(
                hstates, obs_batch, dones, act_rngs, avail_actions
            )
            env_actions = {a: actions[:, i] for i, a in enumerate(env.agents)}
            rng, step_rng = jax.random.split(rng)
            step_rngs = jax.random.split(step_rng, num_envs)
            new_obs, new_env_states, rewards, done_dict, info = v_step(
                step_rngs, env_states, env_actions
            )
            ep_done = done_dict["__all__"]
            new_dones = jnp.stack([done_dict[a] for a in env.agents], axis=1)
            new_hstates = jnp.where(
                ep_done[:, None, None],
                jnp.tile(init_hstate[None, :, :], (num_envs, 1, 1)),
                new_hstates,
            )
            carry = (new_env_states, new_obs, new_hstates, new_dones, rng)
            return carry, info

        carry = (env_states, obs, hstates, dones, rng)
        _, infos = jax.lax.scan(_eval_step, carry, None, length=num_steps)
        return infos

    infos = _eval_loop(rng)
    returned_episode = infos["returned_episode"]
    metrics = jax.tree.map(lambda x: jnp.nanmean(jnp.where(returned_episode, x, jnp.nan)), infos)
    num_completed = int(returned_episode.sum()) // num_agents

    results = {"num_completed_episodes": num_completed}
    results["mean_episode_return"] = float(metrics.get("returned_episode_returns", 0.0))
    results["mean_episode_length"] = float(metrics.get("returned_episode_lengths", 0.0))
    if "user_info" in metrics:
        _flatten_user_info_prefix("", metrics["user_info"], results)
    return results


def _make_eval_env(config, env_name, difficulty, static_env_params=None):
    """Create a fixed-difficulty eval env from config."""
    from alem.alem_coop.alem_state import EnvParams, get_coordination_params
    from alem.alem_env import make_alem_env_from_name

    scale_base = config.get("SCALE_BASE_DIFFICULTY", False)
    coord_kwargs = get_coordination_params(difficulty, scale_base=scale_base)
    eval_env_params = EnvParams(
        shared_reward=config.get("SHARED_REWARD", True),
        randomize_alpha=False,  # Always fixed for eval
        # Coordination params (includes soft_specialization + non_specialist_efficiency)
        **coord_kwargs,
    )

    env = make_alem_env_from_name(
        env_name, env_params=eval_env_params, static_env_params=static_env_params
    )
    if config.get("APPEND_AGENT_ID", False):
        env = AppendAgentIDWrapper(env, obs_with_agent_id=True)
    return env


def _save_eval_trajectories(trajectories, difficulty):
    """Save evaluation trajectories as compressed npz and upload as wandb artifact."""
    # Concatenate all episodes along time axis, with episode_ids and timesteps for reconstruction
    all_obs = np.concatenate([t["obs"] for t in trajectories], axis=0)
    all_actions = np.concatenate([t["actions"] for t in trajectories], axis=0)
    all_rewards = np.concatenate([t["rewards"] for t in trajectories], axis=0)
    all_dones = np.concatenate([t["dones"] for t in trajectories], axis=0)
    episode_ids = np.concatenate(
        [np.full(t["obs"].shape[0], i) for i, t in enumerate(trajectories)]
    )
    timesteps = np.concatenate([np.arange(t["obs"].shape[0]) for t in trajectories])

    save_path = os.path.join(wandb.run.dir, f"eval_trajectories_{difficulty}.npz")
    np.savez_compressed(
        save_path,
        obs=all_obs,
        actions=all_actions,
        rewards=all_rewards,
        dones=all_dones,
        episode_ids=episode_ids,
        timesteps=timesteps,
    )

    artifact_name = f"{wandb.run.name}-eval-{difficulty}".replace(" ", "")
    artifact = wandb.Artifact(name=artifact_name, type="eval_trajectories")
    artifact.add_file(save_path)
    wandb.log_artifact(artifact)
    print(
        f"  Saved eval trajectories ({difficulty}): {all_obs.shape[0]} steps across {len(trajectories)} episodes"
    )


def run_final_eval(config, env, policy_fn, init_hstate, rng):
    """Run sequential final evaluation on easy/medium/hard — matches LLM eval setup.

    Runs exactly TEST_NUM_EPISODES episodes per difficulty (default 10), one at a time,
    using a Python loop + JIT-compiled step. Mirrors the LLM evaluator's episode loop.

    Config keys:
        TEST_NUM_EPISODES: episodes per difficulty (default 10, matches LLM eval)
        TEST_MAX_STEPS: max steps per episode (default 10000, matches LLM eval)
        EVAL_DIFFICULTIES: list of difficulties (default ["easy", "medium", "hard"])
        COLLECT_EVAL_DATA: whether to save trajectory data (default True)
        ENV_NAME: env name string

    Returns:
        dict mapping difficulty -> aggregated metrics
    """
    num_episodes = config.get("TEST_NUM_EPISODES", 10)
    max_steps = config.get("TEST_MAX_STEPS", 10000)
    eval_difficulties = config.get("EVAL_DIFFICULTIES", ["easy", "medium", "hard"])
    env_name = config.get("ENV_NAME", "Alem-Coop-Symbolic")
    tags = config.get("RUN_TAGS", [])
    collect_data = "final" in tags

    # Re-use the same static_env_params as training so obs/action shapes match
    static_env_params = getattr(env, "static_env_params", None)

    all_results = {}

    for difficulty in eval_difficulties:
        print(f"Running eval [{difficulty}]: {num_episodes} episodes, max_steps={max_steps}...")
        rng, eval_rng = jax.random.split(rng)

        eval_env = _make_eval_env(config, env_name, difficulty, static_env_params=static_env_params)
        aggregated, per_episode, trajectories = _run_eval_sequential(
            eval_env,
            policy_fn,
            init_hstate,
            eval_rng,
            num_episodes,
            max_steps,
            config=config,
            collect_data=collect_data,
        )

        wandb.log({f"eval/{difficulty}/{k}": v for k, v in aggregated.items()})

        # Log raw per-episode metrics as individual scalars — retrieve via run.history().
        # Same key schema as LLM eval so both pipelines can be pulled and compared identically.
        _SKIP = {
            "mean_episode_return",
            "mean_episode_length",
            "done",
            "seed",
            "num_episodes",
            "num_completed_episodes",
            "success_rate",
        }
        score_keys = sorted({k for ep in per_episode for k in ep if k not in _SKIP})
        for ep in per_episode:
            wandb.log(
                {
                    f"eval_ep/{difficulty}/episode_return": ep.get("mean_episode_return", 0.0),
                    f"eval_ep/{difficulty}/episode_length": ep.get("mean_episode_length", 0.0),
                    f"eval_ep/{difficulty}/done": ep.get("done", 0.0),
                    f"eval_ep/{difficulty}/seed": ep.get("seed", 0),
                    **{f"eval_ep/{difficulty}/{k}": ep[k] for k in score_keys if k in ep},
                }
            )

        ep_ret = aggregated.get("mean_episode_return", 0.0)
        ep_len = aggregated.get("mean_episode_length", 0.0)
        print(
            f"  {difficulty}: mean_return={ep_ret:.2f}, mean_length={ep_len:.1f}, "
            f"episodes={aggregated['num_completed_episodes']}"
        )

        if trajectories is not None:
            _save_eval_trajectories(trajectories, difficulty)

        all_results[difficulty] = aggregated

    return all_results


# We can def make this faster later, but for now just do a simple Python loop
def run_visualization_ippo_rnn(
    config, env, trained_params, rng, network, scanned_rnn_class, nonshared_params=False
):
    """Run visualization rollouts and save GIF for IPPO (single ActorCritic network)."""
    import distrax

    env_name = config.get("ENV_NAME", "Alem-Coop-Symbolic")
    action_masking = config.get("ACTION_MASKING", False)
    if action_masking:
        from alem.alem_coop.action_masking import compute_action_mask

        _inner = getattr(env, "_env", env)
        _mask_fn = getattr(_inner, "action_mask_fn", compute_action_mask)

    from alem.alem_coop.constants import (
        BLOCK_PIXEL_SIZE_HUMAN,
        TEXTURES,
        load_player_specific_textures,
    )
    from alem.alem_coop.renderer.renderer_pixels import render_alem_pixels

    # Load textures for rendering
    pixel_size = BLOCK_PIXEL_SIZE_HUMAN
    player_textures = load_player_specific_textures(TEXTURES[pixel_size], env.num_agents)

    # JIT compile the step function (without rendering)
    @jax.jit
    def policy_step(obs, hstate, rng, env_state):
        obs_batch = jnp.stack([obs[a] for a in env.agents])
        done_batch = jnp.zeros(env.num_agents, dtype=bool)
        if nonshared_params:
            # Per-agent params: run one single-env forward pass per agent
            ac_obs = obs_batch[:, None, None, :]  # (num_agents, 1, 1, obs_dim)
            ac_done = done_batch[:, None, None]  # (num_agents, 1, 1)
            hstate, pi, _ = jax.vmap(
                lambda p, h, o, d: network.apply(p, h, (o, d)),
                in_axes=(0, 0, 0, 0),
            )(trained_params, hstate, ac_obs, ac_done)
        else:
            ac_in = (obs_batch[np.newaxis, :], done_batch[np.newaxis, :])
            hstate, pi, _ = network.apply(trained_params, hstate, ac_in)
        if action_masking:
            mask = _mask_fn(env_state, env.default_params, env.static_env_params)
            if nonshared_params:
                masked_logits = pi.logits + jnp.where(mask[:, None, None, :], 0.0, -1e10)
            else:
                masked_logits = pi.logits + jnp.where(mask[None, :, :], 0.0, -1e10)
            pi = distrax.Categorical(logits=masked_logits)
        if nonshared_params:
            action = pi.sample(seed=rng)[:, 0, 0]
            env_act = {a: action[i] for i, a in enumerate(env.agents)}
        else:
            action = pi.sample(seed=rng)
            env_act = {a: action[0, i] for i, a in enumerate(env.agents)}
        return env_act, hstate

    # Render function (called outside JIT)
    def render_frame(env_state, downscale=2):
        pixels = render_alem_pixels(env_state, pixel_size, env.static_env_params, player_textures)
        pixels = np.array(pixels)
        # pixels shape: (num_agents, H, W, 3)
        # pixels[0] is agent 0's view, etc.
        if downscale > 1:
            pixels = pixels[:, ::downscale, ::downscale, :]  # (num_agents, H, W, 3)
        return pixels

    # Run rollout with Python loop (rendering doesn't need to be fast)
    rng, reset_rng = jax.random.split(rng)
    obs, env_state = env.reset(reset_rng)
    if nonshared_params:
        single_h = scanned_rnn_class.initialize_carry(1, config["GRU_HIDDEN_DIM"])
        hstate = jnp.tile(single_h[None, :, :], (env.num_agents, 1, 1))
    else:
        hstate = scanned_rnn_class.initialize_carry(env.num_agents, config["GRU_HIDDEN_DIM"])

    # Default to 1000 steps for visualization (env max is 10k which would be too large for GIF)
    max_steps = config.get("VIS_STEPS", 1000)
    frames = []

    for _ in range(max_steps):
        # Render current state
        # We could collect all env state and then render properly in parallel, but I had memory issues doing this (locally on laptop)
        frame = render_frame(env_state)
        frames.append(frame)

        # Get action from policy
        rng, act_rng, step_rng = jax.random.split(rng, 3)
        env_act, hstate = policy_step(obs, hstate, act_rng, env_state)

        # Step environment
        obs, env_state, _, dones, _ = env.step(step_rng, env_state, env_act)

        if dones["__all__"]:
            break

    # Convert frames to numpy and create GIF
    _save_visualization_gifs(frames, config)


def run_visualization_pqn_vdn_rnn(config, env, trained_state, rng, network, init_hstate):
    """Run visualization rollouts and save GIF for PQN-VDN (QNetwork with LSTM)."""
    from jaxmarl.wrappers.baselines import CTRolloutManager, LogWrapper

    env_name = config.get("ENV_NAME", "Alem-Coop-Symbolic")
    action_masking = config.get("ACTION_MASKING", True)

    from alem.alem_coop.constants import (
        BLOCK_PIXEL_SIZE_HUMAN,
        TEXTURES,
        load_player_specific_textures,
    )
    from alem.alem_coop.renderer.renderer_pixels import render_alem_pixels

    pixel_size = BLOCK_PIXEL_SIZE_HUMAN
    player_textures = load_player_specific_textures(TEXTURES[pixel_size], env.num_agents)

    # Use CTRolloutManager(preprocess_obs=True) so obs are preprocessed identically to training
    wrapped_env = CTRolloutManager(LogWrapper(env), batch_size=1, preprocess_obs=True)
    trained_params = trained_state.params
    trained_batch_stats = trained_state.batch_stats

    @jax.jit
    def policy_step(obs, hstate, rng, env_state):
        # obs[a] has shape (batch_size=1, obs_dim) from CTRolloutManager — keep batch dim
        obs_batch = jnp.stack([obs[a] for a in env.agents])  # (num_agents, 1, obs_dim)
        done_batch = jnp.zeros((env.num_agents, 1), dtype=bool)
        _obs = obs_batch[:, np.newaxis]  # (num_agents, 1, 1, obs_dim) — (agents, time, batch, obs)
        _dones = done_batch[:, np.newaxis]  # (num_agents, 1, 1)
        new_hs, q_vals = jax.vmap(network.apply, in_axes=(None, 0, 0, 0, None))(
            {"params": trained_params, "batch_stats": trained_batch_stats},
            hstate,
            _obs,
            _dones,
            False,
        )
        q_vals = q_vals.squeeze(axis=1)  # (num_agents, 1, action_dim)
        if action_masking:
            avail = wrapped_env.get_valid_actions(env_state)
            avail_batch = jnp.stack([avail[a] for a in env.agents])  # (num_agents, 1, action_dim)
            q_vals = q_vals - (1 - avail_batch) * 1e10
        actions = jnp.argmax(q_vals, axis=-1)  # (num_agents, 1)
        env_act = {a: actions[i] for i, a in enumerate(env.agents)}  # (1,) per agent
        return env_act, new_hs

    def render_frame(env_state, downscale=2):
        # env_state is batched LogWrapper state; .env_state is batched raw env state
        inner_state = jax.tree.map(lambda x: x[0], env_state.env_state)
        pixels = render_alem_pixels(inner_state, pixel_size, env.static_env_params, player_textures)
        pixels = np.array(pixels)
        if downscale > 1:
            pixels = pixels[:, ::downscale, ::downscale, :]
        return pixels

    rng, reset_rng = jax.random.split(rng)
    obs, env_state = wrapped_env.batch_reset(reset_rng)
    hstate = init_hstate

    max_steps = config.get("VIS_STEPS", 1000)
    frames = []

    for _ in range(max_steps):
        frame = render_frame(env_state)
        frames.append(frame)

        rng, act_rng, step_rng = jax.random.split(rng, 3)
        env_act, hstate = policy_step(obs, hstate, act_rng, env_state)

        obs, env_state, _, dones, _ = wrapped_env.batch_step(step_rng, env_state, env_act)

        if dones["__all__"][0]:
            break

    _save_visualization_gifs(frames, config)


def run_visualization_mappo_rnn(config, env, trained_params, rng, networks, scanned_rnn_class):
    """Run visualization rollouts and save GIF for MAPPO (separate Actor and Critic networks)."""
    import distrax

    env_name = config.get("ENV_NAME", "Alem-Coop-Symbolic")
    action_masking = config.get("ACTION_MASKING", False)
    if action_masking:
        from alem.alem_coop.action_masking import compute_action_mask

        _inner = getattr(env, "_env", env)
        _mask_fn = getattr(_inner, "action_mask_fn", compute_action_mask)

    from alem.alem_coop.constants import (
        BLOCK_PIXEL_SIZE_HUMAN,
        TEXTURES,
        load_player_specific_textures,
    )
    from alem.alem_coop.renderer.renderer_pixels import render_alem_pixels

    # Load textures for rendering
    pixel_size = BLOCK_PIXEL_SIZE_HUMAN
    player_textures = load_player_specific_textures(TEXTURES[pixel_size], env.num_agents)

    # Unpack networks and params
    actor_network, critic_network = networks
    actor_params, critic_params = trained_params

    # JIT compile the step function (without rendering)
    @jax.jit
    def policy_step(obs, hstate, rng, env_state):
        obs_batch = jnp.stack([obs[a] for a in env.agents])
        done_batch = jnp.zeros(env.num_agents, dtype=bool)
        ac_in = (obs_batch[np.newaxis, :], done_batch[np.newaxis, :])
        # Only need actor for inference
        hstate, pi = actor_network.apply(actor_params, hstate, ac_in)
        if action_masking:
            mask = _mask_fn(env_state, env.default_params, env.static_env_params)
            masked_logits = pi.logits + jnp.where(mask[None, :, :], 0.0, -1e10)
            pi = distrax.Categorical(logits=masked_logits)
        action = pi.sample(seed=rng)
        env_act = {a: action[0, i] for i, a in enumerate(env.agents)}
        return env_act, hstate

    # Render function (called outside JIT)
    def render_frame(env_state, downscale=2):
        pixels = render_alem_pixels(env_state, pixel_size, env.static_env_params, player_textures)
        pixels = np.array(pixels)
        if downscale > 1:
            pixels = pixels[:, ::downscale, ::downscale, :]
        return pixels

    # Run rollout with Python loop
    rng, reset_rng = jax.random.split(rng)
    obs, env_state = env.reset(reset_rng)
    hstate = scanned_rnn_class.initialize_carry(env.num_agents, config["GRU_HIDDEN_DIM"])

    max_steps = config.get("VIS_STEPS", 1000)
    frames = []

    for _ in range(max_steps):
        frame = render_frame(env_state)
        frames.append(frame)

        rng, act_rng, step_rng = jax.random.split(rng, 3)
        env_act, hstate = policy_step(obs, hstate, act_rng, env_state)

        obs, env_state, _, dones, _ = env.step(step_rng, env_state, env_act)

        if dones["__all__"]:
            break

    # Convert frames to numpy and create GIF
    _save_visualization_gifs(frames, config)


def _save_visualization_gifs(frames, config):
    """Shared logic for saving visualization GIFs."""
    frames_np = np.array(frames)
    if frames_np.max() <= 1.0:
        frames_np = (frames_np * 255).astype(np.uint8)
    else:
        frames_np = frames_np.astype(np.uint8)

    # frames_np shape: (T, num_agents, H, W, 3)
    # log per agent gifs
    num_agents = frames_np.shape[1]
    alem_env_path = os.environ.get("ALEM_BASE_DIR", ".")
    for agent_idx in range(num_agents):
        agent_frames = frames_np[:, agent_idx, :, :, :]  # shape (T, H, W, 3)
        gif_path = f"{alem_env_path}/outputs/{config['RUN_NAME']}_rollout_agent{agent_idx}.gif"
        os.makedirs(f"{alem_env_path}/outputs", exist_ok=True)
        imageio.mimsave(gif_path, agent_frames, fps=10)
        print(f"Saved visualization for agent {agent_idx} to {gif_path}")
        # Log to wandb
        wandb.log({f"rollout_video_agent{agent_idx}": wandb.Video(gif_path, fps=10, format="gif")})

    # log one large gif with all agents side by side
    combined_frames = []
    padding_width = 10  # pixels of spacing between agents
    for frame in frames_np:
        # Add padding between agent frames
        padded_frame = frame[0]
        for agent_idx in range(1, frame.shape[0]):
            padding = np.ones((frame.shape[1], padding_width, 3), dtype=frame.dtype) * 255
            padded_frame = np.hstack([padded_frame, padding, frame[agent_idx]])
        combined_frames.append(padded_frame)
    combined_np = np.array(combined_frames)

    # Save GIF
    gif_path = f"{alem_env_path}/outputs/{config['RUN_NAME']}_rollout.gif"
    os.makedirs(f"{alem_env_path}/outputs", exist_ok=True)
    imageio.mimsave(gif_path, combined_np, fps=10)
    print(f"Saved visualization to {gif_path}")

    # Log to wandb
    wandb.log({"rollout_video_all": wandb.Video(gif_path, fps=10, format="gif")})
