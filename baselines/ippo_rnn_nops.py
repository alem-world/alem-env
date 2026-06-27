"""
IPPO RNN with non-shared (per-agent) policy/value weights.

Code is adapted from the IPPO RNN implementation of JaxMARL (https://github.com/FLAIROx/JaxMARL/tree/main)
Credit goes to the original authors: Rutherford et al.
"""

# ===========================
# Imports and Configuration
# ===========================
import os
import sys

from utils import (
    restore_baseline_checkpoint,
    run_final_eval,
    run_visualization_ippo_rnn,
    save_checkpoint,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import argparse
import functools
from collections.abc import Sequence
from typing import Dict, NamedTuple

import distrax
import flax.linen as nn
import hydra
import imageio
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
import yaml
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from jaxmarl.wrappers.baselines import LogWrapper
from omegaconf import OmegaConf

from alem.alem_coop.action_masking import compute_action_mask
from alem.alem_coop.alem_state import (
    COORDINATION_PRESETS,
    DIFFICULTY_ALPHAS,
    EnvParams,
    StaticEnvParams,
    get_coordination_params,
)
from alem.alem_env import make_alem_env_from_name


# ===========================
# Model Definitions
# ===========================
class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(*rnn_state.shape),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


def get_activation(activation: str):
    """Return activation function from config string."""
    activation = activation.lower()
    if activation == "relu":
        return nn.relu
    elif activation == "tanh":
        return nn.tanh
    else:
        raise ValueError(f"Unsupported activation: {activation}")


class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: dict

    @nn.compact
    def __call__(self, hidden, x):
        activation = get_activation(self.config["ACTIVATION"])
        obs, dones = x
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = activation(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(
            self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        actor_mean = activation(actor_mean)
        action_logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=action_logits)

        critic = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1)


# ===========================
# Data Structures and Utilities
# ===========================
class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray


def stack_agents(x: dict, agent_list):
    """Stack agent observations into array with shape (num_agents, num_envs, ...)."""
    return jnp.stack([x[a] for a in agent_list])


# ===========================
# Training Function
# ===========================
def make_train(config, env):
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    assert config["NUM_ACTORS"] % config["NUM_MINIBATCHES"] == 0, (
        "NUM_ACTORS (= num_agents * NUM_ENVS = "
        f"{env.num_agents} * {config['NUM_ENVS']} = {config['NUM_ACTORS']}) must be "
        f"divisible by NUM_MINIBATCHES ({config['NUM_MINIBATCHES']}); "
        "pick a NUM_ENVS such that num_agents * NUM_ENVS is a multiple of NUM_MINIBATCHES."
    )
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    env = LogWrapper(env)

    def linear_schedule(count):
        update_step = count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])

        warmup_fraction = config.get("LR_WARMUP", 0.0)

        if warmup_fraction > 0.0:
            warmup_steps = config["NUM_UPDATES"] * warmup_fraction
            decay_steps = config["NUM_UPDATES"] - warmup_steps

            # 1. Warmup phase: 0 to LR
            warmup_lr = config["LR"] * (update_step / jnp.maximum(1.0, warmup_steps))

            # 2. Decay phase: LR to 0 (starting only AFTER warmup_steps)
            steps_since_warmup = update_step - warmup_steps
            decay_frac = 1.0 - (steps_since_warmup / jnp.maximum(1.0, decay_steps))

            # Prevent learning rate from going negative at the very end
            decay_lr = config["LR"] * jnp.maximum(0.0, decay_frac)

            # 3. Choose the right phase based on the current step
            lr = jnp.where(update_step < warmup_steps, warmup_lr, decay_lr)
        else:
            # Fallback if no warmup is configured
            frac = 1.0 - (update_step / config["NUM_UPDATES"])
            lr = config["LR"] * jnp.maximum(0.0, frac)

        return lr

    def train(rng):
        # INIT NETWORK
        # Each agent gets its own parameters (initialized with different RNG keys)
        network = ActorCriticRNN(env.action_space(env.agents[0]).n, config=config)
        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros((1, config["NUM_ENVS"], env.observation_space(env.agents[0]).shape[0])),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])

        # Per-agent params: vmap init over different RNG keys
        agent_rngs = jax.random.split(_rng, env.num_agents)
        network_params = jax.vmap(lambda prng: network.init(prng, init_hstate, init_x))(agent_rngs)
        # network_params pytree leaves have shape (num_agents, ...)

        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        # hstate: (num_agents, NUM_ENVS, hidden_dim) — one hidden state per agent
        init_hstate = jnp.tile(
            ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])[None, :, :],
            (env.num_agents, 1, 1),
        )

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            runner_state, update_steps = update_runner_state

            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, hstate, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)

                # Stack obs/done per agent: (num_agents, NUM_ENVS, ...)
                obs_agents = stack_agents(last_obs, env.agents)

                # Per-agent forward pass: add time dim -> (num_agents, 1, NUM_ENVS, ...)
                ac_obs = obs_agents[:, None, :, :]
                ac_done = last_done[:, None, :]

                hstate, pi, value = jax.vmap(
                    lambda p, h, o, d: network.apply(p, h, (o, d)),
                    in_axes=(0, 0, 0, 0),
                )(train_state.params, hstate, ac_obs, ac_done)
                # pi.logits: (num_agents, 1, NUM_ENVS, action_dim)
                # value: (num_agents, 1, NUM_ENVS)

                # ACTION MASKING
                if config.get("ACTION_MASKING", False):
                    inner_env = env._env
                    mask = jax.vmap(
                        lambda s: compute_action_mask(
                            s, inner_env.default_params, inner_env.static_env_params
                        )
                    )(env_state.env_state)  # (NUM_ENVS, num_agents, num_actions)
                    mask_batch = mask.transpose(1, 0, 2)  # (num_agents, NUM_ENVS, num_actions)
                    masked_logits = pi.logits + jnp.where(mask_batch[:, None, :, :], 0.0, -1e10)
                    pi = distrax.Categorical(logits=masked_logits)
                else:
                    mask_batch = jnp.ones(
                        (env.num_agents, config["NUM_ENVS"], env.action_space(env.agents[0]).n),
                        dtype=jnp.bool_,
                    )

                # Sample actions: squeeze time dim
                action = pi.sample(seed=_rng)[:, 0, :]  # (num_agents, NUM_ENVS)
                log_prob = pi.log_prob(action[:, None, :])[:, 0, :]  # (num_agents, NUM_ENVS)

                env_act = {a: action[i] for i, a in enumerate(env.agents)}

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                    rng_step, env_state, env_act
                )

                done_agents = stack_agents(done, env.agents)  # (num_agents, NUM_ENVS)
                reward_agents = stack_agents(reward, env.agents)  # (num_agents, NUM_ENVS)
                global_done = jnp.tile(done["__all__"][None, :], (env.num_agents, 1))

                transition = Transition(
                    global_done,
                    last_done,
                    action,
                    value[:, 0, :],  # squeeze time dim
                    reward_agents,
                    log_prob,
                    obs_agents,
                    info,
                    mask_batch,
                )
                runner_state = (train_state, env_state, obsv, done_agents, hstate, rng)
                return runner_state, transition

            initial_hstate = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )
            # traj_batch fields: (NUM_STEPS, num_agents, NUM_ENVS, ...)

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, hstate, rng = runner_state
            last_obs_agents = stack_agents(last_obs, env.agents)
            ac_in_obs = last_obs_agents[:, None, :, :]
            ac_in_done = last_done[:, None, :]
            _, _, last_val = jax.vmap(
                lambda p, h, o, d: network.apply(p, h, (o, d)),
                in_axes=(0, 0, 0, 0),
            )(train_state.params, hstate, ac_in_obs, ac_in_done)
            last_val = last_val[:, 0, :]  # (num_agents, NUM_ENVS)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.global_done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)
            # advantages/targets: (NUM_STEPS, num_agents, NUM_ENVS)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    (
                        init_hstate,
                        obs_mb,
                        done_mb,
                        action_mb,
                        old_value_mb,
                        old_log_prob_mb,
                        avail_actions_mb,
                        adv_mb,
                        targets_mb,
                    ) = batch_info
                    # Each field: (num_agents, NUM_STEPS, mb_envs, ...)

                    def _loss_fn(
                        params,
                        init_hstate,
                        obs,
                        done,
                        action,
                        old_value,
                        old_log_prob,
                        avail_actions,
                        gae,
                        targets,
                    ):
                        # RERUN NETWORK
                        _, pi, value = network.apply(
                            params,
                            init_hstate,
                            (obs, done),
                        )
                        # Apply action mask to logits (same mask used during rollout)
                        if config.get("ACTION_MASKING", False):
                            masked_logits = pi.logits + jnp.where(avail_actions, 0.0, -1e10)
                            pi = distrax.Categorical(logits=masked_logits)
                        log_prob = pi.log_prob(action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = old_value + (value - old_value).clip(
                            -config["CLIP_EPS"], config["CLIP_EPS"]
                        )
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                        # CALCULATE ACTOR LOSS
                        logratio = log_prob - old_log_prob
                        ratio = jnp.exp(logratio)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        # debug
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (
                            value_loss,
                            loss_actor,
                            entropy,
                            ratio,
                            approx_kl,
                            clip_frac,
                        )

                    # Vmap loss+grad across agents (each agent has own params)
                    grad_fn = jax.vmap(
                        jax.value_and_grad(_loss_fn, has_aux=True),
                        in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                    )
                    total_loss, grads = grad_fn(
                        train_state.params,
                        init_hstate,
                        obs_mb,
                        done_mb,
                        action_mb,
                        old_value_mb,
                        old_log_prob_mb,
                        avail_actions_mb,
                        adv_mb,
                        targets_mb,
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    # Average losses across agents for logging
                    total_loss = (
                        total_loss[0].mean(),
                        jax.tree.map(lambda x: x.mean(), total_loss[1]),
                    )
                    return train_state, total_loss

                (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)

                # Shuffle across envs (agent dim stays intact)
                permutation = jax.random.permutation(_rng, config["NUM_ENVS"])
                num_mb = config["NUM_MINIBATCHES"]

                # init_hstate: (num_agents, NUM_ENVS, hidden_dim)
                shuffled_hstate = jnp.take(init_hstate, permutation, axis=1)
                # -> (num_mb, num_agents, mb_envs, hidden_dim)
                init_hstate_mb = jnp.swapaxes(
                    shuffled_hstate.reshape(env.num_agents, num_mb, -1, shuffled_hstate.shape[-1]),
                    0,
                    1,
                )

                # For per-agent traj fields with shape (NUM_STEPS, num_agents, NUM_ENVS, ...)
                # shuffle on axis=2, then reshape to (num_mb, num_agents, NUM_STEPS, mb_envs, ...)
                def _make_traj_mb(x):
                    shuffled = jnp.take(x, permutation, axis=2)
                    return jnp.swapaxes(
                        shuffled.reshape(x.shape[0], x.shape[1], num_mb, -1, *x.shape[3:]),
                        0,
                        2,
                    )

                obs_mb = _make_traj_mb(traj_batch.obs)
                done_mb = _make_traj_mb(traj_batch.done)
                action_mb = _make_traj_mb(traj_batch.action)
                old_value_mb = _make_traj_mb(traj_batch.value)
                old_log_prob_mb = _make_traj_mb(traj_batch.log_prob)
                avail_actions_mb = _make_traj_mb(traj_batch.avail_actions)

                # advantages/targets: (NUM_STEPS, num_agents, NUM_ENVS)
                def _make_adv_mb(x):
                    shuffled = jnp.take(x, permutation, axis=2)
                    reshaped = shuffled.reshape(x.shape[0], x.shape[1], num_mb, -1)
                    return jnp.swapaxes(reshaped, 0, 2)

                adv_mb = _make_adv_mb(advantages)
                targets_mb = _make_adv_mb(targets)

                minibatches = (
                    init_hstate_mb,
                    obs_mb,
                    done_mb,
                    action_mb,
                    old_value_mb,
                    old_log_prob_mb,
                    avail_actions_mb,
                    adv_mb,
                    targets_mb,
                )

                train_state, total_loss = jax.lax.scan(_update_minbatch, train_state, minibatches)
                update_state = (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            update_state = (
                train_state,
                initial_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info

            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)
            metric["update_steps"] = update_steps
            metric["loss"] = {
                "total_loss": loss_info[0],
                "value_loss": loss_info[1][0],
                "actor_loss": loss_info[1][1],
                "entropy": loss_info[1][2],
                "ratio": loss_info[1][3],
                "approx_kl": loss_info[1][4],
                "clip_frac": loss_info[1][5],
            }

            rng = update_state[-1]

            def callback(metrics, actor_state: TrainState, step):
                env_step = metrics["update_steps"] * config["NUM_ENVS"] * config["NUM_STEPS"]
                to_log = {
                    "env_step": env_step,
                    **metrics["loss"],
                }
                if metrics["returned_episode"].any():
                    to_log.update(
                        jax.tree.map(
                            lambda x: x[metrics["returned_episode"]].mean(), metrics["user_info"]
                        )
                    )
                    to_log["mean_episode_length"] = (
                        metrics["returned_episode_lengths"]
                        .mean(axis=-1)[metrics["returned_episode"][:, :, 0]]
                        .mean()
                    )
                    to_log["mean_episode_return"] = (
                        metrics["returned_episode_returns"]
                        .mean(axis=-1)[metrics["returned_episode"][:, :, 0]]
                        .mean()
                    )
                wandb.log(to_log, step=metrics["update_steps"])

            jax.experimental.io_callback(callback, None, metric, train_state, update_steps)
            update_steps = update_steps + 1
            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)
            return (runner_state, update_steps), metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros((env.num_agents, config["NUM_ENVS"]), dtype=bool),
            init_hstate,
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, (runner_state, 0), None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state}

    return train


# ===========================
# Main Run Function
# ===========================
def single_run(config):
    alg_name = config.get("ALG_NAME", "ippo-rnn-nops")
    env_name = config.get("ENV_NAME", "Alem-Coop-Symbolic")

    # Build coordination params from TRAINING_COORDINATION_DIFFICULTY
    train_coord_diff = config.get("TRAINING_COORDINATION_DIFFICULTY", "none")
    scale_base = config.get("SCALE_BASE_DIFFICULTY", False)
    if train_coord_diff == "sampled":
        # Domain randomisation: α ~ U[ALPHA_MIN, ALPHA_MAX] each episode.
        # Use full opportunity params as structural base so coordination is enabled.
        coord_kwargs = get_coordination_params("hard", scale_base=scale_base)
        randomize_alpha = True
    else:
        coord_kwargs = get_coordination_params(train_coord_diff, scale_base=scale_base)
        randomize_alpha = False

    # Create EnvParams with soft specialization and coordination settings
    env_params = EnvParams(
        # Game mode
        shared_reward=config.get("SHARED_REWARD", True),
        # Randomized difficulty (α domain randomization)
        randomize_alpha=randomize_alpha,
        alpha_min=config.get("ALPHA_MIN", 0.2),
        alpha_max=config.get("ALPHA_MAX", 0.85),
        # Coordination params (includes soft_specialization + non_specialist_efficiency)
        **coord_kwargs,
    )
    # Build StaticEnvParams from config (communication channels, player count, etc.)
    static_env_kwargs = {}
    num_comm = config.get("NUM_COMM_CHANNELS", 0)
    if num_comm > 0:
        static_env_kwargs["num_comm_channels"] = int(num_comm)
    num_agents = config.get("NUM_AGENTS", StaticEnvParams.player_count)
    static_env_kwargs["player_count"] = int(num_agents)
    static_env_params = StaticEnvParams(**static_env_kwargs)

    env = make_alem_env_from_name(
        env_name, env_params=env_params, static_env_params=static_env_params
    )

    tags = [
        alg_name.upper(),
        env_name.upper(),
        f"jax_{jax.__version__}",
        StaticEnvParams.version,
    ]
    if "RUN_TAGS" in config:
        tags += config["RUN_TAGS"]

    config["env_version"] = StaticEnvParams.version
    # Log all resolved coordination params so runs are fully reproducible from wandb
    config["coord_alpha"] = DIFFICULTY_ALPHAS.get(train_coord_diff, None)
    for k, v in coord_kwargs.items():
        config[f"coord_{k}"] = v

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=tags,
        name=config["RUN_NAME"],
        config=config,
        mode=config["WANDB_MODE"],
        save_code=True,
    )

    rng = jax.random.PRNGKey(config["SEED"])

    # Count and log network params
    _network = ActorCriticRNN(env.action_space(env.agents[0]).n, config=config)
    _init_x = (
        jnp.zeros((1, 1, env.observation_space(env.agents[0]).shape[0])),
        jnp.zeros((1, 1)),
    )
    _init_hstate = ScannedRNN.initialize_carry(1, config["GRU_HIDDEN_DIM"])
    _params = _network.init(jax.random.PRNGKey(0), _init_hstate, _init_x)
    per_agent_params = sum(x.size for x in jax.tree.leaves(_params))
    num_params = per_agent_params * env.num_agents
    print(
        f"Network parameters: {num_params:,} ({per_agent_params:,} per agent x {env.num_agents} agents)"
    )
    wandb.log({"num_params": num_params, "per_agent_params": per_agent_params})

    load_ckpt = config.get("LOAD_CHECKPOINT", None)
    if load_ckpt:
        # Eval-only mode: restore a stored policy and skip training.
        print(f"LOAD_CHECKPOINT set — skipping training, restoring: {load_ckpt}")
        restored = restore_baseline_checkpoint(load_ckpt)
        trained_params = restored["params"]
    else:
        rngs = jax.random.split(rng, config["NUM_SEEDS"])
        train_vjit = jax.jit(jax.vmap(make_train(config, env)))
        outs = jax.block_until_ready(train_vjit(rngs))

        # Save checkpoint
        train_state_vmapped = outs["runner_state"][0][0]  # Get train_state (still vmapped)
        trained_state = jax.tree.map(lambda x: x[0], train_state_vmapped)  # Get first seed
        save_checkpoint(trained_state, config)
        trained_params = trained_state.params

    network = ActorCriticRNN(env.action_space(env.agents[0]).n, config=config)

    # Run visualization rollouts (skip with VISUALIZE=False, e.g. for batch eval)
    if config.get("VISUALIZE", True):
        print("Running visualization rollouts...")
        rng, vis_rng = jax.random.split(rng)
        run_visualization_ippo_rnn(
            config, env, trained_params, vis_rng, network, ScannedRNN, nonshared_params=True
        )

    # Run final evaluation
    def ippo_policy_fn(hstate, obs_batch, done_batch, rng, avail_actions):
        """Policy fn for IPPO-NOPS: per-agent forward pass."""
        ac_obs = obs_batch[:, None, None, :]  # (num_agents, 1, 1, obs_dim)
        ac_done = done_batch[:, None, None]  # (num_agents, 1, 1)
        hstate, pi, _ = jax.vmap(
            lambda p, h, o, d: network.apply(p, h, (o, d)),
            in_axes=(0, 0, 0, 0),
        )(trained_params, hstate, ac_obs, ac_done)
        masked_logits = pi.logits + jnp.where(avail_actions[:, None, None, :], 0.0, -1e10)
        pi = distrax.Categorical(logits=masked_logits)
        actions = pi.mode()[:, 0, 0]  # squeeze time/env dims
        return hstate, actions

    # Per-agent hidden state: (num_agents, 1, hidden_dim)
    single_h = ScannedRNN.initialize_carry(1, config["GRU_HIDDEN_DIM"])
    eval_hstate = jnp.tile(single_h[None, :, :], (env.num_agents, 1, 1))
    rng, eval_rng = jax.random.split(rng)
    run_final_eval(config, env, ippo_policy_fn, eval_hstate, eval_rng)

    wandb.finish()


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="ippo_rnn_nops.yaml",
)
def main(config):
    config = OmegaConf.to_container(config, resolve=True)
    single_run(config)


if __name__ == "__main__":
    main()
