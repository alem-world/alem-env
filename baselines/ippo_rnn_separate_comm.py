"""Parameter-sharing IPPO with separate gameplay and communication actions.

This is a sibling of ``ippo_rnn.py``. Each agent uses the shared recurrent
policy to choose an independent gameplay action and discrete broadcast message
on every step. The original baseline intentionally remains unchanged.
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
import time
from typing import NamedTuple

import distrax
import flax.linen as nn
import hydra
import imageio
import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from jaxmarl.wrappers.baselines import LogWrapper
from omegaconf import OmegaConf

import wandb
from alem.alem_coop.action_masking import compute_action_mask
from alem.alem_coop.game_logic import can_take_action
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
    action_dim: int
    communication_dim: int
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

        communication_logits = nn.Dense(
            self.communication_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(actor_mean)
        pi = FactorisedCategorical(
            distrax.Categorical(logits=action_logits),
            distrax.Categorical(logits=communication_logits),
        )

        critic = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1)


# ===========================
# Data Structures and Utilities
# ===========================
class FactorisedCategorical(NamedTuple):
    """Independent gameplay and communication categorical distributions."""

    gameplay: distrax.Categorical
    communication: distrax.Categorical

    def sample(self, seed):
        gameplay_key, communication_key = jax.random.split(seed)
        return jnp.stack(
            [
                self.gameplay.sample(seed=gameplay_key),
                self.communication.sample(seed=communication_key),
            ],
            axis=-1,
        )

    def mode(self):
        return jnp.stack([self.gameplay.mode(), self.communication.mode()], axis=-1)

    def log_prob(self, value):
        return self.gameplay.log_prob(value[..., 0]) + self.communication.log_prob(value[..., 1])

    def entropy(self):
        return self.gameplay.entropy() + self.communication.entropy()

    def with_gameplay_mask(self, mask):
        logits = self.gameplay.logits + jnp.where(mask, 0.0, -1e10)
        return self._replace(gameplay=distrax.Categorical(logits=logits))

    def with_communication_mask(self, mask):
        logits = self.communication.logits + jnp.where(mask, 0.0, -1e10)
        return self._replace(communication=distrax.Categorical(logits=logits))


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
    avail_communications: jnp.ndarray


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def get_policy_dims(env):
    """Return the two categorical head sizes from the product action space."""
    action_space = env.action_space(env.agents[0])
    if not hasattr(action_space, "num_categories"):
        raise ValueError("ippo_rnn_separate_comm requires Alem-Coop-Symbolic-Separate-Comm")
    categories = action_space.num_categories.tolist()
    if len(categories) != 2:
        raise ValueError("Separate communication expects exactly two action components")
    return int(categories[0]), int(categories[1])


def apply_gameplay_mask(pi, mask):
    """Mask gameplay logits while leaving every communication choice valid."""
    return pi.with_gameplay_mask(mask)


def build_comm_mask(env_state, config, num_actors, communication_action_dim, inner_env):
    """Legal-communication mask laid out to match ``batchify`` (agent-major).

    Mirrors the single-action-space baseline: an agent that cannot act (dead,
    sleeping or resting) may only stay silent, exactly as ``compute_action_mask``
    collapses those rows to NOOP-only. Without this the communication head keeps
    a full ln(K+1) of entropy and a live policy gradient on steps where
    ``step_env`` discards the message, injecting pure variance into the shared
    trunk and into the joint PPO ratio.

    ``DISABLE_COMM`` forces silence everywhere, giving a comms-off ablation of
    the split action space that is otherwise identical to the split runs.
    """
    if config.get("DISABLE_COMM", False):
        mask = jnp.zeros((num_actors, communication_action_dim), dtype=jnp.bool_)
        return mask.at[:, 0].set(True)

    if not config.get("ACTION_MASKING", False):
        return jnp.ones((num_actors, communication_action_dim), dtype=jnp.bool_)

    can_act = jax.vmap(can_take_action)(env_state)  # (NUM_ENVS, player_count)
    can_act = can_act.transpose(1, 0).reshape(num_actors)  # agent-major, matches batchify
    mask = jnp.ones((num_actors, communication_action_dim), dtype=jnp.bool_)
    return mask.at[:, 1:].set(can_act[:, None])


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
    gameplay_action_dim, communication_action_dim = get_policy_dims(env)

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
        network = ActorCriticRNN(
            gameplay_action_dim,
            config=config,
            communication_dim=communication_action_dim,
        )
        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros((1, config["NUM_ENVS"], env.observation_space(env.agents[0]).shape[0])),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
        network_params = network.init(_rng, init_hstate, init_x)
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
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"])

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            runner_state, update_steps = update_runner_state

            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, hstate, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                ac_in = (
                    obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                )
                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)

                # ACTION MASKING
                if config.get("ACTION_MASKING", False):
                    inner_env = env._env
                    mask_fn = getattr(inner_env, "action_mask_fn", compute_action_mask)
                    mask = jax.vmap(
                        lambda s: mask_fn(s, inner_env.default_params, inner_env.static_env_params)
                    )(env_state.env_state)  # (NUM_ENVS, player_count, num_actions)
                    mask_batch = mask.transpose(1, 0, 2).reshape(config["NUM_ACTORS"], -1)
                    pi = apply_gameplay_mask(pi, mask_batch)
                else:
                    mask_batch = jnp.ones(
                        (config["NUM_ACTORS"], gameplay_action_dim), dtype=jnp.bool_
                    )

                # COMMUNICATION MASKING
                comm_mask_batch = build_comm_mask(
                    env_state.env_state,
                    config,
                    config["NUM_ACTORS"],
                    communication_action_dim,
                    env._env,
                )
                pi = pi.with_communication_mask(comm_mask_batch)

                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                    rng_step, env_state, env_act
                )
                done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    last_done,
                    action.squeeze(),
                    value.squeeze(),
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    info,
                    mask_batch.squeeze(),
                    comm_mask_batch.squeeze(),
                )
                runner_state = (train_state, env_state, obsv, done_batch, hstate, rng)
                return runner_state, transition

            initial_hstate = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, hstate, rng = runner_state
            last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
            ac_in = (
                last_obs_batch[np.newaxis, :],
                last_done[np.newaxis, :],
            )
            _, _, last_val = network.apply(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze()

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

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        # RERUN NETWORK
                        _, pi, value = network.apply(
                            params,
                            init_hstate.squeeze(),
                            (traj_batch.obs, traj_batch.done),
                        )
                        # Apply action mask to logits (same mask used during rollout)
                        if config.get("ACTION_MASKING", False):
                            pi = apply_gameplay_mask(pi, traj_batch.avail_actions)
                        # Replay the exact rollout comm mask; deriving it here instead
                        # would desync the ratio from the stored log-probs.
                        pi = pi.with_communication_mask(traj_batch.avail_communications)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                            -config["CLIP_EPS"], config["CLIP_EPS"]
                        )
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                        # CALCULATE ACTOR LOSS
                        logratio = log_prob - traj_batch.log_prob
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

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, init_hstate, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
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

                # adding an additional "fake" dimensionality to perform minibatching correctly
                init_hstate = jnp.reshape(init_hstate, (1, config["NUM_ACTORS"], -1))
                batch = (
                    init_hstate,
                    traj_batch,
                    advantages.squeeze(),
                    targets.squeeze(),
                )
                permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])

                shuffled_batch = jax.tree.map(lambda x: jnp.take(x, permutation, axis=1), batch)

                minibatches = jax.tree.map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1] + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                train_state, total_loss = jax.lax.scan(_update_minbatch, train_state, minibatches)
                update_state = (
                    train_state,
                    init_hstate.squeeze(),
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

            ratio_0 = loss_info[1][3].at[0, 0].get().mean()
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)
            metric["update_steps"] = update_steps
            metric["loss"] = {
                "total_loss": loss_info[0],
                "value_loss": loss_info[1][0],
                "actor_loss": loss_info[1][1],
                "entropy": loss_info[1][2],
                "ratio": loss_info[1][3],
                "ratio_0": ratio_0,
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
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
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
    alg_name = config.get("ALG_NAME", "ippo-rnn")
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

    # COORDINATION_NONE_PARAMS omits soft_specialization and non_specialist_efficiency,
    # so they would fall back to EnvParams defaults (False / 0.2) without this override.
    if "SOFT_SPECIALIZATION" in config:
        coord_kwargs["soft_specialization"] = bool(config["SOFT_SPECIALIZATION"])
    if "NON_SPECIALIST_EFFICIENCY" in config:
        coord_kwargs["non_specialist_efficiency"] = float(config["NON_SPECIALIST_EFFICIENCY"])
    if "SPECIALIST_EFFICIENCY" in config:
        coord_kwargs["specialist_efficiency"] = float(config["SPECIALIST_EFFICIENCY"])

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
    # Build StaticEnvParams from config (communication channels, etc.)
    static_env_kwargs = {}
    num_agents = config.get("NUM_AGENTS", 3)
    if num_agents != 3:
        static_env_kwargs["player_count"] = int(num_agents)
    num_comm = config.get("NUM_COMM_CHANNELS", 0)
    if num_comm > 0:
        static_env_kwargs["num_comm_channels"] = int(num_comm)
    static_env_params = StaticEnvParams(**static_env_kwargs) if static_env_kwargs else None

    env = make_alem_env_from_name(
        env_name, env_params=env_params, static_env_params=static_env_params
    )
    gameplay_action_dim, communication_action_dim = get_policy_dims(env)

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
    _network = ActorCriticRNN(
        gameplay_action_dim,
        config=config,
        communication_dim=communication_action_dim,
    )
    _init_x = (
        jnp.zeros((1, 1, env.observation_space(env.agents[0]).shape[0])),
        jnp.zeros((1, 1)),
    )
    _init_hstate = ScannedRNN.initialize_carry(1, config["GRU_HIDDEN_DIM"])
    _params = _network.init(jax.random.PRNGKey(0), _init_hstate, _init_x)
    num_params = sum(x.size for x in jax.tree.leaves(_params))
    print(f"Network parameters: {num_params:,}")
    wandb.log({"num_params": num_params})

    load_ckpt = config.get("LOAD_CHECKPOINT", None)
    if load_ckpt:
        # Eval-only mode: restore a stored policy and skip training.
        print(f"LOAD_CHECKPOINT set — skipping training, restoring: {load_ckpt}")
        restored = restore_baseline_checkpoint(load_ckpt)
        trained_params = restored["params"]
    else:
        rngs = jax.random.split(rng, config["NUM_SEEDS"])
        train_vjit = jax.jit(jax.vmap(make_train(config, env)))
        t0 = time.time()
        outs = jax.block_until_ready(train_vjit(rngs))
        elapsed = time.time() - t0
        total_steps = config["NUM_UPDATES"] * config["NUM_STEPS"] * config["NUM_ENVS"]
        sps = total_steps / elapsed
        print(f"SPS: {sps:.0f} (total_steps={total_steps}, elapsed={elapsed:.1f}s)")
        wandb.log({"sps": sps})

        # Save checkpoint
        train_state_vmapped = outs["runner_state"][0][0]  # Get train_state (still vmapped)
        trained_state = jax.tree.map(lambda x: x[0], train_state_vmapped)  # Get first seed
        save_checkpoint(trained_state, config)
        trained_params = trained_state.params

    network = ActorCriticRNN(
        gameplay_action_dim,
        config=config,
        communication_dim=communication_action_dim,
    )

    # Run visualization rollouts (skip with VISUALIZE=False, e.g. for batch eval)
    if config.get("VISUALIZE", True):
        print("Running visualization rollouts...")
        rng, vis_rng = jax.random.split(rng)
        run_visualization_ippo_rnn(config, env, trained_params, vis_rng, network, ScannedRNN)

    # Run final evaluation
    def ippo_policy_fn(hstate, obs_batch, done_batch, rng, avail_actions):
        """Policy fn for IPPO: obs_batch (num_agents, obs_dim) -> actions (num_agents,)"""
        ac_in = (obs_batch[None, :], done_batch[None, :])
        hstate, pi, _ = network.apply(trained_params, hstate, ac_in)
        pi = apply_gameplay_mask(pi, avail_actions[None, :, :])
        if config.get("DISABLE_COMM", False):
            # step_env already silences agents that cannot act, so the only mask
            # that changes eval behaviour is the comms-off ablation.
            comm_mask = jnp.zeros((env.num_agents, communication_action_dim), dtype=jnp.bool_)
            comm_mask = comm_mask.at[:, 0].set(True)
            pi = pi.with_communication_mask(comm_mask[None, :, :])
        actions = pi.mode()
        return hstate, actions[0]  # squeeze time dim

    eval_hstate = ScannedRNN.initialize_carry(env.num_agents, config["GRU_HIDDEN_DIM"])
    rng, eval_rng = jax.random.split(rng)
    run_final_eval(config, env, ippo_policy_fn, eval_hstate, eval_rng)

    wandb.finish()


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="ippo_rnn_separate_comm.yaml",
)
def main(config):
    config = OmegaConf.to_container(config, resolve=True)
    single_run(config)


if __name__ == "__main__":
    main()
