"""Focused policy-head tests for simultaneous-communication IPPO."""

import os
import sys

import distrax
import jax
import jax.numpy as jnp

from alem.alem_coop.alem_state import StaticEnvParams
from alem.alem_env import make_alem_env_from_name

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "baselines")
)

from ippo_rnn_separate_comm import (  # noqa: E402
    ActorCriticRNN,
    FactorisedCategorical,
    ScannedRNN,
    apply_gameplay_mask,
    get_policy_dims,
)


def test_factorised_policy_samples_and_scores_joint_action():
    config = {
        "ACTIVATION": "tanh",
        "FC_DIM_SIZE": 16,
        "GRU_HIDDEN_DIM": 32,
    }
    network = ActorCriticRNN(7, communication_dim=5, config=config)
    hidden = ScannedRNN.initialize_carry(3, 32)
    inputs = (jnp.zeros((1, 3, 11)), jnp.zeros((1, 3), dtype=jnp.bool_))
    params = network.init(jax.random.PRNGKey(0), hidden, inputs)

    _, policy, value = network.apply(params, hidden, inputs)
    actions = policy.sample(jax.random.PRNGKey(1))

    assert isinstance(policy, FactorisedCategorical)
    assert actions.shape == (1, 3, 2)
    assert policy.log_prob(actions).shape == (1, 3)
    assert policy.entropy().shape == (1, 3)
    assert value.shape == (1, 3)


def test_gameplay_mask_does_not_mask_communication_head():
    gameplay = distrax.Categorical(logits=jnp.zeros((2, 3)))
    communication = distrax.Categorical(logits=jnp.zeros((2, 4)))
    policy = FactorisedCategorical(gameplay, communication)

    masked = apply_gameplay_mask(
        policy,
        jnp.array([[True, False, True], [False, True, False]]),
    )

    assert jnp.array_equal(masked.communication.logits, policy.communication.logits)
    assert masked.gameplay.mode().tolist() == [0, 1]


def test_shared_policy_joint_action_steps_separate_communication_env():
    env = make_alem_env_from_name(
        "Alem-Coop-Symbolic-Separate-Comm",
        static_env_params=StaticEnvParams(player_count=3, num_comm_channels=4),
        compute_full_info=False,
    )
    observations, state = env.reset(jax.random.PRNGKey(0))
    gameplay_dim, communication_dim = get_policy_dims(env)
    config = {
        "ACTIVATION": "tanh",
        "FC_DIM_SIZE": 16,
        "GRU_HIDDEN_DIM": 32,
    }
    network = ActorCriticRNN(
        gameplay_dim,
        communication_dim=communication_dim,
        config=config,
    )
    hidden = ScannedRNN.initialize_carry(env.num_agents, 32)
    inputs = (
        jnp.stack([observations[agent] for agent in env.agents])[None, ...],
        jnp.zeros((1, env.num_agents), dtype=jnp.bool_),
    )
    params = network.init(jax.random.PRNGKey(1), hidden, inputs)
    _, policy, _ = network.apply(params, hidden, inputs)
    gameplay_mask = env.action_mask_fn(state, env.default_params, env.static_env_params)
    actions = apply_gameplay_mask(policy, gameplay_mask).sample(jax.random.PRNGKey(2))[0]

    env_actions = {agent: actions[i] for i, agent in enumerate(env.agents)}
    _, next_state, _, _, _ = env.step_env(jax.random.PRNGKey(3), state, env_actions)

    channel = actions[:, 1] - 1
    is_message = (channel >= 0) & (channel < 4)
    expected_messages = jax.nn.one_hot(channel, 4) * is_message[:, None]
    assert jnp.array_equal(next_state.comm_messages, expected_messages)
