"""Tests for simultaneous gameplay and communication actions."""

import jax
import jax.numpy as jnp

from alem.alem_coop.action_masking import compute_action_mask
from alem.alem_coop.alem_state import StaticEnvParams
from alem.alem_coop.constants import Action
from alem.alem_env import make_alem_env_from_name


def _make_env():
    return make_alem_env_from_name(
        "Alem-Coop-Symbolic-Separate-Comm",
        static_env_params=StaticEnvParams(player_count=3, num_comm_channels=4),
        compute_full_info=False,
    )


def test_product_action_space_has_gameplay_and_silence_plus_channels():
    env = _make_env()
    space = env.action_space(env.agents[0])
    assert space.shape == (2,)
    assert space.num_categories.tolist() == [len(Action) + 1, 5]


def test_gameplay_action_and_message_are_applied_on_same_step():
    env = _make_env()
    _, state = env.reset(jax.random.PRNGKey(0))
    actions = {
        agent: jnp.array([Action.REQUEST_FOOD.value, idx + 1], dtype=jnp.int32)
        for idx, agent in enumerate(env.agents)
    }

    observations, next_state, _, _, _ = env.step_env(jax.random.PRNGKey(1), state, actions)

    assert jnp.all(next_state.request_count == 1)
    assert jnp.array_equal(next_state.comm_messages, jnp.eye(4, dtype=jnp.float32)[:3])
    assert jnp.all(next_state.comm_count == 1)

    # The action chosen at t is broadcast in every agent's observation at t + 1.
    dashboard_start = env.get_flat_map_obs_shape()
    dashboard_size_with_directions = env.get_teammate_dashboard_obs_shape()
    dashboard_row_size = dashboard_size_with_directions // env.num_agents - 8
    dashboard_size = dashboard_row_size * env.num_agents
    num_channels = env.communication_action_dim - 1
    communication_start = dashboard_row_size - num_channels
    for observation in observations.values():
        dashboard = observation[dashboard_start : dashboard_start + dashboard_size].reshape(
            env.num_agents, dashboard_row_size
        )
        messages = dashboard[:, communication_start : communication_start + num_channels]
        assert jnp.array_equal(messages.sum(axis=0), jnp.array([1.0, 1.0, 1.0, 0.0]))


def test_agents_that_cannot_act_are_silenced():
    """alem_step forces their gameplay action to NOOP, so they must not broadcast either."""
    env = _make_env()
    _, state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(
        player_alive=state.player_alive.at[0].set(False),
        is_sleeping=state.is_sleeping.at[1].set(True),
        is_resting=state.is_resting.at[2].set(True),
    )
    actions = {agent: jnp.array([Action.NOOP.value, 1], dtype=jnp.int32) for agent in env.agents}

    _, next_state, _, _, _ = env.step_env(jax.random.PRNGKey(1), state, actions)

    assert jnp.all(next_state.comm_messages == 0)
    assert jnp.all(next_state.comm_count == 0)


def test_communication_mask_allows_only_silence_when_agent_cannot_act():
    env = _make_env()
    _, state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(is_sleeping=state.is_sleeping.at[1].set(True))

    masks = env.get_avail_communications(state)

    assert jnp.all(masks["agent_0"])
    assert masks["agent_1"].tolist() == [True, False, False, False, False]
    assert jnp.all(masks["agent_2"])


def test_gameplay_mask_matches_the_gameplay_prefix_of_the_combined_mask():
    env = _make_env()
    _, state = env.reset(jax.random.PRNGKey(0))
    combined = compute_action_mask(state, env.default_params, env.static_env_params)

    masks = env.get_avail_actions(state)

    assert combined.shape[1] == env.gameplay_action_dim + env.static_env_params.num_comm_channels
    for i, agent in enumerate(env.agents):
        assert masks[agent].shape == (env.gameplay_action_dim,)
        assert jnp.array_equal(masks[agent], combined[i, : env.gameplay_action_dim])
        # A masked-out gameplay row would make the policy's masked logits degenerate.
        assert masks[agent].any()


def test_silence_resets_one_step_messages_without_incrementing_counts():
    env = _make_env()
    _, state = env.reset(jax.random.PRNGKey(0))
    send = {agent: jnp.array([Action.NOOP.value, 1], dtype=jnp.int32) for agent in env.agents}
    _, state, _, _, _ = env.step_env(jax.random.PRNGKey(1), state, send)
    silent = {agent: jnp.array([Action.NOOP.value, 0], dtype=jnp.int32) for agent in env.agents}

    _, state, _, _, _ = env.step_env(jax.random.PRNGKey(2), state, silent)

    assert jnp.all(state.comm_messages == 0)
    assert jnp.all(state.comm_count == 1)
