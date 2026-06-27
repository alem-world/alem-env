from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import chex
import jax
import jax.numpy as jnp
from jax import lax

from alem.environment_base.jaxmarl_compat import MultiAgentEnv, spaces

if TYPE_CHECKING:
    from jaxtyping import Array, Float, Int

from ..alem_state import EnvParams, EnvState, StaticEnvParams
from ..constants import (
    BLOCK_PIXEL_SIZE_AGENT,
    INVENTORY_OBS_HEIGHT,
    OBS_DIM,
    TEXTURES,
    Action,
    load_player_specific_textures,
)
from ..game_logic import alem_step
from ..renderer.renderer_pixels import render_alem_pixels
from ..util.game_logic_utils import has_beaten_boss
from ..world_gen.world_gen import generate_world
from .common import compute_score


class AlemCoopPixelsEnv(MultiAgentEnv):
    def __init__(
        self,
        num_agents: int = None,
        env_params: EnvParams = None,
        static_env_params: StaticEnvParams = None,
        compute_full_info: bool = True,
    ):
        """Initialize the multi-agent pixel environment.

        Args:
            num_agents: Player count. Used to build StaticEnvParams when
                static_env_params is not provided. Ignored if static_env_params
                is given explicitly (its player_count takes precedence).
            env_params: Optional episode and gameplay parameters.
            static_env_params: Optional parameters controlling static array shapes.
            compute_full_info: Whether steps should calculate all score metrics.
        """
        if static_env_params is not None:
            self.static_env_params = static_env_params
        elif num_agents is not None:
            self.static_env_params = StaticEnvParams(player_count=num_agents)
        else:
            self.static_env_params = AlemCoopPixelsEnv.default_static_params()
        self._env_params = env_params  # Store custom params if provided
        self.num_agents = self.static_env_params.player_count
        self.compute_full_info = compute_full_info
        self.pixel_size = BLOCK_PIXEL_SIZE_AGENT

        self.agents = [f"agent_{i}" for i in range(self.static_env_params.player_count)]
        self.action_spaces = {name: self.action_shape() for name in self.agents}
        self.observation_spaces = {name: self.observation_shape() for name in self.agents}

        self.player_specific_textures = load_player_specific_textures(
            TEXTURES[self.pixel_size], self.static_env_params.player_count
        )

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey, _=None) -> tuple[dict[str, chex.Array], EnvState]:
        """Generate a new world and its per-agent pixel observations.

        Args:
            key: JAX random key used for world generation.
            _: Ignored compatibility parameter.

        Returns:
            Per-agent observations and the initial environment state.
        """
        state = generate_world(key, self.default_params, self.static_env_params)
        return self.get_obs(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self, key: chex.PRNGKey, state: EnvState, actions: dict[str, Int[Array, ""]]
    ) -> tuple[dict[str, chex.Array], EnvState, dict[str, Float[Array, ""]], dict[str, bool], dict]:
        """Advance the environment by one simultaneous multi-agent action.

        Args:
            key: JAX random key used by stochastic transition logic.
            state: Current environment state.
            actions: Scalar action for each named agent.

        Returns:
            Observations, next state, rewards, termination flags, and metrics.
        """
        actions = jnp.array([actions[a] for a in self.agents])
        state, reward = alem_step(key, state, actions, self.default_params, self.static_env_params)

        obs = self.get_obs(state)
        done = self.is_terminal(state, self.default_params)
        info = {}
        if self.compute_full_info:
            info["user_info"] = compute_score(state, done, self.static_env_params)
        else:
            info["user_info"] = {}
        info["discount"] = self.discount(state, self.default_params)

        agent_rewards = {n: r for n, r in zip(self.agents, reward)}

        agent_done = {n: done for n in self.agents}
        agent_done["__all__"] = done

        return (
            obs,
            lax.stop_gradient(state),
            agent_rewards,
            agent_done,
            info,
        )

    @partial(jax.jit, static_argnums=(0,))
    def get_obs(self, state: EnvState) -> dict[str, Float[Array, "height width 3"]]:
        """Render a normalized RGB observation for each agent.

        Args:
            state: Environment state to render.

        Returns:
            Mapping from agent names to floating-point RGB images.
        """
        pixels = lax.stop_gradient(
            render_alem_pixels(
                state, self.pixel_size, self.static_env_params, self.player_specific_textures
            )
            / 255.0
        )
        obs = {n: o for n, o in zip(self.agents, pixels)}
        return obs

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: EnvState) -> dict[str, chex.Array]:
        """Return the pixel environment's per-agent action availability.

        Args:
            state: Current state, retained for the common environment API.

        Returns:
            Mapping from agent names to action availability values.
        """
        aa = jnp.full(len(Action), True)
        return {agent: aa[i] for i, agent in enumerate(self.agents)}

    @property
    def default_params(self) -> EnvParams:
        """Return custom environment parameters or a default instance.

        Returns:
            Dynamic parameters used by resets and transitions.
        """
        return self._env_params if self._env_params is not None else EnvParams()

    @staticmethod
    def default_static_params() -> StaticEnvParams:
        """Return the default static environment parameters.

        Returns:
            A new default static-parameter instance.
        """
        return StaticEnvParams()

    def action_shape(self) -> spaces.Discrete:
        """Return the discrete action space for one agent.

        Returns:
            Discrete space covering gameplay, give, and communication actions.
        """
        return spaces.Discrete(
            len(Action)
            + (self.static_env_params.player_count - 2)
            + self.static_env_params.num_comm_channels
        )

    def observation_shape(self) -> spaces.Box:
        """Return the composed map, inventory, and teammate image space.

        Returns:
            Bounded RGB image space for one player observation.
        """
        map_height = OBS_DIM[0]
        inventory_height = INVENTORY_OBS_HEIGHT
        teammate_dashboard_height = (self.static_env_params.player_count + 1) // 2
        return spaces.Box(
            0.0,
            1.0,
            (
                OBS_DIM[1] * self.pixel_size,
                (map_height + inventory_height + teammate_dashboard_height) * self.pixel_size,
                3,
            ),
            dtype=jnp.float32,
        )

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Determine whether time, team death, or boss defeat ended the episode.

        Args:
            state: Current environment state.
            params: Episode parameters containing the time limit.

        Returns:
            Scalar JAX boolean indicating episode termination.
        """
        done_steps = state.timestep >= params.max_timesteps
        is_dead = jnp.logical_not(state.player_alive).all()
        defeated_boss = has_beaten_boss(state, self.static_env_params)
        is_terminal = jnp.logical_or(is_dead, done_steps)
        is_terminal = jnp.logical_or(is_terminal, defeated_boss)
        return is_terminal

    def discount(self, state, params) -> float:
        """Return a discount of zero if the episode has terminated.

        Args:
            state: Current environment state.
            params: Dynamic parameters containing terminal conditions.

        Returns:
            Scalar discount equal to zero when terminal and one otherwise.
        """
        return jax.lax.select(self.is_terminal(state, params), 0.0, 1.0)
