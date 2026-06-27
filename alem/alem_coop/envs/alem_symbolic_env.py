from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import chex
import jax
import jax.numpy as jnp
from jax import lax

if TYPE_CHECKING:
    from jaxtyping import Array, Float, Int
from alem.environment_base.jaxmarl_compat import MultiAgentEnv, spaces

from ..action_masking import compute_action_mask
from ..alem_state import EnvParams, EnvState, StaticEnvParams
from ..constants import OBS_DIM, Action, BlockType, ItemType, Specialization
from ..game_logic import alem_step
from ..renderer.renderer_symbolic import render_alem_symbolic
from ..util.game_logic_utils import has_beaten_boss
from ..world_gen.world_gen import generate_world
from .common import compute_score


class AlemCoopSymbolicEnv(MultiAgentEnv):
    def __init__(
        self,
        num_agents: int = None,
        env_params: EnvParams = None,
        static_env_params: StaticEnvParams = None,
        compute_full_info: bool = True,
    ):
        """Initialize the multi-agent symbolic environment.

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
            self.static_env_params = AlemCoopSymbolicEnv.default_static_params()
        self._env_params = env_params  # Store custom params if provided
        self.num_agents = self.static_env_params.player_count
        # When False, skips compute_score (200+ derived metrics) inside step_env.
        # Use for benchmarks; default True for training/eval where the metrics matter.
        self.compute_full_info = compute_full_info

        self.agents = [f"agent_{i}" for i in range(self.static_env_params.player_count)]
        self.action_spaces = {name: self.action_shape() for name in self.agents}
        self.observation_spaces = {name: self.observation_shape() for name in self.agents}

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey, _=None) -> tuple[dict[str, chex.Array], EnvState]:
        """Generate a new world and its per-agent symbolic observations.

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

        # info["discount"] = self.discount(state, self.default_params)
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
    def get_obs(self, state: EnvState) -> dict[str, Float[Array, "obs_dim"]]:
        """Render a flat symbolic observation for each agent.

        Args:
            state: Environment state to observe.

        Returns:
            Mapping from agent names to symbolic observation vectors.
        """
        obs_sym = lax.stop_gradient(
            render_alem_symbolic(
                state,
                self.static_env_params,
            )
        )
        obs = {n: o for n, o in zip(self.agents, obs_sym)}
        return obs

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: EnvState) -> dict[str, chex.Array]:
        """Compute the legal-action mask for each agent.

        Args:
            state: Environment state used to determine action validity.

        Returns:
            Mapping from agent names to boolean action masks.
        """
        mask = compute_action_mask(state, self.default_params, self.static_env_params)
        return {agent: mask[i] for i, agent in enumerate(self.agents)}

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
            + max(0, self.static_env_params.player_count - 2)
            + self.static_env_params.num_comm_channels
        )

    def get_flat_map_obs_shape(self):
        """Return the flattened local-map feature count.

        Returns:
            Number of spatial features in one symbolic observation.
        """
        num_mob_classes = 5
        num_mob_types = 8
        num_blocks = len(BlockType)
        num_items = len(ItemType)
        num_players = self.static_env_params.player_count
        teammate_dead_alive_bit = 1
        light_map = 1
        # Coordination channels: coord_value (normalized), soft_mask
        num_coord_channels = 2
        # Mob coordination markers: requires_coord, is_hard_coord
        num_mob_coord_channels = 2
        # Pending handover obs: normalized time_remaining (0=inactive)
        num_handover_channels = 1

        return (
            OBS_DIM[0]
            * OBS_DIM[1]
            * (
                num_players
                + teammate_dead_alive_bit
                + num_blocks
                + num_items
                + num_mob_classes * num_mob_types
                + light_map
                + num_coord_channels
                + num_mob_coord_channels
                + num_handover_channels
            )
        )

    def get_teammate_dashboard_obs_shape(self):
        """Return the teammate-dashboard feature count.

        Returns:
            Number of teammate status and direction features.
        """
        num_players = self.static_env_params.player_count
        num_health = 1
        num_alive = 1
        num_specialization = len(Specialization) - 1
        num_req_mats = Action.REQUEST_SAPPHIRE.value - Action.REQUEST_FOOD.value + 1
        num_comm = self.static_env_params.num_comm_channels
        num_directions = 8

        return num_players * (
            num_health + num_alive + num_specialization + num_req_mats + num_comm + num_directions
        )

    def get_inventory_obs_shape(self):
        """Return the inventory and intrinsic-stat feature count.

        Returns:
            Number of non-spatial player features.
        """
        num_inventory = (
            16  # ordinal scalars: materials + pickaxe/sword/sword_enchant/bow_enchant/bow
        )
        num_potions = 6
        num_intrinsics = 8
        num_directions = 4
        num_armour = 4  # ordinal scalar per slot (head/chest/legs/boots)
        num_armour_enchantments = 4  # ordinal scalar per slot
        num_special_values = 3
        num_special_level_values = 4
        return (
            num_inventory
            + num_potions
            + num_intrinsics
            + num_directions
            + num_armour
            + num_armour_enchantments
            + num_special_values
            + num_special_level_values
        )

    def observation_shape(self) -> spaces.Box:
        """Return the symbolic observation space for one agent.

        Returns:
            Flat bounded space combining map, dashboard, and inventory features.
        """
        obs_shape = (
            self.get_flat_map_obs_shape()
            + self.get_teammate_dashboard_obs_shape()
            + self.get_inventory_obs_shape()
        )

        return spaces.Box(
            0.0,
            1.0,
            (obs_shape,),
            dtype=jnp.int32,
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
