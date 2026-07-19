"""Symbolic ALEM variant with simultaneous gameplay and communication actions."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import chex
import jax
import jax.numpy as jnp
from jax import lax

if TYPE_CHECKING:
    from jaxtyping import Array, Float, Int

from alem.environment_base.jaxmarl_compat import spaces

from ..action_masking import compute_gameplay_action_mask
from ..alem_state import EnvState
from ..constants import Action
from ..game_logic import alem_step, can_take_action, process_separate_communication
from .alem_symbolic_env import AlemCoopSymbolicEnv
from .common import compute_score


class AlemCoopSymbolicSeparateCommEnv(AlemCoopSymbolicEnv):
    """ALEM with a product action ``[gameplay_action, communication_action]``.

    The gameplay component has the same actions as standard ALEM, excluding its
    legacy communication-as-action suffix. Communication component 0 is silence;
    components 1..K select one of K one-hot communication channels. Both choices
    are applied on every environment step.
    """

    action_mask_fn = staticmethod(compute_gameplay_action_mask)

    @property
    def gameplay_action_dim(self) -> int:
        return len(Action) + max(0, self.static_env_params.player_count - 2)

    @property
    def communication_action_dim(self) -> int:
        return self.static_env_params.num_comm_channels + 1

    def action_shape(self) -> spaces.MultiDiscrete:
        return spaces.MultiDiscrete([self.gameplay_action_dim, self.communication_action_dim])

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self, key: chex.PRNGKey, state: EnvState, actions: dict[str, Int[Array, "2"]]
    ) -> tuple[dict[str, chex.Array], EnvState, dict[str, Float[Array, ""]], dict[str, bool], dict]:
        action_components = jnp.stack([jnp.asarray(actions[a]) for a in self.agents])
        gameplay_actions = action_components[:, 0]
        # Silence players that cannot act, matching how alem_step forces their gameplay
        # action to NOOP. Read before the step: alem_step revives and wakes players.
        communication_actions = jnp.where(can_take_action(state), action_components[:, 1], 0)

        state, reward = alem_step(
            key,
            state,
            gameplay_actions,
            self.default_params,
            self.static_env_params,
        )
        # alem_step runs the legacy comm stage internally, but gameplay actions never
        # reach its comm offset, so that pass only clears last step's messages. The
        # separate comm component is applied here, on top of that reset.
        state = process_separate_communication(state, communication_actions, self.static_env_params)

        obs = self.get_obs(state)
        done = self.is_terminal(state, self.default_params)
        info = {
            "user_info": compute_score(state, done, self.static_env_params)
            if self.compute_full_info
            else {}
        }
        agent_rewards = {name: value for name, value in zip(self.agents, reward)}
        agent_done = {name: done for name in self.agents}
        agent_done["__all__"] = done
        return obs, lax.stop_gradient(state), agent_rewards, agent_done, info

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: EnvState) -> dict[str, chex.Array]:
        """Return legal gameplay masks for each agent."""
        mask = compute_gameplay_action_mask(state, self.default_params, self.static_env_params)
        return {agent: mask[i] for i, agent in enumerate(self.agents)}

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_communications(self, state: EnvState) -> dict[str, chex.Array]:
        """Return legal communication masks; agents that cannot act may only stay silent."""
        mask = jnp.ones((self.num_agents, self.communication_action_dim), dtype=jnp.bool_)
        mask = mask.at[:, 1:].set(can_take_action(state)[:, None])
        return {agent: mask[i] for i, agent in enumerate(self.agents)}
