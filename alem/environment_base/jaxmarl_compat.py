"""Small JaxMARL-compatible surface used by ALEM environments.

ALEM only needs the multi-agent ``step`` wrapper plus ``Discrete`` and ``Box``
spaces. Keeping this local avoids pulling the full JaxMARL/Brax/MuJoCo stack
for environment-only installs.
"""

from functools import partial
from types import SimpleNamespace

import chex
import jax
import jax.numpy as jnp

# When JaxMARL is installed (e.g. for the RL baselines), subclass its space
# types so ALEM's spaces satisfy the strict ``isinstance`` checks inside JaxMARL
# wrappers such as ``CTRolloutManager`` (used by the PQN-VDN baseline). When it
# is absent (the environment-only install) we fall back to plain ``object`` and
# never pull in the JaxMARL/Brax/MuJoCo stack. Behaviour is identical either
# way -- the attributes and methods below are fully self-contained.
try:
    from jaxmarl.environments.spaces import Box as _BoxBase
    from jaxmarl.environments.spaces import Discrete as _DiscreteBase
except ImportError:
    _DiscreteBase = object
    _BoxBase = object


class Discrete(_DiscreteBase):
    def __init__(self, num_categories: int, dtype=jnp.int32):
        """Create a scalar discrete space.

        Args:
            num_categories: Number of integer values in the space.
            dtype: JAX dtype used for sampled values.
        """
        assert num_categories >= 0
        self.n = num_categories
        self.shape = ()
        self.dtype = dtype

    def sample(self, rng: chex.PRNGKey) -> chex.Array:
        """Sample a value using the supplied JAX random key.

        Args:
            rng: Random key used for sampling.

        Returns:
            Scalar integer in the discrete range.
        """
        return jax.random.randint(rng, shape=self.shape, minval=0, maxval=self.n).astype(self.dtype)

    def contains(self, x) -> bool:
        """Return whether a value lies within the discrete bounds.

        Args:
            x: Candidate scalar value.

        Returns:
            Scalar boolean indicating membership.
        """
        return jnp.logical_and(x >= 0, x < self.n)


class Box(_BoxBase):
    def __init__(self, low: float, high: float, shape: tuple[int, ...], dtype=jnp.float32):
        """Create a uniformly bounded array space.

        Args:
            low: Inclusive lower bound for every element.
            high: Inclusive upper bound for every element.
            shape: Shape of values in the space.
            dtype: JAX dtype used for sampled values.
        """
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype

    def sample(self, rng: chex.PRNGKey) -> chex.Array:
        """Sample an array uniformly using the supplied random key.

        Args:
            rng: Random key used for sampling.

        Returns:
            Array with this space's shape and dtype.
        """
        return jax.random.uniform(rng, shape=self.shape, minval=self.low, maxval=self.high).astype(
            self.dtype
        )

    def contains(self, x) -> bool:
        """Return whether every array element lies within the bounds.

        Args:
            x: Candidate array value.

        Returns:
            Scalar boolean indicating membership.
        """
        return jnp.logical_and(jnp.all(x >= self.low), jnp.all(x <= self.high))


spaces = SimpleNamespace(Discrete=Discrete, Box=Box)


class MultiAgentEnv:
    def observation_space(self, agent: str):
        """Return the observation space registered for an agent.

        Args:
            agent: Registered agent name.

        Returns:
            The agent's observation space.
        """
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        """Return the action space registered for an agent.

        Args:
            agent: Registered agent name.

        Returns:
            The agent's action space.
        """
        return self.action_spaces[agent]

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state,
        actions: dict[str, chex.Array],
        reset_state: object | None = None,
    ):
        """Step all agents and automatically reset terminal episodes.

        Args:
            key: JAX random key split between stepping and resetting.
            state: Current environment state.
            actions: Scalar action for each named agent.
            reset_state: Optional pre-generated state used for terminal resets.

        Returns:
            Observations, selected state, rewards, termination flags, and info.
        """
        key, key_reset = jax.random.split(key)
        obs_st, states_st, rewards, dones, infos = self.step_env(key, state, actions)

        if reset_state is None:
            obs_re, states_re = self.reset(key_reset)
        else:
            states_re = reset_state
            obs_re = self.get_obs(states_re)

        states = jax.tree_util.tree_map(
            lambda reset_value, step_value: jax.lax.select(
                dones["__all__"], reset_value, step_value
            ),
            states_re,
            states_st,
        )
        obs = jax.tree_util.tree_map(
            lambda reset_value, step_value: jax.lax.select(
                dones["__all__"], reset_value, step_value
            ),
            obs_re,
            obs_st,
        )
        return obs, states, rewards, dones, infos

    @property
    def name(self) -> str:
        """Return the concrete environment class name.

        Returns:
            Concrete class name.
        """
        return type(self).__name__
