from functools import partial

import chex
import jax


class EnvironmentNoAutoReset:
    """Similar to the base Gymnax environment but without auto-resets."""

    @property
    def default_params(self):
        """Return the environment's default dynamic parameters."""
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0, 4))
    def step(
        self,
        key: chex.PRNGKey,
        state,
        action: int | float,
        params=None,
    ):
        """Perform a transition without automatically resetting.

        Args:
            key: JAX random key used by the transition.
            state: Current environment state.
            action: Action applied to the state.
            params: Optional dynamic parameters; defaults to ``default_params``.

        Returns:
            Observation, next state, reward, terminal flag, and info.
        """
        # Use default env parameters if no others specified
        if params is None:
            params = self.default_params
        obs, state, reward, done, info = self.step_env(key, state, action, params)
        return obs, state, reward, done, info

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key: chex.PRNGKey, params=None):
        """Reset the environment with optional dynamic parameters.

        Args:
            key: JAX random key used to generate the initial state.
            params: Optional dynamic parameters; defaults to ``default_params``.

        Returns:
            Initial observation and environment state.
        """
        # Use default env parameters if no others specified
        if params is None:
            params = self.default_params
        obs, state = self.reset_env(key, params)
        return obs, state

    def step_env(
        self,
        key: chex.PRNGKey,
        state,
        action: int | float,
        params,
    ):
        """Implement an environment-specific step transition.

        Args:
            key: JAX random key used by the transition.
            state: Current environment state.
            action: Action applied to the state.
            params: Dynamic environment parameters.

        Raises:
            NotImplementedError: Always; subclasses must implement this method.
        """
        raise NotImplementedError

    def reset_env(self, key: chex.PRNGKey, params):
        """Implement an environment-specific reset.

        Args:
            key: JAX random key used to generate the initial state.
            params: Dynamic environment parameters.

        Raises:
            NotImplementedError: Always; subclasses must implement this method.
        """
        raise NotImplementedError

    def get_obs(self, state) -> chex.Array:
        """Apply the observation function to a state.

        Args:
            state: Environment state to observe.

        Raises:
            NotImplementedError: Always; subclasses must implement this method.
        """
        raise NotImplementedError

    def is_terminal(self, state, params) -> bool:
        """Check whether a state is terminal.

        Args:
            state: Environment state to inspect.
            params: Dynamic environment parameters.

        Raises:
            NotImplementedError: Always; subclasses must implement this method.
        """
        raise NotImplementedError

    def discount(self, state, params) -> float:
        """Return a discount of zero if the episode has terminated.

        Args:
            state: Environment state to inspect.
            params: Dynamic environment parameters.

        Returns:
            Scalar discount equal to zero when terminal and one otherwise.
        """
        return jax.lax.select(self.is_terminal(state, params), 0.0, 1.0)

    @property
    def name(self) -> str:
        """Return the concrete environment name.

        Returns:
            Concrete class name.
        """
        return type(self).__name__

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        raise NotImplementedError

    def action_space(self, params):
        """Return the action space for dynamic parameters.

        Args:
            params: Dynamic environment parameters.

        Raises:
            NotImplementedError: Always; subclasses must implement this method.
        """
        raise NotImplementedError

    def observation_space(self, params):
        """Return the observation space for dynamic parameters.

        Args:
            params: Dynamic environment parameters.

        Raises:
            NotImplementedError: Always; subclasses must implement this method.
        """
        raise NotImplementedError

    def state_space(self, params):
        """Return the state space for dynamic parameters.

        Args:
            params: Dynamic environment parameters.

        Raises:
            NotImplementedError: Always; subclasses must implement this method.
        """
        raise NotImplementedError
