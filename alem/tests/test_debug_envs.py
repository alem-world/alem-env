"""Tests for debug environments (single-level overworld only)."""

import unittest

import jax

from alem.alem_coop.envs.alem_symbolic_env import AlemCoopSymbolicEnv
from alem.alem_coop.envs.alem_symbolic_env_debug import AlemCoopSymbolicEnvDebug


class TestAlemCoopDebugEnv(unittest.TestCase):
    """Tests for Alem-Coop debug environment."""

    def test_debug_has_1_level(self):
        env = AlemCoopSymbolicEnvDebug(num_agents=3)
        rng = jax.random.PRNGKey(0)
        _, state = env.reset(rng)

        self.assertEqual(env.static_env_params.num_levels, 1)
        self.assertEqual(state.map.shape[0], 1)

    def test_non_debug_has_9_levels(self):
        env = AlemCoopSymbolicEnv(num_agents=3)
        rng = jax.random.PRNGKey(0)
        _, state = env.reset(rng)

        self.assertEqual(env.static_env_params.num_levels, 9)
        self.assertEqual(state.map.shape[0], 9)

    def test_debug_has_no_boss(self):
        env = AlemCoopSymbolicEnvDebug(num_agents=3)
        rng = jax.random.PRNGKey(0)
        _, state = env.reset(rng)

        self.assertEqual(state.boss_progress, 0)

    def test_non_debug_can_have_boss(self):
        env = AlemCoopSymbolicEnv(num_agents=3)
        rng = jax.random.PRNGKey(0)
        _, state = env.reset(rng)

        self.assertGreaterEqual(state.boss_timesteps_to_spawn_this_round, 0)


if __name__ == "__main__":
    unittest.main()
