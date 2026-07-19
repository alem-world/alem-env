"""Tests for the shared alem environment factory."""

import unittest

from alem.alem_coop.envs.alem_pixels_env import AlemCoopPixelsEnv
from alem.alem_coop.envs.alem_symbolic_env import AlemCoopSymbolicEnv
from alem.alem_coop.envs.alem_symbolic_env_debug import (
    AlemCoopSymbolicEnvDebug,
)
from alem.alem_coop.envs.alem_symbolic_separate_comm_env import (
    AlemCoopSymbolicSeparateCommEnv,
)
from alem.alem_env import make_alem_env_from_name


class TestAlemEnvFactory(unittest.TestCase):
    """Tests for environment-name normalization and factory dispatch."""

    def test_accepts_alem_symbolic_alias(self):
        env = make_alem_env_from_name("Alem-Coop-Symbolic")
        self.assertIsInstance(env, AlemCoopSymbolicEnv)

    def test_accepts_alem_debug_alias(self):
        env = make_alem_env_from_name("Alem-Coop-Symbolic-Debug")
        self.assertIsInstance(env, AlemCoopSymbolicEnvDebug)

    def test_accepts_alem_pixels_alias(self):
        env = make_alem_env_from_name("Alem-Coop-Pixels")
        self.assertIsInstance(env, AlemCoopPixelsEnv)

    def test_accepts_separate_communication_variant(self):
        env = make_alem_env_from_name("Alem-Coop-Symbolic-Separate-Comm")
        self.assertIsInstance(env, AlemCoopSymbolicSeparateCommEnv)


if __name__ == "__main__":
    unittest.main()
