"""The difficulty preset must reach the LLM environment unchanged.

Regression test for v0.2.2: up to v0.2.1 ``make_alem_env`` fell back to
``non_specialist_efficiency=0.2`` at every coordination difficulty, while RL eval
took the preset value (1 - alpha), so the two tracks were not evaluated on the
same role gates.
"""

import unittest

from alem.alem_coop.alem_state import EnvParams, get_coordination_params
from alem.llm.alem_language_wrapper import make_alem_env


class TestLLMEnvSpecialisationParams(unittest.TestCase):
    def test_preset_non_specialist_efficiency_reaches_env(self):
        for difficulty in ("easy", "medium", "hard"):
            env = make_alem_env({"coordination_difficulty": difficulty, "num_agents": 3})
            expected = get_coordination_params(difficulty)["non_specialist_efficiency"]
            self.assertAlmostEqual(
                float(env.default_params.non_specialist_efficiency), float(expected), places=4,
                msg=f"{difficulty}: LLM env must use the preset value RL eval uses",
            )
            self.assertTrue(bool(env.default_params.soft_specialization))

    def test_preset_overwrites_config_value(self):
        env = make_alem_env(
            {"coordination_difficulty": "hard", "num_agents": 3, "non_specialist_efficiency": 0.2}
        )
        expected = get_coordination_params("hard")["non_specialist_efficiency"]
        self.assertAlmostEqual(
            float(env.default_params.non_specialist_efficiency), float(expected), places=4
        )

    def test_config_value_applies_without_difficulty(self):
        env = make_alem_env(
            {"coordination_difficulty": "none", "num_agents": 3, "non_specialist_efficiency": 0.33}
        )
        self.assertAlmostEqual(float(env.default_params.non_specialist_efficiency), 0.33, places=4)
        env = make_alem_env({"coordination_difficulty": "none", "num_agents": 3})
        self.assertAlmostEqual(
            float(env.default_params.non_specialist_efficiency),
            float(EnvParams().non_specialist_efficiency),
            places=4,
        )


if __name__ == "__main__":
    unittest.main()
