"""Record reference observation-space sizes for configured ALEM environments."""

import math
import unittest

from alem.alem_coop.alem_state import (
    EnvParams,
    StaticEnvParams,
    get_coordination_params,
)
from alem.alem_env import make_alem_env_from_name

ENV_NAMES = ("Alem-Coop-Symbolic", "Alem-Coop-Pixels")

EXPECTED_OBSERVATION_SPACE_RECORDS = {
    "Alem-Coop-Symbolic": {
        "shape": (9730,),
        "flat_size": 9730,
        "dtype": "<class 'jax.numpy.int32'>",
    },
    "Alem-Coop-Pixels": {
        "shape": (110, 150, 3),
        "flat_size": 49500,
        "dtype": "<class 'jax.numpy.float32'>",
    },
}


def make_reference_easy_env(env_name):
    """Build a fixed easy-coordination environment for shape regression tests."""
    coord_kwargs = get_coordination_params("easy", scale_base=True)
    env_params = EnvParams(
        shared_reward=False,
        randomize_alpha=False,
        alpha_min=0.2,
        alpha_max=0.85,
        **coord_kwargs,
    )
    static_env_params = StaticEnvParams(num_comm_channels=4)
    return make_alem_env_from_name(
        env_name,
        env_params=env_params,
        static_env_params=static_env_params,
    )


def observation_space_record(env_name):
    env = make_reference_easy_env(env_name)
    agent = env.agents[0]
    space = env.observation_space(agent)
    shape = tuple(space.shape)
    return {
        "env_name": env_name,
        "agent": agent,
        "shape": shape,
        "flat_size": math.prod(shape),
        "dtype": str(space.dtype),
    }


def observation_space_records():
    return [observation_space_record(env_name) for env_name in ENV_NAMES]


def format_observation_space_records(records):
    lines = ["Reference easy observation-space record:"]
    for record in records:
        lines.append(
            f"{record['env_name']} "
            f"agent={record['agent']} "
            f"shape={record['shape']} "
            f"flat_size={record['flat_size']} "
            f"dtype={record['dtype']}"
        )
    return "\n".join(lines)


class TestReferenceEasyObservationSpaceSizes(unittest.TestCase):
    def test_symbolic_and_pixel_observation_space_sizes(self):
        records = observation_space_records()
        print("\n" + format_observation_space_records(records))

        for record in records:
            expected = EXPECTED_OBSERVATION_SPACE_RECORDS[record["env_name"]]
            self.assertEqual(record["shape"], expected["shape"])
            self.assertEqual(record["flat_size"], expected["flat_size"])
            self.assertEqual(record["dtype"], expected["dtype"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
