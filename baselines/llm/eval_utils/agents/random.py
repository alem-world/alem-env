"""
Random Agent for Alem-Coop Evaluation Testing
Selects actions uniformly at random - useful for testing the evaluation pipeline
"""

import logging
import random
from collections import namedtuple

try:
    from .base import BaseAgent
except ImportError:

    class BaseAgent:
        def __init__(self, client_factory, prompt_builder):
            self.client = None
            self.prompt_builder = prompt_builder

        def reset(self):
            if hasattr(self.prompt_builder, "reset"):
                self.prompt_builder.reset()


logger = logging.getLogger(__name__)


LLMResponse = namedtuple(
    "LLMResponse",
    ["model_id", "completion", "stop_reason", "input_tokens", "output_tokens", "reasoning"],
)


# Alem-Coop actions (56 total)
CRAFTAX_ACTIONS = [
    "Noop",
    "Move West",
    "Move East",
    "Move North",
    "Move South",
    "Do",
    "Sleep",
    "Place Stone",
    "Place Table",
    "Place Furnace",
    "Place Plant",
    "Make Wood Pickaxe",
    "Make Stone Pickaxe",
    "Make Iron Pickaxe",
    "Make Wood Sword",
    "Make Stone Sword",
    "Make Iron Sword",
    "Rest",
    "Descend",
    "Ascend",
    "Make Diamond Pickaxe",
    "Make Diamond Sword",
    "Make Iron Armour",
    "Make Diamond Armour",
    "Shoot Arrow",
    "Make Arrow",
    "Cast Spell",
    "Place Torch",
    "Drink Potion Red",
    "Drink Potion Green",
    "Drink Potion Blue",
    "Drink Potion Pink",
    "Drink Potion Cyan",
    "Drink Potion Yellow",
    "Read Book",
    "Enchant Sword",
    "Enchant Armour",
    "Make Torch",
    "Level Up Dexterity",
    "Level Up Strength",
    "Level Up Intelligence",
    "Enchant Bow",
    "Request Food",
    "Request Drink",
    "Request Wood",
    "Request Stone",
    "Request Iron",
    "Request Coal",
    "Request Diamond",
    "Request Ruby",
    "Request Sapphire",
    "Give",
    "Build Shelter",
    "Build Forge",
    "Build Beacon",
]


class RandomAgent(BaseAgent):
    """Agent that selects actions uniformly at random."""

    def __init__(self, client_factory, prompt_builder, seed=None):
        super().__init__(client_factory, prompt_builder)
        if seed is not None:
            random.seed(seed)
            logger.info(f"RandomAgent initialized with seed: {seed}")
        else:
            logger.info("RandomAgent initialized with random seed")
        self.step_count = 0

    def act(self, obs, prev_action=None):
        action = random.choice(CRAFTAX_ACTIONS)
        self.step_count += 1

        if self.step_count % 50 == 0:
            long_obs = obs.get("text", {}).get("long_term_context", "")
            short_obs = obs.get("text", {}).get("short_term_context", "")
            obs_text = f"{long_obs}\n\n{short_obs}" if long_obs else short_obs
            logger.info(f"RandomAgent step {self.step_count}: selected '{action}'")
            logger.info(f" Observation:\n{obs_text}")

        return LLMResponse(
            model_id="random",
            completion=action,
            stop_reason="random_selection",
            input_tokens=0,
            output_tokens=0,
            reasoning=f"Randomly selected action: {action}",
        )

    def reset(self):
        if self.prompt_builder is not None:
            self.prompt_builder.reset()
        self.step_count = 0
        logger.debug("RandomAgent reset")


def make_random_action():
    """Convenience function to create a random action response."""
    action = random.choice(CRAFTAX_ACTIONS)
    return LLMResponse(
        model_id="random",
        completion=action,
        stop_reason="random_selection",
        input_tokens=0,
        output_tokens=0,
        reasoning=f"Randomly selected: {action}",
    )
