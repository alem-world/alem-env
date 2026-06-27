"""
Alem-Coop Environment Wrapper for BALROG-style Evaluation
"""

import logging
import re

import jax
import jax.numpy as jnp

from alem.alem_coop.constants import Action
from alem.alem_coop.envs.common import compute_score
from alem.llm.alem_language_wrapper import (
    ACTIONS,
    AlemLanguageWrapper,
    get_instruction_prompt,
    make_alem_env,
)
from alem.llm.alem_language_wrapper_single import (
    AlemLanguageWrapperSingle,
    get_instruction_prompt_single,
)

# from alem.llm.alem_language_wrapper import AlemLanguageWrapper, make_alem_env, ACTIONS, get_instruction_prompt


logger = logging.getLogger(__name__)


class CraftaxEnv:
    """Wrapper for Alem-Coop environment to work with the evaluator.

    This class wraps the AlemLanguageWrapper to provide the interface
    expected by the Evaluator class.
    """

    def __init__(self, task, config):
        """Initialize Alem-Coop environment.

        Args:
            task: Task name (e.g., "default")
            config: Configuration object
        """
        self.task = task
        self.config = config

        # Get Alem-specific config
        alem_config = config.get("alem", {})

        self.num_agents = alem_config.get("num_agents", 3)
        self.max_steps = config.eval.max_steps_per_episode
        self.failed_candidates = []

        env_name = config.get("ENV_NAME", "Alem-Coop-Symbolic")

        # Build env config — keep these params aligned across the symbolic,
        # pixel, and text interfaces for comparable results.
        env_config = {
            "max_timesteps": alem_config.get("max_timesteps", 10000),
            "god_mode": alem_config.get("god_mode", False),
            "coordination_difficulty": alem_config.get("coordination_difficulty", "none"),
            "soft_specialization": alem_config.get("soft_specialization", True),
            "shared_reward": alem_config.get("shared_reward", False),
            "specialist_efficiency": alem_config.get("specialist_efficiency", 1.0),
            "non_specialist_efficiency": alem_config.get("non_specialist_efficiency", 0.2),
            "randomize_alpha": alem_config.get("randomize_alpha", False),
            "num_agents": self.num_agents,
            "ENV_NAME": env_name,
        }

        logger.info(f"Creating {env_name} environment for task '{task}'")
        self.env = make_alem_env(config=env_config)
        self.env_params = self.env.default_params

        # Wrap for LLM control. Single-agent envs use AlemLanguageWrapperSingle
        # so the system prompt and per-turn observations drop teammate /
        # coordination text and the Request/Give actions.
        wrapper_config = alem_config.get("wrapper", {})
        wrapper_cls = AlemLanguageWrapperSingle if self.num_agents == 1 else AlemLanguageWrapper
        self.wrapper = wrapper_cls(
            self.env,
            self.env_params,
            unique_items=wrapper_config.get("unique_items", True),
            precise_location=wrapper_config.get("precise_location", False),
            exact_coordinates=wrapper_config.get("exact_coordinates", False),
            egocentric=wrapper_config.get("egocentric", False),
            skip_items=wrapper_config.get("skip_items", ["grass", "sand", "path"]),
            edge_only_items=wrapper_config.get("edge_only_items", ["water"]),
            render_pixel_size=wrapper_config.get("render_pixel_size", 10),
            render_downscale=wrapper_config.get("render_downscale", 1),
            llm_mode=wrapper_config.get("llm_mode", "easy"),
            prompt_mode=config.agent.get("prompt_mode", "specific_collaborative"),
            show_affordances=config.agent.get("show_affordances", False),
            debug=config.eval.get("debug", False),
            use_ascii=wrapper_config.get("use_ascii", False),
        )

        # State tracking
        self.rng = None
        self.state = None
        self.current_obs_list = None
        self.step_count = 0
        self.failed_candidates_per_agent = [[] for _ in range(self.num_agents)]

    def reset(self, seed=None):
        """Reset the environment.

        Args:
            seed: Random seed

        Returns:
            obs_list: List of observations for all agents
            info: Additional info dict
        """
        if seed is None:
            seed = 0

        self.rng = jax.random.PRNGKey(seed)
        text_obs_list, self.state, self.rng = self.wrapper.reset(self.rng)

        self.current_obs_list = text_obs_list
        self.step_count = 0
        self.failed_candidates = []
        self.failed_candidates_per_agent = [[] for _ in range(self.num_agents)]

        logger.info(f"Environment reset with seed {seed} for {self.num_agents} agents")

        info = {}
        return self.current_obs_list, info

    def step(self, actions):
        """Take a step in the environment with actions for all agents.

        Args:
            actions: List of action strings (one per agent)

        Returns:
            obs_list, rewards, terminateds, truncateds, info
        """
        if len(actions) != self.num_agents:
            raise ValueError(f"Expected {self.num_agents} actions, got {len(actions)}")

        text_obs_list, self.state, rewards, dones, info, self.rng = self.wrapper.step(
            self.state, actions, self.rng
        )

        self.current_obs_list = text_obs_list

        rewards_list = [float(r) for r in rewards]
        terminateds_list = [bool(d) for d in dones]

        self.step_count += 1
        truncateds_list = [self.step_count >= self.max_steps] * self.num_agents

        # Always recompute user_info with done=True so metrics are non-zero.
        # step_env computes compute_score(state, is_terminal) which is zero on
        # non-terminal steps; we need the actual values at episode end.
        episode_ending = any(terminateds_list) or any(truncateds_list)
        if episode_ending:
            info["user_info"] = compute_score(
                self.state, jnp.array(True), self.env.static_env_params
            )

        return text_obs_list, rewards_list, terminateds_list, truncateds_list, info

    def check_action_validity(self, action_str, agent_idx):
        """Canonicalize an action string for a specific agent.

        Args:
            action_str: Candidate action produced by the agent.
            agent_idx: Index of the acting agent.

        Returns:
            Canonical action string, falling back to ``Noop`` when invalid.
        """
        action_idx = self.wrapper.get_action_index(action_str, agent_idx)

        if action_idx is None:
            self.failed_candidates.append(action_str)
            self.failed_candidates_per_agent[agent_idx].append(action_str)
            logger.warning(
                f"Agent {agent_idx}: Invalid action '{action_str}', defaulting to 'Noop'"
            )
            return "Noop"

        # Preserve explicit GIVE targets for every GIVE slot, including the
        # first slot which shares the base Action.GIVE index.
        if action_idx >= Action.GIVE.value:
            raw = (action_str or "").strip()
            m = re.search(
                r"^\s*give\s+(?:to\s+)?(?:agent|teammate)[_\s-]*(\d+)\s*$",
                raw,
                re.IGNORECASE,
            )
            if m:
                target_idx = int(m.group(1))
                if 0 <= target_idx < self.num_agents and target_idx != agent_idx:
                    return f"Give to Agent {target_idx}"

            give_slot = action_idx - Action.GIVE.value
            if 0 <= give_slot < self.num_agents - 1:
                target_idx = give_slot if give_slot < agent_idx else give_slot + 1
                return f"Give to Agent {target_idx}"
            return "Give"

        return ACTIONS[action_idx]

    def get_instruction_prompt(self, agent_idx, instructions=None, current_level=None):
        """Get the instruction prompt for a specific agent.

        Args:
            agent_idx: Index of the agent.
            instructions: Unused, kept for interface compatibility.
            current_level: Dungeon level for progressive disclosure. If None,
                reads from self.state (falls back to 0 before first reset).

        Returns:
            System prompt configured for the selected agent and level.
        """
        from alem.alem_coop.constants import Specialization

        coordination_enabled = self.config.alem.get("coordination_difficulty", "none") != "none"
        llm_mode = self.config.alem.get("wrapper", {}).get("llm_mode", "easy")
        prompt_mode = self.config.agent.get("prompt_mode", "specific_collaborative")

        # Role assignment matches world_gen: [WARRIOR, FORAGER, MINER] cycling
        spec_order = [Specialization.WARRIOR, Specialization.FORAGER, Specialization.MINER]
        role = spec_order[agent_idx % 3].name.lower()

        include_all_actions = self.config.agent.get("include_all_actions", True)
        progressive_disclosure = self.config.agent.get("progressively_display_game_info", False)

        if current_level is None:
            current_level = int(self.state.player_level) if self.state is not None else 0

        if self.num_agents == 1:
            return get_instruction_prompt_single(
                llm_mode=llm_mode,
                agent_id=agent_idx,
                role=role,
                include_all_actions=include_all_actions,
                progressive_disclosure=progressive_disclosure,
                current_level=current_level,
                prompt_mode=prompt_mode,
            )

        return get_instruction_prompt(
            llm_mode=llm_mode,
            coordination_enabled=coordination_enabled,
            num_agents=self.num_agents,
            agent_id=agent_idx,
            role=role,
            include_all_actions=include_all_actions,
            progressive_disclosure=progressive_disclosure,
            current_level=current_level,
            prompt_mode=prompt_mode,
        )

    def get_stats(self, agent_idx=None):
        """Get current statistics for one or all agents.

        Args:
            agent_idx: Optional index selecting one agent.

        Returns:
            One agent's statistics, or a mapping for every agent.
        """
        if agent_idx is not None:
            return self.wrapper.get_stats(agent_idx)
        else:
            return {f"agent_{i}": self.wrapper.get_stats(i) for i in range(self.num_agents)}


def make_env(env_name, task, config):
    """Create the requested evaluator-facing environment.

    Args:
        env_name: Registered evaluator environment name.
        task: Task description passed to the environment wrapper.
        config: Evaluator and ALEM configuration object.

    Returns:
        Configured evaluator-facing environment.

    Raises:
        ValueError: If ``env_name`` is not registered.
    """
    if env_name == "alem":
        return CraftaxEnv(task, config)
    else:
        raise ValueError(f"Unknown environment: {env_name}")


AlemTextEnv = CraftaxEnv
