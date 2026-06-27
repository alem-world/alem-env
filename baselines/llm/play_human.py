"""
Interactive human play mode for Alem-Coop using the same text interface as LLM agents.

You see exactly what the LLM sees (observation text) and pick actions from the same
action list.  Useful for debugging the LLM interface and understanding the game.

Usage:
    # Play with defaults (3 agents, easy coordination, symbolic env)
    python baselines/llm/play_human.py eval.debug=true agent.type=robust_all

"""

import os
import re
import sys
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

# Path setup (same as eval_alem.py)
_project_root = str(Path(__file__).parent.parent.parent)
_alem_root = os.path.join(_project_root, "alem")
sys.path.insert(0, _project_root)
sys.path.insert(0, _alem_root)
sys.path.insert(0, str(Path(__file__).parent))

from eval_utils.prompt_builder import create_prompt_builder

from alem.llm.alem_env import CraftaxEnv
from alem.llm.alem_language_wrapper import ACTIONS

# ── Helpers ──────────────────────────────────────────────────────────────────

ACTIONS_NUMBERED = {i: a for i, a in enumerate(ACTIONS)}
ACTIONS_BY_NAME = {a.lower(): a for a in ACTIONS}

# Shorthand aliases for common actions
ALIASES = {
    "w": "Move North",
    "s": "Move South",
    "a": "Move West",
    "d": "Move East",
    "e": "Do",
    "q": "Noop",
    "z": "Sleep",
}


def print_separator():
    print("\n" + "=" * 80)


def print_actions():
    """Print the available actions in columns."""
    print("\n  Available actions:")
    cols = 3
    for i, action in enumerate(ACTIONS):
        end = "\n" if (i + 1) % cols == 0 else ""
        print(f"  {i:>2}. {action:<25s}", end=end)
    if len(ACTIONS) % cols != 0:
        print()
    print("\n  Shortcuts: w/a/s/d=move, e=Do, q=Noop, z=Sleep")
    print("  Type 'actions' to show this list, 'quit' to exit, 'help' for more.\n")


def parse_action(raw_input, agent_idx=None, num_agents=None):
    """Parse user input into a valid action string, or None."""
    raw = raw_input.strip()
    if not raw:
        return None

    # Targeted GIVE forms: "Give to Agent X", "Give Agent X", "Give to teammate X"
    # Keep the explicit target string so wrapper routing can select the right GIVE slot.
    give_target_match = re.search(
        r"^\s*give\s+(?:to\s+)?(?:agent|teammate)[_\s-]*(\d+)\s*$", raw, re.IGNORECASE
    )
    if give_target_match:
        target_idx = int(give_target_match.group(1))
        if num_agents is not None and (target_idx < 0 or target_idx >= num_agents):
            return None
        if agent_idx is not None and target_idx == agent_idx:
            return None
        return f"Give to Agent {target_idx}"

    # Check aliases
    if raw.lower() in ALIASES:
        return ALIASES[raw.lower()]

    # Check by number
    try:
        idx = int(raw)
        if idx in ACTIONS_NUMBERED:
            return ACTIONS_NUMBERED[idx]
    except ValueError:
        pass

    # Check by exact name (case-insensitive)
    if raw.lower() in ACTIONS_BY_NAME:
        return ACTIONS_BY_NAME[raw.lower()]

    # Fuzzy prefix match
    matches = [a for a in ACTIONS if a.lower().startswith(raw.lower())]
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        print(f"  Ambiguous: {', '.join(matches)}")
        return None

    return None


def get_human_action(agent_idx, num_agents):
    """Prompt the human for an action for a specific agent."""
    label = f"Agent {agent_idx}" if num_agents > 1 else "Action"
    while True:
        try:
            raw = input(f"  [{label}] > ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            sys.exit(0)

        if raw.strip().lower() == "quit":
            print("Exiting.")
            sys.exit(0)
        if raw.strip().lower() == "actions":
            print_actions()
            continue
        if raw.strip().lower() == "help":
            print("  Enter action by number, name, alias, or prefix.")
            print("  'actions' = show list, 'quit' = exit")
            continue

        action = parse_action(raw, agent_idx=agent_idx, num_agents=num_agents)
        if action is None:
            print(f"  Invalid action: '{raw}'. Type 'actions' to see the list.")
            continue

        return action


def _create_agents(config, env, num_agents):
    """Create real agent instances (with no LLM client) for prompt inspection.

    Uses the same agent classes as the evaluator so build_prompt() produces
    the exact prompt the LLM would receive.
    """
    from eval_utils.agents.robust_all import RobustAllAgent
    from eval_utils.agents.robust_cot import RobustCoTAgent
    from eval_utils.agents.robust_naive import RobustNaiveAgent

    agent_type = config.agent.get("type", "robust_all")

    agents = []
    for agent_idx in range(num_agents):
        pb = create_prompt_builder(config.agent)
        instruction = env.get_instruction_prompt(agent_idx=agent_idx)
        pb.update_instruction_prompt(instruction)
        client_cfg = config.clients[agent_idx]

        # client_factory returns None -> agent.client = None (no LLM calls)
        client_factory = lambda: None

        if agent_type == "robust_all":
            agent = RobustAllAgent(client_factory, pb, config=config, client_config=client_cfg)
        elif agent_type == "robust_cot":
            agent = RobustCoTAgent(client_factory, pb, config=config, client_config=client_cfg)
        elif agent_type == "robust_naive":
            agent = RobustNaiveAgent(client_factory, pb)
        else:
            agent = RobustAllAgent(client_factory, pb, config=config, client_config=client_cfg)

        agent.agent_id = agent_idx
        agents.append(agent)
    return agents


# ── Main loop ────────────────────────────────────────────────────────────────


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(config: DictConfig):
    # print(config)
    original_cwd = get_original_cwd()
    os.chdir(original_cwd)

    seed = config.get("EVAL_SEED", 9999)
    num_agents = config.alem.get("num_agents", 3)
    max_steps = config.eval.max_steps_per_episode
    agent_type = config.agent.get("type", "robust_all")
    show_full_prompt = config.get("show_full_prompt", True)

    print_separator()
    print("  Alem-Coop Human Play Mode")
    print(f"  Env:    {config.get('ENV_NAME', 'Alem-Coop-Symbolic')}")
    print(f"  Agents: {num_agents}")
    print(f"  Coord:  {config.alem.get('coordination_difficulty', 'none')}")
    print(f"  Seed:   {seed}")
    print(f"  Max steps: {max_steps}")
    print(f"  Prompt style: {agent_type} (show_full_prompt={show_full_prompt})")
    print_separator()

    env = CraftaxEnv("default", config)

    # Create real agent instances so we can call build_prompt() and see the
    # exact prompt the LLM would receive (system prompt + history + instructions).
    agents = _create_agents(config, env, num_agents)

    print("\n--- System Prompt (Agent 0) ---")
    print(agents[0].prompt_builder.system_prompt)
    print("--- End System Prompt ---\n")

    print("  Shortcuts: w/a/s/d=move, e=Do, q=Noop, z=Sleep")
    print("  Type 'actions' to show full list, 'quit' to exit, 'help' for more.\n")

    obs_list, info = env.reset(seed=seed)
    prev_actions = [None] * num_agents
    total_rewards = [0.0] * num_agents
    step = 0

    while True:
        print_separator()
        print(f"  Step {step}/{max_steps}  |  Total reward: {[round(r, 2) for r in total_rewards]}")
        print_separator()

        # Show observation then collect action for each agent one at a time
        # (mirrors the LLM flow: each agent sees only its own obs independently)
        actions = []
        for agent_idx in range(num_agents):
            obs = obs_list[agent_idx]
            agent = agents[agent_idx]

            if num_agents > 1:
                print(f"\n--- Agent {agent_idx} Observation ---")
            else:
                print("\n--- Observation ---")

            if show_full_prompt:
                # Use the agent's own build_prompt() to get the exact messages
                # the LLM would receive, including system prompt, history,
                # and agent-specific instructions (CoT, comm, scratchpad).
                messages = agent.build_prompt(obs, prev_action=prev_actions[agent_idx])
                for msg in messages:
                    print(f"\n[{msg.role.upper()}]\n{msg.content}")
            else:
                # Default: show obs text only (same as before)
                long_ctx = obs.get("text", {}).get("long_term_context", "")
                short_ctx = obs.get("text", {}).get("short_term_context", "")
                if short_ctx:
                    print(short_ctx)
                print(long_ctx)
                if prev_actions[agent_idx]:
                    print(f"  (Previous action: {prev_actions[agent_idx]})")

            # Show pixel rendering if debug mode is on
            img = obs.get("image", None)
            if img is not None:
                img.show()

            print()
            action = get_human_action(agent_idx, num_agents)
            validated = env.check_action_validity(action, agent_idx)
            if validated != action:
                print(f"  -> Mapped to: {validated}")
            actions.append(validated)

        # Step
        obs_list, rewards, terminateds, truncateds, info = env.step(actions)
        dones = [t or tr for t, tr in zip(terminateds, truncateds)]

        for i in range(num_agents):
            total_rewards[i] += rewards[i]

        # Show rewards
        if any(r != 0 for r in rewards):
            print(f"\n  Rewards: {[round(r, 4) for r in rewards]}")

        prev_actions = list(actions)
        step += 1

        if any(dones):
            print_separator()
            print(f"  EPISODE ENDED at step {step}")
            print(f"  Total rewards: {[round(r, 2) for r in total_rewards]}")
            if "user_info" in info:
                ui = info["user_info"]
                # Print achievements if available
                achievements = ui.get("achievements", None)
                if achievements is not None:
                    import jax.numpy as jnp

                    n_achieved = int(jnp.sum(achievements > 0))
                    total = (
                        int(achievements.shape[0])
                        if hasattr(achievements, "shape")
                        else len(achievements)
                    )
                    print(f"  Achievements: {n_achieved}/{total}")
            print_separator()

            restart = input("  Play again? [Y/n] ").strip().lower()
            if restart in ("n", "no"):
                break

            seed += 1
            obs_list, info = env.reset(seed=seed)
            prev_actions = [None] * num_agents
            total_rewards = [0.0] * num_agents
            step = 0
            for agent in agents:
                agent.reset()
                # Re-set instruction prompt after reset clears the builder
                instruction = env.get_instruction_prompt(agent_idx=agent.agent_id)
                agent.prompt_builder.update_instruction_prompt(instruction)

    print("Done.")


if __name__ == "__main__":
    main()
