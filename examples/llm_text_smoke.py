"""Smoke-test the 3-agent Alem text interface without calling an LLM."""

import argparse
import os
import textwrap

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")


def _excerpt(text, width=100, lines=8):
    wrapped = textwrap.wrap(" ".join(text.split()), width=width)
    return "\n".join(wrapped[:lines])


def main():
    parser = argparse.ArgumentParser(description="Preview Alem's 3-agent LLM text interface")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--coord", choices=["none", "easy", "medium", "hard"], default="easy")
    parser.add_argument(
        "--ascii", action="store_true", help="Render the local view as an ASCII map"
    )
    parser.add_argument(
        "--show-affordances", action="store_true", help="Append legal actions to each observation"
    )
    args = parser.parse_args()

    import jax

    from alem.llm.alem_language_wrapper import (
        ACTIONS,
        AlemLanguageWrapper,
        get_instruction_prompt,
        make_alem_env,
    )

    env = make_alem_env(
        {
            "ENV_NAME": "Alem-Coop-Symbolic",
            "num_agents": 3,
            "coordination_difficulty": args.coord,
            "soft_specialization": True,
            "shared_reward": False,
            "specialist_efficiency": 1.0,
            "non_specialist_efficiency": 0.2,
            "randomize_alpha": False,
            "max_timesteps": 10000,
            "god_mode": False,
        }
    )
    wrapper = AlemLanguageWrapper(
        env,
        env.default_params,
        prompt_mode="specific_collaborative",
        show_affordances=args.show_affordances,
        use_ascii=args.ascii,
        debug=False,
    )

    prompt = get_instruction_prompt(
        coordination_enabled=args.coord != "none",
        num_agents=3,
        agent_id=0,
        role="warrior",
        include_all_actions=False,
        progressive_disclosure=True,
        current_level=0,
        prompt_mode="specific_collaborative",
    )
    obs_list, state, rng = wrapper.reset(jax.random.PRNGKey(args.seed))

    print(f"Alem 3-agent text smoke | coord={args.coord} | actions={len(ACTIONS)}")
    print("\nSystem prompt excerpt:")
    print(_excerpt(prompt))

    for agent_idx, obs in enumerate(obs_list):
        text = obs["text"]
        print(f"\nAgent {agent_idx} long-term observation:")
        print(_excerpt(text["long_term_context"], lines=5))
        print(f"\nAgent {agent_idx} short-term observation:")
        print(_excerpt(text["short_term_context"], lines=5))

    print("\nAction parser examples:")
    for action in ["<action>Move North</action>", "ACTION: Do", "Give to Agent 2"]:
        print(f"{action!r} -> {wrapper.get_action_index(action, agent_idx=0)}")


if __name__ == "__main__":
    main()
