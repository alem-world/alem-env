"""Run a tiny 3-agent Alem LLM smoke test with an OpenAI-compatible API."""

import argparse
import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

ROLES = ("warrior", "forager", "miner")


def make_wrapper(coord):
    from alem.llm.alem_language_wrapper import AlemLanguageWrapper, make_alem_env

    env = make_alem_env(
        {
            "ENV_NAME": "Alem-Coop-Symbolic",
            "num_agents": 3,
            "coordination_difficulty": coord,
            "soft_specialization": True,
            "shared_reward": False,
            "specialist_efficiency": 1.0,
            "non_specialist_efficiency": 0.2,
            "randomize_alpha": False,
            "max_timesteps": 10000,
            "god_mode": False,
        }
    )
    return AlemLanguageWrapper(
        env,
        env.default_params,
        prompt_mode="specific_collaborative",
        show_affordances=True,
        debug=False,
    )


def build_messages(obs, coord, agent_idx, current_level):
    from alem.llm.alem_language_wrapper import get_instruction_prompt

    prompt = get_instruction_prompt(
        coordination_enabled=coord != "none",
        num_agents=3,
        agent_id=agent_idx,
        role=ROLES[agent_idx % len(ROLES)],
        include_all_actions=False,
        progressive_disclosure=True,
        current_level=current_level,
        prompt_mode="specific_collaborative",
    )
    text = obs["text"]
    observation = (
        f"{text['long_term_context']}\n\n"
        f"{text['short_term_context']}\n\n"
        "Choose exactly one available action. "
        "Return only: <action>ACTION_NAME</action>"
    )
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": observation},
    ]


def parse_args():
    parser = argparse.ArgumentParser(description="Run one tiny Alem 3-agent LLM smoke test")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument(
        "--base-url", default=None, help="OpenAI-compatible base URL, e.g. vLLM /v1"
    )
    parser.add_argument(
        "--api-key", default=None, help="Defaults to OPENAI_API_KEY, or EMPTY for --base-url"
    )
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--coord", choices=["none", "easy", "medium", "hard"], default="easy")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=128)
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Install the OpenAI client first: pip install openai") from exc

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if args.base_url and not api_key:
        api_key = "EMPTY"
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY or pass --api-key.")

    import jax

    from alem.llm.action_parser import extract_action_multistrategy
    from alem.llm.alem_language_wrapper import ACTIONS

    client = OpenAI(api_key=api_key, base_url=args.base_url)
    wrapper = make_wrapper(args.coord)
    obs_list, state, rng = wrapper.reset(jax.random.PRNGKey(args.seed))

    print(f"Alem 3-agent LLM smoke | model={args.model} | coord={args.coord} | steps={args.steps}")

    for step in range(args.steps):
        current_level = int(state.player_level)
        completions = []
        parsed_actions = []

        for agent_idx, obs in enumerate(obs_list):
            response = client.chat.completions.create(
                model=args.model,
                messages=build_messages(obs, args.coord, agent_idx, current_level),
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            completion = response.choices[0].message.content or ""
            parsed = extract_action_multistrategy(completion, ACTIONS) or "Noop"
            completions.append(completion)
            parsed_actions.append(parsed)

        obs_list, state, rewards, dones, info, rng = wrapper.step(state, completions, rng)

        print(f"\nStep {step + 1}")
        for agent_idx, (raw, parsed, reward) in enumerate(
            zip(completions, parsed_actions, rewards)
        ):
            raw_one_line = " ".join(raw.split())
            print(
                f"  agent_{agent_idx}: {parsed} | reward={float(reward):+.3f} | raw={raw_one_line[:160]}"
            )

        if all(bool(done) for done in dones):
            print("Episode ended.")
            break


if __name__ == "__main__":
    main()
