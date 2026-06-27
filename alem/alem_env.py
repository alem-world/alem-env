from alem.alem_coop.alem_state import EnvParams, StaticEnvParams
from alem.alem_coop.envs.alem_pixels_env import AlemCoopPixelsEnv
from alem.alem_coop.envs.alem_symbolic_env import AlemCoopSymbolicEnv
from alem.alem_coop.envs.alem_symbolic_env_debug import AlemCoopSymbolicEnvDebug
from alem.alem_coop.envs.alem_symbolic_single_agent_env import AlemCoopSymbolicSingleAgentEnv


def make_alem_env_from_name(
    name: str,
    env_params: EnvParams | None = None,
    static_env_params: StaticEnvParams | None = None,
    compute_full_info: bool = True,
):
    """Create an alem environment from name.

    Args:
        name: Registered environment name.
        env_params: Optional episode and gameplay parameters.
        static_env_params: Optional parameters that determine static array shapes.
        compute_full_info: Whether steps should compute the full metrics dictionary. Logs extra metrics if true.

    Returns:
        The configured symbolic, pixel, debug, or single-agent environment.

    Raises:
        ValueError: If ``name`` is not a registered ALEM environment.
    """

    kw = dict(
        env_params=env_params,
        static_env_params=static_env_params,
        compute_full_info=compute_full_info,
    )

    if name == "Alem-Coop-Symbolic":
        return AlemCoopSymbolicEnv(**kw)
    # Single Agent is experimental!!
    elif name == "Alem-SingleAgent-Symbolic":
        return AlemCoopSymbolicSingleAgentEnv(**kw)
    elif name == "Alem-Coop-Symbolic-Debug":
        return AlemCoopSymbolicEnvDebug(**kw)
    elif name == "Alem-Coop-Pixels":
        return AlemCoopPixelsEnv(**kw)
    raise ValueError(f"Unknown alem environment: {name}")
