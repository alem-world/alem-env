"""ALEM: a JAX environment for open-ended multi-agent coordination."""

from alem._version import LLM_STACK_VERSION, __version__
from alem.alem_env import make_alem_env_from_name

__all__ = ["LLM_STACK_VERSION", "make_alem_env_from_name", "__version__"]
