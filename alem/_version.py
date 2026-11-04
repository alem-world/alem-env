"""Single source of truth for the Alem version.

Kept in its own module so that both ``alem/__init__.py`` and
``alem_coop/alem_state.py`` can read it: ``__init__`` imports ``alem_env``,
which imports ``alem_state``, so ``alem_state`` cannot import from the package
root without a cycle.
"""

__version__ = "0.2.1"

# The LLM-facing stack versions together and independently of the env package:
# the language wrapper (shipped in alem.llm) and the agent harnesses in
# baselines/ (repo-only, excluded from the distribution) change as one
# interface, on their own release cadence.
LLM_STACK_VERSION = "v0.1.1"
