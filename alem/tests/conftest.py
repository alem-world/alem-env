"""Shared pytest fixtures for the ALEM test suite.

Reduce test memory uses for github actions. Probably makes things slower.
"""

import gc

import jax
import pytest


@pytest.fixture(autouse=True, scope="function")
def _clear_jax_caches():
    yield
    jax.clear_caches()
    gc.collect()
