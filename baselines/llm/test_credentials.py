"""Tests for credential failures: they must stop the run, not become Noops.

An authentication failure is not a transient API error. Retrying it cannot help,
and a run that continues past it burns compute on every remaining step while the
agent contributes nothing.
"""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from eval_utils import client as client_module  # noqa: E402
from eval_utils.client import GoogleGenerativeAIWrapper  # noqa: E402
from eval_utils.prompt_builder import Message  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402


class _AuthError(Exception):
    """Stands in for the SDK's 401 (invalid or missing key)."""

    status_code = 401


class _ServerError(Exception):
    """Stands in for a transient backend failure."""

    status_code = 503


def _client_config(client_name="gemini", **overrides):
    base = {
        "client_name": client_name,
        "model_id": "test-model",
        "base_url": None,
        "timeout": 60,
        "generate_kwargs": {"max_tokens": 128},
        "max_retries": 2,
        "delay": 0,
        "alternate_roles": False,
    }
    base.update(overrides)
    return OmegaConf.create(base)


def _gemini_wrapper_raising(exc):
    """A GoogleGenerativeAIWrapper whose every API call raises `exc`."""
    wrapper = GoogleGenerativeAIWrapper(_client_config())

    def _raise(**_kwargs):
        raise exc

    fake_client = SimpleNamespace(models=SimpleNamespace(generate_content=_raise))
    fake_genai = SimpleNamespace(Client=lambda: fake_client)
    return wrapper, mock.patch.object(client_module, "genai", fake_genai)


class TestAuthFailureStopsTheRun(unittest.TestCase):
    def test_auth_error_is_raised_not_returned_as_an_empty_completion(self):
        """A 401 must propagate.

        GoogleGenerativeAIWrapper.generate wraps everything in `except Exception`
        and returns completion="" on failure. The agent then fails to parse an
        action and defaults to Noop — so a run with a bad key does not stop, it
        Noops for the rest of every episode while the other agents keep paying
        for real API calls.
        """
        wrapper, patched = _gemini_wrapper_raising(_AuthError("invalid api key"))
        with patched:
            with self.assertRaises(Exception) as ctx:
                wrapper.generate([Message(role="user", content="ping")])
        self.assertNotIsInstance(ctx.exception, AssertionError)

    def test_transient_error_still_degrades_to_an_empty_completion(self):
        """The tolerance that exists for flaky backends must survive the fix.

        A 503 is retryable: after exhausting retries the wrapper returns an empty
        completion rather than killing the episode. Only non-retryable failures
        (auth, bad request) should propagate.
        """
        wrapper, patched = _gemini_wrapper_raising(_ServerError("upstream unavailable"))
        with patched:
            response = wrapper.generate([Message(role="user", content="ping")])
        self.assertEqual(response.completion, "")
        self.assertEqual(response.stop_reason, "error_max_retries")


if __name__ == "__main__":
    unittest.main()
