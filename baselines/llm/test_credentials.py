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
from eval_utils.agents import AgentFactory  # noqa: E402
from eval_utils.client import (  # noqa: E402
    ClaudeWrapper,
    CredentialError,
    GoogleGenerativeAIWrapper,
    OpenAIWrapper,
)
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


class _RecordingOpenAIClient:
    """Stands in for the OpenAI SDK client: records calls, optionally raises."""

    def __init__(self, exc=None):
        self.calls = []
        self._exc = exc
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._exc is not None:
            raise self._exc
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="pong"),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=1, completion_tokens=1, completion_tokens_details=None
            ),
        )


def _openai_wrapper(client_name="openai", exc=None):
    wrapper = OpenAIWrapper(_client_config(client_name=client_name, base_url="http://localhost/v1"))
    recorder = _RecordingOpenAIClient(exc=exc)
    wrapper.client = recorder
    wrapper._initialized = True
    return wrapper, recorder


class TestValidateCredentials(unittest.TestCase):
    def test_missing_key_fails_before_any_network_call(self):
        """Absent credentials are knowable for free — don't spend a request to learn it."""
        wrapper, recorder = _openai_wrapper("openai")
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(CredentialError) as ctx:
                wrapper.validate_credentials()
        self.assertIn("OPENAI_API_KEY", str(ctx.exception))
        self.assertEqual(recorder.calls, [])

    def test_local_backend_needs_no_key(self):
        """vLLM authenticates with the literal "EMPTY" — demanding a key would break it."""
        wrapper, recorder = _openai_wrapper("vllm")
        with mock.patch.dict(os.environ, {}, clear=True):
            wrapper.validate_credentials()
        self.assertEqual(len(recorder.calls), 1)

    def test_rejected_key_names_the_client_that_failed(self):
        """A key that exists but is refused can only be found by asking the backend."""
        wrapper, _ = _openai_wrapper("openai", exc=_AuthError("invalid api key"))
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-wrong"}, clear=True):
            with self.assertRaises(CredentialError) as ctx:
                wrapper.validate_credentials()
        message = str(ctx.exception)
        self.assertIn("openai", message)
        self.assertIn("test-model", message)

    def test_probe_is_one_token_and_leaves_the_config_untouched(self):
        """The check must be cheap, and must not become the run's settings."""
        wrapper, recorder = _openai_wrapper("vllm")
        before = dict(wrapper.client_kwargs)
        with mock.patch.dict(os.environ, {}, clear=True):
            wrapper.validate_credentials()
        self.assertEqual(recorder.calls[0]["max_tokens"], 1)
        self.assertEqual(dict(wrapper.client_kwargs), before)


class TestRequiredEnvVarsPerBackend(unittest.TestCase):
    def test_gemini_accepts_either_google_key(self):
        required = GoogleGenerativeAIWrapper(
            _client_config("gemini")
        ).required_credential_env_vars()
        self.assertEqual(set(required), {"GEMINI_API_KEY", "GOOGLE_API_KEY"})

    def test_claude_requires_the_anthropic_key(self):
        required = ClaudeWrapper(_client_config("claude")).required_credential_env_vars()
        self.assertEqual(tuple(required), ("ANTHROPIC_API_KEY",))


def _factory_config(*client_names):
    return OmegaConf.create(
        {
            "alem": {"num_agents": len(client_names)},
            "agent": {"type": "robust_all"},
            "clients": [
                {
                    "client_name": name,
                    "model_id": f"model-{idx}",
                    "base_url": None,
                    "timeout": 60,
                    "generate_kwargs": {"max_tokens": 128},
                    "max_retries": 1,
                    "delay": 0,
                    "alternate_roles": False,
                }
                for idx, name in enumerate(client_names)
            ],
        }
    )


class TestPreflightClients(unittest.TestCase):
    def test_every_broken_client_is_reported_at_once(self):
        """One run, one report. Fixing keys one failed run at a time is the waste."""
        factory = AgentFactory(_factory_config("gemini", "openai"))
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(CredentialError) as ctx:
                factory.preflight_clients()
        message = str(ctx.exception)
        self.assertIn("clients[0]", message)
        self.assertIn("clients[1]", message)
        self.assertIn("GEMINI_API_KEY", message)
        self.assertIn("OPENAI_API_KEY", message)

    def test_clientless_agents_are_not_preflighted(self):
        """The random agent never calls an API — it must not require a key."""
        config = _factory_config("openai")
        config.agent.type = "random"
        factory = AgentFactory(config)
        with mock.patch.dict(os.environ, {}, clear=True):
            factory.preflight_clients()

    def test_extra_client_slots_beyond_num_agents_are_ignored(self):
        """The Docker entrypoint always provides 3 slots; unused ones need no key."""
        config = _factory_config("vllm", "openai")
        config.alem.num_agents = 1
        factory = AgentFactory(config)
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(client_module.OpenAIWrapper, "generate", lambda self, m: None):
                factory.preflight_clients()


if __name__ == "__main__":
    unittest.main()
