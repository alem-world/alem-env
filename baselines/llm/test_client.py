"""Unit tests for OpenAIWrapper's request construction.

These cover what `generate()` actually puts on the wire, using a stand-in
client that records the kwargs instead of calling a backend.

Run with:
    python -m unittest baselines.llm.test_client
"""

import json
import unittest
from types import SimpleNamespace

from omegaconf import OmegaConf

from baselines.llm.eval_utils.client import OpenAIWrapper
from baselines.llm.eval_utils.prompt_builder import Message


def _client_config(client_name="vllm", **generate_kwargs):
    return OmegaConf.create(
        {
            "client_name": client_name,
            "model_id": "test-model",
            "base_url": "http://localhost:11434/v1",
            "timeout": 60,
            "generate_kwargs": {"max_tokens": 128, **generate_kwargs},
            "max_retries": 1,
            "delay": 0,
            "alternate_roles": False,
        }
    )


def _fake_response(content="<action>Noop</action>"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=1,
            completion_tokens=1,
            completion_tokens_details=None,
        ),
    )


class _RecordingClient:
    """Stands in for the OpenAI SDK client, recording the create() kwargs."""

    def __init__(self):
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return _fake_response()


def _generate_with(config):
    """Run one generate() against a recording client; return the api kwargs."""
    wrapper = OpenAIWrapper(config)
    recorder = _RecordingClient()
    wrapper.client = recorder
    wrapper._initialized = True
    wrapper.generate([Message(role="user", content="hi")])
    assert len(recorder.calls) == 1
    return recorder.calls[0]


class TestResponseFormatPassthrough(unittest.TestCase):
    def test_response_format_reaches_the_api_call(self):
        """A configured response_format must be sent, not silently dropped.

        client_kwargs is read key-by-key, so any option not explicitly copied
        into api_kwargs is discarded while the run still reports as configured.
        """
        response_format = {"type": "json_object"}
        kwargs = _generate_with(_client_config(response_format=response_format))
        self.assertEqual(kwargs.get("response_format"), response_format)

    def test_response_format_absent_when_not_configured(self):
        """Unset means omitted, not sent as null — as with every other optional kwarg here."""
        kwargs = _generate_with(_client_config())
        self.assertNotIn("response_format", kwargs)

    def test_response_format_from_yaml_config_is_json_serializable(self):
        """A config-supplied value arrives as a DictConfig, which the SDK cannot encode.

        `generate_kwargs` comes from OmegaConf, so `response_format` is a DictConfig
        rather than a dict. It compares equal to a dict — so an equality assertion
        passes — but json.dumps raises `Object of type DictConfig is not JSON
        serializable` inside the SDK, and the retry wrapper reports it as an API
        error. Forwarding the value is not enough; it has to be forwarded in a form
        the request encoder accepts.
        """
        kwargs = _generate_with(_client_config(response_format={"type": "json_object"}))
        json.dumps(kwargs["response_format"])
        self.assertEqual(kwargs["response_format"], {"type": "json_object"})


if __name__ == "__main__":
    unittest.main()
