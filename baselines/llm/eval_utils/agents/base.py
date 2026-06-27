import copy

try:
    from ..prompt_builder import Message
except ImportError:
    from eval_utils.prompt_builder import Message


class _ClientProxy:
    """Wraps an LLM client to capture the last messages sent to generate().

    Both ``generate()`` and ``generate_with_validation()`` are implemented here
    so prompt history is always captured and the retry logic does not depend on
    the wrapped client having ``generate_with_validation`` defined.
    """

    def __init__(self, client):
        self._client = client
        self.last_prompt_messages = None

    def _capture(self, messages):
        self.last_prompt_messages = [{"role": m.role, "content": m.content} for m in messages]

    def generate(self, messages):
        self._capture(messages)
        return self._client.generate(messages)

    def generate_with_validation(self, messages, validate_fn, error_message, max_parse_retries=2):
        """Generate a response, retrying with feedback when validation fails.

        Implemented on the proxy so it works regardless of whether the wrapped
        client exposes this method. Uses self._client.generate() directly so
        only the initial prompt is captured in last_prompt_messages.
        """
        self._capture(messages)
        first_response = self._client.generate(messages)
        extracted = validate_fn(first_response)

        retries = 0
        last_response = first_response
        while extracted is None and retries < max_parse_retries:
            retries += 1
            retry_messages = copy.deepcopy(messages)
            retry_messages.append(Message(role="assistant", content=last_response.completion))
            retry_messages.append(Message(role="user", content=error_message))
            last_response = self._client.generate(retry_messages)
            extracted = validate_fn(last_response)

        return first_response, last_response, extracted, retries

    def __getattr__(self, name):
        return getattr(self._client, name)


class BaseAgent:
    """Base class for agents using prompt-based interactions."""

    def __init__(self, client_factory, prompt_builder, agent_id=None):
        raw_client = client_factory()
        self.client = _ClientProxy(raw_client) if raw_client is not None else None
        self.prompt_builder = prompt_builder
        self.agent_id = agent_id

    def act(self, obs):
        raise NotImplementedError

    def update_prompt(self, observation, action):
        self.prompt_builder.update_observation(observation)
        self.prompt_builder.update_action(action)

    def reset(self):
        self.prompt_builder.reset()
