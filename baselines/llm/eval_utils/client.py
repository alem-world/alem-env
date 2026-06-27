"""
Based on https://github.com/balrog-ai/BALROG/blob/main/balrog/client.py
"""

import base64
import copy
import logging
import os
import time
from collections import namedtuple
from io import BytesIO

# Optional imports with fallbacks
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


LLMResponse = namedtuple(
    "LLMResponse",
    [
        "model_id",
        "completion",
        "stop_reason",
        "input_tokens",
        "output_tokens",
        "reasoning",
        "reasoning_tokens",
    ],
    defaults=(None, 0),
)

httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def process_image_openai(image):
    """Process an image for OpenAI API by converting it to base64."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
    }


def process_image_claude(image):
    """Process an image for Anthropic's Claude API by converting it to base64."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": base64_image},
    }


def _classify_chat_completion_incomplete(
    stop_reason, completion_text, reasoning_tokens, output_tokens
):
    """Map Chat Completions finish reasons to a coarse incomplete-response label."""
    if stop_reason == "length":
        if not (completion_text or "").strip() and reasoning_tokens >= output_tokens > 0:
            return "max_completion_tokens_during_reasoning"
        return "max_completion_tokens"
    if stop_reason == "content_filter":
        return "content_filter"
    if stop_reason in ("tool_calls", "function_call"):
        return str(stop_reason)
    return None if stop_reason in (None, "stop") else str(stop_reason)


class LLMClientWrapper:
    """Base class for LLM client wrappers."""

    def __init__(self, client_config):
        self.client_name = client_config.client_name
        self.model_id = client_config.model_id
        self.base_url = client_config.base_url
        self.timeout = client_config.timeout
        self.client_kwargs = {**client_config.generate_kwargs}
        self.max_retries = client_config.max_retries
        self.delay = client_config.delay
        self.alternate_roles = client_config.alternate_roles
        # enable_thinking controls whether the model produces <think> blocks.
        # Requires --reasoning-parser <model> on the vLLM server.
        # False = suppress thinking (explicit CoT in response); True = internal thinking.
        self.enable_thinking = getattr(client_config, "enable_thinking", False)

    def generate(self, messages):
        raise NotImplementedError("This method should be overridden by subclasses")

    def _status_code_from_exception(self, exc):
        status_code = getattr(exc, "status_code", None)
        if status_code is None:
            response = getattr(exc, "response", None)
            status_code = getattr(response, "status_code", None)
        return status_code

    def _is_non_retryable_exception(self, exc):
        status_code = self._status_code_from_exception(exc)
        if (
            isinstance(status_code, int)
            and 400 <= status_code < 500
            and status_code not in (408, 429)
        ):
            return True

        message = str(exc).lower()
        context_length_markers = (
            "context length",
            "maximum context length",
            "maximum input length",
            "input_tokens",
        )
        return any(marker in message for marker in context_length_markers)

    def execute_with_retries(self, func, *args, **kwargs):
        retries = 0
        last_exception = None
        while retries < self.max_retries:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if self._is_non_retryable_exception(e):
                    logger.error(f"Non-retryable error during {func.__name__}: {e}")
                    raise
                last_exception = e
                retries += 1
                logger.error(
                    f"Retryable error during {func.__name__}: {e}. Retry {retries}/{self.max_retries}"
                )
                sleep_time = self.delay * (2 ** (retries - 1))
                time.sleep(sleep_time)
        raise Exception(
            f"Failed to execute {func.__name__} after {self.max_retries} retries, last error: {last_exception}"
        )

    def generate_with_validation(self, messages, validate_fn, error_message, max_parse_retries=2):
        """Generate a response, retrying with feedback when validation fails.

        Unlike execute_with_retries (which handles transient API errors), this
        method handles *semantic* failures: the API returned a response but the
        content could not be parsed into a valid action.

        Args:
            messages: List of Message objects for the initial prompt.
            validate_fn: Callable(LLMResponse) -> extracted value or None.
                Returns the extracted/validated value on success, None to retry.
            error_message: User-feedback string appended on each retry.
            max_parse_retries: Maximum re-prompt attempts (default: 2).

        Returns:
            Tuple of (first_response, last_response, extracted_value, retries_used).
            first_response is always the initial generation (useful for CoT reasoning).
            If all retries fail, extracted_value will be None.
        """
        from .prompt_builder import Message  # lazy import to avoid circular deps

        first_response = self.generate(messages)
        extracted = validate_fn(first_response)

        retries = 0
        last_response = first_response
        while extracted is None and retries < max_parse_retries:
            retries += 1
            retry_messages = copy.deepcopy(messages)
            retry_messages.append(Message(role="assistant", content=last_response.completion))
            retry_messages.append(Message(role="user", content=error_message))
            last_response = self.generate(retry_messages)
            extracted = validate_fn(last_response)

        return first_response, last_response, extracted, retries


class OpenAIWrapper(LLMClientWrapper):
    """Wrapper for interacting with the OpenAI API (also handles vLLM, NVIDIA, XAI)."""

    def __init__(self, client_config):
        if OpenAI is None:
            raise ImportError("openai package is required. Install with: pip install openai")
        super().__init__(client_config)
        self._initialized = False

    def _initialize_client(self):
        if not self._initialized:
            if self.client_name.lower() == "vllm":
                self.client = OpenAI(api_key="EMPTY", base_url=self.base_url, timeout=self.timeout)
            elif self.client_name.lower() in ("nvidia", "xai"):
                if not self.base_url or not self.base_url.strip():
                    raise ValueError("base_url must be provided when using NVIDIA or XAI client")
                api_key = (
                    os.environ.get("NVIDIA_API_KEY")
                    if self.client_name.lower() == "nvidia"
                    else None
                )
                self.client = OpenAI(api_key=api_key, base_url=self.base_url, timeout=self.timeout)
            elif self.client_name.lower() == "openai":
                self.client = OpenAI(timeout=self.timeout)
            self._initialized = True

    def convert_messages(self, messages):
        converted_messages = []
        for msg in messages:
            new_content = [{"type": "text", "text": msg.content}]
            if msg.attachment is not None:
                new_content.append(process_image_openai(msg.attachment))
            if (
                self.alternate_roles
                and converted_messages
                and converted_messages[-1]["role"] == msg.role
            ):
                converted_messages[-1]["content"].extend(new_content)
            else:
                converted_messages.append({"role": msg.role, "content": new_content})
        return converted_messages

    def generate(self, messages):
        self._initialize_client()
        converted_messages = self.convert_messages(messages)

        def api_call():
            client_name = self.client_name.lower()
            max_tokens = self.client_kwargs.get("max_tokens", 2048)
            max_completion_tokens = self.client_kwargs.get("max_completion_tokens", max_tokens)
            api_kwargs = {
                "messages": converted_messages,
                "model": self.model_id,
            }

            # The real OpenAI API uses max_completion_tokens; most compatible
            # backends still expect max_tokens.
            if client_name == "openai":
                api_kwargs["max_completion_tokens"] = max_completion_tokens
            else:
                api_kwargs["max_tokens"] = max_tokens

            temperature = self.client_kwargs.get("temperature")
            if temperature is not None:
                api_kwargs["temperature"] = temperature
            top_p = self.client_kwargs.get("top_p")
            if top_p is not None:
                api_kwargs["top_p"] = top_p
            seed = self.client_kwargs.get("seed")
            if seed is not None:
                api_kwargs["seed"] = int(seed)
            reasoning_effort = self.client_kwargs.get("reasoning_effort")
            if reasoning_effort is not None and client_name == "openai":
                api_kwargs["reasoning_effort"] = reasoning_effort
            # Pass enable_thinking to vLLM when a --reasoning-parser is active.
            # Only for vLLM — the real OpenAI API doesn't support extra_body.
            if client_name == "vllm":
                chat_template_kwargs = {"enable_thinking": self.enable_thinking}
                # Qwen 3.6 can preserve historical thinking traces when opted in
                # by a model preset via client.generate_kwargs.preserve_thinking.
                preserve_thinking = self.client_kwargs.get("preserve_thinking")
                if preserve_thinking is not None:
                    chat_template_kwargs["preserve_thinking"] = preserve_thinking
                extra_body = {"chat_template_kwargs": chat_template_kwargs}
                # Gemma reasoning parsing in some vLLM versions requires special
                # tokens to be preserved so channel markers can be parsed.
                if self.enable_thinking and "gemma" in self.model_id.lower():
                    extra_body["skip_special_tokens"] = False
                api_kwargs["extra_body"] = extra_body
            logger.debug(f"API call kwargs for model {self.model_id}: {api_kwargs}")
            response = self.client.chat.completions.create(**api_kwargs)

            if response is None:
                raise RuntimeError("LLM response is None")
            if not getattr(response, "choices", None):
                raise RuntimeError(f"Missing choices in response: {response!r}")

            choice = response.choices[0]
            message = getattr(choice, "message", None)
            if message is None:
                raise RuntimeError(f"Missing message in response: {response!r}")

            completion_text = getattr(message, "content", None)
            if completion_text is None:
                # Responses where reasoning consumed the full output budget
                # (finish_reason='length') or where the model completed but
                # emitted only a reasoning trace with no text content
                # (finish_reason='stop', seen with Qwen3.5) are both
                # deterministic — retrying the same prompt wastes tokens.
                # Pass through with empty completion so the evaluator's
                # consecutive-length early-stop can trigger.
                if getattr(choice, "finish_reason", None) in ("length", "stop"):
                    completion_text = ""
                else:
                    raise RuntimeError(f"Missing content in response: {response!r}")

            return response, completion_text

        response, completion_text = self.execute_with_retries(api_call)

        # Extract reasoning from the response if the model produced it.
        # Different models/vLLM versions use different field names:
        #   - Gemma 4: reasoning_content  (direct attribute)
        #   - Qwen 3.5: reasoning         (direct attribute)
        # The OpenAI SDK may also store non-standard fields in model_extra (Pydantic v2).
        choice = response.choices[0]
        reasoning_source = None
        reasoning_content = getattr(choice.message, "reasoning_content", None)
        if reasoning_content is not None:
            reasoning_source = "message.reasoning_content"
        if reasoning_content is None:
            reasoning_content = getattr(choice.message, "reasoning", None)
            if reasoning_content is not None:
                reasoning_source = "message.reasoning"
        if reasoning_content is None:
            extras = getattr(choice.message, "model_extra", None) or {}
            if extras.get("reasoning_content") is not None:
                reasoning_content = extras["reasoning_content"]
                reasoning_source = "message.model_extra.reasoning_content"
            elif extras.get("reasoning") is not None:
                reasoning_content = extras["reasoning"]
                reasoning_source = "message.model_extra.reasoning"

        logger.info(
            "Reasoning field source for model %s: %s",
            self.model_id,
            reasoning_source or "none",
        )

        completion_details = getattr(response.usage, "completion_tokens_details", None)
        reasoning_tokens = 0
        if completion_details is not None:
            reasoning_tokens = getattr(completion_details, "reasoning_tokens", 0) or 0

        logger.info(
            "Reasoning tokens for model %s: %s",
            self.model_id,
            reasoning_tokens,
        )

        stop_reason = choice.finish_reason
        incomplete_reason = _classify_chat_completion_incomplete(
            stop_reason=stop_reason,
            completion_text=completion_text,
            reasoning_tokens=reasoning_tokens,
            output_tokens=response.usage.completion_tokens,
        )
        if incomplete_reason is not None:
            logger.warning(
                "Incomplete ChatCompletion for model %s: finish_reason=%s classified_reason=%s "
                "output_tokens=%s reasoning_tokens=%s completion_chars=%s",
                self.model_id,
                stop_reason,
                incomplete_reason,
                response.usage.completion_tokens,
                reasoning_tokens,
                len((completion_text or "").strip()),
            )

        return LLMResponse(
            model_id=self.model_id,
            completion=completion_text.strip(),
            stop_reason=stop_reason,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            reasoning=reasoning_content,
            reasoning_tokens=reasoning_tokens,
        )


class GoogleGenerativeAIWrapper(LLMClientWrapper):
    """Wrapper for interacting with Google's Generative AI API."""

    def __init__(self, client_config):
        if genai is None or types is None:
            raise ImportError(
                "google-genai package is required. Install with: pip install google-genai"
            )
        super().__init__(client_config)
        self._initialized = False

    def _initialize_client(self):
        if not self._initialized:
            self.client = genai.Client()
            self.model = None
            client_kwargs = {
                "max_output_tokens": self.client_kwargs.get("max_tokens", 1024),
            }
            temperature = self.client_kwargs.get("temperature")
            if temperature is not None:
                client_kwargs["temperature"] = temperature

            # Keep Gemini thinking config simple:
            # - always include thought summaries
            # - if thinking_level is provided, use it
            # - else if thinking_budget is provided, use it
            thinking_config_kwargs = {"include_thoughts": True}
            # thinking level is gemini 3.0 and above
            thinking_level = self.client_kwargs.get("thinking_level")
            # thinking budget is gemini 2.5
            thinking_budget = self.client_kwargs.get("thinking_budget")
            if thinking_level is not None:
                thinking_config_kwargs["thinking_level"] = thinking_level
            elif thinking_budget is not None:
                thinking_config_kwargs["thinking_budget"] = thinking_budget

            self.generation_config = types.GenerateContentConfig(
                **client_kwargs,
                thinking_config=types.ThinkingConfig(**thinking_config_kwargs),
            )

            logger.debug(f"Gemini Model {self.model_id} with config: {self.generation_config}")

            self._initialized = True

    def convert_messages(self, messages):
        """Convert messages to the format expected by the new Google GenAI SDK."""
        converted_messages = []
        for msg in messages:
            parts = []
            role = msg.role
            if role == "assistant":
                role = "model"
            elif role == "system":
                role = "user"
            if msg.content:
                parts.append(types.Part(text=msg.content))
            if msg.attachment is not None:
                parts.append(types.Part(image=msg.attachment))
            converted_messages.append(types.Content(role=role, parts=parts))
        return converted_messages

    def extract_completion(self, response):
        """Extract the completion text (answer) from the API response.

        This concatenates all non-thinking parts from the first candidate.
        """
        if not response:
            raise Exception("Response is None, cannot extract completion.")
        candidates = getattr(response, "candidates", [])
        if not candidates:
            raise Exception("No candidates found in the response.")
        candidate = candidates[0]
        content = getattr(candidate, "content", None)
        if not content:
            raise Exception("No content found in the candidate.")
        content_parts = getattr(content, "parts", [])
        if not content_parts:
            raise Exception("No content parts found in the candidate.")

        answer_chunks = []
        for part in content_parts:
            text = getattr(part, "text", None)
            if not text:
                continue
            if getattr(part, "thought", False):
                continue
            answer_chunks.append(text)

        if not answer_chunks:
            text = getattr(response, "text", None)
            if callable(text):
                text = text()
            if isinstance(text, str) and text.strip():
                return text.strip()
            raise Exception("No non-thinking text found in the content parts.")

        return "".join(answer_chunks).strip()

    def extract_reasoning(self, response):
        """Extract Gemini thought parts, if the model returns them."""
        if not response:
            return None

        candidates = getattr(response, "candidates", [])
        if not candidates:
            return None

        content = getattr(candidates[0], "content", None)
        if not content:
            return None

        reasoning_chunks = []
        for part in getattr(content, "parts", []) or []:
            if not getattr(part, "thought", False):
                continue
            text = getattr(part, "text", None)
            if isinstance(text, str) and text.strip():
                reasoning_chunks.append(text.strip())

        if not reasoning_chunks:
            return None
        return "\n".join(reasoning_chunks)

    def _extract_reasoning_token_count(self, response):
        usage = getattr(response, "usage_metadata", None) if response else None
        if usage is None:
            return 0

        for attr in (
            "thoughts_token_count",
            "thought_token_count",
            "thinking_token_count",
            "reasoning_token_count",
        ):
            val = getattr(usage, attr, None)
            if isinstance(val, int):
                return val
        return 0

    def generate(self, messages):
        self._initialize_client()
        converted_messages = self.convert_messages(messages)

        def api_call():
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=converted_messages,
                config=self.generation_config,
            )
            completion = self.extract_completion(response)
            reasoning = self.extract_reasoning(response)
            return response, completion, reasoning

        try:
            response, completion, reasoning = self.execute_with_retries(api_call)
            usage = getattr(response, "usage_metadata", None) if response else None
            prompt_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
            candidate_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0
            reasoning_tokens = self._extract_reasoning_token_count(response)
            output_tokens = (candidate_tokens or 0) + (reasoning_tokens or 0)

            if not completion or completion.strip() == "":
                logger.warning(f"Gemini returned an empty completion for model {self.model_id}.")
                return LLMResponse(
                    model_id=self.model_id,
                    completion="",
                    stop_reason="empty_response",
                    input_tokens=prompt_tokens or 0,
                    output_tokens=output_tokens,
                    reasoning=reasoning,
                    reasoning_tokens=reasoning_tokens,
                )
            else:
                return LLMResponse(
                    model_id=self.model_id,
                    completion=completion,
                    stop_reason=(
                        getattr(response.candidates[0], "finish_reason", None)
                        if response and getattr(response, "candidates", [])
                        else "unknown"
                    ),
                    input_tokens=prompt_tokens or 0,
                    output_tokens=output_tokens,
                    reasoning=reasoning,
                    reasoning_tokens=reasoning_tokens,
                )
        except Exception as e:
            logger.error(
                f"API call failed after {self.max_retries} retries: {e}. Returning empty completion."
            )
            return LLMResponse(
                model_id=self.model_id,
                completion="",
                stop_reason="error_max_retries",
                input_tokens=0,
                output_tokens=0,
                reasoning=None,
                reasoning_tokens=0,
            )


class ClaudeWrapper(LLMClientWrapper):
    """Wrapper for interacting with Anthropic's Claude API."""

    def __init__(self, client_config):
        if Anthropic is None:
            raise ImportError("anthropic package is required. Install with: pip install anthropic")
        super().__init__(client_config)
        self._initialized = False

    def _initialize_client(self):
        if not self._initialized:
            self.client = Anthropic()
            self._initialized = True

    def convert_messages(self, messages):
        converted_messages = []
        for msg in messages:
            converted_messages.append(
                {"role": msg.role, "content": [{"type": "text", "text": msg.content}]}
            )
            if converted_messages[-1]["role"] == "system":
                converted_messages[-1]["role"] = "user"
                converted_messages.append({"role": "assistant", "content": "I'm ready!"})
            if msg.attachment is not None:
                converted_messages[-1]["content"].append(process_image_claude(msg.attachment))
        return converted_messages

    def generate(self, messages):
        self._initialize_client()
        converted_messages = self.convert_messages(messages)

        def api_call():
            api_kwargs = {
                "messages": converted_messages,
                "model": self.model_id,
                "max_tokens": self.client_kwargs.get("max_tokens", 2048),
            }
            temperature = self.client_kwargs.get("temperature")
            if temperature is not None:
                api_kwargs["temperature"] = temperature
            return self.client.messages.create(**api_kwargs)

        response = self.execute_with_retries(api_call)

        return LLMResponse(
            model_id=self.model_id,
            completion=response.content[0].text.strip(),
            stop_reason=response.stop_reason,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            reasoning=None,
        )


def create_llm_client(client_config):
    """Factory function to create the appropriate LLM client based on the client name."""

    def client_factory():
        client_name_lower = client_config.client_name.lower()
        if (
            "openai" in client_name_lower
            or "vllm" in client_name_lower
            or "nvidia" in client_name_lower
            or "xai" in client_name_lower
        ):
            return OpenAIWrapper(client_config)
        elif "gemini" in client_name_lower:
            return GoogleGenerativeAIWrapper(client_config)
        elif "claude" in client_name_lower or "anthropic" in client_name_lower:
            return ClaudeWrapper(client_config)
        else:
            raise ValueError(f"Unsupported client name: {client_config.client_name}")

    return client_factory
