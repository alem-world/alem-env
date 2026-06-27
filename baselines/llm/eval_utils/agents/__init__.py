# Handle both relative and absolute imports
try:
    from ..client import create_llm_client
    from ..prompt_builder import create_prompt_builder
except ImportError:
    from eval_utils.client import create_llm_client
    from eval_utils.prompt_builder import create_prompt_builder

from .chain_of_thought import ChainOfThoughtAgent
from .custom import CustomAgent
from .dummy import DummyAgent
from .few_shot import FewShotAgent
from .naive import NaiveAgent
from .random import RandomAgent
from .robust_all import RobustAllAgent
from .robust_cot import RobustCoTAgent
from .robust_naive import RobustNaiveAgent


class AgentFactory:
    """Factory class for creating agents based on configuration."""

    # Agent types that use chain-of-thought reasoning and should enable
    # built-in thinking for models that support it (e.g. Qwen3.5, DeepSeek-R1).
    COT_AGENT_TYPES = {"cot", "robust_cot"}

    def __init__(self, config):
        self.config = config
        self._validate_clients_config()
        # Resolve once so the evaluator can read it without races
        self._resolved_enable_thinking = self._resolve_enable_thinking()

    # TODO(cleanup): remove _LEGACY_CLIENT_COMPAT and all references to it once
    # all callers have been updated to use clients.N.generate_kwargs directly.
    # Tracking: eval_gpt.sh / eval_claude.sh / eval_gemini.sh still pass
    # client.generate_kwargs.* on the command line as of 2026-04.
    _LEGACY_CLIENT_COMPAT = True

    _LEGACY_ROUTING_KEYS = frozenset({"client_name", "model_id", "base_url"})

    def _validate_clients_config(self):
        """Require index-aligned clients config for per-agent client mapping."""
        if "client" in self.config:
            # TODO(cleanup): remove this block when _LEGACY_CLIENT_COMPAT is removed.
            # config.client.* (routing keys + generate_kwargs) are forwarded to all agents
            # for backward compat with the baked-in Docker entrypoint.llm.sh that still
            # passes client.client_name/model_id/base_url in the old single-client format.
            import logging as _logging

            _logging.getLogger(__name__).warning(
                "[COMPAT] config.client.* detected; routing keys and generate_kwargs "
                "will be forwarded to all agents. Rebuild the Docker image with "
                "entrypoint.llm.sh (commit 1ef2f92) and remove _LEGACY_CLIENT_COMPAT."
            )

        clients = self.config.get("clients", None)
        if clients is None:
            raise ValueError(
                "Missing required config.clients list. Expected one client config per agent index."
            )
        if len(clients) == 0:
            raise ValueError("config.clients must contain at least one client config.")

        expected_agents = int(self.config.alem.get("num_agents", 0))
        if len(clients) < expected_agents:
            raise ValueError(
                f"config.clients length ({len(clients)}) must be >= "
                f"alem.num_agents ({expected_agents}). "
                f"The Docker entrypoint always provides 3 client slots; "
                f"extra slots beyond num_agents are ignored."
            )

    def _get_client_config_for_agent(self, agent_idx):
        if agent_idx is None:
            raise ValueError("agent_idx is required to map clients[agent_idx] to each agent.")
        if agent_idx < 0 or agent_idx >= len(self.config.clients):
            raise ValueError(
                f"agent_idx {agent_idx} is out of range for config.clients length {len(self.config.clients)}."
            )
        client_cfg = self.config.clients[agent_idx]
        # TODO(cleanup): remove this block when _LEGACY_CLIENT_COMPAT is removed.
        # Propagate config.client.* (routing keys + generate_kwargs) to each agent.
        # Routing keys (client_name, model_id, base_url) override when non-null — used by
        # the baked-in Docker entrypoint.llm.sh that passes client.* in the old format.
        # generate_kwargs are merged (shared first, per-agent wins).
        if self._LEGACY_CLIENT_COMPAT and "client" in self.config:
            from omegaconf import OmegaConf

            legacy = self.config.client
            overrides = {}
            for key in self._LEGACY_ROUTING_KEYS:
                val = legacy.get(key, None)
                if val is not None:
                    overrides[key] = val
            shared_gkw = (
                OmegaConf.to_container(legacy.get("generate_kwargs", {}), resolve=True) or {}
            )
            per_agent_gkw = (
                OmegaConf.to_container(client_cfg.get("generate_kwargs", {}), resolve=True) or {}
            )
            overrides["generate_kwargs"] = {**shared_gkw, **per_agent_gkw}  # per-agent wins
            client_cfg = OmegaConf.merge(client_cfg, overrides)
        return client_cfg

    def _resolve_enable_thinking(self):
        """Resolve the enable_thinking flag for the LLM client.

        Only enabled when explicitly set via ``agent.reasoning: true`` in
        config. This is for models that produce a separate .reasoning field
        (e.g. Qwen3 on vLLM with --reasoning-parser). GPT and Claude models
        do not have this field, so thinking mode should not be used with them.
        """
        return bool(getattr(self.config.agent, "reasoning", False))

    def create_agent(self, agent_idx=None):
        agent_type = self.config.agent.type

        if agent_type == "random":
            seed = self.config.get("seed", None)
            client_factory = lambda: None
            prompt_builder = create_prompt_builder(self.config.agent)
            return RandomAgent(client_factory, prompt_builder, seed=seed)

        # Inject enable_thinking into client config so the LLM client
        # can pass it to the server (e.g. vLLM's enable_thinking flag).
        from omegaconf import OmegaConf

        client_config = OmegaConf.to_container(
            self._get_client_config_for_agent(agent_idx), resolve=True
        )
        assert isinstance(client_config, dict), (
            f"Expected client config to be a dict, got {type(client_config).__name__}"
        )
        enable_thinking = self._resolve_enable_thinking()
        client_config["enable_thinking"] = enable_thinking
        client_config = OmegaConf.create(client_config)

        client_factory = create_llm_client(client_config)
        prompt_builder = create_prompt_builder(self.config.agent)

        if agent_type == "naive":
            return NaiveAgent(client_factory, prompt_builder)
        elif agent_type == "cot":
            return ChainOfThoughtAgent(client_factory, prompt_builder, config=self.config)
        elif agent_type == "dummy":
            return DummyAgent(client_factory, prompt_builder)
        elif agent_type == "custom":
            return CustomAgent(client_factory, prompt_builder)
        elif agent_type == "few_shot":
            return FewShotAgent(client_factory, prompt_builder, self.config.agent.max_icl_history)
        elif agent_type == "robust_naive":
            return RobustNaiveAgent(client_factory, prompt_builder)
        elif agent_type == "robust_cot":
            return RobustCoTAgent(
                client_factory,
                prompt_builder,
                config=self.config,
                client_config=client_config,
            )
        elif agent_type == "robust_all":
            return RobustAllAgent(
                client_factory,
                prompt_builder,
                config=self.config,
                client_config=client_config,
            )
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
