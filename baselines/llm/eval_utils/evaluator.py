# based on https://github.com/balrog-ai/BALROG/blob/b20b5e18e537aeb452cfa9f851e66b398e8d49f4/balrog/evaluator.py#L25

import base64
import csv
import gzip
import io
import json
import logging
import os
import pickle
import random
import re
import sys
import threading
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jax
import numpy as np

# Add project root, alem/, and baselines/llm/ to path
_llm_root = str(Path(__file__).parent.parent)
_project_root = str(Path(__file__).parent.parent.parent.parent)
_alem_root = os.path.join(_project_root, "alem")
for _p in (_llm_root, _project_root, _alem_root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import imageio
from omegaconf import OmegaConf
from tqdm import tqdm

try:
    from .agents.few_shot import FewShotAgent

    FEW_SHOT_AVAILABLE = True
except ImportError:
    try:
        from eval_utils.agents.few_shot import FewShotAgent

        FEW_SHOT_AVAILABLE = True
    except ImportError:
        FEW_SHOT_AVAILABLE = False

from alem.llm.alem_env import make_env

try:
    from .debug_visualiser import generate_debug_html, generate_step_log_txt
except ImportError:
    from eval_utils.debug_visualiser import generate_debug_html, generate_step_log_txt

logger = logging.getLogger(__name__)

_MAX_CONSECUTIVE_LENGTH_INCOMPLETES = 3
_PAID_API_CLIENT_MARKERS = ("openai", "anthropic", "claude", "gemini")
_PAID_API_MODEL_MARKERS = ("gpt-", "claude", "gemini")

# Surrogate characters (U+D800–U+DFFF) and null bytes are not valid in JSON
# strings. LLM thinking output occasionally contains them, causing json.loads
# to fail when reading the JSONL back. Strip them before serializing.
_INVALID_JSON_STR = re.compile(r"[\x00\ud800-\udfff]", re.UNICODE)


def _sanitize_str(s):
    """Remove characters that are invalid in JSON strings."""
    if isinstance(s, str):
        return _INVALID_JSON_STR.sub("\ufffd", s)
    return s


def _sanitize_record(obj):
    """Recursively sanitize all string values in a dict/list for JSON safety."""
    if isinstance(obj, dict):
        return {k: _sanitize_record(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_record(v) for v in obj]
    if isinstance(obj, str):
        return _sanitize_str(obj)
    return obj


def _safe_json_dumps(obj):
    """json.dumps with sanitization fallback for invalid string content."""
    try:
        return json.dumps(obj)
    except (ValueError, UnicodeEncodeError):
        return json.dumps(_sanitize_record(obj))


def _classify_incomplete_response(response):
    """Classify likely incomplete/truncated responses for logging and summaries."""
    stop_reason = getattr(response, "stop_reason", None)
    completion = getattr(response, "completion", None) or ""
    reasoning_tokens = getattr(response, "reasoning_tokens", 0) or 0
    output_tokens = getattr(response, "output_tokens", 0) or 0

    if stop_reason == "length":
        if not completion.strip() and reasoning_tokens >= output_tokens > 0:
            return "max_completion_tokens_during_reasoning"
        return "max_completion_tokens"
    if stop_reason == "content_filter":
        return "content_filter"
    if stop_reason in ("tool_calls", "function_call"):
        return str(stop_reason)
    return None if stop_reason in (None, "stop") else str(stop_reason)


def _should_early_stop_on_length(client_cfg):
    """Only stop early on repeated length truncation for paid hosted APIs.

    For self-hosted/local inference (e.g. vLLM), keep trying because there is
    no per-token external API spend to protect.
    """
    if client_cfg is None:
        return False

    client_name = str(getattr(client_cfg, "client_name", "") or "").strip().lower()
    model_id = str(getattr(client_cfg, "model_id", "") or "").strip().lower()

    # Explicitly treat vLLM as self-hosted/local.
    if "vllm" in client_name:
        return False

    if any(marker in client_name for marker in _PAID_API_CLIENT_MARKERS):
        return True
    return any(marker in model_id for marker in _PAID_API_MODEL_MARKERS)


def _save_episode_trajectory(
    output_dir,
    env_name,
    task,
    episode_idx,
    traj_obs,
    traj_actions,
    traj_rewards,
    traj_dones,
    traj_text_obs,
    traj_text_actions,
):
    """Save per-episode trajectory as .npz matching the RL eval format (baselines/utils.py).

    Numeric arrays match RL _run_eval_sequential exactly:
      obs       (T, num_agents, obs_dim) - raw symbolic obs (pre-step)
      actions   (T, num_agents)          - discrete action indices
      rewards   (T, num_agents)          - per-agent rewards
      dones     (T,)                     - episode-done flag
      timesteps (T,)                     - step indices

    Extra LLM-specific fields (load with allow_pickle=True):
      text_obs     (T, num_agents) - long-term text observation seen by the agent
      text_actions (T, num_agents) - canonical action name chosen by the agent
    """
    T = len(traj_obs)
    if T == 0 or any(x is None for x in traj_obs):
        return
    try:
        traj_dir = os.path.join(output_dir, env_name, task)
        Path(traj_dir).mkdir(exist_ok=True, parents=True)
        save_path = os.path.join(traj_dir, f"{task}_run_{episode_idx:02d}_trajectory.npz")
        np.savez_compressed(
            save_path,
            obs=np.stack(traj_obs),  # (T, num_agents, obs_dim)
            actions=np.stack(traj_actions),  # (T, num_agents)
            rewards=np.stack(traj_rewards),  # (T, num_agents)
            dones=np.array(traj_dones, dtype=bool),  # (T,)
            timesteps=np.arange(T, dtype=np.int32),  # (T,)
            text_obs=np.array(traj_text_obs, dtype=object),  # (T, num_agents)
            text_actions=np.array(traj_text_actions, dtype=object),  # (T, num_agents)
        )
        logger.info(f"Saved trajectory ({T} steps) to {save_path}")
    except Exception as e:
        logger.warning(f"Failed to save trajectory for episode {episode_idx}: {e}")


def _save_episode_states(
    output_dir,
    env_name,
    task,
    episode_idx,
    traj_states,
    static_env_params,
):
    """Save pre-step env states to a separate compressed pickle for exact replay.

    We keep this out of the main `.npz` because EnvState is a nested flax/JAX
    pytree, not a plain numeric array. The replay script can prefer this file
    whenever it exists, which avoids seed-based reconstruction entirely.
    """
    if not traj_states:
        return

    try:
        traj_dir = os.path.join(output_dir, env_name, task)
        Path(traj_dir).mkdir(exist_ok=True, parents=True)
        save_path = os.path.join(traj_dir, f"{task}_run_{episode_idx:02d}_states.pkl.gz")
        payload = {
            "states": traj_states,
            "static_env_params": static_env_params,
            "num_steps": len(traj_states),
        }
        with gzip.open(save_path, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Saved state bundle ({len(traj_states)} steps) to {save_path}")
    except Exception as e:
        logger.warning(f"Failed to save states for episode {episode_idx}: {e}")


class EvaluatorManager:
    """Manages evaluation of agents across multiple environments and tasks."""

    def __init__(self, config, original_cwd="", output_dir="."):
        self.config = config
        self.original_cwd = original_cwd
        self.output_dir = output_dir

        self.env_names = config.envs.names.split("-")
        self.env_evaluators = {}
        self.tasks = []
        for env_name in self.env_names:
            evaluator = Evaluator(
                env_name, config, original_cwd=original_cwd, output_dir=self.output_dir
            )
            self.env_evaluators[env_name] = evaluator
            for task in evaluator.tasks:
                for episode_idx in range(evaluator.num_episodes):
                    json_filename = os.path.join(
                        self.output_dir,
                        env_name,
                        task,
                        f"{task}_run_{episode_idx:02d}.json",
                    )
                    if os.path.exists(json_filename):
                        logging.info(
                            f"Skipping completed task: {env_name}, {task}, episode {episode_idx}"
                        )
                    else:
                        self.tasks.append((env_name, task, episode_idx))
        # Cap workers at actual task count — extra workers would sit idle.
        self.num_workers = min(config.eval.num_workers, max(len(self.tasks), 1))

    def run(self, agent_factory):
        # Only parallelize if there are multiple tasks to run concurrently.
        # num_workers > num_tasks just adds thread overhead with no benefit,
        # and causes the main thread to block on futures[0].result() looking like a hang.
        if self.num_workers > 1 and len(self.tasks) > 1:
            results = self._run_parallel_threads(agent_factory)
        else:
            results = self._run_sequential(agent_factory)
        return results

    def _run_sequential(self, agent_factory):
        results = defaultdict(list)
        total_episodes = len(self.tasks)
        with tqdm(total=total_episodes, desc="Evaluating Episodes", position=0) as pbar:
            for env_name, task, episode_idx in self.tasks:
                evaluator = self.env_evaluators[env_name]
                episode_log = evaluator.run_episode(
                    task, agent_factory, position=1, episode_idx=episode_idx
                )
                results[env_name].append(episode_log)
                pbar.update(1)
        return results

    def _run_parallel_threads(self, agent_factory):
        """Run episodes in parallel using threads.

        Threads are used instead of processes because the workload is IO-bound
        (LLM API calls). This avoids the memory overhead of forking (each
        process would duplicate JAX context + env at ~2-3GB). More concurrent
        episodes also means more concurrent requests to the vLLM server, which
        improves GPU utilization via continuous batching.

        Two progress bars are shown:
          - Top: total steps across ALL workers (true throughput)
          - Bottom: completed episodes out of total
        """
        num_tasks = len(self.tasks)
        num_agents = self.config.alem.get("num_agents", 3)
        max_steps = self.config.eval.max_steps_per_episode or 10000
        logging.info(
            f"Parallel eval: {num_tasks} episodes × {num_agents} agents, "
            f"{self.num_workers} workers → up to {self.num_workers * num_agents} concurrent LLM requests. "
            f"Max {max_steps} steps/episode — episodes end early when all agents die (typically much shorter)."
        )

        results = defaultdict(list)
        lock = threading.Lock()
        import time as _time

        start_time = _time.monotonic()

        # Episodes bar has a known total; steps bar counts up (no total since
        # episodes terminate early when agents die).
        episodes_bar = tqdm(
            total=num_tasks,
            desc=f"Episodes ({self.num_workers} workers)",
            position=0,
            leave=True,
            unit="ep",
        )
        steps_bar = tqdm(
            desc="Total steps",
            position=1,
            leave=True,
            unit="step",
        )

        def _run_task(item, worker_idx):
            env_name, task, episode_idx = item
            try:
                evaluator = self.env_evaluators[env_name]
                result = evaluator.run_episode(
                    task,
                    agent_factory,
                    process_num=f"thread-{worker_idx}",
                    position=worker_idx + 1,
                    episode_idx=episode_idx,
                    step_callback=_on_step,
                )
                result["process_num"] = f"thread-{worker_idx}"
                result["env_name"] = env_name
                with lock:
                    results[env_name].append(result)
            except Exception as e:
                tb = traceback.format_exc()
                logging.error(
                    f"Error in thread-{worker_idx} processing {task} ep {episode_idx}: {e}\n{tb}"
                )
                with lock:
                    results[env_name].append(
                        {
                            "env_name": env_name,
                            "task": task,
                            "error": str(e),
                            "traceback": tb,
                            "process_num": f"thread-{worker_idx}",
                        }
                    )
            finally:
                with lock:
                    episodes_bar.update(1)

        def _on_step():
            """Called by run_episode after each step to update aggregate step counter."""
            with lock:
                steps_bar.update(1)

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks immediately — JAX JIT compilation is thread-safe
            # and caches globally, so concurrent threads just redundantly compile
            # on the first step (a few seconds) rather than serializing an entire episode.
            futures = [
                executor.submit(_run_task, item, idx % self.num_workers)
                for idx, item in enumerate(self.tasks)
            ]
            for f in futures:
                f.result()

        steps_bar.close()
        episodes_bar.close()

        elapsed = _time.monotonic() - start_time
        total_steps = sum(
            r.get("num_steps", 0) for env_results in results.values() for r in env_results
        )
        rate = total_steps / (elapsed / 60) if elapsed > 0 else 0
        logging.info(
            f"Parallel eval complete: {num_tasks} episodes, {total_steps} total steps "
            f"in {elapsed / 60:.1f} min ({rate:.0f} steps/min across {self.num_workers} workers)"
        )
        return results


class Evaluator:
    """Evaluator for a single environment and task."""

    def __init__(self, env_name, config, original_cwd="", output_dir="."):
        self.env_name = env_name.strip()
        self.config = config
        self.output_dir = output_dir
        self.tasks = config.tasks[f"{self.env_name}_tasks"]
        self.num_episodes = config.eval.num_episodes[self.env_name]
        self.num_workers = config.eval.num_workers
        self.max_steps_per_episode = config.eval.max_steps_per_episode

    @staticmethod
    def _flatten_user_info(user_info_tree):
        """Flatten a nested user_info dict from compute_score into a flat {key: float} dict.

        Arrays (e.g. per-agent metrics) are averaged to a scalar.
        NaN/Inf values are skipped.
        """
        flat = {}

        def _walk(prefix, tree):
            if isinstance(tree, dict):
                for k, v in tree.items():
                    _walk(f"{prefix}/{k}" if prefix else k, v)
            else:
                val = tree.mean() if hasattr(tree, "mean") else tree
                v = float(val)
                if v == v and abs(v) != float("inf"):
                    flat[prefix] = v

        _walk("", user_info_tree)
        return flat

    @staticmethod
    def _format_messages_received(agents, agent_idx, step_communications):
        """Format communications received by an agent into the debug viewer's expected format.

        Returns a list of {sender_idx, sender_role, content} dicts matching
        the messages_received schema used by the debug HTML viewer.
        """
        comm_history = getattr(agents[agent_idx], "communication_history", [])
        if not comm_history:
            return []
        last_round = comm_history[-1]
        if not last_round or not isinstance(last_round, dict):
            return []
        msgs = []
        for sender_idx, content in sorted(last_round.items(), key=lambda x: str(x[0])):
            if content is None:
                continue
            # Resolve sender role name if available
            sender_agent = agents[int(sender_idx)] if int(sender_idx) < len(agents) else None
            sender_role = None
            if sender_agent and hasattr(sender_agent, "agent_id"):
                # Try to get role from the agent's prompt builder or observation
                pass
            msgs.append(
                {
                    "sender_idx": int(sender_idx),
                    "sender_role": sender_role,
                    "content": str(content),
                }
            )
        return msgs

    def run_episode(
        self, task, agent_factory, process_num=None, position=0, episode_idx=0, step_callback=None
    ):
        """Run a single evaluation episode with multi-agent support.

        Args:
            step_callback: Optional callable invoked after each step. Used by
                the parallel runner to update an aggregate progress bar.
                When provided, the per-episode tqdm bar is suppressed.
        """
        env = make_env(self.env_name, task, self.config)

        num_agents = env.num_agents
        agents = []
        for agent_idx in range(num_agents):
            agent = agent_factory.create_agent(agent_idx=agent_idx)
            if hasattr(agent, "agent_id"):
                agent.agent_id = agent_idx
            agent.reset()
            agents.append(agent)

        # Seed matches RL eval: jax.random.PRNGKey(EVAL_SEED + ep_idx)
        # (see _run_eval_sequential in baselines/utils.py line 144)
        seed = self.config.envs.env_kwargs.seed
        if seed is None:
            seed = self.config.get("EVAL_SEED", 9999) + episode_idx
        random.seed(seed)
        np.random.seed(seed)
        obs_list, info = env.reset(seed=seed)

        episode_log = {
            "task": task,
            "action_frequency": defaultdict(int),
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "stop_reason_counts": defaultdict(int),
            "incomplete_response_count": 0,
            "incomplete_response_reasons": defaultdict(int),
            "num_agents": num_agents,
        }

        client_cfg = getattr(self.config, "client", None)
        length_early_stop_enabled = _should_early_stop_on_length(client_cfg)
        episode_log["length_incomplete_early_stop_enabled"] = length_early_stop_enabled
        logging.info(
            "Length-based early stop on repeated finish_reason=length is %s for client=%s model=%s",
            "enabled" if length_early_stop_enabled else "disabled",
            getattr(client_cfg, "client_name", "unknown"),
            getattr(client_cfg, "model_id", "unknown"),
        )

        for i in range(num_agents):
            episode_log[f"agent_{i}_action_frequency"] = defaultdict(int)
            episode_log[f"agent_{i}_input_tokens"] = 0
            episode_log[f"agent_{i}_output_tokens"] = 0
            episode_log[f"agent_{i}_reasoning_tokens"] = 0
            episode_log[f"agent_{i}_stop_reason_counts"] = defaultdict(int)
            episode_log[f"agent_{i}_incomplete_response_count"] = 0

        instructions = None

        # Track max dungeon level reached for progressive disclosure prompt refresh.
        # Prompts only ever grow — sections unlocked by descending are never removed,
        # even if the team later ascends back to a higher level.
        max_level_seen = 0

        for agent_idx in range(num_agents):
            instruction_prompt = env.get_instruction_prompt(agent_idx, instructions=instructions)
            if (
                hasattr(agents[agent_idx], "prompt_builder")
                and agents[agent_idx].prompt_builder is not None
            ):
                # Use set_instruction_prompt if available (e.g. RobustAllAgent) so that
                # format instructions are re-injected after the game prompt is set.
                if hasattr(agents[agent_idx], "set_instruction_prompt"):
                    agents[agent_idx].set_instruction_prompt(instruction_prompt)
                else:
                    agents[agent_idx].prompt_builder.update_instruction_prompt(instruction_prompt)
                if agent_idx == 0 or episode_idx == 0:
                    logging.info(
                        f"\n{'=' * 80}\nINSTRUCTION PROMPT (Agent {agent_idx}):\n{'=' * 80}\n{instruction_prompt}\n{'=' * 80}"
                    )

        episode_return = 0.0
        episode_returns = [0.0] * num_agents

        max_steps_per_episode = (
            env.max_steps if self.max_steps_per_episode is None else self.max_steps_per_episode
        )

        # For cheap local sweeps we only save image-heavy artifacts for the first
        # episode, but expensive API runs can opt into full per-episode debug.
        save_images_every_episode = bool(self.config.eval.get("save_images_every_episode", False))
        save_images = self.config.eval.save_images and (
            save_images_every_episode or episode_idx == 0
        )
        agent_frames = {i: [] for i in range(num_agents)} if save_images else {}
        llm_frames = [] if save_images else None
        llm_agent_data_per_step = [] if save_images else None

        csv_filename = os.path.join(
            self.output_dir, self.env_name, task, f"{task}_run_{episode_idx:02d}.csv"
        )
        debug_filename = os.path.join(
            self.output_dir, self.env_name, task, f"{task}_run_{episode_idx:02d}_debug.jsonl"
        )
        Path(csv_filename).parent.mkdir(exist_ok=True, parents=True)

        with (
            open(csv_filename, mode="w", newline="", encoding="utf-8") as csv_file,
            open(debug_filename, mode="w", encoding="utf-8") as debug_file,
        ):
            csv_writer = csv.writer(csv_file, escapechar="˘", quoting=csv.QUOTE_MINIMAL)
            header = ["Step"]
            for i in range(num_agents):
                header.extend([f"Agent{i}_Action", f"Agent{i}_Reasoning"])
            header.extend(["Observations", "Rewards", "Dones"])
            csv_writer.writerow(header)

            # In parallel mode (step_callback provided), suppress per-episode bars
            # to avoid 10+ stacked bars that obscure the aggregate throughput.
            use_episode_pbar = step_callback is None
            if use_episode_pbar:
                pbar_desc = (
                    f"Ep {episode_idx} (max {max_steps_per_episode} steps, ends early on death)"
                )
                pbar = tqdm(
                    total=max_steps_per_episode,
                    desc=pbar_desc,
                    position=position,
                    leave=False,
                    dynamic_ncols=True,
                )
            else:
                pbar = None

            prev_actions = [None] * num_agents
            # Maps agent_idx -> message sent this step. Fed to all agents at
            # the START of the next step via receive_communication(), so each
            # agent sees what others said after acting on the same observation.
            step_communications = {}
            # Track action parse success/failure per agent
            parse_success = [0] * num_agents
            parse_fail = [0] * num_agents
            parse_skipped_inactive = [0] * num_agents
            # Per-step data for windowed ICL metrics
            step_total_rewards = []
            step_parse_successes = []
            step_parse_attempts = []
            episode_error = None
            consecutive_length_incompletes = 0

            # Trajectory collection — mirrors RL _run_eval_sequential (baselines/utils.py)
            _traj_obs = []  # raw symbolic obs (pre-step)
            _traj_actions = []  # discrete action indices
            _traj_rewards = []  # per-agent rewards
            _traj_dones = []  # episode done flag
            _traj_text_obs = []  # long-term text obs per agent
            _traj_text_actions = []  # canonical text action per agent
            _traj_states = []  # exact pre-step EnvState snapshots for replay

            agent_executor = ThreadPoolExecutor(max_workers=num_agents)
            # Initialise to -1 so the except handler can report the failing step
            # even if the error occurs before the first iteration completes.
            step = -1
            try:
                for step in range(max_steps_per_episode):
                    stop_episode_due_to_length = False
                    # Provide communication from the previous step to agents
                    for idx, agent in enumerate(agents):
                        if hasattr(agent, "receive_communication"):
                            receiver_id = getattr(agent, "agent_id", idx)
                            filtered_communications = {
                                sender_id: msg
                                for sender_id, msg in step_communications.items()
                                if sender_id != receiver_id
                            }
                            agent.receive_communication(filtered_communications)

                    # Snapshot observations BEFORE env.step() mutates obs_list.
                    # This captures what the agent actually saw when choosing its
                    # action, which is needed for accurate debug logging/HTML.
                    pre_step_obs = []
                    for agent_idx in range(num_agents):
                        snapshot = {
                            "obs_long_term": obs_list[agent_idx]
                            .get("text", {})
                            .get("long_term_context", ""),
                            "obs_short_term": obs_list[agent_idx]
                            .get("text", {})
                            .get("short_term_context", ""),
                        }
                        img = obs_list[agent_idx].get("image") if save_images else None
                        if img is not None:
                            buf = io.BytesIO()
                            img.save(buf, format="PNG")
                            snapshot["image_base64"] = base64.b64encode(buf.getvalue()).decode(
                                "ascii"
                            )
                        else:
                            snapshot["image_base64"] = None
                        pre_step_obs.append(snapshot)

                    # Trajectory: raw symbolic obs from JAX state (pre-step)
                    try:
                        _raw_obs = env.env.get_obs(env.state)
                        _traj_obs.append(np.stack([np.array(_raw_obs[a]) for a in env.env.agents]))
                    except Exception as _e:
                        logger.debug(f"Traj symbolic obs capture failed step {step}: {_e}")
                        _traj_obs.append(None)
                    try:
                        # Store a host-side snapshot so replay can use the real
                        # renderer later without relying on seeds matching.
                        _traj_states.append(jax.device_get(env.state))
                    except Exception as _e:
                        logger.debug(f"Traj state capture failed step {step}: {_e}")
                    # Trajectory: text obs (long-term context seen by each agent this turn)
                    _traj_text_obs.append(
                        [pre_step_obs[i]["obs_long_term"] for i in range(num_agents)]
                    )

                    actions = [None] * num_agents
                    reasonings = [None] * num_agents
                    responses = [None] * num_agents
                    prompt_histories = [None] * num_agents

                    def _agent_act(agent_idx):
                        return agents[agent_idx].act(
                            obs_list[agent_idx], prev_action=prev_actions[agent_idx]
                        )

                    # Call all agents' LLMs in parallel. Each agent has its own
                    # client/prompt_builder so no shared mutable state is touched.
                    # Results are collected sequentially to update episode_log safely.
                    futures = {agent_executor.submit(_agent_act, i): i for i in range(num_agents)}
                    for future in futures:
                        agent_idx = futures[future]
                        response = future.result()
                        action = env.check_action_validity(response.completion, agent_idx)
                        actions[agent_idx] = action
                        reasonings[agent_idx] = (
                            response.reasoning if hasattr(response, "reasoning") else ""
                        )
                        responses[agent_idx] = response

                        client = getattr(agents[agent_idx], "client", None)
                        if client is not None and hasattr(client, "last_prompt_messages"):
                            prompt_histories[agent_idx] = client.last_prompt_messages

                        episode_log[f"agent_{agent_idx}_action_frequency"][action] += 1
                        episode_log[f"agent_{agent_idx}_input_tokens"] += response.input_tokens
                        episode_log[f"agent_{agent_idx}_output_tokens"] += response.output_tokens
                        episode_log[f"agent_{agent_idx}_reasoning_tokens"] += getattr(
                            response, "reasoning_tokens", 0
                        )
                        stop_reason = getattr(response, "stop_reason", None) or "unknown"
                        episode_log["stop_reason_counts"][stop_reason] += 1
                        episode_log[f"agent_{agent_idx}_stop_reason_counts"][stop_reason] += 1
                        if length_early_stop_enabled:
                            if stop_reason == "length":
                                consecutive_length_incompletes += 1
                                if (
                                    consecutive_length_incompletes
                                    >= _MAX_CONSECUTIVE_LENGTH_INCOMPLETES
                                ):
                                    stop_episode_due_to_length = True
                            else:
                                consecutive_length_incompletes = 0
                        incomplete_reason = _classify_incomplete_response(response)
                        if incomplete_reason is not None:
                            episode_log["incomplete_response_count"] += 1
                            episode_log["incomplete_response_reasons"][incomplete_reason] += 1
                            episode_log[f"agent_{agent_idx}_incomplete_response_count"] += 1
                        episode_log["action_frequency"][action] += 1
                        episode_log["input_tokens"] += response.input_tokens
                        episode_log["output_tokens"] += response.output_tokens
                        episode_log["reasoning_tokens"] += getattr(response, "reasoning_tokens", 0)

                    # Trajectory: discrete action indices + text actions.
                    # Standard actions map directly via ACTIONS list. "Give to Agent X"
                    # slots are >= len(ACTIONS) and encoded as Action.GIVE + slot offset;
                    # we compute the offset directly to avoid wrapper side-effects.
                    try:
                        from alem.alem_coop.constants import Action as _Action
                        from alem.llm.alem_env import ACTIONS as _ACTIONS

                        _act_to_idx = {a: i for i, a in enumerate(_ACTIONS)}
                        _give_base = _Action.GIVE.value
                        _act_indices = []
                        for _i in range(num_agents):
                            _a = actions[_i]
                            _idx = _act_to_idx.get(_a)
                            if _idx is None:
                                # "Give to Agent X" — compute targeted give slot
                                import re as _re

                                _m = _re.search(r"give\s+to\s+agent\s+(\d+)", _a, _re.IGNORECASE)
                                if _m:
                                    _tgt = int(_m.group(1))
                                    _slot = _tgt if _tgt < _i else _tgt - 1
                                    _idx = _give_base + _slot
                                else:
                                    _idx = _act_to_idx.get("Give", 0)
                            _act_indices.append(_idx)
                        _traj_actions.append(np.array(_act_indices, dtype=np.int32))
                    except Exception as _e:
                        logger.debug(f"Traj action capture failed step {step}: {_e}")
                        _traj_actions.append(np.zeros(num_agents, dtype=np.int32))
                    _traj_text_actions.append(list(actions))

                    step_communications = {}
                    for agent in agents:
                        if (
                            hasattr(agent, "current_communication")
                            and agent.current_communication is not None
                        ):
                            step_communications[getattr(agent, "agent_id", None)] = (
                                agent.current_communication
                            )

                    if step % 10 == 0 or step == 0:
                        for agent_idx in range(num_agents):
                            obs_text = pre_step_obs[agent_idx]["obs_long_term"]
                            logging.debug(
                                f"\n--- Step {step}, Agent {agent_idx} ---\nObservation: {obs_text[:100]}...\nAction: {actions[agent_idx]}\nReasoning: {reasonings[agent_idx][:100] if reasonings[agent_idx] else 'None'}..."
                            )

                    obs_list, rewards, terminateds, truncateds, info = env.step(actions)
                    dones = [t or tr for t, tr in zip(terminateds, truncateds)]
                    done = any(dones)

                    # Trajectory: rewards and done flag
                    _traj_rewards.append(np.array(rewards, dtype=np.float32))
                    _traj_dones.append(done)

                    # Progressive disclosure: expand system prompts when team reaches a new max level.
                    # We pass max_level_seen (not current level) so sections are never removed
                    # if the team ascends back after descending.
                    if (
                        hasattr(env, "state")
                        and env.state is not None
                        and hasattr(env.state, "player_level")
                    ):
                        cur_level = int(env.state.player_level)
                        if cur_level > max_level_seen:
                            max_level_seen = cur_level
                            logging.info(
                                f"Step {step}: new max dungeon level {max_level_seen}, expanding system prompts"
                            )
                            for agent_idx in range(num_agents):
                                new_prompt = env.get_instruction_prompt(
                                    agent_idx, current_level=max_level_seen
                                )
                                if (
                                    hasattr(agents[agent_idx], "prompt_builder")
                                    and agents[agent_idx].prompt_builder is not None
                                ):
                                    if hasattr(agents[agent_idx], "set_instruction_prompt"):
                                        agents[agent_idx].set_instruction_prompt(new_prompt)
                                    else:
                                        agents[agent_idx].prompt_builder.update_instruction_prompt(
                                            new_prompt
                                        )

                    for agent_idx in range(num_agents):
                        episode_returns[agent_idx] += rewards[agent_idx]
                    episode_return = sum(episode_returns) / num_agents

                    # Detect parse failures using the agent's _last_parse_failed flag,
                    # which is set reliably in act(). The old heuristic of checking
                    # resp.completion for "noop" was broken: completion is already
                    # replaced with the canonical action name before reaching here.
                    step_parse_ok = 0
                    step_parse_attempt_count = 0
                    for agent_idx in range(num_agents):
                        parsed_action = actions[agent_idx]
                        # is agent alive
                        is_inactive_next = bool(obs_list[agent_idx].get("is_inactive", False))
                        parse_failed = False

                        # Dead/inactive agents can only Noop; exclude these turns
                        # from action parse metrics rather than counting failures.
                        if is_inactive_next:
                            parse_skipped_inactive[agent_idx] += 1
                        else:
                            step_parse_attempt_count += 1
                            parse_failed = getattr(agents[agent_idx], "_last_parse_failed", False)
                            if parse_failed:
                                parse_fail[agent_idx] += 1
                            else:
                                parse_success[agent_idx] += 1
                                step_parse_ok += 1
                        # Append parse failure feedback AFTER the observation so
                        # the agent sees the world state first, then the correction.
                        if self.config.eval.feedback_on_invalid_action:
                            feedback_parts = []
                            if parse_failed and not is_inactive_next:
                                raw = getattr(agents[agent_idx], "_last_raw_completion", None)
                                if not raw:
                                    feedback_parts.append(
                                        "Your previous response was empty. You must output "
                                        "<action>YOUR_CHOSEN_ACTION</action>."
                                    )
                                else:
                                    feedback_parts.append(
                                        "Your previous output did not contain a valid action. "
                                        "Defaulted to Noop. You must include <action>YOUR_CHOSEN_ACTION</action>."
                                    )
                            if getattr(agents[agent_idx], "_last_comm_failed", False):
                                feedback_parts.append(
                                    "Your communication could not be parsed. Use exactly: "
                                    "<communication>YOUR_MESSAGE</communication>"
                                )
                            if getattr(agents[agent_idx], "_last_scratchpad_failed", False):
                                feedback_parts.append(
                                    "Your scratchpad could not be parsed. Use exactly: "
                                    "<scratchpad>YOUR_NOTES</scratchpad>"
                                )
                            if feedback_parts:
                                obs_list[agent_idx]["text"]["long_term_context"] = (
                                    " ".join(feedback_parts)
                                    + "\n\n"
                                    + obs_list[agent_idx]["text"]["long_term_context"]
                                )

                    step_total_rewards.append(sum(rewards))
                    step_parse_successes.append(step_parse_ok)
                    step_parse_attempts.append(step_parse_attempt_count)

                    prev_actions = list(actions)

                    row = [step]
                    for agent_idx in range(num_agents):
                        row.extend([actions[agent_idx], reasonings[agent_idx]])
                    row.append(
                        str([obs_list[i]["text"]["long_term_context"] for i in range(num_agents)])
                    )
                    row.append(str(rewards))
                    row.append(str(dones))
                    csv_writer.writerow(row)

                    # Write debug record using pre-step obs (what the agent actually saw)
                    total_attempts = [parse_success[i] + parse_fail[i] for i in range(num_agents)]
                    debug_record = {
                        "step": step,
                        "agents": {},
                        "rewards": rewards,
                        "dones": dones,
                        "action_parse_stats": {
                            str(i): {
                                "success": parse_success[i],
                                "fail": parse_fail[i],
                                "skipped_inactive": parse_skipped_inactive[i],
                                "total": total_attempts[i],
                                "parse_rate": round(
                                    parse_success[i] / max(total_attempts[i], 1), 4
                                ),
                            }
                            for i in range(num_agents)
                        },
                    }
                    for agent_idx in range(num_agents):
                        raw_completion = getattr(agents[agent_idx], "_last_raw_completion", None)
                        if raw_completion is None:
                            raw_completion = responses[agent_idx].completion
                        agent_debug = {
                            "obs_long_term": pre_step_obs[agent_idx]["obs_long_term"],
                            "obs_short_term": pre_step_obs[agent_idx]["obs_short_term"],
                            "image_base64": pre_step_obs[agent_idx]["image_base64"],
                            "llm_raw_output": raw_completion,
                            "raw_reasoning": getattr(
                                agents[agent_idx], "_last_raw_reasoning", None
                            ),
                            "reasoning": responses[agent_idx].reasoning
                            if hasattr(responses[agent_idx], "reasoning")
                            else None,
                            "stop_reason": getattr(responses[agent_idx], "stop_reason", None),
                            "reasoning_tokens": getattr(
                                responses[agent_idx], "reasoning_tokens", 0
                            ),
                            "output_tokens": getattr(responses[agent_idx], "output_tokens", 0),
                            "incomplete_reason": _classify_incomplete_response(
                                responses[agent_idx]
                            ),
                            "parsed_action": actions[agent_idx],
                            "scratchpad": getattr(agents[agent_idx], "scratchpad_history", [])[-1]
                            if getattr(agents[agent_idx], "scratchpad_history", [])
                            else None,
                            "message_sent": (
                                {
                                    "content": getattr(
                                        agents[agent_idx], "current_communication", None
                                    )
                                }
                                if getattr(agents[agent_idx], "current_communication", None)
                                else None
                            ),
                            "messages_received": self._format_messages_received(
                                agents, agent_idx, step_communications
                            ),
                        }
                        if prompt_histories[agent_idx] is not None:
                            agent_debug["prompt_messages"] = prompt_histories[agent_idx]
                        debug_record["agents"][str(agent_idx)] = agent_debug
                    debug_file.write(_safe_json_dumps(debug_record) + "\n")

                    if pbar is not None:
                        pbar.update(1)
                    if step_callback is not None:
                        step_callback()

                    if save_images and obs_list[0].get("image"):
                        for agent_idx in range(num_agents):
                            if obs_list[agent_idx].get("image"):
                                agent_frames[agent_idx].append(
                                    np.array(obs_list[agent_idx]["image"])
                                )

                        # Collect LLM composite frame data
                        try:
                            from alem.alem_coop.constants import Specialization
                            from alem.alem_coop.renderer.renderer_llm import render_llm_frame

                            spec_order = [
                                Specialization.WARRIOR,
                                Specialization.FORAGER,
                                Specialization.MINER,
                            ]
                            agent_images = [obs_list[i].get("image") for i in range(num_agents)]
                            agent_data = []
                            for i in range(num_agents):
                                received = {}
                                comm_hist = getattr(agents[i], "communication_history", [])
                                if comm_hist:
                                    last = comm_hist[-1]
                                    if isinstance(last, dict):
                                        received = {k: v for k, v in last.items() if v}
                                agent_data.append(
                                    {
                                        "id": i,
                                        "role": spec_order[i % 3].name.lower(),
                                        "comm_sent": getattr(
                                            agents[i], "current_communication", None
                                        ),
                                        "comm_received": received or None,
                                        "scratchpad": getattr(agents[i], "scratchpad_history", [])[
                                            -1
                                        ]
                                        if getattr(agents[i], "scratchpad_history", [])
                                        else None,
                                        "action": actions[i],
                                        "reasoning": reasonings[i] if reasonings[i] else None,
                                    }
                                )
                            llm_frame = render_llm_frame(
                                agent_images=agent_images,
                                agent_data=agent_data,
                                step=step,
                                max_steps=max_steps_per_episode,
                            )
                            llm_frames.append(llm_frame)
                            llm_agent_data_per_step.append(agent_data)
                        except Exception as e:
                            logging.debug(f"LLM frame render failed: {e}")

                    if stop_episode_due_to_length:
                        episode_log["done"] = False
                        episode_log["early_stop_reason"] = "consecutive_length_incomplete_responses"
                        episode_log["early_stop_step"] = step
                        episode_log["early_stop_consecutive_length_incompletes"] = (
                            consecutive_length_incompletes
                        )
                        logging.warning(
                            "Ending episode early at step %s after %s consecutive responses with "
                            "finish_reason=length to avoid wasting API calls.",
                            step,
                            consecutive_length_incompletes,
                        )
                        if pbar is not None:
                            if pbar.n < pbar.total:
                                pbar.update(pbar.total - pbar.n)
                            pbar.set_postfix_str("EARLY STOP")
                        break

                    if done:
                        if "user_info" in info:
                            episode_log["user_info"] = self._flatten_user_info(info["user_info"])

                        logging.info(f"Episode done with mean reward per agent: {episode_return}")
                        for agent_idx in range(num_agents):
                            logging.info(
                                f"  Agent {agent_idx} return: {episode_returns[agent_idx]}"
                            )
                        episode_log["done"] = True
                        if pbar is not None:
                            if pbar.n < pbar.total:
                                pbar.update(pbar.total - pbar.n)
                            pbar.set_postfix_str("DONE")
                        break

            except Exception as e:
                episode_error = str(e)
                logging.error(f"Episode failed at step {step}: {e}\n{traceback.format_exc()}")
            finally:
                agent_executor.shutdown(wait=False)

            # Fallback: if episode was truncated by max_steps (not a natural done),
            # capture user_info from the last step. The env recomputes metrics
            # on truncation so these are still valid end-of-episode values.
            if "user_info" not in episode_log and info and "user_info" in info:
                episode_log["user_info"] = self._flatten_user_info(info["user_info"])

            if pbar is not None:
                if pbar.n < pbar.total:
                    pbar.update(pbar.total - pbar.n)
                if "done" not in episode_log:
                    pbar.set_postfix_str("DONE")
                pbar.close()

            episode_log["episode_return"] = episode_return
            for agent_idx in range(num_agents):
                episode_log[f"agent_{agent_idx}_return"] = episode_returns[agent_idx]
            episode_log["num_steps"] = step + 1
            if episode_error is not None:
                episode_log["error"] = episode_error
            episode_log["failed_candidates"] = env.failed_candidates
            for agent_idx in range(num_agents):
                episode_log[f"agent_{agent_idx}_failed_candidates"] = (
                    env.failed_candidates_per_agent[agent_idx]
                )

            # Action parse statistics
            total_parse_success = sum(parse_success)
            total_parse_fail = sum(parse_fail)
            total_parse_attempts = total_parse_success + total_parse_fail
            episode_log["action_parse_rate"] = round(
                total_parse_success / max(total_parse_attempts, 1), 4
            )
            episode_log["action_parse_success"] = total_parse_success
            episode_log["action_parse_fail"] = total_parse_fail
            episode_log["action_parse_skipped_inactive"] = sum(parse_skipped_inactive)
            for agent_idx in range(num_agents):
                agent_total = parse_success[agent_idx] + parse_fail[agent_idx]
                episode_log[f"agent_{agent_idx}_parse_rate"] = round(
                    parse_success[agent_idx] / max(agent_total, 1), 4
                )
                episode_log[f"agent_{agent_idx}_parse_success"] = parse_success[agent_idx]
                episode_log[f"agent_{agent_idx}_parse_fail"] = parse_fail[agent_idx]
                episode_log[f"agent_{agent_idx}_parse_skipped_inactive"] = parse_skipped_inactive[
                    agent_idx
                ]

            # Compute windowed ICL metrics (reward & parse rate over time)
            WINDOW_SIZE = 100
            windowed_metrics = []
            num_steps_actual = len(step_total_rewards)
            for w_start in range(0, num_steps_actual, WINDOW_SIZE):
                w_end = min(w_start + WINDOW_SIZE, num_steps_actual)
                w_rewards = step_total_rewards[w_start:w_end]
                w_parses = step_parse_successes[w_start:w_end]
                w_attempts = sum(step_parse_attempts[w_start:w_end])
                windowed_metrics.append(
                    {
                        "window_start": w_start,
                        "window_end": w_end,
                        "mean_reward": round(sum(w_rewards) / len(w_rewards), 6),
                        "cumulative_reward": round(sum(step_total_rewards[:w_end]), 6),
                        "parse_rate": round(sum(w_parses) / max(w_attempts, 1), 4),
                    }
                )
            episode_log["windowed_metrics"] = windowed_metrics
            episode_log["windowed_window_size"] = WINDOW_SIZE

            # env.get_stats() returns per-agent {score, progression, achievements}
            # keyed as "agent_0", "agent_1", etc. These come from the env's
            # internal tracking, separate from the user_info metrics above.
            all_stats = env.get_stats()
            episode_log.update(all_stats)

            # Capture agent-level retry stats (robust agents only) and
            # aggregate comm/scratchpad parse rates across all agents.
            total_comm_attempted = 0
            total_comm_parsed = 0
            total_scratchpad_attempted = 0
            total_scratchpad_parsed = 0
            for agent_idx in range(num_agents):
                if hasattr(agents[agent_idx], "get_retry_stats"):
                    retry_stats = agents[agent_idx].get_retry_stats()
                    for key, val in retry_stats.items():
                        episode_log[f"agent_{agent_idx}_retry_{key}"] = val
                    total_comm_attempted += retry_stats.get("comm_attempted", 0)
                    total_comm_parsed += retry_stats.get("comm_parsed", 0)
                    total_scratchpad_attempted += retry_stats.get("scratchpad_attempted", 0)
                    total_scratchpad_parsed += retry_stats.get("scratchpad_parsed", 0)

            if total_comm_attempted > 0:
                episode_log["comm_parse_rate"] = round(total_comm_parsed / total_comm_attempted, 4)
                episode_log["comm_attempted"] = total_comm_attempted
                episode_log["comm_parsed"] = total_comm_parsed
            if total_scratchpad_attempted > 0:
                episode_log["scratchpad_parse_rate"] = round(
                    total_scratchpad_parsed / total_scratchpad_attempted, 4
                )
                episode_log["scratchpad_attempted"] = total_scratchpad_attempted
                episode_log["scratchpad_parsed"] = total_scratchpad_parsed

            # Log parse rates summary: parsed/attempted = tag quality (closed when opened)
            parse_parts = [f"action={episode_log['action_parse_rate']:.1%}"]
            if total_comm_attempted > 0:
                parse_parts.append(
                    f"comm={total_comm_parsed}/{total_comm_attempted} ({episode_log['comm_parse_rate']:.1%})"
                )
            if total_scratchpad_attempted > 0:
                parse_parts.append(
                    f"scratchpad={total_scratchpad_parsed}/{total_scratchpad_attempted} ({episode_log['scratchpad_parse_rate']:.1%})"
                )
            logging.info(f"Parse rates: {', '.join(parse_parts)}")

            # Save trajectory (matches RL _run_eval_sequential npz format)
            _save_episode_trajectory(
                self.output_dir,
                self.env_name,
                task,
                episode_idx,
                _traj_obs,
                _traj_actions,
                _traj_rewards,
                _traj_dones,
                _traj_text_obs,
                _traj_text_actions,
            )
            _save_episode_states(
                self.output_dir,
                self.env_name,
                task,
                episode_idx,
                _traj_states,
                env.env.static_env_params,
            )

            # Save GIFs: per-agent + combined side-by-side
            if save_images and any(len(f) > 0 for f in agent_frames.values()):
                gif_dir = os.path.join(self.output_dir, self.env_name, task)
                Path(gif_dir).mkdir(exist_ok=True, parents=True)

                per_agent_arrays = {}
                for agent_idx in range(num_agents):
                    if agent_frames[agent_idx]:
                        frames_np = np.array(agent_frames[agent_idx]).astype(np.uint8)
                        per_agent_arrays[agent_idx] = frames_np
                        gif_path = os.path.join(
                            gif_dir, f"{task}_run_{episode_idx:02d}_agent{agent_idx}.gif"
                        )
                        imageio.mimsave(gif_path, frames_np, fps=10)
                        logging.info(f"Saved agent {agent_idx} GIF to {gif_path}")

                # Combined side-by-side GIF with all agents
                if len(per_agent_arrays) == num_agents:
                    num_frames = min(arr.shape[0] for arr in per_agent_arrays.values())
                    padding_width = 10
                    combined_frames = []
                    for t in range(num_frames):
                        frame = per_agent_arrays[0][t]
                        for agent_idx in range(1, num_agents):
                            padding = (
                                np.ones((frame.shape[0], padding_width, 3), dtype=np.uint8) * 255
                            )
                            frame = np.hstack([frame, padding, per_agent_arrays[agent_idx][t]])
                        combined_frames.append(frame)
                    combined_path = os.path.join(gif_dir, f"{task}_run_{episode_idx:02d}.gif")
                    imageio.mimsave(combined_path, combined_frames, fps=10)
                    logging.info(f"Saved combined GIF to {combined_path}")

                # LLM composite outputs (video + HTML viewer)
                if llm_frames:
                    from alem.alem_coop.renderer.renderer_llm import (
                        render_llm_gif,
                        render_llm_html,
                        render_llm_video,
                    )

                    prefix = f"{task}_run_{episode_idx:02d}_llm"
                    llm_fps = 1  # 1 fps — one step per second, easy to follow
                    # MP4 video (pauseable)
                    try:
                        mp4_path = os.path.join(gif_dir, f"{prefix}.mp4")
                        render_llm_video(llm_frames, mp4_path, fps=llm_fps)
                        logging.info(f"Saved LLM video to {mp4_path}")
                    except Exception as e:
                        logging.warning(f"Failed to save LLM video: {e}")
                    # Interactive HTML viewer (play/pause/scrub)
                    try:
                        html_path = os.path.join(gif_dir, f"{prefix}.html")
                        render_llm_html(llm_frames, llm_agent_data_per_step, html_path, fps=llm_fps)
                        logging.info(f"Saved LLM HTML viewer to {html_path}")
                    except Exception as e:
                        logging.warning(f"Failed to save LLM HTML: {e}")
                    # GIF fallback
                    try:
                        gif_path = os.path.join(gif_dir, f"{prefix}.gif")
                        render_llm_gif(llm_frames, gif_path, fps=llm_fps)
                        logging.info(f"Saved LLM GIF to {gif_path}")
                    except Exception as e:
                        logging.warning(f"Failed to save LLM GIF: {e}")

            # --- Post-episode debrief ---
            if self.config.eval.get("generate_debriefs", self.config.eval.get("debug", False)):
                logging.info(
                    "Debrief start: task=%s episode=%s step=%s done=%s max_steps=%s",
                    task,
                    episode_idx,
                    step + 1,
                    episode_log.get("done", False),
                    max_steps_per_episode,
                )
                spec_order = ["warrior", "forager", "miner"]
                debriefs = {}
                for agent_idx in range(num_agents):
                    agent = agents[agent_idx]
                    if not hasattr(agent, "get_debrief"):
                        logging.warning(
                            "Skipping debrief for agent %s: no get_debrief()", agent_idx
                        )
                        continue
                    try:
                        role = spec_order[agent_idx % 3]

                        # Achievements: list of achievement names
                        agent_stats = env.get_stats(agent_idx)
                        achieved_names = list(agent_stats.get("achievements", {}).keys())
                        total_ach = len(achieved_names)
                        try:
                            from alem.alem_coop.constants import Achievement

                            max_ach = len(Achievement)
                        except Exception:
                            max_ach = total_ach or 1

                        # Inventory + vitals from env state (JAX arrays → readable string)
                        # describe_inventory already includes health/food/mana/xp
                        inventory_str = "(unavailable)"
                        stats_at_death = {}
                        try:
                            if (
                                hasattr(env, "wrapper")
                                and hasattr(env, "state")
                                and env.state is not None
                            ):
                                inventory_str = env.wrapper.describe_inventory(env.state, agent_idx)
                        except Exception:
                            pass

                        # Death cause
                        try:
                            is_dead = (
                                hasattr(env, "state")
                                and env.state is not None
                                and bool(env.state.player_health[agent_idx] <= 0)
                            )
                            death_cause = "health depleted" if is_dead else "time limit reached"
                        except Exception:
                            death_cause = "unknown"

                        # Communication sample: flatten history into labeled strings
                        comm_sample = []
                        for round_idx, round_msgs in enumerate(agent.communication_history):
                            if isinstance(round_msgs, dict):
                                for sender_id, content in round_msgs.items():
                                    if content:
                                        comm_sample.append(
                                            f"[step ~{round_idx}] Agent {sender_id}: {content}"
                                        )

                        final_scratchpad = (
                            agent.scratchpad_history[-1] if agent.scratchpad_history else ""
                        )

                        debrief_text = agent.get_debrief(
                            agent_id=agent_idx,
                            role=role,
                            final_scratchpad=final_scratchpad,
                            achievements=achieved_names,
                            total_achievements=total_ach,
                            max_achievements=max_ach,
                            steps_survived=step + 1,
                            max_steps=max_steps_per_episode,
                            deepest_level=max_level_seen,
                            death_cause=death_cause,
                            inventory_at_death=inventory_str,
                            stats_at_death=stats_at_death,
                            communication_log_sample=comm_sample if comm_sample else None,
                        )
                        if debrief_text:
                            debriefs[f"agent_{agent_idx}"] = debrief_text
                            logging.info(f"Agent {agent_idx} debrief:\n{debrief_text}")
                        else:
                            logging.warning(
                                "Agent %s produced empty debrief (comm_items=%s, scratchpad_chars=%s)",
                                agent_idx,
                                len(comm_sample),
                                len(final_scratchpad),
                            )
                    except Exception as e:
                        logging.error(
                            "Debrief generation failed for agent %s: %s\n%s",
                            agent_idx,
                            e,
                            traceback.format_exc(),
                        )

                if debriefs:
                    episode_log["debriefs"] = debriefs
                    # Append debrief record to the already-open debug JSONL handle
                    try:
                        debug_file.write(
                            _safe_json_dumps({"type": "debrief", "agents": debriefs}) + "\n"
                        )
                    except Exception as e:
                        logging.warning(f"Failed to append debriefs to debug JSONL: {e}")
                else:
                    logging.warning(
                        "No debriefs generated for task=%s episode=%s", task, episode_idx
                    )

            episode_log["process_num"] = process_num
            episode_log["seed"] = seed
            episode_log["agent"] = OmegaConf.to_container(self.config.agent, resolve=True)
            clients_log = OmegaConf.to_container(self.config.clients, resolve=True)
            if not isinstance(clients_log, list):
                raise ValueError("config.clients must resolve to a list for episode logging.")
            # Log the actual runtime enable_thinking (may differ from config default)
            for client_cfg in clients_log:
                if isinstance(client_cfg, dict):
                    client_cfg["enable_thinking_resolved"] = getattr(
                        agent_factory,
                        "_resolved_enable_thinking",
                        None,
                    )
            episode_log["clients"] = clients_log

            json_filename = os.path.join(
                self.output_dir,
                self.env_name,
                task,
                f"{task}_run_{episode_idx:02d}.json",
            )
            Path(json_filename).parent.mkdir(exist_ok=True, parents=True)
            with open(json_filename, "w") as f:
                json.dump(episode_log, f, indent=4)

            # Save debrief to a separate plain-text file for easy inspection
            if episode_log.get("debriefs"):
                debrief_filename = os.path.join(
                    self.output_dir,
                    self.env_name,
                    task,
                    f"{task}_run_{episode_idx:02d}_debrief.txt",
                )
                try:
                    with open(debrief_filename, "w", encoding="utf-8") as df:
                        for agent_key, text in sorted(episode_log["debriefs"].items()):
                            df.write(f"{'=' * 60}\n{agent_key.upper()}\n{'=' * 60}\n")
                            df.write((text or "(no debrief)") + "\n\n")
                    logging.info(f"Saved debrief to {debrief_filename}")
                except Exception as e:
                    logging.warning(f"Failed to save debrief file: {e}")

            # Flush debug JSONL before reading it back for HTML/txt generation
            debug_file.flush()

            # Generate debug HTML visualisation
            try:
                run_id = str(self.config.get("wandb", {}).get("run_id", "")).strip()
                html_output_path = None
                if run_id:
                    base_name = Path(debug_filename).name.replace("_debug.jsonl", "")
                    html_output_path = str(
                        Path(debug_filename).with_name(f"{base_name}_{run_id}_debug.html")
                    )
                html_path = generate_debug_html(
                    debug_filename,
                    episode_json_path=json_filename,
                    output_path=html_output_path,
                )
                logging.info(f"Saved debug HTML to {html_path}")
            except Exception as e:
                logging.warning(f"Failed to generate debug HTML: {e}")

            # Generate human-readable step log (messages + reasoning per step)
            try:
                run_id = str(self.config.get("wandb", {}).get("run_id", "")).strip()
                txt_output_path = None
                if run_id:
                    base_name = Path(debug_filename).name.replace("_debug.jsonl", "")
                    txt_output_path = str(
                        Path(debug_filename).with_name(f"{base_name}_{run_id}_steps.txt")
                    )
                txt_path = generate_step_log_txt(debug_filename, output_path=txt_output_path)
                logging.info(f"Saved step log to {txt_path}")
            except Exception as e:
                logging.warning(f"Failed to generate step log: {e}")

        return episode_log
