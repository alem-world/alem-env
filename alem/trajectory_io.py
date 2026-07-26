"""Shared episode trajectory/state persistence for RL, LLM, and human-play runs.

``baselines/llm/eval_utils/evaluator.py`` and ``examples/play_with_human_interface.py`` both
call into this module so trajectories collected by any of the three tracks
land in the same file formats and can be loaded/replayed with the same
tooling, regardless of who (or what) generated the episode.
"""

import gzip
import json
import logging
import pickle
import re
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Surrogate characters (U+D800-U+DFFF) and null bytes are not valid in JSON
# strings. LLM thinking output occasionally contains them, causing json.loads
# to fail when reading the JSONL back. Strip them before serializing.
_INVALID_JSON_STR = re.compile(r"[\x00\ud800-\udfff]", re.UNICODE)


def sanitize_str(s):
    """Remove characters that are invalid in JSON strings."""
    if isinstance(s, str):
        return _INVALID_JSON_STR.sub("�", s)
    return s


def sanitize_record(obj):
    """Recursively sanitize all string values in a dict/list for JSON safety."""
    if isinstance(obj, dict):
        return {k: sanitize_record(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_record(v) for v in obj]
    if isinstance(obj, str):
        return sanitize_str(obj)
    return obj


def safe_json_dumps(obj):
    """json.dumps with a sanitization fallback for invalid string content."""
    try:
        return json.dumps(obj)
    except (ValueError, UnicodeEncodeError):
        return json.dumps(sanitize_record(obj))


def save_trajectory_npz(save_path, obs, actions, rewards, dones, text_obs=None, text_actions=None):
    """Save one episode's trajectory as a compressed .npz.

    Numeric arrays match the RL eval format (``baselines/utils.py``
    ``_run_eval_sequential``):
      obs       (T, num_agents, obs_dim) - raw symbolic obs (pre-step)
      actions   (T, num_agents)          - discrete action indices
      rewards   (T, num_agents)          - per-agent rewards
      dones     (T,)                     - episode-done flag
      timesteps (T,)                     - step indices

    Optional extra fields (load with ``allow_pickle=True``), included whenever
    the caller passes them:
      text_obs     (T, num_agents) - long-term text observation seen by the agent
      text_actions (T, num_agents) - canonical/human-readable action name chosen

    Returns the path written, or None if there was nothing to save.
    """
    obs = list(obs)
    T = len(obs)
    if T == 0 or any(o is None for o in obs):
        return None
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(
        obs=np.stack(obs),
        actions=np.stack(list(actions)),
        rewards=np.stack(list(rewards)),
        dones=np.array(list(dones), dtype=bool),
        timesteps=np.arange(T, dtype=np.int32),
    )
    if text_obs is not None:
        arrays["text_obs"] = np.array(text_obs, dtype=object)
    if text_actions is not None:
        arrays["text_actions"] = np.array(text_actions, dtype=object)
    try:
        np.savez_compressed(save_path, **arrays)
        logger.info(f"Saved trajectory ({T} steps) to {save_path}")
        return str(save_path)
    except Exception as e:
        logger.warning(f"Failed to save trajectory to {save_path}: {e}")
        return None


def save_state_bundle(save_path, states, static_env_params):
    """Save pre-step env states to a compressed pickle for exact replay.

    Kept out of the ``.npz`` because EnvState is a nested flax/JAX pytree, not
    a plain numeric array. A replay script should prefer this file whenever it
    exists, which avoids seed-based reconstruction entirely.

    Returns the path written, or None if there was nothing to save.
    """
    if not states:
        return None
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "states": states,
        "static_env_params": static_env_params,
        "num_steps": len(states),
    }
    try:
        with gzip.open(save_path, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Saved state bundle ({len(states)} steps) to {save_path}")
        return str(save_path)
    except Exception as e:
        logger.warning(f"Failed to save state bundle to {save_path}: {e}")
        return None


def load_state_bundle(load_path):
    """Load a state bundle written by ``save_state_bundle``.

    Returns ``(states, static_env_params)``.
    """
    with gzip.open(load_path, "rb") as handle:
        payload = pickle.load(handle)
    return payload["states"], payload["static_env_params"]
