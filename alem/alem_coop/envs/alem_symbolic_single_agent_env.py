from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from alem.environment_base.jaxmarl_compat import spaces

if TYPE_CHECKING:
    from jaxtyping import Array, Bool, Float

from ..action_masking import compute_action_mask_single_agent
from ..constants import OBS_DIM, Action
from .alem_symbolic_env import AlemCoopSymbolicEnv

# ── What changes vs the multi-agent env ───────────────────────────────────────
#
# ACTION SPACE  55 → 42
#   Actions 0-41  (NOOP … ENCHANT_BOW)  are kept — identical to standard Craftax.
#   Actions 42-54 (REQUEST_*, BUILD_*, GIVE) are dropped: they are always masked
#   to False for player_count=1 (no teammates), so the policy can never learn to
#   use them.  Removing them shrinks the output head and eliminates wasted
#   gradient signal.
#
# OBSERVATION  9 476 → 8 972  (504 dead dims removed)
#
#   Spatial (per cell, 95 channels → 90):
#     Dropped channels 89-93 per cell (5 × 99 = 495 dims):
#       coord_obs      (2): coordination type+magnitude, soft/hard flag
#       mob_coord_map  (2): requires_coord, is_hard_coord flags for elite mobs
#       handover_obs   (1): pending handover time remaining
#     All three are always zero when coordination_difficulty="none" and
#     player_count=1.  The light channel (offset 94) is kept.
#
#   Teammate dashboard (14 dims → 5):
#     Dropped indices 5-13 (9 dims):
#       requested_material: one-hot encoding of the pending REQUEST_* action.
#       Always zero for single-agent (no one to request from).
#     Kept: health (1), alive (1), specialization (3).
#     Teammate directions (8 dims, separate concat block) are kept.
#
# ─────────────────────────────────────────────────────────────────────────────

# Actions 0..ENCHANT_BOW inclusive (matches standard Craftax action space).
_SA_ACTION_DIM = Action.ENCHANT_BOW.value + 1  # 42

# Per-cell channel layout for player_count=1 (total 95):
#   BlockType(42) | ItemType(5) | mobs(40) | teammate_map(2) |
#   coord_obs(2)* | mob_coord(2)* | handover(1)* | light(1)
#   * = dead for single-agent with coord_difficulty=none
_CHANNELS_PER_CELL = 95
_DEAD_CELL_OFFSET = 89  # coord_obs starts here
_DEAD_CELL_COUNT = 5  # coord_obs(2) + mob_coord(2) + handover(1)

# Teammate-dashboard layout (14 dims for player_count=1):
#   health(1) | alive(1) | specialization(3) | request_material(9)*
#   * = dead for single-agent
_DASH_DEAD_OFFSET = 5
_DASH_DEAD_COUNT = 9  # REQUEST_FOOD … REQUEST_SAPPHIRE


def _build_keep_indices() -> np.ndarray:
    """Return sorted integer indices of obs columns to keep (shape: 8972)."""
    num_cells = OBS_DIM[0] * OBS_DIM[1]  # 99
    flat_map_end = num_cells * _CHANNELS_PER_CELL  # 9405

    mask = np.ones(9476, dtype=bool)

    # Drop 5 dead channels from each of the 99 spatial cells.
    for c in range(num_cells):
        base = c * _CHANNELS_PER_CELL
        mask[base + _DEAD_CELL_OFFSET : base + _DEAD_CELL_OFFSET + _DEAD_CELL_COUNT] = False

    # Drop request_material from the teammate-dashboard block.
    mask[flat_map_end + _DASH_DEAD_OFFSET : flat_map_end + _DASH_DEAD_OFFSET + _DASH_DEAD_COUNT] = (
        False
    )

    return np.where(mask)[0]  # integer indices, shape (8972,)


class AlemCoopSymbolicSingleAgentEnv(AlemCoopSymbolicEnv):
    """Single-agent symbolic env with multi-agent dead-weight removed.

    Obs: 8972 dims (vs 9476).  Actions: 42 (vs 55).
    Use ENV_NAME=Alem-SingleAgent-Symbolic when a single-agent variant is needed.
    """

    # Callers can read this attribute to pick the right mask function.
    action_mask_fn = staticmethod(compute_action_mask_single_agent)

    def __init__(self, env_params=None, static_env_params=None, compute_full_info: bool = True):
        """Initialize a symbolic environment constrained to one player.

        Args:
            env_params: Optional episode and gameplay parameters.
            static_env_params: Optional static parameters; defaults to one player.
            compute_full_info: Whether steps should calculate all score metrics.
        """
        from ..alem_state import StaticEnvParams

        if static_env_params is None:
            static_env_params = StaticEnvParams(player_count=1)
        super().__init__(
            env_params=env_params,
            static_env_params=static_env_params,
            compute_full_info=compute_full_info,
        )
        self._keep_idx = jnp.array(_build_keep_indices())

    # ── Shape reporting ───────────────────────────────────────────────────────

    def action_shape(self):
        """Return the coordination-free single-agent action space.

        Returns:
            Discrete space containing the 42 solo-playable actions.
        """
        return spaces.Discrete(_SA_ACTION_DIM)

    def get_flat_map_obs_shape(self):
        """Return the map feature count after removing coordination channels.

        Returns:
            Number of retained spatial observation features.
        """
        num_cells = OBS_DIM[0] * OBS_DIM[1]
        return num_cells * (_CHANNELS_PER_CELL - _DEAD_CELL_COUNT)  # 99 * 90 = 8910

    def get_teammate_dashboard_obs_shape(self):
        """Return the dashboard size after removing request features.

        Returns:
            Number of retained single-player dashboard features.
        """
        return super().get_teammate_dashboard_obs_shape() - _DASH_DEAD_COUNT  # 22 - 9 = 13

    # ── Obs stripping ─────────────────────────────────────────────────────────

    def get_obs(self, state) -> dict[str, Float[Array, "obs_dim"]]:
        """Remove dead multi-agent channels from the symbolic observation.

        Args:
            state: Environment state to observe.

        Returns:
            Mapping containing the reduced observation for the sole agent.
        """
        full = super().get_obs(state)  # {agent: (9476,)}
        return {a: obs[self._keep_idx] for a, obs in full.items()}  # {agent: (8972,)}

    # ── Action mask truncation ────────────────────────────────────────────────

    def get_avail_actions(self, state) -> dict[str, Bool[Array, "action_dim"]]:
        """Remove coordination actions from the legal-action mask.

        Args:
            state: Environment state used to determine action validity.

        Returns:
            Mapping containing the reduced mask for the sole agent.
        """
        full = super().get_avail_actions(state)  # {agent: (55,)}
        return {a: m[:_SA_ACTION_DIM] for a, m in full.items()}  # {agent: (42,)}
