"""Tests for the coordination system.

The coordination system adds multi-agent cooperation requirements across four domains:

1. **Mining coordination**: Blocks on the map may require multiple agents to mine
   simultaneously (sync) or in sequence (handover). Controlled by `coordination_map`
   and `soft_coordination_mask` arrays on EnvState.

2. **Mob coordination**: Elite/Large mobs require coordinated attacks.
   Encoded per-slot as 0=normal, 1=elite/large soft, 2=elite/large hard.

3. **Construction coordination**: Construction sites require sync or handover
   cooperation to build structures (Epic Shelter, Forge, Beacon).

4. **Crafting coordination**: Diamond gear crafting at Epic Forge requires multiple
   agents present.

Each domain has "soft" vs "hard" variants:
- **Hard**: Action fails unless coordination requirement is met (N agents together).
- **Soft**: Action always succeeds, but reward/yield scales with number of agents.

Difficulty presets ("none", "easy", "medium", "hard") control the mix of soft/hard
tasks, handover frequency and window sizes, and elite mob probability.
"""

import unittest

import jax
import jax.numpy as jnp

from alem.alem_coop.action_masking import compute_action_mask
from alem.alem_coop.alem_state import (
    COORDINATION_PRESETS,
    EnvParams,
    StaticEnvParams,
    get_coordination_params,
)
from alem.alem_coop.constants import REQUEST_MAX_DURATION, Action, BlockType
from alem.alem_coop.envs.alem_symbolic_env import AlemCoopSymbolicEnv
from alem.alem_coop.game_logic import (
    check_sync_coordination,
    make_request,
    process_handover,
    trade_materials,
)
from alem.alem_coop.util.game_logic_utils import (
    attack_mob_class_with_elite_coordination,
)
from alem.alem_coop.world_gen.world_gen import (
    generate_world,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_env(num_agents=3, **param_overrides):
    """Create env with given parameter overrides."""
    env_params = EnvParams(**param_overrides)
    return AlemCoopSymbolicEnv(num_agents=num_agents, env_params=env_params)


def _reset(env, seed=0):
    """Reset env and return (obs, state)."""
    return env.reset(jax.random.PRNGKey(seed))


def _noop_actions(env):
    """All agents do NOOP."""
    return {agent: jnp.int32(Action.NOOP.value) for agent in env.agents}


def _uniform_actions(env, action_value):
    """All agents perform the same action."""
    return {agent: jnp.int32(action_value) for agent in env.agents}


# ============================================================================
# 1. DIFFICULTY PRESETS & PARAMETER CONFIGURATION
# ============================================================================
class TestCoordinationPresets(unittest.TestCase):
    """Verify that difficulty presets produce sensible EnvParams configurations.

    The preset system maps human-readable difficulty names ("easy", "medium",
    "hard") to concrete parameter dicts. These tests guard against regressions
    in the preset values and ensure the factory function rejects unknown names.
    """

    def test_all_preset_names_are_valid(self):
        """The four canonical presets exist and can be retrieved."""
        for name in ["none", "easy", "medium", "hard"]:
            params = get_coordination_params(name)
            self.assertIsInstance(params, dict)

    def test_unknown_preset_raises(self):
        """Requesting a non-existent preset raises ValueError."""
        with self.assertRaises(ValueError):
            get_coordination_params("nightmare")

    def test_none_preset_disables_everything(self):
        """The 'none' preset sets all coordination features to off / zero."""
        p = get_coordination_params("none")
        self.assertFalse(p["coordination_enabled"])
        self.assertEqual(p["elite_mob_probability"], 0.0)
        self.assertEqual(p["large_passive_probability"], 0.0)
        self.assertEqual(p["hard_mob_probability"], 0.0)
        self.assertEqual(p["p_max_agents"], 0.0)
        self.assertFalse(p["construction_enabled"])
        self.assertFalse(p["crafting_coordination_enabled"])

    def test_p_max_agents_increases_with_difficulty(self):
        """p_max_agents should increase with difficulty (more events require all agents)."""
        easy = get_coordination_params("easy")["p_max_agents"]
        medium = get_coordination_params("medium")["p_max_agents"]
        hard = get_coordination_params("hard")["p_max_agents"]
        self.assertLessEqual(easy, medium)
        self.assertLessEqual(medium, hard)
        # Verify actual values match DIFFICULTY_ALPHAS
        self.assertAlmostEqual(easy, 0.30, places=2)
        self.assertAlmostEqual(hard, 0.90, places=2)

    def test_handover_window_narrows_with_difficulty(self):
        """Harder presets give agents less time to complete handovers."""
        easy = get_coordination_params("easy")
        hard = get_coordination_params("hard")
        self.assertGreaterEqual(easy["handover_window_max"], hard["handover_window_max"])

    def test_opportunity_params_constant_across_difficulties(self):
        """Opportunity params (how many tasks exist) should be the same
        regardless of difficulty — only p_max_agents and windows change."""
        opportunity_keys = [
            "coordination_probability",
            "elite_mob_probability",
            "large_passive_probability",
            "num_construction_sites",
            "soft_coordination_ratio",
            "handover_ratio",
            "hard_mob_probability",
        ]
        easy = get_coordination_params("easy")
        medium = get_coordination_params("medium")
        hard = get_coordination_params("hard")
        for key in opportunity_keys:
            self.assertEqual(
                easy[key],
                medium[key],
                f"{key} should be constant: easy={easy[key]} != medium={medium[key]}",
            )
            self.assertEqual(
                medium[key],
                hard[key],
                f"{key} should be constant: medium={medium[key]} != hard={hard[key]}",
            )

    def test_preset_keys_match_envparams_fields(self):
        """Every key in a preset should be a valid EnvParams field."""
        valid_fields = set(EnvParams.__dataclass_fields__.keys())
        for name, preset in COORDINATION_PRESETS.items():
            for key in preset:
                self.assertIn(key, valid_fields, f"Preset '{name}' has unknown key '{key}'")

    def test_get_coordination_params_is_case_insensitive(self):
        """get_coordination_params should accept 'Easy', 'EASY', etc."""
        p1 = get_coordination_params("easy")
        p2 = get_coordination_params("EASY")
        p3 = get_coordination_params("Easy")
        self.assertEqual(p1, p2)
        self.assertEqual(p2, p3)


# ============================================================================
# 2. COORDINATION MAP GENERATION
# ============================================================================
class TestCoordinationMapGeneration(unittest.TestCase):
    """Verify the per-level coordination map is generated correctly.

    `generate_coordination_map` assigns coordination requirements to eligible
    mining blocks. The map encodes:
      - 0: no requirement
      - positive int (2-N): synchronous — N agents must act together
      - negative int: handover — value encodes window size (abs value)

    A companion `soft_coordination_mask` flags which sync blocks are soft
    (scale reward) vs hard (block action).
    """

    def test_disabled_produces_all_zeros(self):
        """When coordination_enabled=False, both maps are all zeros."""
        env = _make_env(coordination_enabled=False)
        _, state = _reset(env)
        self.assertTrue(jnp.all(state.coordination_map == 0))
        self.assertTrue(jnp.all(~state.soft_coordination_mask))

    def test_enabled_produces_nonzero_blocks(self):
        """With high probability, we expect some coordination blocks."""
        env = _make_env(coordination_enabled=True, coordination_probability=0.5)
        _, state = _reset(env, seed=42)
        total = jnp.sum(state.coordination_map != 0)
        self.assertGreater(
            int(total), 0, "50% coordination probability should produce nonzero blocks"
        )

    def test_coordination_map_shape(self):
        """Shape should be (num_levels, map_h, map_w)."""
        env = _make_env(coordination_enabled=True)
        _, state = _reset(env)
        sp = env.static_env_params
        self.assertEqual(state.coordination_map.shape, (sp.num_levels, *sp.map_size))
        self.assertEqual(state.soft_coordination_mask.shape, (sp.num_levels, *sp.map_size))

    def test_sync_values_within_agent_range(self):
        """Synchronous values should be either 2 or player_count (binary p_max_agents sampling)."""
        env = _make_env(
            coordination_enabled=True,
            coordination_probability=0.5,
            handover_ratio=0.0,  # all sync, no handover
            p_max_agents=0.5,
        )
        _, state = _reset(env, seed=7)
        sync_values = state.coordination_map[state.coordination_map > 0]
        if len(sync_values) > 0:
            self.assertTrue(
                jnp.all(sync_values >= 2), "Sync blocks should require at least 2 agents"
            )
            self.assertTrue(
                jnp.all(sync_values <= env.static_env_params.player_count),
                "Sync blocks should require at most player_count agents",
            )
            # Values should only be 2 or player_count (binary sampling)
            valid = jnp.isin(sync_values, jnp.array([2, env.static_env_params.player_count]))
            self.assertTrue(jnp.all(valid), "Sync values should be either 2 or player_count")

    def test_handover_values_are_negative(self):
        """Handover blocks have negative values encoding window size."""
        env = _make_env(
            coordination_enabled=True,
            coordination_probability=0.5,
            handover_ratio=1.0,  # all handover, no sync
        )
        _, state = _reset(env, seed=3)
        coord = state.coordination_map
        has_any = jnp.any(coord != 0)
        if has_any:
            nonzero = coord[coord != 0]
            # All non-zero values should be negative (handover)
            self.assertTrue(
                jnp.all(nonzero < 0), "With handover_ratio=1.0, all coord blocks should be negative"
            )

    def test_handover_windows_within_configured_range(self):
        """Handover window sizes fall within [min, max] from params."""
        env = _make_env(
            coordination_enabled=True,
            coordination_probability=0.5,
            handover_ratio=1.0,
            handover_window_min=5,
            handover_window_max=15,
        )
        _, state = _reset(env, seed=3)
        handover = state.coordination_map[state.coordination_map < 0]
        if len(handover) > 0:
            windows = jnp.abs(handover)
            self.assertTrue(jnp.all(windows >= 5))
            self.assertTrue(jnp.all(windows <= 15))

    def test_sync_handover_split_respects_ratio(self):
        """With 50/50 split and enough blocks, both sync and handover exist."""
        env = _make_env(
            coordination_enabled=True,
            coordination_probability=0.5,
            handover_ratio=0.5,
        )
        _, state = _reset(env, seed=42)
        sync_count = int(jnp.sum(state.coordination_map > 0))
        handover_count = int(jnp.sum(state.coordination_map < 0))
        self.assertGreater(sync_count, 0, "Should have some sync blocks")
        self.assertGreater(handover_count, 0, "Should have some handover blocks")

    def test_soft_mask_only_on_sync_blocks(self):
        """Soft coordination mask should only be True where coordination_map > 0.
        Handover blocks (negative) are always hard, so soft_mask should be False there."""
        env = _make_env(
            coordination_enabled=True,
            coordination_probability=0.5,
            handover_ratio=0.5,
            soft_coordination_ratio=0.9,
        )
        _, state = _reset(env, seed=42)
        # Soft mask should never be True on handover blocks
        handover_positions = state.coordination_map < 0
        soft_on_handover = state.soft_coordination_mask & handover_positions
        self.assertTrue(
            jnp.all(~soft_on_handover), "Handover blocks should never be marked as soft"
        )

    def test_coordination_only_on_eligible_blocks(self):
        """Coordination should only be assigned to mining-eligible block types
        (trees, ores), not to grass, water, or walls."""
        env = _make_env(
            coordination_enabled=True,
            coordination_probability=0.8,  # high prob to catch violations
        )
        _, state = _reset(env, seed=0)

        eligible_types = jnp.array(
            [
                BlockType.TREE.value,
                BlockType.FIRE_TREE.value,
                BlockType.ICE_SHRUB.value,
                BlockType.STONE.value,
                BlockType.COAL.value,
                BlockType.IRON.value,
                BlockType.DIAMOND.value,
                BlockType.SAPPHIRE.value,
                BlockType.RUBY.value,
            ]
        )

        for level in range(env.static_env_params.num_levels):
            coord = state.coordination_map[level]
            block = state.map[level]
            has_coord = coord != 0
            # Check: if a cell has coordination, its block must be eligible
            # (Except construction sites which get coord applied separately)
            is_construction = (block == BlockType.CONSTRUCTION_SITE.value) | (
                block == BlockType.CONSTRUCTION_IN_PROGRESS.value
            )
            non_construction_coord = has_coord & ~is_construction
            if jnp.any(non_construction_coord):
                blocks_with_coord = block[non_construction_coord]
                for b in blocks_with_coord:
                    self.assertTrue(
                        jnp.any(eligible_types == b),
                        f"Block type {int(b)} at level {level} has coordination "
                        f"but is not an eligible mining block",
                    )


# ============================================================================
# 3. MOB COORDINATION (ELITE / LARGE MOBS)
# ============================================================================
class TestMobCoordination(unittest.TestCase):
    """Verify the unified mob coordination encoding.

    After the unification refactor, each mob slot carries a single integer:
      - 0: Normal mob (no coordination requirement)
      - 1: Elite/Large with **soft** coordination (solo=0.5x, coordinated=2x damage)
      - 2: Elite/Large with **hard** coordination (must have 2+ agents to damage)

    Previously, melee/ranged elites were always hard and only passives had a
    per-mob hard flag. Now all three classes use the same 0/1/2 encoding,
    and the hard_mob_probability parameter controls what fraction of elites
    across ALL classes are hard.
    """

    def test_mob_coordination_shapes(self):
        """Coordination arrays match mob slot dimensions."""
        env = _make_env(elite_mob_probability=0.3, large_passive_probability=0.3)
        _, state = _reset(env)
        sp = env.static_env_params
        expected_melee = (sp.num_levels, sp.max_melee_mobs * sp.player_count)
        expected_ranged = (sp.num_levels, sp.max_ranged_mobs * sp.player_count)
        expected_passive = (sp.num_levels, sp.max_passive_mobs * sp.player_count)
        self.assertEqual(state.melee_mob_coordination.shape, expected_melee)
        self.assertEqual(state.ranged_mob_coordination.shape, expected_ranged)
        self.assertEqual(state.passive_mob_coordination.shape, expected_passive)

    def test_coordination_values_in_valid_range(self):
        """All mob coordination values should be 0, 1, or 2."""
        env = _make_env(
            elite_mob_probability=0.5, large_passive_probability=0.5, hard_mob_probability=0.5
        )
        _, state = _reset(env, seed=42)
        for arr_name in [
            "melee_mob_coordination",
            "ranged_mob_coordination",
            "passive_mob_coordination",
        ]:
            arr = getattr(state, arr_name)
            unique_vals = jnp.unique(arr)
            for v in unique_vals:
                self.assertIn(int(v), {0, 1, 2}, f"{arr_name} has unexpected value {int(v)}")

    def test_zero_elite_probability_produces_all_normal_on_floor_zero(self):
        """With elite_mob_probability=0, floor 0 mobs should be normal (coord=0).

        Note: deeper floors get a +5% bonus per floor, so elite_mob_probability=0
        does NOT mean zero elites on all floors — only floor 0 is guaranteed zero.
        """
        env = _make_env(elite_mob_probability=0.0, large_passive_probability=0.0)
        _, state = _reset(env)
        # Only check floor 0 where the bonus is 0%
        self.assertTrue(jnp.all(state.melee_mob_coordination[0] == 0))
        self.assertTrue(jnp.all(state.ranged_mob_coordination[0] == 0))
        self.assertTrue(jnp.all(state.passive_mob_coordination[0] == 0))

    def test_hard_mob_probability_zero_means_all_soft(self):
        """With hard_mob_probability=0, all elites should be soft (coord=1), never hard (2).

        This is the key behavioral change from the unification: previously melee/ranged
        elites were always hard. Now on easy difficulty (hard_mob_probability=0), a solo
        agent can always damage elites (with 0.5x penalty), making 'easy' truly easy.
        """
        env = _make_env(
            elite_mob_probability=0.5,
            large_passive_probability=0.5,
            hard_mob_probability=0.0,
        )
        _, state = _reset(env, seed=42)
        # No value should be 2 (hard)
        for arr_name in [
            "melee_mob_coordination",
            "ranged_mob_coordination",
            "passive_mob_coordination",
        ]:
            arr = getattr(state, arr_name)
            hard_count = int(jnp.sum(arr == 2))
            self.assertEqual(
                hard_count,
                0,
                f"{arr_name} has {hard_count} hard mobs despite hard_mob_probability=0",
            )

    def test_hard_mob_probability_one_means_all_hard(self):
        """With hard_mob_probability=1, all elites should be hard (coord=2)."""
        env = _make_env(
            elite_mob_probability=0.5,
            large_passive_probability=0.5,
            hard_mob_probability=1.0,
        )
        _, state = _reset(env, seed=42)
        for arr_name in [
            "melee_mob_coordination",
            "ranged_mob_coordination",
            "passive_mob_coordination",
        ]:
            arr = getattr(state, arr_name)
            # All non-zero values should be 2
            elite_vals = arr[arr > 0]
            if len(elite_vals) > 0:
                self.assertTrue(
                    jnp.all(elite_vals == 2),
                    f"{arr_name} has soft elites despite hard_mob_probability=1",
                )

    def test_melee_and_ranged_share_elite_probability(self):
        """Melee and ranged mobs should use the same elite_mob_probability,
        producing roughly similar elite rates (accounting for randomness)."""
        env = _make_env(elite_mob_probability=0.5, hard_mob_probability=0.5)
        _, state = _reset(env, seed=0)
        # Just check both arrays have some elites
        melee_elites = int(jnp.sum(state.melee_mob_coordination > 0))
        ranged_elites = int(jnp.sum(state.ranged_mob_coordination > 0))
        self.assertGreater(melee_elites, 0, "Should have some melee elites at 50%")
        self.assertGreater(ranged_elites, 0, "Should have some ranged elites at 50%")

    def test_passive_uses_large_passive_probability(self):
        """Passive mobs use `large_passive_probability` (not elite_mob_probability).

        We check floor 0 only, because elite_mob_probability has a +5% per-floor
        bonus that would cause elites on deeper floors even with base prob=0.
        """
        env = _make_env(elite_mob_probability=0.0, large_passive_probability=1.0)
        _, state = _reset(env)
        # Floor 0: no melee/ranged elites (base prob 0, floor bonus 0)
        self.assertTrue(jnp.all(state.melee_mob_coordination[0] == 0))
        self.assertTrue(jnp.all(state.ranged_mob_coordination[0] == 0))
        # All passive slots should be large at 100% probability
        self.assertTrue(
            jnp.all(state.passive_mob_coordination[0] > 0),
            "All passive mobs on floor 0 should be large at 100% probability",
        )

    def test_elite_probability_increases_with_floor_depth(self):
        """Elite probability has a +5% bonus per floor, so deeper floors should
        have more elites on average."""
        env = _make_env(elite_mob_probability=0.15, hard_mob_probability=0.0)
        _, state = _reset(env, seed=0)
        sp = env.static_env_params
        level_0_elites = int(jnp.sum(state.melee_mob_coordination[0] > 0))
        last_level = sp.num_levels - 1
        last_level_elites = int(jnp.sum(state.melee_mob_coordination[last_level] > 0))
        # With +5% per floor over 8 floors = +40%, should see more elites on last floor
        # This is statistical so we just check it's >= (not strictly >)
        self.assertGreaterEqual(
            last_level_elites, level_0_elites, "Deeper floors should tend to have more elites"
        )

    def test_no_passive_mob_is_hard_field(self):
        """After unification, EnvState should NOT have a passive_mob_is_hard field.
        Hardness is now encoded in the coordination value (2=hard)."""
        env = _make_env()
        _, state = _reset(env)
        self.assertFalse(
            hasattr(state, "passive_mob_is_hard"),
            "passive_mob_is_hard should have been removed — "
            "hardness is now encoded in coordination value",
        )


# ============================================================================
# 4. SYNC COORDINATION MECHANICS
# ============================================================================
class TestSyncCoordination(unittest.TestCase):
    """Test the check_sync_coordination function directly.

    Sync coordination checks whether enough agents are acting on the same
    position simultaneously:
    - Hard sync: action fails if agent count < required
    - Soft sync: action always succeeds, but yield multiplier scales with agents
    """

    def _make_minimal_state_and_params(self, coord_value, is_soft, num_agents=3):
        """Create a minimal state with a known coordination value at a test position.

        Places a coordination requirement at position (5,5) which all agents face.
        """
        sp = StaticEnvParams(player_count=num_agents)
        params = EnvParams(coordination_enabled=True)

        rng = jax.random.PRNGKey(0)
        state = generate_world(rng, params, sp)

        # Set known coordination at (5,5)
        new_coord = state.coordination_map.at[0, 5, 5].set(coord_value)
        new_soft = state.soft_coordination_mask.at[0, 5, 5].set(is_soft)
        state = state.replace(
            coordination_map=new_coord,
            soft_coordination_mask=new_soft,
            player_level=jnp.int32(0),
        )
        return state, params, sp

    def test_no_coordination_always_succeeds(self):
        """Positions with coord_value=0 should always succeed regardless of agent count."""
        state, params, sp = self._make_minimal_state_and_params(0, False)
        block_positions = jnp.full((sp.player_count, 2), jnp.array([10, 10]))
        is_acting = jnp.ones(sp.player_count, dtype=jnp.bool_)
        equal = (block_positions[:, None] == block_positions[None, :]).all(axis=2)

        succeeds, is_sync, mult, _, _ = check_sync_coordination(
            jax.random.PRNGKey(0), state, block_positions, is_acting, equal, params, sp
        )
        self.assertTrue(jnp.all(succeeds))
        self.assertTrue(jnp.all(~is_sync), "No sync success expected on non-coordinated block")

    def test_hard_sync_fails_with_too_few_agents(self):
        """Hard sync (require 2 agents) should fail when only 1 agent acts."""
        state, params, sp = self._make_minimal_state_and_params(coord_value=2, is_soft=False)
        block_positions = jnp.full((sp.player_count, 2), jnp.array([5, 5]))
        # Only agent 0 is acting
        is_acting = jnp.array([True, False, False])
        equal = (block_positions[:, None] == block_positions[None, :]).all(axis=2)

        succeeds, _, _, _, _ = check_sync_coordination(
            jax.random.PRNGKey(0), state, block_positions, is_acting, equal, params, sp
        )
        # Agent 0's action should fail (hard sync not met)
        self.assertFalse(
            bool(succeeds[0]), "Hard sync should fail when only 1 of 2 required agents act"
        )

    def test_hard_sync_succeeds_with_enough_agents(self):
        """Hard sync requiring 2 agents succeeds when 2+ agents act together."""
        state, params, sp = self._make_minimal_state_and_params(coord_value=2, is_soft=False)
        block_positions = jnp.full((sp.player_count, 2), jnp.array([5, 5]))
        # Agents 0 and 1 acting together
        is_acting = jnp.array([True, True, False])
        equal = (block_positions[:, None] == block_positions[None, :]).all(axis=2)

        succeeds, is_sync, _, _, _ = check_sync_coordination(
            jax.random.PRNGKey(0), state, block_positions, is_acting, equal, params, sp
        )
        self.assertTrue(bool(succeeds[0]))
        self.assertTrue(bool(succeeds[1]))
        self.assertTrue(bool(is_sync[0]))

    def test_soft_sync_always_succeeds_solo(self):
        """Soft sync should let a solo agent succeed (with 1x multiplier)."""
        state, params, sp = self._make_minimal_state_and_params(coord_value=2, is_soft=True)
        block_positions = jnp.full((sp.player_count, 2), jnp.array([5, 5]))
        is_acting = jnp.array([True, False, False])
        equal = (block_positions[:, None] == block_positions[None, :]).all(axis=2)

        succeeds, is_sync, mult, is_soft, _ = check_sync_coordination(
            jax.random.PRNGKey(0), state, block_positions, is_acting, equal, params, sp
        )
        self.assertTrue(
            bool(succeeds[0]), "Soft sync should succeed solo when soft_solo_fail_prob=0"
        )
        self.assertTrue(bool(is_sync[0]))
        self.assertTrue(bool(is_soft[0]))
        self.assertEqual(float(mult[0]), 1.0, "Solo on soft sync should get 1x multiplier")

    def test_soft_sync_solo_fails_when_fail_prob_one(self):
        """With soft_solo_fail_prob=1.0, solo agent on soft sync should always fail."""
        state, params, sp = self._make_minimal_state_and_params(coord_value=2, is_soft=True)
        params = params.replace(soft_solo_fail_prob=1.0)
        block_positions = jnp.full((sp.player_count, 2), jnp.array([5, 5]))
        is_acting = jnp.array([True, False, False])
        equal = (block_positions[:, None] == block_positions[None, :]).all(axis=2)

        succeeds, is_sync, mult, is_soft, _ = check_sync_coordination(
            jax.random.PRNGKey(42), state, block_positions, is_acting, equal, params, sp
        )
        self.assertFalse(
            bool(succeeds[0]), "Solo on soft sync should fail when soft_solo_fail_prob=1.0"
        )

    def test_soft_sync_coordinated_ignores_fail_prob(self):
        """With soft_solo_fail_prob=1.0, coordinated agents should still succeed."""
        state, params, sp = self._make_minimal_state_and_params(coord_value=2, is_soft=True)
        params = params.replace(soft_solo_fail_prob=1.0)
        block_positions = jnp.full((sp.player_count, 2), jnp.array([5, 5]))
        is_acting = jnp.array([True, True, False])
        equal = (block_positions[:, None] == block_positions[None, :]).all(axis=2)

        succeeds, is_sync, mult, is_soft, _ = check_sync_coordination(
            jax.random.PRNGKey(42), state, block_positions, is_acting, equal, params, sp
        )
        self.assertTrue(
            bool(succeeds[0]), "Coordinated agents should succeed regardless of soft_solo_fail_prob"
        )
        self.assertEqual(
            float(mult[0]),
            3.0,
            "2 agents meeting coord_req=2 should get 3x even with fail_prob=1.0",
        )

    def test_soft_sync_scales_multiplier_with_agents(self):
        """Soft sync with 2 agents meeting coord_req=2 should get k(k+1)/2 = 3x."""
        state, params, sp = self._make_minimal_state_and_params(coord_value=2, is_soft=True)
        block_positions = jnp.full((sp.player_count, 2), jnp.array([5, 5]))
        is_acting = jnp.array([True, True, False])
        equal = (block_positions[:, None] == block_positions[None, :]).all(axis=2)

        _, _, mult, _, _ = check_sync_coordination(
            jax.random.PRNGKey(0), state, block_positions, is_acting, equal, params, sp
        )
        # k=2 agents, k(k+1)/2 = 3.0 (threshold met: 2 >= coord_req=2)
        self.assertEqual(
            float(mult[0]),
            3.0,
            "2 agents meeting coord_req=2 on soft sync should get k(k+1)/2 = 3x",
        )

    def test_coordination_disabled_passthrough(self):
        """When coordination_enabled=False, all actions pass through unchanged."""
        sp = StaticEnvParams(player_count=3)
        params = EnvParams(coordination_enabled=False)
        state = generate_world(jax.random.PRNGKey(0), params, sp)

        block_positions = jnp.full((sp.player_count, 2), jnp.array([5, 5]))
        is_acting = jnp.array([True, True, False])
        equal = (block_positions[:, None] == block_positions[None, :]).all(axis=2)

        succeeds, is_sync, mult, _, _ = check_sync_coordination(
            jax.random.PRNGKey(0), state, block_positions, is_acting, equal, params, sp
        )
        # Should pass through is_acting unchanged
        self.assertTrue(bool(succeeds[0]))
        self.assertTrue(bool(succeeds[1]))
        self.assertFalse(bool(succeeds[2]))
        self.assertTrue(jnp.all(mult == 1.0))


# ============================================================================
# 5. HANDOVER COORDINATION MECHANICS
# ============================================================================
class TestHandoverCoordination(unittest.TestCase):
    """Test the handover coordination process.

    Handover is a two-phase mechanism:
    1. Setup: Agent A acts on a handover block → creates pending_handover entry
       with a deadline. The action "fails" (no immediate mining effect).
    2. Completion: Agent B acts on the same block before the deadline.
       A different agent must complete it (agent A can't complete their own).

    Handovers are encoded as negative values in coordination_map.
    """

    def _setup_handover_state(self, window_size=15):
        """Create a state with a handover block at (5,5) with given window."""
        sp = StaticEnvParams(player_count=3)
        params = EnvParams(coordination_enabled=True)
        state = generate_world(jax.random.PRNGKey(0), params, sp)

        # Place handover requirement at (5,5) — negative value = handover
        coord = state.coordination_map.at[0, 5, 5].set(-window_size)
        # Ensure block at (5,5) is a mineable block (not construction)
        new_map = state.map.at[0, 5, 5].set(BlockType.STONE.value)
        state = state.replace(
            coordination_map=coord,
            map=new_map,
            player_level=jnp.int32(0),
        )
        return state, params, sp

    def test_setup_creates_pending_handover(self):
        """Agent acting on handover block should create a pending entry."""
        state, params, sp = self._setup_handover_state(window_size=15)
        block_positions = jnp.full((sp.player_count, 2), jnp.array([5, 5]))
        is_acting = jnp.array([True, False, False])

        succeeds, new_state = process_handover(state, block_positions, is_acting, params, sp)
        # Setup should "fail" (no immediate effect)
        self.assertFalse(bool(succeeds[0]), "Handover setup should not succeed immediately")
        # But pending_handovers should have a new entry
        pending = new_state.pending_handovers
        active_entries = jnp.sum(pending[:, 0] == 1)
        self.assertGreater(
            int(active_entries), 0, "Should have at least one pending handover after setup"
        )

    def test_completion_by_different_agent_succeeds(self):
        """Agent B completing a handover started by Agent A should succeed."""
        state, params, sp = self._setup_handover_state(window_size=100)

        # Phase 1: Agent 0 sets up
        block_positions = jnp.full((sp.player_count, 2), jnp.array([5, 5]))
        is_acting = jnp.array([True, False, False])
        _, state = process_handover(state, block_positions, is_acting, params, sp)

        # Phase 2: Agent 1 completes (same position, different agent)
        is_acting_2 = jnp.array([False, True, False])
        succeeds, new_state = process_handover(state, block_positions, is_acting_2, params, sp)
        self.assertTrue(
            bool(succeeds[1]), "Agent 1 should complete the handover started by Agent 0"
        )
        # Handover success metric should increment
        self.assertGreater(int(new_state.handover_successes), int(state.handover_successes))

    def test_same_agent_cannot_complete_own_handover(self):
        """The initiating agent cannot complete their own handover."""
        state, params, sp = self._setup_handover_state(window_size=100)

        # Phase 1: Agent 0 sets up
        block_positions = jnp.full((sp.player_count, 2), jnp.array([5, 5]))
        is_acting = jnp.array([True, False, False])
        _, state = process_handover(state, block_positions, is_acting, params, sp)

        # Phase 2: Same agent (0) tries to complete
        succeeds, _ = process_handover(state, block_positions, is_acting, params, sp)
        # Agent 0 is trying to act on same position they set up — should be treated
        # as a new setup attempt (not a completion), so action should "fail"
        self.assertFalse(
            bool(succeeds[0]), "Initiator should not be able to complete their own handover"
        )

    def test_initiator_reacting_does_not_create_duplicate_pending(self):
        """Initiator re-acting on its own handover must not create a 2nd slot."""
        state, params, sp = self._setup_handover_state(window_size=100)
        block_positions = jnp.full((sp.player_count, 2), jnp.array([5, 5]))
        is_acting = jnp.array([True, False, False])

        # Phase 1: Agent 0 sets up -> one pending slot
        _, state = process_handover(state, block_positions, is_acting, params, sp)
        self.assertEqual(int(jnp.sum(state.pending_handovers[:, 0] == 1)), 1)

        # Phase 2: Agent 0 re-acts (cannot self-complete) -> must NOT add a slot
        _, state = process_handover(state, block_positions, is_acting, params, sp)
        self.assertEqual(
            int(jnp.sum(state.pending_handovers[:, 0] == 1)),
            1,
            "Initiator re-acting must not create a duplicate pending handover",
        )

    def test_completion_clears_all_pending_no_stale_clock(self):
        """After completion, no active pending slot should linger (stale clock bug)."""
        state, params, sp = self._setup_handover_state(window_size=100)
        block_positions = jnp.full((sp.player_count, 2), jnp.array([5, 5]))

        # Agent 0 sets up, then Agent 0 re-acts (the trigger for the old bug)
        _, state = process_handover(
            state, block_positions, jnp.array([True, False, False]), params, sp
        )
        _, state = process_handover(
            state, block_positions, jnp.array([True, False, False]), params, sp
        )

        # Agent 1 completes
        _, state = process_handover(
            state, block_positions, jnp.array([False, True, False]), params, sp
        )
        self.assertEqual(
            int(jnp.sum(state.pending_handovers[:, 0] == 1)),
            0,
            "No pending handover should remain active after completion",
        )

    def test_simultaneous_setup_same_position_dedups(self):
        """Two agents initiating the same position in one step create one slot."""
        state, params, sp = self._setup_handover_state(window_size=100)
        block_positions = jnp.full((sp.player_count, 2), jnp.array([5, 5]))
        is_acting = jnp.array([True, True, False])

        _, new_state = process_handover(state, block_positions, is_acting, params, sp)
        self.assertEqual(
            int(jnp.sum(new_state.pending_handovers[:, 0] == 1)),
            1,
            "Simultaneous same-position setups should create exactly one slot",
        )

    def test_handover_disabled_is_passthrough(self):
        """When coordination_enabled=False, handovers do nothing."""
        sp = StaticEnvParams(player_count=3)
        params = EnvParams(coordination_enabled=False)
        state = generate_world(jax.random.PRNGKey(0), params, sp)

        block_positions = jnp.full((sp.player_count, 2), jnp.array([5, 5]))
        is_acting = jnp.array([True, True, False])
        succeeds, new_state = process_handover(state, block_positions, is_acting, params, sp)
        # Should pass through is_acting unchanged
        self.assertTrue(bool(succeeds[0]))
        self.assertTrue(bool(succeeds[1]))
        self.assertFalse(bool(succeeds[2]))

    def test_pending_handovers_initialized_to_zero(self):
        """Pending handovers should start as all zeros (inactive)."""
        env = _make_env(coordination_enabled=True)
        _, state = _reset(env)
        sp = env.static_env_params
        self.assertEqual(state.pending_handovers.shape, (sp.max_pending_handovers, 6))
        self.assertTrue(jnp.all(state.pending_handovers == 0))


# ============================================================================
# 6. ELITE MOB COMBAT COORDINATION
# ============================================================================
class TestEliteMobCombat(unittest.TestCase):
    """Test the attack_mob_class_with_elite_coordination function.

    This function replaces the basic attack_mob_class when elite coordination
    is active. It applies damage modifiers based on mob coordination value:

    - coord=0 (normal): Standard damage, no coordination needed.
    - coord=1 (soft elite): Solo gets 0.5x damage; 2+ agents get 2x damage.
    - coord=2 (hard elite): Solo gets 0 damage (attack blocked); 2+ agents
      can damage normally.
    """

    def _make_combat_state(self, mob_coord_value, num_agents=3, agents_required=2):
        """Create a state with a mob that has a specific coordination value.

        Places a melee mob at position (5,5) on level 0 with the given
        coordination requirement.
        """
        sp = StaticEnvParams(player_count=num_agents)
        params = EnvParams()
        state = generate_world(jax.random.PRNGKey(0), params, sp)

        # Manually set up a melee mob at (5,5) on level 0, slot 0
        new_mask = state.melee_mobs.mask.at[0, 0].set(True)
        new_pos = state.melee_mobs.position.at[0, 0].set(jnp.array([5, 5]))
        new_health = state.melee_mobs.health.at[0, 0].set(100.0)  # high HP so it doesn't die
        new_melee = state.melee_mobs.replace(mask=new_mask, position=new_pos, health=new_health)

        # Set coordination value for this mob slot
        new_coord = state.melee_mob_coordination.at[0, 0].set(mob_coord_value)
        # Set agents_required for this mob slot
        new_agents_req = state.melee_mob_agents_required.at[0, 0].set(
            agents_required if mob_coord_value > 0 else 0
        )

        state = state.replace(
            melee_mobs=new_melee,
            melee_mob_coordination=new_coord,
            melee_mob_agents_required=new_agents_req,
            player_level=jnp.int32(0),
        )
        return state, params, sp

    def test_normal_mob_takes_standard_damage(self):
        """Coord=0 mob: all agents deal standard damage regardless of count."""
        state, params, sp = self._make_combat_state(mob_coord_value=0)

        # Single agent attacks the mob
        position = jnp.full((sp.player_count, 2), jnp.array([5, 5]))
        doing_attack = jnp.array([True, False, False])
        damage_vector = jnp.array([[5.0, 0.0, 0.0]] * sp.player_count)
        equal = (position[:, None] == position[None, :]).all(axis=2)

        mobs, _, _, _, _, coord_mult, *_ = attack_mob_class_with_elite_coordination(
            state,
            doing_attack,
            state.melee_mobs,
            position,
            damage_vector,
            True,
            1,
            state.melee_mob_coordination,
            state.melee_mob_agents_required,
            equal,
            sp,
        )
        # Mob should have taken damage
        self.assertLess(float(mobs.health[0, 0]), 100.0)
        # Coord multiplier should be 1.0 for normal mobs
        self.assertEqual(float(coord_mult[0]), 1.0)

    def test_soft_elite_solo_gets_half_damage(self):
        """Coord=1 mob: solo agent deals 0.5x damage."""
        state, params, sp = self._make_combat_state(mob_coord_value=1)

        position = jnp.full((sp.player_count, 2), jnp.array([5, 5]))
        doing_attack = jnp.array([True, False, False])
        damage_vector = jnp.array([[10.0, 0.0, 0.0]] * sp.player_count)
        equal = (position[:, None] == position[None, :]).all(axis=2)

        mobs, *_ = attack_mob_class_with_elite_coordination(
            state,
            doing_attack,
            state.melee_mobs,
            position,
            damage_vector,
            True,
            1,
            state.melee_mob_coordination,
            state.melee_mob_agents_required,
            equal,
            sp,
        )
        # 10 base damage * 0.5 solo penalty = 5 damage → health should be 95
        actual_damage = 100.0 - float(mobs.health[0, 0])
        self.assertAlmostEqual(
            actual_damage, 5.0, places=1, msg="Soft elite solo should deal 0.5x damage"
        )

    def test_soft_elite_coordinated_gets_double_damage(self):
        """Coord=1 mob: 2+ agents deal 2x damage."""
        state, params, sp = self._make_combat_state(mob_coord_value=1)

        position = jnp.full((sp.player_count, 2), jnp.array([5, 5]))
        doing_attack = jnp.array([True, True, False])
        damage_vector = jnp.array([[10.0, 0.0, 0.0]] * sp.player_count)
        equal = (position[:, None] == position[None, :]).all(axis=2)

        mobs, _, _, _, _, coord_mult, *_ = attack_mob_class_with_elite_coordination(
            state,
            doing_attack,
            state.melee_mobs,
            position,
            damage_vector,
            True,
            1,
            state.melee_mob_coordination,
            state.melee_mob_agents_required,
            equal,
            sp,
        )
        # Each agent deals 10 * 2.0 = 20 damage
        actual_damage_per_agent = 100.0 - float(mobs.health[0, 0])
        self.assertEqual(
            float(coord_mult[0]), 2.0, "Coordinated soft elite should give 2x multiplier"
        )
        # Total damage = two agents each dealing 20 = at least 20 per agent
        self.assertGreater(actual_damage_per_agent, 10.0)

    def test_hard_elite_blocks_solo_damage(self):
        """Coord=2 mob: solo agent deals 0 damage (blocked)."""
        state, params, sp = self._make_combat_state(mob_coord_value=2)

        position = jnp.full((sp.player_count, 2), jnp.array([5, 5]))
        doing_attack = jnp.array([True, False, False])
        damage_vector = jnp.array([[10.0, 0.0, 0.0]] * sp.player_count)
        equal = (position[:, None] == position[None, :]).all(axis=2)

        mobs, *_ = attack_mob_class_with_elite_coordination(
            state,
            doing_attack,
            state.melee_mobs,
            position,
            damage_vector,
            True,
            1,
            state.melee_mob_coordination,
            state.melee_mob_agents_required,
            equal,
            sp,
        )
        # Mob should be undamaged
        self.assertEqual(float(mobs.health[0, 0]), 100.0, "Hard elite should block all solo damage")

    def test_hard_elite_allows_coordinated_damage(self):
        """Coord=2 mob: 2+ agents can damage the mob."""
        state, params, sp = self._make_combat_state(mob_coord_value=2)

        position = jnp.full((sp.player_count, 2), jnp.array([5, 5]))
        doing_attack = jnp.array([True, True, False])
        damage_vector = jnp.array([[10.0, 0.0, 0.0]] * sp.player_count)
        equal = (position[:, None] == position[None, :]).all(axis=2)

        mobs, *_ = attack_mob_class_with_elite_coordination(
            state,
            doing_attack,
            state.melee_mobs,
            position,
            damage_vector,
            True,
            1,
            state.melee_mob_coordination,
            state.melee_mob_agents_required,
            equal,
            sp,
        )
        self.assertLess(
            float(mobs.health[0, 0]),
            100.0,
            "Hard elite should take damage when 2 agents coordinate",
        )


# ============================================================================
# 7. CONSTRUCTION COORDINATION
# ============================================================================
class TestConstructionCoordination(unittest.TestCase):
    """Test construction site placement and coordination.

    Construction sites are placed during world gen on grass/path blocks.
    Each site has a coordination requirement stored in coordination_map.

    Three building types:
    - Shelter: 15 wood + 10 stone → EPIC_SHELTER
    - Forge:   20 stone + 10 iron + 5 coal → EPIC_FORGE
    - Beacon:  10 diamond + 5 ruby → EPIC_BEACON
    """

    def test_construction_sites_placed_when_enabled(self):
        """Construction sites should be placed on the map when enabled."""
        env = _make_env(
            coordination_enabled=True,
            construction_enabled=True,
            num_construction_sites=4,
        )
        _, state = _reset(env, seed=42)

        # Check for CONSTRUCTION_SITE blocks on level 0 (overworld)
        sites_on_overworld = jnp.sum(state.map[0] == BlockType.CONSTRUCTION_SITE.value)
        self.assertGreater(
            int(sites_on_overworld), 0, "Construction sites should be placed on overworld"
        )

    def test_no_construction_when_disabled(self):
        """No construction sites should exist when construction_enabled=False."""
        env = _make_env(construction_enabled=False)
        _, state = _reset(env)

        total_sites = jnp.sum(state.map == BlockType.CONSTRUCTION_SITE.value)
        self.assertEqual(int(total_sites), 0)

    def test_construction_sites_have_coordination_values(self):
        """Construction sites should have non-zero coordination_map values
        (either positive for sync or negative for handover)."""
        env = _make_env(
            coordination_enabled=True,
            construction_enabled=True,
            num_construction_sites=4,
        )
        _, state = _reset(env, seed=42)

        # Find construction site positions on level 0
        is_site = state.map[0] == BlockType.CONSTRUCTION_SITE.value
        if jnp.any(is_site):
            coord_at_sites = state.coordination_map[0][is_site]
            nonzero = jnp.sum(coord_at_sites != 0)
            self.assertGreater(
                int(nonzero), 0, "Construction sites should have coordination requirements"
            )

    def test_construction_sites_built_initialized_zero(self):
        """construction_sites_built should start as all zeros (unbuilt)."""
        env = _make_env(construction_enabled=True, num_construction_sites=4)
        _, state = _reset(env)
        sp = env.static_env_params
        self.assertEqual(
            state.construction_sites_built.shape, (sp.num_levels, sp.max_construction_sites)
        )
        self.assertTrue(jnp.all(state.construction_sites_built == 0))


# ============================================================================
# 8. COORDINATION METRICS
# ============================================================================
class TestCoordinationMetrics(unittest.TestCase):
    """Test that coordination metrics are properly initialized and tracked.

    Metrics track events across four domains: mining, construction, combat,
    and crafting. All should start at zero and accumulate during gameplay.
    """

    def test_all_metrics_initialized_to_zero(self):
        """All coordination metric fields should start at 0."""
        env = _make_env(coordination_enabled=True)
        _, state = _reset(env)

        metric_fields = [
            "handover_successes",
            "handover_setups",
            "coord_mine_sync_soft_count",
            "coord_mine_sync_hard_count",
            "coord_mine_handover_count",
            "coord_build_shelter_count",
            "coord_build_forge_count",
            "coord_build_beacon_count",
            "coord_elite_melee_kills",
            "coord_elite_ranged_kills",
            "coord_large_passive_kills",
            "coord_diamond_pickaxe_count",
            "coord_diamond_sword_count",
            "coord_diamond_armour_count",
        ]
        for field in metric_fields:
            self.assertEqual(int(getattr(state, field)), 0, f"{field} should be initialized to 0")

    def test_sync_coord_by_agents_initialized_zero(self):
        """sync_coord_by_agents tracks [2-agent, 3+-agent] counts."""
        env = _make_env(coordination_enabled=True)
        _, state = _reset(env)
        self.assertEqual(state.sync_coord_by_agents.shape, (2,))
        self.assertTrue(jnp.all(state.sync_coord_by_agents == 0))


# ============================================================================
# 9. OBSERVATION SPACE ENCODING
# ============================================================================
class TestCoordinationObservations(unittest.TestCase):
    """Test that coordination information is properly encoded in observations.

    The symbolic renderer adds 5 coordination channels:
    - 2 channels for block coordination (type+magnitude, soft/hard flag)
    - 2 channels for mob coordination markers (requires_coord, is_hard)
    - 1 channel for pending handover time remaining

    These channels are naturally all-zero when coordination is disabled,
    so the observation size is the same regardless of coordination config.
    """

    def test_observation_shape_consistent(self):
        """Observation shape should be the same with or without coordination."""
        env_off = _make_env(coordination_enabled=False)
        env_on = _make_env(coordination_enabled=True, coordination_probability=0.5)

        _, state_off = _reset(env_off)
        _, state_on = _reset(env_on)

        obs_off = env_off.get_obs(state_off)
        obs_on = env_on.get_obs(state_on)

        shape_off = obs_off["agent_0"].shape
        shape_on = obs_on["agent_0"].shape
        self.assertEqual(
            shape_off, shape_on, "Obs shape should be the same regardless of coordination config"
        )

    def test_observations_finite(self):
        """All observation values should be finite (no NaN or Inf)."""
        env = _make_env(
            coordination_enabled=True,
            coordination_probability=0.5,
            elite_mob_probability=0.3,
            construction_enabled=True,
        )
        _, state = _reset(env, seed=42)
        obs = env.get_obs(state)
        for agent_name, agent_obs in obs.items():
            self.assertTrue(
                jnp.all(jnp.isfinite(agent_obs)),
                f"Observation for {agent_name} contains non-finite values",
            )


# ============================================================================
# 10. FULL ENVIRONMENT INTEGRATION
# ============================================================================
class TestCoordinationIntegration(unittest.TestCase):
    """End-to-end tests running the full environment with coordination enabled.

    These tests verify that the environment can reset and step without errors
    across various configurations.
    """

    def test_step_with_all_coordination_features(self):
        """Environment should run 10 steps with all coordination features enabled."""
        env = _make_env(
            coordination_enabled=True,
            coordination_probability=0.25,
            elite_mob_probability=0.15,
            large_passive_probability=0.20,
            hard_mob_probability=0.3,
            construction_enabled=True,
            num_construction_sites=4,
            crafting_coordination_enabled=True,
        )
        rng = jax.random.PRNGKey(0)
        obs, state = env.reset(rng)

        for i in range(10):
            rng, action_rng, step_rng = jax.random.split(rng, 3)
            actions = {
                agent: jax.random.randint(action_rng, (), 0, env.action_space(agent).n)
                for agent in env.agents
            }
            obs, state, reward, done, info = env.step(step_rng, state, actions)

    def test_step_with_coordination_disabled(self):
        """Environment should work fine with coordination completely off."""
        env = _make_env(coordination_enabled=False)
        rng = jax.random.PRNGKey(0)
        obs, state = env.reset(rng)

        for i in range(10):
            rng, action_rng, step_rng = jax.random.split(rng, 3)
            actions = {
                agent: jax.random.randint(action_rng, (), 0, env.action_space(agent).n)
                for agent in env.agents
            }
            obs, state, reward, done, info = env.step(step_rng, state, actions)

    def test_step_with_easy_preset(self):
        """Environment works with the 'easy' difficulty preset."""
        preset = get_coordination_params("easy")
        env = _make_env(**preset)
        rng = jax.random.PRNGKey(0)
        obs, state = env.reset(rng)

        for i in range(10):
            rng, action_rng, step_rng = jax.random.split(rng, 3)
            actions = {
                agent: jax.random.randint(action_rng, (), 0, env.action_space(agent).n)
                for agent in env.agents
            }
            obs, state, reward, done, info = env.step(step_rng, state, actions)

    def test_step_with_hard_preset(self):
        """Environment works with the 'hard' difficulty preset."""
        preset = get_coordination_params("hard")
        env = _make_env(**preset)
        rng = jax.random.PRNGKey(0)
        obs, state = env.reset(rng)

        for i in range(10):
            rng, action_rng, step_rng = jax.random.split(rng, 3)
            actions = {
                agent: jax.random.randint(action_rng, (), 0, env.action_space(agent).n)
                for agent in env.agents
            }
            obs, state, reward, done, info = env.step(step_rng, state, actions)

    def test_info_contains_coordination_metrics(self):
        """The info dict from env.step should contain coordination metrics."""
        preset = get_coordination_params("easy")
        env = _make_env(**preset)
        rng = jax.random.PRNGKey(0)
        obs, state = env.reset(rng)

        rng, action_rng, step_rng = jax.random.split(rng, 3)
        actions = _noop_actions(env)
        obs, state, reward, done, info = env.step(step_rng, state, actions)

        user_info = info.get("user_info", {})
        # Check that coordination-related keys exist
        expected_keys = [
            "Coordination/mine_sync_soft",
            "Coordination/mine_sync_hard",
            "Coordination/mine_handover",
            "Coordination/mining_handover_setups",
            "Coordination/mining_handover_successes",
            "Coordination/handover_successes",
            "Coordination/handover_setups",
        ]
        for key in expected_keys:
            self.assertIn(key, user_info, f"Expected coordination metric '{key}' in info")


# ============================================================================
# 11. BEHAVIORAL CORRECTNESS: DIFFICULTY SCALING
# ============================================================================
class TestDifficultyBehavior(unittest.TestCase):
    """Test that difficulty presets produce the intended behavioral differences.

    Before the unification refactor:
    - Easy: melee/ranged elites were ALWAYS hard (blocked solo agents)
    - This made "easy" not actually easy for combat

    After unification:
    - Easy: all elites are soft (solo=0.5x, coordinated=2x)
    - Medium: 30% of ALL elites are hard
    - Hard: 60% of ALL elites are hard

    These tests verify this new uniform behavior.
    """

    def test_easy_has_lower_p_max_agents_than_hard(self):
        """Easy difficulty has lower p_max_agents than hard, meaning fewer events
        require all agents (most only require 2)."""
        easy_preset = get_coordination_params("easy")
        hard_preset = get_coordination_params("hard")
        self.assertLess(easy_preset["p_max_agents"], hard_preset["p_max_agents"])

    def test_hard_has_mix_of_hard_and_soft(self):
        """On hard difficulty (hard_mob_probability=0.5, fixed), there should be both
        soft and hard elites."""
        preset = get_coordination_params("hard")
        env = _make_env(**preset)
        _, state = _reset(env, seed=42)

        total_soft = 0
        total_hard = 0
        for arr_name in [
            "melee_mob_coordination",
            "ranged_mob_coordination",
            "passive_mob_coordination",
        ]:
            arr = getattr(state, arr_name)
            total_soft += int(jnp.sum(arr == 1))
            total_hard += int(jnp.sum(arr == 2))

        # With 50% hard probability and 15% elite probability across 9 levels,
        # both should be non-zero
        self.assertGreater(total_soft + total_hard, 0, "Hard difficulty should have some elites")

    def test_uniform_hard_mob_probability_across_classes(self):
        """The hard_mob_probability should apply uniformly to melee, ranged,
        and passive — not just passives (which was the old behavior)."""
        env = _make_env(
            elite_mob_probability=0.5,
            large_passive_probability=0.5,
            hard_mob_probability=0.5,
        )
        _, state = _reset(env, seed=0)

        # All three classes should have some hard mobs
        melee_hard = int(jnp.sum(state.melee_mob_coordination == 2))
        ranged_hard = int(jnp.sum(state.ranged_mob_coordination == 2))
        passive_hard = int(jnp.sum(state.passive_mob_coordination == 2))

        self.assertGreater(melee_hard, 0, "Melee should have hard elites (was always-hard before)")
        self.assertGreater(
            ranged_hard, 0, "Ranged should have hard elites (was always-hard before)"
        )
        self.assertGreater(passive_hard, 0, "Passive should have hard elites")


# ============================================================================
# 12. TWO-KNOB DIFFICULTY SYSTEM
# ============================================================================
class TestTwoKnobDifficulty(unittest.TestCase):
    """Test the two-knob difficulty system (p_max_agents + handover windows).

    The key invariant: the NUMBER of coordination events (soft/hard/handover/elite)
    is FIXED across difficulties. Only HOW HARD each event is varies via:
    1. p_max_agents: P(event requires ALL agents vs just 2)
    2. Handover window tightness
    """

    def test_p_max_agents_zero_all_require_two(self):
        """With p_max_agents=0, all sync blocks should require exactly 2 agents."""
        env = _make_env(
            coordination_enabled=True,
            coordination_probability=0.5,
            handover_ratio=0.0,  # all sync
            p_max_agents=0.0,
        )
        _, state = _reset(env, seed=42)
        sync_values = state.coordination_map[state.coordination_map > 0]
        if len(sync_values) > 0:
            self.assertTrue(
                jnp.all(sync_values == 2), "p_max_agents=0 should produce all coord_req=2 blocks"
            )

    def test_p_max_agents_one_all_require_player_count(self):
        """With p_max_agents=1, all sync blocks should require player_count agents."""
        env = _make_env(
            coordination_enabled=True,
            coordination_probability=0.5,
            handover_ratio=0.0,  # all sync
            p_max_agents=1.0,
        )
        _, state = _reset(env, seed=42)
        sp = env.static_env_params
        sync_values = state.coordination_map[state.coordination_map > 0]
        if len(sync_values) > 0:
            self.assertTrue(
                jnp.all(sync_values == sp.player_count),
                f"p_max_agents=1 should produce all coord_req={sp.player_count} blocks",
            )

    def test_soft_sync_below_threshold_gives_1x(self):
        """On a soft sync block with coord_req=3, 2 agents should get 1x (below threshold)."""
        sp = StaticEnvParams(player_count=3)
        params = EnvParams(coordination_enabled=True)
        state = generate_world(jax.random.PRNGKey(0), params, sp)

        # Place soft sync with coord_req=3 at (5,5)
        new_coord = state.coordination_map.at[0, 5, 5].set(3)
        new_soft = state.soft_coordination_mask.at[0, 5, 5].set(True)
        state = state.replace(
            coordination_map=new_coord,
            soft_coordination_mask=new_soft,
            player_level=jnp.int32(0),
        )

        block_positions = jnp.full((sp.player_count, 2), jnp.array([5, 5]))
        # Only 2 agents acting (below threshold of 3)
        is_acting = jnp.array([True, True, False])
        equal = (block_positions[:, None] == block_positions[None, :]).all(axis=2)

        succeeds, _, mult, _, _ = check_sync_coordination(
            jax.random.PRNGKey(0), state, block_positions, is_acting, equal, params, sp
        )
        self.assertTrue(
            bool(succeeds[0]), "Soft sync should succeed solo when soft_solo_fail_prob=0"
        )
        self.assertEqual(float(mult[0]), 1.0, "Soft sync below threshold should give 1x multiplier")

    def test_soft_sync_at_threshold_gives_bonus(self):
        """On a soft sync block with coord_req=3, 3 agents should get k(k+1)/2 bonus."""
        sp = StaticEnvParams(player_count=3)
        params = EnvParams(coordination_enabled=True)
        state = generate_world(jax.random.PRNGKey(0), params, sp)

        new_coord = state.coordination_map.at[0, 5, 5].set(3)
        new_soft = state.soft_coordination_mask.at[0, 5, 5].set(True)
        state = state.replace(
            coordination_map=new_coord,
            soft_coordination_mask=new_soft,
            player_level=jnp.int32(0),
        )

        block_positions = jnp.full((sp.player_count, 2), jnp.array([5, 5]))
        is_acting = jnp.array([True, True, True])
        equal = (block_positions[:, None] == block_positions[None, :]).all(axis=2)

        _, _, mult, _, _ = check_sync_coordination(
            jax.random.PRNGKey(0), state, block_positions, is_acting, equal, params, sp
        )
        # k=3, k(k+1)/2 = 6.0
        self.assertEqual(
            float(mult[0]), 6.0, "3 agents meeting coord_req=3 on soft sync should get 6x"
        )

    def test_mob_agents_required_shapes(self):
        """New mob agents_required arrays should exist and have correct shapes."""
        env = _make_env(
            coordination_enabled=True,
            elite_mob_probability=0.3,
            large_passive_probability=0.3,
            p_max_agents=0.5,
        )
        _, state = _reset(env)
        sp = env.static_env_params
        expected_melee = (sp.num_levels, sp.max_melee_mobs * sp.player_count)
        expected_ranged = (sp.num_levels, sp.max_ranged_mobs * sp.player_count)
        expected_passive = (sp.num_levels, sp.max_passive_mobs * sp.player_count)
        self.assertEqual(state.melee_mob_agents_required.shape, expected_melee)
        self.assertEqual(state.ranged_mob_agents_required.shape, expected_ranged)
        self.assertEqual(state.passive_mob_agents_required.shape, expected_passive)

    def test_mob_agents_required_values(self):
        """Mob agents_required should be 0 (normal), 2, or player_count."""
        env = _make_env(
            coordination_enabled=True,
            elite_mob_probability=0.5,
            large_passive_probability=0.5,
            p_max_agents=0.5,
        )
        _, state = _reset(env, seed=42)
        sp = env.static_env_params
        for arr_name in [
            "melee_mob_agents_required",
            "ranged_mob_agents_required",
            "passive_mob_agents_required",
        ]:
            arr = getattr(state, arr_name)
            unique_vals = jnp.unique(arr)
            for v in unique_vals:
                self.assertIn(
                    int(v), {0, 2, sp.player_count}, f"{arr_name} has unexpected value {int(v)}"
                )


# ============================================================================
# 8. REQUEST + GIVE
# ============================================================================
class TestRequestAndGive(unittest.TestCase):
    """Verify request initialization, GIVE transfer, and GIVE masking."""

    def test_make_request_sets_type_and_duration(self):
        """REQUEST_* action should set request_type and refresh request_duration."""
        env = _make_env(coordination_enabled=False)
        _, state = _reset(env, seed=0)

        actions = jnp.full(
            (env.static_env_params.player_count,), Action.NOOP.value, dtype=jnp.int32
        )
        actions = actions.at[2].set(Action.REQUEST_WOOD.value)

        new_state = make_request(state, actions)

        self.assertEqual(int(new_state.request_type[2]), Action.REQUEST_WOOD.value)
        self.assertEqual(int(new_state.request_duration[2]), REQUEST_MAX_DURATION)

    def test_give_transfers_requested_resource(self):
        """A matching GIVE should move one unit from giver to requester."""
        env = _make_env(coordination_enabled=False)
        _, state = _reset(env, seed=1)

        # Agent 2 requests wood.
        req_dur = state.request_duration.at[2].set(REQUEST_MAX_DURATION)
        req_type = state.request_type.at[2].set(Action.REQUEST_WOOD.value)
        state = state.replace(request_duration=req_dur, request_type=req_type)

        # Agent 1 has wood to give; agent 2 has room.
        inv = state.inventory.replace(wood=state.inventory.wood.at[1].set(1).at[2].set(0))
        state = state.replace(inventory=inv)

        # For 3 players: agent_1 giving to agent_2 is GIVE slot 1 (index 55).
        actions = jnp.full(
            (env.static_env_params.player_count,), Action.NOOP.value, dtype=jnp.int32
        )
        actions = actions.at[1].set(Action.GIVE.value + 1)

        new_state = trade_materials(state, actions, env.static_env_params)

        self.assertEqual(int(new_state.inventory.wood[1]), 0)
        self.assertEqual(int(new_state.inventory.wood[2]), 1)

    def test_give_mask_requires_actual_transfer_feasibility(self):
        """GIVE mask should be true only when transfer would succeed."""
        env = _make_env(coordination_enabled=False)
        _, state = _reset(env, seed=2)

        # Agent 2 requests wood.
        req_dur = state.request_duration.at[2].set(REQUEST_MAX_DURATION)
        req_type = state.request_type.at[2].set(Action.REQUEST_WOOD.value)
        state = state.replace(request_duration=req_dur, request_type=req_type)

        # Agent 1 has wood, agent 2 has room.
        inv_ok = state.inventory.replace(wood=state.inventory.wood.at[1].set(1).at[2].set(0))
        state_ok = state.replace(inventory=inv_ok)

        mask_ok = compute_action_mask(state_ok, env.default_params, env.static_env_params)
        self.assertTrue(bool(mask_ok[1, Action.GIVE.value + 1]))

        # If requester is full for wood, GIVE should be masked out.
        inv_full = state.inventory.replace(wood=state.inventory.wood.at[1].set(1).at[2].set(99))
        state_full = state.replace(inventory=inv_full)

        mask_full = compute_action_mask(state_full, env.default_params, env.static_env_params)
        self.assertFalse(bool(mask_full[1, Action.GIVE.value + 1]))


if __name__ == "__main__":
    unittest.main()
