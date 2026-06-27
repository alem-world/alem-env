"""Tests for the construction system (do_construction).

The construction system allows agents to build structures at pre-placed
CONSTRUCTION_SITE blocks. Three structure types exist:

- **Epic Shelter** (BUILD_SHELTER): Costs 10 wood + 5 stone.
  Effect: faster rest energy regen.
- **Epic Forge** (BUILD_FORGE): Costs 10 stone + 3 iron + 2 coal.
  Effect: enables diamond crafting at this location.
- **Epic Beacon** (BUILD_BEACON): Costs 3 iron + 2 coal.
  Effect: expands light_map in a 9x9 radius.

Construction supports two coordination types:
- **Sync** (coord_value > 0): N agents must build simultaneously.
- **Handover** (coord_value < 0): Agent A initiates (creates IN_PROGRESS),
  Agent B completes within a time window.

Tests call `do_construction` directly with hand-crafted states to verify
map writes, inventory deduction, handover state management, metric tracking,
achievement grants, and edge cases.
"""

import unittest

import jax
import jax.numpy as jnp

from alem.alem_coop.action_masking import compute_action_mask
from alem.alem_coop.alem_state import EnvParams, StaticEnvParams
from alem.alem_coop.constants import (
    BEACON_COST_COAL,
    BEACON_COST_IRON,
    FORGE_COST_COAL,
    FORGE_COST_IRON,
    FORGE_COST_STONE,
    SHELTER_COST_STONE,
    SHELTER_COST_WOOD,
    Achievement,
    Action,
    BlockType,
)

# AlemCoopSymbolicEnv must be imported first: it appends alem/ to
# sys.path so that the internal ``from alem_coop.…`` imports resolve.
from alem.alem_coop.envs.alem_symbolic_env import AlemCoopSymbolicEnv  # noqa: F401
from alem.alem_coop.game_logic import (
    add_pending_handovers,
    clear_completed_handovers,
    do_construction,
    find_pending_matches,
    process_handover,
)
from alem.alem_coop.world_gen.world_gen import generate_world

# ---------------------------------------------------------------------------
# Test scaffold: position agents facing a construction site with materials
# ---------------------------------------------------------------------------

# DIRECTIONS index map:
#   0: NOOP [0,0], 1: LEFT [0,-1], 2: RIGHT [0,1], 3: UP [-1,0], 4: DOWN [1,0]
_SITE_POS = jnp.array([10, 10])  # all tests place the site here
_AGENT_POS = jnp.array([10, 9])  # one column left of site
_DIR_RIGHT = jnp.int32(2)  # DIRECTIONS[2] = [0, 1] → target = (10,10)


def _base_state(num_agents=3, seed=0):
    """Generate a world and return (state, params, static_params).

    Construction and coordination are enabled by default.
    """
    sp = StaticEnvParams(player_count=num_agents)
    params = EnvParams(
        coordination_enabled=True,
        construction_enabled=True,
        num_construction_sites=4,
    )
    state = generate_world(jax.random.PRNGKey(seed), params, sp)
    return state, params, sp


def _place_site(state, pos=_SITE_POS, coord_value=2, level=0):
    """Place a CONSTRUCTION_SITE at *pos* with given coordination value.

    Also registers the site in construction_site_positions (slot 0) so that
    site-tracking logic updates construction_sites_built.
    """
    new_map = state.map.at[level, pos[0], pos[1]].set(BlockType.CONSTRUCTION_SITE.value)
    new_coord = state.coordination_map.at[level, pos[0], pos[1]].set(coord_value)
    # Register site position in slot 0
    new_site_pos = state.construction_site_positions.at[level, 0].set(pos)
    return state.replace(
        map=new_map,
        coordination_map=new_coord,
        construction_site_positions=new_site_pos,
    )


def _position_agents(state, sp, *, agent_pos=_AGENT_POS, direction=_DIR_RIGHT):
    """Move all agents to *agent_pos* facing *direction*."""
    positions = jnp.tile(agent_pos, (sp.player_count, 1))
    directions = jnp.full((sp.player_count,), direction, dtype=jnp.int32)
    return state.replace(
        player_position=positions,
        player_direction=directions,
        player_level=jnp.int32(0),
    )


def _give_materials(state, sp, *, wood=50, stone=50, iron=20, coal=20):
    """Give all agents generous materials so builds aren't blocked by inventory."""
    inv = state.inventory
    inv = inv.replace(
        wood=jnp.full(sp.player_count, wood),
        stone=jnp.full(sp.player_count, stone),
        iron=jnp.full(sp.player_count, iron),
        coal=jnp.full(sp.player_count, coal),
    )
    return state.replace(inventory=inv)


def _setup(num_agents=3, coord_value=2, seed=0):
    """Full convenience setup: site at (10,10), agents one tile left, materials given."""
    state, params, sp = _base_state(num_agents=num_agents, seed=seed)
    state = _place_site(state, coord_value=coord_value)
    state = _position_agents(state, sp)
    state = _give_materials(state, sp)
    return state, params, sp


def _actions_array(sp, agent_actions):
    """Build an actions array. *agent_actions* is a dict {agent_idx: Action.X.value}.

    Agents not mentioned get NOOP.
    """
    arr = jnp.full(sp.player_count, Action.NOOP.value, dtype=jnp.int32)
    for idx, val in agent_actions.items():
        arr = arr.at[idx].set(val)
    return arr


def _rng():
    return jax.random.PRNGKey(42)


# ============================================================================
# 1. SYNC CONSTRUCTION — SHELTER
# ============================================================================
class TestSyncShelter(unittest.TestCase):
    """Verify sync (simultaneous) construction of an Epic Shelter."""

    def test_two_agents_build_shelter_sync(self):
        """Two agents at a sync-2 site → EPIC_SHELTER placed on the map."""
        state, params, sp = _setup(coord_value=2)
        actions = _actions_array(
            sp,
            {
                0: Action.BUILD_SHELTER.value,
                1: Action.BUILD_SHELTER.value,
            },
        )
        new_state = do_construction(_rng(), state, actions, params, sp)

        self.assertEqual(
            int(new_state.map[0, _SITE_POS[0], _SITE_POS[1]]),
            BlockType.EPIC_SHELTER.value,
            "Site should become EPIC_SHELTER after successful sync build",
        )

    def test_solo_agent_at_sync_site_fails(self):
        """One agent at sync-2 site → build fails, map unchanged."""
        state, params, sp = _setup(coord_value=2)
        actions = _actions_array(sp, {0: Action.BUILD_SHELTER.value})
        new_state = do_construction(_rng(), state, actions, params, sp)

        self.assertEqual(
            int(new_state.map[0, _SITE_POS[0], _SITE_POS[1]]),
            BlockType.CONSTRUCTION_SITE.value,
            "Solo agent at sync-2 site should NOT build",
        )

    def test_sync_shelter_deducts_materials_from_first_builder_only(self):
        """Only the lowest-indexed succeeding builder pays materials."""
        state, params, sp = _setup(coord_value=2)
        actions = _actions_array(
            sp,
            {
                0: Action.BUILD_SHELTER.value,
                1: Action.BUILD_SHELTER.value,
            },
        )
        new_state = do_construction(_rng(), state, actions, params, sp)

        # Agent 0 should have been charged
        self.assertEqual(
            int(new_state.inventory.wood[0]),
            50 - SHELTER_COST_WOOD,
        )
        self.assertEqual(
            int(new_state.inventory.stone[0]),
            50 - SHELTER_COST_STONE,
        )
        # Agent 1 should NOT be charged (same position, earlier builder exists)
        self.assertEqual(int(new_state.inventory.wood[1]), 50)
        self.assertEqual(int(new_state.inventory.stone[1]), 50)

    def test_sync_shelter_awards_achievement(self):
        """Both agents should receive COORD_BUILD_SHELTER achievement."""
        state, params, sp = _setup(coord_value=2)
        actions = _actions_array(
            sp,
            {
                0: Action.BUILD_SHELTER.value,
                1: Action.BUILD_SHELTER.value,
            },
        )
        new_state = do_construction(_rng(), state, actions, params, sp)

        ach_idx = Achievement.COORD_BUILD_SHELTER.value
        self.assertTrue(bool(new_state.achievements[0, ach_idx]))
        self.assertTrue(bool(new_state.achievements[1, ach_idx]))
        # Agent 2 did nothing
        self.assertFalse(bool(new_state.achievements[2, ach_idx]))

    def test_sync_shelter_updates_metric(self):
        """coord_build_shelter_count should increment by 1."""
        state, params, sp = _setup(coord_value=2)
        actions = _actions_array(
            sp,
            {
                0: Action.BUILD_SHELTER.value,
                1: Action.BUILD_SHELTER.value,
            },
        )
        new_state = do_construction(_rng(), state, actions, params, sp)
        self.assertEqual(int(new_state.coord_build_shelter_count), 1)

    def test_sync_shelter_only_one_builder_needs_materials(self):
        """Sync construction should succeed when only one participant can pay."""
        state, params, sp = _setup(coord_value=2)
        state = state.replace(
            inventory=state.inventory.replace(
                wood=state.inventory.wood.at[1].set(0),
                stone=state.inventory.stone.at[1].set(0),
            )
        )
        actions = _actions_array(
            sp,
            {
                0: Action.BUILD_SHELTER.value,
                1: Action.BUILD_SHELTER.value,
            },
        )
        new_state = do_construction(_rng(), state, actions, params, sp)

        self.assertEqual(
            int(new_state.map[0, _SITE_POS[0], _SITE_POS[1]]),
            BlockType.EPIC_SHELTER.value,
            "Only one sync builder should need to hold the site materials",
        )
        self.assertEqual(int(new_state.inventory.wood[0]), 50 - SHELTER_COST_WOOD)
        self.assertEqual(int(new_state.inventory.stone[0]), 50 - SHELTER_COST_STONE)
        self.assertEqual(int(new_state.inventory.wood[1]), 0)
        self.assertEqual(int(new_state.inventory.stone[1]), 0)

    def test_soft_sync_shelter_counts_as_success(self):
        """Soft construction that actually builds should increment success metrics."""
        state, params, sp = _setup(coord_value=2)
        state = state.replace(
            soft_coordination_mask=state.soft_coordination_mask.at[
                0, _SITE_POS[0], _SITE_POS[1]
            ].set(True)
        )
        actions = _actions_array(sp, {0: Action.BUILD_SHELTER.value})
        new_state = do_construction(_rng(), state, actions, params, sp)

        self.assertEqual(int(new_state.coord_construction_attempts), 1)
        self.assertEqual(int(new_state.coord_construction_successes), 1)
        self.assertEqual(int(new_state.coord_build_shelter_count), 1)

    def test_sync_clears_coordination_map(self):
        """After a successful sync build, coordination_map should be 0 at site."""
        state, params, sp = _setup(coord_value=2)
        actions = _actions_array(
            sp,
            {
                0: Action.BUILD_SHELTER.value,
                1: Action.BUILD_SHELTER.value,
            },
        )
        new_state = do_construction(_rng(), state, actions, params, sp)
        self.assertEqual(
            int(new_state.coordination_map[0, _SITE_POS[0], _SITE_POS[1]]),
            0,
        )

    def test_sync_updates_construction_sites_built(self):
        """construction_sites_built[0, 0] should be 1 (shelter) after sync build."""
        state, params, sp = _setup(coord_value=2)
        actions = _actions_array(
            sp,
            {
                0: Action.BUILD_SHELTER.value,
                1: Action.BUILD_SHELTER.value,
            },
        )
        new_state = do_construction(_rng(), state, actions, params, sp)
        self.assertEqual(int(new_state.construction_sites_built[0, 0]), 1)


# ============================================================================
# 2. SYNC CONSTRUCTION — FORGE
# ============================================================================
class TestSyncForge(unittest.TestCase):
    """Verify sync construction of an Epic Forge."""

    def test_sync_forge_placed(self):
        """Two agents BUILD_FORGE at sync-2 → EPIC_FORGE on map."""
        state, params, sp = _setup(coord_value=2)
        actions = _actions_array(
            sp,
            {
                0: Action.BUILD_FORGE.value,
                1: Action.BUILD_FORGE.value,
            },
        )
        new_state = do_construction(_rng(), state, actions, params, sp)
        self.assertEqual(
            int(new_state.map[0, _SITE_POS[0], _SITE_POS[1]]),
            BlockType.EPIC_FORGE.value,
        )

    def test_forge_deducts_correct_materials(self):
        """First builder pays 10 stone + 3 iron + 2 coal; second builder unpaid."""
        state, params, sp = _setup(coord_value=2)
        actions = _actions_array(
            sp,
            {
                0: Action.BUILD_FORGE.value,
                1: Action.BUILD_FORGE.value,
            },
        )
        new_state = do_construction(_rng(), state, actions, params, sp)
        self.assertEqual(int(new_state.inventory.stone[0]), 50 - FORGE_COST_STONE)
        self.assertEqual(int(new_state.inventory.iron[0]), 20 - FORGE_COST_IRON)
        self.assertEqual(int(new_state.inventory.coal[0]), 20 - FORGE_COST_COAL)
        # Wood should be untouched (forge doesn't use wood)
        self.assertEqual(int(new_state.inventory.wood[0]), 50)
        # Second builder not charged
        self.assertEqual(int(new_state.inventory.stone[1]), 50)

    def test_forge_achievement(self):
        state, params, sp = _setup(coord_value=2)
        actions = _actions_array(
            sp,
            {
                0: Action.BUILD_FORGE.value,
                1: Action.BUILD_FORGE.value,
            },
        )
        new_state = do_construction(_rng(), state, actions, params, sp)
        self.assertTrue(bool(new_state.achievements[0, Achievement.COORD_BUILD_FORGE.value]))

    def test_forge_metric(self):
        state, params, sp = _setup(coord_value=2)
        actions = _actions_array(
            sp,
            {
                0: Action.BUILD_FORGE.value,
                1: Action.BUILD_FORGE.value,
            },
        )
        new_state = do_construction(_rng(), state, actions, params, sp)
        self.assertEqual(int(new_state.coord_build_forge_count), 1)

    def test_forge_sites_built_tracking(self):
        state, params, sp = _setup(coord_value=2)
        actions = _actions_array(
            sp,
            {
                0: Action.BUILD_FORGE.value,
                1: Action.BUILD_FORGE.value,
            },
        )
        new_state = do_construction(_rng(), state, actions, params, sp)
        self.assertEqual(int(new_state.construction_sites_built[0, 0]), 2)


# ============================================================================
# 3. SYNC CONSTRUCTION — BEACON
# ============================================================================
class TestSyncBeacon(unittest.TestCase):
    """Verify sync construction of an Epic Beacon, including light map effects."""

    def test_sync_beacon_placed(self):
        state, params, sp = _setup(coord_value=2)
        actions = _actions_array(
            sp,
            {
                0: Action.BUILD_BEACON.value,
                1: Action.BUILD_BEACON.value,
            },
        )
        new_state = do_construction(_rng(), state, actions, params, sp)
        self.assertEqual(
            int(new_state.map[0, _SITE_POS[0], _SITE_POS[1]]),
            BlockType.EPIC_BEACON.value,
        )

    def test_beacon_deducts_iron_and_coal(self):
        state, params, sp = _setup(coord_value=2)
        actions = _actions_array(
            sp,
            {
                0: Action.BUILD_BEACON.value,
                1: Action.BUILD_BEACON.value,
            },
        )
        new_state = do_construction(_rng(), state, actions, params, sp)
        self.assertEqual(int(new_state.inventory.iron[0]), 20 - BEACON_COST_IRON)
        self.assertEqual(int(new_state.inventory.coal[0]), 20 - BEACON_COST_COAL)
        # Wood and stone untouched
        self.assertEqual(int(new_state.inventory.wood[0]), 50)
        self.assertEqual(int(new_state.inventory.stone[0]), 50)

    def test_beacon_expands_light_map(self):
        """Light map should increase around the beacon after construction."""
        state, params, sp = _setup(coord_value=2)
        # Zero out the light map so we can observe the beacon's effect
        state = state.replace(
            light_map=state.light_map.at[0].set(jnp.zeros_like(state.light_map[0]))
        )
        actions = _actions_array(
            sp,
            {
                0: Action.BUILD_BEACON.value,
                1: Action.BUILD_BEACON.value,
            },
        )
        new_state = do_construction(_rng(), state, actions, params, sp)

        # The center of the beacon light (at the site position) should be > 0
        center_light = float(new_state.light_map[0, _SITE_POS[0], _SITE_POS[1]])
        self.assertGreater(center_light, 0.0, "Beacon should add light at its position")

        # Some surrounding cells should also be lit
        nearby_light = float(new_state.light_map[0, _SITE_POS[0] + 1, _SITE_POS[1]])
        self.assertGreater(nearby_light, 0.0, "Beacon light should extend beyond center")

    def test_beacon_metric_and_achievement(self):
        state, params, sp = _setup(coord_value=2)
        actions = _actions_array(
            sp,
            {
                0: Action.BUILD_BEACON.value,
                1: Action.BUILD_BEACON.value,
            },
        )
        new_state = do_construction(_rng(), state, actions, params, sp)
        self.assertEqual(int(new_state.coord_build_beacon_count), 1)
        self.assertTrue(bool(new_state.achievements[0, Achievement.COORD_BUILD_BEACON.value]))


# ============================================================================
# 4. MATERIAL CHECKS
# ============================================================================
class TestMaterialRequirements(unittest.TestCase):
    """Verify that builds respect shared material requirements."""

    def test_shelter_succeeds_when_only_one_builder_has_wood(self):
        """Sync shelter should still succeed when one teammate can cover the cost."""
        state, params, sp = _setup(coord_value=2)
        # Remove wood from agent 0
        state = state.replace(
            inventory=state.inventory.replace(wood=state.inventory.wood.at[0].set(0))
        )
        actions = _actions_array(
            sp,
            {
                0: Action.BUILD_SHELTER.value,
                1: Action.BUILD_SHELTER.value,
            },
        )
        new_state = do_construction(_rng(), state, actions, params, sp)

        self.assertEqual(
            int(new_state.map[0, _SITE_POS[0], _SITE_POS[1]]),
            BlockType.EPIC_SHELTER.value,
            "Sync construction should succeed when agent 1 can pay the shelter cost",
        )
        self.assertEqual(int(new_state.inventory.wood[0]), 0)
        self.assertEqual(int(new_state.inventory.wood[1]), 50 - SHELTER_COST_WOOD)
        self.assertEqual(int(new_state.inventory.stone[1]), 50 - SHELTER_COST_STONE)

    def test_shelter_fails_without_stone(self):
        state, params, sp = _setup(coord_value=2)
        state = state.replace(
            inventory=state.inventory.replace(stone=state.inventory.stone.at[0].set(0).at[1].set(0))
        )
        actions = _actions_array(
            sp,
            {
                0: Action.BUILD_SHELTER.value,
                1: Action.BUILD_SHELTER.value,
            },
        )
        new_state = do_construction(_rng(), state, actions, params, sp)
        self.assertEqual(
            int(new_state.map[0, _SITE_POS[0], _SITE_POS[1]]),
            BlockType.CONSTRUCTION_SITE.value,
        )

    def test_forge_fails_without_iron(self):
        state, params, sp = _setup(coord_value=2)
        state = state.replace(
            inventory=state.inventory.replace(iron=jnp.zeros(sp.player_count, dtype=jnp.int32))
        )
        actions = _actions_array(
            sp,
            {
                0: Action.BUILD_FORGE.value,
                1: Action.BUILD_FORGE.value,
            },
        )
        new_state = do_construction(_rng(), state, actions, params, sp)
        self.assertEqual(
            int(new_state.map[0, _SITE_POS[0], _SITE_POS[1]]),
            BlockType.CONSTRUCTION_SITE.value,
        )

    def test_beacon_fails_without_coal(self):
        state, params, sp = _setup(coord_value=2)
        state = state.replace(
            inventory=state.inventory.replace(coal=jnp.zeros(sp.player_count, dtype=jnp.int32))
        )
        actions = _actions_array(
            sp,
            {
                0: Action.BUILD_BEACON.value,
                1: Action.BUILD_BEACON.value,
            },
        )
        new_state = do_construction(_rng(), state, actions, params, sp)
        self.assertEqual(
            int(new_state.map[0, _SITE_POS[0], _SITE_POS[1]]),
            BlockType.CONSTRUCTION_SITE.value,
        )


# ============================================================================
# 5. HANDOVER CONSTRUCTION
# ============================================================================
class TestHandoverConstruction(unittest.TestCase):
    """Verify two-phase handover construction.

    Phase 1 (setup): Agent A acts on a handover CONSTRUCTION_SITE. The block
    becomes CONSTRUCTION_IN_PROGRESS and a pending_handover entry is created.

    Phase 2 (completion): Agent B (different agent) acts on the IN_PROGRESS
    block within the deadline. The structure is built.
    """

    def test_handover_setup_creates_in_progress(self):
        """Single agent at handover site → CONSTRUCTION_IN_PROGRESS."""
        state, params, sp = _setup(coord_value=-15)  # negative = handover, window=15
        actions = _actions_array(sp, {0: Action.BUILD_SHELTER.value})
        new_state = do_construction(_rng(), state, actions, params, sp)

        self.assertEqual(
            int(new_state.map[0, _SITE_POS[0], _SITE_POS[1]]),
            BlockType.CONSTRUCTION_IN_PROGRESS.value,
            "Handover setup should convert CONSTRUCTION_SITE to IN_PROGRESS",
        )

    def test_handover_setup_creates_pending_entry(self):
        """Setup phase should add a pending_handovers entry."""
        state, params, sp = _setup(coord_value=-15)
        actions = _actions_array(sp, {0: Action.BUILD_SHELTER.value})
        new_state = do_construction(_rng(), state, actions, params, sp)

        active = jnp.sum(new_state.pending_handovers[:, 0] == 1)
        self.assertGreater(int(active), 0, "Should have at least one active pending handover")

        # The entry should reference our site position
        pending = new_state.pending_handovers
        active_mask = pending[:, 0] == 1
        # Find the active entry
        idx = jnp.argmax(active_mask)
        self.assertEqual(int(pending[idx, 1]), int(_SITE_POS[0]))
        self.assertEqual(int(pending[idx, 2]), int(_SITE_POS[1]))
        self.assertEqual(int(pending[idx, 4]), 0, "Initiator should be agent 0")

    def test_handover_setup_increments_setup_metric(self):
        state, params, sp = _setup(coord_value=-15)
        actions = _actions_array(sp, {0: Action.BUILD_SHELTER.value})
        new_state = do_construction(_rng(), state, actions, params, sp)
        self.assertGreater(int(new_state.handover_setups), int(state.handover_setups))

    def test_handover_completion_builds_structure(self):
        """Agent B completes a handover started by Agent A → EPIC_SHELTER."""
        state, params, sp = _setup(coord_value=-15)

        # Phase 1: Agent 0 sets up
        actions_setup = _actions_array(sp, {0: Action.BUILD_SHELTER.value})
        state = do_construction(_rng(), state, actions_setup, params, sp)
        self.assertEqual(
            int(state.map[0, _SITE_POS[0], _SITE_POS[1]]),
            BlockType.CONSTRUCTION_IN_PROGRESS.value,
        )

        # Phase 2: Agent 1 completes
        actions_complete = _actions_array(sp, {1: Action.BUILD_SHELTER.value})
        new_state = do_construction(_rng(), state, actions_complete, params, sp)

        self.assertEqual(
            int(new_state.map[0, _SITE_POS[0], _SITE_POS[1]]),
            BlockType.EPIC_SHELTER.value,
            "Handover completion should build EPIC_SHELTER",
        )

    def test_handover_completion_awards_achievement(self):
        """Both setup and completing agents get HANDOVER_COMPLETE at completion."""
        state, params, sp = _setup(coord_value=-15)

        # Phase 1: Agent 0 sets up — achievement not yet awarded
        actions_setup = _actions_array(sp, {0: Action.BUILD_SHELTER.value})
        state = do_construction(_rng(), state, actions_setup, params, sp)
        self.assertFalse(
            bool(state.achievements[0, Achievement.HANDOVER_COMPLETE.value]),
            "Setup agent should NOT get HANDOVER_COMPLETE until completion",
        )

        # Phase 2: Agent 1 completes — both agents get the achievement
        actions_complete = _actions_array(sp, {1: Action.BUILD_SHELTER.value})
        new_state = do_construction(_rng(), state, actions_complete, params, sp)
        self.assertTrue(
            bool(new_state.achievements[0, Achievement.HANDOVER_COMPLETE.value]),
            "Setup agent should get HANDOVER_COMPLETE after completion",
        )
        self.assertTrue(
            bool(new_state.achievements[1, Achievement.HANDOVER_COMPLETE.value]),
            "Completing agent should get HANDOVER_COMPLETE",
        )

    def test_handover_completion_increments_success_metric(self):
        state, params, sp = _setup(coord_value=-15)

        actions_setup = _actions_array(sp, {0: Action.BUILD_SHELTER.value})
        state = do_construction(_rng(), state, actions_setup, params, sp)
        pre_successes = int(state.handover_successes)

        actions_complete = _actions_array(sp, {1: Action.BUILD_SHELTER.value})
        new_state = do_construction(_rng(), state, actions_complete, params, sp)
        self.assertGreater(int(new_state.handover_successes), pre_successes)

    def test_same_agent_cannot_complete_own_handover(self):
        """Agent A cannot complete the handover they initiated."""
        state, params, sp = _setup(coord_value=-15)

        # Agent 0 sets up
        actions_setup = _actions_array(sp, {0: Action.BUILD_SHELTER.value})
        state = do_construction(_rng(), state, actions_setup, params, sp)

        # Agent 0 tries to complete → should NOT succeed (still IN_PROGRESS)
        actions_again = _actions_array(sp, {0: Action.BUILD_SHELTER.value})
        new_state = do_construction(_rng(), state, actions_again, params, sp)

        self.assertEqual(
            int(new_state.map[0, _SITE_POS[0], _SITE_POS[1]]),
            BlockType.CONSTRUCTION_IN_PROGRESS.value,
            "Same agent cannot complete their own handover",
        )

    def test_wrong_build_type_cannot_complete_handover(self):
        """An in-progress shelter handover must be completed as a shelter."""
        state, params, sp = _setup(coord_value=-15)

        actions_setup = _actions_array(sp, {0: Action.BUILD_SHELTER.value})
        state = do_construction(_rng(), state, actions_setup, params, sp)

        actions_wrong = _actions_array(sp, {1: Action.BUILD_FORGE.value})
        new_state = do_construction(_rng(), state, actions_wrong, params, sp)

        self.assertEqual(
            int(new_state.map[0, _SITE_POS[0], _SITE_POS[1]]),
            BlockType.CONSTRUCTION_IN_PROGRESS.value,
            "Wrong BUILD_* type should not complete an in-progress handover",
        )
        self.assertEqual(
            int(new_state.handover_successes),
            int(state.handover_successes),
            "Wrong BUILD_* type should not increment handover successes",
        )

    def test_action_mask_limits_in_progress_to_pending_type(self):
        """Only the pending build type should be offered at IN_PROGRESS sites."""
        state, params, sp = _setup(coord_value=-15)

        actions_setup = _actions_array(sp, {0: Action.BUILD_SHELTER.value})
        state = do_construction(_rng(), state, actions_setup, params, sp)
        mask = compute_action_mask(state, params, sp)

        self.assertFalse(bool(mask[0, Action.BUILD_SHELTER.value]))
        self.assertTrue(bool(mask[1, Action.BUILD_SHELTER.value]))
        self.assertFalse(bool(mask[1, Action.BUILD_FORGE.value]))
        self.assertFalse(bool(mask[1, Action.BUILD_BEACON.value]))

    def test_handover_forge_completion(self):
        """Agent A sets up forge, Agent B completes → EPIC_FORGE."""
        state, params, sp = _setup(coord_value=-20)
        actions_setup = _actions_array(sp, {0: Action.BUILD_FORGE.value})
        state = do_construction(_rng(), state, actions_setup, params, sp)

        actions_complete = _actions_array(sp, {1: Action.BUILD_FORGE.value})
        new_state = do_construction(_rng(), state, actions_complete, params, sp)

        self.assertEqual(
            int(new_state.map[0, _SITE_POS[0], _SITE_POS[1]]),
            BlockType.EPIC_FORGE.value,
        )

    def test_handover_beacon_completion_with_light(self):
        """Agent A sets up beacon, Agent B completes → light map expands."""
        state, params, sp = _setup(coord_value=-20)
        state = state.replace(
            light_map=state.light_map.at[0].set(jnp.zeros_like(state.light_map[0]))
        )

        actions_setup = _actions_array(sp, {0: Action.BUILD_BEACON.value})
        state = do_construction(_rng(), state, actions_setup, params, sp)

        actions_complete = _actions_array(sp, {1: Action.BUILD_BEACON.value})
        new_state = do_construction(_rng(), state, actions_complete, params, sp)

        self.assertEqual(
            int(new_state.map[0, _SITE_POS[0], _SITE_POS[1]]),
            BlockType.EPIC_BEACON.value,
        )
        self.assertGreater(
            float(new_state.light_map[0, _SITE_POS[0], _SITE_POS[1]]),
            0.0,
            "Beacon should illuminate after handover completion",
        )


# ============================================================================
# 6. HANDOVER EXPIRY
# ============================================================================
class TestHandoverExpiry(unittest.TestCase):
    """Test that expired handovers revert CONSTRUCTION_IN_PROGRESS blocks.

    Expiry is handled by process_handover (runs before do_construction in
    alem_step), so we test it indirectly by setting up an expired pending
    entry with an IN_PROGRESS block and calling process_handover.
    """

    def test_expired_handover_does_not_complete(self):
        """Agent B acting after deadline → block stays IN_PROGRESS (no build).

        We simulate expiry by setting a very short window (1) and advancing
        the timestep past it before Agent B acts.
        """
        state, params, sp = _setup(coord_value=-1)  # window = 1 timestep

        # Phase 1: Agent 0 sets up at timestep 0
        actions_setup = _actions_array(sp, {0: Action.BUILD_SHELTER.value})
        state = do_construction(_rng(), state, actions_setup, params, sp)

        # Advance timestep well past the deadline
        state = state.replace(timestep=100)

        # The pending entry is now expired. If process_handover ran, it would
        # revert the block. But since we're testing do_construction directly,
        # Agent 1 should NOT find a valid match (deadline exceeded).
        actions_complete = _actions_array(sp, {1: Action.BUILD_SHELTER.value})
        new_state = do_construction(_rng(), state, actions_complete, params, sp)

        # find_pending_matches checks pending[:, 3] > timestep; since we're
        # past the deadline the entry won't match. The IN_PROGRESS block stays
        # (process_handover would revert it, but we didn't call it here).
        self.assertEqual(
            int(new_state.map[0, _SITE_POS[0], _SITE_POS[1]]),
            BlockType.CONSTRUCTION_IN_PROGRESS.value,
            "Expired handover should not be completable",
        )


# ============================================================================
# 7. MISMATCHED BUILD TYPES
# ============================================================================
class TestBuildTypeMismatch(unittest.TestCase):
    """Verify that agents building different structure types at the same site
    don't count towards each other's sync requirements."""

    def test_different_build_types_dont_count(self):
        """Agent 0 builds shelter, Agent 1 builds forge → neither meets sync-2."""
        state, params, sp = _setup(coord_value=2)
        actions = _actions_array(
            sp,
            {
                0: Action.BUILD_SHELTER.value,
                1: Action.BUILD_FORGE.value,
            },
        )
        new_state = do_construction(_rng(), state, actions, params, sp)
        self.assertEqual(
            int(new_state.map[0, _SITE_POS[0], _SITE_POS[1]]),
            BlockType.CONSTRUCTION_SITE.value,
            "Mismatched build types should not satisfy sync requirement",
        )

    def test_three_agents_two_same_type_at_sync_2(self):
        """Two agents building shelter + one building forge at sync-2 site.
        Only shelter should succeed (2 agents match); forge fails (only 1)."""
        state, params, sp = _setup(coord_value=2)
        actions = _actions_array(
            sp,
            {
                0: Action.BUILD_SHELTER.value,
                1: Action.BUILD_SHELTER.value,
                2: Action.BUILD_FORGE.value,
            },
        )
        new_state = do_construction(_rng(), state, actions, params, sp)
        self.assertEqual(
            int(new_state.map[0, _SITE_POS[0], _SITE_POS[1]]),
            BlockType.EPIC_SHELTER.value,
            "Two shelter builders should succeed at sync-2 even if third builds forge",
        )


# ============================================================================
# 8. CONSTRUCTION DISABLED
# ============================================================================
class TestConstructionDisabled(unittest.TestCase):
    """Verify early return when construction_enabled=False."""

    def test_disabled_returns_state_unchanged(self):
        """With construction_enabled=False, state should be unchanged."""
        state, _, sp = _base_state()
        params = EnvParams(construction_enabled=False, coordination_enabled=True)
        state = _place_site(state, coord_value=2)
        state = _position_agents(state, sp)
        state = _give_materials(state, sp)

        actions = _actions_array(
            sp,
            {
                0: Action.BUILD_SHELTER.value,
                1: Action.BUILD_SHELTER.value,
            },
        )
        new_state = do_construction(_rng(), state, actions, params, sp)

        # Map should be unchanged
        self.assertEqual(
            int(new_state.map[0, _SITE_POS[0], _SITE_POS[1]]),
            BlockType.CONSTRUCTION_SITE.value,
        )
        # Inventory unchanged
        self.assertEqual(int(new_state.inventory.wood[0]), 50)


# ============================================================================
# 9. NOT AT CONSTRUCTION SITE
# ============================================================================
class TestNotAtSite(unittest.TestCase):
    """Build actions on non-construction blocks should do nothing."""

    def test_build_on_grass_does_nothing(self):
        """BUILD_SHELTER on a grass block shouldn't build anything."""
        state, params, sp = _setup(coord_value=2)
        # Overwrite the site back to grass
        state = state.replace(
            map=state.map.at[0, _SITE_POS[0], _SITE_POS[1]].set(BlockType.GRASS.value)
        )
        actions = _actions_array(
            sp,
            {
                0: Action.BUILD_SHELTER.value,
                1: Action.BUILD_SHELTER.value,
            },
        )
        new_state = do_construction(_rng(), state, actions, params, sp)
        self.assertEqual(
            int(new_state.map[0, _SITE_POS[0], _SITE_POS[1]]),
            BlockType.GRASS.value,
        )
        # No materials deducted
        self.assertEqual(int(new_state.inventory.wood[0]), 50)


# ============================================================================
# 10. MULTIPLE SITES
# ============================================================================
class TestMultipleSites(unittest.TestCase):
    """Verify that two pairs of agents can build at two different sites."""

    def test_two_sites_built_independently(self):
        """Agents 0,1 build shelter at site A; agent 2 faces elsewhere (no build).

        Tests that building at one site doesn't interfere with map elsewhere.
        """
        state, params, sp = _setup(num_agents=4, coord_value=2)
        state, params, sp = _base_state(num_agents=4, seed=0)
        state = _place_site(state, coord_value=2)
        state = _position_agents(state, sp)
        state = _give_materials(state, sp)

        # Place a second site at (15, 15) for agents 2 & 3
        site2 = jnp.array([15, 15])
        state = state.replace(
            map=state.map.at[0, site2[0], site2[1]].set(BlockType.CONSTRUCTION_SITE.value),
            coordination_map=state.coordination_map.at[0, site2[0], site2[1]].set(2),
            construction_site_positions=state.construction_site_positions.at[0, 1].set(site2),
        )

        # Move agents 2 & 3 to face site2: position (15, 14) facing right
        positions = state.player_position.at[2].set(jnp.array([15, 14]))
        positions = positions.at[3].set(jnp.array([15, 14]))
        state = state.replace(player_position=positions)

        actions = _actions_array(
            sp,
            {
                0: Action.BUILD_SHELTER.value,
                1: Action.BUILD_SHELTER.value,
                2: Action.BUILD_FORGE.value,
                3: Action.BUILD_FORGE.value,
            },
        )
        new_state = do_construction(_rng(), state, actions, params, sp)

        # Site 1 should be shelter
        self.assertEqual(
            int(new_state.map[0, _SITE_POS[0], _SITE_POS[1]]),
            BlockType.EPIC_SHELTER.value,
        )
        # Site 2 should be forge
        self.assertEqual(
            int(new_state.map[0, site2[0], site2[1]]),
            BlockType.EPIC_FORGE.value,
        )


# ============================================================================
# 11. SHARED HELPERS UNIT TESTS
# ============================================================================
class TestSharedHelpers(unittest.TestCase):
    """Test the shared helper functions used by both process_handover and
    do_construction."""

    def test_find_pending_matches_no_entries(self):
        """No active entries → all has_match=False."""
        sp = StaticEnvParams(player_count=2)
        pending = jnp.zeros((sp.max_pending_handovers, 6), dtype=jnp.int32)
        block_positions = jnp.array([[5, 5], [6, 6]])
        has_match, match_idx = find_pending_matches(pending, block_positions, 0, sp)
        self.assertFalse(bool(has_match[0]))
        self.assertFalse(bool(has_match[1]))

    def test_find_pending_matches_exact(self):
        """Active entry at (5,5) by agent 0 → agent 1 at (5,5) matches."""
        sp = StaticEnvParams(player_count=2)
        pending = jnp.zeros((sp.max_pending_handovers, 6), dtype=jnp.int32)
        # Entry: [active=1, x=5, y=5, deadline=100, initiator=0]
        pending = pending.at[0].set(jnp.array([1, 5, 5, 100, 0, 0]))
        block_positions = jnp.array([[5, 5], [5, 5]])

        has_match, match_idx = find_pending_matches(pending, block_positions, 10, sp)
        # Agent 0 is the initiator → should NOT match their own entry
        self.assertFalse(bool(has_match[0]))
        # Agent 1 should match
        self.assertTrue(bool(has_match[1]))
        self.assertEqual(int(match_idx[1]), 0)

    def test_find_pending_matches_expired_ignored(self):
        """An expired entry (deadline <= timestep) should not match."""
        sp = StaticEnvParams(player_count=2)
        pending = jnp.zeros((sp.max_pending_handovers, 6), dtype=jnp.int32)
        pending = pending.at[0].set(jnp.array([1, 5, 5, 10, 0, 0]))  # deadline=10
        block_positions = jnp.array([[5, 5], [5, 5]])

        has_match, _ = find_pending_matches(pending, block_positions, 10, sp)
        self.assertFalse(
            bool(has_match[1]), "Entry with deadline=10 at timestep=10 should be expired"
        )

    def test_clear_completed_handovers(self):
        """Clearing agent 1's matched slot should zero it out."""
        sp = StaticEnvParams(player_count=2)
        pending = jnp.zeros((sp.max_pending_handovers, 6), dtype=jnp.int32)
        pending = pending.at[0].set(jnp.array([1, 5, 5, 100, 0, 0]))

        is_completing = jnp.array([False, True])
        match_idx = jnp.array([0, 0])  # both "point at" slot 0
        cleared = clear_completed_handovers(pending, is_completing, match_idx, sp)
        self.assertEqual(int(cleared[0, 0]), 0, "Slot 0 should be cleared")

    def test_add_pending_handovers_creates_entry(self):
        """Adding a setup for agent 0 should create a new pending entry."""
        sp = StaticEnvParams(player_count=2)
        pending = jnp.zeros((sp.max_pending_handovers, 6), dtype=jnp.int32)
        block_positions = jnp.array([[5, 5], [6, 6]])
        is_setting_up = jnp.array([True, False])
        window = jnp.array([15, 0])

        new_pending, count = add_pending_handovers(
            pending, block_positions, is_setting_up, window, timestep=10, static_params=sp
        )
        self.assertEqual(int(count), 1)
        self.assertEqual(int(new_pending[0, 0]), 1, "Should be active")
        self.assertEqual(int(new_pending[0, 1]), 5)  # pos_x
        self.assertEqual(int(new_pending[0, 2]), 5)  # pos_y
        self.assertEqual(int(new_pending[0, 3]), 26)  # deadline = 10 + 15 + 1
        self.assertEqual(int(new_pending[0, 4]), 0)  # initiator = agent 0


# ============================================================================
# 12. EDGE CASES
# ============================================================================
class TestConstructionEdgeCases(unittest.TestCase):
    """Various edge cases for construction."""

    def test_noop_actions_do_nothing(self):
        """All agents NOOP → no state change in construction."""
        state, params, sp = _setup(coord_value=2)
        actions = _actions_array(sp, {})  # all NOOP
        new_state = do_construction(_rng(), state, actions, params, sp)
        self.assertEqual(
            int(new_state.map[0, _SITE_POS[0], _SITE_POS[1]]),
            BlockType.CONSTRUCTION_SITE.value,
        )
        # Metrics unchanged
        self.assertEqual(int(new_state.coord_build_shelter_count), 0)
        self.assertEqual(int(new_state.coord_build_forge_count), 0)
        self.assertEqual(int(new_state.coord_build_beacon_count), 0)

    def test_sync_3_needs_three_agents(self):
        """Sync-3 coordination: 2 agents should fail, 3 should succeed."""
        state, params, sp = _setup(coord_value=3)  # require 3 agents
        actions_two = _actions_array(
            sp,
            {
                0: Action.BUILD_SHELTER.value,
                1: Action.BUILD_SHELTER.value,
            },
        )
        state_two = do_construction(_rng(), state, actions_two, params, sp)
        self.assertEqual(
            int(state_two.map[0, _SITE_POS[0], _SITE_POS[1]]),
            BlockType.CONSTRUCTION_SITE.value,
            "2 agents should fail at sync-3",
        )

        actions_three = _actions_array(
            sp,
            {
                0: Action.BUILD_SHELTER.value,
                1: Action.BUILD_SHELTER.value,
                2: Action.BUILD_SHELTER.value,
            },
        )
        state_three = do_construction(_rng(), state, actions_three, params, sp)
        self.assertEqual(
            int(state_three.map[0, _SITE_POS[0], _SITE_POS[1]]),
            BlockType.EPIC_SHELTER.value,
            "3 agents should succeed at sync-3",
        )

    def test_action_mask_allows_helper_when_teammate_can_pay(self):
        """Helpers should see BUILD_* when a teammate can pay the shared cost."""
        state, params, sp = _setup(coord_value=2)
        state = state.replace(
            inventory=state.inventory.replace(
                wood=state.inventory.wood.at[1].set(0),
                stone=state.inventory.stone.at[1].set(0),
            )
        )
        mask = compute_action_mask(state, params, sp)

        self.assertTrue(bool(mask[0, Action.BUILD_SHELTER.value]))
        self.assertTrue(
            bool(mask[1, Action.BUILD_SHELTER.value]),
            "Helper should still be allowed to choose BUILD when a teammate can pay",
        )

    def test_build_at_completed_site_does_nothing(self):
        """Once a site is built (e.g. EPIC_SHELTER), further builds should fail."""
        state, params, sp = _setup(coord_value=2)
        # Pre-set the site to EPIC_SHELTER (already built)
        state = state.replace(
            map=state.map.at[0, _SITE_POS[0], _SITE_POS[1]].set(BlockType.EPIC_SHELTER.value),
            coordination_map=state.coordination_map.at[0, _SITE_POS[0], _SITE_POS[1]].set(0),
        )
        actions = _actions_array(
            sp,
            {
                0: Action.BUILD_FORGE.value,
                1: Action.BUILD_FORGE.value,
            },
        )
        new_state = do_construction(_rng(), state, actions, params, sp)
        # Should still be shelter, not forge
        self.assertEqual(
            int(new_state.map[0, _SITE_POS[0], _SITE_POS[1]]),
            BlockType.EPIC_SHELTER.value,
        )
        # No materials deducted
        self.assertEqual(int(new_state.inventory.stone[0]), 50)

    def test_handover_setup_deducts_materials(self):
        """Setup phase deducts materials from initiator (paid upfront, refunded on expiry)."""
        state, params, sp = _setup(coord_value=-15)
        actions = _actions_array(sp, {0: Action.BUILD_SHELTER.value})
        new_state = do_construction(_rng(), state, actions, params, sp)
        # Initiator pays materials upfront
        self.assertEqual(int(new_state.inventory.wood[0]), 50 - SHELTER_COST_WOOD)
        self.assertEqual(int(new_state.inventory.stone[0]), 50 - SHELTER_COST_STONE)

    def test_handover_completion_does_not_deduct_materials(self):
        """Completer pays nothing — initiator already paid upfront (trust game)."""
        state, params, sp = _setup(coord_value=-15)

        # Setup by agent 0 (pays materials upfront)
        actions_setup = _actions_array(sp, {0: Action.BUILD_SHELTER.value})
        state = do_construction(_rng(), state, actions_setup, params, sp)

        # Complete by agent 1
        actions_complete = _actions_array(sp, {1: Action.BUILD_SHELTER.value})
        new_state = do_construction(_rng(), state, actions_complete, params, sp)

        # Agent 1 (completer) should NOT be charged
        self.assertEqual(int(new_state.inventory.wood[1]), 50)
        self.assertEqual(int(new_state.inventory.stone[1]), 50)
        # Agent 0 (initiator) was already charged at setup
        self.assertEqual(int(new_state.inventory.wood[0]), 50 - SHELTER_COST_WOOD)


# ============================================================================
# 13. INTEGRATION: FULL ENV STEP WITH CONSTRUCTION
# ============================================================================
class TestConstructionIntegration(unittest.TestCase):
    """Run the full environment with construction to verify no crashes."""

    def test_random_actions_with_construction_enabled(self):
        """10 random steps with construction enabled should not crash."""
        from alem.alem_coop.alem_state import get_coordination_params

        preset = get_coordination_params("easy")
        env = AlemCoopSymbolicEnv(num_agents=3, env_params=EnvParams(**preset))
        rng = jax.random.PRNGKey(0)
        obs, state = env.reset(rng)

        for _ in range(10):
            rng, action_rng, step_rng = jax.random.split(rng, 3)
            actions = {
                agent: jax.random.randint(action_rng, (), 0, env.action_space(agent).n)
                for agent in env.agents
            }
            obs, state, reward, done, info = env.step(step_rng, state, actions)

    def test_construction_metrics_in_info(self):
        """Info dict should contain construction coordination metrics."""
        from alem.alem_coop.alem_state import get_coordination_params

        preset = get_coordination_params("easy")
        env = AlemCoopSymbolicEnv(num_agents=3, env_params=EnvParams(**preset))
        rng = jax.random.PRNGKey(0)
        obs, state = env.reset(rng)

        rng, step_rng = jax.random.split(rng)
        noop = {agent: jnp.int32(Action.NOOP.value) for agent in env.agents}
        obs, state, reward, done, info = env.step(step_rng, state, noop)

        user_info = info.get("user_info", {})
        for key in [
            "Coordination/construction_attempts",
            "Coordination/construction_sync_attempts",
            "Coordination/construction_handover_attempts",
            "Coordination/construction_total_attempts",
            "Coordination/construction_success_rate",
            "Coordination/construction_sync_success_rate",
            "Coordination/construction_total_success_rate",
            "Coordination/build_shelter",
            "Coordination/build_forge",
            "Coordination/build_beacon",
        ]:
            self.assertIn(key, user_info, f"Expected construction metric '{key}' in info")


# ============================================================================
# 14. HANDOVER MATERIAL REFUND ON EXPIRY
# ============================================================================
class TestHandoverMaterialRefund(unittest.TestCase):
    """Verify that materials are refunded to the initiator when a construction
    handover expires without being completed.

    This tests the pure-coordination model: failed coordination costs only time,
    not resources.
    """

    def _setup_in_progress(self, build_action, coord_value=-15):
        """Helper: return state with one IN_PROGRESS handover set up by agent 0."""
        state, params, sp = _setup(coord_value=coord_value)
        actions_setup = _actions_array(sp, {0: build_action})
        state = do_construction(_rng(), state, actions_setup, params, sp)
        self.assertEqual(
            int(state.map[0, _SITE_POS[0], _SITE_POS[1]]),
            BlockType.CONSTRUCTION_IN_PROGRESS.value,
        )
        return state, params, sp

    def test_shelter_materials_refunded_on_expiry(self):
        """Shelter handover expiry → wood and stone returned to initiator."""
        state, params, sp = self._setup_in_progress(Action.BUILD_SHELTER.value)

        wood_after_setup = int(state.inventory.wood[0])
        stone_after_setup = int(state.inventory.stone[0])

        # Advance past the deadline by setting timestep beyond it
        pending = state.pending_handovers
        deadline = int(pending[jnp.argmax(pending[:, 0] == 1), 3])
        state = state.replace(timestep=jnp.int32(deadline))  # deadline <= timestep → expired

        # process_handover runs expiry logic
        is_acting = jnp.zeros(sp.player_count, dtype=jnp.bool_)
        _, state = process_handover(
            state, jnp.tile(_SITE_POS, (sp.player_count, 1)), is_acting, params, sp
        )

        self.assertEqual(
            int(state.inventory.wood[0]),
            wood_after_setup + SHELTER_COST_WOOD,
            "Wood should be refunded to initiator on shelter handover expiry",
        )
        self.assertEqual(
            int(state.inventory.stone[0]),
            stone_after_setup + SHELTER_COST_STONE,
            "Stone should be refunded to initiator on shelter handover expiry",
        )

    def test_forge_materials_refunded_on_expiry(self):
        """Forge handover expiry → stone, iron, coal returned to initiator."""
        state, params, sp = self._setup_in_progress(Action.BUILD_FORGE.value)

        stone_after = int(state.inventory.stone[0])
        iron_after = int(state.inventory.iron[0])
        coal_after = int(state.inventory.coal[0])

        pending = state.pending_handovers
        deadline = int(pending[jnp.argmax(pending[:, 0] == 1), 3])
        state = state.replace(timestep=jnp.int32(deadline))

        is_acting = jnp.zeros(sp.player_count, dtype=jnp.bool_)
        _, state = process_handover(
            state, jnp.tile(_SITE_POS, (sp.player_count, 1)), is_acting, params, sp
        )

        self.assertEqual(int(state.inventory.stone[0]), stone_after + FORGE_COST_STONE)
        self.assertEqual(int(state.inventory.iron[0]), iron_after + FORGE_COST_IRON)
        self.assertEqual(int(state.inventory.coal[0]), coal_after + FORGE_COST_COAL)

    def test_beacon_materials_refunded_on_expiry(self):
        """Beacon handover expiry → iron and coal returned to initiator."""
        state, params, sp = self._setup_in_progress(Action.BUILD_BEACON.value)

        iron_after = int(state.inventory.iron[0])
        coal_after = int(state.inventory.coal[0])

        pending = state.pending_handovers
        deadline = int(pending[jnp.argmax(pending[:, 0] == 1), 3])
        state = state.replace(timestep=jnp.int32(deadline))

        is_acting = jnp.zeros(sp.player_count, dtype=jnp.bool_)
        _, state = process_handover(
            state, jnp.tile(_SITE_POS, (sp.player_count, 1)), is_acting, params, sp
        )

        self.assertEqual(int(state.inventory.iron[0]), iron_after + BEACON_COST_IRON)
        self.assertEqual(int(state.inventory.coal[0]), coal_after + BEACON_COST_COAL)

    def test_site_reverts_on_expiry(self):
        """Expired handover reverts CONSTRUCTION_IN_PROGRESS back to CONSTRUCTION_SITE."""
        state, params, sp = self._setup_in_progress(Action.BUILD_SHELTER.value)

        pending = state.pending_handovers
        deadline = int(pending[jnp.argmax(pending[:, 0] == 1), 3])
        state = state.replace(timestep=jnp.int32(deadline))

        is_acting = jnp.zeros(sp.player_count, dtype=jnp.bool_)
        _, state = process_handover(
            state, jnp.tile(_SITE_POS, (sp.player_count, 1)), is_acting, params, sp
        )

        self.assertEqual(
            int(state.map[0, _SITE_POS[0], _SITE_POS[1]]),
            BlockType.CONSTRUCTION_SITE.value,
            "Expired handover should revert IN_PROGRESS back to CONSTRUCTION_SITE",
        )

    def test_non_initiator_not_refunded(self):
        """Only the initiator receives the refund — other agents' inventories unchanged."""
        state, params, sp = self._setup_in_progress(Action.BUILD_SHELTER.value)

        # Record agent 1's inventory before expiry
        wood_agent1_before = int(state.inventory.wood[1])

        pending = state.pending_handovers
        deadline = int(pending[jnp.argmax(pending[:, 0] == 1), 3])
        state = state.replace(timestep=jnp.int32(deadline))

        is_acting = jnp.zeros(sp.player_count, dtype=jnp.bool_)
        _, state = process_handover(
            state, jnp.tile(_SITE_POS, (sp.player_count, 1)), is_acting, params, sp
        )

        self.assertEqual(
            int(state.inventory.wood[1]),
            wood_agent1_before,
            "Non-initiator agent should not receive any refund",
        )

    def test_no_refund_on_successful_completion(self):
        """Completing a handover successfully charges materials and keeps them charged."""
        state, params, sp = _setup(coord_value=-15)

        # Agent 0 sets up (pays)
        actions_setup = _actions_array(sp, {0: Action.BUILD_SHELTER.value})
        state = do_construction(_rng(), state, actions_setup, params, sp)
        wood_after_setup = int(state.inventory.wood[0])

        # Agent 1 completes (no refund should occur — handover succeeded)
        actions_complete = _actions_array(sp, {1: Action.BUILD_SHELTER.value})
        new_state = do_construction(_rng(), state, actions_complete, params, sp)

        self.assertEqual(
            int(new_state.inventory.wood[0]),
            wood_after_setup,
            "No refund should occur on successful completion",
        )
        self.assertEqual(
            int(new_state.map[0, _SITE_POS[0], _SITE_POS[1]]),
            BlockType.EPIC_SHELTER.value,
        )

    def test_building_type_stored_in_pending(self):
        """Pending entry should store correct building type in column 5."""
        state, params, sp = _setup(coord_value=-15)

        actions = _actions_array(sp, {0: Action.BUILD_FORGE.value})
        new_state = do_construction(_rng(), state, actions, params, sp)

        pending = new_state.pending_handovers
        idx = jnp.argmax(pending[:, 0] == 1)
        self.assertEqual(
            int(pending[idx, 5]),
            2,  # 2 = forge
            "Building type should be stored as 2 (forge) in pending entry",
        )


# ============================================================================
# ACTION MASKING — BUILD_SHELTER / BUILD_FORGE / BUILD_BEACON
# ============================================================================
def _clear_all_construction_sites(state):
    """Replace every CONSTRUCTION_SITE and CONSTRUCTION_IN_PROGRESS with GRASS.

    Used to test the "no site nearby" path deterministically, since
    generate_world sprinkles sites across the overworld.
    """
    m = state.map
    is_site = (m == BlockType.CONSTRUCTION_SITE.value) | (
        m == BlockType.CONSTRUCTION_IN_PROGRESS.value
    )
    cleared = jnp.where(is_site, BlockType.GRASS.value, m)
    return state.replace(map=cleared)


class TestBuildActionMask(unittest.TestCase):
    """Verify compute_action_mask exposes BUILD_* actions correctly.

    Mapping checked:
      - site nearby + materials on team → BUILD_X True
      - no site nearby → BUILD_X False (even with materials)
      - site nearby + nobody has materials → BUILD_X False
      - partial materials per-type (wood but no stone) → only buildable types
      - IN_PROGRESS handover → only the pending structure type exposed
      - expired pending → no BUILD_X via handover path
      - dead / sleeping agent → only NOOP (BUILD_X False)
    """

    # ---- positive cases: site nearby + materials ----

    def test_build_shelter_mask_true_when_near_site_with_materials(self):
        state, params, sp = _setup(coord_value=2)
        state = _clear_all_construction_sites(state)  # wipe random sites...
        state = _place_site(state, coord_value=2)  # ...then place ours back
        mask = compute_action_mask(state, params, sp)
        self.assertTrue(bool(mask[0, Action.BUILD_SHELTER.value]))

    def test_build_forge_mask_true_when_near_site_with_materials(self):
        state, params, sp = _setup(coord_value=2)
        state = _clear_all_construction_sites(state)
        state = _place_site(state, coord_value=2)
        mask = compute_action_mask(state, params, sp)
        self.assertTrue(bool(mask[0, Action.BUILD_FORGE.value]))

    def test_build_beacon_mask_true_when_near_site_with_materials(self):
        state, params, sp = _setup(coord_value=2)
        state = _clear_all_construction_sites(state)
        state = _place_site(state, coord_value=2)
        mask = compute_action_mask(state, params, sp)
        self.assertTrue(bool(mask[0, Action.BUILD_BEACON.value]))

    def test_build_mask_true_for_diagonal_adjacency(self):
        """CLOSE_BLOCKS includes diagonals — agent at (9,9), site at (10,10)."""
        state, params, sp = _setup(coord_value=2)
        state = _clear_all_construction_sites(state)
        state = _place_site(state, coord_value=2)
        state = state.replace(
            player_position=jnp.tile(jnp.array([9, 9]), (sp.player_count, 1)),
        )
        mask = compute_action_mask(state, params, sp)
        self.assertTrue(bool(mask[0, Action.BUILD_SHELTER.value]))

    # ---- negative cases: no site / no materials ----

    def test_build_mask_false_when_no_site_nearby(self):
        state, params, sp = _setup(coord_value=2)
        state = _clear_all_construction_sites(state)
        # No site re-placed. Agent has materials; mask should still be False.
        mask = compute_action_mask(state, params, sp)
        self.assertFalse(bool(mask[0, Action.BUILD_SHELTER.value]))
        self.assertFalse(bool(mask[0, Action.BUILD_FORGE.value]))
        self.assertFalse(bool(mask[0, Action.BUILD_BEACON.value]))

    def test_build_mask_false_when_nobody_has_materials(self):
        state, params, sp = _setup(coord_value=2)
        state = _clear_all_construction_sites(state)
        state = _place_site(state, coord_value=2)
        inv = state.inventory.replace(
            wood=jnp.zeros(sp.player_count, dtype=jnp.int32),
            stone=jnp.zeros(sp.player_count, dtype=jnp.int32),
            iron=jnp.zeros(sp.player_count, dtype=jnp.int32),
            coal=jnp.zeros(sp.player_count, dtype=jnp.int32),
        )
        state = state.replace(inventory=inv)
        mask = compute_action_mask(state, params, sp)
        self.assertFalse(bool(mask[0, Action.BUILD_SHELTER.value]))
        self.assertFalse(bool(mask[0, Action.BUILD_FORGE.value]))
        self.assertFalse(bool(mask[0, Action.BUILD_BEACON.value]))

    def test_build_mask_per_type_with_partial_materials(self):
        """Shelter materials only → only BUILD_SHELTER exposed."""
        state, params, sp = _setup(coord_value=2)
        state = _clear_all_construction_sites(state)
        state = _place_site(state, coord_value=2)
        # Enough for shelter (10 wood + 5 stone), NOT for forge/beacon (needs iron/coal).
        inv = state.inventory.replace(
            wood=jnp.full(sp.player_count, SHELTER_COST_WOOD, dtype=jnp.int32),
            stone=jnp.full(sp.player_count, SHELTER_COST_STONE, dtype=jnp.int32),
            iron=jnp.zeros(sp.player_count, dtype=jnp.int32),
            coal=jnp.zeros(sp.player_count, dtype=jnp.int32),
        )
        state = state.replace(inventory=inv)
        mask = compute_action_mask(state, params, sp)
        self.assertTrue(bool(mask[0, Action.BUILD_SHELTER.value]))
        self.assertFalse(bool(mask[0, Action.BUILD_FORGE.value]))
        self.assertFalse(bool(mask[0, Action.BUILD_BEACON.value]))

    # ---- handover completion path ----

    def test_build_mask_allows_completer_for_pending_type_only(self):
        """At IN_PROGRESS shelter site, BUILD_SHELTER True for completers,
        BUILD_FORGE/BEACON False — even if they have the materials."""
        state, params, sp = _setup(coord_value=-15)
        actions_setup = _actions_array(sp, {0: Action.BUILD_SHELTER.value})
        state = do_construction(_rng(), state, actions_setup, params, sp)
        mask = compute_action_mask(state, params, sp)
        # Agent 1 is a potential completer.
        self.assertTrue(bool(mask[1, Action.BUILD_SHELTER.value]))
        self.assertFalse(bool(mask[1, Action.BUILD_FORGE.value]))
        self.assertFalse(bool(mask[1, Action.BUILD_BEACON.value]))

    def test_build_mask_denies_completer_after_pending_expires(self):
        """Once pending deadline passes, BUILD_SHELTER no longer legal."""
        state, params, sp = _setup(coord_value=-3)  # short window
        state = do_construction(
            _rng(), state, _actions_array(sp, {0: Action.BUILD_SHELTER.value}), params, sp
        )
        # Advance timestep past the deadline (window=3, +1 = deadline timestep+4).
        state = state.replace(timestep=state.timestep + jnp.int32(50))
        mask = compute_action_mask(state, params, sp)
        # Initiator is NOT a completer (same agent blocked), but agent 1 would be.
        # After expiry, near_in_progress_* path is False AND the block is still
        # IN_PROGRESS (not CONSTRUCTION_SITE), so no near_construction_site either.
        self.assertFalse(bool(mask[1, Action.BUILD_SHELTER.value]))

    # ---- dead / sleeping / resting: only NOOP ----

    def test_build_mask_false_when_agent_dead(self):
        state, params, sp = _setup(coord_value=2)
        state = _clear_all_construction_sites(state)
        state = _place_site(state, coord_value=2)
        alive = state.player_alive.at[0].set(False)
        state = state.replace(player_alive=alive)
        mask = compute_action_mask(state, params, sp)
        self.assertFalse(bool(mask[0, Action.BUILD_SHELTER.value]))
        self.assertTrue(bool(mask[0, Action.NOOP.value]))
        # A live teammate should still see BUILD.
        self.assertTrue(bool(mask[1, Action.BUILD_SHELTER.value]))

    def test_build_mask_false_when_agent_sleeping(self):
        state, params, sp = _setup(coord_value=2)
        state = _clear_all_construction_sites(state)
        state = _place_site(state, coord_value=2)
        is_sleeping = state.is_sleeping.at[0].set(True)
        state = state.replace(is_sleeping=is_sleeping)
        mask = compute_action_mask(state, params, sp)
        self.assertFalse(bool(mask[0, Action.BUILD_SHELTER.value]))
        self.assertTrue(bool(mask[0, Action.NOOP.value]))


if __name__ == "__main__":
    unittest.main()
