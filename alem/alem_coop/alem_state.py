from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
from flax import struct

if TYPE_CHECKING:
    from jaxtyping import Array, Bool, Float, Int


@struct.dataclass
class Inventory:
    # Per-player item counts — each field shape (player_count,), integer.
    wood: Int[Array, "player_count"]
    stone: Int[Array, "player_count"]
    coal: Int[Array, "player_count"]
    iron: Int[Array, "player_count"]
    diamond: Int[Array, "player_count"]
    sapling: Int[Array, "player_count"]
    pickaxe: Int[Array, "player_count"]
    sword: Int[Array, "player_count"]
    bow: Int[Array, "player_count"]
    arrows: Int[Array, "player_count"]
    armour: Int[Array, "player_count 4"]  # 4 armour slots per player
    torches: Int[Array, "player_count"]
    ruby: Int[Array, "player_count"]
    sapphire: Int[Array, "player_count"]
    potions: Int[Array, "player_count 6"]  # 6 potion colours per player
    books: Int[Array, "player_count"]


@struct.dataclass
class Mobs:
    position: jnp.ndarray
    health: jnp.ndarray
    mask: jnp.ndarray
    attack_cooldown: jnp.ndarray
    type_id: jnp.ndarray


# @struct.dataclass
# class Projectiles(Mobs):
#     directions: jnp.ndarray
#     lifetimes: jnp.ndarray


@struct.dataclass
class InventorySlice:
    """Minimal inventory subset for mob scan carry (only fields accessed by mob logic)."""

    armour: jnp.ndarray


@struct.dataclass
class MobScanState:
    """Lightweight state for update_mobs scan carry.

    Contains ONLY the fields read/written by the 5 mob scan loops,
    avoiding copying all of the coordination/metric arrays 42x per step.
    """

    map: jnp.ndarray
    mob_map: jnp.ndarray
    player_level: int
    player_position: jnp.ndarray
    player_alive: jnp.ndarray
    player_health: jnp.ndarray
    is_sleeping: jnp.ndarray
    is_resting: jnp.ndarray
    achievements: jnp.ndarray
    melee_mobs: Mobs
    passive_mobs: Mobs
    ranged_mobs: Mobs
    mob_projectiles: Mobs
    mob_projectile_directions: jnp.ndarray
    mob_projectile_owners: jnp.ndarray
    player_projectiles: Mobs
    player_projectile_directions: jnp.ndarray
    player_projectile_owners: jnp.ndarray
    monsters_killed: jnp.ndarray
    inventory: InventorySlice
    armour_enchantments: jnp.ndarray
    bow_enchantment: jnp.ndarray
    player_dexterity: jnp.ndarray
    player_intelligence: jnp.ndarray
    player_specialization: jnp.ndarray
    player_food: jnp.ndarray
    player_hunger: jnp.ndarray


@struct.dataclass
class EnvState:
    # Per-level world tensors — shape (num_levels, map_height, map_width).
    map: Int[Array, "num_levels map_height map_width"]  # BlockType ids
    item_map: Int[Array, "num_levels map_height map_width"]  # ItemType ids
    mob_map: Bool[Array, "num_levels map_height map_width"]  # cell occupied by a mob
    light_map: Float[Array, "num_levels map_height map_width"]
    down_ladders: jnp.ndarray
    up_ladders: jnp.ndarray
    chests_opened: Bool[Array, "num_levels player_count"]
    monsters_killed: Int[Array, "num_levels"]

    player_position: Int[Array, "player_count 2"]
    player_level: Int[Array, ""]
    player_direction: Int[Array, "player_count"]
    player_alive: Bool[Array, "player_count"]
    player_death_cause: Int[
        Array, "player_count"
    ]  # DeathCause value per player (0=ALIVE, see constants.DeathCause)
    player_level_at_death: Int[
        Array, "player_count"
    ]  # floor level when player died; -1 if still alive

    # Intrinsics
    player_health: Float[Array, "player_count"]
    player_food: Int[Array, "player_count"]
    player_drink: Int[Array, "player_count"]
    player_energy: Int[Array, "player_count"]
    player_mana: Int[Array, "player_count"]
    is_sleeping: Bool[Array, "player_count"]
    is_resting: Bool[Array, "player_count"]

    # Second order intrinsics
    player_recover: Float[Array, "player_count"]
    player_hunger: Float[Array, "player_count"]
    player_thirst: Float[Array, "player_count"]
    player_fatigue: Float[Array, "player_count"]
    player_recover_mana: Float[Array, "player_count"]

    # Attributes
    player_xp: Int[Array, "player_count"]
    player_dexterity: Int[Array, "player_count"]
    player_strength: Int[Array, "player_count"]
    player_intelligence: Int[Array, "player_count"]
    player_specialization: Int[Array, "player_count"]

    # Request Info
    request_duration: Int[Array, "player_count"]
    request_type: Int[Array, "player_count"]

    inventory: Inventory

    melee_mobs: Mobs
    passive_mobs: Mobs
    ranged_mobs: Mobs

    mob_projectiles: Mobs
    mob_projectile_directions: jnp.ndarray
    mob_projectile_owners: jnp.ndarray
    player_projectiles: Mobs
    player_projectile_directions: jnp.ndarray
    player_projectile_owners: jnp.ndarray

    growing_plants_positions: Int[Array, "num_growing_plants 2"]
    growing_plants_age: Int[Array, "num_growing_plants"]
    growing_plants_mask: Bool[Array, "num_growing_plants"]

    potion_mapping: Int[Array, 6]
    learned_spells: Bool[Array, "player_count"]

    sword_enchantment: Int[Array, "player_count"]
    bow_enchantment: Int[Array, "player_count"]
    armour_enchantments: Int[Array, "player_count 4"]

    boss_progress: Int[Array, ""]
    boss_timesteps_to_spawn_this_round: Int[Array, ""]

    light_level: Float[Array, ""]

    achievements: Bool[Array, "player_count num_achievements"]

    state_rng: Any

    timestep: Int[Array, ""]

    # Cooperation metrics.
    # Per-agent counters are attributed to the acting agent unless noted otherwise.
    trade_count: Int[
        Array, "player_count"
    ]  # (player_count,) successful GIVE transfers initiated by this agent
    food_trade_count: Int[
        Array, "player_count"
    ]  # (player_count,) successful food transfers initiated by this agent
    drink_trade_count: Int[
        Array, "player_count"
    ]  # (player_count,) successful drink transfers initiated by this agent
    give_attempt_count: Int[
        Array, "player_count"
    ]  # (player_count,) GIVE actions selected by this agent
    request_count: Int[
        Array, "player_count"
    ]  # (player_count,) request actions started by this agent
    request_expiry_count: Int[
        Array, "player_count"
    ]  # (player_count,) requests that timed out without being renewed
    request_received_count: Int[
        Array, "player_count"
    ]  # (player_count,) successful requested transfers received
    revives: Int[Array, ""]  # team-level successful revives
    ff_damage_dealt: Float[
        Array, "player_count"
    ]  # (player_count,) friendly-fire damage dealt by this agent
    alive_agent_steps: Int[Array, ""]  # team-level sum of alive agents over the episode
    actionable_agent_steps: Int[
        Array, ""
    ]  # team-level sum of agents able to act (not dead/sleeping/resting)

    # Progression
    max_player_level: Int[Array, ""]  # deepest dungeon floor reached this episode

    # Misc Metrics
    all_necessities_frac: Float[Array, "player_count"]

    # Coordination maps - shape: (num_levels, map_height, map_width)
    # Value encoding: 0=none, 2-N=sync (requires N agents), negative=handover (window size)
    coordination_map: Int[Array, "num_levels map_height map_width"]
    # Soft coordination mask - True = soft (scales reward), False = hard (requires N agents)
    soft_coordination_mask: Bool[Array, "num_levels map_height map_width"]

    # Handover tracking - shape: (max_pending_handovers, 6)
    # Each: [active, pos_x, pos_y, deadline_timestep, initiator_agent_id, building_type]
    # building_type: 0=mining, 1=shelter, 2=forge, 3=beacon
    pending_handovers: Int[Array, "max_pending_handovers 6"]

    # Coordination metrics
    handover_successes: Int[Array, ""]  # Total handover completions across mining + construction
    handover_setups: Int[Array, ""]  # Total handover setups across mining + construction
    handover_expiries: Int[Array, ""]  # Total handovers that timed out unresolved
    # Sync coordination by agent count.
    sync_coord_by_agents: Int[Array, 2]  # shape: (2,) — [2-agent, 3+-agent]

    # Domain-specific coordination metrics
    # Mining coordination (split by soft/hard)
    coord_mine_sync_soft_count: Int[Array, ""]  # Soft sync mining successes (scales reward)
    coord_mine_sync_hard_count: Int[Array, ""]  # Hard sync mining successes (requires N agents)
    coord_mine_handover_count: Int[Array, ""]  # Coordinated handover mining successes
    coord_mine_handover_expiries: Int[Array, ""]  # Mining handovers that timed out unresolved
    # Sync coordination tracking (for coordination_success_rate)
    coord_sync_attempts: Int[
        Array, ""
    ]  # Events where any agent acted on a sync-coordination block (denominator)
    coord_sync_successes: Int[
        Array, ""
    ]  # Events where coord_req was met on a sync block (numerator)
    soft_sync_events: Int[Array, ""]  # Steps where any agent acted on a soft sync block
    soft_sync_bonus_events: Int[
        Array, ""
    ]  # Steps where soft sync bonus threshold was met (agents >= coord_req)
    # Solo soft coordination (agent on soft block without meeting coord_req)
    coord_solo_soft_attempts: Int[
        Array, ""
    ]  # Events where solo agent tried to mine a soft sync block
    coord_solo_soft_successes: Int[
        Array, ""
    ]  # Events where solo agent succeeded (passed probabilistic gate)

    # Construction coordination
    coord_construction_attempts: Int[
        Array, ""
    ]  # Sync construction attempts only (.any per timestep, requires a payer)
    coord_construction_successes: Int[
        Array, ""
    ]  # Sync construction successes only (.any per timestep)
    coord_construction_handover_count: Int[Array, ""]  # Construction handover completions
    coord_construction_handover_setups: Int[
        Array, ""
    ]  # Construction handover setups (initiator placed IN_PROGRESS)
    coord_construction_handover_expiries: Int[
        Array, ""
    ]  # Construction handovers that timed out unresolved
    # Diagnostic: BUILD_X issued while facing a CONSTRUCTION_SITE but no agent
    # on that tile has materials. Funded attempts already register as
    # coord_construction_attempts / handover_setups.
    construction_build_at_site_unfunded: Int[Array, ""]
    coord_build_shelter_count: Int[Array, ""]  # Epic Shelters built with coordination
    coord_build_forge_count: Int[Array, ""]  # Epic Forges built with coordination
    coord_build_beacon_count: Int[Array, ""]  # Epic Beacons built with coordination

    # Combat coordination (elite mob kills)
    coord_elite_attempts: Int[Array, ""]  # Agents attacking an elite/large mob (attempt)
    coord_elite_successes: Int[Array, ""]  # Attack on elite/large mob where coord was met
    coord_elite_melee_kills: Int[Array, ""]  # Elite melee mobs killed with coordination
    coord_elite_ranged_kills: Int[Array, ""]  # Elite ranged mobs killed with coordination
    coord_large_passive_kills: Int[Array, ""]  # Large passive mobs killed with coordination

    # Crafting coordination
    coord_craft_attempts: Int[Array, ""]  # Agents wanting to craft diamond gear (attempt)
    coord_craft_successes: Int[Array, ""]  # Diamond craft where coord requirement met
    coord_diamond_pickaxe_count: Int[Array, ""]  # Diamond pickaxes crafted at Epic Forge
    coord_diamond_sword_count: Int[Array, ""]  # Diamond swords crafted at Epic Forge
    coord_diamond_armour_count: Int[Array, ""]  # Diamond armour crafted at Epic Forge

    # Construction sites - shape: (num_levels, max_construction_sites)
    # 0=unbuilt, 1=shelter, 2=forge, 3=beacon
    construction_sites_built: Int[Array, "num_levels max_construction_sites"]
    construction_site_positions: Int[
        Array, "num_levels max_construction_sites 2"
    ]  # shape: (num_levels, max_construction_sites, 2)
    # Handover deadline for in-progress construction
    construction_handover_deadline: Int[
        Array, "num_levels max_construction_sites"
    ]  # shape: (num_levels, max_construction_sites)

    # Mob coordination: 0=normal, 1=elite/large soft, 2=elite/large hard
    melee_mob_coordination: Int[
        Array, "num_levels num_melee_mobs"
    ]  # shape: (num_levels, max_melee_mobs)
    ranged_mob_coordination: Int[
        Array, "num_levels num_ranged_mobs"
    ]  # shape: (num_levels, max_ranged_mobs)
    passive_mob_coordination: Int[
        Array, "num_levels num_passive_mobs"
    ]  # shape: (num_levels, max_passive_mobs)

    # Per-mob agents_required for elite/large coordination (0=normal, 2+=required count)
    melee_mob_agents_required: Int[
        Array, "num_levels num_melee_mobs"
    ]  # shape: (num_levels, max_melee_mobs)
    ranged_mob_agents_required: Int[
        Array, "num_levels num_ranged_mobs"
    ]  # shape: (num_levels, max_ranged_mobs)
    passive_mob_agents_required: Int[
        Array, "num_levels num_passive_mobs"
    ]  # shape: (num_levels, max_passive_mobs)

    # Communication messages — shape: (player_count, num_comm_channels), one-hot float vectors
    # Reset each step; set when an agent takes a comm action. Zero-sized when num_comm_channels=0.
    comm_messages: Float[Array, "player_count num_comm_channels"]
    # Cumulative count of communication actions per agent — shape: (player_count,)
    comm_count: Int[Array, "player_count"]

    # Sampled difficulty α (for randomized α training, 0.0 when using fixed params)
    sampled_alpha: Float[Array, ""]

    fractal_noise_angles: tuple[int, int, int, int] = (None, None, None, None)


@struct.dataclass
class EnvParams:
    max_timesteps: int = 100000
    day_length: int = 300

    melee_mob_health: int = 5
    passive_mob_health: int = 3
    ranged_mob_health: int = 3

    mob_despawn_distance: int = 14
    max_attribute: int = 5

    fractal_noise_angles: tuple[int, int, int, int] = (None, None, None, None)

    # Game Mode Parameters
    god_mode: bool = False
    shared_reward: bool = True
    friendly_fire: bool = True

    # Soft Specialization
    # When False (default): hard gates - only specialists can perform role-specific actions
    # When True: soft gates - anyone can attempt, but specialists have higher success rate
    soft_specialization: bool = False
    specialist_efficiency: float = 1.0  # Success rate for specialists (100%)
    non_specialist_efficiency: float = 0.2  # Success rate for non-specialists (20%)

    # Coordination System
    coordination_enabled: bool = False
    coordination_probability: float = (
        0.25  # % of eligible mining/placement blocks requiring coordination
    )
    handover_ratio: float = 0.15  # 15% handover, 85% synchronous
    soft_coordination_ratio: float = 0.5  # 50% soft (scales reward), 50% hard (requires N)
    min_agents_required: int = 2
    max_agents_required: int = 3
    handover_window_min: int = 10
    handover_window_max: int = 20
    # Two-knob difficulty parameter: P(coord event requires ALL agents vs just 2)
    p_max_agents: float = 0.5
    # Soft sync solo failure probability: P(solo mining a soft coord block fails).
    # At α=0 soft blocks are trivial; at α=1 they behave like hard blocks.
    # Scales resource slack — same opportunities, less yield without coordination.
    soft_solo_fail_prob: float = 0.0
    # Soft coordination multiplier uses pairwise complementarity: k(k+1)/2
    # (no free parameters — see game_logic.py compute_coordination_success)

    # Elite/Large Mob System (replaces floor-based combat coordination)
    elite_mob_probability: float = 0.15  # 15% base chance for elite hostile mobs
    large_passive_probability: float = 0.20  # 20% chance for large passive mobs
    hard_mob_probability: float = 0.0  # 0% of elites/large are hard coordination

    # Construction System
    construction_enabled: bool = False
    num_construction_sites: int = 8  # Sites per overworld level
    num_mining_construction_sites: int = 4  # Sites per diamond-bearing mining level
    soft_construction_ratio: float = (
        0.5  # 50% soft (solo buildable), 50% (hard, requires sync & handover)
    )

    # Diamond Crafting Coordination
    crafting_coordination_enabled: bool = False
    diamond_crafting_agents_required: int = 2

    # Base difficulty scaling (non-coordination game difficulty)
    # When True, α scales mob health and starting resources.
    # mob_damage is NOT scaled — higher damage teaches mob avoidance, undermining combat coordination.
    scale_base_difficulty: bool = False
    mob_health_multiplier: float = 1.0  # multiplied into mob HP at spawn (β=1/3)
    starting_resource_multiplier: float = (
        1.0  # scales starting food/drink/energy (β=1/3; not health — it rests back)
    )

    # Randomized difficulty (α domain randomization)
    # When True, each episode samples α ~ U[alpha_min, alpha_max] and derives
    # the 7 coordination difficulty params from it, overriding the fixed values.
    randomize_alpha: bool = False
    alpha_min: float = 0.2
    alpha_max: float = 0.85


@struct.dataclass
class StaticEnvParams:
    version: str = "v0.13741-fix-construction-more-metrics"
    map_size: tuple[int, int] = (48, 48)
    num_levels: int = 9
    player_count: int = 3

    # Mobs Per Player
    max_melee_mobs: int = 3
    max_passive_mobs: int = 3
    max_growing_plants: int = 10
    max_ranged_mobs: int = 2
    max_mob_projectiles: int = 3
    max_player_projectiles: int = 3

    # Coordination
    max_pending_handovers: int = 8
    max_construction_sites: int = 12

    # Communication channels (MPE-style discrete messages)
    # 0 = disabled (no extra actions/obs), >0 = num one-hot comm dimensions per agent
    num_comm_channels: int = 0


# Opportunity parameters — base values for coordination.
# These control HOW MUCH coordination exists in the world (density/presence).
# All values here are FIXED across difficulties — only execution difficulty
# (p_max_agents, handover windows, soft_solo_fail_prob) scales with α.
COORDINATION_OPPORTUNITY_PARAMS = dict(
    coordination_enabled=True,
    coordination_probability=0.25,  # fraction of eligible mining blocks with coord requirements
    soft_coordination_ratio=0.5,  # mining: 50% soft (reward-scaled), 50% hard (requires N)
    handover_ratio=0.25,  # mining: handover fraction
    hard_mob_probability=0.5,  # of elite mobs: 50% hard-coord, 50% soft
    elite_mob_probability=0.15,  # hostile mobs: 15% are elite (need coord)
    large_passive_probability=0.20,  # passive mobs: 20% are large (need coord)
    construction_enabled=True,
    num_construction_sites=8,  # construction sites per overworld level
    num_mining_construction_sites=4,  # construction sites per mining level
    soft_construction_ratio=0.5,  # construction: 50% soft, 50% hard
    crafting_coordination_enabled=True,
    diamond_crafting_agents_required=2,
    soft_specialization=True,  # always on when coordination is enabled
)

# ── Difficulty scaling from a single scalar α ∈ [0, 1] ────────────────────
#
# Only EXECUTION DIFFICULTY scales with α — opportunity counts are fixed.
# This ensures the same number and type of coordination events at every
# difficulty; only the bar for succeeding changes.
#
# A. Coordination stringency (harder to execute):
#    - p_max_agents = α: P(coord event requires ALL agents vs just 2)
#    - handover_window_min/max = ⌈base·(1−α)⌉: tighter handover deadlines
#
# B. Resource slack (less yield without coordination):
#    - soft_solo_fail_prob = α: P(solo mining a soft coord block fails).
#      Stag Hunt with degraded outside option — coordination increasingly
#      dominant for scarce resources at higher α.
#
# C. Specialization pressure (role gates tighten):
#    - non_specialist_efficiency = 1 − α: non-specialists succeed less often.
#      At α=0 everyone is a generalist; at α=1 only specialists can act (hard gates).
#      Affects mining indirectly through pickaxe crafting gates.
#
# D. Base game difficulty (only when scale_base_difficulty=True):
#    β = 1/3 applied to two levers (mob health, starting resources).
#    mob_damage is intentionally NOT scaled — higher mob damage teaches agents to
#    avoid mobs entirely, undermining combat coordination. Tankier mobs + fewer
#    resources creates pressure without distorting the combat incentive.
#    - mob_health_multiplier        = 1 + β·α: mobs have up to 30% more HP
#    - starting_resource_multiplier = 1 − β·α: start with up to 30% fewer resources
#
# ── Generated parameter values ──────────────────────────────────────────────
#
#   Parameter                       Formula           Easy(α=.30)  Med(α=.60)  Hard(α=.90)
#   p_max_agents                    α                     0.30        0.60        0.90
#   handover_window_min             max(3,⌈12(1−α)⌉)         9           5           3
#   handover_window_max             max(6,⌈24(1−α)⌉)        17          10           6
#   soft_solo_fail_prob             α                     0.30        0.60        0.90
#   non_specialist_efficiency       1−α                   0.70        0.40        0.10
#   mob_health_multiplier*          1+α/3                 1.10        1.20        1.30
#   starting_resource_multiplier*   1−α/3                 0.90        0.80        0.70
#   (* only when scale_base_difficulty=True)
#
# Fixed across all difficulties (in COORDINATION_OPPORTUNITY_PARAMS):
#   coordination_probability=0.25, handover_ratio=0.25,
#   soft_coordination_ratio=0.5, hard_mob_probability=0.5,
#   elite_mob_probability=0.15, large_passive_probability=0.20,
#   soft_construction_ratio=0.5

DIFFICULTY_ALPHAS = {"easy": 0.30, "medium": 0.60, "hard": 0.90}

# Map geometry constants used in window derivation
_MAP_WIDTH = 48
_WINDOW_MIN_BASE = _MAP_WIDTH // 4  # 12: quarter-map traversal
_WINDOW_MAX_BASE = _MAP_WIDTH // 2  # 24: half-map traversal
_WINDOW_MIN_FLOOR = 3  # prevents free retry spam (2-step degenerate case)
_WINDOW_MAX_FLOOR = 6  # same reasoning for max window


def _difficulty_params_from_alpha(alpha: float, scale_base: bool = False) -> dict:
    """Derive execution difficulty parameters from a single scalar α ∈ [0, 1].

    Only scales HOW HARD each coordination event is, not how many exist.
    Every parameter is a monotonic function of α with zero free constants.

    When scale_base=True, also scales base game difficulty (mob HP/damage, starting resources).
    """
    params = dict(
        p_max_agents=round(alpha, 4),  # P(require ALL agents vs just 2)
        handover_window_min=max(
            _WINDOW_MIN_FLOOR, math.ceil(_WINDOW_MIN_BASE * (1 - alpha))
        ),  # tighter deadlines at high α
        handover_window_max=max(_WINDOW_MAX_FLOOR, math.ceil(_WINDOW_MAX_BASE * (1 - alpha))),
        soft_solo_fail_prob=round(alpha, 4),  # P(solo soft coord mining fails)
        non_specialist_efficiency=round(1.0 - alpha, 4),  # non-specialist success rate
    )
    if scale_base:
        # β = 1/3 caps base perturbations at ≤33% — within the game's survivable regime
        # and strictly weaker than coordination params (which span the full α range).
        # mob_damage excluded: higher damage teaches mob avoidance, undermining combat coordination.
        _B = 1.0 / 3.0
        params.update(
            mob_health_multiplier=round(1.0 + _B * alpha, 4),  # 1.0–1.3× mob HP
            starting_resource_multiplier=round(1.0 - _B * alpha, 4),  # 1.0–0.7× starting resources
        )
    return params


def _jax_difficulty_params_from_alpha(alpha, scale_base=False):
    """JAX-traceable version of _difficulty_params_from_alpha for use inside JIT."""
    params = dict(
        p_max_agents=alpha,
        handover_window_min=jnp.maximum(
            _WINDOW_MIN_FLOOR, jnp.ceil(_WINDOW_MIN_BASE * (1.0 - alpha)).astype(jnp.int32)
        ),
        handover_window_max=jnp.maximum(
            _WINDOW_MAX_FLOOR, jnp.ceil(_WINDOW_MAX_BASE * (1.0 - alpha)).astype(jnp.int32)
        ),
        soft_solo_fail_prob=alpha,
        non_specialist_efficiency=1.0 - alpha,
    )
    if scale_base:
        _B = 1.0 / 3.0
        params.update(
            mob_health_multiplier=1.0 + _B * alpha,
            starting_resource_multiplier=1.0 - _B * alpha,
        )
    return params


COORDINATION_DIFFICULTY_PARAMS = {
    name: _difficulty_params_from_alpha(alpha) for name, alpha in DIFFICULTY_ALPHAS.items()
}

# "none" preset — coordination fully disabled, used as a baseline.
COORDINATION_NONE_PARAMS = dict(
    coordination_enabled=False,
    coordination_probability=0.0,
    soft_coordination_ratio=0.5,
    handover_ratio=0.0,
    hard_mob_probability=0.0,
    p_max_agents=0.0,
    soft_solo_fail_prob=0.0,
    min_agents_required=2,
    max_agents_required=2,
    handover_window_min=10,
    handover_window_max=20,
    elite_mob_probability=0.0,
    large_passive_probability=0.0,
    construction_enabled=False,
    num_construction_sites=0,
    num_mining_construction_sites=0,
    soft_construction_ratio=0.5,
    crafting_coordination_enabled=False,
    diamond_crafting_agents_required=2,
)

# Combined presets for backwards compatibility — merge opportunity + difficulty params.
COORDINATION_PRESETS = {
    "none": COORDINATION_NONE_PARAMS,
    "easy": {**COORDINATION_OPPORTUNITY_PARAMS, **COORDINATION_DIFFICULTY_PARAMS["easy"]},
    "medium": {**COORDINATION_OPPORTUNITY_PARAMS, **COORDINATION_DIFFICULTY_PARAMS["medium"]},
    "hard": {**COORDINATION_OPPORTUNITY_PARAMS, **COORDINATION_DIFFICULTY_PARAMS["hard"]},
}


def get_coordination_params(difficulty: str | float, scale_base: bool = False) -> dict:
    """Return coordination parameter overrides for a given difficulty tier.

    Args:
        difficulty: One of "none", "easy", "medium", "hard", or a numeric
            value (float or string) for a custom α in [0, 1] (e.g. 0.33 or "0.33").
        scale_base: When True, also include base game difficulty params
            (mob_health_multiplier, mob_damage_multiplier, starting_resource_multiplier).

    Returns:
        A dict of parameter names to values suitable for passing to EnvParams.
    """
    difficulty = str(difficulty).lower()
    if difficulty in COORDINATION_PRESETS:
        result = dict(COORDINATION_PRESETS[difficulty])
        if scale_base and difficulty != "none":
            alpha = DIFFICULTY_ALPHAS[difficulty]
            _B = 1.0 / 3.0
            result.update(
                scale_base_difficulty=True,
                mob_health_multiplier=round(1.0 + _B * alpha, 4),
                starting_resource_multiplier=round(1.0 - _B * alpha, 4),
            )
        return result
    # Support custom α values (e.g. "0.33" for intermediate difficulty)
    try:
        alpha = float(difficulty)
    except ValueError:
        raise ValueError(
            f"Unknown coordination difficulty {difficulty!r}. "
            f"Choose from: {list(COORDINATION_PRESETS.keys())} "
            f"or pass a numeric α ∈ [0, 1]."
        )
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"Custom α must be in [0, 1], got {alpha}")
    result = {
        **COORDINATION_OPPORTUNITY_PARAMS,
        **_difficulty_params_from_alpha(alpha, scale_base=scale_base),
    }
    if scale_base:
        result["scale_base_difficulty"] = True
    return result
