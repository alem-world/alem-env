from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from ..alem_state import EnvState, StaticEnvParams
from ..constants import (
    ACHIEVEMENT_REWARD_MAP,
    NUM_ACHIEVEMENTS,
    Achievement,
)

if TYPE_CHECKING:
    from jaxtyping import Array, Bool, Float

# Achievement to Specialization Role Mapping
# maps achievemnts to the role that should primarily achieve it
# 0 = SHARED (anyone can/should do), 1 = FORAGER, 2 = WARRIOR, 3 = MINER
#
# Based on game mechanics:
# - FORAGER: Efficient at saplings, water, planting (soft-gated actions)
# - WARRIOR: Efficient at crafting swords (soft-gated actions)
# - MINER: Efficient at crafting pickaxes (soft-gated actions)

ACHIEVEMENT_ROLE_MAP = {
    # SHARED achievements (0) - Anyone can/should do these (not soft-gated)
    Achievement.COLLECT_WOOD.value: 0,
    Achievement.PLACE_TABLE.value: 0,
    Achievement.PLACE_FURNACE.value: 0,
    Achievement.WAKE_UP.value: 0,
    Achievement.MAKE_WOOD_SWORD.value: 0,
    Achievement.MAKE_IRON_ARMOUR.value: 0,
    Achievement.MAKE_DIAMOND_ARMOUR.value: 0,
    Achievement.ENCHANT_ARMOUR.value: 0,
    Achievement.ENTER_GNOMISH_MINES.value: 0,
    Achievement.ENTER_DUNGEON.value: 0,
    Achievement.ENTER_SEWERS.value: 0,
    Achievement.ENTER_VAULT.value: 0,
    Achievement.ENTER_TROLL_MINES.value: 0,
    Achievement.ENTER_FIRE_REALM.value: 0,
    Achievement.ENTER_ICE_REALM.value: 0,
    Achievement.ENTER_GRAVEYARD.value: 0,
    Achievement.OPEN_CHEST.value: 0,
    Achievement.DRINK_POTION.value: 0,
    # Mining resources - not gated, just needs pickaxe level
    Achievement.COLLECT_STONE.value: 0,
    Achievement.COLLECT_COAL.value: 0,
    Achievement.COLLECT_IRON.value: 0,
    Achievement.COLLECT_DIAMOND.value: 0,
    Achievement.COLLECT_SAPPHIRE.value: 0,
    Achievement.COLLECT_RUBY.value: 0,
    # Placing torch - not gated
    Achievement.PLACE_TORCH.value: 0,
    # Eating plants - not gated
    Achievement.EAT_PLANT.value: 0,
    # Placing saplings - not gated (anyone with sapling can place)
    Achievement.PLACE_PLANT.value: 0,
    # Combat - not gated (anyone can attack)
    Achievement.DEFEAT_ZOMBIE.value: 0,
    Achievement.DEFEAT_SKELETON.value: 0,
    Achievement.DEFEAT_GNOME_WARRIOR.value: 0,
    Achievement.DEFEAT_GNOME_ARCHER.value: 0,
    Achievement.DEFEAT_ORC_SOLIDER.value: 0,
    Achievement.DEFEAT_ORC_MAGE.value: 0,
    Achievement.DEFEAT_LIZARD.value: 0,
    Achievement.DEFEAT_KOBOLD.value: 0,
    Achievement.DEFEAT_KNIGHT.value: 0,
    Achievement.DEFEAT_ARCHER.value: 0,
    Achievement.DEFEAT_TROLL.value: 0,
    Achievement.DEFEAT_DEEP_THING.value: 0,
    Achievement.DEFEAT_PIGMAN.value: 0,
    Achievement.DEFEAT_FIRE_ELEMENTAL.value: 0,
    Achievement.DEFEAT_FROST_TROLL.value: 0,
    Achievement.DEFEAT_ICE_ELEMENTAL.value: 0,
    Achievement.DAMAGE_NECROMANCER.value: 0,
    Achievement.DEFEAT_NECROMANCER.value: 0,
    # Spells - not specifically gated
    Achievement.LEARN_SPELL.value: 0,
    Achievement.CAST_SPELL.value: 0,
    # Bow usage - not gated
    Achievement.FIRE_BOW.value: 0,
    # FORAGER achievements (1) - Soft-gated by is_forager
    Achievement.COLLECT_SAPLING.value: 1,  # sapling_prob = 0.2 * is_forager
    Achievement.COLLECT_DRINK.value: 1,  # is_drinking_water requires is_forager
    Achievement.COLLECT_FOOD.value: 1,  # from killing passive mobs with can_eat=is_forager
    Achievement.EAT_COW.value: 1,  # can_eat gates achievement in attack_mob
    Achievement.EAT_BAT.value: 1,  # can_eat gates achievement in attack_mob
    Achievement.EAT_SNAIL.value: 1,  # can_eat gates achievement in attack_mob
    # WARRIOR achievements (2) - Soft-gated by is_warrior
    Achievement.MAKE_STONE_SWORD.value: 2,  # crafting gated
    Achievement.MAKE_IRON_SWORD.value: 2,  # crafting gated
    Achievement.MAKE_DIAMOND_SWORD.value: 2,  # crafting gated
    Achievement.MAKE_ARROW.value: 2,  # crafting gated
    Achievement.FIND_BOW.value: 2,  # chest loot gated by is_warrior
    Achievement.ENCHANT_SWORD.value: 2,  # enchanting gated
    # MINER achievements (3) - Soft-gated by is_miner
    Achievement.MAKE_WOOD_PICKAXE.value: 3,  # crafting gated
    Achievement.MAKE_STONE_PICKAXE.value: 3,  # crafting gated
    Achievement.MAKE_IRON_PICKAXE.value: 3,  # crafting gated
    Achievement.MAKE_DIAMOND_PICKAXE.value: 3,  # crafting gated
    Achievement.MAKE_TORCH.value: 3,  # crafting gated
    Achievement.PLACE_STONE.value: 3,  # placement gated
    # COORDINATION achievements (0 = SHARED) - require multiple agents
    Achievement.COORD_2_AGENTS_SOFT.value: 0,
    Achievement.COORD_2_AGENTS_HARD.value: 0,
    Achievement.COORD_3_AGENTS_SOFT.value: 0,
    Achievement.COORD_3_AGENTS_HARD.value: 0,
    Achievement.HANDOVER_COMPLETE.value: 0,
    Achievement.COORD_MINE_HANDOVER.value: 0,
    Achievement.COORD_BUILD_SHELTER.value: 0,
    Achievement.COORD_BUILD_FORGE.value: 0,
    Achievement.COORD_BUILD_BEACON.value: 0,
    Achievement.COORD_ELITE_MELEE_KILL.value: 0,
    Achievement.COORD_ELITE_RANGED_KILL.value: 0,
    Achievement.COORD_LARGE_PASSIVE_KILL.value: 0,
    Achievement.COORD_DIAMOND_PICKAXE.value: 0,
    Achievement.COORD_DIAMOND_SWORD.value: 0,
    Achievement.COORD_DIAMOND_ARMOUR.value: 0,
    # Resource-specific coordinated mining
    Achievement.COORD_MINE_STONE_SOFT.value: 0,
    Achievement.COORD_MINE_STONE_HARD.value: 0,
    Achievement.COORD_MINE_COAL_SOFT.value: 0,
    Achievement.COORD_MINE_COAL_HARD.value: 0,
    Achievement.COORD_MINE_IRON_SOFT.value: 0,
    Achievement.COORD_MINE_IRON_HARD.value: 0,
    Achievement.COORD_MINE_DIAMOND_SOFT.value: 0,
    Achievement.COORD_MINE_DIAMOND_HARD.value: 0,
    Achievement.COORD_MINE_SAPPHIRE_SOFT.value: 0,
    Achievement.COORD_MINE_SAPPHIRE_HARD.value: 0,
    Achievement.COORD_MINE_RUBY_SOFT.value: 0,
    Achievement.COORD_MINE_RUBY_HARD.value: 0,
}

# Coordination Achievements - require explicit multi-agent mechanics
# (sync, handover, elite mobs, diamond crafting at epic forge)
COORDINATION_ACHIEVEMENTS = {
    Achievement.COORD_2_AGENTS_SOFT.value,
    Achievement.COORD_2_AGENTS_HARD.value,
    Achievement.COORD_3_AGENTS_SOFT.value,
    Achievement.COORD_3_AGENTS_HARD.value,
    Achievement.HANDOVER_COMPLETE.value,
    Achievement.COORD_MINE_HANDOVER.value,
    Achievement.COORD_BUILD_SHELTER.value,
    Achievement.COORD_BUILD_FORGE.value,
    Achievement.COORD_BUILD_BEACON.value,
    Achievement.COORD_ELITE_MELEE_KILL.value,
    Achievement.COORD_ELITE_RANGED_KILL.value,
    Achievement.COORD_LARGE_PASSIVE_KILL.value,
    Achievement.COORD_DIAMOND_PICKAXE.value,
    Achievement.COORD_DIAMOND_SWORD.value,
    Achievement.COORD_DIAMOND_ARMOUR.value,
    # Resource-specific coordinated mining
    Achievement.COORD_MINE_STONE_SOFT.value,
    Achievement.COORD_MINE_STONE_HARD.value,
    Achievement.COORD_MINE_COAL_SOFT.value,
    Achievement.COORD_MINE_COAL_HARD.value,
    Achievement.COORD_MINE_IRON_SOFT.value,
    Achievement.COORD_MINE_IRON_HARD.value,
    Achievement.COORD_MINE_DIAMOND_SOFT.value,
    Achievement.COORD_MINE_DIAMOND_HARD.value,
    Achievement.COORD_MINE_SAPPHIRE_SOFT.value,
    Achievement.COORD_MINE_SAPPHIRE_HARD.value,
    Achievement.COORD_MINE_RUBY_SOFT.value,
    Achievement.COORD_MINE_RUBY_HARD.value,
}

# Soft coordination: scales reward with agent count but doesn't strictly require N agents
SOFT_COORDINATION_ACHIEVEMENTS = {
    Achievement.COORD_2_AGENTS_SOFT.value,
    Achievement.COORD_3_AGENTS_SOFT.value,
    Achievement.COORD_MINE_STONE_SOFT.value,
    Achievement.COORD_MINE_COAL_SOFT.value,
    Achievement.COORD_MINE_IRON_SOFT.value,
    Achievement.COORD_MINE_DIAMOND_SOFT.value,
    Achievement.COORD_MINE_SAPPHIRE_SOFT.value,
    Achievement.COORD_MINE_RUBY_SOFT.value,
}

# Synchronous hard coordination: strictly requires N agents simultaneously
# Note: COORD_BUILD_* excluded — construction sites can be completed via either sync or
# handover, so they belong in their own construction category rather than sync-hard.
SYNC_HARD_COORDINATION_ACHIEVEMENTS = {
    Achievement.COORD_2_AGENTS_HARD.value,
    Achievement.COORD_3_AGENTS_HARD.value,
    Achievement.COORD_MINE_STONE_HARD.value,
    Achievement.COORD_MINE_COAL_HARD.value,
    Achievement.COORD_MINE_IRON_HARD.value,
    Achievement.COORD_MINE_DIAMOND_HARD.value,
    Achievement.COORD_MINE_SAPPHIRE_HARD.value,
    Achievement.COORD_MINE_RUBY_HARD.value,
    Achievement.COORD_ELITE_MELEE_KILL.value,
    Achievement.COORD_ELITE_RANGED_KILL.value,
    Achievement.COORD_LARGE_PASSIVE_KILL.value,
    Achievement.COORD_DIAMOND_PICKAXE.value,
    Achievement.COORD_DIAMOND_SWORD.value,
    Achievement.COORD_DIAMOND_ARMOUR.value,
}

# Handover coordination: sequential — agent A acts first, agent B completes within a window
HANDOVER_COORDINATION_ACHIEVEMENTS = {
    Achievement.HANDOVER_COMPLETE.value,
    Achievement.COORD_MINE_HANDOVER.value,
}

# Construction coordination: building epic structures, which may require either sync or
# handover depending on the site's coordination_map value.
CONSTRUCTION_COORDINATION_ACHIEVEMENTS = {
    Achievement.COORD_BUILD_SHELTER.value,
    Achievement.COORD_BUILD_FORGE.value,
    Achievement.COORD_BUILD_BEACON.value,
}

# Hard coordination: all non-soft (sync hard + handover + construction) — kept for backwards compatibility
HARD_COORDINATION_ACHIEVEMENTS = (
    SYNC_HARD_COORDINATION_ACHIEVEMENTS
    | HANDOVER_COORDINATION_ACHIEVEMENTS
    | CONSTRUCTION_COORDINATION_ACHIEVEMENTS
)

ROLE_NAMES = ["shared", "forager", "warrior", "miner"]


def _build_role_achievement_mask() -> Float[Array, "4 num_achievements"]:
    """Build a mask array of shape (4, NUM_ACHIEVEMENTS) for each role."""
    mask = jnp.zeros((4, NUM_ACHIEVEMENTS), dtype=jnp.float32)
    for ach in Achievement:
        role = ACHIEVEMENT_ROLE_MAP.get(ach.value, 0)
        mask = mask.at[role, ach.value].set(1.0)
    return mask


def _build_coordination_achievement_mask() -> Float[Array, "num_achievements"]:
    """Build a mask for coordination achievements (multi-agent mechanics)."""
    mask = jnp.zeros(NUM_ACHIEVEMENTS, dtype=jnp.float32)
    for ach_value in COORDINATION_ACHIEVEMENTS:
        mask = mask.at[ach_value].set(1.0)
    return mask


def _build_achievement_set_mask(achievement_set) -> Float[Array, "num_achievements"]:
    """Build a mask for any set of achievement values."""
    mask = jnp.zeros(NUM_ACHIEVEMENTS, dtype=jnp.float32)
    for ach_value in achievement_set:
        mask = mask.at[ach_value].set(1.0)
    return mask


ROLE_ACHIEVEMENT_MASK = _build_role_achievement_mask()
COORDINATION_ACHIEVEMENT_MASK = _build_coordination_achievement_mask()
NORMAL_ACHIEVEMENT_MASK = 1.0 - COORDINATION_ACHIEVEMENT_MASK
SOFT_COORDINATION_ACHIEVEMENT_MASK = _build_achievement_set_mask(SOFT_COORDINATION_ACHIEVEMENTS)
SYNC_HARD_COORDINATION_ACHIEVEMENT_MASK = _build_achievement_set_mask(
    SYNC_HARD_COORDINATION_ACHIEVEMENTS
)
HANDOVER_COORDINATION_ACHIEVEMENT_MASK = _build_achievement_set_mask(
    HANDOVER_COORDINATION_ACHIEVEMENTS
)
CONSTRUCTION_COORDINATION_ACHIEVEMENT_MASK = _build_achievement_set_mask(
    CONSTRUCTION_COORDINATION_ACHIEVEMENTS
)
HARD_COORDINATION_ACHIEVEMENT_MASK = _build_achievement_set_mask(HARD_COORDINATION_ACHIEVEMENTS)


# Achievements achievable on overworld (single-level / debug env)
# Positive list: only these are counted when num_levels == 1.
# New achievements must be added here explicitly to be eligible.
_ON_OVERWORLD = [
    # Resources (8) — all ores except sapphire/ruby spawn on overworld
    Achievement.COLLECT_WOOD.value,
    Achievement.COLLECT_STONE.value,
    Achievement.COLLECT_COAL.value,
    Achievement.COLLECT_IRON.value,
    Achievement.COLLECT_DIAMOND.value,
    Achievement.COLLECT_SAPLING.value,
    Achievement.COLLECT_DRINK.value,
    Achievement.COLLECT_FOOD.value,
    # Crafting (12) — diamond crafting via Epic Forge (built on overworld)
    Achievement.MAKE_WOOD_PICKAXE.value,
    Achievement.MAKE_WOOD_SWORD.value,
    Achievement.MAKE_STONE_PICKAXE.value,
    Achievement.MAKE_STONE_SWORD.value,
    Achievement.MAKE_IRON_PICKAXE.value,
    Achievement.MAKE_IRON_SWORD.value,
    Achievement.MAKE_DIAMOND_PICKAXE.value,
    Achievement.MAKE_DIAMOND_SWORD.value,
    Achievement.MAKE_IRON_ARMOUR.value,
    Achievement.MAKE_DIAMOND_ARMOUR.value,
    Achievement.MAKE_ARROW.value,
    Achievement.MAKE_TORCH.value,
    # Placement (5)
    Achievement.PLACE_TABLE.value,
    Achievement.PLACE_STONE.value,
    Achievement.PLACE_FURNACE.value,
    Achievement.PLACE_PLANT.value,
    Achievement.PLACE_TORCH.value,
    # Survival (3)
    Achievement.EAT_COW.value,
    Achievement.EAT_PLANT.value,
    Achievement.WAKE_UP.value,
    # Combat (2) — zombies and skeletons spawn on overworld
    Achievement.DEFEAT_ZOMBIE.value,
    Achievement.DEFEAT_SKELETON.value,
    # Coordination — generic (4) — mining blocks can have coordination reqs
    Achievement.COORD_2_AGENTS_SOFT.value,
    Achievement.COORD_2_AGENTS_HARD.value,
    Achievement.COORD_3_AGENTS_SOFT.value,
    Achievement.COORD_3_AGENTS_HARD.value,
    # Coordination — handover (1)
    Achievement.HANDOVER_COMPLETE.value,
    # Coordination — mining handover (1)
    Achievement.COORD_MINE_HANDOVER.value,
    # Coordination — construction (3) — sites spawn on overworld
    Achievement.COORD_BUILD_SHELTER.value,
    Achievement.COORD_BUILD_FORGE.value,
    Achievement.COORD_BUILD_BEACON.value,
    # Coordination — combat (3) — elite/large mobs can spawn on overworld
    Achievement.COORD_ELITE_MELEE_KILL.value,
    Achievement.COORD_ELITE_RANGED_KILL.value,
    Achievement.COORD_LARGE_PASSIVE_KILL.value,
    # Coordination — diamond crafting (3) — via Epic Forge on overworld
    Achievement.COORD_DIAMOND_PICKAXE.value,
    Achievement.COORD_DIAMOND_SWORD.value,
    Achievement.COORD_DIAMOND_ARMOUR.value,
    # Coordination — resource-specific mining (12) — soft and hard per resource
    Achievement.COORD_MINE_STONE_SOFT.value,
    Achievement.COORD_MINE_STONE_HARD.value,
    Achievement.COORD_MINE_COAL_SOFT.value,
    Achievement.COORD_MINE_COAL_HARD.value,
    Achievement.COORD_MINE_IRON_SOFT.value,
    Achievement.COORD_MINE_IRON_HARD.value,
    Achievement.COORD_MINE_DIAMOND_SOFT.value,
    Achievement.COORD_MINE_DIAMOND_HARD.value,
    Achievement.COORD_MINE_SAPPHIRE_SOFT.value,
    Achievement.COORD_MINE_SAPPHIRE_HARD.value,
    Achievement.COORD_MINE_RUBY_SOFT.value,
    Achievement.COORD_MINE_RUBY_HARD.value,
    # NOTE: OPEN_CHEST, DRINK_POTION, FIRE_BOW, FIND_BOW, LEARN_SPELL, CAST_SPELL,
    # ENCHANT_SWORD, ENCHANT_ARMOUR are intentionally excluded — verified that
    # chests, enchantment tables, and book-bearing loot only spawn in dungeon
    # rooms / Sewers / Vaults (see world_gen.py add_rooms and SEWER_CONFIG /
    # VAULTS_CONFIG), so these cannot be achieved on a single-level overworld env.
]


def _build_overworld_achievement_mask() -> Float[Array, "num_achievements"]:
    """Build a mask: 1.0 for overworld-achievable achievements, 0.0 otherwise."""
    mask = jnp.zeros(NUM_ACHIEVEMENTS, dtype=jnp.float32)
    for ach_val in _ON_OVERWORLD:
        mask = mask.at[ach_val].set(1.0)
    return mask


OVERWORLD_ACHIEVEMENT_MASK = _build_overworld_achievement_mask()


def _safe_div(num: Array, denom: Array) -> Float[Array, ...]:
    """Divide guarded against zero denominator (returns 0.0 instead of NaN/inf)."""
    return jnp.where(denom > 0, num / jnp.maximum(denom, 1e-8), 0.0)


def compute_score(
    state: EnvState, done: Bool[Array, ""], static_params: StaticEnvParams
) -> dict[str, Array]:
    """Compute episode achievement, reward, and coordination metrics.

    Args:
        state: Current environment state and accumulated counters.
        done: Scalar flag used to expose terminal achievement percentages.
        static_params: Static parameters defining eligible agents and levels.

    Returns:
        Flat metrics dictionary suitable for logging and evaluation.
    """
    achievements = state.achievements * done * 100.0
    info = {}
    num_agents = static_params.player_count

    eligible_mask = jnp.ones(NUM_ACHIEVEMENTS, dtype=jnp.float32)

    # Mask out dungeon-only achievements for overworld-only (debug) env
    eligible_mask = jnp.where(
        static_params.num_levels == 1,
        eligible_mask * OVERWORLD_ACHIEVEMENT_MASK,
        eligible_mask,
    )

    # Single agent: coordination achievements are unreachable, exclude from pct denominators
    if static_params.player_count == 1:
        eligible_mask = eligible_mask * NORMAL_ACHIEVEMENT_MASK
    total_possible = eligible_mask.sum()

    # Eligible coordination and normal masks
    eligible_coord_mask = COORDINATION_ACHIEVEMENT_MASK * eligible_mask
    eligible_normal_mask = NORMAL_ACHIEVEMENT_MASK * eligible_mask
    eligible_soft_coord_mask = SOFT_COORDINATION_ACHIEVEMENT_MASK * eligible_mask
    eligible_sync_hard_coord_mask = SYNC_HARD_COORDINATION_ACHIEVEMENT_MASK * eligible_mask
    eligible_handover_coord_mask = HANDOVER_COORDINATION_ACHIEVEMENT_MASK * eligible_mask
    eligible_construction_coord_mask = CONSTRUCTION_COORDINATION_ACHIEVEMENT_MASK * eligible_mask
    total_possible_coord = eligible_coord_mask.sum()
    total_possible_normal = eligible_normal_mask.sum()
    total_possible_soft_coord = eligible_soft_coord_mask.sum()
    total_possible_sync_hard_coord = eligible_sync_hard_coord_mask.sum()
    total_possible_handover_coord = eligible_handover_coord_mask.sum()
    total_possible_construction_coord = eligible_construction_coord_mask.sum()

    # Reward as % of max achievable (matches paper metric).
    # Uses raw binary achievements (not the *100 scaled version) and eligible mask
    # so overworld-only envs report % of overworld-achievable reward, not full game.
    eligible_reward_map = ACHIEVEMENT_REWARD_MAP * eligible_mask
    max_reward = eligible_reward_map.sum()  # max achievable per agent
    eligible_coord_reward_map = eligible_reward_map * COORDINATION_ACHIEVEMENT_MASK
    eligible_normal_reward_map = eligible_reward_map * NORMAL_ACHIEVEMENT_MASK
    eligible_soft_coord_reward_map = eligible_reward_map * SOFT_COORDINATION_ACHIEVEMENT_MASK
    eligible_sync_hard_coord_reward_map = (
        eligible_reward_map * SYNC_HARD_COORDINATION_ACHIEVEMENT_MASK
    )
    eligible_handover_coord_reward_map = (
        eligible_reward_map * HANDOVER_COORDINATION_ACHIEVEMENT_MASK
    )
    eligible_construction_coord_reward_map = (
        eligible_reward_map * CONSTRUCTION_COORDINATION_ACHIEVEMENT_MASK
    )
    max_coord_reward = eligible_coord_reward_map.sum()
    max_normal_reward = eligible_normal_reward_map.sum()
    max_soft_coord_reward = eligible_soft_coord_reward_map.sum()
    max_sync_hard_coord_reward = eligible_sync_hard_coord_reward_map.sum()
    max_handover_coord_reward = eligible_handover_coord_reward_map.sum()
    max_construction_coord_reward = eligible_construction_coord_reward_map.sum()
    for agent_idx in range(num_agents):
        agent_reward = (state.achievements[agent_idx] * eligible_reward_map).sum()
        info[f"Agent{agent_idx}/reward_pct_of_max"] = _safe_div(agent_reward, max_reward) * done
        agent_normal_reward = (state.achievements[agent_idx] * eligible_normal_reward_map).sum()
        info[f"Agent{agent_idx}/normal_reward_pct_of_max"] = (
            _safe_div(agent_normal_reward, max_normal_reward) * done
        )
        if static_params.player_count > 1:
            agent_coord_reward = (state.achievements[agent_idx] * eligible_coord_reward_map).sum()
            info[f"Agent{agent_idx}/coord_reward_pct_of_max"] = (
                _safe_div(agent_coord_reward, max_coord_reward) * done
            )
    # Team: an achievement counts if ANY agent achieved it
    team_ach = (state.achievements > 0).any(axis=0).astype(jnp.float32)
    team_reward = (team_ach * eligible_reward_map).sum()
    info["Team/reward_pct_of_max"] = _safe_div(team_reward, max_reward) * done
    info["Team/max_achievable_reward"] = max_reward * done
    info["Team/max_achievable_normal_reward"] = max_normal_reward * done
    team_normal_reward = (team_ach * eligible_normal_reward_map).sum()
    info["Team/normal_reward_pct_of_max"] = _safe_div(team_normal_reward, max_normal_reward) * done
    if static_params.player_count > 1:
        info["Team/max_achievable_coord_reward"] = max_coord_reward * done
        team_coord_reward = (team_ach * eligible_coord_reward_map).sum()
        info["Team/coord_reward_pct_of_max"] = _safe_div(team_coord_reward, max_coord_reward) * done
        team_soft_coord_reward = (team_ach * eligible_soft_coord_reward_map).sum()
        team_sync_hard_coord_reward = (team_ach * eligible_sync_hard_coord_reward_map).sum()
        team_handover_coord_reward = (team_ach * eligible_handover_coord_reward_map).sum()
        team_construction_coord_reward = (team_ach * eligible_construction_coord_reward_map).sum()
        info["Team/soft_coord_reward_pct_of_max"] = (
            _safe_div(team_soft_coord_reward, max_soft_coord_reward) * done
        )
        info["Team/sync_hard_coord_reward_pct_of_max"] = (
            _safe_div(team_sync_hard_coord_reward, max_sync_hard_coord_reward) * done
        )
        info["Team/handover_coord_reward_pct_of_max"] = (
            _safe_div(team_handover_coord_reward, max_handover_coord_reward) * done
        )
        info["Team/construction_coord_reward_pct_of_max"] = (
            _safe_div(team_construction_coord_reward, max_construction_coord_reward) * done
        )

    # Per-achievement metrics (existing behavior)
    for achievement in Achievement:
        achievement_name = f"Achievements/{achievement.name.lower()}"
        info[achievement_name] = achievements[:, achievement.value]

    # Per-agent metrics
    for agent_idx in range(num_agents):
        agent_achievements = achievements[agent_idx]

        # Total achievements for this agent
        total = (agent_achievements * eligible_mask > 0).sum()
        info[f"Agent{agent_idx}/total_achievements"] = total
        info[f"Agent{agent_idx}/achievement_pct"] = _safe_div(total, total_possible)

        # Normal vs Coordination split
        normal_count = (agent_achievements * eligible_normal_mask > 0).sum()
        info[f"Agent{agent_idx}/normal_achievements"] = normal_count
        info[f"Agent{agent_idx}/normal_achievement_pct"] = _safe_div(
            normal_count, total_possible_normal
        )
        if static_params.player_count > 1:
            coord_count = (agent_achievements * eligible_coord_mask > 0).sum()
            soft_coord_count = (agent_achievements * eligible_soft_coord_mask > 0).sum()
            sync_hard_coord_count = (agent_achievements * eligible_sync_hard_coord_mask > 0).sum()
            handover_coord_count = (agent_achievements * eligible_handover_coord_mask > 0).sum()
            construction_coord_count = (
                agent_achievements * eligible_construction_coord_mask > 0
            ).sum()
            info[f"Agent{agent_idx}/coordination_achievements"] = coord_count
            info[f"Agent{agent_idx}/coordination_achievement_pct"] = _safe_div(
                coord_count, total_possible_coord
            )
            info[f"Agent{agent_idx}/soft_coordination_achievements"] = soft_coord_count
            info[f"Agent{agent_idx}/soft_coordination_achievement_pct"] = _safe_div(
                soft_coord_count, total_possible_soft_coord
            )
            info[f"Agent{agent_idx}/sync_hard_coordination_achievements"] = sync_hard_coord_count
            info[f"Agent{agent_idx}/sync_hard_coordination_achievement_pct"] = _safe_div(
                sync_hard_coord_count, total_possible_sync_hard_coord
            )
            info[f"Agent{agent_idx}/handover_coordination_achievements"] = handover_coord_count
            info[f"Agent{agent_idx}/handover_coordination_achievement_pct"] = _safe_div(
                handover_coord_count, total_possible_handover_coord
            )
            info[f"Agent{agent_idx}/construction_coordination_achievements"] = (
                construction_coord_count
            )
            info[f"Agent{agent_idx}/construction_coordination_achievement_pct"] = _safe_div(
                construction_coord_count, total_possible_construction_coord
            )

        # Per-role achievement counts for this agent
        for role_idx, role_name in enumerate(ROLE_NAMES):
            role_mask = ROLE_ACHIEVEMENT_MASK[role_idx]
            role_achievements = (agent_achievements * role_mask > 0).sum()
            info[f"Agent{agent_idx}/{role_name}_achievements"] = role_achievements

    # Team-level: role distribution
    for role_idx, role_name in enumerate(ROLE_NAMES):
        role_mask = ROLE_ACHIEVEMENT_MASK[role_idx]
        team_role_achievements = ((achievements * role_mask[None, :]) > 0).any(axis=0).sum()
        info[f"Team/{role_name}_achievements"] = team_role_achievements

    # Team-level: total
    team_total = ((achievements * eligible_mask[None, :]) > 0).any(axis=0).sum()
    info["Team/total_achievements"] = team_total
    info["Team/achievement_pct"] = _safe_div(team_total, total_possible)

    # Team-level: normal vs coordination
    team_normal = ((achievements * eligible_normal_mask[None, :]) > 0).any(axis=0).sum()
    info["Team/normal_achievements"] = team_normal
    info["Team/normal_achievement_pct"] = _safe_div(team_normal, total_possible_normal)
    if static_params.player_count > 1:
        team_coord = ((achievements * eligible_coord_mask[None, :]) > 0).any(axis=0).sum()
        team_soft_coord = ((achievements * eligible_soft_coord_mask[None, :]) > 0).any(axis=0).sum()
        team_sync_hard_coord = (
            ((achievements * eligible_sync_hard_coord_mask[None, :]) > 0).any(axis=0).sum()
        )
        team_handover_coord = (
            ((achievements * eligible_handover_coord_mask[None, :]) > 0).any(axis=0).sum()
        )
        team_construction_coord = (
            ((achievements * eligible_construction_coord_mask[None, :]) > 0).any(axis=0).sum()
        )
        info["Team/coordination_achievements"] = team_coord
        info["Team/coordination_achievement_pct"] = _safe_div(team_coord, total_possible_coord)
        info["Team/soft_coordination_achievements"] = team_soft_coord
        info["Team/soft_coordination_achievement_pct"] = _safe_div(
            team_soft_coord, total_possible_soft_coord
        )
        info["Team/sync_hard_coordination_achievements"] = team_sync_hard_coord
        info["Team/sync_hard_coordination_achievement_pct"] = _safe_div(
            team_sync_hard_coord, total_possible_sync_hard_coord
        )
        info["Team/handover_coordination_achievements"] = team_handover_coord
        info["Team/handover_coordination_achievement_pct"] = _safe_div(
            team_handover_coord, total_possible_handover_coord
        )
        info["Team/construction_coordination_achievements"] = team_construction_coord
        info["Team/construction_coordination_achievement_pct"] = _safe_div(
            team_construction_coord, total_possible_construction_coord
        )
    if static_params.player_count > 1:
        # Event-level bonus activation rate
        info["Team/soft_coordination_bonus_rate"] = (
            jnp.where(
                state.soft_sync_events > 0,
                state.soft_sync_bonus_events / state.soft_sync_events,
                0.0,
            )
            * done
        )

    # Specialization alignment metrics — only meaningful with multiple distinct roles
    if static_params.player_count > 1 and hasattr(state, "player_specialization"):
        specs = state.player_specialization

        for agent_idx in range(num_agents):
            agent_spec = specs[agent_idx]
            agent_achievements = achievements[agent_idx]

            # Count achievements aligned with agent's assigned role
            role_mask = ROLE_ACHIEVEMENT_MASK[agent_spec]
            aligned_count = (agent_achievements * role_mask > 0).sum()

            # Count achievements from other specialist roles (cross-role activity)
            # Exclude shared (0) achievements - only count "stealing" from other specialists
            other_specialist_mask = jnp.zeros_like(role_mask)
            for other_role in [1, 2, 3]:
                other_specialist_mask = jnp.where(
                    other_role != agent_spec,
                    other_specialist_mask + ROLE_ACHIEVEMENT_MASK[other_role],
                    other_specialist_mask,
                )
            cross_role_count = (agent_achievements * other_specialist_mask > 0).sum()

            info[f"Agent{agent_idx}/aligned_achievements"] = aligned_count
            info[f"Agent{agent_idx}/cross_role_achievements"] = cross_role_count

        # Team-level specialization metrics
        total_aligned = jnp.array(0.0)
        total_cross = jnp.array(0.0)
        for agent_idx in range(num_agents):
            total_aligned = total_aligned + info[f"Agent{agent_idx}/aligned_achievements"]
            total_cross = total_cross + info[f"Agent{agent_idx}/cross_role_achievements"]

        info["Team/total_aligned_achievements"] = total_aligned
        info["Team/total_cross_role_achievements"] = total_cross

        # Specialization ratio: aligned / (aligned + cross) - measures role adherence
        # Higher = agents stick to their roles, Lower = agents do others' jobs
        total_role_specific = total_aligned + total_cross
        info["Team/specialization_ratio"] = jnp.where(
            total_role_specific > 0, total_aligned / total_role_specific, 0.0
        )

    if static_params.player_count > 1:
        # Cooperation behavior metrics — trades, requests, revives require multiple agents.
        resource_trade_per_agent = (
            state.trade_count - state.food_trade_count - state.drink_trade_count
        )
        actionable_agent_steps = state.actionable_agent_steps
        alive_agent_steps = state.alive_agent_steps

        def _per_100_actionable(count: Array) -> Float[Array, ""]:
            return jnp.where(
                actionable_agent_steps > 0,
                100.0 * count / actionable_agent_steps,
                0.0,
            )

        for agent_idx in range(num_agents):
            info[f"Agent{agent_idx}/trade_count"] = state.trade_count[agent_idx] * done
            info[f"Agent{agent_idx}/food_trade_count"] = state.food_trade_count[agent_idx] * done
            info[f"Agent{agent_idx}/drink_trade_count"] = state.drink_trade_count[agent_idx] * done
            info[f"Agent{agent_idx}/resource_trade_count"] = (
                resource_trade_per_agent[agent_idx] * done
            )
            info[f"Agent{agent_idx}/give_attempt_count"] = (
                state.give_attempt_count[agent_idx] * done
            )
            info[f"Agent{agent_idx}/give_success_rate"] = (
                jnp.where(
                    state.give_attempt_count[agent_idx] > 0,
                    state.trade_count[agent_idx] / state.give_attempt_count[agent_idx],
                    0.0,
                )
                * done
            )
            info[f"Agent{agent_idx}/request_count"] = state.request_count[agent_idx] * done
            info[f"Agent{agent_idx}/request_expiry_count"] = (
                state.request_expiry_count[agent_idx] * done
            )
            info[f"Agent{agent_idx}/request_received_count"] = (
                state.request_received_count[agent_idx] * done
            )
            info[f"Agent{agent_idx}/ff_damage_dealt"] = state.ff_damage_dealt[agent_idx] * done
        info["Cooperation/trade_count"] = state.trade_count.sum() * done
        info["Cooperation/food_trade_count"] = state.food_trade_count.sum() * done
        info["Cooperation/drink_trade_count"] = state.drink_trade_count.sum() * done
        info["Cooperation/resource_trade_count"] = resource_trade_per_agent.sum() * done
        info["Cooperation/give_attempt_count"] = state.give_attempt_count.sum() * done
        info["Cooperation/give_success_rate"] = (
            jnp.where(
                state.give_attempt_count.sum() > 0,
                state.trade_count.sum() / state.give_attempt_count.sum(),
                0.0,
            )
            * done
        )
        info["Cooperation/request_count"] = state.request_count.sum() * done
        info["Cooperation/request_expiry_count"] = state.request_expiry_count.sum() * done
        info["Cooperation/request_received_count"] = state.request_received_count.sum() * done
        info["Cooperation/request_open_count"] = (state.request_duration > 0).sum() * done
        info["Cooperation/revives"] = state.revives * done
        info["Cooperation/ff_damage_dealt"] = state.ff_damage_dealt.sum() * done
        info["Cooperation/alive_agent_steps"] = alive_agent_steps * done
        info["Cooperation/actionable_agent_steps"] = actionable_agent_steps * done
        info["Cooperation/trades_per_100_actionable_agent_steps"] = (
            _per_100_actionable(state.trade_count.sum()) * done
        )
        info["Cooperation/give_attempts_per_100_actionable_agent_steps"] = (
            _per_100_actionable(state.give_attempt_count.sum()) * done
        )
        info["Cooperation/requests_per_100_actionable_agent_steps"] = (
            _per_100_actionable(state.request_count.sum()) * done
        )
        info["Cooperation/revives_per_100_actionable_agent_steps"] = (
            _per_100_actionable(state.revives) * done
        )

        # Communication metrics (only meaningful when num_comm_channels > 0)
        if static_params.num_comm_channels > 0:
            for agent_idx in range(num_agents):
                info[f"Agent{agent_idx}/comm_count"] = state.comm_count[agent_idx] * done
            info["Communication/total_comm_count"] = state.comm_count.sum() * done
            info["Communication/comms_per_100_actionable_agent_steps"] = (
                _per_100_actionable(state.comm_count.sum()) * done
            )

    # Coordination event metrics — skip for single agent
    if static_params.player_count > 1 and hasattr(state, "coord_mine_sync_soft_count"):
        # Mining / construction handovers share one pending array. Split them by
        # the stored building_type so totals and rates can stay explicit.
        mining_handover_setups = state.handover_setups - state.coord_construction_handover_setups
        mining_handover_successes = (
            state.handover_successes - state.coord_construction_handover_count
        )
        mining_handover_expiries = state.coord_mine_handover_expiries
        construction_sync_attempts = state.coord_construction_attempts
        construction_sync_successes = state.coord_construction_successes
        construction_handover_attempts = state.coord_construction_handover_setups
        construction_handover_successes = state.coord_construction_handover_count
        construction_handover_expiries = state.coord_construction_handover_expiries
        construction_total_attempts = construction_sync_attempts + construction_handover_attempts
        construction_total_successes = construction_sync_successes + construction_handover_successes
        active_handover_mask = state.pending_handovers[:, 0] == 1
        construction_handover_open = (
            active_handover_mask & (state.pending_handovers[:, 5] > 0)
        ).sum()
        mining_handover_open = (active_handover_mask & (state.pending_handovers[:, 5] == 0)).sum()
        total_handover_open = construction_handover_open + mining_handover_open

        resolved_mining_handover_attempts = mining_handover_successes + mining_handover_expiries
        resolved_construction_handover_attempts = (
            construction_handover_successes + construction_handover_expiries
        )
        resolved_total_handover_attempts = state.handover_successes + state.handover_expiries
        construction_total_resolved_attempts = (
            construction_sync_attempts + resolved_construction_handover_attempts
        )

        info["Coordination/mine_sync_soft"] = state.coord_mine_sync_soft_count * done
        info["Coordination/mine_sync_hard"] = state.coord_mine_sync_hard_count * done
        info["Coordination/mine_handover"] = state.coord_mine_handover_count * done
        info["Coordination/mining_handover_setups"] = mining_handover_setups * done
        info["Coordination/mining_handover_successes"] = mining_handover_successes * done
        info["Coordination/mining_handover_expiries"] = mining_handover_expiries * done
        info["Coordination/mining_handover_open"] = mining_handover_open * done
        info["Coordination/mining_handover_resolved_attempts"] = (
            resolved_mining_handover_attempts * done
        )
        info["Coordination/mining_handover_success_rate"] = (
            jnp.where(
                resolved_mining_handover_attempts > 0,
                mining_handover_successes / resolved_mining_handover_attempts,
                0.0,
            )
            * done
        )
        # Total handover metrics across mining + construction.
        info["Coordination/handover_successes"] = state.handover_successes * done
        info["Coordination/handover_setups"] = state.handover_setups * done
        info["Coordination/handover_expiries"] = state.handover_expiries * done
        info["Coordination/handover_open"] = total_handover_open * done
        info["Coordination/total_handover_successes"] = state.handover_successes * done
        info["Coordination/total_handover_setups"] = state.handover_setups * done
        info["Coordination/total_handover_expiries"] = state.handover_expiries * done
        info["Coordination/total_handover_resolved_attempts"] = (
            resolved_total_handover_attempts * done
        )
        total_sync = state.coord_mine_sync_soft_count + state.coord_mine_sync_hard_count
        info["Coordination/total_sync"] = total_sync * done

        # Coordination overview counts blend domains for failure analysis. Rates use
        # only resolved handovers (successes + expiries) so unresolved end-of-episode
        # handovers do not silently bias the denominator.
        all_coord_attempts = (
            state.coord_sync_attempts  # mining sync (.any per timestep)
            + state.handover_setups  # ALL handovers: mining + construction (per setup)
            + state.coord_construction_attempts  # construction SYNC only (.any per timestep)
            + state.coord_elite_attempts  # combat (.any per timestep)
            + state.coord_craft_attempts  # crafting (.any per timestep)
        )
        all_resolved_coord_attempts = (
            state.coord_sync_attempts
            + resolved_total_handover_attempts
            + state.coord_construction_attempts
            + state.coord_elite_attempts
            + state.coord_craft_attempts
        )
        all_coord_successes = (
            state.coord_sync_successes  # mining sync (.any per timestep)
            + state.handover_successes  # ALL handovers: mining + construction (per-slot)
            + state.coord_construction_successes  # construction SYNC only (.any per timestep)
            + state.coord_elite_successes  # combat (.any per timestep)
            + state.coord_craft_successes  # crafting (.any per timestep)
        )
        info["Coordination/sync_attempts"] = state.coord_sync_attempts * done
        info["Coordination/sync_successes"] = state.coord_sync_successes * done
        info["Coordination/coord_sync_success_rate"] = (
            jnp.where(
                state.coord_sync_attempts > 0,
                state.coord_sync_successes / state.coord_sync_attempts,
                0.0,
            )
            * done
        )

        info["Coordination/handover_success_rate"] = (
            jnp.where(
                resolved_total_handover_attempts > 0,
                state.handover_successes / resolved_total_handover_attempts,
                0.0,
            )
            * done
        )

        info["Coordination/construction_sync_success_rate"] = (
            jnp.where(
                construction_sync_attempts > 0,
                construction_sync_successes / construction_sync_attempts,
                0.0,
            )
            * done
        )
        # Keep the blended construction rate, but use resolved attempts so open
        # handovers do not count as silent failures at episode end.
        info["Coordination/construction_success_rate"] = (
            jnp.where(
                construction_total_resolved_attempts > 0,
                construction_total_successes / construction_total_resolved_attempts,
                0.0,
            )
            * done
        )
        info["Coordination/elite_coord_success_rate"] = (
            jnp.where(
                state.coord_elite_attempts > 0,
                state.coord_elite_successes / state.coord_elite_attempts,
                0.0,
            )
            * done
        )
        info["Coordination/craft_coord_success_rate"] = (
            jnp.where(
                state.coord_craft_attempts > 0,
                state.coord_craft_successes / state.coord_craft_attempts,
                0.0,
            )
            * done
        )
        info["Coordination/total_attempts"] = all_coord_attempts * done
        info["Coordination/total_successes"] = all_coord_successes * done
        info["Coordination/total_resolved_attempts"] = all_resolved_coord_attempts * done
        info["Coordination/coordination_success_rate"] = (
            jnp.where(
                all_resolved_coord_attempts > 0,
                all_coord_successes / all_resolved_coord_attempts,
                0.0,
            )
            * done
        )

        # Solo soft coordination (agent on soft block without enough agents)
        info["Coordination/solo_soft_attempts"] = state.coord_solo_soft_attempts * done
        info["Coordination/solo_soft_successes"] = state.coord_solo_soft_successes * done
        info["Coordination/solo_soft_success_rate"] = (
            jnp.where(
                state.coord_solo_soft_attempts > 0,
                state.coord_solo_soft_successes / state.coord_solo_soft_attempts,
                0.0,
            )
            * done
        )

        # Sync coordination by agent count
        info["Coordination/sync_2_agent"] = state.sync_coord_by_agents[0] * done
        info["Coordination/sync_3plus_agent"] = state.sync_coord_by_agents[1] * done

        # Construction coordination
        # Totals keep sync + handover together for high-level failure analysis.
        info["Coordination/construction_attempts"] = construction_total_attempts * done
        info["Coordination/construction_successes"] = construction_total_successes * done
        info["Coordination/construction_sync_attempts"] = construction_sync_attempts * done
        info["Coordination/construction_sync_successes"] = construction_sync_successes * done
        info["Coordination/construction_handover"] = construction_handover_successes * done
        info["Coordination/construction_handover_setups"] = construction_handover_attempts * done
        info["Coordination/construction_handover_attempts"] = construction_handover_attempts * done
        info["Coordination/construction_handover_successes"] = (
            construction_handover_successes * done
        )
        info["Coordination/construction_handover_expiries"] = construction_handover_expiries * done
        info["Coordination/construction_handover_open"] = construction_handover_open * done
        info["Coordination/construction_handover_resolved_attempts"] = (
            resolved_construction_handover_attempts * done
        )
        # Diagnostic: BUILD_X at a CONSTRUCTION_SITE with no materials on tile.
        # Funded attempts are already captured by construction_attempts.
        info["Coordination/construction_build_at_site_unfunded"] = (
            state.construction_build_at_site_unfunded * done
        )
        info["Coordination/construction_total_attempts"] = construction_total_attempts * done
        info["Coordination/construction_total_successes"] = construction_total_successes * done
        info["Coordination/construction_total_resolved_attempts"] = (
            construction_total_resolved_attempts * done
        )
        # Alias kept for backwards compatibility with older analysis scripts.
        info["Coordination/construction_total_success_rate"] = (
            jnp.where(
                construction_total_resolved_attempts > 0,
                construction_total_successes / construction_total_resolved_attempts,
                0.0,
            )
            * done
        )
        info["Coordination/construction_handover_success_rate"] = (
            jnp.where(
                resolved_construction_handover_attempts > 0,
                construction_handover_successes / resolved_construction_handover_attempts,
                0.0,
            )
            * done
        )
        info["Coordination/build_shelter"] = state.coord_build_shelter_count * done
        info["Coordination/build_forge"] = state.coord_build_forge_count * done
        info["Coordination/build_beacon"] = state.coord_build_beacon_count * done
        info["Coordination/total_builds"] = (
            state.coord_build_shelter_count
            + state.coord_build_forge_count
            + state.coord_build_beacon_count
        ) * done

        # Combat coordination
        info["Coordination/elite_attempts"] = state.coord_elite_attempts * done
        info["Coordination/elite_successes"] = state.coord_elite_successes * done
        info["Coordination/elite_melee_kills"] = state.coord_elite_melee_kills * done
        info["Coordination/elite_ranged_kills"] = state.coord_elite_ranged_kills * done
        info["Coordination/large_passive_kills"] = state.coord_large_passive_kills * done
        info["Coordination/total_elite_kills"] = (
            state.coord_elite_melee_kills
            + state.coord_elite_ranged_kills
            + state.coord_large_passive_kills
        ) * done

        # Diamond crafting coordination
        info["Coordination/craft_attempts"] = state.coord_craft_attempts * done
        info["Coordination/craft_successes"] = state.coord_craft_successes * done
        info["Coordination/diamond_pickaxe_crafted"] = state.coord_diamond_pickaxe_count * done
        info["Coordination/diamond_sword_crafted"] = state.coord_diamond_sword_count * done
        info["Coordination/diamond_armour_crafted"] = state.coord_diamond_armour_count * done

    # Sampled alpha (for randomized difficulty training).
    # Always logged for schema stability across configs; -1 signals "not tracked
    # in this state" (e.g. legacy states without the field) so downstream plots
    # can filter without the column silently disappearing.
    if hasattr(state, "sampled_alpha"):
        info["Coordination/sampled_alpha"] = state.sampled_alpha * done
    else:
        info["Coordination/sampled_alpha"] = jnp.asarray(-1.0, dtype=jnp.float32)

    # Progression metrics
    # Subtract the initial offset (level 0 starts at 10 to open the first ladder)
    _monsters_killed = state.monsters_killed.at[0].add(-10)
    info["Progression/total_monsters_killed"] = _monsters_killed.sum() * done
    info["Progression/current_level_monsters_killed"] = _monsters_killed[state.player_level] * done
    info["Progression/player_level"] = state.player_level * done
    info["Progression/max_level_reached"] = state.max_player_level * done

    # Per-agent attributes (XP, stats)
    _attr_names = ["xp", "dexterity", "strength", "intelligence"]
    _attr_arrays = [
        state.player_xp,
        state.player_dexterity,
        state.player_strength,
        state.player_intelligence,
    ]
    for attr_name, attr_arr in zip(_attr_names, _attr_arrays):
        for agent_idx in range(num_agents):
            info[f"Agent{agent_idx}/{attr_name}"] = attr_arr[agent_idx] * done
        info[f"Team/mean_{attr_name}"] = attr_arr.mean() * done

    # Per-agent inventory snapshot at episode end
    # Some fields are multi-dimensional (armour: 4 slots, potions: 6 slots) — sum to scalar.
    _inv_fields = [
        "wood",
        "stone",
        "coal",
        "iron",
        "diamond",
        "sapling",
        "pickaxe",
        "sword",
        "bow",
        "arrows",
        "armour",
        "torches",
        "ruby",
        "sapphire",
    ]
    for field in _inv_fields:
        arr = getattr(state.inventory, field)
        for agent_idx in range(num_agents):
            val = arr[agent_idx]
            info[f"Agent{agent_idx}/inventory_{field}"] = (
                val.sum() if val.ndim > 0 else val
            ) * done
        # Mean across agents (flatten multi-slot first)
        per_agent_totals = arr.reshape(num_agents, -1).sum(axis=1)  # (num_agents,)
        info[f"Team/mean_inventory_{field}"] = per_agent_totals.mean() * done

    # Death cause breakdown — all gated on `done` to avoid leaking mid-episode
    # transitions (e.g. state before a revive) into time-averaged logs.
    _death_causes = ["starvation", "dehydration", "exhaustion", "mob_combat", "friendly_fire"]
    for agent_idx in range(num_agents):
        _cause = state.player_death_cause[agent_idx]
        info[f"Agent{agent_idx}/death_cause"] = _cause * done
        info[f"Agent{agent_idx}/level_at_death"] = state.player_level_at_death[agent_idx] * done
        for _i, _name in enumerate(_death_causes, start=1):
            info[f"Agent{agent_idx}/died_from_{_name}"] = (_cause == _i).astype(jnp.float32) * done
    # Team-level totals
    for _i, _name in enumerate(_death_causes, start=1):
        info[f"Deaths/{_name}_count"] = (state.player_death_cause == _i).sum() * done
    info["Deaths/total_deaths"] = (state.player_death_cause > 0).sum() * done

    # Broadcast all scalar metrics to (num_agents,) for consistent batching
    info = jax.tree.map(lambda x: jnp.broadcast_to(x, (num_agents,)) if x.ndim == 0 else x, info)

    return info
