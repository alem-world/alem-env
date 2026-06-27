"""
Single-agent language wrapper for Alem-Coop run as a 1-player env.

Thin subclass of AlemLanguageWrapper. Differences from the multi-agent version
(everything not listed below is inherited unchanged):

1. Action list (SINGLE_AGENT_ACTIONS):
   - Drops the 9 Request actions (Request Food/Drink/Wood/Stone/Iron/Coal/
     Diamond/Ruby/Sapphire) and the Give action — they only make sense with
     teammates.
   - 45 actions total, vs. 55 in the multi-agent ACTIONS list.

2. Action descriptions (SINGLE_AGENT_ACTION_DICT):
   - "Do" rewritten to drop the teammate/sync-coordination clauses.
   - "Build Shelter" drops the "(all agents)" suffix.

3. Prompt (get_instruction_prompt_single):
   - Intro: "You are a solo agent..." instead of "You are an agent in a
     N-agent cooperative survival game...".
   - End-of-game phrasing: "if your health reaches zero the game ends"
     (was: "if all agents die the game ends").
   - Drops the entire Coordination section (Sync, Handover, Construction
     coordination, Elite mobs, Revive, Epic forge).
   - Drops teammate-only mechanics from "How to play": Request/Give bullet,
     "another player" in Movement, teammate clauses in Do.
   - Resource chain / crafting recipes: drops the "+ enough agents crafting"
     coordination notes on diamond gear.
   - Progression: drops "Only one agent needs to use Descend/Ascend — all
     teammates are teleported with them" and the "coordinating with teammates"
     note on diamond crafting.
   - Crafting: Build Shelter description drops "(all agents)".
   - Achievement list filtered to non-coordination achievements only
     (CORE_ACHIEVEMENTS, never appends COORD_ACHIEVEMENTS).
   - prompt_mode="specific_collaborative" is treated as "specific" — there is
     no collaborative mode for solo play.

4. Per-turn observations (AlemLanguageWrapperSingle):
   - describe_teammates() always returns "" (the parent already iterates
     range(num_agents) skipping self, but the override avoids the loop and
     makes the intent explicit).
   - describe_status() says "ends early if you die" instead of "if all agents
     die".
   - describe_frame() rewrites the dead-state line from "Status: dead (you
     cannot act; you can only use communication to coordinate with teammates)"
     to "Status: dead (game over)".
   - __init__ coerces prompt_mode "specific_collaborative" → "specific" so
     that describe_mobs() never tags elite mobs with the "fight alongside
     teammates" suffix (those tags are gated on
     self.prompt_mode == "specific_collaborative").

5. Things intentionally NOT changed:
   - step()/reset()/process_obs(): the parent already iterates over
     range(self.num_agents); for a 1-player env that's just one agent.
   - Action parsing (get_action_index): the GIVE-slot branch is dead code
     when no Give action is ever issued, so no override needed.
   - Coordination cues in observations (describe_coordination_cues) and
     construction coord cues (describe_construction): already gated on
     self.env_params.coordination_enabled, which is False when
     coordination_difficulty="none" (the single-agent setting).
"""

import numpy as np

from alem.llm.alem_language_wrapper import (
    ACTION_DICT,
    ACTIONS,
    BEACON_COST_COAL,
    BEACON_COST_IRON,
    CORE_ACHIEVEMENTS,
    FORGE_COST_COAL,
    FORGE_COST_IRON,
    FORGE_COST_STONE,
    LEVEL_NAMES,
    SHELTER_COST_STONE,
    SHELTER_COST_WOOD,
    AlemLanguageWrapper,
    _achievements_for_level,
)

# Actions that make no sense without teammates — omit from the single-agent prompt.
_MULTI_AGENT_ONLY_ACTIONS = frozenset(
    {
        "Request Food",
        "Request Drink",
        "Request Wood",
        "Request Stone",
        "Request Iron",
        "Request Coal",
        "Request Diamond",
        "Request Ruby",
        "Request Sapphire",
        "Give",
    }
)

SINGLE_AGENT_ACTIONS = [a for a in ACTIONS if a not in _MULTI_AGENT_ONLY_ACTIONS]

# Override entries in ACTION_DICT whose descriptions reference teammates or
# multi-agent coordination. All other entries are inherited from ACTION_DICT.
_SINGLE_AGENT_OVERRIDES = {
    "Do": (
        "interact with the tile you are facing — chop trees, mine resources "
        "(requires a matching pickaxe tier: stone/coal needs wood pickaxe, "
        "iron needs stone pickaxe, diamond needs iron pickaxe, ruby/sapphire "
        "need diamond pickaxe), attack creatures, drink from water tiles, or "
        "open chests."
    ),
    "Build Shelter": (
        f"build an epic shelter at a construction site "
        f"(needs {SHELTER_COST_WOOD} wood + {SHELTER_COST_STONE} stone; "
        f"effect: +50% energy regeneration while resting)"
    ),
}
SINGLE_AGENT_ACTION_DICT = {**ACTION_DICT, **_SINGLE_AGENT_OVERRIDES}


def get_instruction_prompt_single(
    llm_mode=None,
    include_all_actions=True,
    progressive_disclosure=False,
    current_level=0,
    prompt_mode=None,
    agent_id=None,
    role=None,
):
    """System-prompt for a single LLM agent playing solo.

    API mirrors get_instruction_prompt() but removes all teammate/coordination
    content and strips Request/Give from the action list.

    Args:
        llm_mode: Ignored; kept for call-site compatibility.
        include_all_actions: If True, include the full <all_actions> block.
        progressive_disclosure: Gate late-game sections by current_level.
        current_level: Dungeon level (0 = overworld).
        prompt_mode: "general" | "specific" | "specific_collaborative".
            "specific_collaborative" is treated as "specific" for solo play.
        agent_id: Ignored; kept for call-site compatibility.
        role: Role name shown in the intro if provided.

    Returns:
        Coordination-free system-prompt text for solo play.
    """
    valid_prompt_modes = {"general", "specific", "specific_collaborative"}
    if prompt_mode is None:
        prompt_mode = "specific"
    if prompt_mode not in valid_prompt_modes:
        raise ValueError(
            f"Invalid prompt_mode '{prompt_mode}'. Expected one of: {sorted(valid_prompt_modes)}"
        )

    lvl = current_level if progressive_disclosure else 99

    if progressive_disclosure:
        achievement_str = "\n".join(
            _achievements_for_level(current_level, coordination_enabled=False)
        )
    else:
        achievement_str = "\n".join(CORE_ACHIEVEMENTS)

    action_strings = "\n".join(
        f"{action}: {SINGLE_AGENT_ACTION_DICT[action]}" for action in SINGLE_AGENT_ACTIONS
    )

    role_clause = f" ({role})" if role else ""
    intro = (
        f"You are a solo agent{role_clause} in a survival game. "
        "Your goal is to gather resources, craft gear, fight monsters, and descend "
        "through 9 dungeon levels. You must survive — if your health reaches zero the "
        "game ends. Maximize the number of achievements while staying alive."
    )

    if prompt_mode == "general":
        return f"""{intro}

## Actions
Each turn, choose exactly one action. Your observation will list which actions are currently available.

{action_strings}

## Achievements
{achievement_str}

Choose actions to maximize achievements while staying alive. Your observations show what you see, your inventory, and available actions."""

    # --- "specific" / "specific_collaborative" both use full rules, no coord ---
    core_mechanics = """
## How to play
- Each turn, choose exactly one action.
- **Movement** uses absolute directions: north, south, east, and west. Any move attempt changes your facing to that direction, even if the move is blocked and you stay in place. A move is blocked if the target tile is solid, including trees, stone, ore veins, walls, crafting stations, chests, and plants, or if it contains water, lava, or a mob. If repeated move attempts do not change your position, that direction is blocked. You can also use a blocked move to turn in place, for example to face an adjacent tree.
- **Facing**: your facing direction is set by your last movement action and persists until you move again. **Do** always targets the tile in your current facing direction.
- **Do** is your main interaction: face a tile and use the **Do** action on exactly that tile to chop trees, mine ore, attack creatures, drink water, or open chests.
- **Crafting**: stand next to (including diagonally) the required station and use the craft action; you do NOT need to face it. Diamond items always require an adjacent epic forge, not a table.
- **Placing**: face the target tile, then use the place action. Tables and furnaces need an empty non-solid tile that is not water or lava; stone can also be placed into water (costs 1 stone). Place Plant puts a sapling on the faced tile. Place Torch lights dark areas.
- **Ranged combat**: use Shoot Arrow while facing a creature (requires a bow + arrows). Bows are found in dungeon chests.
- **Elite mobs** are tougher and deal more damage.
"""

    survival_stats = """
## Survival stats
Food, drink, and energy deplete gradually over time — roughly every 20-30 steps you lose 1 point of each (dexterity slows this rate). When food or drink reaches 0, your health starts dropping. When energy reaches 0, you automatically fall asleep and cannot act until energy is full. While sleeping, you take 2.5x damage from all sources. Mana does NOT decay — it is only spent by casting spells or enchanting. Mana slowly regenerates over time (faster while sleeping).
- **Sleep**: choose this voluntarily to recover energy at 2x the passive rate. Ends automatically when energy is full.
- **Rest**: choose this to recover health gradually. Requires food, drink, and energy all > 0; ends when health is full or a stat runs out."""

    resource_chain = """
## Resource chain
Trees → wood (no tool required) → Stone/Coal (needs wood pickaxe) → Iron (needs stone pickaxe) → Diamond (iron pickaxe) → Ruby/Sapphire (diamond pickaxe)"""

    crafting_recipes = f"""
## Crafting recipes
All recipes consume the listed materials.
Stations: Table (2 wood), Furnace (1 stone)
- Wood pickaxe/sword: table + 1 wood
- Stone pickaxe/sword: table + 1 wood + 1 stone
- Iron pickaxe/sword: table + furnace + 1 wood + 1 stone + 1 iron + 1 coal
- Iron armour: table + furnace + 3 iron + 3 coal
- Diamond pickaxe: epic forge + 1 wood + 3 diamond
- Diamond sword: epic forge + 1 wood + 2 diamond
- Diamond armour: epic forge + 3 diamond
- Arrows: table + 1 wood + 1 stone (yields 2)
- Torch: table + 1 wood + 1 coal (yields 4)

Construction (at a construction site, face it and use Build action):
- Build Shelter: needs {SHELTER_COST_WOOD} wood + {SHELTER_COST_STONE} stone. Shelters result in +50% energy regeneration while resting.
- Build Forge: needs {FORGE_COST_STONE} stone + {FORGE_COST_IRON} iron + {FORGE_COST_COAL} coal. Creates an epic forge, which enables diamond gear crafting.
- Build Beacon: needs {BEACON_COST_IRON} iron + {BEACON_COST_COAL} coal. Expands the lit area on this level."""

    chests = ""
    if lvl >= 1:
        chests = """
## Chests
Open chests with Do while facing them. Loot is role-biased and partially random:
- Any role (50% chance): 1-2 potions of a random color
- Miner (60% chance): 1-5 wood, 4-7 torches, or 1-3 ores (coal/iron/diamond/sapphire/ruby)
- Miner (20% chance): pickaxe upgrade to a random higher tier
- Warrior (50% chance): 4-8 arrows
- Warrior only, first chest on dungeon level 1: bow
- Any role, first chest on level 3 or 4 only: spell book"""

    spells = ""
    if lvl >= 3:
        spells = """
## Spells
Learn spells by using Read Book when you have a book in your inventory. Each role learns a different spell:
- Fireball (miner, warrior): costs 2 mana; fires a projectile at the tile you are facing.
- Heal (forager): costs 6 mana; restores +2 health to yourself.
Use Cast Spell while facing the desired target tile."""

    potions = ""
    if lvl >= 1:
        potions = """
## Potions
Each potion action drinks one potion of that color, if you have one. The color-to-effect mapping is randomized at the start of each episode and fixed for its duration — the same color always has the same effect within one game.
- Possible effects: +8 health, -3 health, +8 mana, -3 mana, +8 energy, or -3 energy.
- Potion colors: red, green, blue, pink, cyan, yellow.
- The mapping is not shown to you directly. Observe the changes after each drink to learn which colors are safe."""

    boss = ""
    if lvl >= 5:
        boss = """
## Boss
The Necromancer on the final level is only vulnerable during specific windows (shown in your observation). Attack during vulnerable phases and survive the spawn waves between them."""

    enchantments = ""
    if lvl >= 5:
        enchantments = """
## Enchantments
Enchant Sword and Enchant Bow are warrior-restricted (not guaranteed for non-warriors). Enchant Armour works for all roles. Fire tables use ruby, ice tables use sapphire, and every enchantment costs 1 matching gem + 9 mana.
- Sword enchant: enchants your sword with the table's element (fire or ice).
- Bow enchant: enchants your bow with the table's element (fire or ice).
- Armour enchant: enchants one armour piece with the table's element; it targets an unenchanted piece first, otherwise it can replace a piece with the opposite element."""

    attributes = """
## Attributes
Gain 1 XP each time you descend to a new floor. Spend XP with Level Up actions.
- **Strength**: max health = 8 + strength
- **Dexterity**: max food = 7 + 2*dexterity (+2 extra for foragers); max drink = same; max energy = 7 + 2*dexterity
- **Intelligence**: max mana = 6 + 3*intelligence; enchantment damage +5% per point above 1."""

    progression_steps = [
        "1. Gather wood → place a table → craft a wood pickaxe; craft a wood sword early if combat is likely.",
        "2. Mine stone and coal → place a furnace → craft iron tools and iron armour.",
        "3. To descend: stand on the `ladder_down` tile (visible in your observation when close) and use the Descend action. The ladder only becomes usable after enough monsters on that level have been killed.",
    ]
    if lvl >= 1:
        progression_steps.append(
            "4. In dungeons, open chests for loot such as bows, potions, and spell books, and mine gems as your tools improve."
        )
    if lvl >= 3:
        progression_steps.append(
            "5. Enchant weapons, bows, or armour at enchantment tables by spending 9 mana and 1 gem; ruby gives fire, sapphire gives ice."
        )
    if lvl >= 5:
        progression_steps.append("6. Craft diamond gear at an epic forge.")
        progression_steps.append("7. Repeat across all dungeon levels until the final boss.")

    progression = "\n## Progression\n" + "\n".join(progression_steps)

    roles = """
## Roles
Role-restricted actions succeed with reduced probability for non-specialists. Depending on the difficulty configuration, non-specialist success rates are 10%, 40%, or 70%. Specialist success rate is always 100%.

- **Forager**: collecting water, saplings, eating passive mobs (e.g. cows/bats/snails). Also has 3x base food and drink capacity.
- **Miner**: crafting pickaxes/torches, placing stone.
- **Warrior**: crafting swords and arrows. Also deals 2x melee damage, and specializes in enchanting swords and bows.
- No role restriction: Place Table, Place Furnace, Wood Sword, Iron Armour, Diamond Armour"""

    actions_section = ""
    if include_all_actions:
        actions_section = f"""
<all_actions>
## Actions
Each turn, choose exactly one action. Your observation will list which actions are currently available.

{action_strings}
</all_actions>"""

    achievements_section = f"""
<achievements>
## Achievements
{achievement_str}
</achievements>"""

    late_game_parts = [s for s in [chests, spells, potions, enchantments, boss] if s]
    late_game_section = ""
    if late_game_parts:
        late_game_section = "\n<late_game>" + "".join(late_game_parts) + "\n</late_game>"

    return f"""{intro}
<game_rules>
{core_mechanics}
{survival_stats}
{roles}

{resource_chain}
{crafting_recipes}
{attributes}
{progression}
</game_rules>
{actions_section}
{achievements_section}
{late_game_section}"""


class AlemLanguageWrapperSingle(AlemLanguageWrapper):
    """Single-agent variant of AlemLanguageWrapper.

    See module docstring for the full diff vs. AlemLanguageWrapper. In short:
    no teammate/coordination text in observations, "if you die" instead of
    "if all agents die", and the system prompt routes through
    get_instruction_prompt_single().
    """

    def __init__(self, *args, **kwargs):
        """Initialize the base wrapper and disable collaborative prompt behavior.

        Args:
            *args: Positional arguments forwarded to ``AlemLanguageWrapper``.
            **kwargs: Keyword arguments forwarded to ``AlemLanguageWrapper``.
        """
        super().__init__(*args, **kwargs)
        # describe_mobs() gates elite-coord tags ("fight alongside teammates")
        # on self.prompt_mode == "specific_collaborative". For solo play, demote
        # that mode to "specific" so the tags never fire.
        if self.prompt_mode == "specific_collaborative":
            self.prompt_mode = "specific"

    def describe_teammates(self, state, player_idx):
        """Return no teammate section for the single-agent environment.

        Args:
            state: Current environment state, unused in single-agent mode.
            player_idx: Player index, unused in single-agent mode.

        Returns:
            An empty string.
        """
        return ""

    def describe_status(self, state, player_idx):
        """Describe single-agent time, condition, role, level, and progress.

        Args:
            state: Current environment state.
            player_idx: Player whose status should be described.

        Returns:
            Multi-line status text for the selected player.
        """
        lines = []

        timestep = int(state.timestep)
        max_timesteps = int(self.env_params.max_timesteps)
        lines.append(
            f"Step: {timestep}/{max_timesteps} ({max_timesteps - timestep} remaining, ends early if you die)"
        )
        if self.exact_coordinates:
            lines.append(f"Position: {self._xy_coord_str(state.player_position[player_idx])}")

        if state.is_sleeping[player_idx]:
            lines.append("Status: sleeping (cannot act until energy is full)")
        elif state.player_health[player_idx] <= 0:
            lines.append("Status: dead")
        elif state.is_resting[player_idx]:
            lines.append("Status: resting (recovering health and mana)")

        spec = int(state.player_specialization[player_idx])
        if spec > 0:
            spec_name = self.spec_names.get(spec, "unknown")
            lines.append(f"Role: {spec_name}")

        level = int(state.player_level)
        level_name = LEVEL_NAMES.get(level)
        if level == 0:
            lines.append(f"Location: {level_name} (surface)")
        elif level_name:
            lines.append(f"Location: dungeon level {level} — {level_name}")
        else:
            lines.append(f"Location: dungeon level {level}")

        achievements_arr = np.array(state.achievements[player_idx])
        num_done = int(achievements_arr.sum())
        num_total = len(achievements_arr)
        visible_count = len(_achievements_for_level(level, coordination_enabled=False))
        lines.append(
            f"Achievements: {num_done}/{num_total} ({num_total - visible_count} unlock later)"
        )

        return "\n".join(lines)

    def describe_frame(
        self,
        state,
        player_idx,
        reward=None,
        new_achievements=None,
        last_action=None,
        action_failed=False,
    ):
        """Describe solo state with a game-over message on death.

        Args:
            state: Current environment state.
            player_idx: Player whose perspective should be described.
            reward: Optional reward from the preceding transition.
            new_achievements: Newly earned achievement names.
            last_action: Canonical action attempted on the preceding transition.
            action_failed: Whether the attempted action had no effect.

        Returns:
            Long-term world context and short-term inventory context.
        """
        is_dead = bool(state.player_health[player_idx] <= 0)
        if is_dead:
            inventory_desc = self.describe_inventory(state, player_idx)
            feedback = self.describe_step_feedback(
                reward, new_achievements, last_action=last_action, action_failed=action_failed
            )
            result = ""
            if feedback:
                result += feedback + "\n\n"
            timestep = int(state.timestep)
            max_timesteps = int(self.env_params.max_timesteps)
            result += f"Step: {timestep}/{max_timesteps} ({max_timesteps - timestep} remaining)\n"
            if self.exact_coordinates:
                result += f"Position: {self._xy_coord_str(state.player_position[player_idx])}\n"
            result += "Status: dead (game over)\n"
            return result.strip(), inventory_desc

        # Non-dead: parent path. describe_teammates() is overridden to "" and
        # describe_coordination_cues() is gated on coordination_enabled (False
        # for single-agent), so no multi-agent text leaks through.
        return super().describe_frame(
            state,
            player_idx,
            reward=reward,
            new_achievements=new_achievements,
            last_action=last_action,
            action_failed=action_failed,
        )

    def get_instruction_prompt(
        self,
        agent_id=None,
        role=None,
        include_all_actions=True,
        progressive_disclosure=False,
        current_level=0,
        prompt_mode=None,
        **kwargs,
    ):
        """Build a single-agent prompt without coordination content.

        Args:
            agent_id: Optional agent identifier used in the introduction.
            role: Optional role name used in the introduction.
            include_all_actions: Whether to include the full action block.
            progressive_disclosure: Whether late-game rules are level-gated.
            current_level: Current level used for progressive disclosure.
            prompt_mode: Prompt detail mode; collaborative mode is demoted.
            **kwargs: Ignored compatibility arguments.

        Returns:
            Coordination-free system-prompt text.
        """
        if prompt_mode is None:
            prompt_mode = self.prompt_mode
        return get_instruction_prompt_single(
            include_all_actions=include_all_actions,
            progressive_disclosure=progressive_disclosure,
            current_level=current_level,
            prompt_mode=prompt_mode,
            agent_id=agent_id,
            role=role,
        )
