"""
exp_pstop_vs_length.py — Does the stop head know when an instruction is done?

Experiment design
─────────────────
For each of 25 target word lengths (1–100), issue 4 instructions — one per
semantic type — to the controller and record:

  - steps until p_stop first exceeds threshold  (did the stop head fire?)
  - whether p_stop fired at all within max_steps (break = never fires)
  - p_stop trace over the episode

4 instruction types:
  navigation  — moving/exploring
  collection  — gathering resources
  crafting    — making items
  survival    — dealing with threats

Total: 25 lengths × 4 types = 100 instructions.

Each instruction is issued once; the episode ends when p_stop ≥ threshold
OR max_steps is reached (whichever comes first).

Training distribution reference (mc200m_seed3, n=99748):
  mean=5.45w, std=1.17, min=2, p90=7, max=11

Breaking-point signature:
  - p_stop fires quickly and cleanly  → interface intact
  - p_stop never fires / fires very late  → interface broken for that length
  - steps-to-stop plot shows a cliff at ~12w (first OOD length)

Usage
─────
  # Smoke test (mock, no GPU):
  python experiments/exp_pstop_vs_length.py --mock

  # Real run:
  python experiments/exp_pstop_vs_length.py \\
      --checkpoint /data/terran/instruct_to_act/mc200m_seed3 \\
      --max_steps 300 \\
      --outdir results/pstop_vs_length
"""

import argparse
import datetime
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# Instruction list  (25 lengths × 4 types = 100 instructions)
# ─────────────────────────────────────────────────────────────────────────────
#
# Strategy
# ────────
# In-distribution (lengths 1–11): natural Minecraft imperatives.
# OOD (lengths 12–100): extend by appending more spatial / conditional context.
# Padding rule: always add real words — no repetition — so the embedding
# changes meaningfully with length rather than saturating.
#
# Each entry: (length_target, type, instruction_string)
# length_target = word count (verify with len(s.split()))

INSTRUCTIONS: List[Tuple[int, str, str]] = [
    (1, 'navigation', 'go'),
    (1, 'collection', 'mine'),
    (1, 'crafting', 'craft'),
    (1, 'survival', 'flee'),
    (2, 'navigation', 'go north'),
    (2, 'collection', 'mine wood'),
    (2, 'crafting', 'craft planks'),
    (2, 'survival', 'flee danger'),
    (3, 'navigation', 'move toward cave'),
    (3, 'collection', 'chop down trees'),
    (3, 'crafting', 'craft stone pickaxe'),
    (3, 'survival', 'avoid the creeper'),
    (4, 'navigation', 'walk toward the cave'),
    (4, 'collection', 'collect wood from trees'),
    (4, 'crafting', 'craft a crafting table'),
    (4, 'survival', 'hide from the creeper'),
    (5, 'navigation', 'walk toward the nearby cave'),
    (5, 'collection', 'chop trees to gather wood'),
    (5, 'crafting', 'craft a wooden pickaxe now'),
    (5, 'survival', 'run away from the creeper'),
    (6, 'navigation', 'move toward the cave entrance ahead'),
    (6, 'collection', 'mine iron ore in the cave'),
    (6, 'crafting', 'craft a stone pickaxe using cobblestone'),
    (6, 'survival', 'back away from the hostile mob'),
    (7, 'navigation', 'descend into the cave to find ores'),
    (7, 'collection', 'mine iron ore with the stone pickaxe'),
    (7, 'crafting', 'craft a furnace using eight cobblestone blocks'),
    (7, 'survival', 'retreat upward to escape the lava flow'),
    (8, 'navigation', 'navigate toward the tall trees on the hill'),
    (8, 'collection', 'collect wood from the trees near the hill'),
    (8, 'crafting', 'smelt iron ore in the furnace using coal'),
    (8, 'survival', 'place a torch to light the dark tunnel'),
    (9, 'navigation', 'follow the river downstream until you reach a cave'),
    (9, 'collection', 'mine the coal seam visible on the left cave'),
    (9, 'crafting', 'use the crafting table to make wooden planks from'),
    (9, 'survival', 'build a dirt wall to block the skeleton from'),
    (10, 'navigation', 'explore the cavern to the south until you spot iron'),
    (10, 'collection', 'gather at least eight pieces of cobblestone from the cave'),
    (10, 'crafting', 'craft sticks from planks and then make a wooden pickaxe'),
    (10, 'survival', 'jump over the gap and run away from the incoming'),
    (11, 'navigation', 'head toward the mountain to the east and find the cave'),
    (11, 'collection', 'dig down at the base of the hill to reach the'),
    (11, 'crafting', 'place the crafting table and make an iron pickaxe from iron'),
    (11, 'survival', 'light the area with torches so hostile mobs cannot spawn near'),
    (12, 'navigation', 'travel east across the biome and enter the cave you see on'),
    (12, 'collection', 'mine the iron ore deposit on the left wall and collect all'),
    (12, 'crafting', 'open the crafting table and use six iron ingots and two sticks'),
    (12, 'survival', 'equip your sword and attack the creeper before it explodes and damages'),
    (14, 'navigation', 'navigate through the forest to the north find the river and follow it east'),
    (14, 'collection', 'dig straight down at this location until you reach the stone layer and mine'),
    (14, 'crafting', 'first gather eight cobblestone blocks then return here and use the crafting table to'),
    (14, 'survival', 'move back from the lava pool place a stone block to seal the gap'),
    (16, 'navigation', 'head north past the two oak trees cross the small river and climb the hill until'),
    (16, 'collection', 'mine along the exposed stone wall on the east side of the cave collecting every piece'),
    (16, 'crafting', 'place the furnace adjacent to the crafting table load it with coal and iron ore then'),
    (16, 'survival', 'quickly place three dirt blocks behind you to block the zombie path then turn around and'),
    (18, 'navigation', 'walk north along the riverbank until the biome changes to a dark forest then turn east and locate'),
    (18, 'collection', 'use the iron pickaxe to mine the diamond ore nodes visible at the bottom of the ravine making'),
    (18, 'crafting', 'gather four wooden planks open the crafting table arrange them in a two by two grid and place'),
    (18, 'survival', 'place a temporary dirt shelter around yourself seal the entrance with a single block and wait inside until'),
    (20, 'navigation', 'travel south through the plains biome until you reach the desert then turn west and walk until you find the'),
    (20, 'collection', 'descend to level twelve using your iron pickaxe dig a staircase pattern downward then mine any diamond ore you encounter'),
    (20, 'crafting', 'collect three blocks of iron ore smelt them into ingots using the furnace and coal then open the crafting table'),
    (20, 'survival', 'immediately dig a two by two room in the nearest hillside place your bed inside light the space with torches'),
    (25, 'navigation', 'begin by facing north and walking forward until you reach a large oak tree then turn right and follow the cliff edge east until you'),
    (25, 'collection', 'dig a two block wide staircase going downward at a forty five degree angle continuing until you reach bedrock level then mine horizontally east collecting'),
    (25, 'crafting', 'open the crafting table and arrange three iron ingots in the top row two sticks in the middle column and nothing elsewhere to produce an'),
    (25, 'survival', 'when you hear the hissing sound immediately sprint away from the creeper in the opposite direction and keep running until the sound stops then look'),
    (30, 'navigation', 'start at your current position face the direction of the rising sun which is east then walk forward past the birch forest across the flower plains and up the gradual'),
    (30, 'collection', 'equip the iron pickaxe from your hotbar descend the staircase mine to level eleven by counting blocks as you go then strip mine east for thirty blocks collecting every piece'),
    (30, 'crafting', 'open your inventory check that you have at least eight cobblestone blocks four wooden planks and two sticks then approach the crafting table right click to open it and craft'),
    (30, 'survival', 'as soon as you see the spider sprint toward the nearest tree climb it by jumping and placing blocks beneath you until you are four blocks above ground then wait'),
    (35, 'navigation', 'leave your current base heading northwest through the taiga biome keeping the mountain range visible on your left side cross the frozen river by placing dirt stepping stones continue past the ice spikes formation and'),
    (35, 'collection', 'mine a three block wide tunnel heading due east from your current position at level twelve depth placing torches every eight blocks on the right wall for orientation mark any side tunnels with cobblestone signs'),
    (35, 'crafting', 'gather the following materials before approaching the crafting table eight cobblestone for a furnace four iron ingots and two sticks for an iron pickaxe and six iron ingots for iron leggings then craft each item'),
    (35, 'survival', 'when your health drops below five hearts immediately stop moving crouch behind the nearest boulder or tree trunk eat a golden apple or cooked pork if available wait for regeneration to restore at least eight'),
    (40, 'navigation', 'orient yourself so the sun is setting directly in front of you which means you are facing west then walk forward for approximately one hundred blocks keeping the river on your right side when the river forks take the left'),
    (40, 'collection', 'equip the iron pickaxe and descend through the existing mineshaft using the rope ladder you placed earlier at the bottom of shaft A turn left into the eastern gallery where you previously spotted the large diamond vein mine out all'),
    (40, 'crafting', 'before you begin crafting sort your entire inventory placing raw ores in chest A coal in chest B and excess stone in chest C load the furnace with sixteen coal units and all remaining raw iron and gold ore wait'),
    (40, 'survival', 'when you enter the nether fortress proceed cautiously with your shield held in the off hand facing the direction of travel place a trail of cobblestone markers every ten blocks so you can find your way back when a blaze'),
    (50, 'navigation', 'to reach the stronghold first craft three eyes of ender by combining ender pearls with blaze powder in the crafting table then go outside and throw one eye upward it will float in a direction follow that direction walking for fifty blocks throw another eye repeat this process adjusting your'),
    (50, 'collection', 'to efficiently farm iron ore in the deep slate layer first descend to level negative twenty using the existing staircase then branch mine by digging two parallel tunnels each forty blocks long separated by two blocks of stone walk the left tunnel first breaking every second block at eye level'),
    (50, 'crafting', 'to prepare fully for entering the end dimension gather the following materials in this order twelve obsidian blocks by pouring water over lava then mining with a diamond pickaxe one flint and steel made from iron ingot and flint at least sixteen cooked golden apples crafted by surrounding an apple'),
    (50, 'survival', 'upon entering the end and seeing the ender dragon circle the central fountain immediately run to the nearest obsidian pillar and take cover behind its base look upward to locate the end crystal on top of the pillar shoot it with arrows from cover when it explodes it will remove'),
    (60, 'navigation', 'to locate the nearest ocean monument begin by crafting a respiration helmet by enchanting an iron helmet with respiration three using the enchanting table then stand on the highest point near your base and look for the distinctive dark blue green tint of deep ocean biome tiles on the horizon use a boat to cross shallow ocean sections and sprint'),
    (60, 'collection', 'to obtain silk touch string without destroying spawner blocks first place a chest adjacent to your mob farm sorting system then set the hopper chain so string routes to chest B iron to chest C and experience orbs route nowhere since they cannot be piped check that the lighting in all four spawning platforms is at zero by using a'),
    (60, 'crafting', 'to build a complete set of netherite armor first collect four netherite ingots each requiring four netherite scraps and four gold ingots netherite scraps are obtained by smelting ancient debris in a blast furnace ancient debris spawns at level fifteen in the nether mine using beds which can be detonated in the nether to expose ancient debris clusters safely once'),
    (60, 'survival', 'when the warden emerges from the ground immediately stop all movement crouching reduces your noise signature to near zero switch off any active minecart conveyor systems nearby that generate vibration wait in place without sneaking for fifteen seconds while the warden scans for vibration sources if it begins moving toward you remain crouched and move one block sideways every five'),
    (70, 'navigation', 'to find the woodland mansion start by crafting a woodland explorer map which requires trading an emerald and a compass with a cartographer villager once you have the map open it and orient yourself so your white dot overlaps the mansion icon walk in the direction indicated by the map continuing for potentially thousands of blocks do not attempt shortcuts through swamps at night as drowned zombies will slow you'),
    (70, 'collection', 'to farm enough xp for a maximum enchanting session first build a standard mob tower farm using the following materials sixty four cobblestone for the spawning floor thirty two trapdoors to force mobs to fall sixteen hoppers routed to a double chest and four water source blocks placed in the corners of each twenty four block wide spawning platform build the kill chamber at the base of a twenty three'),
    (70, 'crafting', 'to prepare a complete exploration kit for a multi day nether expedition gather the following items before opening the crafting table two stacks of cobblestone for building emergency shelters sixteen gold ingots to craft golden armor pieces for piglin neutrality two fire resistance potions brewed from magma cream and awkward potions in the brewing stand four ender chests crafted from obsidian and an eye of ender for portable storage two'),
    (70, 'survival', 'when you fall into a lava lake in the nether immediately press the sprint swim button to move toward the nearest obsidian or netherrack ledge if a fire resistance potion is available drink it the moment you enter the lava pool to prevent further burning damage if no potion is available position your crosshair on the ledge edge and use the jump button combined with forward movement to boost yourself'),
    (80, 'navigation', 'to complete the journey from overworld spawn to the end portal room in the stronghold follow these steps in sequence first gather enough resources to create three eyes of ender by killing at least three blazes in the nether for blaze rods converting the rods to blaze powder and combining them with ender pearls obtained from endermen in the end biome second return to the overworld and throw one eye of ender upward which will drift in the direction of'),
    (80, 'collection', 'to gather all materials required for a fully enchanted diamond armor set before the enchanting session collect the following in order sixteen diamonds for the four armor pieces from strip mining at level twelve forty eight lapis lazuli blocks for enchanting from mining blue speckled stone eight sugarcane plants grown adjacent to water for paper to combine into books thirty two leather from cows for the same books and one hundred twenty experience levels from mob farming then at the'),
    (80, 'crafting', 'to build a complete automated iron farm using villagers and zombies first construct a platform twenty four blocks above the ground with a three by three grid of doors to attract villagers place one zombie in a dark minecart at the platform edge so it is visible to the villagers but cannot reach them the villagers will panic and attempt to sleep in beds placed on the platform which causes iron golems to spawn around them build a one block'),
    (80, 'survival', 'when you are trapped underground with no torches and your hunger bar is below three drumsticks execute the following survival protocol immediately stop all movement to prevent further hunger drain open your inventory and eat any raw food available since hunger from raw meat is better than starvation craft a wooden pickaxe if you have logs and no other tools mine upward at a forty five degree angle by digging one block forward and one block up alternating until you'),
    (100, 'navigation', 'to plan and execute a complete expedition from your home base to the nearest jungle temple begin by scouting from an elevated position and identifying the jungle biome on the horizon using the distinctive tall dark green canopy prepare for the journey by stocking sixty four cooked food items for stamina and a full set of iron armor with a sword and bow for defense craft at least two stacks of arrows and bring a compass and empty maps to chart the route as you travel leave trail markers every thirty blocks by placing a single torch on the right'),
    (100, 'collection', 'to conduct a maximally efficient mining session for diamonds and netherite follow this complete protocol start by smelting twenty four iron ingots into an iron pickaxe and two iron shovels for clearing gravel descend using the staircase method to level negative fifty eight which is the optimal layer for diamond spawning using the 1.18 generation rules once at depth dig a three block tall main corridor heading due east for one hundred blocks placing torches on the left wall every eight blocks then branch off perpendicular corridors every three blocks to the right these branches should each extend twenty blocks'),
    (100, 'crafting', 'to prepare a complete end game loadout capable of defeating both the ender dragon and the wither boss follow this crafting sequence in order first smelt all raw metals in the blast furnace prioritizing netherite scraps into ingots then diamonds into blocks for compact storage second open the smithing table and upgrade your diamond sword chestplate helmet leggings and boots to netherite one piece at a time using the netherite upgrade template and one netherite ingot each third open the enchanting table and apply sharpness five to the netherite sword using your stored experience levels fourth enchant the chestplate with'),
    (100, 'survival', 'when you discover you have accidentally entered a deep dark biome without preparing for the warden execute this complete withdrawal protocol as follows immediately stop all movement and switch to sneaking mode which reduces your noise signature to nearly zero scan the area visually for any sculk shriekers the dark blue blocks that summon the warden when activated and note their positions if you have accidentally triggered a shrieker you have at most three activations before a warden spawns begin moving by sneaking one block at a time toward the direction you entered pausing for two full seconds between each'),
]


# Verify all instructions have correct word counts
def _verify_instructions():
    errors = []
    for target_len, itype, text in INSTRUCTIONS:
        actual = len(text.split())
        if actual != target_len:
            errors.append(f"  MISMATCH: target={target_len} actual={actual} [{itype}] {text[:60]}")
    if errors:
        print("WARNING — instruction length mismatches:")
        for e in errors:
            print(e)
    return len(errors) == 0


# Length checkpoints present in the list (derived, not hardcoded)
CHECKPOINTS = sorted(set(t for t, _, _ in INSTRUCTIONS))
TYPES       = ['navigation', 'collection', 'crafting', 'survival']


# ─────────────────────────────────────────────────────────────────────────────
# Episode runner
# ─────────────────────────────────────────────────────────────────────────────

def run_instruction_episode(ctrl, instruction: str, env, encoder,
                            stop_threshold: float, max_steps: int) -> dict:
    """
    Issue one instruction; run until p_stop fires or max_steps reached.

    Returns
    -------
    steps_to_stop : int or None  (None = never fired within max_steps)
    p_stop_fired  : bool
    p_stop_trace  : list[float]
    reward        : float
    steps_taken   : int
    """
    ctrl_state = ctrl.init_policy(batch_size=1)
    lang_size  = getattr(ctrl, '_lang_size', 384)
    embed      = encoder(instruction)

    _SERVER_ERRORS = (ConnectionRefusedError, TimeoutError, OSError,
                      AttributeError, BrokenPipeError, RuntimeError)
    try:
        obs, _     = env.reset()
    except _SERVER_ERRORS as e:
        raise RuntimeError(f"Minecraft server down at reset: {e}") from e

    total_reward  = 0.0
    p_stop_trace: List[float] = []
    steps_to_stop = None

    for step in range(max_steps):
        obs_b = {k: np.expand_dims(v, 0) for k, v in obs.items()
                 if not k.startswith('log/')}
        obs_b['lang_embed'] = embed[np.newaxis]

        ctrl_state, action, extra = ctrl.policy(ctrl_state, obs_b, mode='eval')
        p_stop = float(extra.get('log/p_stop', extra.get('p_stop', 0.0)))
        p_stop_trace.append(p_stop)

        try:
            obs, reward, terminated, truncated, _ = env.step(
                np.array(action['action'][0])
            )
        except _SERVER_ERRORS:
            break

        total_reward += float(reward)

        if p_stop >= stop_threshold and steps_to_stop is None:
            steps_to_stop = step + 1

        if terminated or truncated:
            break

    return {
        'steps_to_stop':  steps_to_stop,
        'p_stop_fired':   steps_to_stop is not None,
        'p_stop_trace':   p_stop_trace,
        'reward':         total_reward,
        'steps_taken':    len(p_stop_trace),
        'p_stop_max':     float(max(p_stop_trace)) if p_stop_trace else 0.0,
        'p_stop_mean':    float(np.mean(p_stop_trace)) if p_stop_trace else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Progress logger
# ─────────────────────────────────────────────────────────────────────────────

class ProgressLogger:
    def __init__(self, outdir: Path, total: int):
        self._path  = outdir / 'progress.jsonl'
        self._total = total
        self._done  = 0
        self._t0    = time.perf_counter()
        outdir.mkdir(parents=True, exist_ok=True)

    def log(self, **kw):
        self._done += 1
        elapsed = time.perf_counter() - self._t0
        avg     = elapsed / self._done
        eta     = str(datetime.timedelta(seconds=int(avg * (self._total - self._done))))
        now     = datetime.datetime.now().strftime('%H:%M:%S')
        entry   = dict(ts=now, elapsed=round(elapsed, 1), eta=eta,
                       done=self._done, total=self._total,
                       pct=round(100 * self._done / self._total, 1), **kw)
        with open(self._path, 'a') as f:
            f.write(json.dumps(entry) + '\n')
        fired = 'FIRED' if kw.get('p_stop_fired') else 'timeout'
        print(f"  {now}  [{kw.get('itype','?'):>12}|L={kw.get('length',0):>3}]  "
              f"{fired:>7}  steps_to_stop={str(kw.get('steps_to_stop',None)):>6}  "
              f"p_max={kw.get('p_stop_max',0):.3f}  "
              f"elapsed={elapsed/60:.1f}m  ETA={eta}")


# ─────────────────────────────────────────────────────────────────────────────
# Mock components
# ─────────────────────────────────────────────────────────────────────────────

class MockController:
    _lang_size = 384

    def init_policy(self, batch_size=1):
        return None

    def policy(self, state, obs_batch, mode='eval'):
        time.sleep(0.001)
        # Simulate: shorter instructions → higher p_stop (controller more confident)
        embed = obs_batch.get('lang_embed', np.zeros((1, 384)))[0]
        embed_norm = float(np.linalg.norm(embed))
        # Mock: p_stop decays with embedding norm (longer = more diluted = lower)
        p_stop = float(np.clip(0.6 / (1.0 + embed_norm * 0.1) + np.random.normal(0, 0.1), 0, 1))
        return state, {'action': np.zeros((1,), dtype=np.int32)}, {'log/p_stop': p_stop}


class MockEnv:
    def reset(self):
        return {'image': np.zeros((64, 64, 3), np.uint8)}, {}

    def step(self, action):
        r = float(np.random.random() < 0.005)
        return {'image': np.zeros((64, 64, 3), np.uint8)}, r, False, False, {}


def mock_encoder(text):
    # Simulate: longer text → lower-norm embedding (dilution)
    n_words = len(text.split())
    scale   = 1.0 / max(1, n_words ** 0.5)
    rng     = np.random.default_rng(abs(hash(text)) % (2**31))
    return (rng.standard_normal(384) * scale).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    'navigation': '#1565C0',
    'collection': '#2E7D32',
    'crafting':   '#F57F17',
    'survival':   '#B71C1C',
}

TRAINING_DIST_RANGE = (2, 11)   # actual training min / max (measured)
TRAINING_DIST_MEAN  = 5.45


def plot_steps_to_stop(results: dict, outdir: Path):
    """
    X = instruction word length, Y = steps until p_stop fires.
    None (timeout) plotted as max_steps with a marker change.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    for itype in TYPES:
        lengths, steps, timed_out = [], [], []
        for L in CHECKPOINTS:
            key = (L, itype)
            if key not in results:
                continue
            r = results[key]
            lengths.append(L)
            s = r['steps_to_stop']
            steps.append(s if s is not None else r['max_steps'])
            timed_out.append(s is None)

        if not lengths:
            continue

        lengths   = np.array(lengths)
        steps     = np.array(steps, dtype=float)
        timed_out = np.array(timed_out)

        # Normal: solid line
        ax.plot(lengths[~timed_out], steps[~timed_out],
                marker='o', color=COLORS[itype], lw=2, label=itype, alpha=0.85)
        # Timeout: open marker
        if timed_out.any():
            ax.scatter(lengths[timed_out], steps[timed_out],
                       marker='x', color=COLORS[itype], s=80, linewidths=2, zorder=5)

    # Training distribution band
    ax.axvspan(*TRAINING_DIST_RANGE, alpha=0.10, color='gray',
               label=f'training dist ({TRAINING_DIST_RANGE[0]}–{TRAINING_DIST_RANGE[1]}w)')
    ax.axvline(TRAINING_DIST_MEAN, color='gray', lw=1.0, ls='--', alpha=0.6)

    ax.set_xlabel('Instruction length (words)', fontsize=11)
    ax.set_ylabel('Steps until p_stop fires', fontsize=11)
    ax.set_title('Stop-head response vs. instruction length\n'
                 '(× = timed out = stop head never fired)', fontsize=10)
    ax.set_xscale('log')
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(outdir / f'steps_to_stop.{ext}',
                    bbox_inches='tight', dpi=150 if ext == 'png' else None)
    plt.close()
    print(f"  Saved {outdir}/steps_to_stop.{{pdf,png}}")


def plot_mean_ci(results: dict, outdir: Path):
    """Two-panel figure: mean p_stop_max and mean reward vs length, with 95% CI."""
    lengths = sorted(set(k[0] for k in results))
    xs      = np.array(lengths)

    def agg(field, raw_field):
        ms, los, his = [], [], []
        for L in lengths:
            vals = []
            for t in TYPES:
                k = (L, t)
                if k not in results: continue
                raw = results[k].get(raw_field)
                if raw:
                    vals.extend(raw)
                else:
                    vals.append(results[k][field])
            if not vals:
                ms.append(0); los.append(0); his.append(0); continue
            boots = [np.mean(np.random.choice(vals, len(vals), replace=True))
                     for _ in range(2000)]
            ms.append(float(np.mean(vals)))
            los.append(float(np.percentile(boots, 2.5)))
            his.append(float(np.percentile(boots, 97.5)))
        return np.array(ms), np.array(los), np.array(his)

    np.random.seed(0)
    pmeans, plos, phis = agg('p_stop_max', 'all_p_stop_max')
    rmeans, rlos, rhis = agg('reward',     'all_rewards')

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    for ax in axes:
        ax.set_xscale('log')
        ax.set_xlim(0.8, 130)
        ax.set_xticks(lengths)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.tick_params(axis='x', labelsize=9)
        ax.axvspan(*TRAINING_DIST_RANGE, alpha=0.10, color='gray')
        ax.grid(True, alpha=0.22, which='both')

    ax = axes[0]
    ax.fill_between(xs, plos, phis, color='#1565C0', alpha=0.18, label='95% CI')
    ax.plot(xs, pmeans, color='#1565C0', lw=2.5, marker='o', ms=5,
            label='Mean max p_stop')
    ax.axhline(0.3, color='black', lw=1.2, ls='--', alpha=0.6, label='threshold = 0.3')
    ax.set_ylabel('Mean max p_stop', fontsize=11)
    ax.set_ylim(-0.02, 1.08)
    ax.set_title('Peak stop-head confidence vs instruction length  '
                 '(mean ± 95% CI across types × episodes)', fontsize=10)
    ax.legend(fontsize=9)
    ax.text(np.sqrt(TRAINING_DIST_RANGE[0] * TRAINING_DIST_RANGE[1]), 1.03,
            'training dist', fontsize=8, color='gray', ha='center', style='italic')

    ax = axes[1]
    ax.fill_between(xs, rlos, rhis, color='#2E7D32', alpha=0.18, label='95% CI')
    ax.plot(xs, rmeans, color='#2E7D32', lw=2.5, marker='o', ms=5,
            label='Mean episode reward')
    ax.axhline(0, color='black', lw=0.8, ls='-', alpha=0.3)
    ax.set_xlabel('Instruction length (words)', fontsize=11)
    ax.set_ylabel('Mean episode reward', fontsize=11)
    ax.set_title('Task score vs instruction length  '
                 f'(mean ± 95% CI, n={list(results.values())[0].get("n_episodes",1)} eps each)',
                 fontsize=10)
    ax.legend(fontsize=9)

    plt.tight_layout(h_pad=2.5)
    for ext in ('pdf', 'png'):
        fig.savefig(outdir / f'pstop_reward_vs_length.{ext}',
                    bbox_inches='tight', dpi=150 if ext == 'png' else None)
    plt.close()
    print(f"  Saved {outdir}/pstop_reward_vs_length.{{pdf,png}}")


def plot_p_stop_max(results: dict, outdir: Path):
    """
    X = instruction length, Y = max p_stop value reached during episode.
    Shows whether the stop head ever gets confident regardless of threshold.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    for itype in TYPES:
        lengths, p_maxes = [], []
        for L in CHECKPOINTS:
            key = (L, itype)
            if key not in results:
                continue
            lengths.append(L)
            p_maxes.append(results[key]['p_stop_max'])

        if lengths:
            ax.plot(lengths, p_maxes, marker='o', color=COLORS[itype],
                    lw=2, label=itype, alpha=0.85)

    ax.axvspan(*TRAINING_DIST_RANGE, alpha=0.10, color='gray',
               label=f'training dist ({TRAINING_DIST_RANGE[0]}–{TRAINING_DIST_RANGE[1]}w)')
    ax.axhline(0.5, color='black', lw=1, ls='--', alpha=0.5, label='threshold=0.5')
    ax.set_xlabel('Instruction length (words)', fontsize=11)
    ax.set_ylabel('Max p_stop during episode', fontsize=11)
    ax.set_title('Peak stop-head confidence vs. instruction length\n'
                 '(below 0.5 = stop head never crossed threshold)', fontsize=10)
    ax.set_xscale('log')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(outdir / f'p_stop_max.{ext}',
                    bbox_inches='tight', dpi=150 if ext == 'png' else None)
    plt.close()
    print(f"  Saved {outdir}/p_stop_max.{{pdf,png}}")


def plot_fired_rate(results: dict, outdir: Path):
    """
    X = instruction length, Y = fraction of types where p_stop fired.
    (All 4 types at each length → 0, 0.25, 0.5, 0.75, or 1.0)
    """
    fired_by_length = {}
    for L in CHECKPOINTS:
        fired = [results[(L, t)]['p_stop_fired']
                 for t in TYPES if (L, t) in results]
        if fired:
            fired_by_length[L] = np.mean(fired)

    lengths = sorted(fired_by_length)
    rates   = [fired_by_length[L] for L in lengths]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(len(lengths)), rates,
           color=[plt.cm.RdYlGn(r) for r in rates], alpha=0.85, width=0.7)
    ax.set_xticks(range(len(lengths)))
    ax.set_xticklabels([str(L) for L in lengths], fontsize=8, rotation=45)
    ax.set_xlabel('Instruction length (words)', fontsize=11)
    ax.set_ylabel('Fraction of types where p_stop fired', fontsize=11)
    ax.set_title('Where does the stop head break?\n'
                 '(green = fired for all 4 types; red = never fired)', fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.axvspan(
        lengths.index(TRAINING_DIST_RANGE[0]) - 0.5 if TRAINING_DIST_RANGE[0] in lengths else -0.5,
        lengths.index(TRAINING_DIST_RANGE[1]) - 0.5 if TRAINING_DIST_RANGE[1] in lengths else len(lengths),
        alpha=0.10, color='gray', label='training dist'
    )
    ax.grid(True, alpha=0.25, axis='y')
    ax.legend(fontsize=9)
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(outdir / f'fired_rate.{ext}',
                    bbox_inches='tight', dpi=150 if ext == 'png' else None)
    plt.close()
    print(f"  Saved {outdir}/fired_rate.{{pdf,png}}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results: dict, max_steps: int):
    print(f"\n{'─'*88}")
    print(f"  {'L':>4}  {'type':>12}  {'fired':>5}  {'steps_to_stop':>14}  "
          f"{'p_max':>6}  {'p_mean':>6}  {'reward':>8}")
    print(f"  {'─'*86}")
    for L in CHECKPOINTS:
        for itype in TYPES:
            key = (L, itype)
            if key not in results:
                continue
            r = results[key]
            fired_str = 'YES' if r['p_stop_fired'] else 'NO '
            stp       = str(r['steps_to_stop']) if r['p_stop_fired'] else f'>{max_steps}'
            print(f"  {L:>4}  {itype:>12}  {fired_str:>5}  {stp:>14}  "
                  f"{r['p_stop_max']:>6.3f}  {r['p_stop_mean']:>6.3f}  {r['reward']:>8.4f}")
        print()
    print(f"{'─'*88}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='p_stop vs. instruction length — interface breaking point experiment'
    )
    p.add_argument('--checkpoint',     default='/data/terran/instruct_to_act/mc200m_seed3')
    p.add_argument('--max_steps',      type=int, default=300,
                   help='Max steps per instruction before declaring timeout')
    p.add_argument('--stop_threshold', type=float, default=0.5)
    p.add_argument('--n_episodes',     type=int, default=1,
                   help='Episodes per (length, type) pair. >1 gives reliable mean/CI.')
    p.add_argument('--outdir',         default='results/pstop_vs_length')
    p.add_argument('--mock',           action='store_true')
    p.add_argument('--seed',           type=int, default=0)
    return p.parse_args()


def _merge_episodes(episodes: list) -> dict:
    """Aggregate a list of single-episode result dicts into one summary dict."""
    p_maxes  = [e['p_stop_max']  for e in episodes]
    p_means  = [e['p_stop_mean'] for e in episodes]
    rewards  = [e['reward']      for e in episodes]
    fired    = [e['p_stop_fired'] for e in episodes]
    stps     = [e['steps_to_stop'] for e in episodes if e['steps_to_stop'] is not None]
    return {
        'p_stop_max':    float(np.mean(p_maxes)),
        'p_stop_max_std': float(np.std(p_maxes)),
        'p_stop_mean':   float(np.mean(p_means)),
        'reward':        float(np.mean(rewards)),
        'reward_std':    float(np.std(rewards)),
        'p_stop_fired':  any(fired),
        'fire_rate':     float(np.mean(fired)),
        'steps_to_stop': float(np.mean(stps)) if stps else None,
        'n_episodes':    len(episodes),
        'max_steps':     episodes[0]['max_steps'],
        'instruction':   episodes[0]['instruction'],
        'length':        episodes[0]['length'],
        'itype':         episodes[0]['itype'],
        # keep all raw values for CI computation in plotting
        'all_p_stop_max': p_maxes,
        'all_rewards':    rewards,
    }


def main():
    args   = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)

    ok = _verify_instructions()
    if not ok:
        print("Fix instruction lengths before running.")
        sys.exit(1)
    print(f"Instruction list OK: {len(INSTRUCTIONS)} instructions, "
          f"{len(CHECKPOINTS)} length checkpoints, {len(TYPES)} types, "
          f"{args.n_episodes} episode(s) each")

    if args.mock:
        ctrl    = MockController()
        env     = MockEnv()
        encoder = mock_encoder
    else:
        root = Path(__file__).parent.parent
        sys.path.insert(0, str(root))
        sys.path.insert(0, str(root / 'dreamerv3'))
        from envs  import make_env
        from train import load_project_config
        from evaluate import load_agent, GymAdapter, encode_fn

        ckpt_dir  = Path(args.checkpoint)
        cfg_paths = [str(root / 'configs' / 'base.yaml'),
                     str(root / 'configs' / 'minecraft.yaml')]
        if '200m' in ckpt_dir.name:
            cfg_paths.append(str(root / 'configs' / 'size200m.yaml'))
            print("── Applying size200m.yaml ──")

        latest = ckpt_dir / 'ckpt' / 'latest'
        if latest.exists():
            tag       = latest.read_text().strip()
            ckpt_path = ckpt_dir / 'ckpt' / tag / 'agent.pkl'
        else:
            candidates = sorted((ckpt_dir / 'ckpt').glob('*/agent.pkl'))
            ckpt_path  = candidates[-1]
        print(f"── Checkpoint: {ckpt_path} ──")

        proj    = load_project_config(cfg_paths)
        lang_sz = int(proj.get('lang_size', 384))
        encoder = encode_fn(lang_sz)
        env     = GymAdapter(make_env('minecraft', proj, lang_size=lang_sz))
        ctrl    = load_agent(str(ckpt_path), env.obs_space, env.act_space, proj)

    total = len(INSTRUCTIONS) * args.n_episodes
    plog  = ProgressLogger(outdir, total)

    # Load existing partial results to allow resuming.
    # `existing` is used to recover prior episodes per instruction.
    # `results` is pre-seeded with all existing data so that the partial file
    # always contains the union of ALL runs, not just the current one.
    partial_path = outdir / 'results_partial.json'
    existing = {}
    results  = {}
    if partial_path.exists():
        with open(partial_path) as f:
            raw = json.load(f)
        for key_str, v in raw.items():
            L     = int(key_str.split('_')[0])
            itype = '_'.join(key_str.split('_')[1:])
            existing[(L, itype)] = v
            results[(L, itype)]  = v   # pre-seed so partial saves keep old data
        print(f"Loaded {len(existing)} existing results — will append new episodes.")

    print(f"\nRunning {len(INSTRUCTIONS)} instructions × {args.n_episodes} episodes "
          f"= {total} total  (max_steps={args.max_steps})")
    print(f"Monitor: tail -f {outdir}/progress.jsonl\n")

    for length, itype, instruction in INSTRUCTIONS:
        key = (length, itype)

        # Recover already-run episodes from the partial file
        prev = existing.get(key, {})
        prev_episodes = []
        if 'all_p_stop_max' in prev:
            n_prev = len(prev['all_p_stop_max'])
            for i in range(n_prev):
                prev_episodes.append({
                    'p_stop_max':   prev['all_p_stop_max'][i],
                    'p_stop_mean':  prev.get('p_stop_mean', 0),
                    'reward':       prev['all_rewards'][i],
                    'p_stop_fired': prev['all_p_stop_max'][i] >= args.stop_threshold,
                    'steps_to_stop': prev.get('steps_to_stop'),
                    'max_steps':    args.max_steps,
                    'instruction':  instruction,
                    'length':       length,
                    'itype':        itype,
                })
        elif prev:
            # single-episode legacy format
            prev['instruction'] = instruction
            prev['length']      = length
            prev['itype']       = itype
            prev.setdefault('max_steps', args.max_steps)
            prev_episodes.append(prev)

        n_needed = max(0, args.n_episodes - len(prev_episodes))
        episodes = list(prev_episodes)

        for ep in range(n_needed):
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    r = run_instruction_episode(
                        ctrl, instruction, env, encoder,
                        stop_threshold=args.stop_threshold,
                        max_steps=args.max_steps,
                    )
                    break
                except RuntimeError as e:
                    print(f"\n  [WARN] Server error ep {ep+1} attempt {attempt+1}: {e}")
                    if attempt + 1 == max_retries:
                        print(f"  [SKIP] Giving up after {max_retries} attempts.")
                        r = None
                    else:
                        print(f"  [RETRY] Waiting 10s then retrying...")
                        time.sleep(10)
            if r is None:
                continue
            r['max_steps']   = args.max_steps
            r['instruction'] = instruction
            r['length']      = length
            r['itype']       = itype
            episodes.append(r)
            plog.log(length=length, itype=itype, ep=len(episodes),
                     instruction=instruction[:50],
                     p_stop_fired=r['p_stop_fired'],
                     steps_to_stop=r['steps_to_stop'],
                     p_stop_max=r['p_stop_max'])

        results[key] = _merge_episodes(episodes)

        # Save partial results after every instruction
        with open(outdir / 'results_partial.json', 'w') as f:
            json.dump({f"{k[0]}_{k[1]}": v for k, v in results.items()}, f, indent=2)

    with open(outdir / 'results.json', 'w') as f:
        json.dump({f"{k[0]}_{k[1]}": v for k, v in results.items()}, f, indent=2)
    print(f"\nSaved {outdir}/results.json")

    print_summary(results, args.max_steps)
    plot_mean_ci(results, outdir)
    plot_steps_to_stop(results, outdir)
    plot_fired_rate(results, outdir)
    print(f"\nAll outputs in {outdir}/")


if __name__ == '__main__':
    main()
