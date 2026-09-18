"""Per-instruction follow-through from recorded rollouts (evaluate.py --record): an instruction counts as completed if its
target inventory item increased between the step it was issued and the step the next instruction was issued (or episode end).
Also reports milestones (reward) per episode and mean steps spent per instruction. usage: python followthrough.py <record_dir>"""
import glob, sys, numpy as np
try:
    from planners.state_text import StateFormatter; FMT = StateFormatter([])   # alias matching for free-text instructions
except Exception: FMT = None
TARGET = {"collect wood from nearby trees": "log", "craft planks from collected wood": "planks", "craft a crafting table": "crafting_table",
          "craft sticks for tool recipes": "stick", "craft a wooden pickaxe": "wooden_pickaxe", "mine stone to gather cobblestone": "cobblestone",
          "craft a stone pickaxe": "stone_pickaxe", "mine iron ore with the stone pickaxe": "iron_ore", "craft a furnace near your position": "furnace",
          "smelt iron ore into ingots": "iron_ingot", "craft an iron pickaxe": "iron_pickaxe", "mine diamond ore with the iron pickaxe": "diamond"}
rows = []
for f in sorted(glob.glob(f"{sys.argv[1]}/record_ep*.npz")):
    d = np.load(f, allow_pickle=True); keys = [k.replace("inventory/", "") for k in d["inv_keys"]]; texts = list(d["instr_texts"]); iid = d["instr_id"]; inv = d["inv_post"]
    # segment by the planner's actual issue steps (instruction_log), so re-issued identical instructions stay separate
    starts = sorted({int(x.split('|', 1)[0]) for x in d['instruction_log']}) if 'instruction_log' in d.files and len(d['instruction_log']) else []
    if not starts or starts[0] != 0: starts = [0] + [s for s in starts if s > 0]
    segs = [[s0, (starts[i + 1] - 1 if i + 1 < len(starts) else len(iid) - 1)] for i, s0 in enumerate(starts)]
    done = []; lens = []
    for s0, s1 in segs:
        txt = texts[iid[min(s0, len(iid) - 1)]]; item = FMT.target_item(txt) if FMT else TARGET.get(txt); lens.append(s1 - s0 + 1)
        if item is None or item not in keys: continue
        j = keys.index(item); base = inv[max(s0 - 1, 0), j]; done.append(int(inv[s0:s1 + 1, j].max() > base))
    rows.append(dict(ep=f, n_instr=len(segs), followed=float(np.mean(done)) if done else float('nan'), n_done=int(np.sum(done)), n_scored=len(done), reward=float(np.sum(d["reward"])) if "reward" in d.files else float('nan'), mean_len=float(np.mean(lens)), med_len=float(np.median(lens))))
ft = np.array([r['followed'] for r in rows]); rw = np.array([r['reward'] for r in rows]); ni = np.array([r['n_instr'] for r in rows]); nd = np.array([r['n_done'] for r in rows]); ml = np.array([r['med_len'] for r in rows])
print(f"{len(rows)} episodes: follow-through {np.nanmean(ft):.3f} (±{1.96*np.nanstd(ft)/np.sqrt(len(ft)):.3f}), completed instructions/ep {nd.mean():.2f} (±{1.96*nd.std()/np.sqrt(len(nd)):.2f}), instructions/ep {ni.mean():.1f}, median steps/instruction {np.median(ml):.0f}, milestones/ep {np.nanmean(rw):.2f}")
