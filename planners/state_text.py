"""Text state and controller feedback for the planner (no more reading the HUD from 64x64 pixels).

StateFormatter turns an observation dict into a short text line (inventory, equipped item, vitals, change since the
current instruction was issued) and can verify an instruction's completion from state when the instruction names an
inventory item. Works for any env exposing an 'inventory' vector plus item names; degrades to '' otherwise."""
from typing import Dict, List, Optional
import numpy as np

# instruction wording -> inventory key (Minecraft). Longest alias wins; plain "stone" maps to cobblestone.
ITEM_ALIASES = {
    'log': ['logs', 'log', 'wood', 'tree'], 'planks': ['planks', 'plank'], 'crafting_table': ['crafting table', 'workbench', 'table'],
    'stick': ['sticks', 'stick'], 'wooden_pickaxe': ['wooden pickaxe', 'wood pickaxe'], 'cobblestone': ['cobblestone', 'stone'],
    'stone_pickaxe': ['stone pickaxe'], 'iron_ore': ['iron ore'], 'furnace': ['furnace'], 'iron_ingot': ['iron ingot', 'ingot'],
    'iron_pickaxe': ['iron pickaxe'], 'diamond': ['diamond'], 'coal': ['coal'], 'torch': ['torch'], 'dirt': ['dirt'], 'sapling': ['sapling'],
}
VITALS = {'health': 20.0, 'hunger': 20.0, 'breath': 300.0}   # obs values are normalised to [0,1]; multiply back for display
HIDDEN_ITEMS = {'air', 'log2'}   # MineRL 'air' = empty slots, not an item


class StateFormatter:
    def __init__(self, inv_keys: Optional[List[str]] = None, equip_names: Optional[List[str]] = None):
        self.inv_keys = [k.replace('inventory/', '') for k in (inv_keys or [])]
        self.equip_names = list(equip_names or [])
        # alias -> item, sorted longest first so 'stone pickaxe' beats 'stone'
        self._alias = sorted(((a, item) for item, al in ITEM_ALIASES.items() for a in al if item in self.inv_keys or not self.inv_keys),
                             key=lambda x: -len(x[0]))

    def inventory(self, obs) -> Dict[str, int]:
        if not isinstance(obs, dict) or 'inventory' not in obs or not self.inv_keys: return {}
        v = np.asarray(obs['inventory']).reshape(-1)
        return {k: int(round(float(v[i]))) for i, k in enumerate(self.inv_keys) if i < len(v)}

    def target_item(self, instruction: str) -> Optional[str]:
        s = (instruction or '').lower()
        for a, item in self._alias:
            if a in s: return item
        return None

    def text(self, obs, issue_inv: Optional[Dict[str, int]] = None) -> str:
        inv = self.inventory(obs)
        parts = []
        have = [f"{k}x{n}" for k, n in inv.items() if n > 0 and k not in HIDDEN_ITEMS]
        parts.append('inventory: ' + (', '.join(have) if have else 'empty'))
        if isinstance(obs, dict) and 'equipped' in obs and self.equip_names:
            e = np.asarray(obs['equipped']).reshape(-1)
            if e.size and e.max() > 0: parts.append('equipped: ' + str(self.equip_names[int(np.argmax(e))]))
        vit = [f"{k} {float(obs[k]) * m:.0f}/{m:.0f}" for k, m in VITALS.items() if isinstance(obs, dict) and k in obs]
        if vit: parts.append(' '.join(vit))
        if issue_inv:
            d = {k: inv.get(k, 0) - issue_inv.get(k, 0) for k in inv}
            ch = [f"{'+' if v > 0 else ''}{v} {k}" for k, v in d.items() if v and k not in HIDDEN_ITEMS]
            parts.append('change since current instruction: ' + (', '.join(ch) if ch else 'none'))
        return ' | '.join(parts)

    def verify(self, instruction: str, obs, issue_inv: Optional[Dict[str, int]]) -> str:
        """'yes' if the item named in the instruction increased since the instruction was issued, 'no' if it did not,
        'unknown' if the instruction names no inventory item (navigation/exploration)."""
        item = self.target_item(instruction)
        if item is None or issue_inv is None: return 'unknown'
        inv = self.inventory(obs)
        return 'yes' if inv.get(item, 0) > issue_inv.get(item, 0) else 'no'


def find_attr(obj, name, depth=0):
    """Walk env wrappers to find an attribute (e.g. _inv_keys, _equip_enum)."""
    if depth > 8 or obj is None: return None
    if hasattr(obj, name): return getattr(obj, name)
    for a in ('_env', 'env', '_gymenv', 'envs'):
        sub = getattr(obj, a, None); sub = sub[0] if isinstance(sub, (list, tuple)) and sub else sub
        r = find_attr(sub, name, depth + 1)
        if r is not None: return r
    return None
