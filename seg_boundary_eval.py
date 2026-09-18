"""How well do VLM-proposed segment ends line up with ground-truth events? Reads annotations_<env>.jsonl written by the
'segment' annotator: entries src=vlm_seg (worker,u,v,instruction) and ground-truth entries (worker,t_e,achievement).
Distances are in replay slots; with W parallel envs one per-worker step = W slots. Reports, for tolerance K steps:
recall = fraction of gt events with a VLM segment end within K; precision = fraction of VLM ends within K of a gt event;
plus a random-boundary baseline with the same number of ends. usage: python seg_boundary_eval.py <annotations.jsonl> [W=16] [K=8]"""
import json, sys, numpy as np, collections
rows = [json.loads(l) for l in open(sys.argv[1])]; W = int(sys.argv[2]) if len(sys.argv) > 2 else 16; K = int(sys.argv[3]) if len(sys.argv) > 3 else 8
seg = collections.defaultdict(list); ev = collections.defaultdict(list); ins = collections.Counter()
for r in rows:
    if r.get('src') == 'vlm_seg': seg[r['worker']].append(r['v']); ins[r['instruction']] += 1
    elif 't_e' in r: ev[r['worker']].append(r['t_e'])
tol = K * W; hit_ev = tot_ev = hit_end = tot_end = 0; rng = np.random.RandomState(0); rand_hit = 0
for w in ev:
    E = np.array(sorted(ev[w])); S = np.array(sorted(seg.get(w, []))); tot_ev += len(E); tot_end += len(S)
    if len(S):
        hit_ev += sum(np.abs(S - e).min() <= tol for e in E); hit_end += sum(np.abs(E - s).min() <= tol for s in S)
        R = rng.uniform(S.min(), S.max(), len(S)); rand_hit += sum(np.abs(R - e).min() <= tol for e in E)
print(f"gt events {tot_ev}, vlm segment ends {tot_end}, tolerance ±{K} steps")
print(f"recall (gt event has a VLM end within K): {hit_ev/max(1,tot_ev):.3f}   random-ends baseline {rand_hit/max(1,tot_ev):.3f}")
print(f"precision (VLM end has a gt event within K): {hit_end/max(1,tot_end):.3f}   (note: events without inventory/achievement effect are not in gt)")
print("top instructions:"); [print(f"  {n:4d}  {t[:80]}") for t, n in ins.most_common(12)]
