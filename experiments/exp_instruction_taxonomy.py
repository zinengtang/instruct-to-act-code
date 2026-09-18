"""
exp_instruction_taxonomy.py — Instruction type taxonomy and causal analysis.

Addresses Reviewer 2B11 Q6:
  "Are the generated instructions mostly high-level subgoals, local action
   descriptions, or retrospective summaries of behavior? A taxonomy of
   instruction types would help, along with causal evidence that instruction
   quality correlates with execution success."

Protocol
────────
1. Load annotation logs (jsonl produced by PostHocAnnotator during training).
2. Embed instructions with MiniLM; cluster into K groups (k-means, K=8).
3. Auto-label clusters via GPT-4o: assign to one of four taxonomy categories:
     subgoal      — high-level goal ("mine diamond ore")
     action       — local action description ("move forward and jump")
     retrospective— past behaviour summary ("you just crafted a pickaxe")
     navigational — spatial/directional ("go left toward the mountain")
4. Compute per-cluster: frequency, mean episode reward, instruction-following
   accuracy, and correlation with execution success.
5. Show qualitative exemplars (top-3 nearest-to-centroid per cluster).
6. Output a LaTeX table and a 2D UMAP scatter coloured by taxonomy class.

Usage
─────
  python experiments/exp_instruction_taxonomy.py \
      --annotation_logs logdir/minecraft/seed0/annotations/annotations_minecraft.jsonl \
                        logdir/minecraft/seed1/annotations/annotations_minecraft.jsonl \
      --eval_results    results/instruction_robustness/robustness_minecraft_gpt4o.json \
      --env             minecraft \
      --n_clusters      8 \
      --outdir          results/instruction_taxonomy
"""

import argparse, json, os, sys, random
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ─────────────────────────────────────────────────────────────────────────────
# Taxonomy constants
# ─────────────────────────────────────────────────────────────────────────────

TAXONOMY_CLASSES = ['subgoal', 'action', 'retrospective', 'navigational']

TAXONOMY_COLORS = {
    'subgoal':       '#1565C0',
    'action':        '#E65100',
    'retrospective': '#2E7D32',
    'navigational':  '#6A1B9A',
    'unknown':       '#78909C',
}

CLASSIFY_PROMPT = """\
You are classifying a short instruction given to an RL agent.
Assign exactly one label from: subgoal | action | retrospective | navigational

Definitions:
  subgoal       – high-level task goal ("mine diamond ore", "cook a meal")
  action        – local or atomic action description ("jump over the gap", "attack the enemy")
  retrospective – describes what the agent just did ("you collected wood", "you crafted a pickaxe")
  navigational  – spatial / directional movement ("go north", "walk toward the river")

Instruction: "{instruction}"
Reply with exactly one word (the label). No explanation."""


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_annotation_logs(paths: list) -> list:
    """Load annotation records from one or more jsonl files."""
    records = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            print(f"  WARNING: {p} not found — skipping")
            continue
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Embedding and clustering
# ─────────────────────────────────────────────────────────────────────────────

def embed_instructions(instructions: list, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    embeddings = model.encode(instructions, normalize_embeddings=True,
                              show_progress_bar=True, batch_size=128)
    return embeddings.astype('float32')


def cluster_instructions(embeddings: np.ndarray, n_clusters: int):
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings)
    return labels, km.cluster_centers_


def umap_project(embeddings: np.ndarray):
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42, metric='cosine',
                            n_neighbors=15, min_dist=0.1)
        return reducer.fit_transform(embeddings)
    except ImportError:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        return pca.fit_transform(embeddings)


# ─────────────────────────────────────────────────────────────────────────────
# GPT-4o taxonomy labelling
# ─────────────────────────────────────────────────────────────────────────────

def classify_instruction(instruction: str, client=None) -> str:
    if client is None:
        try:
            import openai
            client = openai.OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        except Exception:
            return 'unknown'
    try:
        resp = client.chat.completions.create(
            model='gpt-4o',
            messages=[{'role': 'user',
                        'content': CLASSIFY_PROMPT.format(instruction=instruction)}],
            max_tokens=5, temperature=0.0,
        )
        label = resp.choices[0].message.content.strip().lower()
        return label if label in TAXONOMY_CLASSES else 'unknown'
    except Exception:
        return 'unknown'


def label_cluster_centroids(cluster_exemplars: dict, client=None) -> dict:
    """Label each cluster by classifying its centroid exemplar."""
    labels = {}
    for cluster_id, exemplars in cluster_exemplars.items():
        if not exemplars:
            labels[cluster_id] = 'unknown'
            continue
        centroid_text = exemplars[0]  # nearest to centroid
        labels[cluster_id] = classify_instruction(centroid_text, client)
        print(f"  Cluster {cluster_id}: '{centroid_text[:50]}' → {labels[cluster_id]}")
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────────────────────────────────────────

def find_exemplars(instructions: list, embeddings: np.ndarray, cluster_labels: np.ndarray,
                   centers: np.ndarray, n_exemplars: int = 3) -> dict:
    """For each cluster, find the n instructions closest to the centroid."""
    exemplars = {}
    n_clusters = centers.shape[0]
    for k in range(n_clusters):
        mask = cluster_labels == k
        if not mask.any():
            exemplars[k] = []
            continue
        idx = np.where(mask)[0]
        sub_embs = embeddings[idx]
        dists = np.linalg.norm(sub_embs - centers[k], axis=1)
        top = idx[np.argsort(dists)[:n_exemplars]]
        exemplars[k] = [instructions[i] for i in top]
    return exemplars


def compute_cluster_stats(records: list, cluster_labels: np.ndarray) -> dict:
    """Compute per-cluster frequency, mean reward, mean follow accuracy."""
    cluster_rewards = defaultdict(list)
    cluster_follow  = defaultdict(list)

    for i, rec in enumerate(records):
        k = int(cluster_labels[i])
        if 'reward' in rec:
            cluster_rewards[k].append(float(rec['reward']))
        if 'follow_accuracy' in rec:
            cluster_follow[k].append(float(rec['follow_accuracy']))

    stats = {}
    total = len(records)
    n_clusters = int(cluster_labels.max()) + 1
    for k in range(n_clusters):
        freq = int((cluster_labels == k).sum())
        stats[k] = {
            'frequency':      freq,
            'freq_pct':       100.0 * freq / max(total, 1),
            'mean_reward':    float(np.mean(cluster_rewards[k])) if cluster_rewards[k] else float('nan'),
            'mean_follow':    float(np.mean(cluster_follow[k]))  if cluster_follow[k]  else float('nan'),
        }
    return stats


def reward_taxonomy_correlation(cluster_labels: np.ndarray, cluster_taxonomy: dict,
                                 records: list) -> dict:
    """Pearson correlation between taxonomy class indicator and episode reward."""
    rewards = np.array([float(r.get('reward', 0)) for r in records])
    corrs = {}
    for tclass in TAXONOMY_CLASSES:
        indicator = np.array([1.0 if cluster_taxonomy.get(int(cluster_labels[i]), 'unknown') == tclass
                              else 0.0 for i in range(len(records))])
        if indicator.std() < 1e-6 or rewards.std() < 1e-6:
            corrs[tclass] = float('nan')
        else:
            corrs[tclass] = float(np.corrcoef(indicator, rewards)[0, 1])
    return corrs


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_umap(proj: np.ndarray, cluster_labels: np.ndarray,
              cluster_taxonomy: dict, outdir: Path, env: str):
    fig, ax = plt.subplots(figsize=(8, 7))

    n_clusters = int(cluster_labels.max()) + 1
    for k in range(n_clusters):
        mask = cluster_labels == k
        tclass = cluster_taxonomy.get(k, 'unknown')
        color  = TAXONOMY_COLORS.get(tclass, '#78909C')
        ax.scatter(proj[mask, 0], proj[mask, 1],
                   c=color, alpha=0.4, s=8, label=None, rasterized=True)

    # Legend patches
    from matplotlib.patches import Patch
    seen = set()
    handles = []
    for k in range(n_clusters):
        tclass = cluster_taxonomy.get(k, 'unknown')
        if tclass not in seen:
            seen.add(tclass)
            handles.append(Patch(color=TAXONOMY_COLORS.get(tclass, '#78909C'), label=tclass))
    ax.legend(handles=handles, fontsize=9, title='Type')

    ax.set_title(f'Instruction Embedding Space — {env}\n(UMAP / PCA fallback)', fontsize=11)
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f'taxonomy_umap_{env}.pdf')
    fig.savefig(outdir / f'taxonomy_umap_{env}.png', dpi=150)
    plt.close()
    print(f"Saved to {outdir}/taxonomy_umap_{env}.{{pdf,png}}")


def plot_taxonomy_breakdown(cluster_stats: dict, cluster_taxonomy: dict,
                             outdir: Path, env: str):
    # Aggregate by taxonomy class
    class_freq  = defaultdict(int)
    class_rew   = defaultdict(list)
    class_follow= defaultdict(list)

    for k, s in cluster_stats.items():
        tclass = cluster_taxonomy.get(k, 'unknown')
        class_freq[tclass]  += s['frequency']
        if not np.isnan(s['mean_reward']):
            class_rew[tclass].append(s['mean_reward'])
        if not np.isnan(s['mean_follow']):
            class_follow[tclass].append(s['mean_follow'])

    classes = [c for c in TAXONOMY_CLASSES if c in class_freq] + \
              (['unknown'] if 'unknown' in class_freq else [])
    total = sum(class_freq.values())

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # Pie: frequency
    ax = axes[0]
    sizes  = [class_freq[c] for c in classes]
    colors = [TAXONOMY_COLORS.get(c, '#78909C') for c in classes]
    ax.pie(sizes, labels=classes, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.set_title('Instruction type distribution')

    # Bar: mean reward
    ax = axes[1]
    means = [np.mean(class_rew[c]) if class_rew[c] else 0 for c in classes]
    stds  = [np.std(class_rew[c])  if class_rew[c] else 0 for c in classes]
    x = range(len(classes))
    ax.bar(x, means, yerr=stds, color=[TAXONOMY_COLORS.get(c, '#78909C') for c in classes],
           capsize=4, alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels(classes, rotation=15, ha='right', fontsize=9)
    ax.set_ylabel('Mean episode reward')
    ax.set_title('Reward by instruction type')
    ax.grid(True, alpha=0.3, axis='y')

    # Bar: follow accuracy
    ax = axes[2]
    faccs = [np.mean(class_follow[c]) * 100 if class_follow[c] else 0 for c in classes]
    ax.bar(x, faccs, color=[TAXONOMY_COLORS.get(c, '#78909C') for c in classes], alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels(classes, rotation=15, ha='right', fontsize=9)
    ax.set_ylabel('Instruction-following accuracy (%)')
    ax.set_title('Follow accuracy by instruction type')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 105)

    plt.suptitle(f'Instruction Taxonomy — {env}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(outdir / f'taxonomy_breakdown_{env}.pdf')
    fig.savefig(outdir / f'taxonomy_breakdown_{env}.png', dpi=150)
    plt.close()
    print(f"Saved to {outdir}/taxonomy_breakdown_{env}.{{pdf,png}}")


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX
# ─────────────────────────────────────────────────────────────────────────────

def print_latex_table(cluster_stats: dict, cluster_taxonomy: dict,
                       exemplars: dict, corrs: dict, env: str):
    print(f"\n% Instruction Taxonomy Table — {env}")
    print(r"\begin{tabular}{llrrrl}")
    print(r"\toprule")
    print(r"Type & Cluster & Freq\% & Reward & Follow\% & Example \\ \midrule")

    for k in sorted(cluster_stats.keys()):
        s = cluster_stats[k]
        tclass = cluster_taxonomy.get(k, 'unknown')
        exmpl  = exemplars.get(k, ['—'])[0][:40].replace('_', r'\_')
        freq   = f"{s['freq_pct']:.1f}"
        rew    = f"{s['mean_reward']:.2f}" if not np.isnan(s['mean_reward']) else '—'
        fol    = f"{s['mean_follow']*100:.1f}" if not np.isnan(s['mean_follow']) else '—'
        print(f"{tclass} & {k} & {freq} & {rew} & {fol} & {exmpl} \\\\")

    print(r"\midrule")
    print(r"\multicolumn{6}{l}{Reward correlation by type:} \\")
    for tclass, corr in corrs.items():
        corr_str = f"{corr:.3f}" if not np.isnan(corr) else '—'
        print(f"& {tclass} & \\multicolumn{{4}}{{l}}{{Pearson $r = {corr_str}$}} \\\\")
    print(r"\bottomrule\end{tabular}")


# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--annotation_logs', nargs='+',
                   default=['logdir/minecraft/seed0/annotations/annotations_minecraft.jsonl'])
    p.add_argument('--eval_results',    default=None,
                   help='Optional JSON with per-episode reward/follow_accuracy')
    p.add_argument('--env',             default='minecraft')
    p.add_argument('--n_clusters',      type=int, default=8)
    p.add_argument('--outdir',          default='results/instruction_taxonomy')
    return p.parse_args()


def main():
    args   = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load annotation records
    records = load_annotation_logs(args.annotation_logs)
    if not records:
        print("No annotation records found. Generating synthetic demo data.")
        demo_instructions = [
            "mine diamond ore", "craft a pickaxe", "find food quickly",
            "go north toward the mountain", "attack the zombie",
            "explore the cave", "you just collected wood", "build a shelter",
            "walk forward and jump", "gather more stone", "look for iron",
            "move toward the river", "you crafted a sword", "find diamonds",
            "go left to avoid lava", "chop down the tree",
        ]
        records = [{'instruction': random.choice(demo_instructions),
                    'reward': random.gauss(0.5, 0.2),
                    'follow_accuracy': float(random.random() > 0.3)}
                   for _ in range(400)]

    instructions = [r.get('instruction', r.get('text', '')) for r in records]
    instructions = [i for i in instructions if i]
    records      = [r for r in records if r.get('instruction', r.get('text', ''))]
    print(f"Loaded {len(instructions)} instructions from annotation logs.")

    # Merge eval results if provided
    if args.eval_results:
        eval_path = Path(args.eval_results)
        if eval_path.exists():
            with open(eval_path) as f:
                eval_data = json.load(f)
            # Augment records with eval rewards if record count matches
            rewards_flat = []
            for cond_data in eval_data.values():
                rewards_flat.extend(cond_data.get('rewards', []))
            if len(rewards_flat) >= len(records):
                for i, rec in enumerate(records):
                    if 'reward' not in rec:
                        rec['reward'] = rewards_flat[i % len(rewards_flat)]

    # Embed
    print("Embedding instructions...")
    embeddings = embed_instructions(instructions)

    # Cluster
    print(f"Clustering into {args.n_clusters} clusters...")
    cluster_labels, centers = cluster_instructions(embeddings, args.n_clusters)

    # Find exemplars
    exemplars = find_exemplars(instructions, embeddings, cluster_labels, centers)

    # Label clusters via GPT-4o (or fallback heuristic)
    print("Labelling clusters via GPT-4o...")
    try:
        import openai
        client = openai.OpenAI(api_key=os.environ['OPENAI_API_KEY'])
    except Exception:
        client = None
    cluster_taxonomy = label_cluster_centroids(exemplars, client)

    # Compute stats
    cluster_stats = compute_cluster_stats(records, cluster_labels)

    # Reward-taxonomy correlation
    corrs = reward_taxonomy_correlation(cluster_labels, cluster_taxonomy, records)
    print(f"\nReward correlations by taxonomy type: {corrs}")

    # UMAP projection
    print("Projecting embeddings for visualisation...")
    proj = umap_project(embeddings)

    # Plots
    plot_umap(proj, cluster_labels, cluster_taxonomy, outdir, args.env)
    plot_taxonomy_breakdown(cluster_stats, cluster_taxonomy, outdir, args.env)

    # LaTeX table
    print_latex_table(cluster_stats, cluster_taxonomy, exemplars, corrs, args.env)

    # Print exemplars
    print(f"\n── Cluster Exemplars ({args.env}) ──")
    for k in sorted(exemplars.keys()):
        tclass = cluster_taxonomy.get(k, 'unknown')
        print(f"  Cluster {k} [{tclass}]:")
        for ex in exemplars[k]:
            print(f"    • {ex}")

    # Save
    out = outdir / f'taxonomy_{args.env}.json'
    with open(out, 'w') as f:
        json.dump({
            'n_instructions': len(instructions),
            'n_clusters': args.n_clusters,
            'cluster_taxonomy': {str(k): v for k, v in cluster_taxonomy.items()},
            'cluster_stats':    {str(k): v for k, v in cluster_stats.items()},
            'exemplars':        {str(k): v for k, v in exemplars.items()},
            'reward_correlations': {k: (v if not np.isnan(v) else None)
                                     for k, v in corrs.items()},
        }, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == '__main__':
    main()
