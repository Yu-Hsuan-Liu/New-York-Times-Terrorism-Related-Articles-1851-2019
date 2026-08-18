# -*- coding: utf-8 -*-
"""
Procrustes Temporal Alignment Analysis
---------------------------------------
Retrains 7 CBOW models with uniform hyperparameters, computes Procrustes
alignment for consecutive period pairs, and measures cosine displacement
for "terrorism" and "terrorist" across transitions.

Outputs:
  - Procrustes_Displacement_Results.csv
  - Figure2_Procrustes_Displacement.png
  - Procrustes_Alignment_Summary.txt
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import cosine as cosine_dist
import os, warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10

# --- repository-relative paths (edit here or set NYT_NLP_DATA / NYT_NLP_RESULTS) ---
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
DATA_ROOT = _os.environ.get('NYT_NLP_DATA', _os.path.join(REPO_ROOT, 'data'))
RESULTS_ROOT = _os.environ.get('NYT_NLP_RESULTS', _os.path.join(REPO_ROOT, 'results'))
_os.makedirs(RESULTS_ROOT, exist_ok=True)
OUT_DIR = REPO_ROOT

# ── Period definitions ────────────────────────────────────────────────────────
PERIODS_ORDERED = [
    '1851-1900', '1901-1930', '1931-1950',
    '1951-1960', '1961-1980', '1981-20010910', '20010911-2019',
]
PERIOD_LABELS = [
    '1851–1900', '1901–1930', '1931–1950',
    '1951–1960', '1961–1980', '1981–2001\n(pre-9/11)', '2001–2019\n(post-9/11)',
]

# Uniform hyperparameters for all periods
VECTOR_SIZE = 100
WINDOW = 5
MIN_COUNT = 5
EPOCHS = 20
SEED = 123
WORKERS = 1

TARGET_WORDS = ['terrorism', 'terrorist']

# ── Load data ─────────────────────────────────────────────────────────────────
print('Loading parquet data...')
df = pd.read_parquet(_os.path.join(DATA_ROOT, 'combined_processed_df.parquet'))
print(f'  Loaded {len(df)} articles')

# ── Train per-period CBOW models ─────────────────────────────────────────────
models = {}
for period in PERIODS_ORDERED:
    sub = df[df['label'] == period]
    sentences = [str(doc).split() for doc in sub['clean_lemmatized_phrased'] if isinstance(doc, str)]
    print(f'Training CBOW for {period}: {len(sentences)} docs')
    model = Word2Vec(
        sentences=sentences,
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        epochs=EPOCHS,
        seed=SEED,
        workers=WORKERS,
        sg=0,  # CBOW
        hs=0,
        negative=5,
    )
    models[period] = model

# ── Procrustes alignment for consecutive periods ─────────────────────────────
results = []

for i in range(len(PERIODS_ORDERED) - 1):
    p1 = PERIODS_ORDERED[i]
    p2 = PERIODS_ORDERED[i + 1]
    m1 = models[p1]
    m2 = models[p2]

    # Shared vocabulary
    vocab1 = set(m1.wv.key_to_index.keys())
    vocab2 = set(m2.wv.key_to_index.keys())
    shared = sorted(vocab1 & vocab2)
    print(f'\n{p1} -> {p2}: {len(shared)} shared words')

    if len(shared) < 50:
        print(f'  WARNING: Too few shared words for reliable alignment')
        continue

    # Build aligned matrices
    X = np.array([m1.wv[w] for w in shared])
    Y = np.array([m2.wv[w] for w in shared])

    # Compute Procrustes rotation: find R that minimizes ||X @ R - Y||
    R, scale = orthogonal_procrustes(X, Y)
    X_aligned = X @ R

    # Compute displacement for target words
    for target in TARGET_WORDS:
        if target in vocab1 and target in vocab2:
            idx = shared.index(target)
            vec_aligned = X_aligned[idx]
            vec_target = Y[idx]
            displacement = cosine_dist(vec_aligned, vec_target)
            results.append({
                'Transition': f'{p1} -> {p2}',
                'Period_From': p1,
                'Period_To': p2,
                'Target_Word': target,
                'Cosine_Displacement': round(displacement, 4),
                'Shared_Vocab_Size': len(shared),
            })
            print(f'  {target}: displacement = {displacement:.4f}')
        else:
            missing_in = []
            if target not in vocab1:
                missing_in.append(p1)
            if target not in vocab2:
                missing_in.append(p2)
            print(f'  {target}: MISSING in {", ".join(missing_in)}')

# ── Save CSV ──────────────────────────────────────────────────────────────────
results_df = pd.DataFrame(results)
csv_path = os.path.join(OUT_DIR, 'Procrustes_Displacement_Results.csv')
results_df.to_csv(csv_path, index=False)
print(f'\nSaved: {csv_path}')

# ── Generate Figure 2 ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

transition_labels = [f'{PERIOD_LABELS[i]}\n->\n{PERIOD_LABELS[i+1]}'
                     for i in range(len(PERIODS_ORDERED) - 1)]
# Simplify transition labels for x-axis
transition_short = []
for i in range(len(PERIODS_ORDERED) - 1):
    l1 = PERIOD_LABELS[i].replace('\n', ' ')
    l2 = PERIOD_LABELS[i+1].replace('\n', ' ')
    transition_short.append(f'{l1}\n-> {l2}')

x = np.arange(len(transition_short))
width = 0.35

for j, target in enumerate(TARGET_WORDS):
    vals = []
    for i in range(len(PERIODS_ORDERED) - 1):
        p1 = PERIODS_ORDERED[i]
        p2 = PERIODS_ORDERED[i + 1]
        match = [r for r in results
                 if r['Period_From'] == p1 and r['Period_To'] == p2
                 and r['Target_Word'] == target]
        vals.append(match[0]['Cosine_Displacement'] if match else 0)

    offset = -width / 2 + j * width
    color = '#2c3e50' if target == 'terrorism' else '#c0392b'
    bars = ax.bar(x + offset, vals, width, label=f'"{target}"',
                  color=color, alpha=0.85, edgecolor='white', linewidth=0.5)

    for bar, v in zip(bars, vals):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(transition_short, fontsize=7.5, ha='center')
ax.set_ylabel('Cosine Displacement\n(Procrustes-aligned)', fontsize=10)
ax.set_title(
    'Figure 2. Semantic Displacement of "terrorism" and "terrorist"\n'
    'Across Consecutive Historical Periods (Procrustes Temporal Alignment, CBOW)',
    fontsize=10, fontweight='bold', loc='left', pad=10,
)
ax.legend(fontsize=9, loc='upper left')
ax.set_ylim(0, max(r['Cosine_Displacement'] for r in results) * 1.25 if results else 1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3)

fig_path = os.path.join(REPO_ROOT, 'figures', 'Figure2_Procrustes_Displacement.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved: {fig_path}')

# ── Summary text ──────────────────────────────────────────────────────────────
summary_lines = [
    'Procrustes Temporal Alignment Summary',
    '=' * 50,
    '',
    f'Uniform hyperparameters: vector_size={VECTOR_SIZE}, window={WINDOW}, '
    f'min_count={MIN_COUNT}, epochs={EPOCHS}, seed={SEED}, workers={WORKERS}',
    f'Algorithm: CBOW (sg=0)',
    f'Number of periods: {len(PERIODS_ORDERED)}',
    f'Number of transitions: {len(PERIODS_ORDERED) - 1}',
    '',
]

for target in TARGET_WORDS:
    summary_lines.append(f'Target word: "{target}"')
    target_results = [r for r in results if r['Target_Word'] == target]
    if target_results:
        for r in target_results:
            summary_lines.append(
                f'  {r["Transition"]}: displacement = {r["Cosine_Displacement"]:.4f} '
                f'(shared vocab = {r["Shared_Vocab_Size"]})'
            )
        displacements = [r['Cosine_Displacement'] for r in target_results]
        summary_lines.append(f'  Mean displacement: {np.mean(displacements):.4f}')
        summary_lines.append(f'  Max displacement:  {np.max(displacements):.4f}')
        max_r = max(target_results, key=lambda r: r['Cosine_Displacement'])
        summary_lines.append(f'  Largest shift: {max_r["Transition"]}')
    summary_lines.append('')

txt_path = os.path.join(OUT_DIR, 'Procrustes_Alignment_Summary.txt')
with open(txt_path, 'w') as f:
    f.write('\n'.join(summary_lines))
print(f'Saved: {txt_path}')
print('\nDone.')
