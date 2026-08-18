# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10

# ── Read data ────────────────────────────────────────────────────────────────
# --- repository-relative paths (edit here or set NYT_NLP_DATA / NYT_NLP_RESULTS) ---
import os as _os
REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
DATA_ROOT = _os.environ.get('NYT_NLP_DATA', _os.path.join(REPO_ROOT, 'data'))
RESULTS_ROOT = _os.environ.get('NYT_NLP_RESULTS', _os.path.join(REPO_ROOT, 'results'))
_os.makedirs(RESULTS_ROOT, exist_ok=True)
df = pd.read_csv(_os.path.join(RESULTS_ROOT, 'FINAL_ALL_MODELS_RESULTS.csv'))

# ── Period ordering and display labels ───────────────────────────────────────
periods_ordered = [
    '1851-1900', '1901-1930', '1931-1950',
    '1951-1960', '1961-1980', '1981-20010910', '20010911-2019',
]
period_labels = [
    '1851–\n1900', '1901–\n1930', '1931–\n1950',
    '1951–\n1960', '1961–\n1980', '1981–2001\n(pre-9/11)', '2001–2019\n(post-9/11)',
]

# ── Thematic indicator terms ──────────────────────────────────────────────────
# Each entry: (match_fn, display_label)
def exact(t):
    return lambda w: w == t

def contains(t):
    return lambda w: t in w

thematic_groups = [
    ('Racial Violence', '#c0392b', [
        (exact('klan'),                        'Klan'),
        (exact('practical_disfranchisement'),  'Practical disfranchisement'),
        (contains('night_rider'),              'Night rider'),
    ]),
    ('Anarchist / Early European', '#2980b9', [
        (contains('reign_terror'),             'Reign of terror'),
        (exact('assassination'),               'Assassination'),
        (exact('regicide'),                    'Regicide'),
    ]),
    ('Anti-Colonial / WWII-era', '#27ae60', [
        (exact('nazis'),                       'Nazis'),
        (exact('zionist'),                     'Zionist'),
        (exact('kenya'),                       'Kenya'),
        (exact('kill_injure'),                 'Kill/injure'),
    ]),
    ('New Left / Cold War', '#e67e22', [
        (contains('urban_guerrilla'),          'Urban guerrilla'),
        (exact('red_army'),                    'Red Army'),
        (exact('basque'),                      'Basque'),
        (contains('islamic_extremist'),        'Islamic extremist'),
    ]),
    ('Post-9/11 Islamist Terror', '#8e44ad', [
        (exact('international_terrorism'),     "Int\u2019l terrorism"),
        (contains('al_qaeda'),                 'Al-Qaeda'),
        (exact('war_terror'),                  'War on terror'),
        (contains('homegrown_terrorist'),      'Homegrown terrorist'),
    ]),
]

# ── Build flat row list with metadata ────────────────────────────────────────
row_labels   = []
row_colors   = []
row_groups   = []
group_bounds = []     # index of first row in each group

for gname, gcolor, terms in thematic_groups:
    group_bounds.append(len(row_labels))
    for match_fn, label in terms:
        row_labels.append(label)
        row_colors.append(gcolor)
        row_groups.append(gname)

n_rows = len(row_labels)
n_cols = len(periods_ordered)

# ── Filter to CBOW only ──────────────────────────────────────────────────────
df = df[df['Model_Type'] == 'CBOW']

# ── Build similarity matrix (max across CBOW models and both targets) ────────
matrix = np.zeros((n_rows, n_cols))

for col_idx, period in enumerate(periods_ordered):
    sub = df[df['Period'] == period]
    row_idx = 0
    for _gname, _gcolor, terms in thematic_groups:
        for match_fn, _label in terms:
            matched = sub[sub['Neighbor_Word'].apply(match_fn)]
            if not matched.empty:
                matrix[row_idx, col_idx] = matched['Similarity'].max()
            row_idx += 1

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 7))
# Main heatmap axes; leave room on left for group labels
ax = fig.add_axes([0.20, 0.13, 0.60, 0.78])   # [left, bottom, width, height]
cax = fig.add_axes([0.82, 0.30, 0.018, 0.40])  # colorbar

# ── Heatmap ───────────────────────────────────────────────────────────────────
im = ax.imshow(matrix, aspect='auto', cmap='Blues',
               vmin=0, vmax=0.92, interpolation='nearest')

# ── Cell annotations ──────────────────────────────────────────────────────────
for i in range(n_rows):
    for j in range(n_cols):
        val = matrix[i, j]
        if val > 0:
            tc = 'white' if val > 0.60 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=7.5, color=tc, fontweight='bold')
        else:
            ax.text(j, i, '\u2014', ha='center', va='center',
                    fontsize=9, color='#cccccc')

# ── Axes ticks ────────────────────────────────────────────────────────────────
ax.set_xticks(np.arange(n_cols))
ax.set_xticklabels(period_labels, fontsize=9)
ax.set_yticks(np.arange(n_rows))
ax.set_yticklabels(row_labels, fontsize=9)

# Color y-tick labels by group
for tick, color in zip(ax.get_yticklabels(), row_colors):
    tick.set_color(color)

# ── Group separators (horizontal dashed lines) ────────────────────────────────
for boundary in group_bounds[1:]:
    ax.axhline(boundary - 0.5, color='#555555', linewidth=0.9, linestyle='--', alpha=0.6)

# ── 9/11 vertical divider ─────────────────────────────────────────────────────
ax.axvline(5.5, color='#333333', linestyle='--', linewidth=1.2, alpha=0.75)
ax.text(5.54, -0.7, '9/11', fontsize=8, color='#333333', va='bottom', style='italic')

# ── Colorbar ──────────────────────────────────────────────────────────────────
cb = fig.colorbar(im, cax=cax)
cb.set_label('Cosine Similarity\n(max across CBOW targets)', fontsize=8.5)
cb.ax.tick_params(labelsize=8)

# ── Group labels as colored text on the left sidebar ─────────────────────────
ax_left = fig.transFigure
for g_idx, (gname, gcolor, terms) in enumerate(thematic_groups):
    start_row = group_bounds[g_idx]
    end_row   = (group_bounds[g_idx + 1] - 1) if g_idx + 1 < len(group_bounds) else (n_rows - 1)
    mid_row   = (start_row + end_row) / 2

    # Convert data coords to figure fraction
    ax_pos = ax.get_position()
    y_frac = ax_pos.y0 + ax_pos.height * (1 - (mid_row + 0.5) / n_rows)

    fig.text(0.01, y_frac, gname, color=gcolor, fontsize=8.5, fontweight='bold',
             va='center', rotation=0, wrap=True)

# ── Title and xlabel ──────────────────────────────────────────────────────────
ax.set_title(
    'Figure 1. Semantic Proximity of Thematic Indicator Terms to \u201cterrorism\u201d/\u201cterrorist\u201d\n'
    'by Historical Period (Word2Vec CBOW; maximum cosine similarity across both targets)',
    fontsize=9.5, fontweight='bold', pad=10, loc='left',
)
ax.set_xlabel('Historical Period', fontsize=10, labelpad=8)

# ── Save ──────────────────────────────────────────────────────────────────────
out = _os.path.join(REPO_ROOT, 'figures', 'Figure1_W2V_Heatmap_CBOW.png')
plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved:', out)
